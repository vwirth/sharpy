#include <opencv2/core/hal/interface.h>
#include <reclib/depth_processing.h>
#include <reclib/dnn/nvdiffrast_autograd.h>
#include <reclib/internal/filesystem_ops.h>
#include <reclib/optim/model_registration.h>
#include <reclib/tracking/mano_rgbd.h>
#include <reclib/tracking/mano_rgbd_optim.h>

#include <algorithm>
#include <filesystem>
#include <future>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <reclib/cuda/device_info.cuh>
#include <stdexcept>
#include <thread>
#include <utility>

#if HAS_CERES_MODULE
#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

#include <ceres/types.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <torch_tensorrt/core/compiler.h>
#include <torch_tensorrt/torch_tensorrt.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

// static const vec3 SKIN_COLOR = (vec3(250, 214, 190) / 255) * 0.8;
static const vec3 SKIN_COLOR = (vec3(230, 184, 255) / 255) * 0.7;

const int OPENPOSE_KEYPOINT_PAIRS[20][2] = {
    {0, 1},  {1, 2},   {2, 3},   {3, 4},    // thumb
    {0, 5},  {5, 6},   {6, 7},   {7, 8},    // pinkie
    {0, 9},  {9, 10},  {10, 11}, {11, 12},  // middle
    {0, 13}, {13, 14}, {14, 15}, {15, 16},  // ring
    {0, 17}, {17, 18}, {18, 19}, {19, 20}   // small
};

static std::vector<int> apose2vis_indices = {
    5,  5,  6,  7,  9, 9, 10, 11, 17, 17, 18, 19,
    13, 13, 14, 15, 1, 1, 1,  2,  2,  2,  3,
};

static std::vector<int> vis2pose_indices = {0, 13, 14, 15, 0,  1,  2, 3, 0, 4,
                                            5, 6,  0,  10, 11, 12, 0, 7, 8, 9};
static std::vector<int> joint2vis_indices = {5,  6,  7,  9,  10, 11, 17, 18,
                                             19, 13, 14, 15, 1,  2,  3};

static std::vector<int> vertices_per_joint = {31, 20, 34, 45, 46, 28, 40,
                                              62, 43, 19, 40, 60, 39, 26,
                                              32, 64, 37, 20, 36, 66};

static std::vector<int> apose2joint = {1,  1,  2,  3,  4,  4,  5,  6,
                                       7,  7,  8,  9,  10, 10, 11, 12,
                                       13, 13, 13, 14, 14, 14, 15};

// -----------------------------------------------------------------------
// HandLimits
// -----------------------------------------------------------------------

const vec3 reclib::tracking::HandLimits::limits_min[16] = {
    vec3(0, 0, 0),  // wrist - ignored

    vec3(-0.5, -0.5, -1),    // index1
    vec3(-0.1, -0.1, -0.1),  // index2
    vec3(-0.1, -0.1, -0.1),  // index3

    vec3(-0.5, -0.5, -1),    // middle1
    vec3(-0.1, -0.1, -0.1),  // middle2
    vec3(-0.1, -0.1, -0.1),  // middle3

    vec3(-0.5, -0.5, -1),    // pinky1
    vec3(-0.1, -0.1, -0.1),  // pinky2
    vec3(-0.1, -0.1, -0.1),  // pinky3

    vec3(-0.5, -0.5, -1),    // ring1
    vec3(-0.1, -0.1, -0.1),  // ring2
    vec3(-0.1, -0.1, -0.1),  // ring3

    vec3(-1, -1, -1),     //  thumb1
    vec3(-1, -1, -1),     // thumb2
    vec3(-0.5, -1, -0.5)  // thumb3
};

const vec3 reclib::tracking::HandLimits::limits_max[16] = {
    vec3(0, 0, 0),  // wrist - ignored

    vec3(0.5, 0.5, 1.6),  // index1
    vec3(0.1, 0.1, 1.6),  // index2
    vec3(0.1, 0.1, 1.6),  // index3

    vec3(0.5, 0.5, 1.6),  // middle1
    vec3(0.1, 0.1, 1.6),  // middle2
    vec3(0.1, 0.1, 1.6),  // middle3

    vec3(0.5, 0.5, 1.6),  // pinky1
    vec3(0.1, 0.1, 1.6),  // pinky2
    vec3(0.1, 0.1, 1.6),  // pinky3

    vec3(0.5, 0.5, 1.6),  // ring1
    vec3(0.1, 0.1, 1.6),  // ring2
    vec3(0.1, 0.1, 1.6),  // ring3

    vec3(1.6, 1, 1.6),   //  thumb1
    vec3(1.6, 1, 1.6),   // thumb2
    vec3(0.5, 0.4, 0.5)  // thumb3
};

// -----------------------------------------------------------------------
// ManoRGBD
// -----------------------------------------------------------------------

template <typename MANOTYPE>
reclib::tracking::ManoRGBD<MANOTYPE>::ManoRGBD(
    const reclib::Configuration& config,
    const reclib::IntrinsicParameters& intrinsics, bool debug)
    : config_(config),
      debug_(config_["Debug"]["general"].as<bool>()),
      intrinsics_(intrinsics),
      network_(config_["Network"]["weights"].as<std::string>(), true, true,
               true, false),
      openpose_(cv::dnn::readNet(
          config_["Network"]["openpose_weights"].as<std::string>(),
          config_["Network"]["openpose_protofile"].as<std::string>())),
      transform_(config_["Network"]["max_size"].as<unsigned int>(),
                 config_["Network"]["preserve_aspect_ratio"].as<bool>(), "BGR"),
      last_frame_(2),
      frame_counter_(0),
      global_frame_counter_(0),
      optimization_counter_(0),
      volume_size_(config_["Voxels"]["size"][0].as<unsigned int>(),
                   config_["Voxels"]["size"][1].as<unsigned int>(),
                   config_["Voxels"]["size"][2].as<unsigned int>()),
      volume_scale_(config_["Voxels"]["scale"].as<float>()),
      voxel_pos_(volume_size_.z() * volume_size_.y(), volume_size_.x(),
                 CV_32FC3),
      tsdf_volume_(cv::cuda::createContinuous(
          volume_size_.z() * volume_size_.y(), volume_size_.x(), CV_16SC2)),
      tsdf_weights_(cv::cuda::createContinuous(
          volume_size_.z() * volume_size_.y(), volume_size_.x(), CV_16UC1)),
      mano_model_left_(reclib::models::HandType::left),
      mano_model_right_(reclib::models::HandType::right),
      mano_left_(mano_model_left_),
      mano_right_(mano_model_right_)

{
  mano_left_.set_zero();
  mano_right_.set_zero();
  mano_left_.add_pose_mean_ = false;
  mano_right_.add_pose_mean_ = false;
  mano_left_.use_anatomic_pose_ = true;
  mano_right_.use_anatomic_pose_ = true;

  if (config_["Optimization"]["use_calculated_pca"].as<bool>()) {
    std::ifstream f;
    f.open(config_["Dataset"]["pca_file"].as<std::string>());
    nlohmann::json pca_params = nlohmann::json::parse(f);
    // torch::Tensor

    std::vector<float> mean_left =
        pca_params["left"]["mean"].get<std::vector<float>>();
    std::vector<float> mean_right =
        pca_params["right"]["mean"].get<std::vector<float>>();

    mano_model_left_.hand_mean =
        torch::from_blob(
            pca_params["left"]["mean"].get<std::vector<float>>().data(),
            {23, 1})
            .clone();
    mano_model_left_.hand_comps =
        torch::from_blob(
            pca_params["left"]["ev"].get<std::vector<float>>().data(), {23, 23})
            .clone();
    mano_left_.use_anatomic_pca_ = true;
    mano_left_.use_anatomic_pose_ = true;
    mano_left_.add_pose_mean_ = false;

    mano_model_right_.hand_mean =
        torch::from_blob(
            pca_params["right"]["mean"].get<std::vector<float>>().data(),
            {23, 1})
            .clone();
    mano_model_right_.hand_comps =
        torch::from_blob(
            pca_params["right"]["ev"].get<std::vector<float>>().data(),
            {23, 23})
            .clone();

    mano_right_.use_anatomic_pca_ = true;
    mano_right_.use_anatomic_pose_ = true;
    mano_right_.add_pose_mean_ = false;
    f.close();
  }

  mano_left_.update(false, true);
  mano_right_.update(false, true);

  if (config_["Optimization"]["pose_prior_weight"].as<float>() > 0) {
    std::ifstream f;
    f.open(config_["Dataset"]["gmm_file"].as<std::string>());
    nlohmann::json gmm_params = nlohmann::json::parse(f);
    f.close();

    gmm_.weights_ = torch::zeros({(int)gmm_params.size()});
    gmm_.means_ = torch::zeros({(int)gmm_params.size(), 23});
    gmm_.inv_covs_ = torch::zeros({(int)gmm_params.size(), 23, 23});
    gmm_.cov_det_ = torch::zeros((int)gmm_params.size());

    int index = 0;
    for (auto entry : gmm_params) {
      gmm_.weights_.index_put_({index}, entry["weight"][0].get<float>());
      std::vector<float> m = entry["mean"].get<std::vector<float>>();
      std::vector<double> c = entry["cov"].get<std::vector<double>>();

      torch::Tensor cov = torch::from_blob(
          c.data(), {23, 23}, torch::TensorOptions().dtype(torch::kDouble));

      torch::Tensor det = torch::linalg::det(cov);
      std::cout << "det: " << det << std::endl;
      if (det.item<float>() < 1e-10) {
        cov = cov + torch::eye(23) * 0.01;
      }
      det = torch::linalg::det(cov);
      torch::Tensor inv_cov = torch::linalg::pinv(cov);

      gmm_.cov_det_.index_put_({index}, torch::linalg::det(cov));
      gmm_.means_.index_put_({index, torch::All},
                             torch::from_blob(m.data(), {23}));
      gmm_.inv_covs_.index_put_({index, torch::All}, inv_cov);

      index++;
    }
    std::cout << "cov det: " << gmm_.cov_det_ << std::endl;
  }

  // mano_model_left_.gpu();
  // mano_left_.gpu();
  // mano_model_right_.gpu();
  // mano_right_.gpu();

  // create a temporal mano model without PCA components
  // to generate HSV space
  reclib::models::Model<reclib::models::MANOConfig> mano_model_tmp(
      reclib::models::HandType::right);
  reclib::models::ModelInstance<reclib::models::MANOConfig> mano_tmp(
      mano_model_tmp);
  std::vector<vec4> tmp =
      reclib::models::mano_canonical_space_colors_hsv(mano_tmp, false, false);
  mano_verts2joints_ = mano_model_tmp.verts2joints;

  reverse_faces_ = torch::ones({mano_tmp.verts().rows(), 3}) * -1;
  for (unsigned int i = 0; i < mano_model_tmp.faces.rows(); i++) {
    for (unsigned int j = 0; j < 3; j++) {
      int vertex_index = mano_model_tmp.faces(i, j);
      int face_index = i;
      for (unsigned int k = 0; k < 3; k++) {
        if (reverse_faces_.index({vertex_index, (int)k}).item<int>() == -1) {
          reverse_faces_.index({vertex_index, (int)k}) = face_index;
        }
      }
    }
  }
  // std::vector<vec4> canonical_rgb =
  //     reclib::models::mano_canonical_space_colors_hsv(mano_tmp, false,
  //     true);

  for (unsigned int i = 0; i < tmp.size(); i++) {
    vec3 v = tmp[i].head<3>();
    v[0] = v[0] / 360.f;
    // v[1] = v[1] * 2;
    mano_corr_space_.push_back(v);
  }

  // if (debug_) {
  if (!reclib::opengl::Shader::valid("mano_shader")) {
    // reclib::opengl::Shader("mano_shader", "MVP_norm_color3.vs",
    // "color3PV.fs");
    reclib::opengl::Shader("mano_shader", "MVP_norm_color4.vs",
                           "pointlight3_color4PV.fs");
  }
  if (!reclib::opengl::Material::valid("mano_material")) {
    reclib::opengl::Material m("mano_material");
    // m->vec3_map["ambient_color"] = vec3(250, 214, 190).normalized();
    // m->vec3_map["diffuse_color"] = vec3(255, 255, 255).normalized();
    // m->vec3_map["specular_color"] = vec3(255, 255, 255).normalized();
    // m->vec3_map["light_dir"] = vec3(0, -0.5, -0.5).normalized();
    // m->vec3_map["light_pos"] = vec3(0, -1, 0);
    // m->float_map["opacity"] = 1;
    // m->float_map["shinyness"] = 2;
    // m->int_map["use_alpha"] = 1;
    m->vec3_map["pl1_pos"] = vec3(-1, -1, 0);
    m->vec3_map["pl2_pos"] = vec3(1, -1, 1);
    m->vec3_map["pl3_pos"] = vec3(0, 1, 0);

    m->float_map["pl1_attenuation"] = 1;
    m->float_map["pl2_attenuation"] = 1;
    m->float_map["pl3_attenuation"] = 1;
  }

  if (!reclib::opengl::Shader::valid("mano_joint_lines")) {
    reclib::opengl::Shader("mano_joint_lines", "MVP_color3_alpha.vs",
                           "color3PV_alphaPV.fs");
  }
  if (!reclib::opengl::Shader::valid("mano_joints")) {
    reclib::opengl::Shader("mano_joints", "MVP_alpha.vs",
                           "color3Uniform_alphaPV.fs");
  }

  if (!reclib::opengl::Material::valid("mano_joints")) {
    reclib::opengl::Material joint_lines_material("mano_joints");
    // joint_lines_material->vec3_map["color"] = vec3(1.f, 0.5f, 0.0f);

    joint_lines_material->vec3_map["color"] = vec3(0.2, 0.2, 0.2);
  }
  if (!reclib::opengl::Material::valid("mano_joint_lines")) {
    reclib::opengl::Material joint_lines_material("mano_joint_lines");
    joint_lines_material->vec3_map["color"] = vec3(0.0f, 0.2f, 0.8f);
  }

  // mat4 transform_left = reclib::translate(vec3(500, 0, 1000)) *
  //                       reclib::rotate(-M_PI / 2.f, vec3(1, 0, 1));
  // mat4 transform_right = reclib::translate(vec3(-500, 0, 1000)) *
  //                        reclib::rotate(-M_PI / 2.f, vec3(1, 0, 1));
  // Eigen::AngleAxisf t_r_aa(transform_right.block<3, 3>(0, 0));
  // Eigen::AngleAxisf t_l_aa(transform_left.block<3, 3>(0, 0));

  // mano_left_.pose().template head<3>() = (t_l_aa.axis() * t_l_aa.angle());
  // mano_right_.pose().template head<3>() = (t_r_aa.axis() * t_r_aa.angle());

  mano_left_.update(false, true);
  mano_right_.update(false, true);

  if (config_["Debug"]["show_hands"].as<bool>()) {
    std::vector<vec4> color_alphas_pv;
    for (unsigned int i = 0; i < mano_left_.verts().sizes()[0]; i++) {
      int joint = mano_verts2joints_[i];
      float alpha = 1;
      color_alphas_pv.push_back(
          vec4(SKIN_COLOR.x(), SKIN_COLOR.y(), SKIN_COLOR.z(), alpha));
    }
    std::vector<float> alphas_pj;
    for (unsigned int i = 0; i < 21; i++) {
      alphas_pj.push_back(1);
    }
    std::vector<vec3> color_pj;
    // root line
    color_pj.push_back(vec3(1, 1, 1));
    // index
    color_pj.push_back(vec3(0, 1, 0));
    color_pj.push_back(vec3(0, 1, 0));
    color_pj.push_back(vec3(0, 1, 0));
    // middle
    color_pj.push_back(vec3(1, 0.5, 0));
    color_pj.push_back(vec3(1, 0.5, 0));
    color_pj.push_back(vec3(1, 0.5, 0));
    // pinky
    color_pj.push_back(vec3(0, 0.5, 1));
    color_pj.push_back(vec3(0, 0.5, 1));
    color_pj.push_back(vec3(0, 0.5, 1));
    // ring
    color_pj.push_back(vec3(0.9, 0, 1));
    color_pj.push_back(vec3(0.9, 0, 1));
    color_pj.push_back(vec3(0.9, 0, 1));
    // thumb
    color_pj.push_back(vec3(1, 0, 0));
    color_pj.push_back(vec3(1, 0, 0));
    color_pj.push_back(vec3(1, 0, 0));
    // last joint to finger tip of each finger
    color_pj.push_back(vec3(0, 1, 0));
    color_pj.push_back(vec3(1, 0.5, 0));
    color_pj.push_back(vec3(0, 0.5, 1));
    color_pj.push_back(vec3(0.9, 0, 1));
    color_pj.push_back(vec3(1, 0, 0));

    std::cout << "alphas_pj: " << alphas_pj.size() << std::endl;

    // mano_right_.gl_instance->mesh->geometry->add_attribute_vec3(
    //     "color", mano_corr_space_.data(), mano_corr_space_.size(), false);

    if (config_["hand_mode"].as<int>() == 1 ||
        config_["hand_mode"].as<int>() == 2) {
      mano_right_.generate_gl_drawelem(
          reclib::opengl::Shader::find("mano_shader"),
          reclib::opengl::Material::find("mano_material"));

      mano_right_.gl_instance->mesh->geometry->float_map.erase("color");
      mano_right_.gl_instance->mesh->geometry->add_attribute_vec4(
          "color", color_alphas_pv.data(), color_alphas_pv.size(), false);
      mano_right_.gl_instance->wireframe_mode = false;
      mano_right_.gl_instance->add_pre_draw_func("pre",
                                                 [&]() { glLineWidth(5.f); });
      mano_right_.gl_instance->add_post_draw_func("post",
                                                  [&]() { glLineWidth(1.f); });

      mano_right_.gl_joints->shader =
          reclib::opengl::Shader::find("mano_joints");
      mano_right_.gl_joints->mesh->geometry->add_attribute_float("alpha",
                                                                 alphas_pj, 1);
      mano_right_.gl_joints->mesh->material =
          reclib::opengl::Material::find("mano_joints");

      mano_right_.gl_joint_lines->shader =
          reclib::opengl::Shader::find("mano_joint_lines");
      mano_right_.gl_joint_lines->mesh->geometry->add_attribute_vec3("color",
                                                                     color_pj);
      mano_right_.gl_joint_lines->mesh->geometry->add_attribute_float(
          "alpha", alphas_pj, 1);

      if (config_["Debug"]["highlight_joints"].as<bool>()) {
        mano_right_.gl_joints->add_pre_draw_func(
            "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
        mano_right_.gl_joints->add_post_draw_func(
            "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
        mano_right_.gl_joint_lines->add_pre_draw_func(
            "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
        mano_right_.gl_joint_lines->add_post_draw_func(
            "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
      }

      // if (config_["Debug"]["highlight_mesh"].as<bool>()) {
      //   mano_right_.gl_instance->add_pre_draw_func(
      //       "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
      //   mano_right_.gl_instance->add_post_draw_func(
      //       "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
      // }
    }

    if (config_["hand_mode"].as<int>() == 0 ||
        config_["hand_mode"].as<int>() == 2) {
      mano_left_.generate_gl_drawelem(
          reclib::opengl::Shader::find("mano_shader"),
          reclib::opengl::Material::find("mano_material"));

      mano_left_.gl_instance->mesh->geometry->float_map.erase("color");
      mano_left_.gl_instance->mesh->geometry->add_attribute_vec4(
          "color", color_alphas_pv.data(), color_alphas_pv.size(), false);
      mano_left_.gl_instance->wireframe_mode = false;
      mano_left_.gl_instance->add_pre_draw_func("pre",
                                                [&]() { glLineWidth(5.f); });
      mano_left_.gl_instance->add_post_draw_func("post",
                                                 [&]() { glLineWidth(1.f); });

      mano_left_.gl_joints->shader =
          reclib::opengl::Shader::find("mano_joints");
      mano_left_.gl_joints->mesh->geometry->add_attribute_float("alpha",
                                                                alphas_pj, 1);
      mano_left_.gl_joints->mesh->material =
          reclib::opengl::Material::find("mano_joints");

      mano_left_.gl_joint_lines->shader =
          reclib::opengl::Shader::find("mano_joint_lines");
      mano_left_.gl_joint_lines->mesh->geometry->add_attribute_vec3("color",
                                                                    color_pj);
      mano_left_.gl_joint_lines->mesh->geometry->add_attribute_float(
          "alpha", alphas_pj, 1);

      if (config_["Debug"]["highlight_joints"].as<bool>()) {
        mano_left_.gl_joints->add_pre_draw_func(
            "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
        mano_left_.gl_joints->add_post_draw_func(
            "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
        mano_left_.gl_joint_lines->add_pre_draw_func(
            "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
        mano_left_.gl_joint_lines->add_post_draw_func(
            "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
      }

      // if (config_["Debug"]["highlight_mesh"].as<bool>()) {
      //   mano_left_.gl_instance->add_pre_draw_func(
      //       "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
      //   mano_left_.gl_instance->add_post_draw_func(
      //       "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
      // }
    }

    // mano_left_.gl_instance->mesh->geometry->add_attribute_vec3(
    //     "color", mano_corr_space_.data(), mano_corr_space_.size(), false);
  }

  mano_left_.update(false, true);
  mano_right_.update(false, true);
  //}

  // solver_options_.use_inner_iterations = true;
  // solver_options_.use_nonmonotonic_steps = true;

  std::cout << config_ << std::endl;

  // if (!reclib::opengl::Shader::valid("vg_shader")) {
  //   reclib::opengl::Shader("vg_shader", "MVP.vs", "color4Uniform.fs");
  // }
  // if (!reclib::opengl::Material::valid("voxel_grid_left")) {
  //   reclib::opengl::Material m("voxel_grid_left");
  //   m->vec4_map["color"] = vec4(1, 0, 0, 1);
  // }
  // if (!reclib::opengl::Material::valid("nn_lines")) {
  //   reclib::opengl::Material m("nn_lines");
  //   m->vec4_map["color"] = vec4(1, 1, 0, 1);
  // }

  // vec3 center = (mano_right_.gl_instance->mesh->geometry->bb_max -
  //                mano_right_.gl_instance->mesh->geometry->bb_min) *
  //                   0.5 +
  //               mano_right_.gl_instance->mesh->geometry->bb_min;
  // float scale = config_["Voxels"]["scale"].as<float>();
  // vec3 vg_center(config_["Voxels"]["size"][0].as<float>() * scale * 0.5,
  //                config_["Voxels"]["size"][1].as<float>() * scale * 0.5,
  //                config_["Voxels"]["size"][2].as<float>() * scale * 0.5);

  // volume_translation_ = center - vg_center;
  // // volume_translation_ = vec3(0, 0, 0.4) - vg_center;

  // std::vector<reclib::opengl::Drawelement> drawelem;
  // for (unsigned int z = 0; z < volume_size_.z(); z++) {
  //   for (unsigned int y = 0; y < volume_size_.y(); y++) {
  //     for (unsigned int x = 0; x < volume_size_.x(); x++) {
  //       reclib::opengl::Cuboid c(
  //           "voxel_" + std::to_string(z) + "_" + std::to_string(y) + "_" +
  //               std::to_string(x),
  //           volume_scale_, volume_scale_, volume_scale_,
  //           vec3((x + 0.5) * volume_scale_, (y + 0.5) * volume_scale_,
  //                (z + 0.5) * volume_scale_),
  //           reclib::opengl::Material::find("voxel_grid_left"));
  //       // c->primitive_type = GL_LINES;
  //       reclib::opengl::Drawelement d(
  //           "voxel_" + std::to_string(z) + "_" + std::to_string(y) + "_" +
  //               std::to_string(x),
  //           reclib::opengl::Shader::find("vg_shader"), c);
  //       d->model.block<3, 1>(0, 3) = volume_translation_;
  //       d->set_wireframe_mode(true);
  //       drawelem.push_back(d);
  //     }
  //   }
  // }

  // gl_voxel_grid_ =
  //     reclib::opengl::GroupedDrawelements("voxelgrid_left", drawelem);

  // tsdf_weights_.setTo(0);
  // tsdf_volume_.setTo(0);

  // // reclib::tracking::ManoRGBD_Cuda::initialize_volume_weights(
  // //     tsdf_weights_, volume_size_, volume_scale_, volume_translation_,
  // //     mano_right_.verts().data(), mano_right_.verts().rows());

  // CpuMat weights_cpu(tsdf_weights_.size(), tsdf_weights_.type());
  // tsdf_weights_.download(weights_cpu);

  mano_left_.set_hand_pca(torch::matmul(mano_left_.model.hand_comps.inverse(),
                                        mano_left_.model.hand_mean)
                              .squeeze(1));

  mano_left_.update();
  mano_right_.set_hand_pca(torch::matmul(mano_right_.model.hand_comps.inverse(),
                                         mano_right_.model.hand_mean)
                               .squeeze(1));
  mano_right_.update();
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::extract_mesh() {
  reclib::kinfu::SurfaceMesh m =
      reclib::tracking::ManoRGBD_Cuda::marching_cubes(
          tsdf_volume_, volume_size_, volume_scale_,
          config_["Voxels"]["triangles_buffer_size"].as<int>());

  if (!reclib::opengl::Shader::valid("mc_mesh")) {
    reclib::opengl::Shader("mc_mesh", "MVP_norm.vs", "phong.fs");
  }

  if (!reclib::opengl::Material::valid("mc_mesh")) {
    reclib::opengl::Material mat("mc_mesh");
    mat->vec3_map["ambient_color"] = vec3(0.5, 0.1, 0.1);
    mat->vec3_map["diffuse_color"] = vec3(0.5, 0.1, 0.1);
    mat->vec3_map["specular_color"] = vec3(0.5, 0.1, 0.1);
    mat->vec3_map["light_dir"] = vec3(0, -1, 0);
    mat->float_map["opacity"] = 1;
    mat->float_map["shinyness"] = 2;
  }
  std::cout << "adding " << m.triangles_.cols << " triangles " << std::endl;
  if (!reclib::opengl::Drawelement::valid("mc_mesh")) {
    reclib::opengl::Drawelement d =
        reclib::opengl::DrawelementImpl::from_geometry(
            "mc_mesh", reclib::opengl::Shader::find("mc_mesh"),
            reclib::opengl::Material::find("mc_mesh"), false,
            (float*)m.triangles_.data, m.triangles_.cols, nullptr, 0,
            (float*)m.normals_.data, m.normals_.cols);
  } else {
    reclib::opengl::Drawelement d =
        reclib::opengl::Drawelement::find("mc_mesh");
    d->mesh->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        (float*)m.triangles_.data, m.triangles_.cols, nullptr, 0,
        (float*)m.normals_.data, m.normals_.cols);
    d->mesh->geometry->update_meshes();
  }
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::fuse() {
  int num_samples = last_frame_.network_output_[0].sizes()[0];

  int index = 0;
  for (int i = 0; i < num_samples; i++) {
    if (last_frame_.network_output_[0]
            .index({i})
            .to(torch::kCPU)
            .item()
            .toInt() == 1) {
      if (config_.i("hand_mode") == int(HandMode::RIGHT) ||
          config_.i("hand_mode") == int(HandMode::BOTH)) {
        index = i;
        break;
      }
    }
  }

  torch::Tensor box = last_frame_.network_output_[2].index({index});
  // CpuMat depth_map_last_frame_ = last_frame_.masked_depth_[index];
  CpuMat depth_map_last_frame_;
  last_frame_.depth_.copyTo(depth_map_last_frame_);
  depth_map_last_frame_.convertTo(depth_map_last_frame_, CV_32FC1);

  CpuMat normal_map_last_frame_ = last_frame_.normal_maps_[index];
  GpuMat depth(cv::cuda::createContinuous(depth_map_last_frame_.size(),
                                          depth_map_last_frame_.type()));
  GpuMat normals(cv::cuda::createContinuous(normal_map_last_frame_.size(),
                                            normal_map_last_frame_.type()));

  depth.upload(depth_map_last_frame_);
  normals.upload(normal_map_last_frame_);
  uvec2 pixel_offset(box.index({0}).item<int>(), box.index({1}).item<int>());
  GpuMat pixels(cv::cuda::createContinuous(tsdf_weights_.size(), CV_16UC2));
  pixels.setTo(0);

  // reclib::tracking::ManoRGBD_Cuda::fuse_surface(
  //     pixels, depth, normals, uvec2(0, 0), tsdf_volume_, tsdf_weights_,
  //     volume_size_, volume_scale_, intrinsics_, volume_translation_,
  //     mano_right_._vert_transforms.data(),
  //     mano_right_._vert_transforms.rows(),
  //     config_["Voxels"]["truncation_distance"].as<float>(),
  //     config_["Voxels"]["max_weight"].as<float>());

  CpuMat pixels_cpu;
  pixels.download(pixels_cpu);

  CpuMat texture(depth.size(), CV_8UC3);
  CpuMat color = last_frame_.rgb_;
  // cv::imshow("color", color);
  // cv::imshow("depth, ", last_frame_.depth_);
  // cv::waitKey(0);
  // cv::destroyAllWindows();

  texture.setTo(0);
  for (unsigned int i = 0; i < pixels_cpu.rows * pixels_cpu.cols; i++) {
    int x = pixels_cpu.ptr<uint16_t>(0)[i * 2 + 0];
    int y = pixels_cpu.ptr<uint16_t>(0)[i * 2 + 1];
    if (x != 0 || y != 0) {
      // ivec3 coord(i % pixels_cpu.cols, i / pixels_cpu.cols,
      //             i / pixels_cpu.rows);

      // std::cout << "CPU found (" << x << "," << y << ")" << std::endl;
      texture.ptr<uint8_t>(0)[(y * texture.cols + x) * 3 + 0] =
          color.ptr<uint8_t>(0)[(y * texture.cols + x) * 3 + 0];
      texture.ptr<uint8_t>(0)[(y * texture.cols + x) * 3 + 1] =
          color.ptr<uint8_t>(0)[(y * texture.cols + x) * 3 + 1];
      texture.ptr<uint8_t>(0)[(y * texture.cols + x) * 3 + 2] =
          color.ptr<uint8_t>(0)[(y * texture.cols + x) * 3 + 2];
    }
  }
  cv::imshow("tex", texture);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::update_voxel_grid() {
  GpuMat vp_cuda(
      cv::cuda::createContinuous(voxel_pos_.size(), voxel_pos_.type()));
  vp_cuda.upload(voxel_pos_);
  // reclib::tracking::ManoRGBD_Cuda::update_voxel_pos(
  //     vp_cuda, tsdf_weights_, volume_size_, volume_scale_,
  //     volume_translation_, mano_right_._vert_transforms.data(),
  //     mano_right_._vert_transforms.rows());
  vp_cuda.download(voxel_pos_);

  CpuMat weights(tsdf_weights_.size(), tsdf_weights_.type());
  tsdf_weights_.download(weights);
  // for (unsigned int z = 0; z < volume_size_.z(); z++) {
  //   for (unsigned int y = 0; y < volume_size_.y(); y++) {
  //     for (unsigned int x = 0; x < volume_size_.x(); x++) {
  //       uint32_t linearized_index = (z * volume_size_.y() *
  //       volume_size_.x()
  //       +
  //                                    y * volume_size_.x() + x);
  //       uint16_t index = weights.ptr(0)[linearized_index];
  //       vec3 vp((x + 0.5) * volume_scale_, (y + 0.5) * volume_scale_,
  //               (z + 0.5) * volume_scale_);

  //       Eigen::Matrix<float, 3, 4> T =
  //           mano_right_._vert_transforms.row(index).reshaped(4,
  //           3).transpose();
  //       T.block<3, 1>(0, 3) += T.block<3, 3>(0, 0) * volume_translation_;

  //       gl_voxel_grid_
  //           ->elems[z * volume_size_.y() * volume_size_.x() +
  //                   y * volume_size_.x() + x]
  //           ->model.block<3, 4>(0, 0) = T;
  //     }
  //   }
  // }
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::reset() {
  optimization_counter_ = 0;
  frame_counter_ = 0;
  mano_left_.set_zero();
  mano_left_.set_hand_pca(torch::matmul(mano_left_.model.hand_comps.inverse(),
                                        mano_left_.model.hand_mean)
                              .squeeze(1));
  mano_right_.set_zero();
  mano_right_.set_hand_pca(torch::matmul(mano_right_.model.hand_comps.inverse(),
                                         mano_right_.model.hand_mean)
                               .squeeze(1));
  mano_left_.update();
  mano_right_.update();

  for (unsigned int i = 0; i < last_frame_.pose_.size(); i++) {
    last_frame_.visibility_map_[i] = std::vector<uint8_t>(20);
    last_frame_.trans_[i] = torch::zeros(
        3, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.rot_[i] = torch::zeros(
        3, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.pose_[i] = torch::zeros(
        23, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.shape_[i] = torch::zeros(
        10, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.loss_[i] = -1;
  }

  std::cout << "resetting tracker..." << std::endl;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::restore_left() {
  optimization_counter_ = 0;
  frame_counter_ = 0;
  mano_left_.set_rot(last_frame_.rot_[0]);
  mano_left_.set_shape(last_frame_.shape_[0]);
  mano_left_.set_hand_pca(last_frame_.pca_[0]);
  mano_left_.set_trans(last_frame_.trans_[0]);
  mano_left_.update();

  std::cout << "restoring tracker LEFT..." << std::endl;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::restore_right() {
  optimization_counter_ = 0;
  frame_counter_ = 0;
  mano_right_.set_rot(last_frame_.rot_[1]);
  mano_right_.set_shape(last_frame_.shape_[1]);
  mano_right_.set_hand_pca(last_frame_.pca_[1]);
  mano_right_.set_trans(last_frame_.trans_[1]);
  mano_right_.update();

  std::cout << "restoring tracker RIGHT..." << std::endl;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::reset_left() {
  optimization_counter_ = 0;
  mano_left_.set_zero();
  mano_left_.set_hand_pca(torch::matmul(mano_left_.model.hand_comps.inverse(),
                                        mano_left_.model.hand_mean)
                              .squeeze(1));
  mano_left_.update();

  for (unsigned int i = 0; i < 1; i++) {
    last_frame_.visibility_map_[i] = std::vector<uint8_t>(20);
    last_frame_.trans_[i] = torch::zeros(
        3, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.rot_[i] = torch::zeros(
        3, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.pose_[i] = torch::zeros(
        23, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.shape_[i] = torch::zeros(
        10, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.loss_[i] = -1;
  }

  std::cout << "resetting tracker LEFT..." << std::endl;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::reset_right() {
  optimization_counter_ = 0;
  mano_right_.set_zero();
  mano_right_.set_hand_pca(torch::matmul(mano_right_.model.hand_comps.inverse(),
                                         mano_right_.model.hand_mean)
                               .squeeze(1));
  mano_right_.update();

  for (unsigned int i = 1; i < 2; i++) {
    last_frame_.visibility_map_[i] = std::vector<uint8_t>(20);
    last_frame_.trans_[i] = torch::zeros(
        3, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.rot_[i] = torch::zeros(
        3, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.pose_[i] = torch::zeros(
        23, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.shape_[i] = torch::zeros(
        10, torch::TensorOptions().device(mano_right_.trans().device()));
    last_frame_.loss_[i] = -1;
  }

  std::cout << "resetting tracker RIGHT..." << std::endl;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::prepare_correspondences(
    const CorrespondenceMode& mode, const torch::Tensor& box,
    const torch::Tensor& mask, const torch::Tensor& corr,
    CpuMat& input_output_depth, CpuMat& output_corr, float& output_mean_depth) {
  _RECLIB_ASSERT_EQ(mask.sizes()[0], input_output_depth.rows);
  _RECLIB_ASSERT_EQ(mask.sizes()[1], input_output_depth.cols);
  _RECLIB_ASSERT_EQ(corr.sizes()[0], 3);
  _RECLIB_ASSERT_EQ(corr.sizes()[1], input_output_depth.rows);
  _RECLIB_ASSERT_EQ(corr.sizes()[2], input_output_depth.cols);

  reclib::Configuration corr_config;
  if (mode == (CorrespondenceMode::REGISTRATION)) {
    corr_config = config_.subconfig({"Correspondences_Registration"});
  } else {
    corr_config = config_.subconfig({"Correspondences_Optimization"});
  }

  if (debug_) {
    timer_.look_and_reset();
  }

  torch::Tensor corr_masked =
      corr.to(torch::kCPU) *
      mask.to(torch::kCPU).index({torch::None, torch::All, torch::All});
  corr_masked[0] *= 360;
  corr_masked = corr_masked.permute({1, 2, 0});
  corr_masked =
      corr_masked.index({torch::indexing::Slice(box.index({1}).item<int>(),
                                                box.index({3}).item<int>()),
                         torch::indexing::Slice(box.index({0}).item<int>(),
                                                box.index({2}).item<int>())});
  corr_masked = corr_masked.contiguous();

  if (corr_masked.sizes().size() == 3 && corr_masked.sizes()[0] > 0 &&
      corr_masked.sizes()[1] > 0) {
    output_corr = reclib::dnn::torch2cv(corr_masked, true);
    cv::cvtColor(output_corr, output_corr, cv::COLOR_HSV2RGB);
  } else {
    output_corr.create(1, 1, CV_32FC3);
    output_corr.setTo(0);
  }

  input_output_depth.convertTo(input_output_depth, CV_32FC1);
  torch::Tensor depth_tensor = reclib::dnn::cv2torch(input_output_depth, true);
  torch::Tensor masked_depth =
      (depth_tensor *
       mask.to(torch::kCPU).index({torch::All, torch::All, torch::None}));

  masked_depth =
      masked_depth.index({torch::indexing::Slice(box.index({1}).item<int>(),
                                                 box.index({3}).item<int>()),
                          torch::indexing::Slice(box.index({0}).item<int>(),
                                                 box.index({2}).item<int>())});

  int median_index = int((masked_depth > 0).sum().item().toFloat() / 2.f);

  if ((masked_depth > 0).sum().item().toFloat() > 0) {
    output_mean_depth =
        std::get<0>(masked_depth.index({masked_depth > 0}).reshape(-1).sort())
            .index({median_index})
            .item()
            .toFloat() /
        1000.f;
  } else {
    output_mean_depth = 0;
  }

  masked_depth = masked_depth.contiguous();

  // float deviation =
  // corr_config.f("pruning_deviation");  // 10cm

  // masked_depth = masked_depth * (masked_depth < (mean + deviation));
  // masked_depth = masked_depth * (masked_depth > (mean - deviation));

  //  masked_depth = masked_depth.contiguous();

  if (masked_depth.sizes().size() == 3 && masked_depth.sizes()[0] > 0 &&
      masked_depth.sizes()[1] > 0) {
    input_output_depth = reclib::dnn::torch2cv(masked_depth, true);
  } else {
    input_output_depth.create(1, 1, CV_32FC1);
    input_output_depth.setTo(0);
  }

  // input_output_depth = input_output_depth;  // / 1000.f;

  if (debug_) {
    std::cout << "---- Preparing correspondences: " << timer_.look_and_reset()
              << " ms." << std::endl;
  }

  // cv::imshow("Depth", input_output_depth);
  // cv::imshow("Corr", output_corr);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
}

template <typename MANOTYPE>
std::vector<torch::Tensor> reclib::tracking::ManoRGBD<MANOTYPE>::reduce(
    std::vector<torch::Tensor>& in) {
  int num_samples = in[0].sizes()[0];

  int left_index = -1;
  int right_index = -1;

  // std::cout << "classes: " << in[0] << std::endl;
  // std::cout << "scores_ " << in[1] << std::endl;

  for (int i = 0; i < num_samples; i++) {
    int cls = in[0].index({i}).to(torch::kCPU).item().toInt();
    torch::Tensor score = in[1].index({i}).to(torch::kCPU);

    if (cls == 0 && left_index < 0 &&
        (config_["hand_mode"].as<int>() == 0 ||
         config_["hand_mode"].as<int>() == 2)) {
      left_index = i;
    }
    if (cls == 1 && right_index < 0 &&
        (config_["hand_mode"].as<int>() == 1 ||
         config_["hand_mode"].as<int>() == 2)) {
      right_index = i;
    }

    if (left_index >= 0 && right_index >= 0) break;
  }

  std::vector<torch::Tensor> out;
  std::vector<int> indices;

  if (left_index >= 0 && right_index >= 0) {
    indices.push_back(left_index);
    indices.push_back(right_index);
  } else if (left_index >= 0) {
    indices.push_back(left_index);

  } else if (right_index >= 0) {
    indices.push_back(right_index);
  }

  torch::Tensor ind = torch::tensor(indices);
  for (unsigned int i = 0; i < in.size(); i++) {
    if (in[i].sizes()[0] == 1) {
      std::vector<torch::Tensor> stacked;
      for (unsigned int j = 0; j < ind.sizes()[0]; j++) {
        stacked.push_back(in[i]);
      }
      torch::Tensor t = torch::cat(stacked, 0);
      out.push_back(t);
    } else {
      out.push_back(in[i].index({ind}));
    }
  }
  return out;
}

static int hsv2cls(vec3 hsv) {
  if (hsv.x() == 0 && hsv.y() == 0 && hsv.z() == 0) {
    return 30;
  }
  hsv.x() *= 360;

  int hsv_finger_seg =
      fmax(0, fmin(int(((hsv.y() - 0.58 - 0.02 * (1 - hsv.z())) / (0.139)) + 1),
                   3));  // yields 0, 1, 2 or 3
  int hsv_finger = 0;
  if (hsv.x() < 180) {
    hsv_finger =
        std::min(int(fmax(0, int(hsv.x() - 20)) / (80.0)), 1);  // yields 0 or 1
  } else {
    hsv_finger = std::min(int(fmax(0, int(hsv.x() - 180)) / (55.5)) + 2,
                          4);  // yields 2,3, or 4
  }
  return hsv_finger * 4 + hsv_finger_seg;
}

vec3 hsv2classification(vec3 hsv) {
  if (0) {
    hsv[0] *= 360;
    if (hsv[0] < 100) {
      // thumb
      if (hsv[1] < 0.58) {
        return (vec3(0.4, 0, 0));
      } else if (hsv[1] >= 0.58 && hsv[1] < 0.76) {
        return (vec3(0.6, 0, 0));
      } else if (hsv[1] >= 0.76 && hsv[1] < 0.89) {
        return (vec3(0.8, 0, 0));
      } else {
        return (vec3(1, 0, 0));
      }
    } else if (hsv[0] < 180) {
      // index
      if (hsv[1] < 0.58) {
        return (vec3(0.4, 0, 0));
      } else if (hsv[1] >= 0.58 && hsv[1] < 0.76) {
        return (vec3(1, 0, 0.6));

      } else if (hsv[1] >= 0.76 && hsv[1] < 0.89) {
        return (vec3(1, 0, 0.8));

      } else {
        return (vec3(1, 0, 1));
      }
    } else if (hsv[0] < 235) {
      if (hsv[1] < 0.58) {
        return (vec3(0.4, 0, 0));
      } else if (hsv[1] >= 0.58 && hsv[1] < 0.76) {
        return (vec3(0, 0.6, 0));
      } else if (hsv[1] >= 0.76 && hsv[1] < 0.89) {
        return (vec3(0, 0.8, 0));
      } else {
        return (vec3(0, 1, 0));
      }
    } else if (hsv[0] < 295) {
      if (hsv[1] < 0.58) {
        return (vec3(0.4, 0, 0));

      } else if (hsv[1] >= 0.58 && hsv[1] < 0.76) {
        return (vec3(0, 0, 0.6));
      } else if (hsv[1] >= 0.76 && hsv[1] < 0.89) {
        return (vec3(0, 0, 0.8));
      } else {
        return (vec3(0, 0, 1));
      }
    } else if (hsv[0] < 360) {
      if (hsv[1] < 0.58) {
        return (vec3(0.4, 0, 0));
      } else if (hsv[1] >= 0.58 && hsv[1] < 0.76) {
        return (vec3(1, 0.6, 0));
      } else if (hsv[1] >= 0.76 && hsv[1] < 0.89) {
        return (vec3(1, 0.8, 0));
      } else {
        return (vec3(1, 1, 0));
      }
    } else {
      return (vec3(0, 0, 0));
    }
  } else {
    if (hsv[0] == 0 || hsv[0] == 4 || hsv[0] == 8 || hsv[0] == 12 ||
        hsv[0] == 16) {
      return (vec3(0.4, 0, 0));
    }
    if (hsv[0] == 1) {
      return (vec3(0.6, 0, 0));
    }
    if (hsv[0] == 2) {
      return (vec3(0.8, 0, 0));
    }
    if (hsv[0] == 3) {
      return (vec3(1, 0, 0));
    }
    if (hsv[0] == 5) {
      return (vec3(1, 0, 0.6));
    }
    if (hsv[0] == 6) {
      return (vec3(1, 0, 0.8));
    }
    if (hsv[0] == 7) {
      return (vec3(1, 0, 1));
    }
    if (hsv[0] == 9) {
      return (vec3(0, 0.6, 0));
    }
    if (hsv[0] == 10) {
      return (vec3(0, 0.8, 0));
    }
    if (hsv[0] == 11) {
      return (vec3(0, 1, 0));
    }
    if (hsv[0] == 13) {
      return (vec3(0, 0, 0.6));
    }
    if (hsv[0] == 14) {
      return (vec3(0, 0, 0.8));
    }
    if (hsv[0] == 15) {
      return (vec3(0, 0, 1));
    }
    if (hsv[0] == 17) {
      return (vec3(1, 0.6, 0));
    }
    if (hsv[0] == 18) {
      return (vec3(1, 0.8, 0));
    }
    if (hsv[0] == 19) {
      return (vec3(1, 1, 0));
    }
    return vec3(0, 0, 0);
  }
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::compute_correspondences(
    const CorrespondenceMode& mode, int index,
    reclib::modelstorch::ModelInstance<MANOTYPE>& hand) {
  int cls_label = last_frame_.network_output_[0]
                      .index({index})
                      .to(torch::kCPU)
                      .item()
                      .toInt();

  torch::Tensor box = last_frame_.network_output_[2].index({index});

  if (optimization_counter_ == 0) {
    CpuMat masked_depth = last_frame_.depth_.clone();
    CpuMat masked_corr;

    prepare_correspondences(mode, last_frame_.network_output_[2].index({index}),
                            last_frame_.network_output_[3].index({index}),
                            last_frame_.network_output_[4].index({index}),
                            masked_depth, masked_corr,
                            last_frame_.mean_depth_[cls_label]);

    last_frame_.corrs_[cls_label] = (masked_corr);
    last_frame_.masked_depth_[cls_label] = (masked_depth);
  }

  if (config_["Debug"]["show_prepared_corr"].as<bool>()) {
    cv::destroyAllWindows();
    cv::imshow("Depth", last_frame_.masked_depth_[cls_label]);
    cv::imshow("Corr", last_frame_.corrs_[cls_label]);
    cv::waitKey(0);
    // cv::destroyAllWindows();
  }

  reclib::Configuration corr_config;
  if (mode == (CorrespondenceMode::REGISTRATION)) {
    corr_config = config_.subconfig({"Correspondences_Registration"});
  } else {
    corr_config = config_.subconfig({"Correspondences_Optimization"});
  }

  if (debug_) {
    timer_.look_and_reset();
  }

  if (optimization_counter_ == 0) {
    GpuMat raw_vertex_map = cv::cuda::createContinuous(
        last_frame_.masked_depth_[cls_label].size(), CV_32FC3);
    GpuMat depth_gpu(last_frame_.masked_depth_[cls_label].size(),
                     last_frame_.masked_depth_[cls_label].type());
    depth_gpu.upload(last_frame_.masked_depth_[cls_label]);

    reclib::cuda::ComputeVertexMap(
        depth_gpu, raw_vertex_map, intrinsics_.Level(0), -1,
        ivec2(box.index({0}).item<int>(), box.index({1}).item<int>()), 0.001f);

    GpuMat raw_normal_map = cv::cuda::createContinuous(
        last_frame_.masked_depth_[cls_label].size(), CV_32FC3);
    // reclib::cuda::ComputeNormalMap(raw_vertex_map, raw_normal_map);

    reclib::cuda::ComputePrunedVertexNormalMap(
        raw_vertex_map, raw_normal_map, corr_config.f("pc_normal_thresh"));

    last_frame_.vertex_maps_[cls_label].create(raw_vertex_map.size(),
                                               raw_vertex_map.type());
    last_frame_.normal_maps_[cls_label].create(raw_normal_map.size(),
                                               raw_normal_map.type());
    raw_vertex_map.download(last_frame_.vertex_maps_[cls_label]);
    raw_normal_map.download(last_frame_.normal_maps_[cls_label]);
  }

  if (debug_) {
    std::cout << "---- Vertex and normal map computation: "
              << timer_.look_and_reset() << " ms." << std::endl;
  }

  if (mode == (CorrespondenceMode::REGISTRATION)) {
    last_frame_.corr_indices_[cls_label].reset();
    last_frame_.corr_indices_[cls_label].comp_canonical_corr_ = true;
    last_frame_.corr_indices_[cls_label].comp_silhouette_corr_ = false;

  } else if (mode == (CorrespondenceMode::OPTIMIZATION) &&
             optimization_counter_ == 1) {
    // this is the multiple frame case where we are at frame > 0 and do not
    // need registration anymore
    last_frame_.corr_indices_[cls_label].comp_canonical_corr_ = true;
    last_frame_.corr_indices_[cls_label].comp_silhouette_corr_ =
        config_["Optimization"]["visibility_weight"].as<float>() > 0;
  } else {
    last_frame_.corr_indices_[cls_label].comp_canonical_corr_ = true;  // false
    last_frame_.corr_indices_[cls_label].comp_silhouette_corr_ =
        config_["Optimization"]["visibility_weight"].as<float>() > 0;
  }

  CpuMat corrs_hsv_space = last_frame_.corrs_[cls_label].clone();
  cv::cvtColor(corrs_hsv_space, corrs_hsv_space, cv::COLOR_RGB2HSV);
  for (unsigned int c = 0; c < corrs_hsv_space.cols; c++) {
    for (unsigned int r = 0; r < corrs_hsv_space.rows; r++) {
      corrs_hsv_space.ptr<vec3>(0)[r * corrs_hsv_space.cols + c][0] /= 360;
      // corrs_hsv_space.ptr<vec3>(0)[r * corrs_hsv_space.cols + c][1] *= 2;
    }
  }

  if (optimization_counter_ == 0) {
    GpuMat seg(last_frame_.corrs_[cls_label].size(),
               last_frame_.corrs_[cls_label].type());
    seg.upload(corrs_hsv_space);
    reclib::tracking::cuda::compute_segmentation(
        seg, last_frame_.visibility_map_[cls_label]);
    seg.download(last_frame_.corrs_segmented_[cls_label]);
  }

  torch::Tensor verts_c = hand.verts().cpu().contiguous();

  // last_frame_.corr_indices_[cls_label].compute(
  //     corr_config, verts_c.data<float>(), verts_c.sizes()[0],
  //     hand.gl_instance->mesh->geometry->normals_ptr(), mano_verts2joints_,
  //     mano_corr_space_, last_frame_.vertex_maps_[cls_label],

  //     last_frame_.normal_maps_[cls_label], corrs_hsv_space,
  //     last_frame_.corrs_segmented_[cls_label], intrinsics_,
  //     ivec2(box.index({0}).item<int>(), box.index({1}).item<int>()),
  //     last_frame_.mean_depth_[cls_label]);

  last_frame_.corr_indices_[cls_label].compute_points2mano(
      corr_config, verts_c.data<float>(), verts_c.sizes()[0],
      mano_verts2joints_, mano_corr_space_, last_frame_.vertex_maps_[cls_label],
      last_frame_.normal_maps_[cls_label], corrs_hsv_space,
      last_frame_.corrs_segmented_[cls_label], intrinsics_,
      ivec2(box.index({0}).item<int>(), box.index({1}).item<int>()),
      last_frame_.mean_depth_[cls_label]);

  if (debug_) {
    std::cout << "---- Computing correspondences: " << timer_.look_and_reset()
              << " ms." << std::endl;

    std::cout
        << "Num correspondences: "
        << last_frame_.corr_indices_[cls_label].pc_corr_.canonical_ind_.size()
        << std::endl;
  }

  const ManoRGBDCorrespondences& corr_pairs =
      last_frame_.corr_indices_.at(last_frame_.network_output_[0]
                                       .index({index})
                                       .to(torch::kCPU)
                                       .item<int>());

  // last_frame_.corr_indices_[last_frame_.network_output_[0]
  //                               .index({index})
  //                               .to(torch::kCPU)
  //                               .item()
  //                               .toInt()] = (corr_pairs);

  const ManoRGBDCorrespondences& corr_indices =
      last_frame_.corr_indices_.at(last_frame_.network_output_[0]
                                       .index({index})
                                       .to(torch::kCPU)
                                       .item<int>());

  if (config_["Debug"]["show_corr_img"].as<bool>() == true) {
    CpuMat last_frame_corrs(last_frame_.corrs_segmented_[cls_label].size(),
                            last_frame_.corrs_segmented_[cls_label].type());
    CpuMat corrs_segmented = last_frame_.corrs_segmented_[cls_label].clone();
    // CpuMat last_frame_corrs = last_frame_.corrs_[cls_label];
    for (unsigned int r = 0; r < corrs_segmented.rows; r++) {
      for (unsigned int c = 0; c < corrs_segmented.cols; c++) {
        vec3 v = corrs_segmented.ptr<vec3>(0)[r * corrs_segmented.cols + c];
        vec3 v_cls = hsv2classification(v);
        last_frame_corrs.ptr<vec3>(0)[r * last_frame_corrs.cols + c] = v_cls;
      }
    }

    std::string name = "pc_frame_masked_" + std::to_string(index);

    if (0) {
      CpuMat rgb_float;
      last_frame_.rgb_.convertTo(rgb_float, CV_32FC3);
      cv::cvtColor(rgb_float, rgb_float, cv::COLOR_BGR2RGB);
      torch::Tensor rgb = reclib::dnn::cv2torch(rgb_float) / 255;
      torch::Tensor mask = last_frame_.network_output_[3].clone().squeeze(0);
      torch::Tensor masked_rgb =
          (rgb *
           mask.to(torch::kCPU).index({torch::All, torch::All, torch::None}));

      masked_rgb = masked_rgb.index(
          {torch::indexing::Slice(box.index({1}).item<int>(),
                                  box.index({3}).item<int>()),
           torch::indexing::Slice(box.index({0}).item<int>(),
                                  box.index({2}).item<int>())});
      masked_rgb = masked_rgb.contiguous();
      CpuMat masked_rgb_ = reclib::dnn::torch2cv(masked_rgb);
      reclib::opengl::Mesh m = reclib::opengl::pointcloud_norm_color(
          last_frame_.masked_depth_[cls_label], intrinsics_, masked_rgb_, true,
          name, last_frame_.vertex_maps_[cls_label],
          ivec2(last_frame_.network_output_[2]
                    .index({index})
                    .index({0})
                    .item<int>(),
                last_frame_.network_output_[2]
                    .index({index})
                    .index({1})
                    .item<int>()),
          0.001f);
    }

    reclib::opengl::Mesh m = reclib::opengl::pointcloud_norm_color(
        last_frame_.masked_depth_[cls_label], intrinsics_, last_frame_corrs,
        true, name, last_frame_.vertex_maps_[cls_label],
        ivec2(last_frame_.network_output_[2]
                  .index({index})
                  .index({0})
                  .item<int>(),
              last_frame_.network_output_[2]
                  .index({index})
                  .index({1})
                  .item<int>()),
        0.001f);

    if (!reclib::opengl::Drawelement::valid(name)) {
      reclib::opengl::Drawelement d(name,
                                    reclib::opengl::Shader::find("colorPV"), m);
      d->add_pre_draw_func("pre", [&]() { glPointSize(5.f); });
      d->add_post_draw_func("pre", [&]() { glPointSize(1.f); });
    } else {
      reclib::opengl::Drawelement d = reclib::opengl::Drawelement::find(name);
      d->mesh = m;
    }

    generated_drawelements_[name] = reclib::opengl::Drawelement::find(name);
  }
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::compute_correspondences(
    const CorrespondenceMode& mode) {
  int num_samples = last_frame_.network_output_[0].sizes()[0];

  for (int i = 0; i < num_samples; i++) {
    if (last_frame_.network_output_[0]
            .index({i})
            .to(torch::kCPU)
            .item()
            .toInt() == 0) {
      if (config_.i("hand_mode") == int(HandMode::LEFT) ||
          config_.i("hand_mode") == int(HandMode::BOTH)) {
        compute_correspondences(mode, i, mano_left_);
      }
    } else if (last_frame_.network_output_[0]
                   .index({i})
                   .to(torch::kCPU)
                   .item()
                   .toInt() == 1) {
      if (config_.i("hand_mode") == int(HandMode::BOTH)) {
        if (num_samples == 1) {
          compute_correspondences(mode, 0, mano_right_);
        } else {
          compute_correspondences(mode, i, mano_right_);
        }

      } else if (config_.i("hand_mode") == int(HandMode::RIGHT)) {
        compute_correspondences(mode, 0, mano_right_);
      }
    } else {
      throw std::runtime_error("Unknown class label: " +
                               std::to_string(last_frame_.network_output_[0]
                                                  .index({i})
                                                  .to(torch::kCPU)
                                                  .item()
                                                  .toInt()));
    }
  }
}

template <typename MANOTYPE>
std::pair<bool, bool> reclib::tracking::ManoRGBD<MANOTYPE>::register_hand(
    bool compute_rotation, bool compute_translation) {
  if (last_frame_.network_output_.size() == 0) {
    std::cout << "output is empty. quit." << std::endl;
    return std::make_pair(false, false);
  }

  bool result_left = true;
  bool result_right = true;
  int num_samples = last_frame_.network_output_[0].sizes()[0];
  for (int i = 0; i < num_samples; i++) {
    if (last_frame_.network_output_[0]
            .index({i})
            .to(torch::kCPU)
            .item()
            .toInt() == 0) {
      if (config_.i("hand_mode") == int(HandMode::LEFT) ||
          config_.i("hand_mode") == int(HandMode::BOTH)) {
        result_left =
            register_hand(0, compute_rotation, compute_translation, mano_left_);
      }
    } else if (last_frame_.network_output_[0]
                   .index({i})
                   .to(torch::kCPU)
                   .item()
                   .toInt() == 1) {
      if (config_.i("hand_mode") == int(HandMode::BOTH)) {
        if (num_samples == 2) {
          result_right = register_hand(1, compute_rotation, compute_translation,
                                       mano_right_);
        } else if (num_samples == 1) {
          result_right = register_hand(0, compute_rotation, compute_translation,
                                       mano_right_);
        }

      } else if (config_.i("hand_mode") == int(HandMode::RIGHT)) {
        result_right = register_hand(0, compute_rotation, compute_translation,
                                     mano_right_);
      }
    } else {
      throw std::runtime_error("Unknown class label: " +
                               std::to_string(last_frame_.network_output_[0]
                                                  .index({i})
                                                  .to(torch::kCPU)
                                                  .item()
                                                  .toInt()));
    }
  }

  optimization_counter_++;

  return std::make_pair(result_left, result_right);
}

template <typename MANOTYPE>
bool reclib::tracking::ManoRGBD<MANOTYPE>::register_hand(
    unsigned int index, bool compute_rotation, bool compute_translation,
    reclib::modelstorch::ModelInstance<MANOTYPE>& hand) {
  reclib::Configuration reg_config = config_.subconfig(
      reclib::optim::RegistrationConfiguration::YAML_PREFIXES);
  reclib::Configuration corr_config =
      config_.subconfig({"Correspondences_Registration"});

  if (debug_) {
    timer_.look_and_reset();
  }

  int cls_label = last_frame_.network_output_[0]
                      .index({(int)index})
                      .to(torch::kCPU)
                      .item()
                      .toInt();
  // const ManoRGBDCorrespondences& corr_pairs =
  //     last_frame_.corr_indices_.at(index);
  ManoRGBDCorrespondences& corr_pairs = last_frame_.corr_indices_.at(cls_label);

  if (corr_pairs.mano_corr_.canonical_ind_.size() == 0) {
    std::cout << "No correspondences. Abort." << std::endl;
    return false;
  }
  if (hand.trans().sum().template item<float>() > 0) {
    std::cout << "Hand is already registered." << std::endl;
    return true;
  }
  CpuMat vertex_map_last_frame_ = last_frame_.vertex_maps_[cls_label];
  CpuMat normal_map_last_frame_ = last_frame_.normal_maps_[cls_label];

  mat4 incr_trans = mat4::Identity();
  incr_trans.block<3, 1>(0, 3) =
      reclib::make_vec3(hand.trans().template data<float>());
  vec3 rot = reclib::make_vec3(hand.pose().template data<float>());
  Eigen::AngleAxisf aa(rot.norm(), rot.normalized());
  incr_trans.block<3, 3>(0, 0) = aa.matrix();

  if (debug_) {
    timer_.look_and_reset();
  }

  if (corr_pairs.mano_corr_.canonical_ind_.size() > 0) {
    for (unsigned int i = 0; i < reg_config.ui("iterations"); i++) {
      if (reg_config.b("register_rotation") && compute_rotation) {
        Sophus::SE3<float> rigid_trans = reclib::optim::pointToPointDirect(
            (float*)hand.verts().template data<float>(),
            hand.verts().sizes()[0], (float*)vertex_map_last_frame_.data,
            vertex_map_last_frame_.cols * vertex_map_last_frame_.rows,
            corr_pairs.mano_corr_.canonical_ind_,
            corr_pairs.pc_corr_.canonical_ind_);
        incr_trans = rigid_trans.matrix() * incr_trans;
        Eigen::AngleAxisf aa(incr_trans.matrix().block<3, 3>(0, 0));
        vec3 a = aa.angle() * aa.axis();
        hand.apose().index({0}) = a.x();
        hand.apose().index({1}) = a.y();
        hand.apose().index({2}) = a.z();

        hand.trans().index({0}) = incr_trans(0, 3);
        hand.trans().index({1}) = incr_trans(1, 3);
        hand.trans().index({2}) = incr_trans(2, 3);
        hand.update();
      }

      if (reg_config.b("register_translation") && compute_translation) {
        vec3 trans = reclib::optim::MeanTranslation(
            (float*)hand.verts().template data<float>(),
            hand.verts().sizes()[0], (float*)vertex_map_last_frame_.data,
            vertex_map_last_frame_.cols * vertex_map_last_frame_.rows,
            corr_pairs.mano_corr_.canonical_ind_,
            corr_pairs.pc_corr_.canonical_ind_);
        incr_trans = reclib::translate(trans) * incr_trans;
        hand.trans().index({0}) = incr_trans(0, 3);
        hand.trans().index({1}) = incr_trans(1, 3);
        hand.trans().index({2}) = incr_trans(2, 3);
        hand.update();
      }
    }
  }

  // hand.trans().index({2}) = hand.trans().index({2}) - 0.2;
  // hand.update();

  if (debug_) {
    std::cout << "---- Registration: " << timer_.look_and_reset() << " ms."
              << std::endl;
  }

  if (config_["Debug"]["show_corr_lines"].as<bool>()) {
    torch::Tensor mano_corr_ind_canonical =
        torch::from_blob((void*)corr_pairs.mano_corr_.canonical_ind_.data(),
                         {(int)corr_pairs.mano_corr_.canonical_ind_.size()},
                         torch::TensorOptions().dtype(torch::kInt))
            .toType(torch::kLong);
    torch::Tensor mano_corr_ind_silhouette =
        torch::from_blob((void*)corr_pairs.mano_corr_.silhouette_ind_.data(),
                         {(int)corr_pairs.mano_corr_.silhouette_ind_.size()},
                         torch::TensorOptions().dtype(torch::kInt))
            .toType(torch::kLong);
    torch::Tensor corr_pos_ref_canonical =
        hand.verts().index({mano_corr_ind_canonical});
    torch::Tensor corr_pos_ref_silhouette =
        hand.verts().index({mano_corr_ind_silhouette});
    // torch::Tensor corr_pos_ref =
    //     torch::cat({corr_pos_ref_canonical, corr_pos_ref_silhouette});
    torch::Tensor corr_pos_ref = torch::cat({corr_pos_ref_canonical});

    std::vector<vec3> corr_pos_src;
    std::vector<vec3> corr_colors(corr_pairs.mano_corr_.canonical_ind_.size() *
                                  2);
    std::vector<uint32_t> indices;
    for (unsigned int i = 0; i < corr_pairs.mano_corr_.canonical_ind_.size();
         i++) {
      vec3 color = vec3::Random().normalized().cwiseAbs();
      // color.x() = 0;
      // corr_colors.push_back(color);
      // corr_colors.push_back(color);
      corr_colors[i] = color;
      corr_colors[corr_pos_ref.sizes()[0] + i] = color;

      vec3 src(&(last_frame_.vertex_maps_[cls_label].ptr<float>(
          0)[corr_pairs.pc_corr_.canonical_ind_[i] * 3]));

      corr_pos_src.push_back(src);
      indices.push_back(i);
      indices.push_back(corr_pos_ref.sizes()[0] + i);
    }
    // for (unsigned int i = 0; i <
    // corr_pairs.mano_corr_.silhouette_ind_.size();
    //      i++) {
    //   vec3 color = vec3::Random().normalized().cwiseAbs();
    //   // color.y() = 0;
    //   // color.z() = 0;
    //   corr_colors.push_back(color);
    //   corr_colors.push_back(color);

    //   vec3 src(&(last_frame_.vertex_maps_[cls_label].ptr<float>(
    //       0)[corr_pairs.pc_corr_.silhouette_ind_[i] * 3]));

    //   corr_pos_src.push_back(src);
    //   indices.push_back(i + corr_pairs.mano_corr_.canonical_ind_.size());
    //   indices.push_back(corr_pos_ref.sizes()[0] + i +
    //                     corr_pairs.mano_corr_.canonical_ind_.size());
    // }
    torch::Tensor corr_pos_src_ =
        torch::from_blob(corr_pos_src[0].data(), {(int)corr_pos_src.size(), 3});

    torch::Tensor corr_pos =
        torch::cat({corr_pos_ref, corr_pos_src_}).contiguous();

    if (!reclib::opengl::Shader::valid(
            "canonicalPointCorrespondences_colorPV")) {
      reclib::opengl::Shader("canonicalPointCorrespondences_colorPV",
                             "MVP_color3.vs", "color3PV.fs");
    }

    if (!reclib::opengl::Drawelement::valid("corr_points_" +
                                            std::to_string(cls_label))) {
      reclib::opengl::Drawelement corr_points =
          reclib::opengl::DrawelementImpl::from_geometry(
              "corr_points_" + std::to_string(cls_label),
              reclib::opengl::Shader::find(
                  "canonicalPointCorrespondences_colorPV"),
              false, corr_pos.data<float>(), corr_pos.sizes()[0],
              (uint32_t*)indices.data(), indices.size());
      corr_points->mesh->primitive_type = GL_POINTS;
      corr_points->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                      false, true);
      corr_points->add_pre_draw_func("pointsize", [&]() {
        glDisable(GL_DEPTH_TEST);
        glPointSize(10.f);
      });
      corr_points->add_post_draw_func("pointsize", [&]() {
        glEnable(GL_DEPTH_TEST);
        glPointSize(1.f);
      });
      corr_points->mesh->geometry->update_meshes();

      reclib::opengl::Drawelement corr_lines =
          reclib::opengl::DrawelementImpl::from_geometry(
              "corr_lines_" + std::to_string(cls_label),
              reclib::opengl::Shader::find(
                  "canonicalPointCorrespondences_colorPV"),
              false, corr_pos.data<float>(), corr_pos.sizes()[0],
              (uint32_t*)indices.data(), indices.size());
      corr_lines->mesh->primitive_type = GL_LINES;
      corr_lines->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                     false, true);
      corr_lines->add_pre_draw_func("linewidth", [&]() {
        // glDisable(GL_DEPTH_TEST);
        glLineWidth(3.f);
      });
      corr_lines->add_post_draw_func("linewidth", [&]() {
        // glEnable(GL_DEPTH_TEST);
        glLineWidth(1.f);
      });
      corr_lines->mesh->geometry->update_meshes();
    } else {
      reclib::opengl::Drawelement corr_points =
          reclib::opengl::Drawelement::find("corr_points_" +
                                            std::to_string(cls_label));
      corr_points->mesh->geometry.cast<reclib::opengl::GeometryImpl>()->set(
          corr_pos.data<float>(), corr_pos.sizes()[0],
          (uint32_t*)indices.data(), indices.size());
      corr_points->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                      false, true);
      corr_points->mesh->geometry->update_meshes();

      reclib::opengl::Drawelement corr_lines =
          reclib::opengl::Drawelement::find("corr_lines_" +
                                            std::to_string(cls_label));
      corr_lines->mesh->geometry.cast<reclib::opengl::GeometryImpl>()->set(
          corr_pos.data<float>(), corr_pos.sizes()[0],
          (uint32_t*)indices.data(), indices.size());
      corr_lines->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                     false, true);
      corr_lines->mesh->geometry->update_meshes();
    }
  }
  return true;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::visualize_uncertainty() {
  if (last_frame_.network_output_.size() == 0) {
    return;
  }

  int num_samples = last_frame_.network_output_[0].sizes()[0];

  for (int i = 0; i < num_samples; i++) {
    if (last_frame_.network_output_[0]
            .index({i})
            .to(torch::kCPU)
            .item()
            .toInt() == 0) {
      if (config_.i("hand_mode") == int(HandMode::LEFT) ||
          config_.i("hand_mode") == int(HandMode::BOTH)) {
        // async mode does not work with openGL yet
        visualize_uncertainty(0, mano_left_, mano_model_left_);
      }
    } else if (last_frame_.network_output_[0]
                   .index({i})
                   .to(torch::kCPU)
                   .item()
                   .toInt() == 1) {
      if (config_.i("hand_mode") == int(HandMode::BOTH)) {
        int index = 1;
        if (num_samples == 1) {
          index = 0;
        } else if (num_samples == 0) {
          continue;
        }

        // async mode does not work with openGL yet

        visualize_uncertainty(index, mano_right_, mano_model_right_);

      } else if (config_.i("hand_mode") == int(HandMode::RIGHT)) {
        // async mode does not work with openGL yet

        visualize_uncertainty(0, mano_right_, mano_model_right_);
      }
    } else {
      throw std::runtime_error("Unknown class label: " +
                               std::to_string(last_frame_.network_output_[0]
                                                  .index({i})
                                                  .to(torch::kCPU)
                                                  .item()
                                                  .toInt()));
    }
  }
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::visualize_uncertainty(
    unsigned int index, reclib::modelstorch::ModelInstance<MANOTYPE>& hand,
    reclib::modelstorch::Model<MANOTYPE>& hand_model) {
  int cls_label = last_frame_.network_output_[0]
                      .index({(int)index})
                      .to(torch::kCPU)
                      .item()
                      .toInt();

  const ManoRGBDCorrespondences& corr_pairs =
      last_frame_.corr_indices_.at(cls_label);
  std::vector<bool> visibility_joints =
      last_frame_.corr_indices_[cls_label].joints_visible_;

  float low_thresh = config_["Visualization"]["error_low_thresh"].as<float>();
  float high_thresh = config_["Visualization"]["error_high_thresh"].as<float>();
  float mid_thresh = (high_thresh - low_thresh) * 0.5 + low_thresh;

  std::vector<vec4> per_vertex_color(hand.verts().sizes()[0]);
  std::vector<float> per_joint_alpha(21);
  std::vector<float> per_joint_alpha_lines(21);

  std::vector<int> reorder_joints = {0,  1,  2,  3,  5,  6, 7, 9,  10, 11, 13,
                                     14, 15, 17, 18, 19, 4, 8, 12, 16, 20};

  {
    torch::NoGradGuard guard;
    int device = reclib::getDevice();
    RasterizeCRStateWrapper context(device);
    int w = intrinsics_.image_width_;
    int h = intrinsics_.image_height_;
    std::tuple<int, int> res = std::make_tuple(h, w);
    mat4 proj = reclib::vision2graphics(intrinsics_.Matrix(), 0.01f, 1000.f,
                                        intrinsics_.image_width_,
                                        intrinsics_.image_height_);
    mat4 view = reclib::lookAt(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, -1, 0));
    mat4 vp = proj * view;

    torch::Tensor VP = reclib::dnn::eigen2torch<float, 4, 4>(vp, true).to(
        hand.trans().device());
    // torch::Tensor colors =
    //     torch::ones(hand.verts().sizes()).to(hand.verts().device());
    torch::Tensor colors =
        torch::from_blob(mano_corr_space_[0].data(), hand.verts().sizes())
            .clone()
            .to(hand.verts().device());
    colors = colors.index({torch::None, "..."});

    torch::Tensor box =
        last_frame_.network_output_[2].index({(int)index}).contiguous();
    torch::Tensor seg =
        torch::zeros({intrinsics_.image_height_, intrinsics_.image_width_},
                     torch::TensorOptions().device(hand.params.device()));

    torch::Tensor pc_indices =
        torch::from_blob((void*)corr_pairs.pc_corr_.canonical_ind_.data(),
                         {(int)corr_pairs.pc_corr_.canonical_ind_.size()},
                         torch::TensorOptions().dtype(torch::kInt))
            .toType(torch::kLong)
            .to(hand.trans().device());
    torch::Tensor mano_indices =
        torch::from_blob((void*)corr_pairs.mano_corr_.canonical_ind_.data(),
                         {(int)corr_pairs.mano_corr_.canonical_ind_.size()},
                         torch::TensorOptions().dtype(torch::kInt))
            .toType(torch::kLong)
            .to(hand.trans().device());
    torch::Tensor vertex_map =
        reclib::dnn::cv2torch(last_frame_.vertex_maps_[cls_label], true)
            .reshape({-1, 3})
            .to(hand.trans().device());

    torch::Tensor normal_map =
        reclib::dnn::cv2torch(last_frame_.normal_maps_[cls_label], true)
            .reshape({-1, 3})
            .to(hand.trans().device());

    torch::Tensor seg_ = last_frame_.network_output_[3].index(
        {(int)index,
         torch::indexing::Slice(box.index({1}).item<int>(),
                                box.index({3}).item<int>()),
         torch::indexing::Slice(box.index({0}).item<int>(),
                                box.index({2}).item<int>())});
    seg.index_put_({torch::indexing::Slice(box.index({1}).item<int>(),
                                           box.index({3}).item<int>()),
                    torch::indexing::Slice(box.index({0}).item<int>(),
                                           box.index({2}).item<int>())},
                   seg_);

    seg = torch::flip(seg, 0).contiguous();

    torch::Tensor corr = last_frame_.network_output_[4]
                             .index({(int)index})
                             .contiguous()
                             .to(seg.device());
    // corr[0] /= 360.f;
    // corr = torch::flip(corr, 1).permute({1, 2, 0}).contiguous() *
    //        seg.unsqueeze(-1);
    corr = torch::flip(corr, 1).permute({1, 2, 0}).contiguous() *
           seg.unsqueeze(-1);

    torch::Tensor verts = hand.verts();

    torch::Tensor ones =
        torch::ones({verts.sizes()[0], 1},
                    torch::TensorOptions().device(verts.device()))
            .contiguous();
    torch::Tensor verts_hom = torch::cat({verts, ones}, 1).contiguous();
    torch::Tensor verts_clip =
        torch::matmul(verts_hom, VP.transpose(1, 0)).contiguous();
    torch::Tensor pos_idx = hand.model.faces.clone().to(torch::kCUDA);
    verts_clip = verts_clip.index({torch::None, "..."}).to(torch::kCUDA);
    colors = colors.to(torch::kCUDA);

    std::vector<torch::Tensor> rast_out =
        reclib::dnn::rasterize(context, verts_clip, pos_idx, res);
    std::vector<torch::Tensor> interp_out =
        reclib::dnn::interpolate(colors, rast_out[0], pos_idx);
    std::vector<torch::Tensor> antialias_out =
        reclib::dnn::antialias(interp_out[0], rast_out[0], verts_clip, pos_idx);
    torch::Tensor color_pred = antialias_out[0].squeeze(0).to(torch::kCPU);
    CpuMat color_tmp = reclib::dnn::torch2cv(color_pred);
    CpuMat color_gt = reclib::dnn::torch2cv(corr);
    // cv::imshow("tmp", color_tmp);
    // cv::imshow("gt: ", color_gt);
    // cv::waitKey(0);

    torch::Tensor point2point =
        (vertex_map.index({pc_indices}) - verts.index({mano_indices}));

    torch::Tensor point2plane = torch::bmm(
        point2point.view({vertex_map.index({pc_indices}).sizes()[0], 1, 3}),
        normal_map.index({pc_indices})
            .view({vertex_map.index({pc_indices}).sizes()[0], 3, 1}));

    // torch::Tensor per_point_error = point2point.abs().sum(1);
    torch::Tensor per_point_error = point2point.abs().index({torch::All, 2});

    torch::Tensor per_pixel_error =
        (torch::l1_loss(corr, color_pred, torch::Reduction::None)).abs().sum(2);

    CpuMat error_mat = reclib::dnn::torch2cv(per_pixel_error);

    CpuMat corr_seg(last_frame_.corrs_segmented_[cls_label].size(), CV_32FC1);
    cv::extractChannel(last_frame_.corrs_segmented_[cls_label], corr_seg, 0);
    // cv::imshow("error_mat", error_mat);
    // cv::imshow("corr_seg", corr_seg / 30);
    // cv::waitKey(0);

    torch::Tensor px_segmentation_ = reclib::dnn::cv2torch(corr_seg);
    torch::Tensor px_segmentation = torch::ones({h, w, 1}) * 30;
    px_segmentation.index_put_(
        {
            torch::indexing::Slice(
                {box.index({1}).item<int>(), box.index({3}).item<int>()}),
            torch::indexing::Slice(
                {box.index({0}).item<int>(), box.index({2}).item<int>()}),
        },
        px_segmentation_);

    px_segmentation = px_segmentation.index({torch::All, torch::All, 0});
    px_segmentation = torch::flip(px_segmentation, 0).contiguous();

    std::vector<bool> joint_errorenous;
    std::vector<bool> joint_visible;
    for (int i = 0; i < 20; i++) {
      torch::Tensor candidates = px_segmentation == i;

      torch::Tensor err = per_pixel_error.index({candidates});
      // std::cout << "err: " << err.sizes() << std::endl;
      // std::cout << "err: " << err.index({torch::indexing::Slice(0, 5)})
      //           << std::endl;
      torch::Tensor median_err = torch::median(err);
      // torch::Tensor outlier = err > median_err;
      // torch::Tensor deviation = err - median_err;

      if (err.sizes()[0] == 0) {
        joint_visible.push_back(false);
        joint_errorenous.push_back(false);
        continue;
      }

      torch::Tensor deviation = err;
      torch::Tensor outlier =
          deviation > config_["Visualization"]["pixel_thresh"].as<float>();
      float outlier_fraction = outlier.sum().item<float>() / outlier.sizes()[0];
      if (outlier_fraction >=
          config_["Visualization"]["pixel_fraction"].as<float>()) {
        joint_errorenous.push_back(true);
      } else {
        joint_errorenous.push_back(false);
      }
      joint_visible.push_back(true);
    }

    std::map<int, std::vector<float>> per_vertexpoint_error;
    for (int i = 0; i < per_point_error.sizes()[0]; i++) {
      int vertex_index = corr_pairs.mano_corr_.canonical_ind_[i];
      int joint_index = mano_verts2joints_[vertex_index];
      float error = per_point_error.index({i}).item<float>();

      if (per_vertexpoint_error.find(vertex_index) ==
          per_vertexpoint_error.end()) {
        per_vertexpoint_error[vertex_index] = std::vector<float>();
        per_vertexpoint_error[vertex_index].push_back(error);
      } else {
        per_vertexpoint_error[vertex_index].push_back(error);
      }
    }

    std::map<int, std::vector<float>> per_joint_error;
    // std::vector<int> vertices_per_joint(20);
    // std::fill(vertices_per_joint.begin(), vertices_per_joint.end(), 0);

    for (auto it : per_vertexpoint_error) {
      int i = it.first;
      int joint_index = mano_verts2joints_[i];
      vec3 hsv = mano_corr_space_[i];
      int seg_index = hsv2cls(hsv);
      joint_index = seg_index;

      if (per_vertexpoint_error.find(i) != per_vertexpoint_error.end()) {
        float sum = 0;
        for (unsigned int j = 0; j < per_vertexpoint_error[i].size(); j++) {
          sum += per_vertexpoint_error[i][j];
        }

        if (per_joint_error.find(joint_index) == per_joint_error.end()) {
          per_joint_error[joint_index] = std::vector<float>();
          per_joint_error[joint_index].push_back(
              sum / per_vertexpoint_error[i].size());
        } else {
          per_joint_error[joint_index].push_back(
              sum / per_vertexpoint_error[i].size());
        }
      }
    }

    std::vector<bool> joint_errorenous_depth(vertices_per_joint.size());
    for (int i = 0; i < vertices_per_joint.size(); i++) {
      int joint_index = i;

      if (per_joint_error.find(joint_index) == per_joint_error.end()) {
        joint_visible[joint_index] = false;
        joint_errorenous_depth[joint_index] = false;
        continue;
      } else {
        std::vector<float> pj = per_joint_error[joint_index];
        float visible_frac = pj.size() / (float)vertices_per_joint[joint_index];

        int mano_index = vis2pose_indices[i];

        // std::cout << "joint: " << joint_index << "," << mano_index << ", "
        //           << reclib::models::MANOConfig::joint_name[mano_index]
        //           << " visible: " << visible_frac
        //           << " vertices total: " << vertices_per_joint[joint_index]
        //           << std::endl;

        if (visible_frac <
            config_["Visualization"]["per_joint_visible_fraction"]
                .as<float>()) {
          joint_visible[joint_index] = false;
          joint_errorenous_depth[joint_index] = false;

          continue;
        }

        int num_errorenous = 0;
        for (unsigned int j = 0; j < pj.size(); j++) {
          if (pj[j] > config_["Visualization"]["depth_thresh"].as<float>()) {
            num_errorenous++;
          }
        }

        float errorenous_fraction = num_errorenous / (float)pj.size();
        if (errorenous_fraction >
            config_["Visualization"]["depth_fraction"].as<float>()) {
          joint_errorenous_depth[joint_index] = true;
          joint_visible[joint_index] = true;
        }
      }
    }

    bool separate =
        config_["Visualization"]["separate_px_and_depth"].as<bool>();

    for (int i = 0; i < hand.verts().sizes()[0]; i++) {
      int joint_index = mano_verts2joints_[i];
      float weight =
          hand.model.weights.index({i, joint_index}).template item<float>();

      vec3 hsv_mano = mano_corr_space_[i];
      int joint_index_seg = hsv2cls(hsv_mano);
      vec3 test(0, 0, 0);
      test.x() = joint_index_seg;
      vec3 hsv = hsv2classification(test);

      vec4 c(SKIN_COLOR.x(), SKIN_COLOR.y(), SKIN_COLOR.z(), 1);
      float alpha = 1;

      if (joint_index_seg % 4 == 0) {
        // if (joint_index_seg == 0) {
        //   c.x() = 1;
        //   c.y() = 0;
        //   c.z() = 0;
        // }
        // if (joint_index_seg == 4) {
        //   c.x() = 0;
        //   c.y() = 0;
        //   c.z() = 1;
        // }
        // if (joint_index_seg == 8) {
        //   c.x() = 1;
        //   c.y() = 0.5;
        //   c.z() = 0;
        // }
        // if (joint_index_seg == 12) {
        //   c.x() = 0;
        //   c.y() = 1;
        //   c.z() = 0;
        // }
        // if (joint_index_seg == 16) {
        //   c.x() = 0.5;
        //   c.y() = 0;
        //   c.z() = 1;
        // }

        // if (weight < 0.85) {
        // per_vertex_color[i] = c;
        // continue;
        //}

      } else if (joint_index == 0) {
        // per_vertex_color[i] = c;
        // continue;

      } else {
        // per_vertex_color[i] = c;
        // continue;

        // joint_index = vis2pose_indices[joint_index_seg];
        joint_index_seg = joint2vis_indices[joint_index - 1];
      }
      if (!joint_visible[joint_index_seg]) {
        alpha = 0.75;
        c.w() = alpha;
        c.x() = 0.3;
        c.y() = 0.3;
        c.z() = 0.3;
        // c.x() = fmax(0.55 + weight * 0.5, 0.55);
        // c.y() -= 0.1;
        // c.z() -= 0.25;

        per_vertex_color[i] = c;
        continue;
      }
      if (!separate && (joint_errorenous[joint_index_seg] ||
                        joint_errorenous_depth[joint_index_seg])) {
        // c.x() = 0.7 * weight;
        // c.y() = 0.7 * weight;
        c.x() = fmax(0.55 + weight * 0.5, 0.55);
        // c.y() -= 0.25 * weight;
        // c.z() -= 0.25 * weight;
        c.y() -= 0.1;
        c.z() -= 0.25;
      }

      if (separate && joint_errorenous[joint_index_seg]) {
        c.x() = fmax(0.5 + weight * 0.5, 0.5);
        c.y() = 0;
        c.z() = 0;
      }
      if (separate && joint_errorenous_depth[joint_index_seg]) {
        c.z() = fmax(0.5 + weight * 0.5, 0.5);
        c.x() = 0;
        c.y() = 0;
      }
      if (separate && (joint_errorenous[joint_index_seg] &&
                       joint_errorenous_depth[joint_index_seg])) {
        c.y() = fmax(0.5 + weight * 0.5, 0.5);
        c.x() = 0;
        c.z() = 0;
      }

      // if (joint_errorenous_depth[joint_index_seg]) {
      //   c.x() += 0.2 * weight;
      //   c.y() -= 0.7 * weight;
      // }
      c.x() = fmax(fmin(c.x(), 1), 0);
      c.y() = fmax(fmin(c.y(), 1), 0);
      c.z() = fmax(fmin(c.z(), 1), 0);
      c.w() = alpha;

      // if (c.y() > c.x() || c.z() > c.x()) {
      //   std::cout << "ERROR: c: " << c << " weight: " << weight
      //             << " SKIN: " << SKIN_COLOR << std::endl;
      // }

      per_vertex_color[i] = c;
    }

    // per_joint_alpha[0] = 1;
    std::fill(per_joint_alpha.begin(), per_joint_alpha.end(), 1);
    std::fill(per_joint_alpha_lines.begin(), per_joint_alpha_lines.end(), 1);
    per_joint_alpha_lines[0] = 0.8;
    std::vector<float> tip_indices = {16, 17, 18, 19, 20};

    // std::cout << "visible: " << joint_visible << std::endl;
    // std::cout << "------------------------------------------------"
    //           << std::endl;
    // std::cout << "errorenous: " << joint_errorenous << std::endl;
    // std::cout << "------------------------------------------------"
    //           << std::endl;
    // std::cout << "errorenous_depth: " << joint_errorenous_depth << std::endl;
    // std::cout << "------------------------------------------------"
    //           << std::endl;

    // std::cout << "joint visible: " << joint_visible << std::endl;
    for (int i = 0; i < 16; i++) {
      int joint_index = i;
      float alpha = 1;

      int target_index = 0;
      int visibility_joint = 0;
      if (i == 0) {
        // if (!joint_visible[0]) {
        //   // std::cout << "joint " << i << " mapped: " << 0 << " target: " <<
        //   13
        //   //           << std::endl;

        //   per_joint_alpha[13] = 0.2;
        // }
        // if (!joint_visible[4]) {
        //   // std::cout << "joint " << i << " mapped: " << 4 << " target: " <<
        //   1
        //   //           << std::endl;
        //   per_joint_alpha[1] = 0.2;
        // }
        // if (!joint_visible[8]) {
        //   // std::cout << "joint " << i << " mapped: " << 8 << " target: " <<
        //   4
        //   //           << std::endl;
        //   per_joint_alpha[4] = 0.2;
        // }
        // if (!joint_visible[12]) {
        //   // std::cout << "joint " << i << " mapped: " << 12 << " target: "
        //   <<
        //   // 10
        //   //           << std::endl;
        //   per_joint_alpha[10] = 0.2;
        // }
        // if (!joint_visible[16]) {
        //   // std::cout << "joint " << i << " mapped: " << 16 << " target: "
        //   << 7
        //   //           << std::endl;
        //   per_joint_alpha[7] = 0.2;
        // }

        if (!joint_visible[0] && !joint_visible[4] && !joint_visible[8] &&
            !joint_visible[12] && !joint_visible[16]) {
          per_joint_alpha[1] = 0.01;
          per_joint_alpha[13] = 0.01;
          per_joint_alpha[4] = 0.01;
          per_joint_alpha[10] = 0.01;
          per_joint_alpha[7] = 0.01;
        }
        if (joint_errorenous[0] && joint_errorenous[4] && joint_errorenous[8] &&
            joint_errorenous[12] && joint_errorenous[16]) {
          per_joint_alpha[1] = 0.01;
          per_joint_alpha[13] = 0.01;
          per_joint_alpha[4] = 0.01;
          per_joint_alpha[10] = 0.01;
          per_joint_alpha[7] = 0.01;
        }
        if (joint_errorenous_depth[0] && joint_errorenous_depth[4] &&
            joint_errorenous_depth[8] && joint_errorenous_depth[12] &&
            joint_errorenous_depth[16]) {
          per_joint_alpha[1] = 0.01;
          per_joint_alpha[13] = 0.01;
          per_joint_alpha[4] = 0.01;
          per_joint_alpha[10] = 0.01;
          per_joint_alpha[7] = 0.01;
        }
        continue;

      } else {
        visibility_joint = joint2vis_indices[i - 1];
        if (!joint_visible[visibility_joint] ||
            joint_errorenous[visibility_joint] ||
            joint_errorenous_depth[visibility_joint]) {
          alpha = 0.01;
        }
        target_index = i + 1;
        if ((i - 1) % 3 == 2) {
          target_index = tip_indices[(i - 1) / 3];  // ((i - 1) / 3) + 16;
        }
        // else if (i > 3) {
        //   target_index = (i + 1) - ((i - 1) / 3);
        // }
      }
      // alpha = 1;
      // std::cout << "joint " << i << " mapped: " << visibility_joint
      //           << " target: " << target_index << " alpha: " << alpha
      //           << std::endl;

      // std::cout << "joint: " << target_index << " alpha: " << alpha
      //           << std::endl;

      per_joint_alpha[target_index] = alpha;
      per_joint_alpha_lines[target_index] = alpha;
      if (i > 0) {
        per_joint_alpha_lines[i] = fmin(per_joint_alpha_lines[i], alpha);
      }

      if ((i == 1 || i == 4 || i == 7 || i == 10 || i == 13) && alpha < 1) {
        // per_joint_alpha_lines[0] = fmin(per_joint_alpha_lines[0], alpha);
        per_joint_alpha[i] = fmin(per_joint_alpha[i], alpha);
      }
    }

    last_frame_.joints_visible_[cls_label] = joint_visible;
    last_frame_.joints_errorenous_[cls_label].clear();
    for (unsigned int i = 0; i < 20; i++) {
      last_frame_.joints_errorenous_[cls_label].push_back(
          joint_errorenous[i] || joint_errorenous_depth[i]);
    }
  }

  hand.gl_instance->mesh->geometry->vec4_map.erase("color");
  hand.gl_instance->mesh->geometry->add_attribute_vec4(
      "color", per_vertex_color.data(), per_vertex_color.size(), false);
  hand.gl_instance->mesh->geometry->update_meshes();

  hand.gl_joint_lines->mesh->geometry->float_map.erase("alpha");
  hand.gl_joint_lines->mesh->geometry->add_attribute_float(
      "alpha", per_joint_alpha_lines, 1);
  hand.gl_joint_lines->mesh->geometry->update_meshes();

  hand.gl_joints->mesh->geometry->float_map.erase("alpha");
  hand.gl_joints->mesh->geometry->add_attribute_float("alpha", per_joint_alpha,
                                                      1);
  hand.gl_joints->mesh->geometry->update_meshes();
}

template <typename MANOTYPE>
std::pair<bool, bool> reclib::tracking::ManoRGBD<MANOTYPE>::optimize_hand(
    int stage) {
  if (last_frame_.network_output_.size() == 0) {
    std::cout << "output is empty. quit." << std::endl;
    return std::make_pair(false, false);
  }

  std::vector<std::thread> threads;

  bool result_left = true;
  bool result_right = true;
  int num_samples = last_frame_.network_output_[0].sizes()[0];

  std::vector<std::future<bool>> futures;
  std::vector<int> queried_indexes;

  for (int i = 0; i < num_samples; i++) {
    if (last_frame_.network_output_[0]
            .index({i})
            .to(torch::kCPU)
            .item()
            .toInt() == 0) {
      if (config_.i("hand_mode") == int(HandMode::LEFT) ||
          config_.i("hand_mode") == int(HandMode::BOTH)) {
        // if (mode == reclib::tracking::OptimizationMode::PCA) {
        //   result = result && optimize_hand_pca(0, mano_left_);
        // } else {
        //   result = result && optimize_hand_angles(0, mano_left_);
        // }
        // optimize_hand(0, mano_left_, stage);

        // threads.push_back(std::thread(
        //     &reclib::tracking::ManoRGBDFunctions::optimize_hand<MANOTYPE>,
        //     std::ref(config_), std::ref(last_frame_), frame_counter_,
        //     std::ref(intrinsics_), debug_, 0, std::ref(mano_corr_space_),
        //     std::ref(mano_left_.model), std::ref(mano_left_), stage));

        // threads.push_back(std::thread([this, stage] {
        //   optimize_hand(0, mano_left_, mano_model_left_, stage);
        // }));

        // auto future_left = std::async([this, stage] {
        //   return optimize_hand(0, mano_left_, mano_model_left_, stage);
        // });

        if (true || config_["Debug"]["show_hands"].as<bool>()) {
          // async mode does not work with openGL yet
          result_left = optimize_hand(0, mano_left_, mano_model_left_, stage);
        } else {
          queried_indexes.push_back(0);
          futures.push_back(std::async(std::launch::async, [this, stage] {
            return optimize_hand(0, mano_left_, mano_model_left_, stage);
          }));
          // auto _ = std::async(std::launch::async, [this, stage] {
          //   return optimize_hand(0, mano_left_, mano_model_left_, stage);
          // });
        }
      }
    } else if (last_frame_.network_output_[0]
                   .index({i})
                   .to(torch::kCPU)
                   .item()
                   .toInt() == 1) {
      if (config_.i("hand_mode") == int(HandMode::BOTH)) {
        int index = 1;
        if (num_samples == 1) {
          index = 0;
        } else if (num_samples == 0) {
          continue;
        }

        if (true || config_["Debug"]["show_hands"].as<bool>()) {
          // async mode does not work with openGL yet
          result_right =
              optimize_hand(index, mano_right_, mano_model_right_, stage);

        } else {
          queried_indexes.push_back(1);
          futures.push_back(std::async(std::launch::async, [this, stage,
                                                            index] {
            return optimize_hand(index, mano_right_, mano_model_right_, stage);
          }));
        }
      } else if (config_.i("hand_mode") == int(HandMode::RIGHT)) {
        if (true || config_["Debug"]["show_hands"].as<bool>()) {
          // async mode does not work with openGL yet
          result_right =
              optimize_hand(0, mano_right_, mano_model_right_, stage);
        } else {
          queried_indexes.push_back(1);
          futures.push_back(std::async(std::launch::async, [this, stage] {
            return optimize_hand(0, mano_right_, mano_model_right_, stage);
          }));
        }
      }
    } else {
      throw std::runtime_error("Unknown class label: " +
                               std::to_string(last_frame_.network_output_[0]
                                                  .index({i})
                                                  .to(torch::kCPU)
                                                  .item()
                                                  .toInt()));
    }
  }

  for (unsigned int i = 0; i < futures.size(); i++) {
    bool result = futures[i].get();
    int index = queried_indexes[i];
    if (index == 0) {
      result_left = result;
    } else {
      result_right = result;
    }
  }

  optimization_counter_++;
  return std::make_pair(result_left, result_right);
}

template <typename MANOTYPE>
bool reclib::tracking::ManoRGBD<MANOTYPE>::optimize_hand(
    unsigned int index, reclib::modelstorch::ModelInstance<MANOTYPE>& hand,
    reclib::modelstorch::Model<MANOTYPE>& hand_model, int stage) {
  if (!config_.b("optimize_hand")) return true;

  reclib::opengl::Timer timer;
  timer.begin();

  reclib::Configuration corr_config =
      config_.subconfig({"Correspondences_Optimization"});

  reclib::Configuration opt_config = config_.subconfig({"Optimization"});

  int cls_label = last_frame_.network_output_[0]
                      .index({(int)index})
                      .to(torch::kCPU)
                      .item()
                      .toInt();

  const ManoRGBDCorrespondences& corr_pairs =
      last_frame_.corr_indices_.at(cls_label);

  std::vector<bool> visibility_joints =
      last_frame_.corr_indices_[cls_label].joints_visible_;

  // std::vector<uint8_t>& visibility = last_frame_.visibility_map_.at(index);

  int num_visible = 0;

  if (debug_) {
    std::cout << "visibility: ";
    for (unsigned int i = 0; i < visibility_joints.size(); i++) {
      if (i > 0 && i - 1 % 3 == 0) {
        std::cout << std::endl;
        std::cout << "-----------" << std::endl;
      }
      if (visibility_joints[i] > 0) {
        num_visible++;
      }
      std::cout << (uint32_t)visibility_joints[i] << " ";
    }
    std::cout << std::endl;
  }

  for (unsigned int i = 0; i < (visibility_joints.size() - 1) / 3; i++) {
    for (unsigned int j = 2; j > 0; j--) {
      if (visibility_joints[i * 3 + j + 1]) {
        for (int k = j - 1; k >= 0; k--) {
          visibility_joints[i * 3 + k + 1] = true;
        }
        break;
      }
    }
  }

  float visible_percentage = num_visible / 16.f;
  if (debug_)
    std::cout << "Visible percentage: " << visible_percentage << std::endl;

  if (corr_pairs.mano_corr_.canonical_ind_.size() == 0) {
    std::cout << "No correspondences. Abort." << std::endl;
    return false;
  }

  int n_pca_comps = opt_config["pca_components"].as<int>();

  hand_model.gpu();
  hand.gpu();

  hand.requires_grad(false);
  torch::Tensor prev_trans;
  torch::Tensor prev_shape;
  torch::Tensor prev_rot;
  torch::Tensor prev_apose;
  torch::Tensor prev_pose;
  torch::Tensor prev_visibility;

  torch::Tensor trans;
  torch::Tensor shape;
  torch::Tensor rot;
  torch::Tensor pose;
  torch::Tensor apose;
  torch::Tensor pca;

  torch::Tensor keypoints;
  torch::Tensor keypoints_mask;

  {
    torch::NoGradGuard no_grad;
    trans = hand.trans().clone().detach();
    shape = hand.shape().clone().detach();
    rot = hand.apose().index({torch::indexing::Slice({0, 3})}).clone().detach();
    // rot = torch::zeros(3);
    // apose =
    //     hand.apose().index({torch::indexing::Slice({3,
    //     26})}).clone().detach();

    pca = hand.hand_pca().clone().detach();
    // keypoints =
    //     last_frame_.keypoints_[cls_label].clone().to(hand.trans().device());
    // keypoints_mask = (keypoints.sum(1) >= 0).to(hand.trans().device());

    apose = torch::matmul(hand.model.hand_comps, pca.clone()).reshape({-1});

    if (frame_counter_ > 1) {
      prev_trans = last_frame_.trans_[cls_label].clone().detach().to(
          hand.trans().device());
    } else {
      prev_trans = hand.trans().clone().detach();
    }

    prev_shape = last_frame_.shape_[cls_label].clone().detach().to(
        hand.trans().device());
    prev_rot =
        last_frame_.rot_[cls_label].clone().detach().to(hand.trans().device());
    prev_apose =
        last_frame_.pose_[cls_label].clone().detach().to(hand.trans().device());
  }

  if (0) {
    std::vector<int> visibility_apose;
    for (int i = 0; i < apose.sizes()[0]; i++) {
      int joint = apose2joint[i];
      visibility_apose.push_back(visibility_joints[joint]);
    }

    torch::Tensor visibility_tensor =
        torch::from_blob(visibility_apose.data(),
                         {(int)visibility_apose.size()},
                         torch::TensorOptions().dtype(torch::kInt32))
            .clone()
            .toType(torch::kBool);

    torch::Tensor Q_given =
        hand.model.hand_comps.index({visibility_tensor, torch::All});
    torch::Tensor mu_given = hand.model.hand_mean.index({visibility_tensor});
    torch::Tensor s_given = apose.index({visibility_tensor}).unsqueeze(1);

    float sigma = 0.01 * 0.01;
    torch::Tensor M =
        torch::matmul(Q_given.transpose(1, 0), Q_given) +
        sigma * torch::eye(23, torch::TensorOptions().device(Q_given.device()));
    torch::Tensor centered = s_given - mu_given;
    torch::Tensor tmp = torch::matmul(hand.model.hand_comps, M.inverse());
    torch::Tensor mu_partial =
        hand.model.hand_mean +
        torch::matmul(torch::matmul(tmp, Q_given.transpose(1, 0)), centered);
    mu_partial = mu_partial.squeeze(1);

    torch::Tensor cov_partial =
        sigma * torch::matmul(tmp, hand.model.hand_comps.transpose(1, 0));

    torch::Tensor det_partial = torch::linalg::det(cov_partial);
    if (det_partial.item<float>() < 1e-10) {
      cov_partial =
          cov_partial +
          torch::eye(23, torch::TensorOptions().device(cov_partial.device())) *
              0.01;
    }

    det_partial = torch::linalg::det(cov_partial);
    torch::Tensor inv_cov_partial = torch::linalg::pinv(cov_partial);
  }

  torch::Tensor pc_indices =
      torch::from_blob((void*)corr_pairs.pc_corr_.canonical_ind_.data(),
                       {(int)corr_pairs.pc_corr_.canonical_ind_.size()},
                       torch::TensorOptions().dtype(torch::kInt))
          .toType(torch::kLong)
          .to(trans.device());
  torch::Tensor mano_indices =
      torch::from_blob((void*)corr_pairs.mano_corr_.canonical_ind_.data(),
                       {(int)corr_pairs.mano_corr_.canonical_ind_.size()},
                       torch::TensorOptions().dtype(torch::kInt))
          .toType(torch::kLong)
          .to(trans.device());
  torch::Tensor vertex_map =
      reclib::dnn::cv2torch(last_frame_.vertex_maps_[cls_label], true)
          .reshape({-1, 3})
          .to(trans.device());

  torch::Tensor normal_map =
      reclib::dnn::cv2torch(last_frame_.normal_maps_[cls_label], true)
          .reshape({-1, 3})
          .to(trans.device());

  // torch::Tensor segmentation =
  //     reclib::dnn::cv2torch(last_frame_.corrs_segmented_[index], true)
  //         .reshape({-1, 3})
  //         .to(trans.device());

  torch::Tensor box =
      last_frame_.network_output_[2].index({(int)index}).contiguous();

  torch::Tensor seg =
      torch::zeros({intrinsics_.image_height_, intrinsics_.image_width_},
                   torch::TensorOptions().device(hand.params.device()));

  torch::Tensor seg_ = last_frame_.network_output_[3].index(
      {(int)index,
       torch::indexing::Slice(box.index({1}).item<int>(),
                              box.index({3}).item<int>()),
       torch::indexing::Slice(box.index({0}).item<int>(),
                              box.index({2}).item<int>())});
  seg.index_put_({torch::indexing::Slice(box.index({1}).item<int>(),
                                         box.index({3}).item<int>()),
                  torch::indexing::Slice(box.index({0}).item<int>(),
                                         box.index({2}).item<int>())},
                 seg_);

  seg = torch::flip(seg, 0).contiguous();

  torch::Tensor corr =
      last_frame_.network_output_[4].index({(int)index}).contiguous();
  // corr[0] /= 360.f;
  corr =
      torch::flip(corr, 1).permute({1, 2, 0}).contiguous() * seg.unsqueeze(-1);

  mat4 proj = reclib::vision2graphics(intrinsics_.Matrix(), 0.01f, 1000.f,
                                      intrinsics_.image_width_,
                                      intrinsics_.image_height_);
  mat4 view = reclib::lookAt(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, -1, 0));
  mat4 vp = proj * view;
  torch::Tensor VP =
      reclib::dnn::eigen2torch<float, 4, 4>(vp, true).to(trans.device());
  torch::Tensor colors =
      torch::from_blob(mano_corr_space_[0].data(), hand.verts().sizes())
          .clone()
          .to(hand.verts().device());
  colors = colors.index({torch::None, "..."});

  int device = reclib::getDevice();
  RasterizeCRStateWrapper context(device);
  int w = intrinsics_.image_width_;
  int h = intrinsics_.image_height_;
  std::tuple<int, int> res = std::make_tuple(h, w);

  torch::Tensor cross_matrix =
      reclib::tracking::compute_apose_matrix(hand.model, shape);

  torch::Tensor apose_indices =
      torch::from_blob(reclib::tracking::apose_indices_.data(),
                       {(int)reclib::tracking::apose_indices_.size()},
                       torch::TensorOptions().dtype(torch::kLong))
          .clone()
          .to(trans.device()) -
      3;

  std::vector<torch::Tensor> params;

  // params.push_back(trans);
  params.push_back(rot);
  // params.push_back(apose);
  params.push_back(pca);
  if (visible_percentage > 0.5) {
    params.push_back(shape);
  }

  torch::optim::AdamOptions adam_opt_visible;
  adam_opt_visible.amsgrad() = true;
  if (frame_counter_ <= 1) {
    adam_opt_visible.set_lr(
        opt_config["lr_visible_initialization"].as<float>());

  } else {
    adam_opt_visible.set_lr(opt_config["lr_visible"].as<float>());
  }
  adam_opt_visible.weight_decay() = opt_config["weight_decay"].as<float>();

  torch::optim::LBFGSOptions opt_visible;
  if (frame_counter_ <= 1) {
    opt_visible.max_iter(
        opt_config["inner_iterations_visible_initialization"].as<int>());
  } else {
    opt_visible.max_iter(opt_config["inner_iterations_visible"].as<int>());
  }

  opt_visible.line_search_fn() = "strong_wolfe";

  if (debug_)
    std::cout << "preparation: " << timer.look_and_reset() << " ms."
              << std::endl;

  for (unsigned int i = 0;
       i < opt_config["outer_iterations"].as<unsigned int>(); i++) {
    // torch::optim::LBFGS optim_rot_pose = torch::optim::LBFGS({apose}, opt);

    torch::optim::LBFGS optim_rot_pose_trans =
        torch::optim::LBFGS(params, opt_visible);

    bool visible = true;

    auto loss_func_rot_apose = [&](bool visible) {
      // optim_rot_pose.zero_grad();
      // optim_rot_pose_invisible.zero_grad();
      // optim_trans.zero_grad();
      visible = true;

      torch::Tensor data_term =
          torch::zeros(1, torch::TensorOptions().device(trans.device()));
      torch::Tensor temp_reg_term_pose =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      torch::Tensor temp_reg_term_rot =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      torch::Tensor temp_reg_term_trans =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      torch::Tensor temp_reg_term_shape =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      torch::Tensor reg_term_pose =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      torch::Tensor reg_term_shape =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      torch::Tensor silhouette_term =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      torch::Tensor pose_prior =
          torch::zeros(1, torch::TensorOptions().device(data_term.device()));
      // torch::Tensor keypoint_term =
      //     torch::zeros(1, torch::TensorOptions().device(trans.device()));

      std::pair<torch::Tensor, torch::Tensor> t;

      if (pca.requires_grad()) {
        t = reclib::tracking::torch_lbs_pca_anatomic(
            hand.model, trans, rot, shape, pca, cross_matrix, false);
      } else {
        t = reclib::tracking::torch_lbs_anatomic(hand.model, trans, rot, shape,
                                                 apose, cross_matrix, false);
      }

      torch::Tensor verts = t.first.contiguous();

      torch::Tensor ones =
          torch::ones({verts.sizes()[0], 1},
                      torch::TensorOptions().device(verts.device()))
              .contiguous();
      torch::Tensor verts_hom = torch::cat({verts, ones}, 1).contiguous();
      torch::Tensor verts_clip =
          torch::matmul(verts_hom, VP.transpose(1, 0)).contiguous();
      torch::Tensor pos_idx = hand.model.faces;

      verts_clip = verts_clip.index({torch::None, "..."});

      std::vector<torch::Tensor> rast_out =
          reclib::dnn::rasterize(context, verts_clip, pos_idx, res);
      std::vector<torch::Tensor> interp_out =
          reclib::dnn::interpolate(colors, rast_out[0], pos_idx);
      std::vector<torch::Tensor> antialias_out = reclib::dnn::antialias(
          interp_out[0], rast_out[0], verts_clip, pos_idx);
      torch::Tensor color_pred = antialias_out[0].squeeze(0);

      torch::Tensor positive_samples_pred = (color_pred.sum(2) > 0).sum();

      torch::Tensor iou_intersection =
          torch::logical_and(color_pred.sum(2) > 0, seg);
      torch::Tensor iou_union = torch::logical_or(color_pred.sum(2) > 0, seg);
      torch::Tensor iou_gt = seg.sum() / positive_samples_pred;
      torch::Tensor iou_pred = iou_intersection.sum() / iou_union.sum();

      float weight = std::exp(-iou_pred.item<float>() *
                                  opt_config["data_weight_scale"].as<float>() +
                              opt_config["data_weight_offset"].as<float>()) *
                     opt_config["data_weight"].as<float>();

      if (pca.requires_grad() || apose.requires_grad() ||
          trans.requires_grad() || rot.requires_grad() || true) {
        data_term =
            weight *
            torch::bmm(
                torch::nn::functional::l1_loss(
                    vertex_map.index({pc_indices}),
                    t.first.index({mano_indices}),
                    torch::nn::functional::L1LossFuncOptions().reduction(
                        torch::kNone))
                    .view({vertex_map.index({pc_indices}).sizes()[0], 1, 3}),
                normal_map.index({pc_indices})
                    .view({vertex_map.index({pc_indices}).sizes()[0], 3, 1}));

        torch::Tensor point2point = weight * (vertex_map.index({pc_indices}) -
                                              t.first.index({mano_indices}));

        torch::Tensor point2plane =
            weight *
            torch::bmm(
                point2point.view(
                    {vertex_map.index({pc_indices}).sizes()[0], 1, 3}),
                normal_map.index({pc_indices})
                    .view({vertex_map.index({pc_indices}).sizes()[0], 3, 1}));

        // data_term =
        //     point2point.abs().mean() * 0.66 + point2plane.abs().mean() *
        //     0.33;

        data_term =
            point2point.abs().mean() * 0.33 + point2plane.abs().mean() * 0.66;
      }
      if (apose.requires_grad() || pca.requires_grad() ||
          shape.requires_grad() || trans.requires_grad()) {
        silhouette_term =
            opt_config["silhouette_weight"].as<float>() *
            torch::l1_loss(corr, color_pred, torch::Reduction::Sum) / seg.sum();
      } else {
        verts_hom.set_requires_grad(false);
        ones.set_requires_grad(false);
        verts_clip.set_requires_grad(false);
        for (unsigned int i = 0; i < rast_out.size(); i++) {
          rast_out[i].set_requires_grad(false);
        }
        for (unsigned int i = 0; i < interp_out.size(); i++) {
          interp_out[i].set_requires_grad(false);
        }
        for (unsigned int i = 0; i < antialias_out.size(); i++) {
          antialias_out[i].set_requires_grad(false);
        }

        color_pred.set_requires_grad(false);
        positive_samples_pred.set_requires_grad(false);
        iou_intersection.set_requires_grad(false);
        iou_union.set_requires_grad(false);
        iou_pred.set_requires_grad(false);
      }

      float regularization_weight_iou = std::exp(-iou_pred.item<float>() + 1);
      float regularization_weight_data = std::log(1 + data_term.item<float>());

      // if (!visible &&
      //     config_["Optimization"]["pose_prior_weight"].as<float>() > 0 &&
      //     apose.requires_grad()) {
      //   if (frame_counter_ <= 1) {
      //     gmm_.weights_ = gmm_.weights_.to(apose.device());
      //     gmm_.means_ = gmm_.means_.to(apose.device());
      //     gmm_.inv_covs_ = gmm_.inv_covs_.to(apose.device());
      //     gmm_.cov_det_ = gmm_.cov_det_.to(apose.device());
      //   }

      //   torch::Tensor likelihood;

      //   if (visible_percentage == 1) {
      //     if (hand.model.hand_type == reclib::models::HandType::right) {
      //       likelihood =
      //           compute_gmm_likelihood(gmm_.weights_, gmm_.means_,
      //                                  gmm_.inv_covs_, gmm_.cov_det_, apose);
      //     } else {
      //       torch::Tensor mirrored_pose = apose.clone();
      //       int c = 0;
      //       for (unsigned int i = 1; i < 16; i++) {
      //         int dof =
      //             reclib::models::MANOConfigExtra::dof_per_anatomic_joint[i];
      //         if (dof > 1) {
      //           mirrored_pose.index({c + 1}) =
      //               -1 * mirrored_pose.index({c + 1});
      //         }
      //         if (dof > 2) {
      //           mirrored_pose.index({c + 2}) =
      //               -1 * mirrored_pose.index({c + 2});
      //         }
      //         c += dof;
      //       }
      //       likelihood = compute_gmm_likelihood(gmm_.weights_, gmm_.means_,
      //                                           gmm_.inv_covs_,
      //                                           gmm_.cov_det_,
      //                                           mirrored_pose);
      //     }
      //   } else {
      //     likelihood = compute_gmm_likelihood(
      //         torch::ones(1, torch::TensorOptions().device(apose.device())),
      //         mu_partial.unsqueeze(0), inv_cov_partial.unsqueeze(0),
      //         det_partial.unsqueeze(0), apose);
      //   }

      //   pose_prior =
      //       config_["Optimization"]["pose_prior_weight"].as<float>() *
      //       -torch::log(
      //           likelihood *
      //           config_["Optimization"]["likelihood_scale"].as<float>());

      //   if (debug_) std::cout << "likelihood: " << likelihood << std::endl;
      // }

      if (apose.requires_grad()) {
        reg_term_pose =
            (opt_config["param_regularizer_pose_weight"].as<float>() * visible +
             opt_config["param_regularizer_pose_weight_invisible"].as<float>() *
                 !visible) *
            apose.norm() * apose.norm();

        if (frame_counter_ > 1 && prev_apose.sizes()[0] > 0) {
          temp_reg_term_pose =
              opt_config["temp_regularizer_pose_weight"].as<float>() *
              (regularization_weight_iou + regularization_weight_data) *
              (torch::mse_loss(apose, prev_apose, torch::Reduction::None) *
               (prev_apose > 0))
                  .sum();
        }
      }

      if (shape.requires_grad()) {
        reg_term_shape =
            opt_config["param_regularizer_shape_weight"].as<float>() *
            shape.norm() * shape.norm();

        if (frame_counter_ > 1) {
          temp_reg_term_shape =
              opt_config["temp_regularizer_shape_weight"].as<float>() *
              torch::nn::functional::mse_loss(shape, prev_shape);
        }
      }

      if (trans.requires_grad()) {
        temp_reg_term_trans =
            opt_config["temp_regularizer_trans_weight"].as<float>() *
            (regularization_weight_iou + regularization_weight_data) *
            torch::nn::functional::mse_loss(trans, prev_trans);
      }

      if (rot.requires_grad()) {
        if (frame_counter_ > 1) {
          temp_reg_term_rot =
              opt_config["temp_regularizer_rot_weight"].as<float>() *
              (regularization_weight_iou + regularization_weight_data) *
              torch::nn::functional::mse_loss(rot, prev_rot);
        }
      }

      if (debug_) {
        if (data_term.item<float>() != 0) {
          std::cout << "-- data: " << data_term.item<float>()
                    << ", weight: " << weight
                    << " (iou:" << iou_pred.item<float>() << ")" << std::endl;
        }
        if (silhouette_term.item<float>() != 0) {
          std::cout << "-- silhouette: " << silhouette_term.item<float>()
                    << std::endl;
        }
        if (pose_prior.item<float>() != 0) {
          std::cout << "-- pose prior (apose): " << pose_prior.item<float>()
                    << std::endl;
        }
        if (reg_term_pose.item<float>() != 0) {
          std::cout << "-- param reg (apose): " << reg_term_pose.item<float>()
                    << std::endl;
        }
        if (reg_term_shape.item<float>() != 0) {
          std::cout << "-- param reg (shape): " << reg_term_shape.item<float>()
                    << std::endl;
        }
        if (temp_reg_term_pose.item<float>() != 0) {
          std::cout << "-- temp reg (apose): "
                    << temp_reg_term_pose.item<float>()
                    << " iou_w: " << regularization_weight_iou
                    << " data_w: " << regularization_weight_data << std::endl;
        }
        if (temp_reg_term_shape.item<float>() != 0) {
          std::cout << "-- temp reg (shape): "
                    << temp_reg_term_shape.item<float>()
                    << " iou_w: " << regularization_weight_iou
                    << " data_w: " << regularization_weight_data << std::endl;
        }
        if (temp_reg_term_trans.item<float>() != 0) {
          std::cout << "-- temp reg (trans): "
                    << temp_reg_term_trans.item<float>()
                    << " iou_w: " << regularization_weight_iou
                    << " data_w: " << regularization_weight_data << std::endl;
        }
        if (temp_reg_term_rot.item<float>() != 0) {
          std::cout << "-- temp reg (rot): " << temp_reg_term_rot.item<float>()
                    << " iou_w: " << regularization_weight_iou
                    << " data_w: " << regularization_weight_data << std::endl;
        }

        std::cout << "--------------------------------------" << std::endl;
      }

      torch::Tensor loss = data_term + reg_term_pose + temp_reg_term_pose +
                           silhouette_term + temp_reg_term_trans +
                           temp_reg_term_shape + reg_term_shape + pose_prior;

      loss.backward();

      if (1 && apose.requires_grad()) {
        for (int i = 0; i < apose.sizes()[0]; i++) {
          int joint = apose2joint[i];
          if ((bool)visibility_joints[joint] != visible) {
            apose.mutable_grad().index({i}) = 0;
          } else {
            // std::cout << "keep grad " << i << "," << joint << std::endl;
          }
        }
      }

      return loss;
    };

    auto loss_func_wrapper = [&]() {
      optim_rot_pose_trans.zero_grad();
      torch::Tensor loss = loss_func_rot_apose(true);
      return loss;
    };

    // for (int j = 0; j < opt_config["inner_iterations_visible"].as<int>();
    // j++) {

    if (stage == 0 || !config_["Debug"]["separate_stages"].as<bool>()) {
      if ((pca -
           torch::matmul(hand.model.hand_comps.inverse(), hand.model.hand_mean)
               .squeeze(1))
              .abs()
              .max()
              .template item<float>() > 1e-2f) {
        std::cout << "PCA already optimized." << std::endl;
        std::cout << "diff: "
                  << (pca - torch::matmul(hand.model.hand_comps.inverse(),
                                          hand.model.hand_mean)
                                .squeeze(1))
                         .abs()
                         .max()
                  << std::endl;

        hand_model.cpu();
        hand.cpu();

        return true;
      }

      // apose.set_requires_grad(true);
      pca.set_requires_grad(true);
      rot.set_requires_grad(true);
      shape.set_requires_grad(true);
      // trans.set_requires_grad(true);

      {
        torch::Tensor prev_apose_optim = apose.clone().detach();
        torch::Tensor prev_pca_optim = pca.clone().detach();

        visible = true;

        optim_rot_pose_trans.step(loss_func_wrapper);

        if (debug_)
          std::cout << "diff: "
                    << (pca - prev_pca_optim).abs().sum().item<float>()
                    << std::endl;
      }

      if (debug_)
        std::cout << "[ITERATIONS]:" << opt_visible.max_iter() << std::endl;

      // apose.set_requires_grad(false);
      pca.set_requires_grad(false);
      rot.set_requires_grad(false);
      shape.set_requires_grad(false);
      // trans.set_requires_grad(false);
    }

    if (!config_["Debug"]["separate_stages"].as<bool>()) {
      torch::NoGradGuard guard;
      apose = torch::matmul(hand.model.hand_comps, pca.clone()).reshape({-1});
    }

    torch::optim::Adam optim_rot_pose_trans_refined =
        torch::optim::Adam({trans, rot, shape, apose}, adam_opt_visible);
    torch::optim::Adam optim_rot_pose_refined_ =
        torch::optim::Adam({shape, apose}, adam_opt_visible);
    torch::optim::Adam optim_trans_ =
        torch::optim::Adam({trans, rot}, adam_opt_visible);

    int epochs = opt_config.i("epochs");
    if (frame_counter_ <= 1) {
      epochs = opt_config.i("epochs_initial");
    }

    torch::optim::StepLR optim_rot_pose_refined = torch::optim::StepLR(
        optim_rot_pose_refined_, epochs, opt_config.f("step_size"));
    torch::optim::StepLR optim_trans =
        torch::optim::StepLR(optim_trans_, epochs, opt_config.f("step_size"));

    if (stage == 1 || !config_["Debug"]["separate_stages"].as<bool>()) {
      int termination_iter =
          opt_config["termination_max_iter_visible"].as<int>();
      if (frame_counter_ <= 1 || last_frame_.loss_[cls_label] < 0) {
        termination_iter =
            opt_config["termination_max_iter_visible_initialization"].as<int>();
      }

      apose.set_requires_grad(true);
      rot.set_requires_grad(true);
      shape.set_requires_grad(true);
      trans.set_requires_grad(true);
      torch::Tensor best_apose = apose.clone().detach();
      torch::Tensor best_trans = trans.clone().detach();
      torch::Tensor best_rot = rot.clone().detach();
      torch::Tensor best_shape = shape.clone().detach();
      float best_loss = -1;

      int iter = 0;
      torch::Tensor prev_apose_optim;
      torch::Tensor prev_rot_optim;
      torch::Tensor prev_trans_optim;
      torch::Tensor loss;
      while (true) {
        prev_apose_optim = apose.clone().detach();
        prev_rot_optim = rot.clone().detach();
        prev_trans_optim = trans.clone().detach();

        visible = true;

        if (opt_config.b("use_flip_flop")) {
          apose.set_requires_grad(true);
          rot.set_requires_grad(false);
          shape.set_requires_grad(true);
          trans.set_requires_grad(false);
          optim_rot_pose_refined_.zero_grad();
          loss = loss_func_rot_apose(visible);
          optim_rot_pose_refined_.step();
          optim_rot_pose_refined.step();

          if (apose.isnan().any().item<bool>()) {
            throw std::runtime_error("Apose is NaN");
          }
          if (shape.isnan().any().item<bool>()) {
            throw std::runtime_error("Shape is NaN");
          }

          apose.set_requires_grad(false);
          rot.set_requires_grad(true);
          shape.set_requires_grad(false);
          trans.set_requires_grad(true);
          optim_trans_.zero_grad();
          loss = loss_func_rot_apose(visible);
          optim_trans_.step();
          optim_trans.step();

          if (rot.isnan().any().item<bool>()) {
            throw std::runtime_error("Rot is NaN");
          }
          if (trans.isnan().any().item<bool>()) {
            throw std::runtime_error("Trans is NaN");
          }

        } else {
          optim_rot_pose_trans_refined.zero_grad();
          loss = loss_func_rot_apose(visible);
          optim_rot_pose_trans_refined.step();
        }

        if (best_loss == -1) {
          torch::NoGradGuard guard;
          best_apose = apose.clone().detach();
          best_trans = trans.clone().detach();
          best_rot = rot.clone().detach();
          best_shape = shape.clone().detach();
          best_loss = loss.clone().detach().item<float>();
        } else if (iter > 0 && (best_loss > loss).all().item<bool>()) {
          best_apose = apose.clone().detach();
          best_trans = trans.clone().detach();
          best_rot = rot.clone().detach();
          best_shape = shape.clone().detach();
          best_loss = loss.clone().detach().item<float>();
        }

        float diff = (apose - prev_apose_optim).abs().sum().item<float>();
        diff += (rot - prev_rot_optim).abs().sum().item<float>();
        diff += (trans - prev_trans_optim).abs().sum().item<float>();
        // diff = diff / 3.f;

        // std::cout << "diff: " << diff << std::endl;

        if ((diff <= opt_config["termination_eps_visible"].as<float>()) ||
            iter > termination_iter) {
          break;
        }
        // if (debug_)
        //   std::cout << "diff: "
        //             << (pca - prev_pca_optim).abs().sum().item<float>()
        //             << std::endl;

        // if ((pca - prev_pca_optim).abs().sum().item<float>() <=
        //         opt_config["termination_eps_visible"].as<float>() ||
        //     iter > opt_config["termination_max_iter_visible"].as<int>()) {
        //   break;
        // }

        iter++;
      }
      if (debug_) {
        std::cout << "Loss: " << loss << std::endl;
        std::cout << "Best loss: " << best_loss << std::endl;
        std::cout << "Last frame best loss: " << last_frame_.loss_[cls_label]
                  << std::endl;
        std::cout << "Difference in losses: "
                  << (best_loss - last_frame_.loss_[cls_label]) /
                         last_frame_.loss_[cls_label]
                  << std::endl;
      }

      apose.set_requires_grad(false);
      rot.set_requires_grad(false);
      shape.set_requires_grad(false);
      trans.set_requires_grad(false);

      pca = torch::matmul(hand.model.hand_comps.inverse(), apose.unsqueeze(1))
                .reshape({-1});
      if (pca.isnan().any().item<bool>()) {
        throw std::runtime_error("PCA is NaN");
      }

      // pca = torch::matmul(hand.model.hand_comps.inverse(),
      //                     best_apose.unsqueeze(1))
      //           .reshape({-1});

      if (debug_) {
        std::cout << "[ITERATIONS]:" << iter << ", " << termination_iter
                  << std::endl;
      }

      if (last_frame_.loss_[cls_label] > 0 &&
          (best_loss - last_frame_.loss_[cls_label]) /
                  last_frame_.loss_[cls_label] >
              opt_config.f("loss_relative_threshold")) {
        if (debug_) {
          std::cout << "Loss higher than last one: " << best_loss << " <-> "
                    << last_frame_.loss_[cls_label] << std::endl;
        }
        hand_model.cpu();
        hand.cpu();

        return false;
      }
      if (last_frame_.loss_[cls_label] > 0 &&
          best_loss > opt_config.f("loss_absolute_threshold")) {
        if (debug_) {
          std::cout << "Loss higher than absolute thresh: " << best_loss
                    << " <-> " << opt_config.f("loss_absolute_threshold")
                    << std::endl;
        }
        hand_model.cpu();
        hand.cpu();

        return false;
      }
      last_frame_.loss_[cls_label] = best_loss;
    }

    if (stage == 2) {
      shape.set_requires_grad(true);
      apose.set_requires_grad(false);
      trans.set_requires_grad(false);
      rot.set_requires_grad(false);
      pca.set_requires_grad(false);
      torch::optim::Adam optim_shape =
          torch::optim::Adam({shape}, adam_opt_visible);

      int iter = 0;
      while (true) {
        torch::Tensor prev_shape_optim = shape.clone().detach();

        visible = false;
        optim_shape.zero_grad();
        torch::Tensor loss = loss_func_rot_apose(visible);
        optim_shape.step();

        if (debug_)
          std::cout << "diff: "
                    << (shape - prev_shape_optim).abs().sum().item<float>()
                    << std::endl;

        if ((shape - prev_shape_optim).abs().sum().item<float>() <=
                opt_config["termination_eps_visible"].as<float>() ||
            iter > opt_config["termination_max_iter_visible"].as<int>()) {
          break;
        }

        iter++;
      }
      shape.set_requires_grad(false);

      if (debug_) std::cout << "Iterations: " << iter << std::endl;
    }
  }

  if (debug_)
    std::cout << "optimization: " << timer.look_and_reset() << " ms."
              << std::endl;

  hand.params.index_put_({torch::indexing::Slice(0, 3)}, trans);
  hand.params.index_put_({torch::indexing::Slice(3 + 3 + 45, 3 + 3 + 45 + 23)},
                         pca);
  hand.params.index_put_(
      {torch::indexing::Slice(3 + 3 + 45 + 23, 3 + 3 + 45 + 23 + 10)}, shape);
  hand.anatomic_params.index_put_({torch::indexing::Slice(0, 3)}, rot);
  hand.update();

  hand_model.cpu();
  hand.cpu();

  if (debug_)
    std::cout << "postprocessing: " << timer.look_and_reset() << " ms."
              << std::endl;

  if (config_["Debug"]["show_corr_lines"].as<bool>()) {
    torch::Tensor mano_corr_ind_canonical =
        torch::from_blob((void*)corr_pairs.mano_corr_.canonical_ind_.data(),
                         {(int)corr_pairs.mano_corr_.canonical_ind_.size()},
                         torch::TensorOptions().dtype(torch::kInt))
            .toType(torch::kLong);
    torch::Tensor mano_corr_ind_silhouette =
        torch::from_blob((void*)corr_pairs.mano_corr_.silhouette_ind_.data(),
                         {(int)corr_pairs.mano_corr_.silhouette_ind_.size()},
                         torch::TensorOptions().dtype(torch::kInt))
            .toType(torch::kLong);
    torch::Tensor corr_pos_ref_canonical =
        hand.verts().index({mano_corr_ind_canonical});
    torch::Tensor corr_pos_ref_silhouette =
        hand.verts().index({mano_corr_ind_silhouette});
    torch::Tensor corr_pos_ref =
        torch::cat({corr_pos_ref_canonical, corr_pos_ref_silhouette});
    std::vector<vec3> corr_pos_src;
    std::vector<vec3> corr_colors;
    std::vector<uint32_t> indices;
    for (unsigned int i = 0; i < corr_pairs.mano_corr_.canonical_ind_.size();
         i++) {
      vec3 color = vec3::Random().normalized().cwiseAbs();
      color.x() = 0;
      corr_colors.push_back(color);
      corr_colors.push_back(color);

      vec3 src(&(last_frame_.vertex_maps_[cls_label].ptr<float>(
          0)[corr_pairs.pc_corr_.canonical_ind_[i] * 3]));

      corr_pos_src.push_back(src);
      indices.push_back(i);
      indices.push_back(corr_pos_ref.sizes()[0] + i);
    }
    // for (unsigned int i = 0; i <
    // corr_pairs.mano_corr_.silhouette_ind_.size();
    //      i++) {
    //   vec3 color = vec3::Random().normalized().cwiseAbs();
    //   color.y() = 0;
    //   color.z() = 0;
    //   corr_colors.push_back(color);
    //   corr_colors.push_back(color);

    //   vec3 src(&(last_frame_.vertex_maps_[cls_label].ptr<float>(
    //       0)[corr_pairs.pc_corr_.silhouette_ind_[i] * 3]));

    //   corr_pos_src.push_back(src);
    //   indices.push_back(i + corr_pairs.mano_corr_.canonical_ind_.size());
    //   indices.push_back(corr_pos_ref.sizes()[0] + i +
    //                     corr_pairs.mano_corr_.canonical_ind_.size());
    // }
    torch::Tensor corr_pos_src_ =
        torch::from_blob(corr_pos_src[0].data(), {(int)corr_pos_src.size(), 3});

    torch::Tensor corr_pos =
        torch::cat({corr_pos_ref, corr_pos_src_}).contiguous();

    if (!reclib::opengl::Shader::valid(
            "canonicalPointCorrespondences_colorPV")) {
      reclib::opengl::Shader("canonicalPointCorrespondences_colorPV",
                             "MVP_color3.vs", "color3PV.fs");
    }

    if (!reclib::opengl::Drawelement::valid("corr_points_" +
                                            std::to_string(cls_label))) {
      reclib::opengl::Drawelement corr_points =
          reclib::opengl::DrawelementImpl::from_geometry(
              "corr_points_" + std::to_string(cls_label),
              reclib::opengl::Shader::find(
                  "canonicalPointCorrespondences_colorPV"),
              false, corr_pos.data<float>(), corr_pos.sizes()[0],
              (uint32_t*)indices.data(), indices.size());
      corr_points->mesh->primitive_type = GL_POINTS;
      corr_points->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                      false, true);
      corr_points->add_pre_draw_func("pointsize", [&]() { glPointSize(10.f); });
      corr_points->add_post_draw_func("pointsize", [&]() { glPointSize(1.f); });
      corr_points->mesh->geometry->update_meshes();

      reclib::opengl::Drawelement corr_lines =
          reclib::opengl::DrawelementImpl::from_geometry(
              "corr_lines_" + std::to_string(cls_label),
              reclib::opengl::Shader::find(
                  "canonicalPointCorrespondences_colorPV"),
              false, corr_pos.data<float>(), corr_pos.sizes()[0],
              (uint32_t*)indices.data(), indices.size());
      corr_lines->mesh->primitive_type = GL_LINES;
      corr_lines->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                     false, true);
      corr_lines->add_pre_draw_func("linewidth", [&]() { glLineWidth(3.f); });
      corr_lines->add_post_draw_func("linewidth", [&]() { glLineWidth(1.f); });
      corr_lines->mesh->geometry->update_meshes();
    } else {
      reclib::opengl::Drawelement corr_points =
          reclib::opengl::Drawelement::find("corr_points_" +
                                            std::to_string(cls_label));
      corr_points->mesh->geometry.cast<reclib::opengl::GeometryImpl>()->set(
          corr_pos.data<float>(), corr_pos.sizes()[0],
          (uint32_t*)indices.data(), indices.size());
      corr_points->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                      false, true);
      corr_points->mesh->geometry->update_meshes();

      reclib::opengl::Drawelement corr_lines =
          reclib::opengl::Drawelement::find("corr_lines_" +
                                            std::to_string(cls_label));
      corr_lines->mesh->geometry.cast<reclib::opengl::GeometryImpl>()->set(
          corr_pos.data<float>(), corr_pos.sizes()[0],
          (uint32_t*)indices.data(), indices.size());
      corr_lines->mesh->geometry->add_attribute_vec3("color", corr_colors,
                                                     false, true);
      corr_lines->mesh->geometry->update_meshes();
    }
  }
  return true;
}

template <typename MANOTYPE>
cv::Rect reclib::tracking::ManoRGBD<MANOTYPE>::compute_crop_factor(
    const CpuMat& rgb) {
  torch::Tensor box = last_frame_.network_output_[2].to(torch::kCPU);
  torch::Tensor union_box = torch::zeros(4);
  union_box.index({0}) = torch::min(box.index({torch::All, 0}));
  union_box.index({1}) = torch::min(box.index({torch::All, 1}));
  union_box.index({2}) = torch::max(box.index({torch::All, 2}));
  union_box.index({3}) = torch::max(box.index({torch::All, 3}));

  cv::Rect crop_factor_new;
  crop_factor_new.x =
      std::fmax(0, union_box.index({0}).item<float>() -
                       config_["Input"]["crop_pad_x"].as<float>());
  crop_factor_new.y =
      std::fmax(0, union_box.index({1}).item<float>() -
                       config_["Input"]["crop_pad_y"].as<float>());
  crop_factor_new.width =
      std::fmin(rgb.cols - 1, union_box.index({2}).item<float>() +
                                  config_["Input"]["crop_pad_x"].as<float>()) -
      crop_factor_new.x;
  crop_factor_new.height =
      std::fmin(rgb.rows - 1, union_box.index({3}).item<float>() +
                                  config_["Input"]["crop_pad_y"].as<float>()) -
      crop_factor_new.y;

  return crop_factor_new;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::detect_openpose(
    unsigned int index, const CpuMat& rgb, cv::Rect crop_factor) {
  int cls_label = last_frame_.network_output_[0]
                      .index({(int)index})
                      .to(torch::kCPU)
                      .item()
                      .toInt();

  CpuMat rgb_cropped = rgb;
  if (crop_factor.x >= 0) {
    CpuMat img_float_tmp(crop_factor.height, crop_factor.width, rgb.type());

    img_float_tmp = rgb(crop_factor).clone();
    rgb_cropped = img_float_tmp;
  }

  CpuMat input_blob = cv::dnn::blobFromImage(
      rgb_cropped, config_["Network"]["openpose_scale"].as<double>(),
      cv::Size(368, 368));
  openpose_.setInput(input_blob);
  CpuMat result = openpose_.forward();
  int H = result.size[2];
  int W = result.size[3];

  std::vector<vec3> color_pj;

  // connect body parts and draw it !
  float SX = float(rgb_cropped.cols) / W;
  float SY = float(rgb_cropped.rows) / H;

  // thumb
  color_pj.push_back(vec3(1, 0, 0));
  color_pj.push_back(vec3(1, 0, 0));
  color_pj.push_back(vec3(1, 0, 0));
  color_pj.push_back(vec3(1, 0, 0));
  // index
  color_pj.push_back(vec3(0, 1, 0));
  color_pj.push_back(vec3(0, 1, 0));
  color_pj.push_back(vec3(0, 1, 0));
  color_pj.push_back(vec3(0, 1, 0));
  // middle
  color_pj.push_back(vec3(1, 0.5, 0));
  color_pj.push_back(vec3(1, 0.5, 0));
  color_pj.push_back(vec3(1, 0.5, 0));
  color_pj.push_back(vec3(1, 0.5, 0));
  // ring
  color_pj.push_back(vec3(0.9, 0, 1));
  color_pj.push_back(vec3(0.9, 0, 1));
  color_pj.push_back(vec3(0.9, 0, 1));
  color_pj.push_back(vec3(0.9, 0, 1));
  // pinky
  color_pj.push_back(vec3(0, 0.5, 1));
  color_pj.push_back(vec3(0, 0.5, 1));
  color_pj.push_back(vec3(0, 0.5, 1));
  color_pj.push_back(vec3(0, 0.5, 1));

  float thresh = config_["Network"]["openpose_threshold"].as<float>();
  torch::Tensor keypoints = torch::ones({22, 2}) * -1;
  for (int n = 0; n < 22; n++) {
    // Slice heatmap of corresponding body's part.
    CpuMat heatMap(H, W, CV_32F, result.ptr(0, n));
    // std::cout << "rows: " << heatMap.rows << "," << heatMap.cols <<
    // std::endl;

    // CpuMat resized_heatMap;
    // cv::resize(heatMap, resized_heatMap, cv::Size(800, 800));
    // resized_heatMap = resized_heatMap * 10;
    // cv::imshow("heatmap", resized_heatMap);
    // cv::waitKey(0);
    // 1 maximum per heatmap
    cv::Point p(-1, -1), pm;
    double conf;
    cv::minMaxLoc(heatMap, 0, &conf, 0, &pm);
    if (conf > thresh) p = pm;

    if (p.x > -1 && p.y > -1) {
      p.x = p.x * SX + crop_factor.x;
      p.y = p.y * SY + crop_factor.y;
    }

    keypoints.index({n, 0}) = p.x;
    keypoints.index({n, 1}) = p.y;
  }

  if (config_["Debug"]["show_keypoints"].as<bool>()) {
    std::vector<cv::Point> points(22);
    for (int n = 0; n < 22; n++) {
      cv::Point p(-1, -1), pm;
      p.x = keypoints.index({n, 0}).item<float>();
      p.y = keypoints.index({n, 1}).item<float>();
      points[n] = p;
    }

    for (int n = 0; n < 20; n++) {
      // lookup 2 connected body/hand parts
      cv::Point2f a = points[OPENPOSE_KEYPOINT_PAIRS[n][0]];
      cv::Point2f b = points[OPENPOSE_KEYPOINT_PAIRS[n][1]];
      // we did not find enough confidence before
      if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0) continue;
      // scale to image size
      // a.x = a.x * SX + crop_factor.x;
      // a.y = a.y * SY + crop_factor.y;
      // b.x = b.x * SX + crop_factor.x;
      // b.y = b.y * SY + crop_factor.y;
      vec3 c = color_pj[n] * 255;
      ;

      cv::line(rgb, a, b, cv::Scalar(c.z(), c.y(), c.x()), 5);
      cv::circle(rgb, a, 3, cv::Scalar(255, 255, 255), 5);
      cv::circle(rgb, b, 3, cv::Scalar(255, 255, 255), 5);
    }

    if (config_["Debug"]["save_openpose"].as<bool>()) {
      fs::path output_dir =
          config_["Debug"]["screenshot_output_folder"].as<std::string>();
      fs::path openpose_path = "openpose";
      std::stringstream filename;
      filename << std::setw(6) << std::setfill('0') << global_frame_counter_
               << ".png";
      if (!fs::exists(output_dir / openpose_path)) {
        reclib::utils::CreateDirectories({output_dir / openpose_path});
      }

      cv::imwrite(output_dir / openpose_path / fs::path(filename.str()), rgb);
    } else if (config_["Debug"]["show_keypoints"].as<bool>()) {
      cv::imshow("OpenPose", rgb);
      cv::waitKey(0);
    }
  }

  last_frame_.keypoints_[cls_label] = keypoints;
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::detect(const CpuMat& rgb,
                                                  cv::Rect crop_factor) {
  CpuMat img_float;
  rgb.convertTo(img_float, CV_32FC3);

  unsigned int orig_image_h = img_float.rows;
  unsigned int orig_image_w = img_float.cols;
  if (crop_factor.height > 0) {
    CpuMat img_float_tmp(crop_factor.height, crop_factor.width,
                         img_float.type());

    img_float_tmp = img_float(crop_factor).clone();
    img_float = img_float_tmp;
  }
  torch::Tensor input_float = reclib::dnn::cv2torch(img_float);
  unsigned int image_h = input_float.sizes()[0];
  unsigned int image_w = input_float.sizes()[1];

  std::vector<torch::Tensor> output = transform_.transform({input_float});
  torch::Tensor net_input = output[0].permute({2, 0, 1});
  unsigned int transformed_h = output[0].sizes()[0];
  unsigned int transformed_w = output[0].sizes()[1];

  if (debug_) {
    std::cout << "---- Transforming input: " << timer_.look_and_reset()
              << " ms." << std::endl;
  }

  if (debug_ && config_["Debug"]["show_input"].as<bool>() == true) {
    torch::Tensor output_img = output[0].abs().contiguous();
    CpuMat output_mat = reclib::dnn::torch2cv(output_img, true);
    cv::imshow("Input (transformed)", output_mat);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    timer_.look_and_reset();
  }

  auto tmp = network_.forward(net_input.to(torch::kCUDA), false);

  if (debug_) {
    std::cout << "---- Network forward: " << timer_.look_and_reset() << " ms."
              << std::endl;
  }

  std::vector<std::vector<torch::Tensor>> out_detection = network_.detect(tmp);

  std::vector<torch::Tensor> out_processed = network_.postprocess(
      out_detection, transformed_w, transformed_h, 0, false, false);

  out_processed = reduce(out_processed);

  out_processed = transform_.untransform(out_processed, image_w, image_h);
  // last_frame_.network_output_ = reduce(out_processed);

  if (crop_factor.width > 0) {
    // rescale box
    out_processed[2].index({torch::All, 0}) += crop_factor.x;
    out_processed[2].index({torch::All, 1}) += crop_factor.y;
    out_processed[2].index({torch::All, 2}) += crop_factor.x;
    out_processed[2].index({torch::All, 3}) += crop_factor.y;

    if (config_["Input"]["manual_pad_x"].as<int>() > 0) {
      int pad = config_["Input"]["manual_pad_x"].as<int>();
      out_processed[2].index({torch::All, 0}) = torch::clamp(
          out_processed[2].index({torch::All, 0}) - pad, 0, (int)orig_image_w);
      out_processed[2].index({torch::All, 2}) = torch::clamp(
          out_processed[2].index({torch::All, 2}) + pad, 0, (int)orig_image_w);
    }
    if (config_["Input"]["manual_pad_y"].as<int>() > 0) {
      int pad = config_["Input"]["manual_pad_y"].as<int>();
      out_processed[2].index({torch::All, 1}) = torch::clamp(
          out_processed[2].index({torch::All, 1}) - pad, 0, (int)orig_image_h);
      out_processed[2].index({torch::All, 3}) = torch::clamp(
          out_processed[2].index({torch::All, 3}) + pad, 0, (int)orig_image_h);
    }

    // rescale mask
    torch::Tensor expanded_masks =
        torch::zeros({out_processed[3].sizes()[0], rgb.rows, rgb.cols},
                     torch::TensorOptions().device(out_processed[3].device()));

    expanded_masks.index_put_(
        {torch::All,
         torch::indexing::Slice(crop_factor.y,
                                crop_factor.y + crop_factor.height),
         torch::indexing::Slice(crop_factor.x,
                                crop_factor.x + crop_factor.width)},
        out_processed[3]);
    out_processed[3] = expanded_masks;

    torch::Tensor expanded_corrs =
        torch::zeros({out_processed[4].sizes()[0], 3, rgb.rows, rgb.cols},
                     torch::TensorOptions().device(out_processed[4].device()));
    expanded_corrs.index_put_(
        {torch::All, torch::All,
         torch::indexing::Slice(crop_factor.y,
                                crop_factor.y + crop_factor.height),
         torch::indexing::Slice(crop_factor.x,
                                crop_factor.x + crop_factor.width)},
        out_processed[4]);
    out_processed[4] = expanded_corrs;
  }
  last_frame_.network_output_ = out_processed;

  if (debug_) {
    std::cout << "---- Untransforming network output: "
              << timer_.look_and_reset() << " ms." << std::endl;
  }
}

template <typename MANOTYPE>
void reclib::tracking::ManoRGBD<MANOTYPE>::process(
    const CpuMat& rgb, const CpuMat& depth,
    const reclib::IntrinsicParameters& intrinsics, cv::Rect crop_factor) {
  if (frame_counter_ == 0) {
    for (unsigned int i = 0; i < last_frame_.pose_.size(); i++) {
      last_frame_.visibility_map_[i] = std::vector<uint8_t>(20);
      last_frame_.trans_[i] = torch::zeros(
          3, torch::TensorOptions().device(mano_right_.trans().device()));
      last_frame_.rot_[i] = torch::zeros(
          3, torch::TensorOptions().device(mano_right_.trans().device()));
      last_frame_.pose_[i] = torch::zeros(
          23, torch::TensorOptions().device(mano_right_.trans().device()));
      last_frame_.shape_[i] = torch::zeros(
          10, torch::TensorOptions().device(mano_right_.trans().device()));
      last_frame_.pca_[i] = torch::zeros(
          23, torch::TensorOptions().device(mano_right_.trans().device()));
      last_frame_.loss_[i] = -1;
    }
  }

  last_frame_.rgb_ = rgb;
  last_frame_.depth_ = depth;
  for (unsigned int i = 0; i < last_frame_.corr_indices_.size(); i++) {
    last_frame_.corr_indices_[i].reset();  // reset kdtree
  }

  if (frame_counter_ > 0) {
    {
      torch::Tensor pose =
          torch::matmul(mano_left_.model.hand_comps, mano_left_.hand_pca());
      std::vector<uint8_t>& visibility = last_frame_.visibility_map_.at(0);
      for (int i = 0; i < 23; i++) {
        if (visibility[apose2vis_indices[i]] > 0) {
          // last_frame_.pose_[0].index_put_({i},
          // mano_left_.apose().index({i}));
          last_frame_.pose_[0].index_put_({i}, pose.index({i}));
        }
      }
    }
    {
      torch::Tensor pose =
          torch::matmul(mano_right_.model.hand_comps, mano_right_.hand_pca());
      std::vector<uint8_t>& visibility = last_frame_.visibility_map_.at(1);
      for (int i = 0; i < 23; i++) {
        if (visibility[apose2vis_indices[i]] > 0) {
          // last_frame_.pose_[1].index_put_({i},
          // mano_right_.apose().index({i}));
          last_frame_.pose_[1].index_put_({i}, pose.index({i}));
        }
      }
    }

    last_frame_.trans_[0] = mano_left_.trans();
    last_frame_.shape_[0] = mano_left_.shape();
    last_frame_.rot_[0] = mano_left_.rot();
    last_frame_.pca_[0] = mano_left_.hand_pca();
    last_frame_.trans_[1] = mano_right_.trans();
    last_frame_.shape_[1] = mano_right_.shape();
    last_frame_.rot_[1] = mano_right_.rot();
    last_frame_.pca_[1] = mano_right_.hand_pca();
  }

  // save MANO parameters for temporal regularization
  // last_frame_.trans_[0] = mano_left_.trans();
  // last_frame_.shape_[0] = mano_left_.shape();
  // last_frame_.pose_[0] = mano_left_.pose();
  // last_frame_.pose_pca_[0] = mano_left_.hand_pca();

  // last_frame_.trans_[1] = mano_right_.trans();
  // last_frame_.shape_[1] = mano_right_.shape();
  // last_frame_.pose_[1] = mano_right_.pose();
  // last_frame_.pose_pca_[1] = mano_right_.hand_pca();
  frame_counter_++;
  global_frame_counter_++;
  optimization_counter_ = 0;

  if (intrinsics.image_height_ > 0) {
    intrinsics_ = intrinsics;
  }

  if (config_["Debug"]["show_rgbd"].as<bool>() == true) {
    CpuMat rgb_float;
    cv::cvtColor(rgb, rgb_float, cv::COLOR_BGR2RGB);
    rgb_float.convertTo(rgb_float, CV_32FC3);
    rgb_float = rgb_float / 255;

    CpuMat depth_float;
    depth.convertTo(depth_float, CV_32FC1);

    if (!reclib::opengl::Shader::valid("colorPV")) {
      reclib::opengl::Shader("colorPV", "MVP_norm_color3.vs", "color3PV.fs");
    }

    std::string name = "pc_frame";
    reclib::opengl::Mesh m = reclib::opengl::pointcloud_norm_color(
        depth_float, intrinsics_, rgb_float, true, name, CpuMat(0, 0, CV_32FC1),
        ivec2(0, 0), 0.001f);
    if (!reclib::opengl::Drawelement::valid(name)) {
      reclib::opengl::Drawelement d(name,
                                    reclib::opengl::Shader::find("colorPV"), m);
      d->add_pre_draw_func("pre", [&]() { glPointSize(3.f); });
      d->add_post_draw_func("post", [&]() { glPointSize(1.f); });
    } else {
      reclib::opengl::Drawelement d = reclib::opengl::Drawelement::find(name);
      d->mesh = m;
    }

    generated_drawelements_[name] = reclib::opengl::Drawelement::find(name);
  }

  if (config_["Input"]["crop_to_bb"].as<bool>() && frame_counter_ > 1) {
    crop_factor = compute_crop_factor(rgb);
  }
  last_frame_.network_output_.clear();

  if (config_["Debug"]["show_input"].as<bool>() == true) {
    cv::imshow("Input: ", rgb);
  }

  if (debug_) {
    timer_.begin();
  }

  detect(rgb, crop_factor);

  cv::Rect crop_factor_new;
  if (frame_counter_ <= 1 && config_["Input"]["crop_to_bb"].as<bool>()) {
    cv::Rect crop_factor_new = compute_crop_factor(rgb);
    detect(rgb, crop_factor_new);
  }

  if (config_["Network"]["use_openpose"].as<bool>()) {
    for (int i = 0; i < last_frame_.network_output_[2].sizes()[0]; i++) {
      // cv::Rect crop_factor_openpose;
      // crop_factor_openpose.x =
      //     last_frame_.network_output_[2].index({i, 0}).item<int>();
      // crop_factor_openpose.y =
      //     last_frame_.network_output_[2].index({i, 1}).item<int>();
      // crop_factor_openpose.width =
      //     last_frame_.network_output_[2].index({i, 2}).item<int>() -
      //     last_frame_.network_output_[2].index({i, 0}).item<int>();
      // crop_factor_openpose.height =
      //     last_frame_.network_output_[2].index({i, 3}).item<int>() -
      //     last_frame_.network_output_[2].index({i, 1}).item<int>();

      cv::Rect crop_factor_openpose;
      crop_factor_openpose.x = std::fmax(
          0, last_frame_.network_output_[2].index({i, 0}).item<float>() -
                 config_["Input"]["crop_pad_x_openpose"].as<float>());
      crop_factor_openpose.y = std::fmax(
          0, last_frame_.network_output_[2].index({i, 1}).item<float>() -
                 config_["Input"]["crop_pad_y_openpose"].as<float>());
      crop_factor_openpose.width =
          std::fmin(rgb.cols - 1,
                    last_frame_.network_output_[2].index({i, 2}).item<float>() +
                        config_["Input"]["crop_pad_x_openpose"].as<float>()) -
          crop_factor_openpose.x;
      crop_factor_openpose.height =
          std::fmin(rgb.rows - 1,
                    last_frame_.network_output_[2].index({i, 3}).item<float>() +
                        config_["Input"]["crop_pad_y_openpose"].as<float>()) -
          crop_factor_openpose.y;

      // crop_factor_openpose.x = -1;

      detect_openpose(i, rgb, crop_factor_openpose);
    }
  }

  if (config_["Debug"]["show_output"].as<bool>() == true) {
    CpuMat img_float;
    rgb.convertTo(img_float, CV_32FC3);
    torch::Tensor input_float = reclib::dnn::cv2torch(img_float);

    int num_samples = last_frame_.network_output_[0].sizes()[0];
    for (int i = 0; i < num_samples; i++) {
      torch::Tensor box = last_frame_.network_output_[2]
                              .index({i})
                              .to(torch::kCPU)
                              .contiguous();
      torch::Tensor mask = last_frame_.network_output_[3]
                               .index({i})
                               .to(torch::kCPU)
                               .contiguous();
      // mask.index({torch::indexing::Slice(box.index({1}).item<int>(),
      //                                    box.index({3}).item<int>()),
      //             torch::indexing::Slice(box.index({0}).item<int>(),
      //                                    box.index({2}).item<int>())}) =
      //                                    0.5;
      torch::Tensor corr_masked =
          last_frame_.network_output_[4]
              .index({i})
              .to(torch::kCPU)
              .contiguous() *
          mask.index({torch::None, torch::All, torch::All});

      torch::Tensor corr_hsv = corr_masked;
      corr_hsv[0] *= 360;

      torch::Tensor input_masked =
          (input_float *
           (mask.index({torch::All, torch::All, torch::None}) == 0))
              .permute({2, 0, 1}) /
          255;

      corr_hsv = corr_hsv.permute({1, 2, 0}).contiguous();
      input_masked = input_masked.permute({1, 2, 0}).contiguous();

      CpuMat mask_img = reclib::dnn::torch2cv(mask, false);
      CpuMat corr_img = reclib::dnn::torch2cv(corr_hsv, false);
      CpuMat input_masked_img = reclib::dnn::torch2cv(input_masked, false);
      cv::cvtColor(corr_img, corr_img, cv::COLOR_HSV2BGR);

      cv::Point p1;
      p1.x = box.index({0}).item<int>();
      p1.y = box.index({1}).item<int>();

      cv::Point p2;
      p2.x = box.index({0}).item<int>();
      p2.y = box.index({3}).item<int>();

      cv::Point p3;
      p3.x = box.index({2}).item<int>();
      p3.y = box.index({1}).item<int>();

      cv::Point p4;
      p4.x = box.index({2}).item<int>();
      p4.y = box.index({3}).item<int>();

      // cv::line(mask_img, p1, p2, {0.5, 0, 0});
      // cv::line(mask_img, p1, p3, {0.5, 0, 0});
      // cv::line(mask_img, p2, p4, {0.5, 0, 0});
      // cv::line(mask_img, p3, p4, {0.5, 0, 0});

      // corr_img = corr_img  + input_masked_img;

      // CpuMat corr_img = reclib::dnn::torch2cv(corr_rgb, false);
      // cv::cvtColor(corr_img, corr_img, cv::COLOR_RGB2BGR);
      cv::imshow("mask", mask_img);
      cv::imshow("corr", corr_img);
      cv::imwrite("output_mask.png", mask_img * 255);
      cv::imwrite("output_corr.png", corr_img * 255);
      cv::waitKey(0);
    }
  }
}

// instantiation
// template class reclib::tracking::ManoRGBD<reclib::models::MANOConfig>;
// template class reclib::tracking::ManoRGBD<reclib::models::MANOConfigPCA>;
template class reclib::tracking::ManoRGBD<
    reclib::models::MANOConfigAnglePCAGeneric<45>>;
template class reclib::tracking::ManoRGBD<
    reclib::models::MANOConfigAnglePCAGeneric<23>>;

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE
#endif  // HAS_CERES_MODULE