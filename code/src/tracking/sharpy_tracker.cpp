#include <ATen/ops/tensor.h>
#include <opencv2/core/hal/interface.h>
#include <reclib/depth_processing.h>
#include <reclib/dnn/nvdiffrast_autograd.h>
#include <reclib/internal/filesystem_ops.h>
#include <reclib/opengl/rgbd_utils.h>
#include <reclib/optim/model_registration.h>
#include <reclib/tracking/mano_rgbd_optim.h>
#include <reclib/tracking/sharpy_tracker.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

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

#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

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

extern const vec3 reclib::tracking::SKIN_COLOR =
    (vec3(230, 184, 255) / 255) * 0.7;
extern const std::vector<int> reclib::tracking::JOINT2VIS_INDICES = {
    5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3};

static const std::vector<int> APOSE2VIS_INDICES = {
    5,  5,  6,  7,  9, 9, 10, 11, 17, 17, 18, 19,
    13, 13, 14, 15, 1, 1, 1,  2,  2,  2,  3,
};

static std::vector<int> VIS2POSE_INDICES = {0, 13, 14, 15, 0,  1,  2, 3, 0, 4,
                                            5, 6,  0,  10, 11, 12, 0, 7, 8, 9};

// -----------------------------------------------------------------------
// HandState
// -----------------------------------------------------------------------

bool reclib::tracking::HandState::global_initialized_ = false;
std::vector<vec4> reclib::tracking::HandState::color_alphas_mano_;
std::vector<vec3> reclib::tracking::HandState::color_mano_joints_;
std::vector<float> reclib::tracking::HandState::alpha_mano_joints_;

void reclib::tracking::HandState::initialize_globals() {
  torch::NoGradGuard guard;

  if (reclib::tracking::HandState::global_initialized_) return;

  for (unsigned int i = 0; i < 778; i++) {
    color_alphas_mano_.push_back(
        vec4(SKIN_COLOR.x(), SKIN_COLOR.y(), SKIN_COLOR.z(), 1));
  }

  for (unsigned int i = 0; i < 21; i++) {
    alpha_mano_joints_.push_back(1);
  }

  // initialize color per joint
  // root line
  color_mano_joints_.push_back(vec3(1, 1, 1));
  // index
  color_mano_joints_.push_back(vec3(0, 1, 0));
  color_mano_joints_.push_back(vec3(0, 1, 0));
  color_mano_joints_.push_back(vec3(0, 1, 0));
  // middle
  color_mano_joints_.push_back(vec3(1, 0.5, 0));
  color_mano_joints_.push_back(vec3(1, 0.5, 0));
  color_mano_joints_.push_back(vec3(1, 0.5, 0));
  // pinky
  color_mano_joints_.push_back(vec3(0, 0.5, 1));
  color_mano_joints_.push_back(vec3(0, 0.5, 1));
  color_mano_joints_.push_back(vec3(0, 0.5, 1));
  // ring
  color_mano_joints_.push_back(vec3(0.9, 0, 1));
  color_mano_joints_.push_back(vec3(0.9, 0, 1));
  color_mano_joints_.push_back(vec3(0.9, 0, 1));
  // thumb
  color_mano_joints_.push_back(vec3(1, 0, 0));
  color_mano_joints_.push_back(vec3(1, 0, 0));
  color_mano_joints_.push_back(vec3(1, 0, 0));
  // last joint to finger tip of each finger
  color_mano_joints_.push_back(vec3(0, 1, 0));
  color_mano_joints_.push_back(vec3(1, 0.5, 0));
  color_mano_joints_.push_back(vec3(0, 0.5, 1));
  color_mano_joints_.push_back(vec3(0.9, 0, 1));
  color_mano_joints_.push_back(vec3(1, 0, 0));

  reclib::tracking::HandState::global_initialized_ = true;
}

reclib::tracking::HandState::HandState(
    reclib::Configuration& config,
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<23>>& model)
    : config_(config),
      trans_(torch::zeros(3)),
      rot_(torch::zeros(3)),
      shape_(torch::zeros(10)),
      instance_(std::make_shared<reclib::modelstorch::ModelInstance<
                    reclib::models::MANOConfigAnglePCAGeneric<23>>>(model)),
      loss_(-1),
      mean_depth_(-1),
      failed_(false),
      stage_(0),
      visibility_map_(20),
      joints_visible_(20),
      joints_errorenous_(20) {
  initialize_globals();
  torch::NoGradGuard guard;

  if (config["MANO"]["use_calculated_pca"].as<bool>()) {
    pca_ = torch::zeros(23);
  } else {
    pca_ = torch::zeros(45);
  }
  if (config["MANO"]["use_anatomic_pose"].as<bool>()) {
    pose_ = torch::zeros(23);
  } else {
    pose_ = torch::zeros(45);
  }

  // initialize MANO model
  instance_->set_zero();
  instance_->add_pose_mean_ = false;
  instance_->use_anatomic_pose_ =
      config["MANO"]["use_anatomic_pose"].as<bool>();
  instance_->use_anatomic_pca_ =
      config["MANO"]["use_calculated_pca"].as<bool>();
  instance_->gpu();
  bool update_meshes = !config["Optimization"]["multithreading"].as<bool>();
  instance_->update(false, true, false, update_meshes);

  // OpenGL stuff
  if (config["Debug"]["show_hands"].as<bool>()) {
    // initialize openGL stuff to render the MANO hand
    if (!reclib::opengl::Shader::valid("mano_shader")) {
      // reclib::opengl::Shader("mano_shader", "MVP_norm_color3.vs",
      // "color3PV.fs");
      reclib::opengl::Shader("mano_shader", "MVP_norm_color4.vs",
                             "pointlight3_color4PV.fs");
    }
    if (!reclib::opengl::Material::valid("mano_material")) {
      reclib::opengl::Material m("mano_material");
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
      joint_lines_material->vec3_map["color"] = vec3(0.2, 0.2, 0.2);
    }
    if (!reclib::opengl::Material::valid("mano_joint_lines")) {
      reclib::opengl::Material joint_lines_material("mano_joint_lines");
      joint_lines_material->vec3_map["color"] = vec3(0.0f, 0.2f, 0.8f);
    }

    instance_->generate_gl_drawelem(
        reclib::opengl::Shader::find("mano_shader"),
        reclib::opengl::Material::find("mano_material"));

    // MANO mesh
    instance_->gl_instance->mesh->geometry->float_map.erase("color");
    instance_->gl_instance->mesh->geometry->add_attribute_vec4(
        "color", color_alphas_mano_.data(), color_alphas_mano_.size(), false);
    instance_->gl_instance->wireframe_mode = false;
    instance_->gl_instance->add_pre_draw_func("pre",
                                              [&]() { glLineWidth(5.f); });
    instance_->gl_instance->add_post_draw_func("post",
                                               [&]() { glLineWidth(1.f); });

    // MANO joints (spheres)
    instance_->gl_joints->shader = reclib::opengl::Shader::find("mano_joints");
    instance_->gl_joints->mesh->geometry->add_attribute_float(
        "alpha", alpha_mano_joints_, 1);
    instance_->gl_joints->mesh->material =
        reclib::opengl::Material::find("mano_joints");

    // MANO joint lines connecting the joints
    instance_->gl_joint_lines->shader =
        reclib::opengl::Shader::find("mano_joint_lines");
    instance_->gl_joint_lines->mesh->geometry->add_attribute_vec3(
        "color", color_mano_joints_);
    instance_->gl_joint_lines->mesh->geometry->add_attribute_float(
        "alpha", alpha_mano_joints_, 1);

    if (config["Debug"]["highlight_joints"].as<bool>()) {
      instance_->gl_joints->add_pre_draw_func(
          "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
      instance_->gl_joints->add_post_draw_func(
          "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
      instance_->gl_joint_lines->add_pre_draw_func(
          "depthtest", [&]() { glDisable(GL_DEPTH_TEST); });
      instance_->gl_joint_lines->add_post_draw_func(
          "depthtest", [&]() { glEnable(GL_DEPTH_TEST); });
    }
  }

  // initialize PCA values with the mean value
  instance_->set_hand_pca(torch::matmul(instance_->model.hand_comps.inverse(),
                                        instance_->model.hand_mean)
                              .squeeze(1));

  instance_->update(false, true, false, update_meshes);
}

void reclib::tracking::HandState::prepare_new_frame() {
  torch::NoGradGuard guard;

  // reset correspondences from last frame
  corrs_state_.reset();

  // store MANO Parameters in tensors
  torch::Tensor pose =
      torch::matmul(instance_->model.hand_comps, instance_->hand_pca());

  for (int i = 0; i < 45; i++) {
    if (instance_->use_anatomic_pose_) {
      // anatomic case, only 23 pose parameters
      if (i == 23) break;
      if (visibility_map_[APOSE2VIS_INDICES[i]] > 0)
        pose_.index_put_({i}, pose.index({i}));
    } else {
      // non-anatomic case, 45 pose parameters
      int joint_index = i / 3;
      if (visibility_map_[JOINT2VIS_INDICES[joint_index]])
        pose_.index_put_({i}, pose.index({i}));
    }
  }

  trans_ = instance_->trans();
  shape_ = instance_->shape();
  rot_ = instance_->rot();
  pca_ = instance_->hand_pca();

  // set to uninitialized state
  corr_ = torch::Tensor();
  corr_segmented_ = torch::Tensor();
  vertex_map_ = torch::Tensor();
  normal_map_ = torch::Tensor();
  masked_depth_ = torch::Tensor();
  nonzero_indices_ = torch::Tensor();

  failed_ = false;
  // registration + optimization stage 1 is only done for the first
  // frame, or in case of tracking failures
  stage_ = 2;
}

void reclib::tracking::HandState::reset() {
  torch::NoGradGuard guard;

  // reset correspondences from last frame
  corrs_state_.reset();

  trans_ = torch::zeros(3);
  rot_ = torch::zeros(3);
  shape_ = torch::zeros(10);
  if (config_["MANO"]["use_calculated_pca"].as<bool>()) {
    pca_ = torch::zeros(23);
  } else {
    pca_ = torch::zeros(45);
  }
  if (config_["MANO"]["use_anatomic_pose"].as<bool>()) {
    pose_ = torch::zeros(23);
  } else {
    pose_ = torch::zeros(45);
  }

  loss_ = -1;
  mean_depth_ = -1;
  failed_ = false;
  stage_ = 0;

  // initialize MANO model
  instance_->set_zero();
  bool update_meshes = !config_["Optimization"]["multithreading"].as<bool>();
  instance_->update(false, true, false, update_meshes);

  std::fill(visibility_map_.begin(), visibility_map_.end(), 0);
  std::fill(joints_visible_.begin(), joints_visible_.end(), true);
  std::fill(joints_errorenous_.begin(), joints_errorenous_.end(), false);
}

// -----------------------------------------------------------------------
// HandTracker
// -----------------------------------------------------------------------

reclib::tracking::HandTracker::HandTracker(const reclib::Configuration& config)
    : config_(config),

      network_(config_["Network"]["weights"].as<std::string>(), true, true,
               true, false),
      transform_(config_["Network"]["max_size"].as<unsigned int>(),
                 config_["Network"]["preserve_aspect_ratio"].as<bool>(), "BGR"),
      mano_model_left_(reclib::models::HandType::left),
      mano_model_right_(reclib::models::HandType::right),
      debug_(config_["Debug"]["general"].as<bool>()),
      sequence_frame_counter_(0),
      global_frame_counter_(0)

{
  torch::NoGradGuard guard;

  if (config_["OpenPose"]["use"].as<bool>()) {
    openpose_ =
        cv::dnn::readNet(config_["OpenPose"]["weights"].as<std::string>(),
                         config_["OpenPose"]["protofile"].as<std::string>());
  }

  if (config_["MANO"]["use_calculated_pca"].as<bool>()) {
    // load PCA parameter file
    std::ifstream f;
    f.open(config_["Dataset"]["pca_file"].as<std::string>());
    nlohmann::json pca_params = nlohmann::json::parse(f);

    // initialize mano models with precomputed PCA parameters
    mano_model_left_.hand_mean =
        torch::from_blob(
            pca_params["left"]["mean"].get<std::vector<float>>().data(),
            {23, 1})
            .clone();
    mano_model_left_.hand_comps =
        torch::from_blob(
            pca_params["left"]["ev"].get<std::vector<float>>().data(), {23, 23})
            .clone();
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
    f.close();
  }
  mano_model_left_.gpu();
  mano_model_right_.gpu();

  // initialize MANO-specific variables
  {
    reclib::models::Model<reclib::models::MANOConfig> mano_model_tmp(
        reclib::models::HandType::right);
    reclib::models::ModelInstance<reclib::models::MANOConfig> mano_tmp(
        mano_model_tmp);
    std::vector<vec4> tmp =
        reclib::models::mano_canonical_space_colors_hsv(mano_tmp, false, false);
    torch::Tensor corr_space =
        torch::from_blob(tmp[0].data(), {(int)tmp.size(), 4});

    corr_space =
        corr_space.index({torch::All, torch::indexing::Slice(0, 3)}).clone();

    mano_corr_space_ = corr_space;
    mano_corr_space_.index({torch::All, 0}) =
        mano_corr_space_.index({torch::All, 0}) / 360.0;

    // a vector that assigns each vertex to its most influencing hand joint
    mano_verts2joints_ = mano_model_tmp.verts2joints;
  }

  // initialize OpenGL variables
}

void reclib::tracking::HandTracker::reset() {
  torch::NoGradGuard guard;
  sequence_frame_counter_ = 0;
  // hand_states_.clear();
  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    // if (hand_states_[i].failed_) {
    hand_states_[i].reset();
    //}
  }
}

void reclib::tracking::HandTracker::reset_failed() {
  torch::NoGradGuard guard;
  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    if (hand_states_[i].failed_) {
      hand_states_[i].reset();
    }
  }
}

void reclib::tracking::HandTracker::retry_failed() {
  torch::NoGradGuard guard;
  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    if (hand_states_[i].failed_) {
      int stage = hand_states_[i].stage_;
      std::cout << "hand " << i << " failed at stage: " << stage << std::endl;
      hand_states_[i].reset();
      if (stage >= 0) {
        compute_correspondences(
            reclib::tracking::CorrespondenceMode::REGISTRATION);
        register_hand(i);
      }
      if (stage >= 1) {
        compute_correspondences(
            reclib::tracking::CorrespondenceMode::OPTIMIZATION);
        optimize_hand(i, 0);
        optimize_hand(i, 1);
      }
    }
  }
}

void reclib::tracking::HandTracker::restore_failed() {
  torch::NoGradGuard guard;
  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    if (hand_states_[i].failed_) {
      HandState& state = hand_states_.at(i);
      state.instance_->set_trans(state.trans_);
      state.instance_->set_rot(state.rot_);
      state.instance_->set_hand_pca(state.pca_);
      state.instance_->set_shape(state.shape_);
      state.failed_ = false;
      state.stage_ = 0;
    }
  }
}

void reclib::tracking::HandTracker::process_image(
    const CpuMat& rgb, const CpuMat& depth,
    const reclib::IntrinsicParameters& intrinsics, cv::Rect crop_factor) {
  torch::NoGradGuard guard;

  // prepare tracker state
  rgb_ = rgb.clone();
  rgb_.convertTo(rgb_, CV_32FC3);
  // rgb_ = rgb_ / 255;
  depth_ = depth.clone();
  depth_.convertTo(depth_, CV_32FC1);

  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    hand_states_[i].prepare_new_frame();
  }

  sequence_frame_counter_++;
  global_frame_counter_++;

  intrinsics_ = intrinsics;

  if (config_["Debug"]["show_rgbd"].as<bool>() == true) {
    CpuMat rgb_float;
    cv::cvtColor(rgb, rgb_float, cv::COLOR_BGR2RGB);
    rgb_float.convertTo(rgb_float, CV_32FC3);
    rgb_float = rgb_float / 255;

    if (!reclib::opengl::Shader::valid("colorPV")) {
      reclib::opengl::Shader("colorPV", "MVP_norm_color3.vs", "color3PV.fs");
    }

    std::string name = "pc_frame";
    reclib::opengl::Mesh m = reclib::opengl::pointcloud_norm_color(
        depth_, intrinsics_, rgb_float, true, name, CpuMat(0, 0, CV_32FC1),
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

  // reuse the predicted bounidng box from last frame to crop
  // the image of current frame
  if (config_["Input"]["crop_to_bb"].as<bool>() &&
      sequence_frame_counter_ > 1) {
    crop_factor = compute_crop_factor(rgb_.cols, rgb_.rows);
  }
  network_output_.clear();

  if (config_["Debug"]["show_input"].as<bool>() == true) {
    cv::imshow("Input: ", rgb);
  }

  if (debug_) {
    timer_.begin();
  }

  detect(crop_factor);

  // run the first frame through the network two times
  // use the predicted bounding box of first attempt in second
  // pass
  if (sequence_frame_counter_ <= 1 &&
      config_["Input"]["crop_to_bb"].as<bool>()) {
    cv::Rect crop_factor_new = compute_crop_factor(rgb.cols, rgb.rows);
    detect(crop_factor_new);
  }

  if (config_["OpenPose"]["use"].as<bool>()) {
    for (int i = 0; i < network_output_[2].sizes()[0]; i++) {
      cv::Rect crop_factor_openpose;
      crop_factor_openpose.x =
          std::fmax(0, network_output_[2].index({i, 0}).item<float>() -
                           config_["OpenPose"]["bb_pad_x"].as<float>());
      crop_factor_openpose.y =
          std::fmax(0, network_output_[2].index({i, 1}).item<float>() -
                           config_["OpenPose"]["bb_pad_y"].as<float>());
      crop_factor_openpose.width =
          std::fmin(rgb.cols - 1,
                    network_output_[2].index({i, 2}).item<float>() +
                        config_["Input"]["bb_pad_x"].as<float>()) -
          crop_factor_openpose.x;
      crop_factor_openpose.height =
          std::fmin(rgb.rows - 1,
                    network_output_[2].index({i, 3}).item<float>() +
                        config_["Input"]["bb_pad_y"].as<float>()) -
          crop_factor_openpose.y;

      // Beware: we intentionally give rgb and not rgb_ as an argument
      // since openpose needs the uint8_t datatype per channel
      detect_openpose(i, rgb, crop_factor_openpose);
    }
  }

  if (config_["Debug"]["show_output"].as<bool>() == true) {
    torch::Tensor input_float = reclib::dnn::cv2torch(rgb_);

    int num_samples = network_output_[0].sizes()[0];
    for (int i = 0; i < num_samples; i++) {
      torch::Tensor box =
          network_output_[2].index({i}).clone().to(torch::kCPU).contiguous();
      torch::Tensor mask =
          network_output_[3].index({i}).clone().to(torch::kCPU).contiguous();
      // mask.index({torch::indexing::Slice(box.index({1}).item<int>(),
      //                                    box.index({3}).item<int>()),
      //             torch::indexing::Slice(box.index({0}).item<int>(),
      //                                    box.index({2}).item<int>())}) =
      //                                    0.5;
      torch::Tensor corr_masked =
          network_output_[4].index({i}).clone().to(torch::kCPU).contiguous() *
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

      CpuMat mask_img = reclib::dnn::torch2cv(mask);
      CpuMat corr_img = reclib::dnn::torch2cv(corr_hsv);
      CpuMat input_masked_img = reclib::dnn::torch2cv(input_masked);
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

      cv::line(mask_img, p1, p2, {0.5, 0, 0});
      cv::line(mask_img, p1, p3, {0.5, 0, 0});
      cv::line(mask_img, p2, p4, {0.5, 0, 0});
      cv::line(mask_img, p3, p4, {0.5, 0, 0});

      // corr_img = corr_img  + input_masked_img;

      // CpuMat corr_img = reclib::dnn::torch2cv(corr_rgb, false);
      // cv::cvtColor(corr_img, corr_img, cv::COLOR_RGB2BGR);
      cv::imshow("mask", mask_img);
      cv::imshow("corr", corr_img);
      // cv::imwrite("output_mask.png", mask_img * 255);
      // cv::imwrite("output_corr.png", corr_img * 255);
      cv::waitKey(0);
    }
  }

  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    // prepare correspondence data types for succeeding stages
    prepare_correspondences(i);
  }
}

void reclib::tracking::HandTracker::process(
    const CpuMat& rgb, const CpuMat& depth,
    const reclib::IntrinsicParameters& intrinsics, cv::Rect crop_factor) {
  torch::NoGradGuard guard;

  process_image(rgb, depth, intrinsics, crop_factor);

  compute_all_correspondences();
}

void reclib::tracking::HandTracker::compute_all_correspondences(
    int index,
    at::Stream stream = at::Stream(at::Stream::DEFAULT,
                                   at::Device(at::DeviceType::CUDA, 0))) {
  at::cuda::CUDAStreamGuard guard(stream);
  // auto afterstream = at::cuda::getCurrentCUDAStream();
  // std::cout << "after: " << afterstream.id() << std::endl;

  int num_tries = 2;
  int try_counter = 0;
  do {
    compute_correspondences(reclib::tracking::CorrespondenceMode::REGISTRATION,
                            index);
    std::optional<bool> res = register_hand(index);
    if (res.has_value()) {
      hand_states_[index].failed_ = res.value();
      if (!res.value()) {
        break;  // hand registration has failed, there's nothing that
        // can be fixed here...
      }
    }
    compute_correspondences(reclib::tracking::CorrespondenceMode::OPTIMIZATION,
                            index);
    res = optimize_hand(index, 0);
    if (res.has_value()) {
      hand_states_[index].failed_ = res.value();
      if (!res.value()) {
        // try resetting the hand and retry
        hand_states_[index].reset();
        try_counter++;
        continue;
      }
    }
    res = optimize_hand(index, 1);
    if (res.has_value()) {
      hand_states_[index].failed_ = res.value();
      if (!res.value()) {
        // try resetting the hand and retry
        hand_states_[index].reset();
        try_counter++;
        continue;
      }
    }
    // everything worked fine, break out of loop
    try_counter = num_tries;
  } while (try_counter < num_tries);
}

void reclib::tracking::HandTracker::compute_all_correspondences() {
  bool multithreading = config_["Optimization"]["multithreading"].as<bool>();

  std::vector<at::Stream> streams;
  if (multithreading) {
    // create two streams for the two hands.
    // TODO: move stream creation out of here. Can be done once in the
    // beginning. Maybe get CUDA_API_PER_THREAD_DEFAULT_STREAM to work

    // auto beforestream = at::cuda::getCurrentCUDAStream();
    // std::cout << "before: " << beforestream.id() << std::endl;

    at::DeviceIndex current_device_index = at::cuda::current_device();
    at::Device current_device(at::DeviceType::CUDA, current_device_index);
    for (int i = 0; i < hand_states_.size(); i++) {
      cudaStream_t raw_stream;
      cudaStreamCreate(&raw_stream);
      streams.emplace_back(at::Stream::UNSAFE, current_device,
                           (int64_t)raw_stream);
    }
  }

  if (debug_) {
    // for benchmarks, disable the Debug::general flag in sharpy.yaml and pull
    // these lines out of the if
    reclib::opengl::Timer timer_;
    timer_.look_and_reset();
  }

  std::vector<std::future<void>> futures;
  for (int i = 0; i < hand_states_.size(); i++) {
    if (!multithreading) {
      compute_all_correspondences(i);
      continue;
    }

    // explicitly define the type of the function we call with async
    // so the compiler knows which of the overloaded functions to call.
    using single_hand_type = void (HandTracker::*)(int, at::Stream);
    single_hand_type compute_hand = &HandTracker::compute_all_correspondences;
    std::future<void> future =
        std::async(std::launch::async, compute_hand, this, i, streams[i]);
    futures.push_back(std::move(future));
  }

  if (multithreading) {
    for (std::future<void>& future : futures) {
      future.wait();
    }

    for (int i = 0; i < hand_states_.size(); i++) {
      visualize_correspondences(i);
      visualize_correspondence_pointcloud(i);
    }

    for (at::Stream stream : streams) {
      cudaStreamDestroy((cudaStream_t)(stream.id()));
    }
  }

  if (debug_) {
    std::cout << "All hand correspondences: " << timer_.look_and_reset()
              << " ms." << std::endl;
  }
}

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE