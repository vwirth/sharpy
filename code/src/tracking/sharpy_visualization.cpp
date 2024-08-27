#include <ATen/ops/nonzero.h>
#include <c10/core/DeviceType.h>
#include <reclib/depth_processing.h>
#include <reclib/optim/model_registration.h>
#include <reclib/tracking/sharpy_utils.h>

#include "nvdiffrast/torch/torch_types.h"
#include "reclib/cuda/device_info.cuh"
#include "reclib/dnn/dnn_utils.h"
#include "reclib/dnn/nvdiffrast_autograd.h"
#include "reclib/math/eigen_glm_interface.h"
#include "reclib/tracking/mano_rgbd_optim.h"
#include "reclib/tracking/sharpy.cuh"
#include "reclib/tracking/sharpy_tracker.h"

#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <torch_tensorrt/core/compiler.h>
#include <torch_tensorrt/torch_tensorrt.h>

static std::vector<int> vertices_per_joint = {31, 20, 34, 45, 46, 28, 40,
                                              62, 43, 19, 40, 60, 39, 26,
                                              32, 64, 37, 20, 36, 66};

void reclib::tracking::HandTracker::visualize_correspondence_pointcloud(
    unsigned int index) {
  reclib::tracking::HandState& state = hand_states_.at(index);

  torch::Tensor segmentation_colors =
      reclib::tracking::sharpy::batch_cls2color(state.corr_segmented_);
  // linearize, clone and move to CPU to upload to OpenGL memory
  // cloning is important because otherwise we would move the
  // original tensor to the CPU as well
  torch::Tensor linearized_color =
      segmentation_colors
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1}), torch::All})
          .clone()
          .to(torch::kCPU)
          .contiguous();
  torch::Tensor linearized_vertices =
      state.vertex_map_
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1}), torch::All})
          .clone()
          .to(torch::kCPU)
          .contiguous();
  torch::Tensor linearized_normals =
      state.normal_map_
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1}), torch::All})
          .clone()
          .to(torch::kCPU)
          .contiguous();

  std::string name = "pc_frame_masked_" + std::to_string(index);
  if (!reclib::opengl::Mesh::valid(name)) {
    reclib::opengl::Mesh m = reclib::opengl::MeshImpl::from_geometry(
        name, false, linearized_vertices.data_ptr<float>(),
        linearized_vertices.sizes()[0], nullptr, 0,
        linearized_normals.data_ptr<float>(), linearized_normals.sizes()[0]);
    m->geometry->add_attribute_float("colors",
                                     linearized_color.data_ptr<float>(),
                                     linearized_color.sizes()[0], 3);
    m->primitive_type = GL_POINTS;
    m->geometry->update_meshes();
    reclib::opengl::Drawelement d(name, reclib::opengl::Shader::find("colorPV"),
                                  m);
    d->add_pre_draw_func("pre", [&]() { glPointSize(5.f); });
    d->add_post_draw_func("pre", [&]() { glPointSize(1.f); });

  } else {
    reclib::opengl::Mesh m = reclib::opengl::Mesh::find(name);
    m->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        linearized_vertices.data_ptr<float>(), linearized_vertices.sizes()[0],
        nullptr, 0, linearized_normals.data_ptr<float>(),
        linearized_normals.sizes()[0]);
    m->geometry->add_attribute_float(
        "colors", linearized_color.data_ptr<float>(),
        linearized_color.sizes()[0], 3, false, true);
    m->geometry->update_meshes();
    reclib::opengl::Drawelement d = reclib::opengl::Drawelement::find(name);
    d->mesh = m;
  }

  generated_drawelements_[name] = reclib::opengl::Drawelement::find(name);
}

void reclib::tracking::HandTracker::visualize_correspondences(
    unsigned int index) {
  HandState& state = hand_states_.at(index);

  // linearize, clone and move to CPU to upload to OpenGL memory
  // cloning is important because otherwise we would move the
  // original tensor to the CPU as well
  torch::Tensor corr_pos_ref =
      state.instance_->verts()
          .index({state.corrs_state_.mano_corr_indices_})
          .clone()
          .to(torch::kCPU);
  torch::Tensor vertex_map_linearized =
      state.vertex_map_
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1}), torch::All})
          .clone()
          .to(torch::kCPU)
          .contiguous();

  torch::Tensor corr_pos_src = vertex_map_linearized.index(
      {state.corrs_state_.pc_corr_indices_linearized_});
  std::vector<vec3> corr_colors(
      state.corrs_state_.mano_corr_indices_.sizes()[0] * 2);
  std::vector<uint32_t> indices;
  for (unsigned int i = 0; i < state.corrs_state_.mano_corr_indices_.sizes()[0];
       i++) {
    vec3 color = vec3::Random().normalized().cwiseAbs();

    corr_colors[i] = color;
    corr_colors[corr_pos_ref.sizes()[0] + i] = color;

    indices.push_back(i);
    indices.push_back(corr_pos_ref.sizes()[0] + i);
  }

  torch::Tensor corr_pos =
      torch::cat({corr_pos_ref, corr_pos_src}).contiguous();

  if (!reclib::opengl::Shader::valid("canonicalPointCorrespondences_colorPV")) {
    reclib::opengl::Shader("canonicalPointCorrespondences_colorPV",
                           "MVP_color3.vs", "color3PV.fs");
  }

  if (!reclib::opengl::Drawelement::valid("corr_points_" +
                                          std::to_string(index))) {
    reclib::opengl::Drawelement corr_points =
        reclib::opengl::DrawelementImpl::from_geometry(
            "corr_points_" + std::to_string(index),
            reclib::opengl::Shader::find(
                "canonicalPointCorrespondences_colorPV"),
            false, corr_pos.data<float>(), corr_pos.sizes()[0],
            (uint32_t*)indices.data(), indices.size());
    corr_points->mesh->primitive_type = GL_POINTS;
    corr_points->mesh->geometry->add_attribute_vec3("color", corr_colors, false,
                                                    true);
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
            "corr_lines_" + std::to_string(index),
            reclib::opengl::Shader::find(
                "canonicalPointCorrespondences_colorPV"),
            false, corr_pos.data<float>(), corr_pos.sizes()[0],
            (uint32_t*)indices.data(), indices.size());
    corr_lines->mesh->primitive_type = GL_LINES;
    corr_lines->mesh->geometry->add_attribute_vec3("color", corr_colors, false,
                                                   true);
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
    reclib::opengl::Drawelement corr_points = reclib::opengl::Drawelement::find(
        "corr_points_" + std::to_string(index));
    corr_points->mesh->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        corr_pos.data<float>(), corr_pos.sizes()[0], (uint32_t*)indices.data(),
        indices.size());
    corr_points->mesh->geometry->add_attribute_vec3("color", corr_colors, false,
                                                    true);
    corr_points->mesh->geometry->update_meshes();

    reclib::opengl::Drawelement corr_lines = reclib::opengl::Drawelement::find(
        "corr_lines_" + std::to_string(index));
    corr_lines->mesh->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        corr_pos.data<float>(), corr_pos.sizes()[0], (uint32_t*)indices.data(),
        indices.size());
    corr_lines->mesh->geometry->add_attribute_vec3("color", corr_colors, false,
                                                   true);
    corr_lines->mesh->geometry->update_meshes();
  }
}

void reclib::tracking::HandTracker::visualize_uncertainty() {
  if (network_output_.size() == 0) {
    std::cout << "output is empty. quit." << std::endl;
  }

  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    visualize_uncertainty(i);
  }
}

void reclib::tracking::HandTracker::visualize_uncertainty(unsigned int index) {
  HandState& state = hand_states_.at(index);

  std::vector<bool> visibility_joints = state.corrs_state_.joints_visible_;

  std::vector<vec4> per_vertex_color(state.instance_->verts().sizes()[0]);
  std::vector<float> per_joint_alpha(21);
  std::vector<float> per_joint_alpha_lines(21);

  std::vector<int> reorder_joints = {0,  1,  2,  3,  5,  6, 7, 9,  10, 11, 13,
                                     14, 15, 17, 18, 19, 4, 8, 12, 16, 20};

  torch::Device dev = state.instance_->params.device();
  {
    torch::NoGradGuard guard;

    torch::Tensor box = network_output_[2].index({(int)index}).contiguous();
    torch::Tensor mask = network_output_[3].index(
        {(int)index,
         torch::indexing::Slice(box.index({1}).item<int>(),
                                box.index({3}).item<int>()),
         torch::indexing::Slice(box.index({0}).item<int>(),
                                box.index({2}).item<int>())});
    torch::Tensor corr = network_output_[4].index({(int)index}).contiguous();
    torch::Tensor linearized_vertex_map = state.vertex_map_.index(
        {state.nonzero_indices_.index({torch::All, 0}),
         state.nonzero_indices_.index({torch::All, 1}), torch::All});
    torch::Tensor linearized_normal_map = state.normal_map_.index(
        {state.nonzero_indices_.index({torch::All, 0}),
         state.nonzero_indices_.index({torch::All, 1}), torch::All});

    // crop the hand mask and insert it into a container of the
    // original image size -> everything is black except the hand region
    // within the box coordinates
    torch::Tensor extended_mask =
        torch::zeros({intrinsics_.image_height_, intrinsics_.image_width_},
                     torch::TensorOptions().device(dev));

    extended_mask.index_put_(
        {torch::indexing::Slice(box.index({1}).item<int>(),
                                box.index({3}).item<int>()),
         torch::indexing::Slice(box.index({0}).item<int>(),
                                box.index({2}).item<int>())},
        mask);
    // Nvdiffrast is similar to OpenGL and expects images to have flipped
    // y-coordinates
    extended_mask = torch::flip(extended_mask, 0).contiguous();

    // Nvdiffrast is similar to OpenGL and expects images to have flipped
    // y-coordinates
    corr = torch::flip(corr, 1).permute({1, 2, 0}).contiguous() *
           extended_mask.unsqueeze(-1);

    // compute view projection matrix as torch tensor
    mat4 proj = reclib::vision2graphics(intrinsics_.Matrix(), 0.01f, 1000.f,
                                        intrinsics_.image_width_,
                                        intrinsics_.image_height_);
    mat4 view = reclib::lookAt(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, -1, 0));
    mat4 vp = proj * view;
    torch::Tensor VP = reclib::dnn::eigen2torch<float, 4, 4>(vp, true).to(dev);

    // compute ground-truth correspondence image for Nvdiffrast
    torch::Tensor colors = mano_corr_space_.clone().to(dev);
    colors = colors.index({torch::None, "..."});

    // initialize Nvdiffrast
    int device = reclib::getDevice();
    RasterizeCRStateWrapper context(device);

    int w = intrinsics_.image_width_;
    int h = intrinsics_.image_height_;
    std::tuple<int, int> res = std::make_tuple(h, w);

    torch::Tensor verts = state.instance_->verts();

    torch::Tensor ones =
        torch::ones({verts.sizes()[0], 1}, torch::TensorOptions().device(dev))
            .contiguous();
    torch::Tensor verts_hom = torch::cat({verts, ones}, 1).contiguous();
    torch::Tensor verts_clip =
        torch::matmul(verts_hom, VP.transpose(1, 0)).contiguous();

    torch::Tensor pos_idx =
        state.instance_->model.faces.clone().to(torch::kCUDA);

    // compute ground-truth correspondence image for Nvdiffrast
    verts_clip = verts_clip.index({torch::None, "..."}).to(torch::kCUDA);
    colors = colors.to(torch::kCUDA);

    std::vector<torch::Tensor> rast_out =
        reclib::dnn::rasterize(context, verts_clip, pos_idx, res);
    std::vector<torch::Tensor> interp_out =
        reclib::dnn::interpolate(colors, rast_out[0], pos_idx);
    std::vector<torch::Tensor> antialias_out =
        reclib::dnn::antialias(interp_out[0], rast_out[0], verts_clip, pos_idx);
    // predicted MANO colors (aka canonical coordinates) from Nvdiffrast
    torch::Tensor color_pred = antialias_out[0].squeeze(0);

    torch::Tensor point2point =
        (linearized_vertex_map.index(
             {state.corrs_state_.pc_corr_indices_linearized_}) -
         verts.index({state.corrs_state_.mano_corr_indices_}));

    // take the z component -> depth2depth error
    torch::Tensor per_point_error = point2point.abs().index({torch::All, 2});
    torch::Tensor per_pixel_error =
        (torch::l1_loss(corr, color_pred, torch::Reduction::None)).abs().sum(2);

    // compute segmentation over whole image instead of cropped region
    torch::Tensor px_segmentation =
        torch::ones({h, w}, torch::TensorOptions().device(dev)) * 30;
    px_segmentation.index_put_(
        {
            torch::indexing::Slice(
                {box.index({1}).item<int>(), box.index({3}).item<int>()}),
            torch::indexing::Slice(
                {box.index({0}).item<int>(), box.index({2}).item<int>()}),
        },
        state.corr_segmented_);

    // px_segmentation = px_segmentation.index({torch::All, torch::All, 0});
    px_segmentation = torch::flip(px_segmentation, 0).contiguous();

    std::vector<bool> joint_errorenous;
    std::vector<bool> joint_visible;

    // -------------------------------------
    // compute 2D error-prone regions
    // -------------------------------------
    for (int i = 0; i < 20; i++) {
      // compute error within a segment i
      torch::Tensor candidates = px_segmentation == i;
      torch::Tensor err = per_pixel_error.index({candidates});

      if (err.sizes()[0] == 0) {
        joint_visible.push_back(false);
        joint_errorenous.push_back(false);
        continue;
      }

      // compute number of outliers within segment i
      torch::Tensor outlier =
          err > config_["Visualization"]["pixel_thresh"].as<float>();
      // compute fraction of outliers in relation to all pixels in segment i
      float outlier_fraction = outlier.sum().item<float>() / outlier.sizes()[0];
      // thresholding to mark segmentation i as errorenous
      if (outlier_fraction >=
          config_["Visualization"]["pixel_fraction"].as<float>()) {
        joint_errorenous.push_back(true);
      } else {
        joint_errorenous.push_back(false);
      }
      joint_visible.push_back(true);
    }

    // -------------------------------------
    // compute 3D error-prone regions
    // -------------------------------------

    // compute error per 3D MANO vertex in the correspondence set
    // keep a list of the error of all correspondences in which a particular
    // vertex appears
    std::map<int, std::vector<float>> per_vertex_error;
    for (int i = 0; i < per_point_error.sizes()[0]; i++) {
      int vertex_index =
          state.corrs_state_.mano_corr_indices_.index({i}).item<int>();
      int joint_index = mano_verts2joints_[vertex_index];
      float error = per_point_error.index({i}).item<float>();

      if (per_vertex_error.find(vertex_index) == per_vertex_error.end()) {
        // vertex is not yet present -> construct a new vector and add per-point
        // error to that list
        per_vertex_error[vertex_index] = std::vector<float>();
        per_vertex_error[vertex_index].push_back(error);
      } else {
        // vertex is already present in the list -> add per-point error to that
        // list
        per_vertex_error[vertex_index].push_back(error);
      }
    }

    torch::Tensor mano_segmentation =
        reclib::tracking::sharpy::batch_corr2seg(mano_corr_space_);

    // map the per-vertex errors to per-joint errors
    // keep a list of the error of all per-vertex errors that belong to a
    // particular joint
    std::map<int, std::vector<float>> per_joint_error;
    for (auto it : per_vertex_error) {
      int i = it.first;
      int joint_index = mano_verts2joints_[i];
      int seg_index = mano_segmentation.index({i}).item<int>();
      joint_index = seg_index;

      if (per_vertex_error.find(i) != per_vertex_error.end()) {
        // sum up the total error per vertex
        float sum = 0;
        for (unsigned int j = 0; j < per_vertex_error[i].size(); j++) {
          sum += per_vertex_error[i][j];
        }

        if (per_joint_error.find(joint_index) == per_joint_error.end()) {
          // joint not present in list yet -> add first per-vertex error to
          // joint index
          per_joint_error[joint_index] = std::vector<float>();
          per_joint_error[joint_index].push_back(sum /
                                                 per_vertex_error[i].size());
        } else {
          // joint is already present in the list -> add per-vertex error to
          // that list
          per_joint_error[joint_index].push_back(sum /
                                                 per_vertex_error[i].size());
        }
      }
    }

    // compute the final 3D error to determine errorenous joints
    std::vector<bool> joint_errorenous_depth(vertices_per_joint.size());
    for (int i = 0; i < vertices_per_joint.size(); i++) {
      int joint_index = i;

      if (per_joint_error.find(joint_index) == per_joint_error.end()) {
        joint_visible[joint_index] = false;
        joint_errorenous_depth[joint_index] = false;
        continue;
      } else {
        std::vector<float> pj = per_joint_error[joint_index];
        // compute fraction of visible vertices for a particular joint
        float visible_frac = pj.size() / (float)vertices_per_joint[joint_index];

        // visibility thresholding
        if (visible_frac <
            config_["Visualization"]["visible_fraction"].as<float>()) {
          joint_visible[joint_index] = false;
          joint_errorenous_depth[joint_index] = false;

          continue;
        }

        // compute number of outlier vertices that belong to joint
        int num_errorenous = 0;
        for (unsigned int j = 0; j < pj.size(); j++) {
          if (pj[j] > config_["Visualization"]["depth_thresh"].as<float>()) {
            num_errorenous++;
          }
        }

        // compute fraction of errorenous vertices
        float errorenous_fraction = num_errorenous / (float)pj.size();
        // errorenous thresholding
        if (errorenous_fraction >
            config_["Visualization"]["depth_fraction"].as<float>()) {
          joint_errorenous_depth[joint_index] = true;
          joint_visible[joint_index] = true;
        }
      }
    }

    // whether to use separate colors for 2D errorenous and 3D errorenous joints
    bool separate =
        config_["Visualization"]["separate_px_and_depth"].as<bool>();

    for (int i = 0; i < state.instance_->verts().sizes()[0]; i++) {
      int joint_index = mano_verts2joints_[i];
      float weight = state.instance_->model.weights.index({i, joint_index})
                         .template item<float>();

      // compute segmentation label for mano vertex i
      int joint_index_seg = mano_segmentation.index({i}).item<int>();

      // compute a color for the current vertex
      vec4 c(SKIN_COLOR.x(), SKIN_COLOR.y(), SKIN_COLOR.z(), 1);
      float alpha = 1;

      // vertex does not belong to the wrist
      if (!(joint_index_seg % 4 == 0) && !(joint_index == 0)) {
        // use the segmentation label that can be directly inferred from the
        // joint that most influences the deformation of the current vertex
        joint_index_seg = JOINT2VIS_INDICES[joint_index - 1];
      }
      if (!joint_visible[joint_index_seg]) {
        // joint is not visible -> color as 'unobserved'
        alpha = 0.75;
        c.w() = alpha;
        c.x() = 0.3;
        c.y() = 0.3;
        c.z() = 0.3;

        per_vertex_color[i] = c;
        continue;
      }
      if (!separate && (joint_errorenous[joint_index_seg] ||
                        joint_errorenous_depth[joint_index_seg])) {
        // joint is errorenous -> color accordingly
        c.x() = fmax(0.55 + weight * 0.5, 0.55);
        c.y() -= 0.1;
        c.z() -= 0.25;
      }

      if (separate && joint_errorenous[joint_index_seg]) {
        // separate color for 2D errorenous joint
        c.x() = fmax(0.5 + weight * 0.5, 0.5);
        c.y() = 0;
        c.z() = 0;
      }
      if (separate && joint_errorenous_depth[joint_index_seg]) {
        // separate color for 3D errorenous joint
        c.z() = fmax(0.5 + weight * 0.5, 0.5);
        c.x() = 0;
        c.y() = 0;
      }
      if (separate && (joint_errorenous[joint_index_seg] &&
                       joint_errorenous_depth[joint_index_seg])) {
        // separate color for both, 2D and 3D errorenous joint
        c.y() = fmax(0.5 + weight * 0.5, 0.5);
        c.x() = 0;
        c.z() = 0;
      }

      //  clamping
      c.x() = fmax(fmin(c.x(), 1), 0);
      c.y() = fmax(fmin(c.y(), 1), 0);
      c.z() = fmax(fmin(c.z(), 1), 0);
      c.w() = alpha;

      per_vertex_color[i] = c;
    }

    std::fill(per_joint_alpha.begin(), per_joint_alpha.end(), 1);
    std::fill(per_joint_alpha_lines.begin(), per_joint_alpha_lines.end(), 1);
    per_joint_alpha_lines[0] = 0.8;
    std::vector<float> tip_indices = {16, 17, 18, 19, 20};

    // beautifying of visualization
    // the joint lines and joint spheres that are used to visualize
    // the skeleton of the hand should be shaded accordingly to
    // the previously determined unobserved and error-prone
    // joints
    for (int i = 0; i < 16; i++) {
      int joint_index = i;
      float alpha = 1;

      int target_index = 0;
      int visibility_joint = 0;
      if (i == 0) {
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
        // if previous joint in the kinematic chain is
        // not visible, then current joint should also be
        // visualized as not visible
        visibility_joint = JOINT2VIS_INDICES[i - 1];
        if (!joint_visible[visibility_joint] ||
            joint_errorenous[visibility_joint] ||
            joint_errorenous_depth[visibility_joint]) {
          alpha = 0.01;
        }
        target_index = i + 1;
        if ((i - 1) % 3 == 2) {
          target_index = tip_indices[(i - 1) / 3];  // ((i - 1) / 3) + 16;
        }
      }

      per_joint_alpha[target_index] = alpha;
      per_joint_alpha_lines[target_index] = alpha;
      if (i > 0) {
        per_joint_alpha_lines[i] = fmin(per_joint_alpha_lines[i], alpha);
      }

      if ((i == 1 || i == 4 || i == 7 || i == 10 || i == 13) && alpha < 1) {
        per_joint_alpha[i] = fmin(per_joint_alpha[i], alpha);
      }
    }

    // update hand state
    state.joints_visible_ = joint_visible;
    state.joints_errorenous_.clear();
    for (unsigned int i = 0; i < 20; i++) {
      state.joints_errorenous_.push_back(joint_errorenous[i] ||
                                         joint_errorenous_depth[i]);
    }
  }

  // assign color and alpha attributes to OpenGL mesh
  state.instance_->gl_instance->mesh->geometry->vec4_map.erase("color");
  state.instance_->gl_instance->mesh->geometry->add_attribute_vec4(
      "color", per_vertex_color.data(), per_vertex_color.size(), false);
  state.instance_->gl_instance->mesh->geometry->update_meshes();

  state.instance_->gl_joint_lines->mesh->geometry->float_map.erase("alpha");
  state.instance_->gl_joint_lines->mesh->geometry->add_attribute_float(
      "alpha", per_joint_alpha_lines, 1);
  state.instance_->gl_joint_lines->mesh->geometry->update_meshes();

  state.instance_->gl_joints->mesh->geometry->float_map.erase("alpha");
  state.instance_->gl_joints->mesh->geometry->add_attribute_float(
      "alpha", per_joint_alpha, 1);
  state.instance_->gl_joints->mesh->geometry->update_meshes();
}

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE