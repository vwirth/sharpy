#include <ATen/ops/nonzero.h>
#include <c10/core/DeviceType.h>
#include <reclib/depth_processing.h>
#include <reclib/optim/model_registration.h>
#include <reclib/tracking/sharpy_utils.h>

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

static std::vector<int> apose2joint = {1,  1,  2,  3,  4,  4,  5,  6,
                                       7,  7,  8,  9,  10, 10, 11, 12,
                                       13, 13, 13, 14, 14, 14, 15};

std::vector<bool> reclib::tracking::HandTracker::register_hand() {
  std::vector<bool> results(hand_states_.size());
  std::fill(results.begin(), results.end(), false);
  if (network_output_.size() == 0) {
    std::cout << "output is empty. quit." << std::endl;
    return results;
  }

  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    std::optional<bool> res = register_hand(i);
    if (res.has_value()) {
      results[i] = res.value();
      hand_states_[i].failed_ = !results[i];
    }
  }

  return results;
}

std::optional<bool> reclib::tracking::HandTracker::register_hand(
    unsigned int index) {
  torch::NoGradGuard guard;
  reclib::Configuration reg_config = config_.subconfig({"Registration"});
  bool update_meshes = !config_["Optimization"]["multithreading"].as<bool>();

  if (debug_) {
    timer_.look_and_reset();
  }

  HandState& state = hand_states_.at(index);

  if (state.corrs_state_.mano_corr_indices_.sizes()[0] == 0) {
    std::cout << "No correspondences. Abort." << std::endl;
    return false;
  }
  if (state.stage_ >= 1) {
    std::cout << "Hand is already registered." << std::endl;
    return {};
  }

  if (debug_) {
    timer_.look_and_reset();
  }

  torch::Tensor linearized_pc = state.vertex_map_.index(
      {state.nonzero_indices_.index({torch::All, 0}),
       state.nonzero_indices_.index({torch::All, 1}), torch::All});
  torch::Tensor pc_corrs =
      linearized_pc.index({state.corrs_state_.pc_corr_indices_linearized_});
  torch::Tensor mano_corrs =
      state.instance_->verts().index({state.corrs_state_.mano_corr_indices_});

  torch::Tensor incr_trans = torch::eye(4);

  for (unsigned int i = 0; i < reg_config.ui("iterations"); i++) {
    if (reg_config.b("register_rotation")) {
      torch::Tensor rigid_trans =
          reclib::optim::pointToPointDirect(mano_corrs, pc_corrs);
      incr_trans = torch::matmul(rigid_trans, incr_trans);

      reclib::dnn::TorchVector aa_tensor = reclib::tracking::batch_mat2aa(
          incr_trans
              .index(
                  {torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)})
              .unsqueeze(0));

      state.instance_->set_trans(
          incr_trans.index({torch::indexing::Slice(0, 3), 3}));
      state.instance_->set_rot(aa_tensor.tensor());
      state.instance_->update(false, true, false, update_meshes);
    }

    // update tensor with updated vertices
    mano_corrs =
        state.instance_->verts().index({state.corrs_state_.mano_corr_indices_});

    if (reg_config.b("register_translation")) {
      torch::Tensor trans =
          reclib::optim::MeanTranslation(mano_corrs, pc_corrs);
      torch::Tensor T = torch::eye(4);
      T.index_put_({torch::indexing::Slice(0, 3), 3}, trans);
      incr_trans = torch::matmul(T, incr_trans);
      state.instance_->set_trans(
          incr_trans.index({torch::indexing::Slice(0, 3), 3}));
      state.instance_->update(false, true, false, update_meshes);
    }
  }

  if (debug_) {
    std::cout << "---- Registration: " << timer_.look_and_reset() << " ms."
              << std::endl;
  }

  if (config_["Debug"]["show_corr_lines"].as<bool>() &&
      config_["Optimization"]["multithreading"].as<bool>() == false) {
    visualize_correspondences(index);
  }
  state.stage_ = 1;
  return true;
}

std::vector<bool> reclib::tracking::HandTracker::optimize_hand(int stage) {
  std::vector<bool> results(hand_states_.size());
  std::fill(results.begin(), results.end(), false);
  if (network_output_.size() == 0) {
    std::cout << "output is empty. quit." << std::endl;
    return results;
  }

  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    std::optional<bool> res = optimize_hand(i, stage);
    if (res.has_value()) {
      results[i] = res.value();
      hand_states_[i].failed_ = !results[i];
    }
  }

  return results;
}

std::optional<bool> reclib::tracking::HandTracker::optimize_hand(
    unsigned int index, int stage) {
  reclib::Configuration opt_config = config_.subconfig({"Optimization"});
  bool update_meshes = !config_["Optimization"]["multithreading"].as<bool>();

  HandState& state = hand_states_.at(index);

  // stage 0 is only executed once, directly after registration
  if (stage == 0 || !config_["Optimization"]["separate_stages"].as<bool>()) {
    if (state.stage_ >= 2) {
      return {};
    }
  }

  std::vector<bool> visibility_joints = state.corrs_state_.joints_visible_;

  // number of visible joint segments
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

  // if a child within the finger segment is visible,
  // then all its parents are also visible
  // -> change their value to true
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

  if (state.corrs_state_.mano_corr_indices_.sizes()[0] == 0) {
    std::cout << "No correspondences. Abort." << std::endl;
    return false;
  }

  // upload to gpu
  if (!state.instance_->verts().is_cuda()) {
    if (state.instance_->model.hand_type == reclib::models::HandType::left) {
      mano_model_left_.gpu();
    } else {
      mano_model_right_.gpu();
    }
    state.instance_->gpu();
  }
  state.instance_->requires_grad(false);

  // load all parameters of current frame
  torch::Tensor trans;
  torch::Tensor shape;
  torch::Tensor rot;
  torch::Tensor pose;
  torch::Tensor apose;
  torch::Tensor pca;

  torch::Device dev = state.instance_->params.device();
  {
    torch::NoGradGuard guard;
    trans = state.instance_->trans().clone().detach();
    shape = state.instance_->shape().clone().detach();
    rot = state.instance_->rot().clone().detach();
    pca = state.instance_->hand_pca().clone().detach();
    apose = torch::matmul(state.instance_->model.hand_comps, pca.clone())
                .reshape({-1});
  }

  // load all parameters of previous frame
  torch::Tensor prev_trans;
  torch::Tensor prev_shape;
  torch::Tensor prev_rot;
  torch::Tensor prev_apose;
  torch::Tensor prev_pose;
  torch::Tensor prev_pca;
  torch::Tensor prev_visibility;

  if (sequence_frame_counter_ > 1) {
    torch::NoGradGuard guard;
    // load temporal information from last frame
    prev_trans = state.trans_.clone().detach().to(dev);
    prev_shape = state.shape_.clone().detach().to(dev);
    prev_rot = state.rot_.clone().detach().to(dev);
    prev_apose = state.pose_.clone().detach().to(dev);
  }

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

  extended_mask.index_put_({torch::indexing::Slice(box.index({1}).item<int>(),
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

  // pre-compute the cross_matrix, which transforms the original MANO pose
  // orientation to anatomically correct axes
  torch::Tensor cross_matrix =
      reclib::tracking::compute_apose_matrix(state.instance_->model, shape);

  std::vector<torch::Tensor> params;
  params.push_back(rot);
  params.push_back(pca);
  if (visible_percentage > 0.5) {
    params.push_back(shape);
  }

  // Configure Adam Options
  torch::optim::AdamOptions adam_opt;
  adam_opt.amsgrad() = true;
  if (sequence_frame_counter_ <= 1) {
    adam_opt.set_lr(opt_config["adam_lr_initialization"].as<float>());
  } else {
    adam_opt.set_lr(opt_config["adam_lr"].as<float>());
  }
  adam_opt.weight_decay() = opt_config["adam_weight_decay"].as<float>();

  // Configure LBFGS Options
  torch::optim::LBFGSOptions lbfgs_opt;
  if (sequence_frame_counter_ <= 1) {
    lbfgs_opt.max_iter(
        opt_config["lbfgs_inner_iterations_initialization"].as<int>());
  } else {
    lbfgs_opt.max_iter(opt_config["lbfgs_inner_iterations"].as<int>());
  }
  lbfgs_opt.line_search_fn() = "strong_wolfe";

  if (debug_)
    std::cout << "---- Preparation: " << timer_.look_and_reset() << " ms."
              << std::endl;

  for (unsigned int i = 0;
       i < opt_config["outer_iterations"].as<unsigned int>(); i++) {
    torch::optim::LBFGS optim_rot_pose_trans =
        torch::optim::LBFGS(params, lbfgs_opt);

    auto loss_func_rot_apose = [&]() {
      // ---------------------------------------------
      // Compute LBS
      // ---------------------------------------------
      std::pair<torch::Tensor, torch::Tensor> t;
      if (pca.requires_grad()) {
        // optimize over PCA
        t = reclib::tracking::torch_lbs_pca_anatomic(state.instance_->model,
                                                     trans, rot, shape, pca,
                                                     cross_matrix, false);
      } else {
        // optimize over pose
        t = reclib::tracking::torch_lbs_anatomic(state.instance_->model, trans,
                                                 rot, shape, apose,
                                                 cross_matrix, false);
      }

      // ---------------------------------------------
      // Silhouette Term
      // ---------------------------------------------
      const auto [silhouette_term, iou_pred] = rasterizer_loss(
          state, opt_config, context, t.first, colors, VP, corr, extended_mask);

      // ---------------------------------------------
      // Data Term
      // ---------------------------------------------
      torch::Tensor data_term =
          data_loss(state, opt_config, t.first, linearized_vertex_map,
                    linearized_normal_map, iou_pred);

      // ---------------------------------------------
      // Regularization terms
      // ---------------------------------------------
      std::vector<torch::Tensor> reg_losses =
          param_regularizer(opt_config, trans, rot, pose, pca, shape);
      std::vector<torch::Tensor> temp_losses = temp_regularizer(
          opt_config, {trans, prev_trans}, {rot, prev_rot}, {pose, prev_pose},
          {pca, prev_pca}, {shape, prev_shape}, iou_pred, data_term);

      // ---------------------------------------------
      // Compute final loss
      // ---------------------------------------------

      torch::Tensor loss = data_term + silhouette_term;
      for (unsigned int i = 0; i < reg_losses.size(); i++) {
        loss = loss + reg_losses[i];
      }
      for (unsigned int i = 0; i < temp_losses.size(); i++) {
        loss = loss + temp_losses[i];
      }

      loss.backward();

      if (1 && apose.requires_grad()) {
        for (int i = 0; i < apose.sizes()[0]; i++) {
          int joint = apose2joint[i];
          // do not update gradients of pose parameters in which
          // the corresponding joints were not visible
          if (!(bool)visibility_joints[joint]) {
            apose.mutable_grad().index({i}) = 0;
          }
        }
      }

      if (debug_)
        std::cout << "--------------------------------------" << std::endl;

      return loss;
    };

    // LBFGS requires a wrapping function
    auto lbfgs_loss_func_wrapper = [&]() {
      optim_rot_pose_trans.zero_grad();
      torch::Tensor loss = loss_func_rot_apose();
      return loss;
    };

    // -----------------------------------------------------------------------------------------------------------
    // Optimization Stage 0
    // -----------------------------------------------------------------------------------------------------------
    if (stage == 0 || !config_["Optimization"]["separate_stages"].as<bool>()) {
      pca.set_requires_grad(true);
      rot.set_requires_grad(true);
      shape.set_requires_grad(true);

      {
        torch::Tensor prev_apose_optim = apose.clone().detach();
        torch::Tensor prev_pca_optim = pca.clone().detach();

        optim_rot_pose_trans.step(lbfgs_loss_func_wrapper);

        if (debug_)
          std::cout << "diff: "
                    << (pca - prev_pca_optim).abs().sum().item<float>()
                    << std::endl;
      }

      if (debug_)
        std::cout << "[ITERATIONS]:" << lbfgs_opt.max_iter() << std::endl;

      pca.set_requires_grad(false);
      rot.set_requires_grad(false);
      shape.set_requires_grad(false);

      state.stage_ = 2;
    }

    if (!config_["Optimization"]["separate_stages"].as<bool>()) {
      torch::NoGradGuard guard;
      // recompute apose from optimized pca
      apose = torch::matmul(state.instance_->model.hand_comps, pca.clone())
                  .reshape({-1});
    }

    // -----------------------------------------------------------------------------------------------------------
    // Optimization Stage 1
    // -------------------------------------------------------------------------------------------------------

    torch::optim::Adam optim_rot_pose_trans_refined =
        torch::optim::Adam({trans, rot, shape, apose}, adam_opt);
    torch::optim::Adam optim_rot_pose_refined_ =
        torch::optim::Adam({shape, apose}, adam_opt);
    torch::optim::Adam optim_trans_rot =
        torch::optim::Adam({trans, rot}, adam_opt);

    int epochs = opt_config.i("adam_lr_epochs");
    if (sequence_frame_counter_ <= 1) {
      epochs = opt_config.i("adam_lr_epochs_initial");
    }

    torch::optim::StepLR optim_rot_pose_refined = torch::optim::StepLR(
        optim_rot_pose_refined_, epochs, opt_config.f("adam_lr_step_size"));
    torch::optim::StepLR optim_trans = torch::optim::StepLR(
        optim_trans_rot, epochs, opt_config.f("adam_lr_step_size"));

    if (stage == 1 || !config_["Optimization"]["separate_stages"].as<bool>()) {
      int termination_iter = opt_config["adam_inner_iterations"].as<int>();
      if (sequence_frame_counter_ <= 1 || state.loss_ < 0) {
        termination_iter =
            opt_config["adam_inner_iterations_initialization"].as<int>();
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

        if (opt_config.b("use_flip_flop")) {
          apose.set_requires_grad(true);
          rot.set_requires_grad(false);
          shape.set_requires_grad(true);
          trans.set_requires_grad(false);

          // optimize only apose + shape
          optim_rot_pose_refined_.zero_grad();
          // loss_func_rot_apose takes by far the most time, split between this
          // call and the one below
          loss = loss_func_rot_apose();
          optim_rot_pose_refined_.step();
          optim_rot_pose_refined.step();

          if (apose.isnan().any().item<bool>()) {
            throw std::runtime_error("Apose is NaN");
          }
          if (shape.isnan().any().item<bool>()) {
            throw std::runtime_error("Shape is NaN");
          }

          // optimize only rot + trans
          apose.set_requires_grad(false);
          rot.set_requires_grad(true);
          shape.set_requires_grad(false);
          trans.set_requires_grad(true);

          optim_trans_rot.zero_grad();
          loss = loss_func_rot_apose();
          optim_trans_rot.step();
          optim_trans.step();

          if (rot.isnan().any().item<bool>()) {
            throw std::runtime_error("Rot is NaN");
          }
          if (trans.isnan().any().item<bool>()) {
            throw std::runtime_error("Trans is NaN");
          }

        } else {
          // optimize only rot + trans + pose + shape
          optim_rot_pose_trans_refined.zero_grad();
          loss = loss_func_rot_apose();
          optim_rot_pose_trans_refined.step();
        }

        // store the best parameters of the whole optimization
        if (best_loss == -1) {
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

        // compute the difference in parameter updates
        float diff = (apose - prev_apose_optim).abs().sum().item<float>();
        diff += (rot - prev_rot_optim).abs().sum().item<float>();
        diff += (trans - prev_trans_optim).abs().sum().item<float>();

        // terminate algorithm if parameters barely changed
        if ((diff <= opt_config["adam_termination_eps"].as<float>()) ||
            iter > termination_iter) {
          break;
        }

        iter++;
      }
      if (debug_) {
        std::cout << "Loss: " << loss << std::endl;
        std::cout << "Best loss: " << best_loss << std::endl;
        std::cout << "Last frame best loss: " << state.loss_ << std::endl;
        std::cout << "Difference in losses: "
                  << (best_loss - state.loss_) / state.loss_ << std::endl;
      }

      apose.set_requires_grad(false);
      rot.set_requires_grad(false);
      shape.set_requires_grad(false);
      trans.set_requires_grad(false);

      // compute PCA since pose updates are stored as a PCA within the instance
      pca = torch::matmul(state.instance_->model.hand_comps.inverse(),
                          apose.unsqueeze(1))
                .reshape({-1});
      if (pca.isnan().any().item<bool>()) {
        throw std::runtime_error("PCA is NaN");
      }

      if (debug_) {
        std::cout << "[ITERATIONS]:" << iter << ", " << termination_iter
                  << std::endl;
      }

      // Reset the tracking state if the relative or absolute loss
      // becomes too high
      if (state.loss_ > 0 && (best_loss - state.loss_) / state.loss_ >
                                 opt_config.f("loss_relative_threshold")) {
        if (debug_) {
          std::cout << "Loss higher than last one: " << best_loss << " <-> "
                    << state.loss_ << std::endl;
        }

        return false;
      }
      if (state.loss_ > 0 &&
          best_loss > opt_config.f("loss_absolute_threshold")) {
        if (debug_) {
          std::cout << "Loss higher than absolute thresh: " << best_loss
                    << " <-> " << opt_config.f("loss_absolute_threshold")
                    << std::endl;
        }

        return false;
      }
      state.loss_ = best_loss;
      state.stage_ = 3;
    }

    // -----------------------------------------------------------------------------------------------------------
    // Optimization Stage 2
    // -------------------------------------------------------------------------------------------------------

    if (stage == 2) {
      shape.set_requires_grad(true);
      apose.set_requires_grad(false);
      trans.set_requires_grad(false);
      rot.set_requires_grad(false);
      pca.set_requires_grad(false);

      // only optimize the hand shape
      torch::optim::Adam optim_shape = torch::optim::Adam({shape}, adam_opt);

      int iter = 0;
      while (true) {
        torch::Tensor prev_shape_optim = shape.clone().detach();

        optim_shape.zero_grad();
        torch::Tensor loss = loss_func_rot_apose();
        optim_shape.step();

        if (debug_)
          std::cout << "diff: "
                    << (shape - prev_shape_optim).abs().sum().item<float>()
                    << std::endl;

        if ((shape - prev_shape_optim).abs().sum().item<float>() <=
                opt_config["adam_termination_eps"].as<float>() ||
            iter > opt_config["adam_inner_iterations"].as<int>()) {
          break;
        }

        iter++;
      }
      shape.set_requires_grad(false);

      if (debug_) std::cout << "Iterations: " << iter << std::endl;
      state.stage_ = 4;
    }
  }

  if (debug_)
    std::cout << "---- Optimization: " << timer_.look_and_reset() << " ms."
              << std::endl;

  int pca_len = 45;
  if (state.instance_->use_anatomic_pca_) {
    pca_len = 23;
  }

  state.instance_->set_trans(trans);
  state.instance_->params.index_put_(
      {torch::indexing::Slice(3 + 3 + 45, 3 + 3 + 45 + pca_len)}, pca);
  state.instance_->params.index_put_(
      {torch::indexing::Slice(3 + 3 + 45 + pca_len, 3 + 3 + 45 + pca_len + 10)},
      shape);
  state.instance_->set_rot(rot);
  state.instance_->update(false, true, false, update_meshes);

  if (debug_)
    std::cout << "---- Postprocessing: " << timer_.look_and_reset() << " ms."
              << std::endl;

  if (config_["Debug"]["show_corr_lines"].as<bool>() &&
      config_["Optimization"]["multithreading"].as<bool>() == false) {
    visualize_correspondences(index);
  }
  return true;
}

torch::Tensor reclib::tracking::HandTracker::data_loss(
    HandState& state, reclib::Configuration& opt_config,
    torch::Tensor mano_verts, torch::Tensor linearized_vertex_map,
    torch::Tensor linearized_normal_map, torch::Tensor iou_pred) {
  // iou is a weighting factor between data term and silhouette term
  // the greater iou -> higher weight for silhouette
  // the smaller iou -> higher weight for data (because less overlap)
  float weight = std::exp(-iou_pred.item<float>() + 1) *
                 opt_config["data_weight"].as<float>();

  torch::Tensor point2point =
      weight * (linearized_vertex_map.index(
                    {state.corrs_state_.pc_corr_indices_linearized_}) -
                mano_verts.index({state.corrs_state_.mano_corr_indices_}));

  torch::Tensor point2plane =
      weight *
      torch::bmm(point2point.index({torch::All, torch::None, torch::All}),
                 linearized_normal_map
                     .index({state.corrs_state_.pc_corr_indices_linearized_})
                     .index({torch::All, torch::All, torch::None}));

  torch::Tensor loss = weight * (point2point.abs().mean() * 0.33 +
                                 point2plane.abs().mean() * 0.66);

  if (debug_) {
    std::cout << "-- data: " << loss.item<float>() << ", weight: " << weight
              << " (iou:" << iou_pred.item<float>() << ")" << std::endl;
  }

  return loss;
}

std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::HandTracker::rasterizer_loss(
    HandState& state, reclib::Configuration& opt_config,
    RasterizeCRStateWrapper& context, torch::Tensor mano_verts,
    torch::Tensor colors, torch::Tensor VP, torch::Tensor corr,
    torch::Tensor extended_mask) {
  torch::Device dev = mano_verts.device();
  int w = intrinsics_.image_width_;
  int h = intrinsics_.image_height_;
  std::tuple<int, int> res = std::make_tuple(h, w);

  torch::Tensor verts = mano_verts.contiguous();

  torch::Tensor ones =
      torch::ones({verts.sizes()[0], 1}, torch::TensorOptions().device(dev))
          .contiguous();
  // compute vertices in homogeneous coordinates
  torch::Tensor verts_hom = torch::cat({verts, ones}, 1).contiguous();
  // transform to clip space
  torch::Tensor verts_clip =
      torch::matmul(verts_hom, VP.transpose(1, 0)).contiguous();
  verts_clip = verts_clip.index({torch::None, "..."});
  torch::Tensor pos_idx = state.instance_->model.faces;

  // ---------------------------------------------
  // Apply Nvdiffrast (differential Rasterizer)
  // ---------------------------------------------
  std::vector<torch::Tensor> rast_out =
      reclib::dnn::rasterize(context, verts_clip, pos_idx, res);
  std::vector<torch::Tensor> interp_out =
      reclib::dnn::interpolate(colors, rast_out[0], pos_idx);
  std::vector<torch::Tensor> antialias_out =
      reclib::dnn::antialias(interp_out[0], rast_out[0], verts_clip, pos_idx);
  // predicted MANO colors (aka canonical coordinates) from Nvdiffrast
  torch::Tensor color_pred = antialias_out[0].squeeze(0);

  // compute nonzero pixels within prediction
  torch::Tensor positive_samples_pred = (color_pred.sum(2) > 0).sum();
  // compute pixel intersection between prediction and ground truth
  torch::Tensor iou_intersection =
      torch::logical_and(color_pred.sum(2) > 0, extended_mask);
  // compute pxiel union between prediction and ground truth
  torch::Tensor iou_union =
      torch::logical_or(color_pred.sum(2) > 0, extended_mask);
  // compute intersection over union (iou)
  torch::Tensor iou_pred = iou_intersection.sum() / iou_union.sum();

  torch::Tensor loss = opt_config["silhouette_weight"].as<float>() *
                       torch::l1_loss(corr, color_pred, torch::Reduction::Sum) /
                       extended_mask.sum();

  if (debug_) {
    std::cout << "-- silhouette: " << loss.item<float>() << std::endl;
  }

  return std::make_pair(loss, iou_pred);
}

std::vector<torch::Tensor> reclib::tracking::HandTracker::param_regularizer(
    reclib::Configuration& opt_config, torch::Tensor trans, torch::Tensor rot,
    torch::Tensor pose, torch::Tensor pca, torch::Tensor shape) {
  std::vector<torch::Tensor> losses;

  if (pose.requires_grad()) {
    losses.push_back(opt_config["param_regularizer_pose_weight"].as<float>() *
                     pose.norm() * pose.norm());

    if (debug_) {
      std::cout << "-- param reg (pose): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (shape.requires_grad()) {
    losses.push_back(opt_config["param_regularizer_shape_weight"].as<float>() *
                     shape.norm() * shape.norm());

    if (debug_) {
      std::cout << "-- param reg (shape): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }

  return losses;
}

std::vector<torch::Tensor> reclib::tracking::HandTracker::temp_regularizer(
    reclib::Configuration& opt_config,
    std::pair<torch::Tensor, torch::Tensor> trans,
    std::pair<torch::Tensor, torch::Tensor> rot,
    std::pair<torch::Tensor, torch::Tensor> pose,
    std::pair<torch::Tensor, torch::Tensor> pca,
    std::pair<torch::Tensor, torch::Tensor> shape, torch::Tensor iou_pred,
    torch::Tensor data_term) {
  std::vector<torch::Tensor> losses;
  if (sequence_frame_counter_ <= 1) return losses;

  float regularization_weight_iou = std::exp(-iou_pred.item<float>() + 1);
  float regularization_weight_data = std::log1p(data_term.item<float>());

  if (debug_) {
    std::cout << "[TEMP REG] iou_w: " << regularization_weight_iou
              << " data_w: " << regularization_weight_data << std::endl;
  }

  // ----------- APOSE -----------
  if (pose.first.requires_grad() && pose.second.sizes()[0] > 0) {
    losses.push_back(
        opt_config["temp_regularizer_pose_weight"].as<float>() *
        (regularization_weight_iou + regularization_weight_data) *
        (torch::mse_loss(pose.first, pose.second, torch::Reduction::None) *
         (pose.second > 0))
            .sum());

    if (debug_) {
      std::cout << "-- temp reg (pose): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (shape.first.requires_grad() && shape.second.sizes()[0] > 0) {
    losses.push_back(
        opt_config["temp_regularizer_shape_weight"].as<float>() *
        torch::nn::functional::mse_loss(shape.first, shape.second));

    if (debug_) {
      std::cout << "-- temp reg (shape): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (trans.first.requires_grad() && trans.second.sizes()[0] > 0) {
    losses.push_back(
        opt_config["temp_regularizer_trans_weight"].as<float>() *
        (regularization_weight_iou + regularization_weight_data) *
        torch::nn::functional::mse_loss(trans.first, trans.second));

    if (debug_) {
      std::cout << "-- temp reg (trans): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (rot.first.requires_grad() && rot.second.sizes()[0] > 0) {
    losses.push_back(opt_config["temp_regularizer_rot_weight"].as<float>() *
                     (regularization_weight_iou + regularization_weight_data) *
                     torch::nn::functional::mse_loss(rot.first, rot.second));

    if (debug_) {
      std::cout << "-- temp reg (rot): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  return losses;
}

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE