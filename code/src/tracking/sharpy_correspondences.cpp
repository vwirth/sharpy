
#include <ATen/ops/nonzero.h>
#include <c10/core/DeviceType.h>
#include <reclib/depth_processing.h>
#include <reclib/optim/model_registration.h>
#include <reclib/tracking/sharpy_utils.h>

#include <opencv2/core/matx.hpp>

#include "reclib/assert.h"
#include "reclib/dnn/dnn_utils.h"
#include "reclib/math/eigen_glm_interface.h"
#include "reclib/tracking/sharpy.cuh"
#include "reclib/tracking/sharpy_tracker.h"

#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <torch_tensorrt/core/compiler.h>
#include <torch_tensorrt/torch_tensorrt.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include "reclib/tracking/matches.cuh"

// -----------------------------------------------------------------------
// CorrespondenceState
// -----------------------------------------------------------------------

void reclib::tracking::HandTracker::compute_matches(
    reclib::Configuration config, int index) {
  torch::NoGradGuard guard;

  // read from config file
  float pc_normal_thresh = config.f("pc_normal_thresh");
  float pc_dist_thresh = config.f("pc_dist_thresh");
  float nn_dist_thresh = config.f("nn_dist_thresh");

  HandState& state = hand_states_.at(index);

  // clone creates a new memory pool for the tensor,
  // creating a deep copy without memory holes.
  torch::Tensor linearized_pc =
      state.vertex_map_
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1}), torch::All})
          .clone();
  torch::Tensor linearized_pc_canonicals =
      state.corr_
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1}), torch::All})
          .clone();
  torch::Tensor linearized_pc_normals =
      state.normal_map_
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1}), torch::All})
          .clone();
  torch::Tensor linearized_segmentation =
      state.corr_segmented_
          .index({state.nonzero_indices_.index({torch::All, 0}),
                  state.nonzero_indices_.index({torch::All, 1})})
          .clone();
  torch::Tensor mano_segmentation =
      reclib::tracking::sharpy::batch_corr2seg(mano_corr_space_).clone();

  if (!state.corrs_state_.initialized_) {
    state.corrs_state_.pc_wrapper_.points_ = mano_corr_space_.data_ptr<float>();
    state.corrs_state_.pc_wrapper_.num_points_ = mano_corr_space_.sizes()[0];
    state.corrs_state_.kdtree_.buildIndex();

    state.corrs_state_.initialized_ = true;
  }
  std::fill(state.corrs_state_.joints_visible_.begin(),
            state.corrs_state_.joints_visible_.end(), false);

  state.corrs_state_.pc_wrapper_.points_ = mano_corr_space_.data_ptr<float>();
  state.corrs_state_.pc_wrapper_.num_points_ = mano_corr_space_.sizes()[0];

  // TODO: Maybe use torch::kInt16 for indices?
  torch::Tensor correlation_indices =
      torch::zeros(linearized_pc.sizes()[0], at::kLong).to(torch::kCUDA);

  torch::Tensor mano_corr_space = mano_corr_space_.clone().to(torch::kCUDA);
  torch::Tensor mano_vertices = state.instance_->verts().to(torch::kCUDA);
  mano_segmentation = mano_segmentation.to(torch::kCUDA);
  linearized_pc_canonicals = linearized_pc_canonicals.to(torch::kCUDA);
  linearized_pc = linearized_pc.to(torch::kCUDA);
  linearized_pc_normals = linearized_pc_normals.to(torch::kCUDA);
  linearized_segmentation = linearized_segmentation.to(torch::kCUDA);
  torch::Tensor verts2joints =
      torch::from_blob(mano_verts2joints_.data(), mano_verts2joints_.size(),
                       torch::kInt)
          .clone()
          .to(torch::kCUDA);
  torch::Tensor joints_visible =
      torch::zeros(state.corrs_state_.joints_visible_.size(), torch::kBool)
          .to(torch::kCUDA);
  // The overwhelming majority of time is spent in this function
  // (153/154ms, >99%)
  // -> preparation and postprocessing is negligible
  reclib::tracking::cuda::compute_matches(
      mano_corr_space, mano_vertices, mano_segmentation,
      linearized_pc_canonicals, linearized_pc, linearized_pc_normals,
      linearized_segmentation, state.stage_, state.mean_depth_, pc_dist_thresh,
      pc_normal_thresh, nn_dist_thresh, verts2joints, correlation_indices,
      joints_visible);
  // Make sure *not* to copy tensors back to the cpu unnecessarily.
  // Especially shared memory like mano_corr_space_ will make problems when
  // computing hands in parallel
  linearized_pc_canonicals = linearized_pc_canonicals.to(torch::kCPU);
  joints_visible = joints_visible.to(torch::kCPU);
  // TOOD: make make state.corrs_state_.joints_visible a Tensor so we can pass
  // it directly and avoid copying?
  for (int i = 0; i < joints_visible.sizes()[0]; i++) {
    state.corrs_state_.joints_visible_[i] = joints_visible[i].item<bool>();
  }

  // Postprocess the results.
  // First, we remove all entries that have no valid match.
  // valid_matches is a mask that is True where the query point was assigned to
  // a mano point
  torch::Tensor valid_matches = correlation_indices.not_equal(-1);
  state.corrs_state_.mano_corr_indices_ =
      correlation_indices.masked_select(valid_matches);
  // the indices of the canonicals with matches are exactly the indices where
  // the valid mask is true. nonzero gives us these indices (False==0, True==1)
  // nonzero(as_tuple=True) should also work instead of flatten,
  // but that api seems to work only in Python and not in C++
  state.corrs_state_.pc_corr_indices_linearized_ =
      valid_matches.nonzero().flatten();
}

// -----------------------------------------------------------------------
// HandTracker
// -----------------------------------------------------------------------

void reclib::tracking::HandTracker::prepare_correspondences(int index) {
  torch::NoGradGuard guard;

  torch::Tensor box = network_output_[2].index({index});
  torch::Tensor mask = network_output_[3].index({index});
  torch::Tensor corr = network_output_[4].index({index});

  HandState& state = hand_states_.at(index);
  state.masked_depth_ = reclib::dnn::cv2torch(depth_);

  _RECLIB_ASSERT_EQ(mask.sizes()[0], state.masked_depth_.sizes()[0]);
  _RECLIB_ASSERT_EQ(mask.sizes()[1], state.masked_depth_.sizes()[1]);
  _RECLIB_ASSERT_EQ(corr.sizes()[0], 3);
  _RECLIB_ASSERT_EQ(corr.sizes()[1], state.masked_depth_.sizes()[0]);
  _RECLIB_ASSERT_EQ(corr.sizes()[2], state.masked_depth_.sizes()[1]);

  if (debug_) {
    timer_.look_and_reset();
  }

  // mask correspondences w.r.t. hand region
  torch::Tensor corr_masked =
      corr * mask.index({torch::None, torch::All, torch::All});
  corr_masked = corr_masked.permute({1, 2, 0});
  corr_masked =
      corr_masked.index({torch::indexing::Slice(box.index({1}).item<int>(),
                                                box.index({3}).item<int>()),
                         torch::indexing::Slice(box.index({0}).item<int>(),
                                                box.index({2}).item<int>())});
  state.corr_ = corr_masked.contiguous();

  // mask depth w.r.t. hand region
  torch::Tensor depth_tensor =
      reclib::dnn::cv2torch(depth_, true).to(mask.device());
  torch::Tensor masked_depth =
      (depth_tensor * mask.index({torch::All, torch::All, torch::None}));

  masked_depth =
      masked_depth.index({torch::indexing::Slice(box.index({1}).item<int>(),
                                                 box.index({3}).item<int>()),
                          torch::indexing::Slice(box.index({0}).item<int>(),
                                                 box.index({2}).item<int>())});
  state.masked_depth_ = masked_depth.contiguous();

  if (debug_) {
    std::cout << "---- Depth/Corr Masking Operations: "
              << timer_.look_and_reset() << " ms." << std::endl;
  }

  // compute median depth of the hand (later used for pruning correspondences)
  int median_index = int((masked_depth > 0).sum().item().toFloat() / 2.f);
  if ((masked_depth > 0).sum().item().toFloat() > 0) {
    state.mean_depth_ =
        std::get<0>(masked_depth.index({masked_depth > 0}).reshape(-1).sort())
            .index({median_index})
            .item()
            .toFloat() /
        1000.f;
  } else {
    state.mean_depth_ = 0;
  }

  if (debug_) {
    std::cout << "---- Median Depth computation: " << timer_.look_and_reset()
              << " ms." << std::endl;
  }

  state.corr_ = state.corr_.to(torch::kCUDA);
  reclib::tracking::cuda::compute_segmentation_from_corrs(
      state.corr_, state.corr_segmented_, state.visibility_map_);
  state.corr_ = state.corr_.to(torch::kCPU);

  if (debug_) {
    std::cout << "---- Segmentation from Correspondences computation: "
              << timer_.look_and_reset() << " ms." << std::endl;
  }

  state.masked_depth_ = state.masked_depth_.to(torch::kCUDA);
  reclib::cuda::ComputeVertexMap(
      state.masked_depth_, state.vertex_map_, intrinsics_.Level(0), -1,
      ivec2(box.index({0}).item<int>(), box.index({1}).item<int>()), 0.001f);
  reclib::cuda::ComputePrunedVertexNormalMap(
      state.vertex_map_, state.normal_map_,
      config_["Correspondences"]["pc_normal_thresh"].as<float>());

  // store all non-zero indices of the vertex map
  // used for linearization later on
  state.nonzero_indices_ =
      torch::nonzero(state.vertex_map_.index({torch::All, torch::All, 2}));

  if (debug_) {
    std::cout << "---- Vertex and normal map computation: "
              << timer_.look_and_reset() << " ms." << std::endl;
  }
}

void reclib::tracking::HandTracker::compute_correspondences(
    const CorrespondenceMode& mode) {
  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    compute_correspondences(mode, i);
  }
}

void reclib::tracking::HandTracker::compute_correspondences(
    const CorrespondenceMode& mode, int index) {
  HandState& state = hand_states_.at(index);
  torch::Tensor box = network_output_[2].index({index});

  if (config_["Debug"]["show_prepared_corr"].as<bool>()) {
    cv::destroyAllWindows();
    CpuMat masked_d = reclib::dnn::torch2cv(state.masked_depth_);
    CpuMat masked_c = reclib::dnn::torch2cv(state.corr_);

    cv::imshow("Depth", masked_d);
    cv::imshow("Corr", masked_c);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }

  reclib::Configuration corr_config = config_.subconfig({"Correspondences"});

  if (debug_) {
    timer_.look_and_reset();
  }

  compute_matches(corr_config, index);

  if (debug_) {
    std::cout << "---- Computing correspondences: " << timer_.look_and_reset()
              << " ms." << std::endl;

    std::cout << "---- Num correspondences: "
              << state.corrs_state_.mano_corr_indices_.sizes()[0] << std::endl;

    /*
    With CPU kd-tree (average over 4 repeated registrations)
---- Computing correspondences: 1718ms / 1564ms
---- Num correspondences: 23542 / 15194
    With naive cuda
---- Computing correspondences: 152.4ms / 113.7ms
---- Num correspondences: 23960 / 15720
    (More correspondences because all mano vertices were considered)
    */
  }

  // generate openGl pointcloud for visualzation
  if (config_["Debug"]["show_corr_img"].as<bool>() == true &&
      config_["Optimization"]["multithreading"].as<bool>() == false) {
    visualize_correspondence_pointcloud(index);
  }
}

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE
