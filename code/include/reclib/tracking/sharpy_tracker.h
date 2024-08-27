#ifndef RECLIB_TRACKING_SHARPY_TRACKER_H
#define RECLIB_TRACKING_SHARPY_TRACKER_H

#include <yaml-cpp/yaml.h>

#include <cstdint>

#include "nvdiffrast/torch/torch_types.h"
#include "reclib/benchmark/mano.h"
#include "reclib/configuration.h"
#include "reclib/data_types.h"
#include "reclib/dnn/dnn.h"
#include "reclib/dnn/yolact.h"
#include "reclib/models/smpl.h"
#include "reclib/models/smpl_torch.h"
#include "reclib/optim/correspondences.h"
#include "reclib/tracking/mano_rgbd.cuh"
#include "reclib/tracking/mano_rgbd_corrs.h"
#include "reclib/voxel.h"

#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

#include <opencv2/dnn.hpp>

namespace reclib {
namespace tracking {

extern const vec3 SKIN_COLOR;
extern const std::vector<int> JOINT2VIS_INDICES;

enum class HandType {
  ANY = -1,
  LEFT = 0,
  RIGHT = 1,
};

enum class CorrespondenceMode {
  REGISTRATION = 0,
  OPTIMIZATION = 1,
};

struct CorrespondenceState {
  // indices of mano and pointcloud (pc) correspondences

  reclib::optim::PointCloud<float> pc_wrapper_;
  reclib::optim::KdTree<float, 3> kdtree_;

  bool initialized_;
  std::vector<bool> joints_visible_;

  torch::Tensor mano_corr_indices_;
  // linearized index to only the VALID point clouds (with nonzero indices)
  torch::Tensor pc_corr_indices_linearized_;

  CorrespondenceState()
      : kdtree_(3, pc_wrapper_, {32 /* max leaf*/}),
        initialized_(false),
        joints_visible_(16) {}
  CorrespondenceState(const CorrespondenceState& other)
      : kdtree_(3, pc_wrapper_, {32 /* max leaf*/}),
        initialized_(false),
        joints_visible_(16) {}
  void reset() { initialized_ = false; }
};

class HandState {
 private:
  static bool global_initialized_;
  static void initialize_globals();

  reclib::Configuration config_;

 public:
  // OpenGL stuff
  static std::vector<vec4> color_alphas_mano_;
  static std::vector<vec3> color_mano_joints_;
  static std::vector<float> alpha_mano_joints_;

  // hand-related images (2D representation)
  torch::Tensor vertex_map_;
  torch::Tensor normal_map_;
  torch::Tensor corr_;
  torch::Tensor corr_segmented_;
  torch::Tensor masked_depth_;
  torch::Tensor nonzero_indices_;

  reclib::tracking::CorrespondenceState corrs_state_;

  // MANO parameters of the previous frame
  torch::Tensor trans_;  // translation of the wrist
  torch::Tensor rot_;    // rotation of the wrist
  torch::Tensor shape_;  // MANO shape parameters
  torch::Tensor pose_;   // MANO anatomical pose parameters
  torch::Tensor pca_;    // MANO PCA parameters

  // MANO instance to be modified continuously during the pipeline
  const std::shared_ptr<reclib::modelstorch::ModelInstance<
      reclib::models::MANOConfigAnglePCAGeneric<23>>>
      instance_;

  float loss_;
  float mean_depth_;
  bool failed_;
  // per-frame stage,
  // 0 = initialized,
  // 1 = registered,
  // 2 = optimized (stage 0),
  // 3 = optimized (stage 1),
  // ....
  int stage_;

  std::vector<uint8_t> visibility_map_;
  std::vector<bool> joints_visible_;
  std::vector<bool> joints_errorenous_;

  HandState(reclib::Configuration& config,
            const reclib::modelstorch::Model<
                reclib::models::MANOConfigAnglePCAGeneric<23>>& model);

  void prepare_new_frame();
  void reset();
};

class HandTracker {
  reclib::opengl::Timer timer_;
  reclib::Configuration config_;

  // pipeline inputs
  CpuMat rgb_;
  CpuMat depth_;
  reclib::IntrinsicParameters intrinsics_;

  // networks
  reclib::dnn::Yolact network_;
  cv::dnn::Net openpose_;

  // image transforms: transforms and untransforms images into
  // network-appropriate format
  reclib::dnn::BaseTransform transform_;

  // 0 = class label (left or right hand)
  // 1 = confidence score for class label
  // 2 = box coordinates
  // 3 = hand mask
  // 4 = hand correspondences
  std::vector<torch::Tensor> network_output_;

  // MANO-specific parameters
  Eigen::Vector<int, reclib::models::MANOConfig::n_verts()> mano_verts2joints_;
  torch::Tensor mano_corr_space_;
  reclib::modelstorch::Model<reclib::models::MANOConfigAnglePCAGeneric<23>>
      mano_model_left_;
  reclib::modelstorch::Model<reclib::models::MANOConfigAnglePCAGeneric<23>>
      mano_model_right_;

  // Hand states
  std::vector<HandState> hand_states_;

  // other state-specific stuff
  bool debug_;
  unsigned int sequence_frame_counter_;
  unsigned int global_frame_counter_;

  std::map<std::string, reclib::opengl::Drawelement> generated_drawelements_;

  // -------------------------- NETWORKS -----------------------
  std::vector<torch::Tensor> reduce(std::vector<torch::Tensor>& in);
  cv::Rect compute_crop_factor(int width, int height);
  void detect(cv::Rect crop_factor = cv::Rect(0, 0, -1, -1));
  void detect_openpose(unsigned int index, const CpuMat& rgb,
                       cv::Rect crop_factor);

  // -------------------------- CORRESPONDENCES -----------------------
  // computes point-to-vertex correspondences
  // matches every point in the depth cloud to its nearest MANO vertex
  // filters out correspondence matches w.r.t. certain thresholds
  void compute_matches(reclib::Configuration config, int index);
  void compute_correspondences(const CorrespondenceMode& mode, int index);

  // -------------------------- OPTIMIZATION -----------------------
  std::optional<bool> register_hand(unsigned int index);
  std::optional<bool> optimize_hand(unsigned int index, int stage);

  // -------------------------- VISUALIZATION -----------------------
  void visualize_uncertainty(unsigned int index);
  void visualize_correspondences(unsigned int index);
  void visualize_correspondence_pointcloud(unsigned int index);

 public:
  explicit HandTracker(const reclib::Configuration& config);

  // ---------------------- Getter & Setter -----------------------
  unsigned int get_global_fc() { return global_frame_counter_; }
  void set_global_fc(unsigned int val) { global_frame_counter_ = val; }
  std::vector<bool> get_joints_visible(int index) {
    return hand_states_[index].joints_visible_;
  }
  std::vector<bool> get_joints_errorenous(int index) {
    return hand_states_[index].joints_errorenous_;
  }
  const std::vector<HandState>& hand_states() { return hand_states_; }
  reclib::Configuration& config() { return config_; };

  // reset tracking system, useful in case of a new video sequence
  void reset();
  // reset hand states of those in which the 'failed_' flag is set to true
  void reset_failed();
  // reset hand states of those in which the 'failed_' flag is set to true
  // and retry pipeline
  void retry_failed();
  // restore hand states (of previous frame) of those in which the 'failed_'
  // flag is set to true
  void restore_failed();

  // -------------------- PROCESS INPUTS ---------------------
  void process_image(
      const CpuMat& rgb, const CpuMat& depth,
      const reclib::IntrinsicParameters& intrinsics = IntrinsicParameters(),
      cv::Rect crop_factor = cv::Rect(0, 0, -1, -1));
  void prepare_correspondences(int index);

  // -------------------------- CORRESPONDENCES -----------------------
  void compute_correspondences(const CorrespondenceMode& mode);

  // -------------------------- OPTIMIZATION -----------------------
  std::vector<bool> register_hand();
  std::vector<bool> optimize_hand(int stage);
  torch::Tensor data_loss(HandState& state, reclib::Configuration& opt_config,
                          torch::Tensor mano_verts,
                          torch::Tensor linearized_vertex_map,
                          torch::Tensor linearized_normal_map,
                          torch::Tensor iou_pred);

  std::pair<torch::Tensor, torch::Tensor> rasterizer_loss(
      HandState& state, reclib::Configuration& opt_config,
      RasterizeCRStateWrapper& context, torch::Tensor mano_verts,
      torch::Tensor colors, torch::Tensor VP, torch::Tensor corr,
      torch::Tensor extended_mask);

  std::vector<torch::Tensor> param_regularizer(
      reclib::Configuration& opt_config, torch::Tensor trans, torch::Tensor rot,
      torch::Tensor pose, torch::Tensor pca, torch::Tensor shape);

  std::vector<torch::Tensor> temp_regularizer(
      reclib::Configuration& opt_config,
      std::pair<torch::Tensor, torch::Tensor> trans,
      std::pair<torch::Tensor, torch::Tensor> rot,
      std::pair<torch::Tensor, torch::Tensor> pose,
      std::pair<torch::Tensor, torch::Tensor> pca,
      std::pair<torch::Tensor, torch::Tensor> shape, torch::Tensor iou_pred,
      torch::Tensor data_term);

  // -------------------------- VISUALIZATION -----------------------
  void visualize_uncertainty();

  // -------------------- EXECUTE THE FULL PIPELINE ---------------------
  void process(
      const CpuMat& rgb, const CpuMat& depth,
      const reclib::IntrinsicParameters& intrinsics = IntrinsicParameters(),
      cv::Rect crop_factor = cv::Rect(0, 0, -1, -1));
  void compute_all_correspondences(int index, at::Stream stream);
  void compute_all_correspondences();
};

}  // namespace tracking
}  // namespace reclib

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE

#endif
