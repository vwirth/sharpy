#ifndef RECLIB_MANO_RGBD_H
#define RECLIB_MANO_RGBD_H

#include <yaml-cpp/yaml.h>

#include <cstdint>

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

struct HandLimits {
  static const vec3 limits_min[16];
  static const vec3 limits_max[16];
};

struct GMM {
  torch::Tensor weights_;
  torch::Tensor means_;
  torch::Tensor inv_covs_;
  torch::Tensor cov_det_;
};

struct LastFrame {
  // 0 = cls index
  // 2 = box
  // 3 = mask
  // 4 = corrs
  std::vector<torch::Tensor> network_output_;

  CpuMat rgb_;
  CpuMat depth_;
  std::vector<CpuMat> vertex_maps_;
  std::vector<CpuMat> normal_maps_;
  std::vector<CpuMat> corrs_;
  std::vector<CpuMat> corrs_segmented_;
  std::vector<std::vector<uint8_t>> visibility_map_;
  std::vector<CpuMat> masked_depth_;
  std::vector<float> mean_depth_;
  std::vector<ManoRGBDCorrespondences> corr_indices_;

  std::vector<torch::Tensor> trans_;
  std::vector<torch::Tensor> shape_;
  std::vector<torch::Tensor> pose_;
  std::vector<torch::Tensor> rot_;
  std::vector<torch::Tensor> pca_;
  std::vector<float> loss_;
  std::vector<torch::Tensor> keypoints_;

  std::vector<std::vector<bool>> joints_visible_;
  std::vector<std::vector<bool>> joints_errorenous_;
  // std::vector<torch::Tensor> pose_pca_;

  // std::vector<Eigen::Vector<float, 3>> trans_;
  // std::vector<Eigen::Vector<
  //     float,
  //     reclib::models::MANOConfigAnglePCAGeneric<45>::n_shape_blends()>>
  //     shape_;
  // std::vector<Eigen::Vector<
  //     float,
  //     reclib::models::MANOConfigAnglePCAGeneric<45>::n_explicit_joints() *
  //     3>> pose_;
  // std::vector<Eigen::Vector<
  //     float, reclib::models::MANOConfigAnglePCAGeneric<45>::n_hand_pca()>>
  //     pose_pca_;

  LastFrame(unsigned int n_hands)
      : vertex_maps_(n_hands),
        normal_maps_(n_hands),
        corrs_(n_hands),
        corrs_segmented_(n_hands),
        visibility_map_(n_hands),
        masked_depth_(n_hands),
        mean_depth_(n_hands),
        corr_indices_(n_hands),
        trans_(n_hands),
        shape_(n_hands),
        pose_(n_hands),
        rot_(n_hands),
        pca_(n_hands),
        loss_(n_hands),
        keypoints_(n_hands),
        joints_visible_(n_hands),
        joints_errorenous_(n_hands)
        // pose_pca_(n_hands)
        {

        };
};

enum class CorrespondenceMode {
  REGISTRATION = 0,
  OPTIMIZATION = 1,
};

template <typename MANOTYPE>
class ManoRGBD {
  enum class HandMode {
    LEFT = 0,
    RIGHT = 1,
    BOTH = 2,
  };

  reclib::opengl::Timer timer_;

  reclib::Configuration config_;
  bool debug_;
  reclib::IntrinsicParameters intrinsics_;

  reclib::dnn::Yolact network_;
  cv::dnn::Net openpose_;
  reclib::dnn::BaseTransform transform_;

  Eigen::Vector<int, reclib::models::MANOConfig::n_verts()> mano_verts2joints_;

  std::vector<vec3> mano_corr_space_;
  LastFrame last_frame_;

  unsigned int frame_counter_;
  unsigned int global_frame_counter_;
  unsigned int optimization_counter_;

  uvec3 volume_size_;
  float volume_scale_;
  vec3 volume_translation_;
  CpuMat voxel_pos_;
  GpuMat tsdf_volume_;   // short2
  GpuMat tsdf_weights_;  // int16

  torch::Tensor reverse_faces_;

  GMM gmm_;

  std::vector<torch::Tensor> reduce(std::vector<torch::Tensor>& in);

  void initialize_volume_weights();

  bool register_hand(unsigned int index, bool compute_rotation,
                     bool compute_translation,
                     reclib::modelstorch::ModelInstance<MANOTYPE>& hand);
  bool optimize_hand(unsigned int index,
                     reclib::modelstorch::ModelInstance<MANOTYPE>& hand,
                     reclib::modelstorch::Model<MANOTYPE>& hand_model,
                     int stage);
  void visualize_uncertainty(unsigned int index,
                             reclib::modelstorch::ModelInstance<MANOTYPE>& hand,
                             reclib::modelstorch::Model<MANOTYPE>& hand_model);
  void compute_correspondences(
      const CorrespondenceMode& mode, int index,
      reclib::modelstorch::ModelInstance<MANOTYPE>& hand);
  void prepare_correspondences(const CorrespondenceMode& mode,
                               const torch::Tensor& box,
                               const torch::Tensor& mask,
                               const torch::Tensor& corr,
                               CpuMat& input_output_depth, CpuMat& output_corr,
                               float& output_mean_depth);

  cv::Rect compute_crop_factor(const CpuMat& rgb);
  void detect(const CpuMat& rgb, cv::Rect crop_factor = cv::Rect(0, 0, -1, -1));
  void detect_openpose(unsigned int index, const CpuMat& rgb,
                       cv::Rect crop_factor);

 public:
  reclib::opengl::GroupedDrawelements gl_voxel_grid_;

  reclib::modelstorch::Model<MANOTYPE> mano_model_left_;
  reclib::modelstorch::Model<MANOTYPE> mano_model_right_;
  reclib::modelstorch::ModelInstance<MANOTYPE> mano_left_;
  reclib::modelstorch::ModelInstance<MANOTYPE> mano_right_;

  std::map<std::string, reclib::opengl::Drawelement> generated_drawelements_;

  ManoRGBD(const reclib::Configuration& config,
           const reclib::IntrinsicParameters& intrinsics, bool debug = false);

  unsigned int get_global_fc() { return global_frame_counter_; }
  void set_global_fc(unsigned int val) { global_frame_counter_ = val; }
  std::vector<bool> get_joints_visible(int index) {
    return last_frame_.joints_visible_[index];
  }
  std::vector<bool> get_joints_errorenous(int index) {
    return last_frame_.joints_errorenous_[index];
  }
  torch::Tensor get_kpts(int index) { return last_frame_.keypoints_[index]; }

  void fuse();
  void extract_mesh();
  void update_voxel_grid();
  void reset();
  void reset_right();
  void reset_left();
  void restore_right();
  void restore_left();
  void process(
      const CpuMat& rgb, const CpuMat& depth,
      const reclib::IntrinsicParameters& intrinsics = IntrinsicParameters(),
      cv::Rect crop_factor = cv::Rect(0, 0, -1, -1));
  void compute_correspondences(const CorrespondenceMode& mode);
  std::pair<bool, bool> register_hand(bool compute_rotation,
                                      bool compute_translation);
  std::pair<bool, bool> optimize_hand(int stage);
  void visualize_uncertainty();
  reclib::Configuration& config() { return config_; };
};
}  // namespace tracking
}  // namespace reclib

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE

#endif
