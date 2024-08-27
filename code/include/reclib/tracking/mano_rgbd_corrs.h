
#ifndef RECLIB_MANO_RGBD_CORRS_H
#define RECLIB_MANO_RGBD_CORRS_H

#include <yaml-cpp/yaml.h>

#include "reclib/benchmark/mano.h"
#include "reclib/configuration.h"
#include "reclib/data_types.h"
#include "reclib/dnn/dnn.h"
#include "reclib/dnn/yolact.h"
#include "reclib/models/smpl.h"
#include "reclib/models/smpl_torch.h"
#include "reclib/optim/correspondences.h"
#include "reclib/tracking/mano_rgbd.cuh"
#include "reclib/voxel.h"

#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

namespace reclib {
namespace tracking {

class ManoRGBDCorrespondences {
 private:
  struct MANO {
    std::vector<unsigned int> canonical_ind_;
    std::vector<unsigned int> silhouette_ind_;

    void clear() {
      canonical_ind_.clear();
      silhouette_ind_.clear();
    }
  };

  struct PC {
    std::vector<unsigned int> canonical_ind_;
    std::vector<unsigned int> silhouette_ind_;

    PC() {}

    void clear() {
      canonical_ind_.clear();
      silhouette_ind_.clear();
    }
  };

  std::vector<vec2> valid_px_coords_;
  std::vector<vec3> valid_canonical_coords_;
  std::map<unsigned int, unsigned int> wrist2valid_;
  std::map<unsigned int, unsigned int> thumb2valid_;
  std::map<unsigned int, unsigned int> index2valid_;
  std::map<unsigned int, unsigned int> middle2valid_;
  std::map<unsigned int, unsigned int> pinky2valid_;
  std::map<unsigned int, unsigned int> ring2valid_;
  std::vector<vec3> c_wrist_;
  std::vector<vec3> c_thumb_;
  std::vector<vec3> c_index_;
  std::vector<vec3> c_middle_;
  std::vector<vec3> c_pinky_;
  std::vector<vec3> c_ring_;
  std::unordered_map<unsigned int, unsigned int> target_to_source_;
  std::unordered_map<unsigned int, unsigned int> target_to_source_silhouette_;
  // std::unordered_map<unsigned int, unsigned int> source_to_target_;

  reclib::optim::PointCloud<float> pc_wrapper_wrist_;
  reclib::optim::PointCloud<float> pc_wrapper_thumb_;
  reclib::optim::PointCloud<float> pc_wrapper_index_;
  reclib::optim::PointCloud<float> pc_wrapper_middle_;
  reclib::optim::PointCloud<float> pc_wrapper_pinky_;
  reclib::optim::PointCloud<float> pc_wrapper_ring_;
  // reclib::optim::PointCloud<float> pc_wrapper_;
  reclib::optim::PointCloud<float, 2> pc_wrapper_px_;
  // reclib::optim::KdTree<float, 3> kdtree_;

  reclib::optim::KdTree<float, 3> kdtree_wrist_;
  reclib::optim::KdTree<float, 3> kdtree_thumb_;
  reclib::optim::KdTree<float, 3> kdtree_index_;
  reclib::optim::KdTree<float, 3> kdtree_mid_;
  reclib::optim::KdTree<float, 3> kdtree_pinky_;
  reclib::optim::KdTree<float, 3> kdtree_ring_;

  reclib::optim::KdTree<float, 2> kdtree_px_;

  bool initialized_;

 public:
  std::vector<bool> joints_visible_;
  bool comp_canonical_corr_;
  bool comp_silhouette_corr_;

  std::vector<float> canonical_weights_;
  MANO mano_corr_;
  PC pc_corr_;

  ManoRGBDCorrespondences(bool canonical_corr = true,
                          bool silhouette_corr = true)
      :  // kdtree_(3, pc_wrapper_, {32 /* max leaf*/}),
        kdtree_wrist_(3, pc_wrapper_wrist_, {32 /* max leaf*/}),
        kdtree_thumb_(3, pc_wrapper_thumb_, {32 /* max leaf*/}),
        kdtree_index_(3, pc_wrapper_index_, {32 /* max leaf*/}),
        kdtree_mid_(3, pc_wrapper_middle_, {32 /* max leaf*/}),
        kdtree_pinky_(3, pc_wrapper_pinky_, {32 /* max leaf*/}),
        kdtree_ring_(3, pc_wrapper_ring_, {32 /* max leaf*/}),
        kdtree_px_(2, pc_wrapper_px_, {32 /* max leaf*/}),
        initialized_(false),
        joints_visible_(16),
        comp_canonical_corr_(canonical_corr),
        comp_silhouette_corr_(silhouette_corr) {}

  void reset() { initialized_ = false; }

  void compute(const reclib::Configuration& config, const float* mano_verts,
               const uint32_t num_verts, const float* mano_normals,
               const Eigen::Vector<int, Eigen::Dynamic>& mano_verts2joints,
               const std::vector<vec3>& mano_canonicals, const CpuMat& pc,
               const CpuMat& pc_normals, const CpuMat& pc_canonicals,
               const CpuMat& pc_segmentation,
               const reclib::IntrinsicParameters& intrinsics,
               const ivec2& box_offset, const float mean_depth);

  void compute_points2mano(
      const reclib::Configuration& config, const float* mano_verts,
      const uint32_t num_verts,
      const Eigen::Vector<int, Eigen::Dynamic>& mano_verts2joints,
      const std::vector<vec3>& mano_canonicals, const CpuMat& pc,
      const CpuMat& pc_normals, const CpuMat& pc_canonicals,
      const CpuMat& pc_segmentation,
      const reclib::IntrinsicParameters& intrinsics, const ivec2& box_offset,
      const float mean_depth);
};

}  // namespace tracking
}  // namespace reclib

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE

#endif
