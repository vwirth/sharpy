
#ifndef DYNAMICFUSION_DATA_TYPES_H
#define DYNAMICFUSION_DATA_TYPES_H

#if WITH_CUDA
  #include <cuda_runtime.h>
  #include <thrust/device_vector.h>
  #include <thrust/host_vector.h>
#endif

#include <Eigen/Eigen>
#include <iterator>
#include <nanoflann.hpp>

#if HAS_OPENCV_MODULE
  #include <opencv2/core/cuda.hpp>
#endif

#include "reclib/assert.h"
#include "reclib/camera_parameters.h"
#include "reclib/depth_processing.h"
#include "reclib/math/exp_coords.h"
#include "reclib/math/quaternion.h"
#include "reclib/voxel.h"

namespace reclib {
namespace dynfu {


#if HAS_OPENCV_MODULE
struct MatWrapper {
  const CpuMat& points_;

  MatWrapper(const CpuMat& points) : points_(points){};

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return points_.cols; }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return (points_).ptr<float>(0)[3 * idx + 0];
    else if (dim == 1)
      return (points_).ptr<float>(0)[3 * idx + 1];
    else
      return (points_).ptr<float>(0)[3 * idx + 2];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }
};


struct PointCloud {
  CpuVec<vec3> points_;

  PointCloud() : points_(0){};
  PointCloud(CpuVec<vec3> points) : points_(points){};

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return points_.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (idx >= points_.size()) {
      std::cout << "get index: " << idx << " dim: " << dim
                << " size: " << points_.size() << std::endl;
      exit(0);
    }

    if (dim == 0)
      return (points_)[idx].x();
    else if (dim == 1)
      return (points_)[idx].y();
    else
      return (points_)[idx].z();
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }

  inline auto size() const { return points_.size(); }

  void insert(CpuVec<vec3> input) {
    points_.insert(points_.end(), input.begin(), input.end());
  }

  void insert(CpuMat input) {
    points_.insert(points_.end(), input.ptr<vec3>(0),
                   input.ptr<vec3>(0) + input.rows);
  }

  CpuVec<vec3>::iterator begin() { return points_.begin(); }

  CpuVec<vec3>::iterator end() { return points_.end(); }

  CpuVec<vec3>& points() { return points_; }

  vec3 operator[](const int i) const { return points_[i]; }
};

typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3 /* dim */
    >
    KdTree_t;


class _API Warpfield {
 public:
  static const unsigned int kRadiusSearchNN;
  const float kMinRadius;
  const unsigned int kDQBNN;
  const float kMaxLeafSize;
  const unsigned int kRegularizationLevel;

  std::vector<PointCloud> node_positions_;
  std::vector<CpuVec<float>> node_radi_;
  std::vector<CpuVec<reclib::DualQuaternion>> node_transformations_;
  std::vector<unsigned int> num_nodes_;

  std::vector<std::shared_ptr<KdTree_t>> reg_index_;
  std::vector<CpuVec<CpuVec<unsigned int>>> reg_edges_;

  Warpfield(const float min_radius, const unsigned int knn,
            const float max_leaf_size, const unsigned int reg_graph_levels)
      : kMinRadius(min_radius),
        kDQBNN(knn),
        kMaxLeafSize(max_leaf_size),
        kRegularizationLevel(reg_graph_levels),
        node_positions_(kRegularizationLevel),
        node_radi_(kRegularizationLevel),
        node_transformations_(kRegularizationLevel),
        num_nodes_(kRegularizationLevel),
        reg_edges_(kRegularizationLevel - 1) {
    for (unsigned int i = 0; i < kRegularizationLevel; i++) {
      node_positions_[i] = PointCloud();
      reg_index_.push_back(std::make_shared<KdTree_t>(
          3, node_positions_[i],
          nanoflann::KDTreeSingleIndexAdaptorParams(kMaxLeafSize)));
    }
  };
  ~Warpfield(){};

  inline void RebuildRegularizationGraph() {
    for (unsigned int i = 1; i < kRegularizationLevel; i++) {
      node_positions_[i].points().clear();
      node_radi_[i].clear();
      node_transformations_[i].clear();
      reg_index_[i].reset();
      reg_index_[i] = std::make_shared<KdTree_t>(
          3, node_positions_[i],
          nanoflann::KDTreeSingleIndexAdaptorParams(kMaxLeafSize));

      reg_edges_[i - 1].clear();
      num_nodes_[i] = 0;
    }
  }

  inline void AddRegNodes(const unsigned int level, const CpuVec<vec3>& pos,
                          const CpuVec<float>& weights,
                          const CpuVec<reclib::DualQuaternion>& trans) {
    node_positions_[level].insert(pos);
    node_radi_[level].insert(node_radi_[level].end(), weights.begin(),
                             weights.end());
    node_transformations_[level].insert(node_transformations_[level].end(),
                                        trans.begin(), trans.end());

    reg_index_[level]->addPoints(node_positions_[level].size() - pos.size(),
                                 node_positions_[level].size() - 1);

    if (level > 0) {
      std::cout << "added " << pos.size()
                << " new regularization nodes at level " << level << std::endl;
    } else {
      std::cout << "added " << pos.size() << " new nodes" << std::endl;
    }

    num_nodes_[level] += pos.size();
  }

  inline KdTree_t& index() { return *reg_index_[0]; }

  inline KdTree_t& reg_index(unsigned int level) { return *reg_index_[level]; }

  inline unsigned int warpsize() { return node_radi_[0].size(); }

  float GetNodeWeight(const vec3& pos, unsigned int index) {
    vec3 pnt = node_positions_[0].points()[index];
    float l2_diff = (pnt - pos).squaredNorm();
    return expf(-l2_diff / (2.f * pow(node_radi_[0][index], 2)));
  }

  reclib::DualQuaternion GetNodetransformation(unsigned int index) {
    return node_transformations_[0][index];
  }
};
#endif

}  // namespace dynfu

}  // namespace reclib

#endif
