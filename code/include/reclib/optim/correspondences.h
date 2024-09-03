#ifndef RECLIB_OPTIM_CORRESPONDENCES_H
#define RECLIB_OPTIM_CORRESPONDENCES_H

#include <yaml-cpp/yaml.h>

#include <nanoflann.hpp>
#include <sophus/se3.hpp>

#include "reclib/camera_parameters.h"
#include "reclib/configuration.h"
#include "reclib/internal/filesystem.h"
#include "reclib/math/eigen_util_funcs.h"
#include "reclib/models/smpl.h"

namespace reclib {
namespace optim {

// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/darglein/saiga
//
// MIT License
// Copyright (c) 2021 Darius Rückert <darius.rueckert@fau.de>
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ---------------------------------------------------------------------------

template <typename T, int DIM, int INDEX_DIM = 1>
struct CorrespondenceBase {
  Eigen::Vector<T, DIM> refPoint;
  Eigen::Vector<T, DIM> refNormal;
  Eigen::Vector<T, DIM> srcPoint;
  Eigen::Vector<T, DIM> srcNormal;
  double weight = 1;

  Eigen::Vector<uint32_t, INDEX_DIM> srcIndex;
  Eigen::Vector<uint32_t, INDEX_DIM> refIndex;

  // Apply this transfomration to the src point and normal
  void apply(const Sophus::SE3<T>& trans) {
    srcPoint = trans * srcPoint;
    srcNormal = trans.so3() * srcNormal;
  }
  void apply(const mat4& trans) {
    mat4 trans_normal = mat4::Identity();
    trans_normal.block<3, 3>(0, 0) =
        trans.block<3, 3>(0, 0).inverse().transpose();

    srcPoint = (trans * srcPoint.homogeneous()).template head<3>();
    srcNormal = (trans_normal * srcNormal.homogeneous()).template head<3>();
  }

  void applyRef(const Sophus::SE3<T>& trans) {
    refPoint = trans * refPoint;
    refNormal = trans.so3() * refNormal;
  }

  void applyRef(const mat4& trans) {
    mat4 trans_normal = mat4::Identity();
    trans_normal.block<3, 3>(0, 0) =
        trans.block<3, 3>(0, 0).inverse().transpose();

    refPoint = (trans * refPoint.homogeneous()).template head<3>();
    refNormal = (trans_normal * refNormal.homogeneous()).template head<3>();
  }

  double residualPointToPoint() const {
    return (refPoint - srcPoint).squaredNorm();
  }

  double residualPointToPlane() const {
    double d = refNormal.dot(refPoint - srcPoint);
    return d * d;
  }
};
template <int DIM>
using Corr = CorrespondenceBase<double, DIM>;
template <int DIM>
using corr = CorrespondenceBase<float, DIM>;
using Corr3 = Corr<3>;
using corr3 = corr<3>;

// ---------------------------------------------------------------------------
// End of adapted code
// ---------------------------------------------------------------------------

// data type T, dimension of one point POINTS_DIM
template <typename T, int POINTS_DIM = 3>
struct PointCloud {
  const T* points_;
  uint32_t num_points_;  // number of points (num_floats / points_dim)
  const bool transform_{true};

  PointCloud()
      : points_(nullptr),
        num_points_(0),
        transform_(false),
        trans_(Eigen::Matrix<T, 4, 4>::Identity()){};

  PointCloud(const T* points, uint32_t num_points)
      : points_(points),
        num_points_(num_points),
        transform_(true),
        trans_(mat4::Identity()){};

  PointCloud(const T* points, uint32_t num_points,
             const Eigen::Matrix<T, 4, 4>& trans)
      : points_(points),
        num_points_(num_points),
        transform_(true),
        trans_(trans){};

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return num_points_; }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
    _RECLIB_ASSERT_LT(idx, num_points_);
    _RECLIB_ASSERT_LT(dim, POINTS_DIM);

    if (transform_) {
      Eigen::Map<const Eigen::Vector<T, POINTS_DIM>> pnt(
          &(points_[idx * POINTS_DIM]));

      return (trans_ * reclib::make_Vector3(pnt).homogeneous())[dim];
    }

    return points_[idx * POINTS_DIM + dim];
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

 private:
  Eigen::Matrix<T, 4, 4> trans_;
};

template <typename T, int DIM>
using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<T, PointCloud<T, DIM>>, PointCloud<T, DIM>,
    DIM /* dim */
    >;

struct CorrespondencesConfiguration : public reclib::Configuration {
  static const std::vector<std::string> YAML_PREFIXES;  // = {"Correspondences"}
  static const int NO_WEIGHTS;
  static const int SOURCE_WEIGHTS;
  static const int TARGET_WEIGHTS;
};

std::vector<CorrespondenceBase<float, 3>> nearestNeighborPointCorrespondences(
    const reclib::Configuration& config, const float* source_vertices,
    uint32_t source_size, const float* target_vertices, uint32_t target_size,
    const float* source_normals = nullptr,
    const float* target_normals = nullptr,
    const mat4& source_trans = mat4::Identity(),
    const mat4& target_trans = mat4::Identity(),
    const float* predefined_weights = nullptr, uint32_t weight_size = 0);

std::vector<CorrespondenceBase<float, 3>> nearestNeighborPointCorrespondences(
    const reclib::Configuration& config, const float* source_vertices,
    uint32_t source_size, const float* target_vertices, uint32_t target_size,
    const float* source_normals, const float* target_normals,
    const mat4& source_vertex_trans, const mat4& target_vertex_trans,
    const mat4& source_normal_trans, const mat4& target_normal_trans,
    const float* predefined_weights = nullptr, uint32_t weight_size = 0);

std::vector<CorrespondenceBase<float, 3>> projectivePointCorrespondences(
    const reclib::Configuration& config, const float* source_vertices,
    uint32_t source_size, const float* target_vertex_map,
    const reclib::ExtrinsicParameters& camera_extr,
    const reclib::IntrinsicParameters& camera_intr,
    const float* source_normals = nullptr,
    const float* target_normal_map = nullptr,
    const mat4& source_trans = mat4::Identity(),
    const mat4& target_trans = mat4::Identity());

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
canonicalPointCorrespondences(
    const reclib::Configuration& config, const float* source_vertices,
    uint32_t source_size, const float* target_vertices, uint32_t target_size,
    const float* source_canonicals, const float* target_canonicals);

template <typename MANOCONFIG>
std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
canonicalMANO2PointCorrespondences(
    const reclib::Configuration& config,
    reclib::models::ModelInstance<MANOCONFIG>& mano,
    const float* target_vertices, uint32_t target_size,
    const float* source_canonicals, const float* target_canonicals);

template <typename MANOCONFIG>
std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
nearestNeighborCanonicalMANO2PointCorrespondences(
    const reclib::Configuration& config,
    reclib::models::ModelInstance<MANOCONFIG>& mano,
    const float* target_vertices, uint32_t target_size,
    const float* source_canonicals, const float* target_canonicals);

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
nearestNeighborCanonicalPointCorrespondences(
    const reclib::Configuration& config, const float* source_vertices,
    uint32_t source_size, const float* target_vertices, uint32_t target_size,
    const float* source_canonicals, const float* target_canonicals);

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
nearestNeighborPointCorrespondences_(const reclib::Configuration& config,
                                     const float* source_vertices,
                                     uint32_t source_size,
                                     const float* target_vertices,
                                     uint32_t target_size);

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
nearestNeighborZeroDepthPointCorrespondences(
    const reclib::Configuration& config,
    const reclib::IntrinsicParameters& intrinsics, const float* source_vertices,
    uint32_t source_size, const float* target_vertex_map, uint32_t target_size);

}  // namespace optim
}  // namespace reclib

#endif