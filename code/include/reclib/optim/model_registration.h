#if __unix__

#ifndef RECLIB_OPTIM_MODEL_REGISTRATION_H
#define RECLIB_OPTIM_MODEL_REGISTRATION_H

#include "reclib/data_types.h"
#include "reclib/fusion/tsdf_volume.h"
#include "reclib/models/model_config.h"
#include "reclib/models/smpl.h"
#include "reclib/optim/correspondences.h"
#include "reclib/optim/registration.h"
#include "sophus/se3.hpp"

namespace reclib {
namespace optim {

#if HAS_OPENCV_MODULE
// Special registration from mano hand to 3D point cloud
// with nearest neighbor correspondences
// and direct p2p optimization
mat4 registerMANO2Pointcloud(const reclib::Configuration& config,
                             const float* mano_vertices,
                             uint32_t mano_n_vertices, const float* pc_vertices,
                             uint32_t pc_n_vertices,
                             const float* mano_normals = nullptr,
                             const float* pc_normals = nullptr,
                             const Eigen::Matrix<float, 4, 4>& mano_pre_trans =
                                 Eigen::Matrix<float, 4, 4>::Identity(),
                             const Eigen::Matrix<float, 4, 4>& pc_pre_trans =
                                 Eigen::Matrix<float, 4, 4>::Identity(),
                             vec3* pca_mean = nullptr);

// More generalized registration from mano hand to 2D vertex map
// uses direct p2p optimization
Sophus::SE3<float> registerMANO2Pointcloud(
    reclib::Configuration config,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& mano,
    const float* pc_vertices, const reclib::ExtrinsicParameters& cam_extr,
    const reclib::IntrinsicParameters& cam_intr,
    const float* mano_normals = nullptr, const float* pc_normals = nullptr,
    const Eigen::Matrix<float, 4, 4>& mano_pre_trans =
        Eigen::Matrix<float, 4, 4>::Identity(),
    const Eigen::Matrix<float, 4, 4>& pc_pre_trans =
        Eigen::Matrix<float, 4, 4>::Identity(),
    vec3* pca_mean = nullptr, vec3* mano_mean = nullptr);
#endif  // HAS_OPENCV_MODULE

#if HAS_OPENCV_MODULE
template <class ModelConfig>
struct SMPL2SDFFunctor {
  const reclib::models::ModelInstance<ModelConfig>& instance;
  mat4 smpl_affine;
  unsigned int vertex_index;

  const CpuMat_<short2>& voxel_volume;
  float voxel_scale;
  ivec3 grid_size;

  template <typename T>
  bool operator()(const T* const shape, const T* const pose_incr,
                  T* residual) const {
    Eigen::Map<const Eigen::Vector<T, ModelConfig::n_shape_blends()>> shape_map(
        shape);
    Eigen::Map<const Eigen::Vector<T, ModelConfig::n_explicit_joints() * 3>>
        pose_incr_map(pose_incr);

    Eigen::Matrix<T, 1, 3> verts;
    LBS<T, ModelConfig>(instance.model, shape_map, pose_incr_map,
                        instance.pose().template cast<T>(), verts, true,
                        vertex_index);

    Eigen::Vector<T, 3> vertex =
        (smpl_affine.cast<T>() * verts.transpose().homogeneous())
            .template head<3>();

    Eigen::Vector<T, 3> vertex_in_grid = vertex / T(voxel_scale);
    T x = vertex_in_grid[0];
    T y = vertex_in_grid[1];
    T z = vertex_in_grid[2];
    ivec3 grid_index(get_integer_part(x), get_integer_part(y),
                     get_integer_part(z));
    T res = T(0);
    bool has_tsdf = reclib::interpolate_trilinearly<T>(
        vertex_in_grid, voxel_volume, grid_size, res, grid_index);

    if (has_tsdf) {
      residual[0] = res * res;
    } else {
      residual[0] = T(2);
    }

    return true;
  }
};

// Localizes the mean of the hand shape
// Underlying assumption: the point cloud is an arm
// When computing the mean of an arm, it is typically biased by the length
// of the arm itself
// For Hand registration we need the mean of the hand itself
vec3 PCAHandMean(const float* pc_vertices, uint32_t pc_n_vertices,
                 float variance_thresh = 0.8);

#endif  // HAS_OPENCV_MODULE

}  // namespace optim
}  // namespace reclib

#endif

#endif  //__unix__