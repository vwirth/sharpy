#ifndef DYNAMIC_FUSION_CUDA_H2
#define DYNAMIC_FUSION_CUDA_H2

#if WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <Eigen/Eigen>

#if HAS_OPENCV_MODULE
#include <opencv2/core/cuda.hpp>
#endif

#include "reclib/camera_parameters.h"
#include "reclib/depth_processing.h"
#include "reclib/fusion/dynfu_types.h"
#include "reclib/fusion/kinectfusion.h"
#include "reclib/fusion/kinfu_types.h"
#include "reclib/math/quaternion.h"
#include "reclib/voxel.h"

namespace reclib {
namespace dynfu {

template <typename T>
RECLIB_HD inline T computeWarpWeight(const Eigen::Vector<T, 3>& point,
                                               const Eigen::Vector<T, 3>& node,
                                               T node_radius) {
  return exp(-(node - point).squaredNorm() /
             (T(2) * node_radius * node_radius));
}

namespace cuda {


#if HAS_OPENCV_MODULE
void EstimateTransformation(
    const GpuMat& frame_vertex_map, unsigned int iterations,
    const ivec3& grid_size, const float voxel_size,
    reclib::dynfu::Warpfield& warpfield, GpuMat& nn_voxel_grid,
    const CpuMat& warped_model_vmap, const CpuMat& warped_model_nmap,
    const mat4& current_pose,
    const reclib::ExtendedIntrinsicParameters& camera_parameters,
    float tukey_constant, float huber_constant, float lambda);

void GenerateKNNVoxelGrid(const ivec3& grid_size, const float voxel_scale,
                          const float knn, const CpuVec<vec3>& all_nodes,
                          const CpuVec<vec3>& new_nodes, GpuMat& nn_voxel_grid,
                          GpuMat& nn_voxel_dists);

void Warp(const reclib::kinfu::SurfaceMesh& input_mesh,
          const CpuVec<vec3>& warp_nodes,
          const CpuVec<reclib::DualQuaternion>& wn_transformations,
          const CpuVec<float> wn_radi, const GpuMat_<uint16_t>& nn_voxel_grid,
          const ivec3 grid_size, const float voxel_scale, const uint32_t knn,
          reclib::kinfu::SurfaceMesh& output_mesh);

void SurfaceReconstruction(
    const GpuMat& depth_image, const GpuMat& normal_map,
    const GpuMat& color_image, GpuMat& tsdf_volume, GpuMat& color_volume,
    const ivec3& grid_size, const float voxel_size,
    const reclib::IntrinsicParameters& cam_params,
    const CpuVec<vec3>& node_positions, const CpuVec<float>& node_radi,
    const CpuVec<reclib::DualQuaternion>& node_transformations,
    const GpuMat_<uint16_t>& nn_voxel_grid, const uint32_t knn,
    const float truncation_distance, const Eigen::Matrix4f& model_view,
    float max_weight);
#endif


}  // namespace cuda

}  // namespace dynfu
}  // namespace reclib

#endif
