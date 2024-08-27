#ifndef RECLIB_MANO_RGBD_CUH
#define RECLIB_MANO_RGBD_CUH

#if HAS_OPENCV_MODULE
#if WITH_CUDA
#include "reclib/camera_parameters.h"
#include "reclib/data_types.h"
#include "reclib/fusion/kinfu_types.h"

namespace reclib {
namespace tracking {
namespace cuda {
void compute_segmentation(GpuMat& input, std::vector<uint8_t>& visible_ids);
void compute_segmentation(GpuVec<vec3>& input,
                          std::vector<uint8_t>& visible_ids);
}  // namespace cuda

namespace ManoRGBD_Cuda {

void initialize_volume_weights(GpuMat& tsdf_weights, uvec3 size, float scale,
                               vec3 initial_trans, const float* mano_verts,
                               uint32_t num_verts);

void update_voxel_pos(GpuMat& voxel_pos, GpuMat& tsdf_weights, uvec3 size,
                      float scale, vec3 initial_trans,
                      const float* mano_verts_transforms, uint32_t num_verts);

void fuse_surface(GpuMat& pixels, const GpuMat& depth_map,
                  const GpuMat& normal_map, const uvec2 pixel_offset,
                  const GpuMat& tsdf_values, const GpuMat& tsdf_weights,
                  uvec3 size, float scale,
                  const reclib::IntrinsicParameters& cam_params,
                  vec3 initial_trans, const float* mano_verts_transforms,
                  uint32_t num_verts, float truncation_distance,
                  float max_weight);

reclib::kinfu::SurfaceMesh marching_cubes(const GpuMat& tsdf_volume,
                                          const uvec3 size,
                                          const float voxel_size,
                                          const int triangles_buffer_size);

}  // namespace ManoRGBD_Cuda
}  // namespace tracking
}  // namespace reclib

#endif  // WITH_CUDA
#endif  // HAS_OPENCV_MODULE

#endif