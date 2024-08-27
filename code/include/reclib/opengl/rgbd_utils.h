#ifndef RECLIB_OPENGL_RGBD_UTILS_H
#define RECLIB_OPENGL_RGBD_UTILS_H

#include "reclib/camera_parameters.h"
#include "reclib/data_types.h"
#include "reclib/opengl/mesh.h"

#if HAS_OPENCV_MODULE
namespace reclib {
/// @brief
namespace opengl {

CpuMat fill_depth_nearest_neighbor(CpuMat depth, float random_noise_scale = 0);

// generates point cloud from depth image with normal and color attribute
reclib::opengl::Mesh pointcloud_norm_color(
    const CpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    const CpuMat& rgb, bool pinhole2opengl_normals = false,
    const std::string& name = "pc_norm_col",
    const CpuMat& xyz = CpuMat(0, 0, CV_32FC1),
    const ivec2& xy_offset = ivec2(0, 0), const float scale = 1.f);
// generates triangle mesh from depth image with normal and color attribute
reclib::opengl::Mesh triangle_mesh_norm_color(
    const CpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    const CpuMat& rgb, bool pinhole2opengl_normals = false,
    const std::string& name = "pc_norm_col",
    const CpuMat& xyz = CpuMat(0, 0, CV_32FC1),
    const ivec2& xy_offset = ivec2(0, 0), const float scale = 1.f);
// generates point cloud from depth image with normal attribute
reclib::opengl::Mesh pointcloud_norm(
    const CpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    bool pinhole2opengl_normals = false, const std::string& name = "pc_norm",
    const CpuMat& xyz = CpuMat(0, 0, CV_32FC1),
    const ivec2& xy_offset = ivec2(0, 0), const float scale = 1.f);

// generates point cloud from depth image with normal and color attribute
reclib::opengl::Mesh pointcloud_norm_color(
    const CpuMat& vertex_map, const CpuMat& normal_map,
    bool pinhole2opengl_normals = false,
    const std::string& name = "pc_norm_col",
    const ivec2& xy_offset = ivec2(0, 0), const float scale = 1.f);

#if WITH_CUDA

namespace cuda {
// generates point cloud from depth image with normal and
// color attribute
reclib::opengl::Mesh pointcloud_norm_color(
    const GpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    const GpuMat& rgb, bool pinhole2opengl_normals = false,
    const std::string& name = "pc_norm_col",
    const GpuMat& xyz = GpuMat(0, 0, CV_32FC1),
    const ivec2& xy_offset = ivec2(0, 0), const float scale = 1.f);
// generates point cloud from depth image with normal attribute
reclib::opengl::Mesh pointcloud_norm(
    const GpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    bool pinhole2opengl_normals = false, const std::string& name = "pc_norm",
    const GpuMat& xyz = GpuMat(0, 0, CV_32FC1),
    const ivec2& xy_offset = ivec2(0, 0), const float scale = 1.f);

void fill_depth_nearest_neighbor(const GpuMat depth_in, GpuMat depth_out,
                                 float random_noise_scale);

}  // namespace cuda

#endif

}  // namespace opengl
}  // namespace reclib

#endif

#endif