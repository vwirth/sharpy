
#ifndef KINECTFUSION_DATA_TYPES_H
#define KINECTFUSION_DATA_TYPES_H

#if WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include "reclib/assert.h"

// clang-format off
#include <Eigen/Eigen>
// #include <gvdb.h>
// clang-format on

#include <iterator>
#include <nanoflann.hpp>

#if HAS_OPENCV_MODULE
#include <opencv2/core/cuda.hpp>
#endif

#include "reclib/camera_parameters.h"
#include "reclib/depth_processing.h"
#include "reclib/math/exp_coords.h"
#include "reclib/math/quaternion.h"
#include "reclib/voxel.h"

namespace reclib {
namespace kinfu {


#if HAS_OPENCV_MODULE
// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/chrdiller/KinectFusionLib
// ---------------------------------------------------------------------------

/**
 * \brief Representation of a cloud of three-dimensional points (vertices)
 *
 * This data structure contains
 * (1) the world coordinates of the vertices,
 * (2) the corresponding normals and
 * (3) the corresponding RGB color value
 *
 * - vertices: A 1 x buffer_size opencv Matrix with CV_32FC3 values,
 * representing the coordinates
 * - normals: A 1 x buffer_size opencv Matrix with CV_32FC3 values,
 * representing the normal direction
 * - color: A 1 x buffer_size opencv Matrix with CV_8UC3 values, representing
 * the RGB color
 *
 * Same indices represent the same point
 *
 * The total number of valid points in those buffers is stored in num_points
 *
 */
struct PointCloud {
  // World coordinates of all vertices
  CpuMat vertices_;
  // Normal directions
  CpuMat normals_;
  // RGB color values
  CpuMat color_;

  // Total number of valid points
  int num_points_;
};

/**
 * \brief Representation of a dense surface mesh
 *
 * This data structure contains
 * (1) the mesh triangles (triangular faces) and
 * (2) the colors of the corresponding vertices
 *
 * - triangles: A 1 x num_vertices opencv Matrix with CV_32FC3 values,
 * representing the coordinates of one vertex; a sequence of three vertices
 * represents one triangle
 * - colors: A 1 x num_vertices opencv Matrix with CV_8Uc3 values,
 * representing the RGB color of each vertex
 *
 * Same indices represent the same point
 *
 * Total number of vertices stored in num_vertices, total number of triangles in
 * num_triangles
 *
 */
struct SurfaceMesh {
  // Triangular faces
  CpuMat triangles_;
  // Per-vertex normals
  CpuMat normals_;
  // Colors of the vertices
  CpuMat colors_;

  // Total number of vertices
  int num_vertices_;
  // Total number of triangles
  int num_triangles_;

  void clear() {
    triangles_.release();
    normals_.release();
    colors_.release();
    num_vertices_ = 0;
    num_triangles_ = 0;
  }
};

/*
 * Contains the internal data representation of one single frame as read by the
 * depth camera Consists of depth, smoothed depth and color pyramids as well as
 * vertex and normal pyramids
 */
struct FrameData {
  std::vector<GpuMat> depth_pyramid_;
  std::vector<GpuMat> smoothed_depth_pyramid_;
  std::vector<GpuMat> color_pyramid_;

  std::vector<GpuMat> vertex_pyramid_;
  std::vector<GpuMat> normal_pyramid_;

  FrameData(){};

  FrameData(const size_t pyramid_height)
      : depth_pyramid_(pyramid_height),
        smoothed_depth_pyramid_(pyramid_height),
        color_pyramid_(pyramid_height),
        vertex_pyramid_(pyramid_height),
        normal_pyramid_(pyramid_height) {}

  // No copying
  FrameData(const FrameData&) = delete;
  FrameData& operator=(const FrameData& other) = delete;

  FrameData(FrameData&& data) noexcept
      : depth_pyramid_(std::move(data.depth_pyramid_)),
        smoothed_depth_pyramid_(std::move(data.smoothed_depth_pyramid_)),
        color_pyramid_(std::move(data.color_pyramid_)),
        vertex_pyramid_(std::move(data.vertex_pyramid_)),
        normal_pyramid_(std::move(data.normal_pyramid_)) {}

  FrameData& operator=(FrameData&& data) noexcept {
    depth_pyramid_ = std::move(data.depth_pyramid_);
    smoothed_depth_pyramid_ = std::move(data.smoothed_depth_pyramid_);
    color_pyramid_ = std::move(data.color_pyramid_);
    vertex_pyramid_ = std::move(data.vertex_pyramid_);
    normal_pyramid_ = std::move(data.normal_pyramid_);
    return *this;
  }

  std::vector<CpuMat> DownloadDepthPyramid();
  std::vector<CpuMat> DownloadSmoothedDepthPyramid();
  std::vector<CpuMat> DownloadColorPyramid();
  std::vector<CpuMat> DownloadVertexPyramid();
  std::vector<CpuMat> DownloadNormalPyramid();
};

/*
 * Contains the internal data representation of one single frame as raycast by
 * surface prediction Consists of depth, smoothed depth and color pyramids as
 * well as vertex and normal pyramids
 */
struct ModelData {
  std::vector<GpuMat> vertex_pyramid_;
  std::vector<GpuMat> normal_pyramid_;
  std::vector<GpuMat> color_pyramid_;

  ModelData(const size_t pyramid_height,
            const IntrinsicParameters camera_parameters)
      : vertex_pyramid_(pyramid_height),
        normal_pyramid_(pyramid_height),
        color_pyramid_(pyramid_height) {
    for (size_t level = 0; level < pyramid_height; ++level) {
      vertex_pyramid_[level] = cv::cuda::createContinuous(
          camera_parameters.Level(level).image_height_,
          camera_parameters.Level(level).image_width_, CV_32FC3);
      normal_pyramid_[level] = cv::cuda::createContinuous(
          camera_parameters.Level(level).image_height_,
          camera_parameters.Level(level).image_width_, CV_32FC3);
      color_pyramid_[level] = cv::cuda::createContinuous(
          camera_parameters.Level(level).image_height_,
          camera_parameters.Level(level).image_width_, CV_8UC3);
      vertex_pyramid_[level].setTo(0);
      normal_pyramid_[level].setTo(0);
    }
  }

  // No copying
  ModelData(const ModelData&) = delete;
  ModelData& operator=(const ModelData& data) = delete;

  ModelData(ModelData&& data) noexcept
      : vertex_pyramid_(std::move(data.vertex_pyramid_)),
        normal_pyramid_(std::move(data.normal_pyramid_)),
        color_pyramid_(std::move(data.color_pyramid_)) {}

  ModelData& operator=(ModelData&& data) noexcept {
    vertex_pyramid_ = std::move(data.vertex_pyramid_);
    normal_pyramid_ = std::move(data.normal_pyramid_);
    color_pyramid_ = std::move(data.color_pyramid_);
    return *this;
  }
};

/*
 * \brief Contains the internal volume representation
 *
 * This internal representation contains two volumes:
 * (1) TSDF volume: The global volume used for depth frame fusion and
 * (2) Color volume: Simple color averaging for colorized vertex output
 *
 * It also contains two important parameters:
 * (1) Volume size: The x, y and z dimensions of the volume (in mm)
 * (2) Voxel scale: The scale of a single voxel (in mm)
 *
 */
class _API VolumeData {
 public:
  // template <typename T>

  const ivec3& GridSize() const { return tsdf_volume_.GridSize(); }

  float VoxelSize() const { return tsdf_volume_.VoxelSize(); }

  VolumeData(const ivec3 volume_size, const float voxel_scale)
      : tsdf_volume_(volume_size, voxel_scale, CV_16SC2),
        color_volume_(volume_size, voxel_scale, CV_8UC3) {}

  reclib::ArrayVoxelGrid<cv::Vec2s> tsdf_volume_;   // short2
  reclib::ArrayVoxelGrid<cv::Vec3b> color_volume_;  // uchar3
};

/*
 * \brief Contains the internal pointcloud representation
 *
 * This is only used for exporting the data kept in the internal volumes
 *
 * It holds GPU containers for vertices, normals and vertex colors
 * It also contains host containers for this data and defines the total number
 * of points
 *
 */
struct CloudData {
  GpuMat vertices_;
  GpuMat normals_;
  GpuMat color_;

  CpuMat host_vertices_;
  CpuMat host_normals_;
  CpuMat host_color_;

  int* point_num_;
  int host_point_num_;

  explicit CloudData(const int max_number)
      : vertices_{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
        normals_{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
        color_{cv::cuda::createContinuous(1, max_number, CV_8UC3)},

        point_num_{nullptr},
        host_point_num_{} {
    vertices_.setTo(0.F);
    normals_.setTo(0.F);
    color_.setTo(0.F);

    cudaMalloc(&point_num_, sizeof(int));
    cudaMemset(point_num_, 0, sizeof(int));
  }

  // No copying
  CloudData(const CloudData&) = delete;
  CloudData& operator=(const CloudData& data) = delete;

  void Download() {
    vertices_.download(host_vertices_);
    normals_.download(host_normals_);
    color_.download(host_color_);

    cudaMemcpy(&host_point_num_, point_num_, sizeof(int),
               cudaMemcpyDeviceToHost);
  }
};

/*
 * \brief Contains the internal surface mesh representation
 *
 * This is only used for exporting the data kept in the internal volumes
 *
 * It holds several GPU containers needed for the MarchingCubes algorithm
 *
 */
struct MeshData {
  GpuMat occupied_voxel_ids_buffer_;
  GpuMat number_vertices_buffer_;
  GpuMat vertex_offsets_buffer_;
  GpuMat triangle_buffer_;
  GpuMat normal_buffer_;

  GpuMat occupied_voxel_ids_;
  GpuMat number_vertices_;
  GpuMat vertex_offsets_;

  explicit MeshData(const int buffer_size)
      : occupied_voxel_ids_buffer_{cv::cuda::createContinuous(1, buffer_size,
                                                              CV_32SC1)},
        number_vertices_buffer_{
            cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
        vertex_offsets_buffer_{
            cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
        triangle_buffer_{
            cv::cuda::createContinuous(1, buffer_size * 3, CV_32FC3)},
        normal_buffer_{
            cv::cuda::createContinuous(1, buffer_size * 3, CV_32FC3)},
        occupied_voxel_ids_{cv::cuda::createContinuous(0, 0, CV_32FC3)},
        number_vertices_{cv::cuda::createContinuous(0, 0, CV_32FC3)},
        vertex_offsets_{cv::cuda::createContinuous(0, 0, CV_32FC3)} {
    occupied_voxel_ids_buffer_.setTo(0);
    number_vertices_buffer_.setTo(0);
    vertex_offsets_buffer_.setTo(0);
    triangle_buffer_.setTo(0);
    normal_buffer_.setTo(0);
    occupied_voxel_ids_.setTo(0);
    number_vertices_.setTo(0);
    vertex_offsets_.setTo(0);
  }

  void Clear() {
    occupied_voxel_ids_buffer_.release();
    number_vertices_buffer_.release();
    vertex_offsets_buffer_.release();
    triangle_buffer_.release();
    normal_buffer_.release();

    occupied_voxel_ids_.release();
    number_vertices_.release();
    vertex_offsets_.release();
  }

  void CreateView(const int length) {
    occupied_voxel_ids_ =
        GpuMat(1, length, CV_32SC1, occupied_voxel_ids_buffer_.ptr<int>(0),
               occupied_voxel_ids_buffer_.step);
    number_vertices_ =
        GpuMat(1, length, CV_32SC1, number_vertices_buffer_.ptr<int>(0),
               number_vertices_buffer_.step);
    vertex_offsets_ =
        GpuMat(1, length, CV_32SC1, vertex_offsets_buffer_.ptr<int>(0),
               vertex_offsets_buffer_.step);
  }
};

// ---------------------------------------------------------------------------
// End of adapted code
// ---------------------------------------------------------------------------
#endif

}  // namespace kinfu

}  // namespace reclib

#endif
