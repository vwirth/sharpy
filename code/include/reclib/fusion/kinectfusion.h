#ifndef KINECTFUSION_H
#define KINECTFUSION_H

#include <Eigen/Eigen>

#if HAS_OPENCV_MODULE
#include <opencv2/core/cuda.hpp>
#endif

#include "reclib/camera_parameters.h"
#include "reclib/depth_processing.h"
#include "reclib/fusion/kinfu_types.h"
#include "reclib/internal/filesystem.h"
#include "reclib/utils/yaml.h"
#include "reclib/voxel.h"

namespace reclib {
namespace kinfu {

// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/chrdiller/KinectFusionLib

// MIT License
// Copyright (c) 2018 Christian Diller
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

/**
 * \brief The global configuration
 *
 * This data structure contains several parameters that control the workflow of
 * the overall pipeline. Most of them are based on the KinectFusion paper
 *
 * For an explanation of a specific parameter, see the corresponding comment
 *
 * The struct is preset with some default values so that you can use the
 * configuration without modification. However, you will probably want to adjust
 * most of them to your specific use case.
 *
 * Spatial parameters are always represented in millimeters (mm).
 *
 */
struct Configuration {
  // The overall size of the volume (in voxel units). Will be allocated on the
  // GPU and is thus limited by the amount of storage you have available.
  // Dimensions are (x, y, z).
  // the overall volume size in millimeters is
  // volume_size * voxel_scale
  // BEWARE: MUST BE A MULTIPLE OF 32
  ivec3 volume_size_{512, 512, 512};
  // ivec3 volume_size_{256, 256, 256};
  // ivec3 volume_size_{64, 64, 64};

  // The amount of mm one single voxel will represent in each dimension.
  // Controls the resolution of the volume.
  float voxel_scale_{2.0F};  // in millimeters

  // Parameters for the Bilateral Filter, applied to incoming depth frames.
  // Directly passed to cv::cuda::bilateralFilter(...); for further
  // information, have a look at the opencv docs.
  int bfilter_kernel_size_{5};        // 5
  float bfilter_color_sigma_{1.F};    // 1.F
  float bfilter_spatial_sigma_{1.F};  // 1.F

  // The initial distance of the camera from the volume center along the z-axis
  // (in mm)
  float init_depth_{1000.F};  // 1000.F

  // Downloads the model frame for each frame (for visualization purposes). If
  // this is set to true, you can retrieve the frame with
  // Pipeline::get_last_model_frame()
  bool use_output_frame_ = {true};

  // The truncation distance for both updating and raycasting the TSDF volume
  float truncation_distance_{25.F};

  // The distance (in mm) after which to set the depth in incoming depth frames
  // to 0. Can be used to separate an object you want to scan from the
  // background
  float depth_cutoff_distance_{1500.F};

  // The number of pyramid levels to generate for each frame, including the
  // original frame level
  int num_levels_{3};

  // The maximum buffer size for exporting triangles; adjust if you run out of
  // memory when exporting
  int triangles_buffer_size_{3 * 2000000};
  // The maximum buffer size for exporting pointclouds; adjust if you run out of
  // memory when exporting
  int pointcloud_buffer_size_{3 * 2000000};

  // ICP configuration
  // The distance threshold (as described in the paper) in mm
  float distance_threshold_{10.F};
  // The angle threshold (as described in the paper) in degrees
  float angle_threshold_{20.F};
  // Number of ICP iterations for each level from original level 0 to highest
  // scaled level (sparse to coarse)
  std::vector<int> icp_iterations_{10, 5, 4};

  // maximum weight value for a voxel
  float voxel_max_weight_{128.f};

  static Configuration from_file(std::filesystem::path& filepath);
  void to_file(std::filesystem::path& filepath);

  friend std::ostream& operator<<(std::ostream& os, const Configuration& q) {
    os << "Configuration:" << std::endl;
    os << std::setw(30) << "volume_size_:" << std::setw(30) << q.volume_size_
       << std::endl;
    os << std::setw(30) << "voxel_scale_:" << std::setw(30) << q.voxel_scale_
       << std::endl;
    os << std::setw(30) << "bfilter_kernel_size_:" << std::setw(30)
       << q.bfilter_kernel_size_ << std::endl;
    os << std::setw(30) << "bfilter_color_sigma_:" << std::setw(30)
       << q.bfilter_color_sigma_ << std::endl;
    os << std::setw(30) << "bfilter_spatial_sigma_:" << std::setw(30)
       << q.bfilter_spatial_sigma_ << std::endl;
    os << std::setw(30) << "init_depth_:" << std::setw(30) << q.init_depth_
       << std::endl;
    os << std::setw(30) << "use_output_frame_:" << std::setw(30)
       << q.use_output_frame_ << std::endl;
    os << std::setw(30) << "truncation_distance_:" << std::setw(30)
       << q.truncation_distance_ << std::endl;
    os << std::setw(30) << "depth_cutoff_distance_:" << std::setw(30)
       << q.depth_cutoff_distance_ << std::endl;
    os << std::setw(30) << "num_levels_:" << std::setw(30) << q.num_levels_
       << std::endl;
    os << std::setw(30) << "triangles_buffer_size_:" << std::setw(30)
       << q.triangles_buffer_size_ << std::endl;
    os << std::setw(30) << "pointcloud_buffer_size_:" << std::setw(30)
       << q.pointcloud_buffer_size_ << std::endl;
    os << std::setw(30) << "distance_threshold_:" << std::setw(30)
       << q.distance_threshold_ << std::endl;
    os << std::setw(30) << "angle_threshold_:" << std::setw(30)
       << q.angle_threshold_ << std::endl;
    os << std::setw(30) << "icp_iterations_:" << std::setw(30)
       << q.icp_iterations_.size() << std::endl;
    os << std::setw(30) << "voxel_max_weight_:" << std::setw(30)
       << q.voxel_max_weight_ << std::endl;

    return os;
  }
};

#if HAS_OPENCV_MODULE
/*
 *
 * \brief This is the KinectFusion pipeline that processes incoming frames and
 * fuses them into one volume
 *
 * It implements the basic four steps described in the KinectFusion paper:
 * (1) Surface Measurement: Compute vertex and normal maps and their pyramids
 * (2) Pose Estimation: Use ICP with measured depth and predicted surface to
 * localize camera (3) Surface reconstruction: Integration of surface
 * measurements into a global volume (4) Surface prediction: Raycast volume in
 * order to compute a surface prediction
 *
 * After construction, the pipeline allows you to insert new frames consisting
 * of depth and color. In the end, you can export the internal volume either as
 * a pointcloud or a dense surface mesh. You can also export the camera poses
 * and (depending on your configuration) visualize the last model frame.
 *
 */
class _API Pipeline {
 public:
  /**
   * Constructs the pipeline, sets up the interal volume and camera.
   * @param _camera_parameters The \ref{reclib::IntrinsicParameters} that you
   * want this pipeline to use
   * @param _configuration The \ref{GlobalConfiguration} with all parameters the
   * pipeline should use
   */
  Pipeline(reclib::IntrinsicParameters camera_parameters,
           const Configuration& configuration);

  ~Pipeline() = default;

  /**
   * Invoke this for every frame you want to fuse into the global volume
   * @param depth_map The depth map for the current frame. Must consist of float
   * values representing the depth in mm
   * @param color_map The RGB color map. Must be a Matrix (datatype CV_8UC3)
   * @return Whether the frame has been fused successfully. Will only be false
   * if the ICP failed.
   */
  bool ProcessFrame(const CpuMat_<float>& depth_map,
                    const CpuMat_<cv::Vec3b>& color_map);

  /**
   * Retrieve all camera poses computed so far
   * @return A vector for 4x4 camera poses, consisting of rotation and
   * translation
   */
  std::vector<Eigen::Matrix4f> GetPoses() const;

  /**
   * Use this to get a visualization of the last raycasting
   * @return The last (colorized) model frame from raycasting the internal
   * volume
   */
  CpuMat GetModelRgb() const;
  CpuMat GetModelVertices() const;
  CpuMat GetModelNormals() const;
  ExtrinsicParameters GetCameraExtrinsics();

  /**
   * Extract a point cloud
   * @return A PointCloud representation (see description of PointCloud for more
   * information on the data layout)
   */
  PointCloud ExtractPointcloud();

  /**
   * Extract a dense surface mesh
   * @return A SurfaceMesh representation (see description of SurfaceMesh for
   * more information on the data layout)
   */
  SurfaceMesh ExtractMesh();

  /**
   * Raycast volume in order to compute a surface prediction
   * This method does not modify the internal state of the kinectfusion
   * module. It is merely intended to render the volume from a different
   * camera view (pose).
   */
  void SurfacePrediction(CpuMat& model_vertex, CpuMat& model_normal,
                         CpuMat& model_color, const Eigen::Matrix4f& pose);

 private:
  // Internal parameters, not to be changed after instantiation
  const IntrinsicParameters camera_parameters_;
  const Configuration configuration_;

  // The global volume (containing tsdf and color)
  VolumeData volume_;

  // The model data for the current frame
  ModelData model_data_;

  // Poses: Current and all previous
  Eigen::Matrix4f current_pose_;
  std::vector<Eigen::Matrix4f> poses_;

  // Frame ID and raycast result for output purposes
  size_t frame_id_;
  CpuMat last_model_frame_rgb_;
  CpuMat last_model_frame_vertices_;
  CpuMat last_model_frame_normals_;
};

/**
 * Store a PointCloud instance as a PLY file.
 * If file cannot be saved, nothing will be done
 * @param filename The path and name of the file to write to; if it does not
 * exists, it will be created and if it exists it will be overwritten
 * @param point_cloud The PointCloud instance
 */
void ExportPly(const std::string& filename, const PointCloud& point_cloud);

/**
 * Store a SurfaceMesh instance as a PLY file.
 * If file cannot be saved, nothing will be done
 * @param filename The path and name of the file to write to; if it does not
 * exists, it will be created and if it exists it will be overwritten
 * @param surface_mesh The SurfaceMesh instance
 */
void ExportPly(const std::string& filename, const SurfaceMesh& surface_mesh);

namespace internal {

/*
 * Step 1: SURFACE MEASUREMENT
 * Compute vertex and normal maps and their pyramids
 */
FrameData SurfaceMeasurement(const CpuMat_<float>& input_frame,
                             const reclib::IntrinsicParameters& camera_params,
                             size_t num_levels, float depth_cutoff,
                             int kernel_size, float color_sigma,
                             float spatial_sigma);

/*
 *
 * Step 2: POSE ESTIMATION
 * Use ICP with measured depth and predicted surface to localize camera
 */
bool PoseEstimation(Eigen::Matrix4f& pose, const FrameData& frame_data,
                    const ModelData& model_data,
                    const reclib::IntrinsicParameters& cam_params,
                    int pyramid_height, float distance_threshold,
                    float angle_threshold, const std::vector<int>& iterations);

namespace cuda {

/*
 *
 * Step 2: Helper function for POSE ESTIMATION
 * Use ICP with measured depth and predicted surface to localize camera
 */
void EstimateStep(
    const Eigen::Matrix3f& rotation_current,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current,
    const GpuMat& vertex_map_current, const GpuMat& normal_map_current,
    const Eigen::Matrix3f& rotation_previous_inv,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous,
    const IntrinsicParameters& cam_params, const GpuMat& vertex_map_previous,
    const GpuMat& normal_map_previous, float distance_threshold,
    float angle_threshold, Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& a,
    Eigen::Matrix<double, 6, 1>& b);

/*
 * Step 3: SURFACE RECONSTRUCTION
 * Integration of surface measurements into a global volume
 */
void SurfaceReconstruction(const GpuMat& depth_image, const GpuMat& normal_map,
                           const GpuMat& color_image, GpuMat& tsdf_volume,
                           GpuMat& color_volume, const ivec3& grid_size,
                           const float voxel_size,
                           const reclib::IntrinsicParameters& cam_params,
                           float truncation_distance,
                           const Eigen::Matrix4f& model_view, float max_weight);

/*
 * Step 4: SURFACE PREDICTION
 * Raycast volume in order to compute a surface prediction
 */
void SurfacePrediction(const VolumeData& volume, GpuMat& model_vertex,
                       GpuMat& model_normal, GpuMat& model_color,
                       const reclib::IntrinsicParameters& cam_parameters,
                       const float truncation_distance,
                       const Eigen::Matrix4f& pose);

PointCloud ExtractPoints(const VolumeData& volume, int buffer_size);

SurfaceMesh MarchingCubes(const GpuMat& tsdf_volume, const GpuMat& color_volume,
                          const ivec3 grid_size, const float voxel_size,
                          int triangles_buffer_size);

// ---------------------------------------------------------------------------
// End of adapted code
// ---------------------------------------------------------------------------

}  // namespace cuda

}  // namespace internal

#endif

}  // namespace kinfu
}  // namespace reclib

#endif
