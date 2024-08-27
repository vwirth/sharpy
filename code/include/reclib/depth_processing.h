#ifndef DEPTH_H
#define DEPTH_H

#include <Eigen/Eigen>

#if HAS_OPENCV_MODULE
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#if HAS_DNN_MODULE
#include <torch/torch.h>
#endif

#include "reclib/camera_parameters.h"

namespace reclib {

struct PhongParameters {
  mat31da ambient_color_{0.5F, 0.5F, 0.5F};
  mat31da diffuse_color_{0.5F, 0.5F, 0.5F};
  mat31da specular_color_{0.5F, 0.5F, 0.5F};

  mat31da light_dir_{0.5F, -1.5F, 0.F};
  float shinyness_{2};
  float ambient_coeff_{0.6};
  float diffuse_coeff_{0.3};
  float specular_coeff_{0.1};

  PhongParameters() = default;

  PhongParameters(mat31da& color, mat31da& dir, float shiny,
                  float ambient_coeff = 0.6, float diffuse_coeff = 0.3,
                  float specular_coeff = 0.1)
      : ambient_color_(color),
        diffuse_color_(color),
        specular_color_(color),
        light_dir_(dir),
        shinyness_(shiny),
        ambient_coeff_(ambient_coeff),
        diffuse_coeff_(diffuse_coeff),
        specular_coeff_(specular_coeff){};

  PhongParameters(mat31da& ambient_color, mat31da& diffuse_color,
                  mat31da& specular_color, mat31da& dir, float shiny,
                  float ambient_coeff = 0.6, float diffuse_coeff = 0.3,
                  float specular_coeff = 0.1)
      : ambient_color_(ambient_color),
        diffuse_color_(diffuse_color),
        specular_color_(specular_color),
        light_dir_(dir),
        shinyness_(shiny),
        ambient_coeff_(ambient_coeff),
        diffuse_coeff_(diffuse_coeff),
        specular_coeff_(specular_coeff){};
};

#if HAS_OPENCV_MODULE
void ComputeVertexMap(const CpuMat& depth_map, CpuMat& vertex_map,
                      const IntrinsicParameters& cam_params,
                      float depth_cutoff = -1.0F,
                      const ivec2& pixel_offsets = ivec2(0, 0),
                      const float scale = 1.f);

void ComputeNormalMap(const CpuMat& vertex_map, CpuMat& normal_map);

void ComputePrunedVertexNormalMap(CpuMat& vertex_map, CpuMat& normal_map,
                                  float normal_thresh);

void ComputePhongMap(const CpuMat& normal_map, CpuMat& phong_map,
                     const IntrinsicParameters& cam_params,
                     const PhongParameters& phong_params,
                     const mat31da& view_dir);

cv::Mat ColorizeDepth(const CpuMat& depth, const cv::ColormapTypes& style);
#endif  // HAS_OPENCV_MODULE

namespace cuda {

#if HAS_OPENCV_MODULE
void ComputeVertexMap(const GpuMat& depth_map, GpuMat& vertex_map,
                      const IntrinsicParameters& cam_params,
                      float depth_cutoff = -1.0F,
                      const ivec2& pixel_offsets = ivec2(0, 0),
                      const float scale = 1.f);

void ComputeNormalMap(const GpuMat& vertex_map, GpuMat& normal_map);

void ComputePrunedVertexNormalMap(GpuMat& vertex_map, GpuMat& normal_map,
                                  float normal_thresh);

void ComputePhongMap(const GpuMat& normal_map, GpuMat& phong_map,
                     const IntrinsicParameters& cam_params,
                     const PhongParameters& phong_params,
                     const mat31da& view_dir);
#endif  // HAS_OPENCV_MODULE

#if HAS_DNN_MODULE
void ComputeVertexMap(const torch::Tensor& depth_map, torch::Tensor& vertex_map,
                      const IntrinsicParameters& cam_params,
                      float depth_cutoff = -1.0F,
                      const ivec2& pixel_offsets = ivec2(0, 0),
                      const float scale = 1.f);

void ComputePrunedVertexNormalMap(torch::Tensor& vertex_map,
                                  torch::Tensor& normal_map,
                                  float normal_thresh);

#endif  // HAS_DNN_MODULE

}  // namespace cuda

}  // namespace reclib

#endif
