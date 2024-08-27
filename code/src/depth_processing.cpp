#include <reclib/depth_processing.h>

#include <iostream>

#if HAS_OPENCV_MODULE
void reclib::ComputeVertexMap(const CpuMat& depth_map, CpuMat& vertex_map,
                              const IntrinsicParameters& cam_params,
                              const float depth_cutoff,
                              const ivec2& pixel_offsets, const float scale) {
  GpuMat depth_map_gpu;
  depth_map_gpu.upload(depth_map);

  GpuMat vertex_map_gpu;
  cuda::ComputeVertexMap(depth_map_gpu, vertex_map_gpu, cam_params,
                         depth_cutoff, pixel_offsets, scale);

  if (vertex_map.empty()) {
    vertex_map.create(vertex_map_gpu.size(), vertex_map_gpu.type());
  }

  vertex_map_gpu.download(vertex_map);
  vertex_map_gpu.release();
}

void reclib::ComputeNormalMap(const CpuMat& vertex_map, CpuMat& normal_map) {
  GpuMat vertex_map_gpu;
  vertex_map_gpu.upload(vertex_map);

  GpuMat normal_map_gpu;
  cuda::ComputeNormalMap(vertex_map_gpu, normal_map_gpu);

  if (normal_map.empty()) {
    normal_map.create(normal_map_gpu.size(), normal_map_gpu.type());
  }

  normal_map_gpu.download(normal_map);
  normal_map_gpu.release();
}

void reclib::ComputePrunedVertexNormalMap(CpuMat& vertex_map,
                                          CpuMat& normal_map,
                                          float normal_thresh) {
  GpuMat vertex_map_gpu;
  vertex_map_gpu.upload(vertex_map);

  GpuMat normal_map_gpu;
  cuda::ComputePrunedVertexNormalMap(vertex_map_gpu, normal_map_gpu,
                                     normal_thresh);

  if (normal_map.empty()) {
    normal_map.create(normal_map_gpu.size(), normal_map_gpu.type());
  }

  normal_map_gpu.download(normal_map);
  normal_map_gpu.release();

  vertex_map_gpu.download(vertex_map);
  vertex_map_gpu.release();
}

void reclib::ComputePhongMap(
    const CpuMat& normal_map, CpuMat& phong_map,
    const IntrinsicParameters& cam_params, const PhongParameters& phong_params,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& view_dir) {
  GpuMat normal_map_gpu;
  normal_map_gpu.upload(normal_map);

  GpuMat phong_map_gpu;
  cuda::ComputePhongMap(normal_map_gpu, phong_map_gpu, cam_params, phong_params,
                        view_dir);

  if (phong_map.elemSize() < phong_map_gpu.elemSize()) {
    phong_map.release();
    phong_map.create(phong_map_gpu.size(), phong_map_gpu.type());
  }

  phong_map_gpu.download(phong_map);
  phong_map_gpu.release();
}

cv::Mat reclib::ColorizeDepth(const CpuMat& depth,
                              const cv::ColormapTypes& style) {
  double min, max;
  cv::minMaxLoc(depth, &min, &max);

  // normalize depth to [0,1]
  cv::Mat normalized_depth = (depth / max);
  cv::Mat depth_8bit(depth.size(), CV_8UC1);

  normalized_depth = (normalized_depth * (pow(2, 8) - 1));
  normalized_depth.convertTo(depth_8bit, CV_8UC1);

  // create color map
  cv::Mat color_map(depth_8bit.size(), CV_8UC3);
  cv::applyColorMap(depth_8bit, color_map, style);

  return color_map;
}
#endif  // HAS_OPENCV_MODULE
