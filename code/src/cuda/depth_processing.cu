
#include <reclib/assert.h>
#include <reclib/camera_parameters.h>
#include <reclib/data_types.h>
#include <reclib/depth_processing.h>

#include <iostream>
#include <reclib/cuda/common.cuh>

namespace reclib {
namespace cuda {

#if HAS_OPENCV_MODULE
__global__ void kernel_compute_vertex_map(
    const cv::cuda::PtrStepSz<float> depth_map,
    cv::cuda::PtrStep<float3> vertex_map, const IntrinsicParameters cam_params,
    const float depth_cutoff, const int offset_x, const int offset_y,
    const float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= depth_map.cols || y >= depth_map.rows) return;

  float depth_value = depth_map.ptr(y)[x];

  if (depth_value > depth_cutoff && depth_cutoff >= 0.0f)
    depth_value = 0.f;  // Depth cutoff

  vec3 vertex((x + offset_x - cam_params.principal_x_) * depth_value /
                  cam_params.focal_x_,
              (y + offset_y - cam_params.principal_y_) * depth_value /
                  cam_params.focal_y_,
              depth_value);

  if (isinf(vertex.x()) || isinf(vertex.y()) || isinf(vertex.z())) {
    vertex = vec3(0, 0, 0);
  }

  vertex_map.ptr(y)[x] =
      make_float3(vertex.x() * scale, vertex.y() * scale, vertex.z() * scale);

  // if (depth_value == 0) {
  //   vertex_map.ptr(y)[x] = make_float3(1, 0, 0);
  // }
}

__global__ void kernel_compute_normal_map_4neighbors(
    const cv::cuda::PtrStepSz<float3> vertex_map,
    cv::cuda::PtrStep<float3> normal_map) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= vertex_map.cols - 1 || y < 1 || y >= vertex_map.rows - 1)
    return;

  const vec3 left(&vertex_map.ptr(y)[x - 1].x);
  const vec3 right(&vertex_map.ptr(y)[x + 1].x);
  const vec3 upper(&vertex_map.ptr(y - 1)[x].x);
  const vec3 lower(&vertex_map.ptr(y + 1)[x].x);

  vec3 normal;

  if (left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
    normal = vec3(0.f, 0.f, 0.f);
  else {
    vec3 hor(right.x() - left.x(), right.y() - left.y(), right.z() - left.z());
    vec3 ver(upper.x() - lower.x(), upper.y() - lower.y(),
             upper.z() - lower.z());

    normal = hor.cross(ver);
    normal.normalize();
    normal = -normal;

    // if (normal.z() > 0) normal *= -1;
  }

  normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
}

__global__ void kernel_compute_pruned_vertex_normal_map_4neighbors(
    float normal_thresh, cv::cuda::PtrStepSz<float3> vertex_map,
    cv::cuda::PtrStep<float3> normal_map) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= vertex_map.cols - 1 || y < 1 || y >= vertex_map.rows - 1)
    return;

  const vec3 left(&vertex_map.ptr(y)[x - 1].x);
  const vec3 right(&vertex_map.ptr(y)[x + 1].x);
  const vec3 upper(&vertex_map.ptr(y - 1)[x].x);
  const vec3 lower(&vertex_map.ptr(y + 1)[x].x);

  vec3 normal;

  if (left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
    normal = vec3(0.f, 0.f, 0.f);
  else {
    vec3 hor(right.x() - left.x(), right.y() - left.y(), right.z() - left.z());
    vec3 ver(upper.x() - lower.x(), upper.y() - lower.y(),
             upper.z() - lower.z());

    normal = hor.cross(ver);
    normal.normalize();
    normal = -normal;

    // if (normal.z() > 0) normal *= -1;
  }
  if (abs(normal.z()) > normal_thresh) {
    normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
  } else {
    normal_map.ptr(y)[x] = make_float3(0, 0, 0);
    vertex_map.ptr(y)[x] = make_float3(0, 0, 0);
  }
}

__global__ void kernel_compute_normal_map_2neighbors(
    const cv::cuda::PtrStepSz<float3> vertex_map,
    cv::cuda::PtrStep<float3> normal_map) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || x >= vertex_map.cols - 1 || y < 0 || y >= vertex_map.rows - 1)
    return;

  const vec3 cur(&vertex_map.ptr(y)[x].x);
  const vec3 right(&vertex_map.ptr(y)[x + 1].x);
  const vec3 lower(&vertex_map.ptr(y + 1)[x].x);

  vec3 normal;

  if (right.z() == 0 || lower.z() == 0)
    normal = vec3(0.f, 0.f, 0.f);
  else {
    vec3 hor(right.x() - cur.x(), right.y() - cur.y(), right.z() - cur.z());
    vec3 ver(lower.x() - cur.x(), lower.y() - cur.y(), lower.z() - cur.z());

    normal = hor.cross(ver);
    normal.normalize();
    normal = -normal;

    // if (normal.z() > 0) normal *= -1;
  }

  normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
}

__global__ void kernel_compute_pruned_vertex_normal_map_2neighbors(
    float normal_thresh, cv::cuda::PtrStepSz<float3> vertex_map,
    cv::cuda::PtrStep<float3> normal_map) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || x >= vertex_map.cols - 1 || y < 0 || y >= vertex_map.rows - 1)
    return;

  const vec3 cur(&vertex_map.ptr(y)[x].x);
  const vec3 right(&vertex_map.ptr(y)[x + 1].x);
  const vec3 lower(&vertex_map.ptr(y + 1)[x].x);

  vec3 normal;

  if (right.z() == 0 || lower.z() == 0)
    normal = vec3(0.f, 0.f, 0.f);
  else {
    vec3 hor(right.x() - cur.x(), right.y() - cur.y(), right.z() - cur.z());
    vec3 ver(lower.x() - cur.x(), lower.y() - cur.y(), lower.z() - cur.z());

    normal = hor.cross(ver);
    normal.normalize();
    normal = -normal;

    // if (normal.z() > 0) normal *= -1;
  }

  if (abs(normal.z()) > normal_thresh) {
    normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
  } else {
    normal_map.ptr(y)[x] = make_float3(0, 0, 0);
    vertex_map.ptr(y)[x] = make_float3(0, 0, 0);
  }
}

__global__ void kernel_compute_phong_map(
    const cv::cuda::PtrStepSz<float3> normal_map,
    cv::cuda::PtrStep<float3> phong_map, const IntrinsicParameters cam_params,
    const PhongParameters phong_params,
    Eigen::Matrix<float, 3, 1, Eigen::DontAlign> view_dir) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= normal_map.cols || y >= normal_map.rows) return;

  const vec3 normal(&normal_map.ptr(y)[x].x);
  vec3 color(0, 0, 0);
  if (normal.x() != 0 && normal.y() != 0 && normal.z() != 0) {
    vec3 light_dir = phong_params.light_dir_.normalized();
    vec3 n = normal.normalized();

    float n_dot_l = max(min(n.dot(light_dir), 1.f), 0.f);
    vec3 reflection = 2 * n_dot_l * n - light_dir;
    float r_dot_v = max(min(reflection.dot(view_dir), 1.f), 0.f);

    vec3 diffuse =
        phong_params.diffuse_coeff_ * n_dot_l * phong_params.diffuse_color_;

    vec3 specular = phong_params.specular_coeff_ *
                    pow(r_dot_v, phong_params.shinyness_) *
                    phong_params.specular_color_;

    color = phong_params.ambient_coeff_ * phong_params.ambient_color_;
    color = color + diffuse + specular;
  }

  phong_map.ptr(y)[x] = make_float3(color.x(), color.y(), color.z());
}
#endif  // HAS_OPENCV_MODULE

}  // namespace cuda
}  // namespace reclib

#if HAS_OPENCV_MODULE

__global__ void kernel_naive_compute_vertex_map(
    const float* depth_map, int width, int height, float* vertex_map,
    const reclib::IntrinsicParameters cam_params, const float depth_cutoff,
    const int offset_x, const int offset_y, const float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float depth_value = depth_map[y * width + x];
  if (depth_value > depth_cutoff && depth_cutoff >= 0.0f)
    depth_value = 0.f;  // Depth cutoff

  float val = (x + offset_x - cam_params.principal_x_) * depth_value /
              cam_params.focal_x_;
  if (!isinf(val)) {
    vertex_map[(y * width + x) * 3 + 0] = val * scale;
  } else {
    vertex_map[(y * width + x) * 3 + 0] = 0;
    vertex_map[(y * width + x) * 3 + 1] = 0;
    vertex_map[(y * width + x) * 3 + 2] = 0;
    return;
  }

  val = (y + offset_y - cam_params.principal_y_) * depth_value /
        cam_params.focal_y_;
  if (!isinf(val)) {
    vertex_map[(y * width + x) * 3 + 1] = val * scale;
  } else {
    vertex_map[(y * width + x) * 3 + 0] = 0;
    vertex_map[(y * width + x) * 3 + 1] = 0;
    vertex_map[(y * width + x) * 3 + 2] = 0;
    return;
  }

  vertex_map[(y * width + x) * 3 + 2] = depth_value * scale;
}

__global__ void kernel_naive_compute_pruned_vertex_normal_map_4neighbors(
    float normal_thresh, int width, int height, float* vertex_map,
    float* normal_map) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

  Eigen::Map<vec3> left(&vertex_map[(y * width + x - 1) * 3]);
  Eigen::Map<vec3> right(&vertex_map[(y * width + x + 1) * 3]);
  Eigen::Map<vec3> upper(&vertex_map[((y - 1) * width + x) * 3]);
  Eigen::Map<vec3> lower(&vertex_map[((y + 1) * width + x) * 3]);

  Eigen::Map<vec3> normal(&normal_map[(y * width + x) * 3]);
  Eigen::Map<vec3> vertex(&vertex_map[(y * width + x) * 3]);

  if (left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
    normal = vec3(0.f, 0.f, 0.f);
  else {
    vec3 hor(right.x() - left.x(), right.y() - left.y(), right.z() - left.z());
    vec3 ver(upper.x() - lower.x(), upper.y() - lower.y(),
             upper.z() - lower.z());

    normal = hor.cross(ver);
    normal.normalize();
    normal = -normal;
  }
  if (abs(normal.z()) <= normal_thresh) {
    normal.x() = 0;
    normal.y() = 0;
    normal.z() = 0;
    vertex.x() = 0;
    vertex.y() = 0;
    vertex.z() = 0;
  }
}

#if HAS_OPENCV_MODULE

void reclib::cuda::ComputeVertexMap(const GpuMat& depth_map, GpuMat& vertex_map,
                                    const IntrinsicParameters& cam_params,
                                    const float depth_cutoff,
                                    const ivec2& pixel_offsets,
                                    const float scale) {
  if (vertex_map.empty()) {
    vertex_map.create(depth_map.size(), CV_32FC3);
  }

  _RECLIB_ASSERT_EQ(depth_map.type(), CV_32FC1);
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(depth_map.type(), CV_32FC1);
  _RECLIB_ASSERT_EQ(vertex_map.type(), CV_32FC3);
#endif

  dim3 threads(32, 32);
  dim3 blocks((depth_map.cols + threads.x - 1) / threads.x,
              (depth_map.rows + threads.y - 1) / threads.y);

  kernel_compute_vertex_map<<<blocks, threads>>>(
      depth_map, vertex_map, cam_params, depth_cutoff, pixel_offsets.x(),
      pixel_offsets.y(), scale);

  cudaThreadSynchronize();
}

void reclib::cuda::ComputeNormalMap(const GpuMat& vertex_map,
                                    GpuMat& normal_map) {
  dim3 threads(32, 32);
  dim3 blocks((vertex_map.cols + threads.x - 1) / threads.x,
              (vertex_map.rows + threads.y - 1) / threads.y);

  if (normal_map.empty()) {
    normal_map.create(vertex_map.size(), vertex_map.type());
  }

  _RECLIB_ASSERT_EQ(vertex_map.type(), CV_32FC3);
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(vertex_map.type(), CV_32FC3);
  _RECLIB_ASSERT_EQ(normal_map.type(), CV_32FC3);
#endif

  kernel_compute_normal_map_4neighbors<<<blocks, threads>>>(vertex_map,
                                                            normal_map);

  cudaThreadSynchronize();
}

void reclib::cuda::ComputePrunedVertexNormalMap(GpuMat& vertex_map,
                                                GpuMat& normal_map,
                                                float normal_thresh) {
  dim3 threads(32, 32);
  dim3 blocks((vertex_map.cols + threads.x - 1) / threads.x,
              (vertex_map.rows + threads.y - 1) / threads.y);

  if (normal_map.empty()) {
    normal_map.create(vertex_map.size(), vertex_map.type());
  }

  _RECLIB_ASSERT_EQ(vertex_map.type(), CV_32FC3);
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(vertex_map.type(), CV_32FC3);
  _RECLIB_ASSERT_EQ(normal_map.type(), CV_32FC3);
#endif

  kernel_compute_pruned_vertex_normal_map_4neighbors<<<blocks, threads>>>(
      normal_thresh, vertex_map, normal_map);

  cudaThreadSynchronize();
}

void reclib::cuda::ComputePhongMap(
    const GpuMat& normal_map, GpuMat& phong_map,
    const IntrinsicParameters& cam_params, const PhongParameters& phong_params,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& view_dir) {
  dim3 threads(32, 32);
  dim3 blocks((normal_map.cols + threads.x - 1) / threads.x,
              (normal_map.rows + threads.y - 1) / threads.y);

  if (phong_map.empty()) {
    phong_map.create(normal_map.size(), normal_map.type());
  }

  _RECLIB_ASSERT_EQ(normal_map.type(), CV_32FC3);
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(normal_map.type(), CV_32FC3);
  _RECLIB_ASSERT_EQ(phong_map.type(), CV_32FC3);
#endif

  kernel_compute_phong_map<<<blocks, threads>>>(
      normal_map, phong_map, cam_params, phong_params, view_dir);
  cudaThreadSynchronize();
}

#endif  // HAS_OPENCV_MODULE

#if HAS_DNN_MODULE

void reclib::cuda::ComputeVertexMap(const torch::Tensor& depth_map,
                                    torch::Tensor& vertex_map,
                                    const IntrinsicParameters& cam_params,
                                    const float depth_cutoff,
                                    const ivec2& pixel_offsets,
                                    const float scale) {
  if (!vertex_map.defined()) {
    vertex_map = torch::zeros({depth_map.sizes()[0], depth_map.sizes()[1], 3},
                              torch::TensorOptions().device(torch::kCUDA));
  }
  int height = depth_map.sizes()[0];
  int width = depth_map.sizes()[1];

  _RECLIB_ASSERT_EQ(depth_map.scalar_type(), torch::ScalarType::Float);
  _RECLIB_ASSERT_EQ(vertex_map.scalar_type(), torch::ScalarType::Float);
  _RECLIB_ASSERT_EQ(vertex_map.sizes()[2], 3);
  _RECLIB_ASSERT_EQ(height, vertex_map.sizes()[0]);
  _RECLIB_ASSERT_EQ(width, vertex_map.sizes()[1]);

  _RECLIB_ASSERT(vertex_map.is_cuda());
  _RECLIB_ASSERT(depth_map.is_cuda());

  dim3 threads(32, 32);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

  kernel_naive_compute_vertex_map<<<blocks, threads>>>(
      depth_map.data_ptr<float>(), width, height, vertex_map.data_ptr<float>(),
      cam_params, depth_cutoff, pixel_offsets.x(), pixel_offsets.y(), scale);

  cudaThreadSynchronize();
}

void reclib::cuda::ComputePrunedVertexNormalMap(torch::Tensor& vertex_map,
                                                torch::Tensor& normal_map,
                                                float normal_thresh) {
  if (!normal_map.defined()) {
    normal_map = torch::zeros({vertex_map.sizes()[0], vertex_map.sizes()[1], 3},
                              torch::TensorOptions().device(torch::kCUDA));
  }
  int height = vertex_map.sizes()[0];
  int width = vertex_map.sizes()[1];

  _RECLIB_ASSERT_EQ(vertex_map.scalar_type(), torch::ScalarType::Float);
  _RECLIB_ASSERT_EQ(vertex_map.sizes()[2], 3);
  _RECLIB_ASSERT_EQ(normal_map.scalar_type(), torch::ScalarType::Float);
  _RECLIB_ASSERT_EQ(normal_map.sizes()[2], 3);
  _RECLIB_ASSERT_EQ(height, normal_map.sizes()[0]);
  _RECLIB_ASSERT_EQ(width, normal_map.sizes()[1]);

  _RECLIB_ASSERT(vertex_map.is_cuda());
  _RECLIB_ASSERT(normal_map.is_cuda());

  dim3 threads(32, 32);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

  kernel_naive_compute_pruned_vertex_normal_map_4neighbors<<<blocks, threads>>>(
      normal_thresh, width, height, vertex_map.data_ptr<float>(),
      normal_map.data_ptr<float>());

  cudaThreadSynchronize();
}

#endif  // HAS_DNN_MODULE

#endif  // HAS_OPENCV_MODULE
