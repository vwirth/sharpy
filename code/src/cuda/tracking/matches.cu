// TODO: clean includes
#include <reclib/internal/opencv_utils.h>

#include <chrono>

#if HAS_OPENCV_MODULE
#include <opencv2/cudaarithm.hpp>
#endif
#include <reclib/data_types.h>

#include <reclib/cuda/common.cuh>
#include <reclib/cuda/debug.cuh>
#include <reclib/cuda/helper.cuh>
#include <reclib/cuda/thread_info.cuh>
#include <reclib/fusion/dynamicfusion.cuh>
#include <reclib/tracking/matches.cuh>

#if HAS_OPENCV_MODULE

namespace reclib {
namespace tracking {
namespace cuda {

__global__ void gpu_knn(
    const vec3* reference_canonicals, const vec3* reference_vertices,
    const int32_t* reference_segmentation,
    const uint32_t reference_points_length, const vec3* query_canonicals,
    const vec3* query_vertices, const vec3* query_normals,
    const int32_t* query_segmentation, const uint32_t query_points_length,
    int stage, float mean_depth, float pc_dist_thresh, float pc_normal_thresh,
    float nn_dist_thresh, const int32_t* verts2joints, int64_t* indices,
    bool* joints_visible) {
  if (linearized_input_index() >= query_points_length) return;

  vec3 canonical = query_canonicals[linearized_input_index()];
  vec3 vertex = query_vertices[linearized_input_index()];
  vec3 normal = query_normals[linearized_input_index()];
  int segment = query_segmentation[linearized_input_index()];

  // we need to go over the resulting list on the CPU anyway to remove the
  // -1 values.
  // Removing the computation here does not slow down the cuda code, however.
  // Thus we keep it on the gpu.
  if (abs(vertex.z() - mean_depth) > pc_dist_thresh ||
      abs(normal.z()) < pc_normal_thresh || vertex.norm() == 0 ||
      canonical.norm() == 0) {
    indices[linearized_output_index()] = -1;
    return;
  }

  float mindist = std::numeric_limits<float>::max();
  int closest_index = -1;
#pragma unroll 16
  for (int i = 0; i < reference_points_length; i++) {
    vec3 other_canonical = reference_canonicals[i];
    vec3 other_vertex = reference_vertices[i];

    // Threshold calculation
    if (other_vertex.norm() == 0 || other_canonical.norm() == 0) {
      continue;
    }

    int other_segment = reference_segmentation[i];

    if (other_segment != segment) {
      continue;
    }

    // if the hand is not registered yet it does not
    // make sense to apply a nearest neighbor threshold
    if (stage > 0) {
      if ((abs(vertex.z() - other_vertex.z()) > nn_dist_thresh)) {
        continue;
      }
    }

    float dist = (other_canonical - canonical).norm();
    if (dist < mindist) {
      mindist = dist;
      indices[linearized_output_index()] = i;
      closest_index = i;
    }
  }
  if (closest_index != -1) {
    int joint = verts2joints[closest_index];
    joints_visible[joint] = true;
  }
}

/*
 * For each point in query_points, compute the nearest neighbor in
 * reference_points that satisfies all thresholds.
 * Parameters: mean_depth, pc_dist_thresh, pc_normal_thresh
 * Used to determine thresholds.
 * Returns: indices
 * A Tensor of size query_points.size(), containing for each query point the
 * index of the nearest neighbor in reference_points.
 * If no point in reference_points was found that satisfies all thresholds,
 * index is set to -1.
 */
void compute_matches(const torch::Tensor& reference_canonicals,
                     const torch::Tensor& reference_vertices,
                     const torch::Tensor& reference_segmentation,
                     const torch::Tensor& query_canonicals,
                     const torch::Tensor& query_vertices,
                     const torch::Tensor& query_normals,
                     const torch::Tensor& query_segmentation, int stage,
                     float mean_depth, float pc_dist_thresh,
                     float pc_normal_thresh, float nn_dist_thresh,
                     const torch::Tensor& verts2joints, torch::Tensor& indices,
                     torch::Tensor& joints_visible) {
  _RECLIB_ASSERT_EQ(reference_canonicals.sizes()[0],
                    reference_vertices.sizes()[0]);
  _RECLIB_ASSERT_EQ(reference_vertices.sizes()[0],
                    reference_segmentation.sizes()[0]);
  _RECLIB_ASSERT_EQ(reference_segmentation.sizes()[0], verts2joints.sizes()[0]);
  _RECLIB_ASSERT_EQ(query_canonicals.sizes()[0], query_vertices.sizes()[0]);
  _RECLIB_ASSERT_EQ(query_vertices.sizes()[0], query_normals.sizes()[0]);
  _RECLIB_ASSERT_EQ(query_normals.sizes()[0], query_segmentation.sizes()[0]);

  // divide and round up
  int num_blocks =
      (query_canonicals.sizes()[0] + KNN_BLOCKSIZE - 1) / KNN_BLOCKSIZE;

  // Tensor.data_ptr does not support unsigned int
  // If memory is a problem, we could use at::kInt16 for the indices
  // TODO: Eigen::Map<vec3> like in sharpy.cu
  // cuda has a different memory layout, reinterpret_cast can go wrong
  gpu_knn<<<num_blocks, KNN_BLOCKSIZE>>>(
      reinterpret_cast<vec3*>(reference_canonicals.data_ptr<float>()),
      reinterpret_cast<vec3*>(reference_vertices.data_ptr<float>()),
      reference_segmentation.data_ptr<int32_t>(),
      reference_canonicals.sizes()[0],
      reinterpret_cast<vec3*>(query_canonicals.data_ptr<float>()),
      reinterpret_cast<vec3*>(query_vertices.data_ptr<float>()),
      reinterpret_cast<vec3*>(query_normals.data_ptr<float>()),
      query_segmentation.data_ptr<int32_t>(), query_canonicals.sizes()[0],
      stage, mean_depth, pc_dist_thresh, pc_normal_thresh, nn_dist_thresh,
      verts2joints.data_ptr<int32_t>(), indices.data_ptr<int64_t>(),
      joints_visible.data_ptr<bool>());

  // CUDA_SYNC_CHECK();
  // cudaThreadSynchronize();
}

}  // namespace cuda
}  // namespace tracking
}  // namespace reclib

#endif  // HAS_OPENCV_MODULE
