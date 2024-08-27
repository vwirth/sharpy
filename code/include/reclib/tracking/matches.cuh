#ifndef RECLIB_KNN_CORRS_CUH
#define RECLIB_KNN_CORRS_CUH

#if HAS_OPENCV_MODULE
#if WITH_CUDA

#include <device_launch_parameters.h>
#include <reclib/internal/opencv_utils.h>

namespace reclib {
namespace tracking {
namespace cuda {

#define NUM_KNN 1
#define KNN_BLOCKSIZE 64

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
                     torch::Tensor& joints_visible);

RECLIB_DEVICE __forceinline__ int64_t linearized_input_index() {
  return blockIdx.x * KNN_BLOCKSIZE + threadIdx.x;
}
RECLIB_DEVICE __forceinline__ int64_t linearized_output_index(int k = 0) {
  return (blockIdx.x * KNN_BLOCKSIZE + threadIdx.x) * NUM_KNN + k;
}
RECLIB_DEVICE __forceinline__ int insertion_sort(int64_t* indices, float* dists,
                                                 float new_val, int index) {
  int i = 0;
  // #pragma unroll(NUM_KNN)
  for (; i < NUM_KNN; i++) {
    // if (indices[index] >)
    float dist = dists[i];
    if (dist > new_val) {
      // #pragma unroll(NUM_KNN - 2)
      for (int j = (int)NUM_KNN - 2; j >= (int)i; j--) {
        indices[(j + 1)] = indices[j];
        dists[j + 1] = dists[j];
      }
      indices[i] = index;
      dists[i] = new_val;
      break;
    }
  }
  return i;
}

}  // namespace cuda
}  // namespace tracking
}  // namespace reclib

#endif  // WITH_CUDA
#endif  // HAS_OPENCV_MODULE

#endif
