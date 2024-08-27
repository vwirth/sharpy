#ifndef CUDA_HELPER
#define CUDA_HELPER

#include <limits>
#if WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstdint>

#include "reclib/assert.h"
#include "reclib/cuda/device_info.cuh"
#include "reclib/data_types.h"
#include "reclib/math/math_ops.h"

namespace reclib {

namespace cuda {
template <typename T1, typename T2>
inline RECLIB_HD const T1 getBlockCount(T1 problemSize, T2 threadCount) {
  return (problemSize + (threadCount - T2(1))) / (threadCount);
}

inline RECLIB_HD const dim3 getBlockCount(dim3 problemSize, dim3 threadCount) {
  dim3 a((problemSize.x + (threadCount.x - 1)) / (threadCount.x),
         (problemSize.y + (threadCount.y - 1)) / (threadCount.y),
         (problemSize.z + (threadCount.z - 1)) / (threadCount.z));
  return a;
}

inline RECLIB_HD vec3 float2vec(float3 in) { return vec3(in.x, in.y, in.z); }
inline RECLIB_HD uvec3 char2vec(uchar3 in) { return uvec3(in.x, in.y, in.z); }

}  // namespace cuda

template <typename T>
inline T* vec2ptr(GpuVec<T>& ptr) {
  return thrust::raw_pointer_cast(&(ptr[0]));
}

inline void getChunkedThreadBlocks(const dim3& problemSize, dim3& threads,
                                   dim3& blocks, uint32_t& chunks) {
  int max_threads_per_block = gpuMaxThreadsPerBlock(0);
  int max_threads_per_grid = gpuMaxThreadsPerSM(0);
  int warp_size = gpuWarpSize(0);
  int max_blocks[3];
  gpuMaxGridSize(&max_blocks[0]);
  int num_sm = gpuNumSM(0);
  max_threads_per_grid *= num_sm;

  int num_dims = (int)(problemSize.x > 1) + (int)(problemSize.y > 1) +
                 (int)(problemSize.z > 1);

  if (threads.x <= 1 && threads.y <= 1 && threads.z <= 1) {
    if (num_dims == 0) {
      threads.x = 32;
      threads.y = 1;
      threads.z = 1;
    } else if (num_dims == 1) {
      threads.x =
          std::min((float)max_threads_per_block,
                   std::ceil(problemSize.x / (float)warp_size) * warp_size);
      threads.y = 1;
      threads.z = 1;
    } else if (num_dims == 2 || logn(max_threads_per_block, warp_size) < 3) {
      threads.x = std::min(
          (float)std::pow(
              warp_size,
              (std::floor(logn(max_threads_per_block, warp_size) / 2))),
          std::ceil(problemSize.x / (float)warp_size) * warp_size);
      int remaining_threads = max_threads_per_block / threads.x;
      threads.y =
          std::min((float)std::floor(remaining_threads / warp_size) * warp_size,
                   std::ceil(problemSize.y / (float)warp_size) * warp_size);
      threads.z = 1;
    } else if (num_dims == 3) {
      threads.x = std::min(
          (float)std::pow(
              warp_size,
              (std::floor(logn(max_threads_per_block, warp_size) / 3))),
          std::ceil(problemSize.x / (float)warp_size) * (float)warp_size);

      threads.y = std::min(
          (float)std::pow(
              warp_size,
              (std::floor(logn(max_threads_per_block, warp_size) / 3))),
          std::ceil(problemSize.y / (float)warp_size) * warp_size);

      int remaining_threads = max_threads_per_block / (threads.x * threads.z);
      threads.z =
          std::min((float)std::floor(remaining_threads / warp_size) * warp_size,
                   std::ceil(problemSize.z / (float)warp_size) * warp_size);
    }
  }

  dim3 full_block_count = cuda::getBlockCount(problemSize, threads);
  // int max_blocks_per_grid =
  //     max_threads_per_grid / (threads.x * threads.y * threads.z);
  int max_blocks_per_grid = std::numeric_limits<int>::max();

  if (blocks.x <= 1 && blocks.y <= 1 && blocks.z <= 1) {
    if (num_dims == 0) {
      blocks.x = 1;
      blocks.y = 1;
      blocks.z = 1;
    } else if (num_dims == 1) {
      blocks.x = std::min(max_blocks_per_grid,
                          std::min(max_blocks[0], (int)full_block_count.x));
      blocks.y = 1;
      blocks.z = 1;
    } else if (num_dims == 2) {
      int max_blocks_per_grid_dim = std::floor(max_blocks_per_grid / 2.f);
      blocks.x = std::min(max_blocks_per_grid_dim,
                          std::min(max_blocks[0], (int)full_block_count.x));
      blocks.y = std::min(max_blocks_per_grid_dim,
                          std::min(max_blocks[1], (int)full_block_count.y));
      blocks.z = 1;
    } else if (num_dims == 3) {
      int max_blocks_per_grid_dim = std::floor(max_blocks_per_grid / 3.f);
      blocks.x = std::min(max_blocks_per_grid_dim,
                          std::min(max_blocks[0], (int)full_block_count.x));
      blocks.y = std::min(max_blocks_per_grid_dim,
                          std::min(max_blocks[1], (int)full_block_count.y));
      blocks.z = std::min(max_blocks_per_grid_dim,
                          std::min(max_blocks[2], (int)full_block_count.z));
    }

    // if (max_blocks[0] < full_block_count.x ||
    //     max_blocks[1] < full_block_count.y ||
    //     max_blocks[2] < full_block_count.z) {
    //   blocks.x = max_blocks[0];
    //   blocks.y = max_blocks[1];
    //   blocks.z = max_blocks[2];
    // } else {
    //   blocks.x = full_block_count.x;
    //   blocks.y = full_block_count.y;
    //   blocks.z = full_block_count.z;
    // }
  }

  int chunk_x = std::ceil(full_block_count.x / (float)blocks.x);
  int chunk_y = std::ceil(full_block_count.y / (float)blocks.y);
  int chunk_z = std::ceil(full_block_count.z / (float)blocks.z);
  chunks = std::max(chunk_x, std::max(chunk_y, chunk_z));

  _RECLIB_ASSERT_GT(threads.x, 0);
  _RECLIB_ASSERT_GT(threads.y, 0);
  _RECLIB_ASSERT_GT(threads.z, 0);

  _RECLIB_ASSERT_GT(blocks.x, 0);
  _RECLIB_ASSERT_GT(blocks.y, 0);
  _RECLIB_ASSERT_GT(blocks.z, 0);

  _RECLIB_ASSERT_GE(threads.x * blocks.x * chunks, problemSize.x);
  _RECLIB_ASSERT_GE(threads.y * blocks.y * chunks, problemSize.y);
  _RECLIB_ASSERT_GE(threads.z * blocks.z * chunks, problemSize.z);
}
}  // namespace reclib

#define THREAD_BLOCK(_problemSize, _threadCount) \
  reclib::cuda::getBlockCount(_problemSize, _threadCount), _threadCount

#endif