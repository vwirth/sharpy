#ifndef THREAD_INFO_H
#define THREAD_INFO_H

#include <cuda_occupancy.h>

#include <algorithm>

namespace reclib {
namespace cuda {
/**
 * A simple helper struct that can be used at the beginning of each kernel to
 * compute some usefull variables. Don't worry about variables that you are not
 * going going to use, because they are optimized away :).
 *
 * If you know the number of threads per block at compile time pass it as the
 * first template argument. LOCAL_WARP_SIZE can be used for example in partial
 * warp reductions. LOCAL_WARP_SIZE must be one of these values: 1,2,4,8,16,32
 *
 * Note: In some libraries you will find the __mul24 for these index
 * computations, but from cuda 8 api:
 *  "_[u]mul24 are legacy intrinsic functions that have no longer any reason to
 * be used"
 *
 * Note: It is important to use unsigend ints here, because then the compiler
 * can replace expensive integer divisions with fast bit shifts. The same counts
 * when accessing these variables. Either use "auto" or "unsigend int".
 */
template <unsigned int THREADS_PER_BLOCK = 0, unsigned int LOCAL_WARP_SIZE = 32>
struct ThreadInfo {
  unsigned int local_thread_id;  // local thread id in the block PTX: %tid.x
  unsigned int thread_id;        // global thread index
  unsigned int block_id;         // id of this block PTX: %ctaid.x
  unsigned int lane_id;          // thread index within the warp
  unsigned int warp_id;          // global warp index

  unsigned int threads_per_block;
  unsigned int warp_lane;        // warp index within the block
  unsigned int num_blocks;       // number of blocks in that grid PTX: %nctaid.x
  unsigned int num_warps_block;  // total number of active warps
  unsigned int num_warps;        // total number of active warps
  unsigned int grid_size;        // total number of threads in the grid

  RECLIB_DEVICE ThreadInfo() {
    if (THREADS_PER_BLOCK > 0) {
      threads_per_block = THREADS_PER_BLOCK;
    } else {
      threads_per_block = blockDim.x;
    }
    local_thread_id = threadIdx.x;
    block_id = blockIdx.x;
    num_blocks = gridDim.x;

    // Note: Beccause of this line LOCAL_WARP_SIZE must be a power of 2
    lane_id = local_thread_id & (LOCAL_WARP_SIZE - 1);
    thread_id = threads_per_block * block_id + local_thread_id;

    grid_size = num_blocks * threads_per_block;

    warp_id = thread_id / LOCAL_WARP_SIZE;
    warp_lane = local_thread_id / LOCAL_WARP_SIZE;
    num_warps_block = threads_per_block / LOCAL_WARP_SIZE;
    num_warps = num_warps_block * num_blocks;
  }
};

struct KernelInfo {
  int block_size;
  int max_blocks_per_sm;
  int max_blocks_on_device;

  int LaunchGrid(int n) {
    return std::min((n + block_size - 1) / block_size, max_blocks_on_device);
  }
};

template <typename K>
KernelInfo GetKernelInfo(K kernel, int block_size, int dynamic_smem = 0) {
  cudaDeviceProp props;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  int num_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel, block_size,
                                                dynamic_smem);

  KernelInfo ki;
  ki.block_size = block_size;
  ki.max_blocks_per_sm = num_blocks;
  ki.max_blocks_on_device = ki.max_blocks_per_sm * props.multiProcessorCount;
  return ki;
}
}  // namespace cuda
}  // namespace reclib

#endif