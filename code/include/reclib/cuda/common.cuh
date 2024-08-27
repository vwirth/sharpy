#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

// CUDA kernel header with commonly used definitions, functions and data
// structures Author: Christian Diller, git@christian-diller.de

#if WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <Eigen/Eigen>

#if HAS_OPENCV_MODULE
#include <opencv2/core/cuda.hpp>
#endif

#include "reclib/camera_parameters.h"
#include "reclib/data_types.h"

// If you are working with CUDA code in CLion
#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define RECLIB_HOST
#define RECLIB_DEVICE
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
using blockDim = struct {
  int x;
  int y;
};
using threadIdx = struct {
  int x;
  int y;
  int z;
};
using blockIdx = struct {
  int x;
  int y;
  int z;
};
#endif

#define DIVSHORTMAX 0.0000305185f  // 1.f / SHRT_MAX;
#define SHORTMAX 32767             // SHRT_MAX;

namespace reclib {
namespace cuda {
RECLIB_DEVICE __forceinline__ ivec2da
project(const vec3& camera_pos, const IntrinsicParameters& cam_params) {
  return ivec2da(
      __float2int_rn(camera_pos.x() / camera_pos.z() * cam_params.focal_x_ +
                     cam_params.principal_x_),
      __float2int_rn(camera_pos.y() / camera_pos.z() * cam_params.focal_y_ +
                     cam_params.principal_y_));
}

RECLIB_DEVICE __forceinline__ vec3
backproject(const ivec2da& uv, const float depth,
            const IntrinsicParameters& cam_params) {
  return vec3((uv.x() - cam_params.principal_x_) / cam_params.focal_x_ * depth,
              (uv.y() - cam_params.principal_y_) / cam_params.focal_y_ * depth,
              depth);
}

// converts normalized values in [0,1] to image range in [0, N]
// where N is 2^P -1 where P is the bit-precision
template <typename V, typename I>
RECLIB_DEVICE __forceinline__ I
val2img(V value, V multiplier = (pow(2, sizeof(I) * 8) - 1)) {
  return I(value * multiplier);
}

// converts value from image range to any other
// data type in normalized range in [0,1]
template <typename I, typename V>
RECLIB_DEVICE __forceinline__ V img2val(I value,
                                     V divisor = (pow(2, sizeof(I) * 8) - 1)) {
  return V(value) / V(divisor);
}

// Lane Identifier. (id within a 32 warp)
static RECLIB_DEVICE __forceinline__ unsigned int kernel_lane_ID() {
  unsigned int ret;
  asm("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

// 32-bit mask with bits set in positions less than the thread's lane number in
// the warp.
static RECLIB_DEVICE __forceinline__ int kernel_laneMaskLt() {
  unsigned int ret;
  asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret));
  return ret;
}

// return rank of thread in range [0, active_threads]
static RECLIB_DEVICE __forceinline__ int kernel_binaryExclScan(int ballot_mask) {
  return __popc(kernel_laneMaskLt() & ballot_mask);
}

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)                                \
  (byte & 0x80 ? '1' : '0'), (byte & 0x40 ? '1' : '0'),     \
      (byte & 0x20 ? '1' : '0'), (byte & 0x10 ? '1' : '0'), \
      (byte & 0x08 ? '1' : '0'), (byte & 0x04 ? '1' : '0'), \
      (byte & 0x02 ? '1' : '0'), (byte & 0x01 ? '1' : '0')

// warp-aggregated increment
static RECLIB_DEVICE int kernel_atomicWarpAggregatedIncrement(int* ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  int lane_mask = kernel_laneMaskLt();
  unsigned int rank = __popc(active & kernel_laneMaskLt());
  int warp_res;
  if (rank == 0) warp_res = atomicAdd(ctr, change);
  warp_res = __shfl_sync(active, warp_res, leader);

  return warp_res + rank;
}

}  // namespace cuda

}  // namespace reclib

#endif
