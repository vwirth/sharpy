#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

#if WITH_CUDA
#include <cuda_runtime.h>
#endif
#include <stdio.h>

namespace reclib {
inline int getDevice() {
  int device;
  cudaGetDevice(&device);
  return device;
}

inline void printGpus() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  }
}

inline int gpuSharedMemPerBlock(int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  return deviceProp.sharedMemPerBlock;  // in bytes
}

inline int gpuComputeCapability(int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  return deviceProp.minor;
}

inline int gpuMaxThreadsPerBlock(int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  return deviceProp.maxThreadsPerBlock;
}

inline int gpuWarpSize(int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  return deviceProp.warpSize;
}

inline int gpuMaxThreadsPerSM(int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  return deviceProp.maxThreadsPerMultiProcessor;
}

inline int gpuNumSM(int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  return deviceProp.multiProcessorCount;
}

inline void gpuMaxThreadsDim(int* val, int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  val[0] = deviceProp.maxThreadsDim[0];
  val[1] = deviceProp.maxThreadsDim[1];
  val[2] = deviceProp.maxThreadsDim[2];
}

inline void gpuMaxGridSize(int* val, int device_num = 0) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_num);
  val[0] = deviceProp.maxGridSize[0];
  val[1] = deviceProp.maxGridSize[1];
  val[2] = deviceProp.maxGridSize[2];
}
}  // namespace reclib

#endif