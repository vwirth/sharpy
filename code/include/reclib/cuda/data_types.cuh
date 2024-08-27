#ifndef RECLIB_CUDA_DATATYPES_CUH
#define RECLIB_CUDA_DATATYPES_CUH

#if WITH_CUDA
#include <cuda_runtime.h>
#endif  // WITH_CUDA

#include "reclib/cuda/debug.cuh"
#include "reclib/data_types.h"

template <typename T>
inline T* upload_ptr(const T* cpu_ptr, uint32_t num_elements) {
#if WITH_CUDA
  T* cuda_ptr;
  CUDA_CHECK(cudaMalloc((void**)&cuda_ptr, num_elements * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(cuda_ptr, cpu_ptr, num_elements * sizeof(T),
                        cudaMemcpyHostToDevice));
  return cuda_ptr;
#else
  throw std::runtime_error(
      "Error. Attempt to call cuda functions without cuda enabled.");
#endif
}

template <typename T>
inline void download_ptr(T* cuda_ptr, T* cpu_ptr, uint32_t num_elements) {
#if WITH_CUDA
  CUDA_CHECK(cudaMemcpy(cpu_ptr, cuda_ptr, num_elements * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(cuda_ptr));
#else
  throw std::runtime_error(
      "Error. Attempt to call cuda functions without cuda enabled.");
#endif
}

template <typename T>
inline T* upload_eigen(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cpu_mat) {
#if WITH_CUDA
  T* cuda_ptr;
  CUDA_CHECK(cudaMalloc((void**)&cuda_ptr,
                        cpu_mat.rows() * cpu_mat.cols() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(cuda_ptr, cpu_mat.data(),
                        cpu_mat.rows() * cpu_mat.cols() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return cuda_ptr;
#else
  throw std::runtime_error(
      "Error. Attempt to call cuda functions without cuda enabled.");
#endif
}

template <typename T>
inline void download_eigen(
    T* cuda_ptr, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cpu_mat) {
#if WITH_CUDA
  CUDA_CHECK(cudaMemcpy(cpu_mat.data(), cuda_ptr,
                        cpu_mat.rows() * cpu_mat.cols() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(cuda_ptr));
#else
  throw std::runtime_error(
      "Error. Attempt to call cuda functions without cuda enabled.");
#endif
}

#endif