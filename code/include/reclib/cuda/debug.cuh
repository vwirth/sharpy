#ifndef CUDA_DEBUG_H
#define CUDA_DEBUG_H

#if WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <sstream>
#include <stdexcept>

#define cudaSafeCall(call)                                         \
  do {                                                             \
    cudaError_t err = call;                                        \
    if (cudaSuccess != err) {                                      \
      std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ \
                << "): " << cudaGetErrorString(err);               \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  } while (0)

#define CUDA_CHECK(call)                                    \
  {                                                         \
    cudaError_t rc = call;                                  \
    if (rc != cudaSuccess) {                                \
      std::stringstream txt;                                \
      cudaError_t err = rc; /*cudaGetLastError();*/         \
      txt << "CUDA Error " << cudaGetErrorName(err) << " (" \
          << cudaGetErrorString(err) << ")";                \
      throw std::runtime_error(txt.str());                  \
    }                                                       \
  }

#define CUDA_CHECK_NOEXCEPT(call) \
  { cuda##call; }

#define CUDA_SYNC_CHECK()                                              \
  {                                                                    \
    cudaDeviceSynchronize();                                           \
    cudaError_t error = cudaGetLastError();                            \
    if (error != cudaSuccess) {                                        \
      fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error));                              \
      exit(2);                                                         \
    }                                                                  \
  }

#endif
