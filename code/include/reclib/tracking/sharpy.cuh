#ifndef RECLIB_TRACKING_SHARPY_CUH
#define RECLIB_TRACKING_SHARPY_CUH

#if HAS_DNN_MODULE
#if WITH_CUDA

#include "reclib/camera_parameters.h"
#include "reclib/data_types.h"

// clang-format off
#include <torch/torch.h>
// clang-format on

namespace reclib {
namespace tracking {
namespace cuda {
void compute_segmentation_from_corrs(torch::Tensor& input,
                                     torch::Tensor& segmentation);
void compute_segmentation_from_corrs(torch::Tensor& input,
                                     torch::Tensor& segmentation,
                                     std::vector<uint8_t>& visible_ids);
}  // namespace cuda
}  // namespace tracking
}  // namespace reclib

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE

#endif