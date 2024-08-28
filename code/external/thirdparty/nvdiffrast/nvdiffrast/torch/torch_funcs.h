#ifndef NVDIFFRAST_TORCH_FUNCS_H
#define NVDIFFRAST_TORCH_FUNCS_H

#include "../common/common.h"
#include "../common/cudaraster/CudaRaster.hpp"
#include "../common/cudaraster/impl/Constants.hpp"
#include "../common/rasterize.h"
#include "torch_types.h"

namespace nvdiffrast {

// ----------------------------------------
// Rasterization
// ----------------------------------------
std::tuple<torch::Tensor, torch::Tensor> rasterize_fwd_cuda(
    RasterizeCRStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri,
    std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx);

torch::Tensor rasterize_grad_db(torch::Tensor pos, torch::Tensor tri,
                                torch::Tensor out, torch::Tensor dy,
                                torch::Tensor ddb);

torch::Tensor rasterize_grad(torch::Tensor pos, torch::Tensor tri,
                             torch::Tensor out, torch::Tensor dy);

std::tuple<torch::Tensor, torch::Tensor> rasterize_fwd_gl(
    RasterizeGLStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri,
    std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx);

// Interpolation
// Rasterization
// ----------------------------------------
std::tuple<torch::Tensor, torch::Tensor> interpolate_fwd(torch::Tensor attr,
                                                         torch::Tensor rast,
                                                         torch::Tensor tri);

std::tuple<torch::Tensor, torch::Tensor> interpolate_fwd_da(
    torch::Tensor attr, torch::Tensor rast, torch::Tensor tri,
    torch::Tensor rast_db, bool diff_attrs_all,
    std::vector<int>& diff_attrs_vec);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolate_grad_da(
    torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy,
    torch::Tensor rast_db, torch::Tensor dda, bool diff_attrs_all,
    std::vector<int>& diff_attrs_vec);

std::tuple<torch::Tensor, torch::Tensor> interpolate_grad(torch::Tensor attr,
                                                          torch::Tensor rast,
                                                          torch::Tensor tri,
                                                          torch::Tensor dy);

// ----------------------------------------
// Antialiasing
// ----------------------------------------
TopologyHashWrapper antialias_construct_topology_hash(torch::Tensor tri);

std::tuple<torch::Tensor, torch::Tensor> antialias_fwd(
    torch::Tensor color, torch::Tensor rast, torch::Tensor pos,
    torch::Tensor tri, TopologyHashWrapper topology_hash_wrap);

std::tuple<torch::Tensor, torch::Tensor> antialias_grad(
    torch::Tensor color, torch::Tensor rast, torch::Tensor pos,
    torch::Tensor tri, torch::Tensor dy, torch::Tensor work_buffer);

}  // namespace nvdiffrast

#endif