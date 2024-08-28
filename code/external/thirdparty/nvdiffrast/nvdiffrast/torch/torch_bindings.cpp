// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <tuple>

#include "torch_types.h"

//------------------------------------------------------------------------
// Op prototypes. Return type macros for readability.

#define OP_RETURN_T torch::Tensor
#define OP_RETURN_TT std::tuple<torch::Tensor, torch::Tensor>
#define OP_RETURN_TTT std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTTT \
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTV \
  std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >
#define OP_RETURN_TTTTV                                                  \
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, \
             std::vector<torch::Tensor> >

OP_RETURN_TT rasterize_fwd_cuda(RasterizeCRStateWrapper& stateWrapper,
                                torch::Tensor pos, torch::Tensor tri,
                                std::tuple<int, int> resolution,
                                torch::Tensor ranges, int peeling_idx);
OP_RETURN_T rasterize_grad(torch::Tensor pos, torch::Tensor tri,
                           torch::Tensor out, torch::Tensor dy);
OP_RETURN_T rasterize_grad_db(torch::Tensor pos, torch::Tensor tri,
                              torch::Tensor out, torch::Tensor dy,
                              torch::Tensor ddb);
OP_RETURN_TT interpolate_fwd(torch::Tensor attr, torch::Tensor rast,
                             torch::Tensor tri);
OP_RETURN_TT interpolate_fwd_da(torch::Tensor attr, torch::Tensor rast,
                                torch::Tensor tri, torch::Tensor rast_db,
                                bool diff_attrs_all,
                                std::vector<int>& diff_attrs_vec);
OP_RETURN_TT interpolate_grad(torch::Tensor attr, torch::Tensor rast,
                              torch::Tensor tri, torch::Tensor dy);
OP_RETURN_TTT interpolate_grad_da(torch::Tensor attr, torch::Tensor rast,
                                  torch::Tensor tri, torch::Tensor dy,
                                  torch::Tensor rast_db, torch::Tensor dda,
                                  bool diff_attrs_all,
                                  std::vector<int>& diff_attrs_vec);
TextureMipWrapper texture_construct_mip(torch::Tensor tex, int max_mip_level,
                                        bool cube_mode);
OP_RETURN_T texture_fwd(torch::Tensor tex, torch::Tensor uv, int filter_mode,
                        int boundary_mode);
OP_RETURN_T texture_fwd_mip(torch::Tensor tex, torch::Tensor uv,
                            torch::Tensor uv_da, torch::Tensor mip_level_bias,
                            TextureMipWrapper mip_wrapper,
                            std::vector<torch::Tensor> mip_stack,
                            int filter_mode, int boundary_mode);
OP_RETURN_T texture_grad_nearest(torch::Tensor tex, torch::Tensor uv,
                                 torch::Tensor dy, int filter_mode,
                                 int boundary_mode);
OP_RETURN_TT texture_grad_linear(torch::Tensor tex, torch::Tensor uv,
                                 torch::Tensor dy, int filter_mode,
                                 int boundary_mode);
OP_RETURN_TTV texture_grad_linear_mipmap_nearest(
    torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da,
    torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper,
    std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
OP_RETURN_TTTTV texture_grad_linear_mipmap_linear(
    torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da,
    torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper,
    std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
TopologyHashWrapper antialias_construct_topology_hash(torch::Tensor tri);
OP_RETURN_TT antialias_fwd(torch::Tensor color, torch::Tensor rast,
                           torch::Tensor pos, torch::Tensor tri,
                           TopologyHashWrapper topology_hash);
OP_RETURN_TT antialias_grad(torch::Tensor color, torch::Tensor rast,
                            torch::Tensor pos, torch::Tensor tri,
                            torch::Tensor dy, torch::Tensor work_buffer);

//------------------------------------------------------------------------

//------------------------------------------------------------------------
