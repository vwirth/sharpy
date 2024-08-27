#ifndef RECLIB_DNN_NVDIFFRAST_AUTOGRAD_H
#define RECLIB_DNN_NVDIFFRAST_AUTOGRAD_H

#if HAS_DNN_MODULE
#if HAS_NVDIFFRAST_MODULE
#include <nvdiffrast/torch/torch_types.h>
#include <torch/torch.h>

namespace reclib {
namespace dnn {
class RasterizeFunc : public torch::autograd::Function<RasterizeFunc> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext* ctx, RasterizeCRStateWrapper& context,
      torch::Tensor positions, torch::Tensor triangles,
      std::tuple<int, int> resolution,
      torch::Tensor ranges =
          torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kInt)),
      bool grad_db = true, int peeling_idx = -1);

  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> backward_inputs);
};

std::vector<torch::Tensor> rasterize(
    RasterizeCRStateWrapper& context, torch::Tensor positions,
    torch::Tensor triangles, std::tuple<int, int> resolution,
    torch::Tensor ranges =
        torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kInt)),
    bool grad_db = true, int peeling_idx = -1);

class InterpolateFunc : public torch::autograd::Function<InterpolateFunc> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext* ctx, torch::Tensor attr,
      torch::Tensor rast, torch::Tensor tri,
      torch::Tensor rast_db = torch::empty({0}), bool diff_attrs_all = false,
      std::vector<int> diff_attrs = std::vector<int>());

  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> backward_inputs);
};

std::vector<torch::Tensor> interpolate(
    torch::Tensor attr, torch::Tensor rast, torch::Tensor tri,
    torch::Tensor rast_db = torch::empty({0}), bool diff_attrs_all = false,
    std::vector<int> diff_attrs = std::vector<int>());

class AntialiasFunc : public torch::autograd::Function<AntialiasFunc> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext* ctx, torch::Tensor color,
      torch::Tensor rast, torch::Tensor pos, torch::Tensor tri,
      TopologyHashWrapper topology_hash_wrap = TopologyHashWrapper(),
      float pos_gradient_boost = 1.f);

  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> backward_inputs);
};

std::vector<torch::Tensor> antialias(
    torch::Tensor color, torch::Tensor rast, torch::Tensor pos,
    torch::Tensor tri,
    TopologyHashWrapper topology_hash_wrap = TopologyHashWrapper(),
    float pos_gradient_boost = 1.f);

}  // namespace dnn
}  // namespace reclib

#endif  // HAS_NVDIFFRAST_MODULE
#endif  // HAS_DNN_MODULE

#endif  // RECLIB_DNN_NVDIFFRAST_AUTOGRAD_H