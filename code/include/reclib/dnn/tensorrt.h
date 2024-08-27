#ifndef RECLIB_DNN_TENSORRT
#define RECLIB_DNN_TENSORRT

#if HAS_DNN_MODULE

#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <torch_tensorrt/core/compiler.h>
#include <torch_tensorrt/torch_tensorrt.h>

namespace reclib {
namespace dnn {
void silence_torchtrt_logger();

torch::jit::script::Module torchscript_to_tensorrt(
    const torch::jit::Module& ts_module,
    const std::vector<torch::Tensor>& input, bool use_fp16 = false);

}  // namespace dnn
}  // namespace reclib

#endif

#endif