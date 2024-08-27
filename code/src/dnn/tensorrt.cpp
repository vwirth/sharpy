#include "reclib/dnn/tensorrt.h"

#if HAS_DNN_MODULE

void reclib::dnn::silence_torchtrt_logger() {
  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kERROR);
}

torch::jit::script::Module reclib::dnn::torchscript_to_tensorrt(
    const torch::jit::Module& ts_module,
    const std::vector<torch::Tensor>& inputs, bool use_fp16) {
  std::vector<torch_tensorrt::core::ir::Input> input_specs;
  for (unsigned int i = 0; i < inputs.size(); i++) {
    input_specs.push_back(inputs[i].sizes().vec());
  }

  torch_tensorrt::core::CompileSpec spec(input_specs);

  if (use_fp16) {
    spec.convert_info.engine_settings.enabled_precisions.insert(
        nvinfer1::DataType::kHALF);
  }

  return torch_tensorrt::core::CompileGraph(ts_module, spec);
}

#endif