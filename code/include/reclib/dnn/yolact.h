#ifndef RECLIB_DNN_YOLACT_H
#define RECLIB_DNN_YOLACT_H

#if HAS_DNN_MODULE

#include <ATen/TensorIndexing.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/torch.h>

#include "reclib/data_types.h"
#include "reclib/internal/filesystem.h"
#include "reclib/opengl/query.h"

namespace reclib {
namespace dnn {
class Transform {
 public:
  Transform();
  virtual ~Transform();

  virtual std::vector<torch::Tensor> transform(
      const std::vector<torch::Tensor>& input) = 0;

  virtual std::vector<torch::Tensor> untransform(
      const std::vector<torch::Tensor>& input, unsigned int im_w,
      unsigned int im_h) = 0;
};

class Resize : public Transform {
  bool preserve_aspect_ratio_;
  unsigned int max_size_;
  int discard_box_width_;
  int discard_box_height_;
  bool resize_gt_;

 public:
  Resize(bool preserve_aspect_ratio, unsigned int max_size,
         int discard_box_width, int discard_box_height, bool resize_gt = false);

  std::vector<torch::Tensor> transform(
      const std::vector<torch::Tensor>& input) override;

  std::vector<torch::Tensor> untransform(
      const std::vector<torch::Tensor>& input, unsigned int im_w,
      unsigned int im_h) override;
};

class Pad : public Transform {
  bool preserve_aspect_ratio_;
  unsigned int width_;
  unsigned int height_;
  std::vector<float> mean_;
  bool pad_gt_;

 public:
  Pad(bool preserve_aspect_ratio, unsigned int width, unsigned int height,
      std::vector<float> mean, bool pad_gt = false);

  std::vector<torch::Tensor> transform(
      const std::vector<torch::Tensor>& input) override;
  std::vector<torch::Tensor> untransform(
      const std::vector<torch::Tensor>& input, unsigned int im_w,
      unsigned int im_h) override;
};

class BackboneTransform : public Transform {
  std::string channel_order_;
  std::string in_channel_order_;
  bool normalize_;
  bool subtract_means_;
  bool to_float_;
  std::vector<float> mean_;
  std::vector<float> std_;

  std::vector<int> permutation_order_;

 public:
  BackboneTransform(const std::string& channel_order, bool normalize,
                    bool subtract_means, bool to_float,
                    const std::vector<float>& mean,
                    const std::vector<float> std,
                    const std::string& in_channel_order = "RGB");

  std::vector<torch::Tensor> transform(
      const std::vector<torch::Tensor>& input) override;
  std::vector<torch::Tensor> untransform(
      const std::vector<torch::Tensor>& input, unsigned int im_w,
      unsigned int im_h) override;
};

class BaseTransform : public Transform {
  // we need a vector of shared pointers instead of only the transforms
  // see https://en.wikipedia.org/wiki/Object_slicing
  std::vector<std::shared_ptr<reclib::dnn::Transform>> transforms_;

 public:
  BaseTransform(unsigned int max_size, bool preserve_aspect_ratio = false,
                const std::string& in_channel_order = "BGR",
                const std::vector<float>& mean = {103.94f, 116.78f, 123.68f},
                const std::vector<float>& std = {57.38f, 57.12f, 58.40f});

  std::vector<torch::Tensor> transform(
      const std::vector<torch::Tensor>& input) override;
  std::vector<torch::Tensor> untransform(
      const std::vector<torch::Tensor>& input, unsigned int im_w,
      unsigned int im_h) override;
};

class Yolact {
 public:
  const fs::path BACKBONE_PATH = "backbone.ts";
  const fs::path FPN_PATH = "fpn.ts";
  const fs::path PROTO_NET_PATH = "proto_net.ts";
  const fs::path PROTO_NET_CORR_PATH = "proto_net_corr.ts";
  const fs::path PRED_MOD0_PATH = "prediction_layer_0.ts";
  const fs::path PRED_MOD1_PATH = "prediction_layer_1.ts";
  const fs::path PRED_MOD2_PATH = "prediction_layer_2.ts";
  const fs::path PRED_MOD3_PATH = "prediction_layer_3.ts";
  const fs::path PRED_MOD4_PATH = "prediction_layer_4.ts";

  // ---------------------------------------------------------------
  // MODEL PARAMETERS
  // ---------------------------------------------------------------
  const unsigned int MAX_INPUT_SIZE = 550;
  const float CONF_THRESH = 0.00f;  // 0.05f
  const float NMS_THRESH = 0.5f;
  const unsigned int MAX_NUM_DETECTIONS = 100;
  const unsigned int TOP_K = 200;
  const unsigned int NUM_CLASSES = 3;

  const std::vector<int64_t> backbone_input_ = {1, 3, 550, 550};
  // subsequent module inputs multiple 2nd entry with 2
  // and divide last 2 entries by 2
  const std::vector<int64_t> fpn_input_ = {1, 512, 69, 69};
  const std::vector<int64_t> proto_net_input_ = {1, 256, 69, 69};
  // const std::vector<int64_t> proto_net_corr_input_ = {1, 32, 138, 138};
  const std::vector<int64_t> proto_net_corr_input_ = {1, 256, 69, 69};
  // subsequent module inputs divide the last 2 entries by 2
  const std::vector<int64_t> prediction_module_input_ = {1, 256, 69, 69};

  const std::vector<float> scales_ = {24, 48, 96, 192, 384};
  const std::vector<std::vector<float>> aspect_ratios_ = {
      {1, 0.5f, 2}, {1, 0.5f, 2}, {1, 0.5f, 2}, {1, 0.5f, 2}, {1, 0.5f, 2},
  };

 private:
  // ---------------------------------------------------------------
  // ---------------------------------------------------------------
  // ---------------------------------------------------------------

  bool use_cuda_;
  ::torch::jit::script::Module backbone_;
  ::torch::jit::script::Module fpn_;
  ::torch::jit::script::Module proto_net_;
  ::torch::jit::script::Module proto_net_corr_;
  std::vector<::torch::jit::script::Module> prediction_modules_;
  std::vector<::torch::Tensor> prediction_priors_;

  reclib::opengl::Timer timer_;

  void load(const fs::path& base_path, bool use_cuda = true,
            bool use_trt = true, bool use_fp16 = true,
            bool disable_cache = false);

  static void print_output(const c10::IValue& val,
                           std::string output_string = "tensor",
                           int recursion_depth = 0);

  static torch::Tensor decode(const torch::Tensor& loc,
                              const torch::Tensor& priors,
                              bool use_yolo_regressors = false);

  static torch::Tensor make_priors(float scale,
                                   const std::vector<float>& aspect_ratios,
                                   unsigned int conv_h, unsigned int conv_w,
                                   const c10::Device& device = torch::kCUDA,
                                   float max_size = 550);

  static std::vector<torch::Tensor> detect(
      int batch_idx, const torch::Tensor& conf_preds,
      const torch::Tensor& decoded_boxes, const torch::Tensor& mask_data,
      float conf_thresh = 0.05f, float nms_thresh = 0.5f,
      int max_num_detections = 100, int top_k = 200);

  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  fast_nms(torch::Tensor boxes, torch::Tensor masks, torch::Tensor scores,
           float conf_thresh, int max_num_detections,
           float iou_threshold = 0.5f, int top_k = 200,
           bool second_threshold = false);

  static torch::Tensor intersect(const torch::Tensor& box_a,
                                 const torch::Tensor& box_b);

  static torch::Tensor jaccard(torch::Tensor box_a, torch::Tensor box_b,
                               bool iscrowd = false);

 public:
  Yolact(const fs::path& base_path, bool use_cuda = true, bool use_trt = true,
         bool use_fp16 = true, bool disable_cache = false);

  std::vector<torch::Tensor> forward(const torch::Tensor& input,
                                     bool benchmark = false);

  std::vector<torch::Tensor> postprocess(
      const std::vector<std::vector<torch::Tensor>>& net_output, unsigned int w,
      unsigned int h, float score_threshold = 0, bool crop_masks = true,
      bool benchmark = false);

  std::vector<std::vector<torch::Tensor>> detect(
      const std::vector<torch::Tensor>& pred_outs);
};
}  // namespace dnn
}  // namespace reclib

#endif  // HAS_DNN_MODULE

#endif