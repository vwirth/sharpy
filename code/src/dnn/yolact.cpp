
#if HAS_DNN_MODULE

#include <ATen/TensorIndexing.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <reclib/dnn/tensorrt.h>
#include <reclib/dnn/yolact.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch_tensorrt/core/compiler.h>
#include <torch_tensorrt/torch_tensorrt.h>

#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <reclib/utils/torch_utils.h>

// ---------------------------------------------------------
// Transform
// --------------------------------------------------------

reclib::dnn::Transform::Transform() {}

std::vector<torch::Tensor> reclib::dnn::Transform::transform(
    const std::vector<torch::Tensor>& input) {
  std::vector<torch::Tensor> out;
  out.push_back(torch::zeros({1, 1}));
  return out;
}

reclib::dnn::Transform::~Transform(){};

// ---------------------------------------------------------
// Pad
// --------------------------------------------------------

reclib::dnn::Pad::Pad(bool preserve_aspect_ratio, unsigned int width,
                      unsigned int height, std::vector<float> mean, bool pad_gt)
    : preserve_aspect_ratio_(preserve_aspect_ratio),
      width_(width),
      height_(height),
      mean_(mean),
      pad_gt_(pad_gt) {}

std::vector<torch::Tensor> reclib::dnn::Pad::transform(
    const std::vector<torch::Tensor>& input) {
  torch::Tensor image = input[0];
  // check if 3rd dimension corresponds to the number of channels
  _RECLIB_ASSERT_LT(image.sizes()[2], 5);

  torch::Tensor masks;
  torch::Tensor boxes;
  if (input.size() > 1) {
    masks = input[1];
    _RECLIB_ASSERT_EQ(masks.sizes()[0], image.sizes()[0]);
    _RECLIB_ASSERT_EQ(masks.sizes()[1], image.sizes()[1]);
  }
  if (input.size() > 2) {
    boxes = input[2];
  }

  unsigned int im_h = image.sizes()[0];
  unsigned int im_w = image.sizes()[1];
  unsigned int depth = image.sizes()[2];

  torch::Tensor expand_image = torch::zeros({height_, width_, depth});
  expand_image.index({torch::All, torch::All, torch::All}) =
      torch::tensor(mean_);

  expand_image.index({torch::indexing::Slice({torch::None, im_h}),
                      torch::indexing::Slice({torch::None, im_w})}) =
      image.index({torch::indexing::Slice({torch::None, im_h}),
                   torch::indexing::Slice({torch::None, im_w})});

  if (pad_gt_ && masks.numel() > 0) {
    torch::Tensor expand_masks =
        torch::zeros({masks.sizes()[0], height_, width_});
    expand_masks.index({torch::All, torch::indexing::Slice({torch::None, im_h}),
                        torch::indexing::Slice({torch::None, im_w})}) = masks;
    masks = expand_masks;
  }

  std::vector<torch::Tensor> out;
  out.push_back(expand_image);
  if (masks.numel() > 0) {
    out.push_back(masks);
  }
  if (boxes.numel() > 0) {
    out.push_back(boxes);
  }
  for (unsigned int i = 3; i < input.size(); i++) {
    out.push_back(input[i]);
  }

  return out;
}

std::vector<torch::Tensor> reclib::dnn::Pad::untransform(
    const std::vector<torch::Tensor>& input, unsigned int img_w,
    unsigned int img_h) {
  std::vector<torch::Tensor> out = input;

  torch::Tensor masks;
  torch::Tensor boxes;
  torch::Tensor corrs;

  unsigned int h = 0;
  unsigned int w = 0;
  if (input.size() >= 4) {
    masks = input[3];
  }
  if (input.size() >= 3) {
    boxes = input[2];
  }
  if (input.size() >= 5) {
    corrs = input[4];
  }

  if (preserve_aspect_ratio_) {
    if (img_w > img_h) {
      float ratio = (float)img_h / (float)img_w;
      w = width_;
      h = int(height_ * ratio);
    } else {
      float ratio = (float)img_w / (float)img_h;
      h = height_;
      w = int(width_ * ratio);
    }
  } else {
    w = width_;
    h = height_;
  }

  if (masks.numel() > 0) {
    masks = masks.index({torch::All, torch::indexing::Slice(0, h),
                         torch::indexing::Slice(0, w)});
  }
  if (corrs.numel() > 0) {
    corrs = corrs.index({torch::All, torch::All, torch::indexing::Slice(0, h),
                         torch::indexing::Slice(0, w)});
  }

  if (masks.numel() > 0) {
    out[3] = masks;
  }
  if (boxes.numel() > 0) {
    out[2] = boxes;
  }
  if (corrs.numel() > 0) {
    out[4] = corrs;
  }
  return out;
}

#if HAS_OPENCV_MODULE
// ---------------------------------------------------------
// Resize
// --------------------------------------------------------

reclib::dnn::Resize::Resize(bool preserve_aspect_ratio, unsigned int max_size,
                            int discard_box_width, int discard_box_height,
                            bool resize_gt)
    : preserve_aspect_ratio_(preserve_aspect_ratio),
      max_size_(max_size),
      discard_box_width_(discard_box_width),
      discard_box_height_(discard_box_height),
      resize_gt_(resize_gt) {}

std::vector<torch::Tensor> reclib::dnn::Resize::transform(
    const std::vector<torch::Tensor>& input) {
  torch::Tensor image = input[0];
  // check if 3rd dimension corresponds to the number of channels
  _RECLIB_ASSERT_LT(image.sizes()[2], 5);

  torch::Tensor masks;
  torch::Tensor boxes;

  unsigned int img_h = image.sizes()[0];
  unsigned int img_w = image.sizes()[1];
  unsigned int depth = image.sizes()[2];
  int w = 0;
  int h = 0;

  if (input.size() > 1) {
    masks = input[1];
    _RECLIB_ASSERT_EQ(masks.sizes()[0], image.sizes()[0]);
    _RECLIB_ASSERT_EQ(masks.sizes()[1], image.sizes()[1]);
  }
  if (input.size() > 2) {
    boxes = input[2];
  }

  if (preserve_aspect_ratio_) {
    if (img_w > img_h) {
      float ratio = (float)img_h / (float)img_w;
      w = max_size_;
      h = int(max_size_ * ratio);
    } else {
      float ratio = (float)img_w / (float)img_h;
      h = max_size_;
      w = int(max_size_ * ratio);
    }
  } else {
    w = max_size_;
    h = max_size_;
  }

  torch::Tensor image_rgb = image.index(
      {torch::All, torch::All, torch::indexing::Slice(torch::None, 3)});

  image_rgb = torch::nn::functional::interpolate(
                  image_rgb.permute({2, 0, 1}).unsqueeze(0),
                  torch::nn::functional::InterpolateFuncOptions()
                      .size(std::vector<int64_t>({h, w}))
                      .mode(torch::kBilinear)
                      .align_corners(false))
                  .squeeze(0)
                  .permute({1, 2, 0});

  if (depth == 4) {
    torch::Tensor image_d = image.index({torch::All, torch::All, 3});

    image_d = torch::nn::functional::interpolate(
                  image_d.unsqueeze(0),
                  torch::nn::functional::InterpolateFuncOptions()
                      .size(std::vector<int64_t>({h, w}))
                      .mode(torch::kNearest)
                      .align_corners(false))
                  .squeeze(0);
    // cv::Mat image_d_wrapped = reclib::dnn::torch2cv(image_d, true);
    // cv::resize(image_d_wrapped, image_d_wrapped, cv::Size(w, h), 0, 0,
    //            cv::INTER_NEAREST);

    image = torch::cat(
        {image_rgb, image_d.index({torch::All, torch::All, torch::None})}, -1);
  } else {
    image = image_rgb;
  }

  if (resize_gt_) {
    if (masks.numel() > 0) {
      masks = masks.permute({1, 2, 0});
      masks = torch::nn::functional::interpolate(
                  masks.unsqueeze(0),
                  torch::nn::functional::InterpolateFuncOptions()
                      .size(std::vector<int64_t>({h, w}))
                      .mode(torch::kBilinear)
                      .align_corners(false))
                  .squeeze(0);

      // cv::Mat masks_wrapped(masks.sizes()[0], masks.sizes()[1], CV_32FC1,
      //                       masks.data_ptr());
      // cv::resize(masks_wrapped, masks_wrapped, cv::Size(w, h), 0, 0,
      //            cv::INTER_LINEAR);

      if (masks.sizes().size() == 2) {
        masks = masks.expand(0);
      } else {
        masks = masks.permute({2, 0, 1});
      }
    }

    if (boxes.numel() > 0) {
      boxes.index({torch::All, torch::tensor({0, 2})}) *= (w / (float)img_w);
      boxes.index({torch::All, torch::tensor({1, 3})}) *= (h / (float)img_h);
    }
  }

  if (boxes.numel() > 0) {
    torch::Tensor box_w =
        boxes.index({torch::All, 2}) - boxes.index({torch::All, 0});
    torch::Tensor box_h =
        boxes.index({torch::All, 3}) - boxes.index({torch::All, 1});

    torch::Tensor keep =
        (box_w > (discard_box_width_)) * (box_h > (discard_box_height_));
    boxes = boxes.index({keep});
    masks = masks.index({keep});
  }

  std::vector<torch::Tensor> out;
  out.push_back(image_rgb);
  if (masks.numel() > 0) {
    out.push_back(masks);
  }
  if (boxes.numel() > 0) {
    out.push_back(boxes);
  }
  for (unsigned int i = 3; i < input.size(); i++) {
    out.push_back(input[i]);
  }

  return out;
}

std::vector<torch::Tensor> reclib::dnn::Resize::untransform(
    const std::vector<torch::Tensor>& input, unsigned int img_w,
    unsigned int img_h) {
  std::vector<torch::Tensor> out = input;

  torch::Tensor masks;
  torch::Tensor boxes;
  torch::Tensor corrs;

  unsigned int h = 0;
  unsigned int w = 0;
  if (input.size() >= 4) {
    masks = input[3];
    h = masks.sizes()[1];
    w = masks.sizes()[2];
  }
  if (input.size() >= 3) {
    boxes = input[2];
  }
  if (input.size() >= 5) {
    corrs = input[4];
    h = corrs.sizes()[2];
    w = corrs.sizes()[3];
  }

  if (masks.numel() > 0) {
    // B x C x H x W
    masks =
        torch::nn::functional::interpolate(
            masks.unsqueeze(1), torch::nn::functional::InterpolateFuncOptions()
                                    .size(std::vector<int64_t>({img_h, img_w}))
                                    .mode(torch::kBilinear)
                                    .align_corners(false))
            .squeeze(1)
            .gt(0.5)
            .toType(torch::ScalarType::Float);
  }

  if (corrs.numel() > 0) {
    // B x C x H x W
    corrs = torch::nn::functional::interpolate(
        corrs, torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<int64_t>({img_h, img_w}))
                   .mode(torch::kBilinear)
                   .align_corners(false));
  }

  if (boxes.numel() > 0) {
    boxes = boxes.toType(torch::kFloat32);

    torch::Tensor scales =
        torch::tensor({(img_w / (float)w), (img_h / (float)h),
                       (img_w / (float)w), (img_h / (float)h)})
            .contiguous();
    scales = scales.to(boxes.device());

    boxes = boxes * scales.unsqueeze(0);
    boxes = torch::round(boxes);
    boxes = boxes.toType(torch::kLong);
  }

  if (masks.numel() > 0) {
    out[3] = masks;
  }
  if (boxes.numel() > 0) {
    out[2] = boxes;
  }
  if (corrs.numel() > 0) {
    out[4] = corrs;
  }
  return out;
}

#endif

// ---------------------------------------------------------
// BackboneTransform
// --------------------------------------------------------

reclib::dnn::BackboneTransform::BackboneTransform(
    const std::string& channel_order, bool normalize, bool subtract_means,
    bool to_float, const std::vector<float>& mean, const std::vector<float> std,
    const std::string& in_channel_order)
    : channel_order_(channel_order),
      in_channel_order_(in_channel_order),
      normalize_(normalize),
      subtract_means_(subtract_means),
      to_float_(to_float),
      mean_(mean),
      std_(std) {
  if (channel_order.compare(in_channel_order) == 0) {
    permutation_order_ = {0, 1, 2};
  } else if (channel_order.compare("RGB") == 0 &&
             in_channel_order.compare("BGR") == 0) {
    permutation_order_ = {2, 1, 0};
  } else if (in_channel_order.compare("RGB") == 0 &&
             channel_order.compare("BGR") == 0) {
    permutation_order_ = {2, 1, 0};
  } else {
    throw std::runtime_error("Unknown pair of channels");
  }
}

std::vector<torch::Tensor> reclib::dnn::BackboneTransform::transform(
    const std::vector<torch::Tensor>& input) {
  torch::Tensor image = input[0];
  // check if 3rd dimension corresponds to the number of channels
  _RECLIB_ASSERT_LT(image.sizes()[2], 5);
  torch::Tensor masks;
  torch::Tensor boxes;
  if (input.size() > 1) {
    masks = input[1];
    _RECLIB_ASSERT_EQ(masks.sizes()[0], image.sizes()[0]);
    _RECLIB_ASSERT_EQ(masks.sizes()[1], image.sizes()[1]);
  }
  if (input.size() > 2) {
    boxes = input[2];
  }

  if (normalize_) {
    image.index(
        {torch::All, torch::All, torch::indexing::Slice({torch::None, 3})}) =
        (image.index({torch::All, torch::All,
                      torch::indexing::Slice({torch::None, 3})}) -
         torch::tensor(mean_)) /
        torch::tensor(std_);
  } else if (subtract_means_) {
    image.index(
        {torch::All, torch::All, torch::indexing::Slice({torch::None, 3})}) =
        (image.index({torch::All, torch::All,
                      torch::indexing::Slice({torch::None, 3})}) -
         torch::tensor(mean_));
  } else if (to_float_) {
    image.index({torch::All, torch::All,
                 torch::indexing::Slice({torch::None, 3})}) /= 255;
  }

  image.index(
      {torch::All, torch::All, torch::indexing::Slice({torch::None, 3})}) =
      image.index({torch::All, torch::All, torch::tensor(permutation_order_)});

  std::vector<torch::Tensor> out;
  out.push_back(image);
  if (masks.numel() > 0) {
    out.push_back(masks);
  }
  if (boxes.numel() > 0) {
    out.push_back(boxes);
  }
  for (unsigned int i = 3; i < input.size(); i++) {
    out.push_back(input[i]);
  }

  return out;
}

std::vector<torch::Tensor> reclib::dnn::BackboneTransform::untransform(
    const std::vector<torch::Tensor>& input, unsigned int im_w,
    unsigned int im_h) {
  return input;
}

// ---------------------------------------------------------
// BaseTransform
// --------------------------------------------------------

reclib::dnn::BaseTransform::BaseTransform(unsigned int max_size,
                                          bool preserve_aspect_ratio,
                                          const std::string& in_channel_order,
                                          const std::vector<float>& mean,
                                          const std::vector<float>& std) {
  transforms_.push_back(std::make_shared<reclib::dnn::Resize>(
      preserve_aspect_ratio, max_size, 4 / max_size, 4 / max_size, false));
  transforms_.push_back(std::make_shared<reclib::dnn::Pad>(
      preserve_aspect_ratio, max_size, max_size, mean, false));
  transforms_.push_back(std::make_shared<reclib::dnn::BackboneTransform>(
      "RGB", true, false, false, mean, std, in_channel_order));
}

std::vector<torch::Tensor> reclib::dnn::BaseTransform::transform(
    const std::vector<torch::Tensor>& input) {
  std::vector<torch::Tensor> out = transforms_[0]->transform(input);
  for (unsigned int i = 1; i < transforms_.size(); i++) {
    out = transforms_[i]->transform(out);
  }
  return out;
}

std::vector<torch::Tensor> reclib::dnn::BaseTransform::untransform(
    const std::vector<torch::Tensor>& input, unsigned int im_w,
    unsigned int im_h) {
  std::vector<torch::Tensor> out =
      transforms_[transforms_.size() - 1]->untransform(input, im_w, im_h);
  for (int i = transforms_.size() - 2; i >= 0; i--) {
    out = transforms_[i]->untransform(out, im_w, im_h);
  }
  return out;
}

// ---------------------------------------------------------
// YOLACT
// --------------------------------------------------------

void load_(torch::jit::Module& out, const fs::path& full_path,
           const std::vector<torch_tensorrt::core::ir::Input>& trt_input_specs =
               std::vector<torch_tensorrt::core::ir::Input>(),
           bool use_cuda = true, bool use_trt = true, bool use_fp16 = true,
           bool disable_cache = false) {
  try {
    // Deserialize the ScriptModule from a file using
    // torch::torch::jit::load()

    torch::jit::script::Module mod;
    if (use_trt) {
      fs::path tmp = full_path;
      tmp = tmp.replace_extension("trt");
      if (fs::exists(tmp) && !disable_cache) {
        std::cout << "loading cached tensorrt model: " << tmp << "...."
                  << std::endl;
        mod = torch::jit::load(tmp.string());

      } else {
        std::cout << "Converting torchscript model to tensorrt: " << tmp
                  << "...." << std::endl;

        reclib::dnn::silence_torchtrt_logger();
        torch::jit::script::Module ts_module = torch::jit::load(full_path);
        torch_tensorrt::core::CompileSpec spec(trt_input_specs);

        if (use_fp16) {
          spec.convert_info.engine_settings.enabled_precisions.insert(
              nvinfer1::DataType::kHALF);
        }
        mod = torch_tensorrt::core::CompileGraph(ts_module, spec);
        mod.save(tmp);
      }
    } else {
      std::cout << "loading cached torchscript model: " << full_path << "...."
                << std::endl;
      mod = torch::jit::load(full_path);
    }

    out = mod;
    out.eval();
    if (use_cuda) {
      out.to(torch::kCUDA);
    }

  } catch (const c10::Error& e) {
    std::cerr << "error loading the model at " << full_path << std::endl;
  }
}

void reclib::dnn::Yolact::load(const fs::path& base_path, bool use_cuda,
                               bool use_trt, bool use_fp16,
                               bool disable_cache) {
  // backbone
  load_(backbone_, base_path / BACKBONE_PATH, {backbone_input_}, use_cuda,
        use_trt, use_fp16, disable_cache);

  // fpn
  std::vector<int64_t> fpn_input_0 = fpn_input_;
  std::vector<int64_t> fpn_input_1 = fpn_input_0;
  fpn_input_1[1] *= 2;
  fpn_input_1[2] = round(fpn_input_1[2] / 2.f);
  fpn_input_1[3] = round(fpn_input_1[3] / 2.f);
  std::vector<int64_t> fpn_input_2 = fpn_input_1;
  fpn_input_2[1] *= 2;
  fpn_input_2[2] = round(fpn_input_2[2] / 2.f);
  fpn_input_2[3] = round(fpn_input_2[3] / 2.f);

  load_(fpn_, base_path / FPN_PATH, {fpn_input_0, fpn_input_1, fpn_input_2},
        use_cuda, use_trt, use_fp16, disable_cache);

  // proto net
  load_(proto_net_, base_path / PROTO_NET_PATH, {proto_net_input_}, use_cuda,
        use_trt, use_fp16, disable_cache);
  // proto net_corr
  load_(proto_net_corr_, base_path / PROTO_NET_CORR_PATH,
        {proto_net_corr_input_}, use_cuda, use_trt, use_fp16, disable_cache);

  std::vector<int64_t> pred_input_0 = prediction_module_input_;
  std::vector<int64_t> pred_input_1 = pred_input_0;
  pred_input_1[2] = round(pred_input_1[2] / 2.f);
  pred_input_1[3] = round(pred_input_1[3] / 2.f);
  std::vector<int64_t> pred_input_2 = pred_input_1;
  pred_input_2[2] = round(pred_input_2[2] / 2.f);
  pred_input_2[3] = round(pred_input_2[3] / 2.f);
  std::vector<int64_t> pred_input_3 = pred_input_2;
  pred_input_3[2] = round(pred_input_3[2] / 2.f);
  pred_input_3[3] = round(pred_input_3[3] / 2.f);
  std::vector<int64_t> pred_input_4 = pred_input_3;
  pred_input_4[2] = round(pred_input_4[2] / 2.f);
  pred_input_4[3] = round(pred_input_4[3] / 2.f);

  // prediction modules
  torch::jit::Module pred;
  load_(pred, base_path / PRED_MOD0_PATH, {pred_input_0}, use_cuda, use_trt,
        use_fp16, disable_cache);
  prediction_modules_.push_back(pred);

  load_(pred, base_path / PRED_MOD1_PATH, {pred_input_1}, use_cuda, use_trt,
        use_fp16, disable_cache);
  prediction_modules_.push_back(pred);

  load_(pred, base_path / PRED_MOD2_PATH, {pred_input_2}, use_cuda, use_trt,
        use_fp16, disable_cache);
  prediction_modules_.push_back(pred);

  load_(pred, base_path / PRED_MOD3_PATH, {pred_input_3}, use_cuda, use_trt,
        use_fp16, disable_cache);
  prediction_modules_.push_back(pred);

  load_(pred, base_path / PRED_MOD4_PATH, {pred_input_4}, use_cuda, use_trt,
        use_fp16, disable_cache);
  prediction_modules_.push_back(pred);
}

reclib::dnn::Yolact::Yolact(const fs::path& base_path, bool use_cuda,
                            bool use_trt, bool use_fp16, bool disable_cache)
    : use_cuda_(use_cuda) {
  load(base_path, use_cuda, use_trt, use_fp16, disable_cache);

  uint32_t pred_input_size = prediction_module_input_[3];
  for (unsigned idx = 0; idx < prediction_modules_.size(); idx++) {
    torch::Device dev(torch::kCPU);
    if (use_cuda) {
      dev = torch::Device(torch::kCUDA);
    }
    torch::Tensor prior = make_priors(scales_[idx], aspect_ratios_[idx],
                                      pred_input_size, pred_input_size, dev);

    if (use_cuda) {
      prior = prior.to(torch::kCUDA);
    }
    prediction_priors_.push_back(prior);
    pred_input_size = ceil((float)pred_input_size / 2.f);
  }
}

torch::Tensor reclib::dnn::Yolact::make_priors(
    float scale, const std::vector<float>& aspect_ratios, unsigned int conv_h,
    unsigned int conv_w, const c10::Device& device, float max_size) {
  std::vector<torch::Tensor> tensor_data;

#if defined(_OPENMP)
#pragma omp for
#endif
  for (unsigned int j = 0; j < conv_h; j++) {
#if defined(_OPENMP)
#pragma omp for
#endif
    for (unsigned int i = 0; i < conv_w; i++) {
      float x = ((i + 0.5f) / conv_w);
      float y = (j + 0.5f) / conv_h;

      for (auto ar : aspect_ratios) {
        // !backbone.preapply_sqrt
        float ar_sqrt = sqrt(ar);
        // cfg.backbone.use_pixel_scales, no tuple
        float w = (scale * ar_sqrt / max_size);
        float h = (scale / ar_sqrt / max_size);
        // cfg.backbone.use_square_anchors
        h = w;

        auto tmp = torch::tensor({x, y, w, h});
        tensor_data.push_back(tmp);
      }
    }
  }
  torch::Tensor out =
      torch::stack(tensor_data).to(device).view({-1, 4}).detach();
  out.set_requires_grad(false);
  return out;
}

/**
We resize both tensors to [A,B,2] without new malloc:
[A,2] -> [A,1,2] -> [A,B,2]
[B,2] -> [1,B,2] -> [A,B,2]
Then we compute the area of intersect between box_a and box_b.
Args:
  box_a: (tensor) bounding boxes, Shape: [n,A,4].
  box_b: (tensor) bounding boxes, Shape: [n,B,4].
Return:
  (tensor) intersection area, Shape: [n,A,B].
*/
torch::Tensor reclib::dnn::Yolact::intersect(const torch::Tensor& box_a,
                                             const torch::Tensor& box_b) {
  unsigned int n = box_a.sizes()[0];
  unsigned int A = box_a.sizes()[1];
  unsigned int B = box_b.sizes()[1];

  torch::Tensor A_max = box_a
                            .index({torch::All, torch::All,
                                    torch::indexing::Slice(2, torch::None)})
                            .unsqueeze(2)
                            .expand({n, A, B, 2});
  torch::Tensor B_max = box_b
                            .index({torch::All, torch::All,
                                    torch::indexing::Slice(2, torch::None)})
                            .unsqueeze(1)
                            .expand({n, A, B, 2});
  torch::Tensor A_min = box_a
                            .index({torch::All, torch::All,
                                    torch::indexing::Slice(torch::None, 2)})
                            .unsqueeze(2)
                            .expand({n, A, B, 2});
  torch::Tensor B_min = box_b
                            .index({torch::All, torch::All,
                                    torch::indexing::Slice(torch::None, 2)})
                            .unsqueeze(1)
                            .expand({n, A, B, 2});

  torch::Tensor max_xy = torch::min(A_max, B_max);
  torch::Tensor min_xy = torch::max(A_min, B_min);

  return torch::clamp(max_xy - min_xy, 0).prod(3);
}

/** Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
is simply the intersection over union of two boxes.  Here we operate on
ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
E.g.:
    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
Args:
    box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
    box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
Return:
    jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
"*/
torch::Tensor reclib::dnn::Yolact::jaccard(torch::Tensor box_a,
                                           torch::Tensor box_b, bool iscrowd) {
  // _RECLIB_ASSERT_EQ(box_a.dim(), box_b.dim());
  bool use_batch = true;

  if (box_a.dim() == 2) {
    use_batch = false;
    box_a = box_a.unsqueeze(0);
    box_b = box_b.unsqueeze(0);
  }

  torch::Tensor inter = intersect(box_a, box_b);
  torch::Tensor area_a = ((box_a.index({torch::All, torch::All, 2}) -
                           box_a.index({torch::All, torch::All, 0})) *
                          (box_a.index({torch::All, torch::All, 3}) -
                           box_a.index({torch::All, torch::All, 1})))
                             .unsqueeze(2)
                             .expand_as(inter);
  torch::Tensor area_b = ((box_b.index({torch::All, torch::All, 2}) -
                           box_b.index({torch::All, torch::All, 0})) *
                          (box_b.index({torch::All, torch::All, 3}) -
                           box_b.index({torch::All, torch::All, 1})))
                             .unsqueeze(1)
                             .expand_as(inter);

  torch::Tensor union_ = area_a + area_b - inter;
  torch::Tensor out;
  if (iscrowd) {
    out = inter / area_a;
  } else {
    out = inter / union_;
  }
  if (use_batch) {
    return out;
  } else {
    return out.squeeze(0);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
reclib::dnn::Yolact::fast_nms(torch::Tensor boxes, torch::Tensor masks,
                              torch::Tensor scores, float conf_thresh,
                              int max_num_detections, float iou_threshold,
                              int top_k, bool second_threshold) {
  std::tuple<torch::Tensor, torch::Tensor> ret = scores.sort(1, true);
  scores = std::get<0>(ret);
  torch::Tensor idx = std::get<1>(ret);

  idx = idx.index({torch::All, torch::indexing::Slice(torch::None, top_k)})
            .contiguous();
  scores =
      scores.index({torch::All, torch::indexing::Slice(torch::None, top_k)});

  int num_classes = idx.sizes()[0];
  int num_dets = idx.sizes()[1];

  boxes =
      boxes.index({idx.view(-1), torch::All}).view({num_classes, num_dets, 4});
  masks =
      masks.index({idx.view(-1), torch::All}).view({num_classes, num_dets, -1});

  torch::Tensor iou = jaccard(boxes, boxes);
  iou.triu_(1);
  ret = iou.max(1);
  torch::Tensor iou_max = std::get<0>(ret);

  torch::Tensor keep = (iou_max <= iou_threshold);

  if (second_threshold) {
    keep *= (scores > conf_thresh);
  }

  torch::Tensor classes = torch::arange(num_classes)
                              .to(boxes.device())
                              .index({torch::All, torch::None})
                              .expand_as(keep);
  classes = classes.index({keep});

  boxes = boxes.index({keep});
  masks = masks.index({keep});
  scores = scores.index({keep});

  ret = scores.sort(0, true);
  scores = std::get<0>(ret);
  idx = std::get<1>(ret);
  idx = idx.index({torch::indexing::Slice(torch::None, max_num_detections)});
  scores =
      scores.index({torch::indexing::Slice(torch::None, max_num_detections)});

  classes = classes.index({idx});
  boxes = boxes.index({idx});
  masks = masks.index({idx});

  return std::make_tuple(boxes, masks, classes, scores);
}

torch::Tensor reclib::dnn::Yolact::decode(const torch::Tensor& loc,
                                          const torch::Tensor& priors,
                                          bool use_yolo_regressors) {
  torch::Tensor boxes;
  if (use_yolo_regressors) {
    torch::Tensor t1 =
        loc.index({torch::All, torch::indexing::Slice(torch::None, 2)}) +
        priors.index({torch::All, torch::indexing::Slice(torch::None, 2)});
    torch::Tensor t2 =
        priors.index({torch::All, torch::indexing::Slice(2, torch::None)}) +
        torch::exp(
            loc.index({torch::All, torch::indexing::Slice(2, torch::None)}));

    boxes = torch::cat({t1, t2}, 1);
  } else {
    torch::Tensor t1 =
        priors.index({torch::All, torch::indexing::Slice(torch::None, 2)}) +
        loc.index({torch::All, torch::indexing::Slice(torch::None, 2)}) * 0.1 *
            priors.index({torch::All, torch::indexing::Slice(2, torch::None)});
    torch::Tensor t2 =
        priors.index({torch::All, torch::indexing::Slice(2, torch::None)}) *
        torch::exp(
            loc.index({torch::All, torch::indexing::Slice(2, torch::None)}) *
            0.2);

    boxes = torch::cat({t1, t2}, 1);

    boxes.index({torch::All, torch::indexing::Slice(torch::None, 2)}) -=
        boxes.index({torch::All, torch::indexing::Slice(2, torch::None)}) / 2.f;
    boxes.index({torch::All, torch::indexing::Slice(2, torch::None)}) +=
        boxes.index({torch::All, torch::indexing::Slice(torch::None, 2)});
  }
  return boxes;
}

std::vector<torch::Tensor> reclib::dnn::Yolact::detect(
    int batch_idx, const torch::Tensor& conf_preds,
    const torch::Tensor& decoded_boxes, const torch::Tensor& mask_data,
    float conf_thresh, float nms_thresh, int max_num_detections, int top_k) {
  torch::Tensor cur_scores =
      conf_preds.index({batch_idx, torch::indexing::Slice(1, torch::None)});

  std::tuple<torch::Tensor, torch::Tensor> ret = torch::max(cur_scores, 0);
  torch::Tensor conf_scores = std::get<0>(ret);
  torch::Tensor keep = conf_scores > conf_thresh;

  torch::Tensor scores = cur_scores.index({torch::All, keep});
  torch::Tensor boxes = decoded_boxes.index({keep});
  torch::Tensor masks = mask_data.index({batch_idx, keep});

  if (scores.sizes()[1] == 0) {
    return {};
  }

  auto ret_tuple = fast_nms(boxes, masks, scores, conf_thresh,
                            max_num_detections, nms_thresh, top_k);

  return {std::get<0>(ret_tuple), std::get<1>(ret_tuple),
          std::get<2>(ret_tuple), std::get<3>(ret_tuple)};
}

std::vector<std::vector<torch::Tensor>> reclib::dnn::Yolact::detect(
    const std::vector<torch::Tensor>& pred_outs) {
  torch::Tensor loc_data = pred_outs[0];
  torch::Tensor conf_data = pred_outs[1];
  torch::Tensor mask_data = pred_outs[2];
  torch::Tensor prior_data = pred_outs[3];

  torch::Tensor proto_data = pred_outs[4];
  torch::Tensor proto_corr_data = pred_outs[5];

  int batch_size = loc_data.sizes()[0];
  int num_priors = prior_data.sizes()[0];

  torch::Tensor conf_preds =
      conf_data.view({batch_size, num_priors, NUM_CLASSES})
          .transpose(2, 1)
          .contiguous();

  std::vector<std::vector<torch::Tensor>> out;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    torch::Tensor decoded_boxes =
        decode(loc_data.index({batch_idx}), prior_data);

    std::vector<torch::Tensor> ret_vec =
        detect(batch_idx, conf_preds, decoded_boxes, mask_data, CONF_THRESH,
               NMS_THRESH, MAX_NUM_DETECTIONS, TOP_K);

    if (ret_vec.size() > 0) {
      ret_vec.push_back(proto_data.index({batch_idx}));
      ret_vec.push_back(proto_corr_data.index({batch_idx}));
    }

    out.push_back(ret_vec);
  }
  return out;
}

void reclib::dnn::Yolact::print_output(const c10::IValue& val,
                                       std::string output_string,
                                       int recursion_depth) {
  if (val.isTuple()) {
    c10::ivalue::Tuple* val_cast = val.toTuple().get();
    for (unsigned int i = 0; i < val_cast->elements().size(); i++) {
      print_output(val_cast->elements()[i],
                   output_string + ", i: " + std::to_string(i),
                   recursion_depth + 1);
    }

  } else if (val.isTensor()) {
    torch::Tensor val_cast = val.toTensor();
    std::vector<at::indexing::TensorIndex> indices;
    for (unsigned int i = 0; i < val_cast.sizes().vec().size() - 1; i++) {
      if (i == val_cast.sizes().vec().size() - 2 &&
          val_cast.sizes()[val_cast.sizes().vec().size() - 1] == 4) {
        indices.push_back(torch::indexing::Slice({torch::None, 10}));
      } else {
        indices.push_back(0);
      }
    }
    indices.push_back(torch::indexing::Slice(0, 10));
    std::cout << output_string << " shape: " << val_cast.sizes()
              << ", vals: " << val_cast.index(indices) << std::endl;

  } else if (val.isList()) {
    c10::impl::GenericList val_cast = val.toList();
    for (unsigned int i = 0; i < val_cast.size(); i++) {
      print_output(val_cast[i], output_string + ", i: " + std::to_string(i),
                   recursion_depth + 1);
    }

  } else if (val.isGenericDict()) {
    c10::impl::GenericDict val_cast = val.toGenericDict();
    for (auto iter = val_cast.begin(); iter != val_cast.end(); iter++) {
      print_output(
          iter->value(),
          output_string + ", k: " + iter->key().toString().get()->string(),
          recursion_depth + 1);
    }
  }

  if (recursion_depth == 0) {
    std::cout << "--------------------------------------------" << std::endl;
  }
}

std::vector<torch::Tensor> reclib::dnn::Yolact::forward(
    const torch::Tensor& input, bool benchmark) {
  timer_.look_and_reset();
  c10::IValue out = backbone_.forward({input.unsqueeze(0)});

  if (benchmark) {
    std::cout << "---- backbone: " << timer_.look_and_reset() << std::endl;
  }

  // select layers
  std::vector<c10::IValue> selected;
  c10::ivalue::Tuple* out_tuple = out.toTuple().get();
  for (unsigned int i = 1; i < out_tuple->elements().size(); i++) {
    selected.push_back(out_tuple->elements()[i]);
  }

  if (benchmark) {
    timer_.look_and_reset();
  }

  out = fpn_.forward(selected);

  if (benchmark) {
    std::cout << "---- fpn: " << timer_.look_and_reset() << std::endl;
  }

  out_tuple = out.toTuple().get();

  if (benchmark) {
    timer_.look_and_reset();
  }

  c10::IValue proto_out = proto_net_({out_tuple->elements()[0].toTensor()});

  if (benchmark) {
    std::cout << "---- proto_out: " << timer_.look_and_reset() << std::endl;
  }

  torch::Tensor proto_out_tensor = torch::relu(proto_out.toTensor());
  // torch::Tensor proto_out_tensor = torch::sigmoid(proto_out.toTensor());
  proto_out_tensor = proto_out_tensor.permute({0, 2, 3, 1}).contiguous();

  if (benchmark) {
    std::cout << "(relu + permute): " << timer_.look_and_reset() << std::endl;
  }

  // c10::IValue proto_corr_out = proto_net_corr_({proto_out.toTensor()});
  c10::IValue proto_corr_out =
      proto_net_corr_({out_tuple->elements()[0].toTensor()});

  if (benchmark) {
    std::cout << "---- proto_corr_out: " << timer_.look_and_reset()
              << std::endl;
  }

  torch::Tensor proto_corr_out_tensor =
      torch::sigmoid(proto_corr_out.toTensor());
  proto_corr_out_tensor = proto_corr_out_tensor.reshape(
      {proto_corr_out_tensor.sizes()[0], -1, 3,
       proto_corr_out_tensor.sizes()[2], proto_corr_out_tensor.sizes()[3]});
  proto_corr_out_tensor = proto_corr_out_tensor.permute({0, 2, 3, 4, 1});

  if (benchmark) {
    timer_.look_and_reset();
  }

  std::vector<torch::Tensor> pred_outs;
  for (unsigned idx = 0; idx < prediction_modules_.size(); idx++) {
    torch::Tensor pred_x = out_tuple->elements()[idx].toTensor();

    if (benchmark) {
      timer_.look_and_reset();
    }

    c10::IValue p = prediction_modules_[idx]({pred_x});

    if (benchmark) {
      std::cout << "---- prediction " << idx << ": " << timer_.look_and_reset()
                << std::endl;
    }

    c10::ivalue::Tuple* p_tuple = p.toTuple().get();
    unsigned int tuple_limit = 3;
    torch::Tensor priors = prediction_priors_[idx];

    if (pred_outs.size() < tuple_limit) {
      for (unsigned int i = 0; i < tuple_limit; i++) {
        pred_outs.push_back(p_tuple->elements()[i].toTensor());
      }
      pred_outs.push_back(priors);
    } else {
      for (unsigned int i = 0; i < tuple_limit; i++) {
        torch::Tensor tmp = p_tuple->elements()[i].toTensor();
        pred_outs[i] = torch::cat({pred_outs[i], tmp}, -2);
      }
      pred_outs[tuple_limit] = torch::cat({pred_outs[tuple_limit], priors}, -2);
    }

    if (benchmark) {
      timer_.look_and_reset();
    }

    if (benchmark) {
      std::cout << "(Insert into Dict (in loop)): " << timer_.look_and_reset()
                << std::endl;
    }
  }

  pred_outs.push_back(proto_out_tensor);
  pred_outs.push_back(proto_corr_out_tensor);

  torch::Tensor conf = pred_outs[1];
  conf = torch::softmax(conf, -1);
  pred_outs[1] = conf;

  if (benchmark) {
    std::cout << "(insert to Dict (End))" << timer_.look_and_reset()
              << std::endl;
  }

  return pred_outs;

  // std::vector<std::vector<torch::Tensor>> final_out =
  //     Detect(pred_outs, NUM_CLASSES);

  // if (benchmark) {
  //   std::cout << "---- detect: " << timer_.look_and_reset() << std::endl;
  // }

  // return final_out;
}

/**
Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <=
image_size. Also converts from relative to absolute coordinates and casts the
results to long tensors.

If cast is false, the result won't be cast to longs.
Warning: this does things in-place behind the scenes so copy if necessary.
*/
std::pair<torch::Tensor, torch::Tensor> sanitize_coordinates(
    const torch::Tensor& _x1, const torch::Tensor& _x2, int img_size,
    int padding = 0, bool cast = true) {
  torch::Tensor x1 = _x1 * img_size;
  torch::Tensor x2 = _x2 * img_size;

  if (cast) {
    x1 = x1.toType(torch::ScalarType::Long);
    x2 = x2.toType(torch::ScalarType::Long);
  }

  x1 = torch::min(x1, x2);
  x2 = torch::max(x1, x2);

  x1 = torch::clamp(x1 - padding, 0, img_size);
  x2 = torch::clamp(x2 + padding, 0, img_size);

  return std::make_pair(x1, x2);
}

std::pair<torch::Tensor, torch::Tensor> crop(const torch::Tensor& masks,
                                             const torch::Tensor& corrs,
                                             const torch::Tensor& boxes,
                                             int padding = 1) {
  int h = masks.sizes()[0];
  int w = masks.sizes()[1];
  int n = masks.sizes()[2];

  std::pair<torch::Tensor, torch::Tensor> ret_x =
      sanitize_coordinates(boxes.index({torch::All, 0}),
                           boxes.index({torch::All, 2}), w, padding, false);
  std::pair<torch::Tensor, torch::Tensor> ret_y =
      sanitize_coordinates(boxes.index({torch::All, 1}),
                           boxes.index({torch::All, 3}), h, padding, false);

  torch::Tensor rows = torch::arange(w)
                           .to(masks.device())
                           .toType(ret_x.first.scalar_type())
                           .view({1, -1, 1})
                           .expand({h, w, n});
  torch::Tensor cols = torch::arange(h)
                           .to(masks.device())
                           .toType(ret_x.first.scalar_type())
                           .view({-1, 1, 1})
                           .expand({h, w, n});

  torch::Tensor masks_left = rows >= ret_x.first.view({1, 1, -1});
  torch::Tensor masks_right = rows < ret_x.second.view({1, 1, -1});
  torch::Tensor masks_up = cols >= ret_y.first.view({1, 1, -1});
  torch::Tensor masks_down = cols < ret_y.second.view({1, 1, -1});

  torch::Tensor crop_mask = masks_left * masks_right * masks_up * masks_down;

  torch::Tensor cropped_masks =
      masks * crop_mask.toType(torch::ScalarType::Float);
  torch::Tensor cropped_corrs =
      corrs * crop_mask.toType(torch::ScalarType::Float)
                  .index({torch::None, torch::All, torch::All});
  return {cropped_masks, cropped_corrs};
}

std::vector<torch::Tensor> reclib::dnn::Yolact::postprocess(
    const std::vector<std::vector<torch::Tensor>>& net_output, unsigned int w,
    unsigned int h, float score_threshold, bool do_crop, bool benchmark) {
  if (net_output.size() == 0 || net_output[0].size() == 0) {
    return {};
  }

  timer_.look_and_reset();

  torch::Tensor boxes = net_output[0][0];
  torch::Tensor mask_coeffs = net_output[0][1];
  torch::Tensor classes = net_output[0][2];
  torch::Tensor scores = net_output[0][3];
  torch::Tensor proto_masks = net_output[0][4];
  torch::Tensor proto_corrs = net_output[0][5];

  if (benchmark) {
    std::cout << "---- parsing predictions: " << timer_.look_and_reset()
              << std::endl;
  }

  torch::Tensor keep = scores > score_threshold;
  boxes = boxes.index({keep});
  mask_coeffs = mask_coeffs.index({keep});
  classes = classes.index({keep});
  scores = scores.index({keep});

  if (benchmark) {
    std::cout << "---- Thresholding: " << timer_.look_and_reset() << std::endl;
  }

  std::vector<torch::Tensor> out;
  if (scores.sizes()[0] == 0) {
    return out;
  }

  // torch::Tensor corrs = torch::matmul(proto_corrs, mask_coeffs.transpose(1,
  // 0)); corrs = torch::sigmoid(corrs);
  torch::Tensor corrs = proto_corrs;

  if (benchmark) {
    std::cout << "---- Processing corr prototpes: " << timer_.look_and_reset()
              << std::endl;
  }

  // std::cout << "masks: " << proto_masks.sizes() << std::endl;
  // for (unsigned int i = 0; i < proto_masks.sizes()[2]; i++) {
  //   torch::Tensor m =
  //       proto_masks.index({torch::All, torch::All,
  //       (int)i}).cpu().contiguous();
  //   CpuMat tmp = reclib::dnn::torch2cv(m, true);
  //   cv::imshow("m", tmp);
  //   cv::waitKey(0);
  //   std::cout << "m: " << m.sizes() << std::endl;
  // }
  // cv::destroyAllWindows();
  torch::Tensor masks = torch::matmul(proto_masks, mask_coeffs.transpose(1, 0));

  masks = torch::sigmoid(masks);

  if (benchmark) {
    std::cout << "---- Processing mask prototypes: " << timer_.look_and_reset()
              << std::endl;
  }

  if (do_crop) {
    std::pair<torch::Tensor, torch::Tensor> cropped = crop(masks, corrs, boxes);
    masks = cropped.first;
    corrs = cropped.second;
  }
  if (benchmark) {
    std::cout << "---- Crop: " << timer_.look_and_reset() << std::endl;
  }

  masks = masks.permute({2, 0, 1}).contiguous();
  // cv::Mat mask_wrapped = reclib::dnn::torch2cv(masks);
  // cv::resize(mask_wrapped, mask_wrapped, cv::Size(w, h))
  // masks = reclib::dnn::cv2torch(mask_wrapped);
  masks =
      torch::nn::functional::interpolate(
          masks.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions()
                                  .size(std::vector<int64_t>({h, w}))
                                  .mode(torch::kBilinear)
                                  .align_corners(false))
          .squeeze(0);

  masks = masks.gt(0.5).toType(torch::ScalarType::Float);

  corrs = corrs.permute({3, 0, 1, 2});
  corrs = torch::nn::functional::interpolate(
      corrs, torch::nn::functional::InterpolateFuncOptions()
                 .size(std::vector<int64_t>({h, w}))
                 .mode(torch::kBilinear)
                 .align_corners(false));
  // corrs =
  //     corrs * masks.index({torch::All, torch::None, torch::All, torch::All});

  if (benchmark) {
    std::cout << "---- Processing masks and corrs: " << timer_.look_and_reset()
              << std::endl;
  }

  std::pair<torch::Tensor, torch::Tensor> sanitized_x = sanitize_coordinates(
      boxes.index({torch::All, 0}), boxes.index({torch::All, 2}), w, false);
  boxes.index({torch::All, 0}) = sanitized_x.first;
  boxes.index({torch::All, 2}) = sanitized_x.second;
  std::pair<torch::Tensor, torch::Tensor> sanitized_y = sanitize_coordinates(
      boxes.index({torch::All, 1}), boxes.index({torch::All, 3}), h, false);
  boxes.index({torch::All, 1}) = sanitized_y.first;
  boxes.index({torch::All, 3}) = sanitized_y.second;

  if (benchmark) {
    std::cout << "---- Sanitizing boxes: " << timer_.look_and_reset()
              << std::endl;
  }

  boxes = boxes.toType(torch::ScalarType::Long);

  out.push_back(classes);
  out.push_back(scores);
  out.push_back(boxes);
  out.push_back(masks);
  out.push_back(corrs);

  return out;
}

#endif  // HAS_DNN_MODULE