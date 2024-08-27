#ifndef RECLIB_SHARPY_UTILS_H
#define RECLIB_SHARPY_UTILS_H

#include <ATen/core/TensorBody.h>
#include <ATen/ops/clamp_min.h>
#include <c10/core/TensorOptions.h>
#include <torch/types.h>

#include "reclib/data_types.h"
#include "reclib/dnn/dnn_utils.h"

namespace reclib {
namespace tracking {
namespace sharpy {

// converts hsv coordinate to classification label
inline int hsv2cls(vec3 hsv) {
  if (hsv.x() == 0 && hsv.y() == 0 && hsv.z() == 0) {
    return 30;
  }
  hsv.x() *= 360;

  int hsv_finger_seg =
      fmax(0, int(((hsv.y() - 0.58) / (0.139)) + 1));  // yields 0, 1, 2 or 3
  int hsv_finger = 0;
  if (hsv.x() < 180) {
    hsv_finger = int(fmax(0, int(hsv.x() - 20)) / (80.0));  // yields 0 or 1
  } else {
    hsv_finger =
        int(fmax(0, int(hsv.x() - 180)) / (60.0)) + 2;  // yields 2,3, or 4
  }
  return hsv_finger * 4 + hsv_finger_seg;
}

inline vec3 cls2color(int cls) {
  if (cls == 0 || cls == 4 || cls == 8 || cls == 12 || cls == 16) {
    return (vec3(0.4, 0, 0));
  }
  if (cls == 1) {
    return (vec3(0.6, 0, 0));
  }
  if (cls == 2) {
    return (vec3(0.8, 0, 0));
  }
  if (cls == 3) {
    return (vec3(1, 0, 0));
  }
  if (cls == 5) {
    return (vec3(1, 0, 0.6));
  }
  if (cls == 6) {
    return (vec3(1, 0, 0.8));
  }
  if (cls == 7) {
    return (vec3(1, 0, 1));
  }
  if (cls == 9) {
    return (vec3(0, 0.6, 0));
  }
  if (cls == 10) {
    return (vec3(0, 0.8, 0));
  }
  if (cls == 11) {
    return (vec3(0, 1, 0));
  }
  if (cls == 13) {
    return (vec3(0, 0, 0.6));
  }
  if (cls == 14) {
    return (vec3(0, 0, 0.8));
  }
  if (cls == 15) {
    return (vec3(0, 0, 1));
  }
  if (cls == 17) {
    return (vec3(1, 0.6, 0));
  }
  if (cls == 18) {
    return (vec3(1, 0.8, 0));
  }
  if (cls == 19) {
    return (vec3(1, 1, 0));
  }
  return vec3(0, 0, 0);
}

#if HAS_DNN_MODULE
inline torch::Tensor batch_cls2color(torch::Tensor cls) {
  torch::Device dev = cls.device();
  torch::TensorOptions opt = torch::TensorOptions().device(dev);
  torch::Tensor color = torch::zeros({cls.sizes()[0], cls.sizes()[1], 3}, opt);

  color.index_put_({cls == 0.f, torch::All},
                   torch::tensor({0.4f, 0.f, 0.f}, opt));
  color.index_put_({cls == 4, torch::All},
                   torch::tensor({0.4f, 0.f, 0.f}, opt));
  color.index_put_({cls == 8, torch::All},
                   torch::tensor({0.4f, 0.f, 0.f}, opt));
  color.index_put_({cls == 12, torch::All},
                   torch::tensor({0.4f, 0.f, 0.f}, opt));
  color.index_put_({cls == 16, torch::All},
                   torch::tensor({0.4f, 0.f, 0.f}, opt));

  color.index_put_({cls == 1.f, torch::All},
                   torch::tensor({0.6f, 0.f, 0.f}, opt));
  color.index_put_({cls == 2, torch::All},
                   torch::tensor({0.8f, 0.f, 0.f}, opt));
  color.index_put_({cls == 3, torch::All}, torch::tensor({1.f, 0.f, 0.f}, opt));

  color.index_put_({cls == 5, torch::All},
                   torch::tensor({1.f, 0.f, 0.6f}, opt));
  color.index_put_({cls == 6, torch::All},
                   torch::tensor({1.f, 0.f, 0.8f}, opt));
  color.index_put_({cls == 7, torch::All}, torch::tensor({1.f, 0.f, 1.f}, opt));

  color.index_put_({cls == 9, torch::All},
                   torch::tensor({0.f, 0.6f, 0.f}, opt));
  color.index_put_({cls == 10.f, torch::All},
                   torch::tensor({0.f, 0.8f, 0.f}, opt));
  color.index_put_({cls == 11.f, torch::All},
                   torch::tensor({0.f, 1.f, 0.f}, opt));

  color.index_put_({cls == 13, torch::All},
                   torch::tensor({0.f, 0.f, 0.6f}, opt));
  color.index_put_({cls == 14, torch::All},
                   torch::tensor({0.f, 0.f, 0.8f}, opt));
  color.index_put_({cls == 15, torch::All},
                   torch::tensor({0.f, 0.f, 1.f}, opt));

  color.index_put_({cls == 17, torch::All},
                   torch::tensor({1.f, 0.6f, 0.f}, opt));
  color.index_put_({cls == 18, torch::All},
                   torch::tensor({1.f, 0.8f, 0.f}, opt));
  color.index_put_({cls == 19, torch::All},
                   torch::tensor({1.f, 1.f, 0.f}, opt));

  return color;
}

inline torch::Tensor batch_corr2seg(torch::Tensor hsv) {
  _RECLIB_ASSERT_EQ(hsv.sizes()[1], 3);  // assume linearized hsv

  torch::Tensor summed = hsv.sum(1);
  torch::Tensor segmentation = torch::zeros(
      {hsv.sizes()[0]},
      torch::TensorOptions().device(hsv.device()).dtype(torch::kInt32));

  segmentation.index({summed == 0}) = 30;

  torch::Tensor sat_val = (hsv.index({torch::All, 1}) - 0.58) / (0.139) + 1;
  torch::Tensor hsv_finger_seg = torch::clamp_min(sat_val.toType(torch::kInt32),
                                                  0);  // yields 0, 1, 2 or 3

  torch::Tensor hue_val_thumbindex =
      torch::clamp_min(
          ((hsv.index({torch::All, 0}) * 360) - 20).toType(torch::kInt32), 0)
          .toType(torch::kFloat32) /
      80.0;
  hue_val_thumbindex =
      hue_val_thumbindex.index({(hsv.index({torch::All, 0}) * 360) < 180});

  torch::Tensor hue_val_midtopinky =
      torch::clamp_min(
          ((hsv.index({torch::All, 0}) * 360) - 180).toType(torch::kInt32), 0)
          .toType(torch::kFloat32) /
      60.0;

  torch::Tensor hsv_finger = hue_val_midtopinky.toType(torch::kInt32) + 2;

  hsv_finger.index_put_({(hsv.index({torch::All, 0}) * 360) < 180},
                        hue_val_thumbindex.toType(torch::kInt32));

  return hsv_finger * 4 + hsv_finger_seg;
}
#endif

}  // namespace sharpy
}  // namespace tracking
}  // namespace reclib

#endif