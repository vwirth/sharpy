#ifndef RECLIB_DNN_DNN_UTILS_H
#define RECLIB_DNN_DNN_UTILS_H

#include <Eigen/Eigen>

#include "reclib/assert.h"

#if HAS_DNN_MODULE
#include <torch/torch.h>

namespace torch {
extern torch::indexing::Slice All;
extern c10::nullopt_t None;

}  // namespace torch

namespace reclib {
namespace dnn {

template <typename T = float>
class TorchVector {
 public:
  torch::Tensor storage_;

  TorchVector() {}
  TorchVector(torch::Tensor storage) : storage_(storage) {}

  T operator[](int index) { return storage_.index({index}).item<T>(); }
  T x() const { return storage_.index({0}).item<T>(); }
  T y() const { return storage_.index({1}).item<T>(); }
  T z() const { return storage_.index({2}).item<T>(); }
  T w() const { return storage_.index({3}).item<T>(); }
  float norm() const { return storage_.norm().item<float>(); }
  int dims() const { return storage_.sizes().size(); }
  T* data_ptr() { return storage_.data_ptr<T>(); }
  torch::Tensor tensor() { return storage_; };
};

template <typename T>
inline torch::Tensor eigen2torch(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& m,
    bool copy = true) {
  if (copy) {
    return torch::from_blob(m.data(), {m.rows(), m.cols()})
        .transpose(0, 1)
        .clone();
  } else {
    return torch::from_blob(m.data(), {m.rows(), m.cols()}).transpose(0, 1);
  }
}

template <typename T, int R, int C>
inline torch::Tensor eigen2torch(Eigen::Matrix<T, R, C, Eigen::ColMajor>& m,
                                 bool copy = true) {
  if (copy) {
    return torch::from_blob(m.data(), {m.rows(), m.cols()})
        .transpose(0, 1)
        .clone();
  } else {
    return torch::from_blob(m.data(), {m.rows(), m.cols()}).transpose(0, 1);
  }
}

template <typename T>
inline torch::Tensor eigen2torch(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& m,
    bool copy = true) {
  if (copy) {
    return torch::from_blob(m.data(), {m.cols(), m.rows()}).clone();
  } else {
    return torch::from_blob(m.data(), {m.cols(), m.rows()});
  }
}

template <typename T, int R, int C>
inline torch::Tensor eigen2torch(Eigen::Matrix<T, R, C, Eigen::RowMajor>& m,
                                 bool copy = true) {
  if (copy) {
    return torch::from_blob(m.data(), {m.cols(), m.rows()}).clone();
  } else {
    return torch::from_blob(m.data(), {m.cols(), m.rows()});
  }
}

// template <typename T>
// inline Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> torch2eigen(
//     torch::Tensor& t) {
//   _RECLIB_ASSERT(t.is_cpu());
//   _RECLIB_ASSERT(t.is_contiguous() ||
//                  t.is_contiguous(torch::MemoryFormat::ChannelsLast));
//   _RECLIB_ASSERT_EQ(t.sizes().size(), 1);

//   return Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(t.data<T>(),
//                                                       t.sizes()[0]);
// }

// template <typename T>
// inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
// torch2eigen(
//     torch::Tensor& t) {
//   _RECLIB_ASSERT(t.is_cpu());
//   _RECLIB_ASSERT(t.is_contiguous() ||
//                  t.is_contiguous(torch::MemoryFormat::ChannelsLast));
//   _RECLIB_ASSERT_LE(t.sizes().size(), 2);
//   _RECLIB_ASSERT_EQ(t.element_size(), sizeof(T));

//   if (t.sizes().size() == 1) {
//     return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
//         t.data<T>(), t.sizes()[0], 1);
//   } else {
//     return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
//         t.data<T>(), t.sizes()[0], t.sizes()[1]);
//   }
// }

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> torch2eigen(
    torch::Tensor t) {
  _RECLIB_ASSERT(t.is_cpu());
  _RECLIB_ASSERT(t.is_contiguous() ||
                 t.is_contiguous(torch::MemoryFormat::ChannelsLast));
  _RECLIB_ASSERT_LE(t.sizes().size(), 2);
  _RECLIB_ASSERT_EQ(t.element_size(), sizeof(T));

  if (t.sizes().size() == 1) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        t.data<T>(), t.sizes()[0], 1);
  } else {
    if (t.sizes()[0] == t.sizes()[1]) {
      torch::Tensor tmp = t.clone().transpose(1, 0).contiguous();
      return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
          tmp.data<T>(), tmp.sizes()[0], tmp.sizes()[1]);
    }
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        t.data<T>(), t.sizes()[0], t.sizes()[1]);
  }
}

#if HAS_OPENCV_MODULE
#include <opencv2/core.hpp>

inline torch::Tensor cv2torch(cv::Mat& mat, bool copy = true) {
  torch::Tensor out;
  _RECLIB_ASSERT(mat.isContinuous());
  switch (mat.depth()) {
    case CV_32F: {
      unsigned int num_channels = mat.elemSize() / sizeof(float);

      out = torch::from_blob(mat.data, {mat.rows, mat.cols, num_channels},
                             {torch::ScalarType::Float});
      break;
    }
    case CV_8U: {
      unsigned int num_channels = mat.elemSize() / sizeof(char);
      out = torch::from_blob(mat.data, {mat.rows, mat.cols, num_channels},
                             {torch::ScalarType::Byte});
      break;
    }
    case CV_8S: {
      unsigned int num_channels = mat.elemSize() / sizeof(char);
      out = torch::from_blob(mat.data, {mat.rows, mat.cols, num_channels},
                             {torch::ScalarType::Char});
      break;
    }
    default: {
      throw std::runtime_error("Unsupported mat type");
      out = torch::zeros({1});
    }
  }

  if (copy) {
    return out.clone();
  }
  return out;
}

inline cv::Mat torch2cv(torch::Tensor& b, bool copy = true) {
  cv::Mat out;
  torch::Tensor t;
  if (copy) {
    torch::NoGradGuard guard;
    t = b.clone();
    if (t.is_cuda()) {
      t = t.cpu();
    }
  }
  _RECLIB_ASSERT(t.is_cpu());
  _RECLIB_ASSERT(t.is_contiguous() ||
                 t.is_contiguous(torch::MemoryFormat::ChannelsLast));

  unsigned int channels = 1;
  if (t.dim() > 2) {
    channels = t.sizes()[2];

    if (t.sizes()[0] < t.sizes()[2] || t.sizes()[1] < t.sizes()[2]) {
      std::cout << "[torch2cv] Warning: Tensor size might be incorrect: "
                << t.sizes() << std::endl;
    }
    // _RECLIB_ASSERT_GT(t.sizes()[0], t.sizes()[2]);
    // _RECLIB_ASSERT_GT(t.sizes()[1], t.sizes()[2]);
  }
  switch (t.scalar_type()) {
    case torch::ScalarType::Char: {
      out = cv::Mat(t.sizes()[0], t.sizes()[1], CV_8SC(channels), t.data_ptr());
      break;
    }
    case torch::ScalarType::Byte: {
      out = cv::Mat(t.sizes()[0], t.sizes()[1], CV_8UC(channels), t.data_ptr());
      break;
    }
    case torch::ScalarType::Float: {
      out =
          cv::Mat(t.sizes()[0], t.sizes()[1], CV_32FC(channels), t.data_ptr());
      break;
    }
    case torch::ScalarType::Int: {
      out =
          cv::Mat(t.sizes()[0], t.sizes()[1], CV_32SC(channels), t.data_ptr());
      break;
    }
    default: {
      throw std::runtime_error("Unsupported tensor type");
      out = cv::Mat(1, 1, CV_32FC1);
    }
  }

  if (copy) {
    cv::Mat out_out;
    out.copyTo(out_out);
    return out_out;
  }
  return out;
}
#endif  // HAS_OPENCV_MODULE
}  // namespace dnn
}  // namespace reclib

#endif  // HAS_DNN_MODULE
#endif