#ifndef RECLIB_UTILS_TORCH_UTILS_H
#define RECLIB_UTILS_TORCH_UTILS_H

#if HAS_DNN_MODULE

#include <torch/torch.h>

#include "reclib/dnn/dnn_utils.h"

namespace torch {
inline auto segment(torch::Tensor& t, int start_index, unsigned int length) {
  return t.index({torch::indexing::Slice({start_index, start_index + length})});
}
inline auto segment(const torch::Tensor& t, int start_index,
                    unsigned int length) {
  return t.index({torch::indexing::Slice({start_index, start_index + length})});
}
template <int L>
inline auto segment(const torch::Tensor& t, int start_index) {
  return t.index({torch::indexing::Slice({start_index, start_index + L})});
}
template <int L>
inline auto segment(torch::Tensor& t, int start_index) {
  return t.index({torch::indexing::Slice({start_index, start_index + L})});
}

inline auto head(torch::Tensor& t, unsigned int length) {
  return t.index({torch::indexing::Slice({0, length})});
}
inline auto head(const torch::Tensor& t, unsigned int length) {
  return t.index({torch::indexing::Slice({0, length})});
}
template <int L>
inline auto head(const torch::Tensor& t) {
  return t.index({torch::indexing::Slice({0, L})});
}
template <int L>
inline auto head(torch::Tensor& t) {
  return t.index({torch::indexing::Slice({0, L})});
}

inline auto tail(torch::Tensor& t, unsigned int length) {
  return t.index(
      {torch::indexing::Slice({t.sizes()[0] - length, t.sizes()[0]})});
}
inline auto tail(const torch::Tensor& t, unsigned int length) {
  return t.index(
      {torch::indexing::Slice({t.sizes()[0] - length, t.sizes()[0]})});
}
template <int L>
inline auto tail(const torch::Tensor& t) {
  return t.index({torch::indexing::Slice({t.sizes()[0] - L, t.sizes()[0]})});
}
template <int L>
inline auto tail(torch::Tensor& t) {
  return t.index({torch::indexing::Slice({t.sizes()[0] - L, t.sizes()[0]})});
}

inline auto topRows(torch::Tensor& t, unsigned int length) {
  return t.index({torch::indexing::Slice({0, length}), torch::All});
}
inline auto topRows(const torch::Tensor& t, unsigned int length) {
  return t.index({torch::indexing::Slice({0, length}), torch::All});
}
template <int L>
inline auto topRows(const torch::Tensor& t) {
  return t.index({torch::indexing::Slice({0, L}), torch::All});
}
template <int L>
inline auto topRows(torch::Tensor& t) {
  return t.index({torch::indexing::Slice({0, L}), torch::All});
}

inline auto leftCols(const torch::Tensor& t, unsigned int length) {
  return t.index({torch::All, torch::indexing::Slice({0, length})});
}
inline auto leftCols(torch::Tensor& t, unsigned int length) {
  return t.index({torch::All, torch::indexing::Slice({0, length})});
}
template <int L>
inline auto leftCols(torch::Tensor& t) {
  return t.index({torch::All, torch::indexing::Slice({0, L})});
}
template <int L>
inline auto leftCols(const torch::Tensor& t) {
  return t.index({torch::All, torch::indexing::Slice({0, L})});
}

inline auto rightCols(const torch::Tensor& t, unsigned int length) {
  return t.index({torch::All, torch::indexing::Slice(
                                  {t.sizes()[1] - length, t.sizes()[1]})});
}
inline auto rightCols(torch::Tensor& t, unsigned int length) {
  return t.index({torch::All, torch::indexing::Slice(
                                  {t.sizes()[1] - length, t.sizes()[1]})});
}
template <int L>
inline auto rightCols(torch::Tensor& t) {
  return t.index(
      {torch::All, torch::indexing::Slice({t.sizes()[1] - L, t.sizes()[1]})});
}
template <int L>
inline auto rightCols(const torch::Tensor& t) {
  return t.index(
      {torch::All, torch::indexing::Slice({t.sizes()[1] - L, t.sizes()[1]})});
}

inline torch::indexing::Slice make_slice(int start, int length) {
  return torch::indexing::Slice({start, start + length});
}

template <int R, int C>
inline auto block(const torch::Tensor& t) {
  return t.index(
      {torch::indexing::Slice({0, R}), torch::indexing::Slice({0, C})});
}

}  // namespace torch

#endif  // HAS_DNN_MODULE

#endif