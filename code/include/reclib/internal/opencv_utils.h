#ifndef INTERNAL_OPENCV_UTILS
#define INTERNAL_OPENCV_UTILS

#include <reclib/assert.h>

#if WITH_CUDA
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#endif

#include "reclib/common.h"
#include "reclib/data_types.h"



namespace reclib {
namespace utils {

#if WITH_CUDA  
#if HAS_OPENCV_MODULE

template <typename T>
struct step_functor : public thrust::unary_function<int, int> {
  int columns;
  int step;
  int channels;
  RECLIB_HD step_functor(int columns_, int step_, int channels_ = 1)
      : columns(columns_), step(step_), channels(channels_){};
  RECLIB_HOST step_functor(GpuMat& mat) {
    _RECLIB_ASSERT_EQ(mat.depth(), cv::DataType<T>::depth);

    columns = mat.cols;
    step = mat.step / sizeof(T);
    channels = mat.channels();
  }
  RECLIB_HD int operator()(int x) const {
    int row = x / columns;
    int idx = (row * step) + (x % columns) * channels;
    return idx;
  }
};

/*
    @Brief GpuMatBeginItr returns a thrust compatible iterator to the
beginning of a GPU CpuMat's memory.
    @Param CpuMat is the input Matrix
    @Param channel is the channel of the Matrix that the iterator is
accessing. If set to -1, the iterator will access every element in sequential
order
*/
template <typename T>
thrust::permutation_iterator<
    thrust::device_ptr<T>,
    thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatBeginItr(GpuMat mat, int offset = 0, int channel = 0) {
  if (channel == -1) {
    mat = mat.reshape(1);
    channel = 0;
  }
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(mat.depth(), cv::DataType<T>::depth);
  _RECLIB_ASSERT_LT(channel, mat.channels());
#endif

  return thrust::make_permutation_iterator(
      thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator(offset),
          step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}
/*
@Brief GpuMatEndItr returns a thrust compatible iterator to the end of a GPU
CpuMat's memory.
@Param CpuMat is the input Matrix
@Param channel is the channel of the Matrix that the iterator is accessing.
If set to -1, the iterator will access every element in sequential order
*/
template <typename T>
thrust::permutation_iterator<
    thrust::device_ptr<T>,
    thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatEndItr(GpuMat mat, int channel = 0) {
  if (channel == -1) {
    mat = mat.reshape(1);
    channel = 0;
  }
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(mat.depth(), cv::DataType<T>::depth);
  _RECLIB_ASSERT_LT(channel, mat.channels());
#endif

  return thrust::make_permutation_iterator(
      thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator(mat.rows * mat.cols),
          step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}

template <typename T>
thrust::permutation_iterator<
    T*,
    thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
CpuMatBeginItr(CpuMat mat, int offset = 0, int channel = 0) {
  if (channel == -1) {
    mat = mat.reshape(1);
    channel = 0;
  }
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(mat.depth(), cv::DataType<T>::depth);
  _RECLIB_ASSERT_LT(channel, mat.channels());
#endif

  return thrust::make_permutation_iterator(
      thrust::raw_pointer_cast(mat.ptr<T>(0) + channel),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator(offset),
          step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}
/*
@Brief GpuMatEndItr returns a thrust compatible iterator to the end of a GPU
CpuMat's memory.
@Param CpuMat is the input Matrix
@Param channel is the channel of the Matrix that the iterator is accessing.
If set to -1, the iterator will access every element in sequential order
*/
template <typename T>
thrust::permutation_iterator<
    T*,
    thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
CpuMatEndItr(CpuMat mat, int channel = 0) {
  if (channel == -1) {
    mat = mat.reshape(1);
    channel = 0;
  }
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(mat.depth(), cv::DataType<T>::depth);
  _RECLIB_ASSERT_LT(channel, mat.channels());
#endif
  return thrust::make_permutation_iterator(
      thrust::raw_pointer_cast(mat.ptr<T>(0) + channel),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator(mat.rows * mat.cols),
          step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}

#endif // HAS_OPENCV_MODULE
#endif // WITH_CUDA

}  // namespace utils
}  // namespace reclib

#endif