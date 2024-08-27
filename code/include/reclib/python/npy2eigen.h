#ifndef RECLIB_PYTHON_NPY2EIGEN_H
#define RECLIB_PYTHON_NPY2EIGEN_H

#include <cnpy.h>
#if HAS_DNN_MODULE
#include <torch/torch.h>
#endif

#include "reclib/data_types.h"

namespace reclib {
namespace python {

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
load_float_matrix_cm(const cnpy::NpyArray& raw, size_t rows, size_t cols);

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
load_float_matrix_rm(const cnpy::NpyArray& raw, size_t rows, size_t cols);

Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
load_uint_matrix_rm(const cnpy::NpyArray& raw, size_t rows, size_t cols);

Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
load_uint_matrix_cm(const cnpy::NpyArray& raw, size_t rows, size_t cols);

#if HAS_DNN_MODULE

namespace tensor {
torch::Tensor load_float_matrix(cnpy::NpyArray& raw, int rows, int cols);
torch::Tensor load_uint_matrix(cnpy::NpyArray& raw, int rows, int cols);
}  // namespace tensor
#endif  // HAS_DNN_MODULE

}  // namespace python
}  // namespace reclib

#endif  // RECLIB_PYTHON_NPY2EIGEN_H