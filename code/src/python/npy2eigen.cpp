#include "reclib/python/npy2eigen.h"

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
reclib::python::load_float_matrix_cm(const cnpy::NpyArray& raw, size_t rows,
                                     size_t cols) {
  size_t dwidth = raw.word_size;
  assert(dwidth == 4 || dwidth == 8);  // float or double
  if (raw.fortran_order) {
    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<float>(), rows, cols)
          .template cast<float>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<double>(), rows, cols)
          .template cast<float>();
    }
  } else {
    // throw std::runtime_error(
    //     "Error: raw numpy array is row major. Try invoking "
    //     "'load_float_matrix_rm'.");
    // return Eigen::Matrix<float, 1, 1, Eigen::ColMajor>::Zero();
    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<float>(), rows, cols)
          .template cast<float>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<double>(), rows, cols)
          .template cast<float>();
    }
  }
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
reclib::python::load_float_matrix_rm(const cnpy::NpyArray& raw, size_t rows,
                                     size_t cols) {
  size_t dwidth = raw.word_size;
  assert(dwidth == 4 || dwidth == 8);  // float or double
  if (raw.fortran_order) {
    // throw std::runtime_error(
    //     "Error: raw numpy array is column major. Try invoking "
    //     "'load_float_matrix_cm'.");
    // return Eigen::Matrix<float, 1, 1, Eigen::RowMajor>::Zero();

    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<float>(), rows, cols)
          .template cast<float>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<double>(), rows, cols)
          .template cast<float>();
    }
  } else {
    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<float>(), rows, cols)
          .template cast<float>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<double>(), rows, cols)
          .template cast<float>();
    }
  }
}

Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
reclib::python::load_uint_matrix_cm(const cnpy::NpyArray& raw, size_t rows,
                                    size_t cols) {
  size_t dwidth = raw.word_size;
  assert(dwidth == 4 || dwidth == 8);
  if (raw.fortran_order) {
    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<uint32_t>(), rows, cols)
          .template cast<uint32_t>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<uint64_t>(), rows, cols)
          .template cast<uint32_t>();
    }
  } else {
    // throw std::runtime_error(
    //     "Error: raw numpy array is row major. Try invoking "
    //     "'load_float_matrix_rm'.");
    // return Eigen::Matrix<uint32_t, 1, 1, Eigen::ColMajor>::Zero();
    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<uint32_t>(), rows, cols)
          .template cast<uint32_t>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<uint64_t>(), rows, cols)
          .template cast<uint32_t>();
    }
  }
}

Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
reclib::python::load_uint_matrix_rm(const cnpy::NpyArray& raw, size_t rows,
                                    size_t cols) {
  size_t dwidth = raw.word_size;
  assert(dwidth == 4 || dwidth == 8);
  if (raw.fortran_order) {
    // throw std::runtime_error(
    //     "Error: raw numpy array is column major. Try invoking "
    //     "'load_float_matrix_cm'.");
    // return Eigen::Matrix<uint32_t, 1, 1, Eigen::RowMajor>::Zero();
    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<uint32_t>(), rows, cols)
          .template cast<uint32_t>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<uint64_t>(), rows, cols)
          .template cast<uint32_t>();
    }
  } else {
    if (dwidth == 4) {
      return Eigen::template Map<const Eigen::Matrix<
          uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<uint32_t>(), rows, cols)
          .template cast<uint32_t>();
    } else {
      return Eigen::template Map<const Eigen::Matrix<
          uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                 raw.data<uint64_t>(), rows, cols)
          .template cast<uint32_t>();
    }
  }
}

#if HAS_DNN_MODULE

torch::Tensor reclib::python::tensor::load_float_matrix(cnpy::NpyArray& raw,
                                                        int rows, int cols) {
  size_t dwidth = raw.word_size;
  assert(dwidth == 4 || dwidth == 8);  // float or double
  if (raw.fortran_order) {
    // column-major
    if (dwidth == 4) {
      return torch::from_blob(raw.data<void>(), {cols, rows},
                              c10::TensorOptions().dtype(torch::kFloat32))
          .transpose(0, 1)
          .detach()
          .clone()
          .contiguous();
    } else {
      return torch::from_blob(raw.data<void>(), {cols, rows},
                              c10::TensorOptions().dtype(torch::kFloat64))
          .transpose(0, 1)
          .toType(torch::kFloat32)
          .detach()
          .clone()
          .contiguous();
    }
  } else {
    // row-major
    if (dwidth == 4) {
      return torch::from_blob(raw.data<void>(), {rows, cols},
                              c10::TensorOptions().dtype(torch::kFloat32))
          .detach()
          .clone();
    } else {
      return torch::from_blob(raw.data<void>(), {rows, cols},
                              c10::TensorOptions().dtype(torch::kFloat64))
          .toType(torch::kFloat32)
          .detach()
          .clone();
    }
  }
}

torch::Tensor reclib::python::tensor::load_uint_matrix(cnpy::NpyArray& raw,
                                                       int rows, int cols) {
  size_t dwidth = raw.word_size;
  assert(dwidth == 4 || dwidth == 8);  // float or double
  if (raw.fortran_order) {
    // column-major
    if (dwidth == 4) {
      return torch::from_blob(raw.data<void>(), {cols, rows},
                              c10::TensorOptions().dtype(torch::kInt32))
          .transpose(0, 1)
          .detach()
          .clone()
          .contiguous();
    } else {
      return torch::from_blob(raw.data<void>(), {cols, rows},
                              c10::TensorOptions().dtype(torch::kInt64))
          .transpose(0, 1)
          .detach()
          .clone()
          .contiguous();
    }
  } else {
    // row-major
    if (dwidth == 4) {
      return torch::from_blob(raw.data<void>(), {rows, cols},
                              c10::TensorOptions().dtype(torch::kInt32))
          .detach()
          .clone();
    } else {
      return torch::from_blob(raw.data<void>(), {rows, cols},
                              c10::TensorOptions().dtype(torch::kInt64))
          .detach()
          .clone();
    }
  }
}

#endif  // HAS_DNN_MODULE