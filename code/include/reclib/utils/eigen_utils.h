#ifndef RECLIB_UTILS_EIGEN_UTILS_H
#define RECLIB_UTILS_EIGEN_UTILS_H

#if WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <Eigen/Eigen>
#include <memory>

#include "reclib/assert.h"
#include "reclib/common.h"
#include "reclib/data_types.h"

// euler angles must be in radians
template <typename T>
inline RECLIB_HD Eigen::Matrix<T, 4, 4> MatrixFromEulerTrans(T alpha, T beta,
                                                             T gamma, T tx,
                                                             T ty, T tz) {
  Eigen::Matrix<T, 3, 3> yaw =
      Eigen::AngleAxis<T>(alpha, Eigen::Vector<T, 3>::UnitX())
          .toRotationMatrix();
  Eigen::Matrix<T, 3, 3> pitch =
      Eigen::AngleAxis<T>(beta, Eigen::Vector<T, 3>::UnitY())
          .toRotationMatrix();
  Eigen::Matrix<T, 3, 3> roll =
      Eigen::AngleAxis<T>(gamma, Eigen::Vector<T, 3>::UnitZ())
          .toRotationMatrix();

  Eigen::Matrix<T, 4, 4> mat = Eigen::Matrix<T, 4, 4>::Identity(4, 4);
  Eigen::Matrix<T, 3, 3> rot = roll * pitch * yaw;

  mat.template block<3, 3>(0, 0) = rot;
  mat.template block<3, 1>(0, 3) << tx, ty, tz;

  for (unsigned int x = 0; x < 4; x++) {
    for (unsigned int y = 0; y < 4; y++) {
      if (abs(mat(x, y)) < T(1e-4f)) {
        mat(x, y) = T(0);
      }
      // mat(x, y) = round(mat(x, y));
    }
  }
  return mat;
}

namespace reclib {
template <int BLOCK_ROWS, int BLOCK_COLS, typename Derived>
inline void printBlocksAboveThresh(const Eigen::MatrixBase<Derived>& mat,
                                   float thresh) {
#if RECLIB_DEBUG_MODE
  _RECLIB_ASSERT_EQ(mat.rows() % BLOCK_ROWS, 0);
  _RECLIB_ASSERT_EQ(mat.cols() % BLOCK_COLS, 0);
#endif

  unsigned int num_blocks_row = mat.rows() / BLOCK_ROWS;
  unsigned int num_blocks_col = mat.cols() / BLOCK_COLS;

  for (unsigned int i = 0; i < num_blocks_row; i++) {
    bool nonzero_row = false;
    for (unsigned int j = 0; j < num_blocks_col; j++) {
      if (mat.template block<BLOCK_ROWS, BLOCK_COLS>(i * BLOCK_ROWS,
                                                     j * BLOCK_COLS)
              .cwiseAbs()
              .sum() > thresh) {
        std::cout << mat.template block<BLOCK_ROWS, BLOCK_COLS>(i * BLOCK_ROWS,
                                                                j * BLOCK_COLS)
                  << std::endl;
        std::cout << "--" << std::endl;
        nonzero_row = true;
      }
    }
    if (nonzero_row) std::cout << "--------------------------" << std::endl;
  }
}

template <typename Derived>
inline void printAboveThresh(const Eigen::MatrixBase<Derived>& mat,
                             float thresh) {
  bool first_nonzero = true;

  for (unsigned int i = 0; i < mat.rows(); i++) {
    bool nonzero_row = false;
    for (unsigned int j = 0; j < mat.cols(); j++) {
      if (abs(mat(i, j)) > thresh) {
        std::cout << mat(i, j);
        first_nonzero = true;
        nonzero_row = true;
        if (j < mat.cols() - 1) {
          std::cout << ",";
        }
      } else if (first_nonzero) {
        if (j < mat.cols() - 1) std::cout << "...";
        first_nonzero = false;
      }
    }
    if (nonzero_row) std::cout << std::endl;
  }
}

template <typename Derived>
inline void printNonzero(const Eigen::MatrixBase<Derived>& mat) {
  printAboveThresh(mat, 0);
}

template <typename T, int DIMS>
inline Eigen::Matrix<T, DIMS, DIMS> triangular2dense(float* tri_mat) {
  Eigen::Matrix<T, DIMS, DIMS> mat;

  unsigned int offset = 0;
  for (unsigned int row = 0; row < DIMS; row++) {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> row_vec(tri_mat + offset,
                                                         DIMS - row);
    mat.block(row, row, 1, DIMS - row) = row_vec.transpose();
    mat.block(row, row, DIMS - row, 1) = row_vec;
    offset += DIMS - row;
  }

  return mat;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> triangular2dense(
    float* tri_mat, unsigned int dims) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat(dims, dims);

  unsigned int offset = 0;
  for (unsigned int row = 0; row < dims; row++) {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> row_vec(tri_mat + offset,
                                                         dims - row);
    mat.block(row, row, 1, dims - row) = row_vec.transpose();
    mat.block(row, row, dims - row, 1) = row_vec;
    offset += dims - row;
  }

  return mat;
}

}  // namespace reclib

#endif