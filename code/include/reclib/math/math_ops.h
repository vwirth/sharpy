#ifndef RECLIB_MATH_OPS_H
#define RECLIB_MATH_OPS_H

#if WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <Eigen/Eigen>
#include "reclib/common.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b) ? (a) : (b))
#endif

namespace reclib {

template <typename T, typename Derived>
Eigen::Vector<T, 4> quat_conj(const Eigen::MatrixBase<Derived>& a) {
  assert(a.rows() == 4);

  return Eigen::Vector<T, 4>(-a[0], -a[1], -a[2], a[3]);
}

/**
 * Construct a skew symmetric matrix from a vector.
 * Also know as 'cross product matrix' or 'hat operator'.
 * https://en.wikipedia.org/wiki/Hat_operator
 *
 * Vector [x,y,z] transforms to Matrix
 *
 * |  0  -z   y |
 * |  z   0  -x |
 * | -y   x   0 |
 *
 */

template <typename T, typename Derived>
RECLIB_HD Eigen::Matrix<T, 3, 3> skew(
    const Eigen::MatrixBase<Derived>& a) {
  assert(a.rows() == 3);

  Matrix<T, 3, 3> m;
  // clang-format off
    m <<
        T(0),      T(-a(2)),  T(a(1)),
        T(a(2)),   T(0),      T(-a(0)),
        T(-a(1)),  T(a(0)),   T(0);
  // clang-format on

  return m;
}

template <typename T1, typename T2>
int logn(T1 value, T2 base) {
  return (int)(std::log(value) / std::log(base));
}

template <typename T>
RECLIB_HD int sign(T a) {
  if (a == T(0)) {
    return 0;
  } else if (a > T(0)) {
    return 1;
  } else {
    return -1;
  }
}

template <typename T>
RECLIB_HD int sign_nonzero(T a) {
  if (a == T(0)) {
    return 1;
  } else if (a > T(0)) {
    return 1;
  } else {
    return -1;
  }
}

inline RECLIB_HD float rad2deg(float x) {
  return x * (180.0f / M_PI);
}
}  // namespace reclib

#endif