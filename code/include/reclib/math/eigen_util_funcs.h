#ifndef RECLIB_MATH_EIGEN_UTIL_FUNCS
#define RECLIB_MATH_EIGEN_UTIL_FUNCS

#include "reclib/assert.h"
#include "reclib/data_types.h"

namespace reclib {

template <typename T = float>
Eigen::Vector<T, 3> make_Vector3(const T* data) {
  return Eigen::Vector<T, 3>(data[0], data[1], data[2]);
}

template <typename T = float>
Eigen::Vector<T, 4> make_Vector4(const T* data) {
  return Eigen::Vector<T, 4>(data[0], data[1], data[2], data[3]);
}

template <typename Derived, typename T = float>
Eigen::Vector<T, 4> make_Vector4(const Eigen::MatrixBase<Derived>& v,
                                 T fill = T(0)) {
  _RECLIB_ASSERT_EQ(v.cols(), 1);
  _RECLIB_ASSERT_LE(v.rows(), 4);

  Eigen::Vector<T, 4> r = Eigen::Vector<T, 4>::Zero();
  for (unsigned int i = 0; i < v.rows(); i++) {
    r[i] = T(v[i]);
  }
  if (fill != 0) {
    for (unsigned int i = v.rows(); i < 4; i++) {
      r[i] = fill;
    }
  }

  return r;
}

template <typename Derived>
vec4 make_vec4(const Eigen::MatrixBase<Derived>& v, float fill = 0) {
  return make_Vector4(v, fill);
}

template <typename Derived, typename T = float>
Eigen::Vector<T, 3> make_Vector3(const Eigen::MatrixBase<Derived>& v,
                                 T fill = T(0)) {
  _RECLIB_ASSERT_EQ(v.cols(), 1);
  _RECLIB_ASSERT_LE(v.rows(), 3);

  Eigen::Vector<T, 3> r = Eigen::Vector<T, 3>::Zero();
  for (unsigned int i = 0; i < v.rows(); i++) {
    r[i] = T(v[i]);
  }
  if (fill != 0) {
    for (unsigned int i = v.rows(); i < 3; i++) {
      r[i] = fill;
    }
  }

  return r;
}
}  // namespace reclib

#endif