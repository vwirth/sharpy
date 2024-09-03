/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#ifndef RECLIB_MATH_EIGEN_GLM_INTERFACE_H
#define RECLIB_MATH_EIGEN_GLM_INTERFACE_H

#include <Eigen/Eigen>

#include "reclib/data_types.h"

// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/darglein/saiga
//
// MIT License
// Copyright (c) 2021 Darius Rückert <darius.rueckert@fau.de>
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ---------------------------------------------------------------------------

namespace reclib {
template <typename Derived>
typename Derived::PlainObject clamp(const Eigen::EigenBase<Derived>& x,
                                    const Eigen::EigenBase<Derived>& minVal,
                                    const Eigen::EigenBase<Derived>& maxVal) {
  typename Derived::PlainObject tmp =
      x.derived().array().max(minVal.derived().array());
  return tmp.array().min(maxVal.derived().array());
}

template <typename Derived>
typename Derived::PlainObject saturate(const Eigen::EigenBase<Derived>& x) {
  typename Derived::PlainObject z, o;
  z.setZero();
  o.setOnes();
  return clamp<Derived>(x, z, o);
}

template <typename Derived1, typename Derived2>
Derived1 mix(const Eigen::MatrixBase<Derived1>& a,
             const Eigen::MatrixBase<Derived2>& b,
             typename Derived1::Scalar alpha) {
  return (1 - alpha) * a + alpha * b;
}

template <typename Derived1, typename Derived2>
typename Derived1::Scalar dot(const Eigen::MatrixBase<Derived1>& a,
                              const Eigen::MatrixBase<Derived2>& b) {
  return a.dot(b);
}

template <typename Derived1, typename Derived2>
typename Eigen::MatrixBase<Derived1>::PlainObject cross(
    const Eigen::MatrixBase<Derived1>& v1,
    const Eigen::MatrixBase<Derived2>& v2) {
  return v1.cross(v2);
}

template <typename Derived1, typename Derived2>
typename Derived1::Scalar distance(const Eigen::MatrixBase<Derived1>& v1,
                                   const Eigen::MatrixBase<Derived2>& v2) {
  return (v1 - v2).norm();
}

template <typename Derived1>
typename Eigen::MatrixBase<Derived1>::PlainObject inverse(
    const Eigen::MatrixBase<Derived1>& v1) {
  return v1.inverse();
}

template <typename Derived1>
Derived1 transpose(const Eigen::MatrixBase<Derived1>& v1) {
  return v1.transpose();
}

template <typename Derived>
constexpr typename Derived::Scalar length(const Eigen::MatrixBase<Derived>& v) {
  return v.norm();
}

//  constexpr float abs(float v)
//{
//     return std::abs(v);
// }
//
//  constexpr double abs(double v)
//{
//     return std::abs(v);
// }

using std::abs;

template <typename Derived>
constexpr Derived abs(const Eigen::MatrixBase<Derived>& v) {
  return v.array().abs();
}

inline vec3 abs(const vec3& v) { return v.cwiseAbs(); }
inline vec3 sign(const vec3& v) { return v.cwiseSign(); }

template <typename Derived>
typename Eigen::MatrixBase<Derived>::PlainObject normalize(
    const Eigen::MatrixBase<Derived>& v) {
  return v.normalized();
}

template <typename Derived>
typename Eigen::QuaternionBase<Derived>::PlainObject normalize(
    const Eigen::QuaternionBase<Derived>& q) {
  return q.normalized();
}

template <typename Derived>
typename Eigen::QuaternionBase<Derived>::PlainObject slerp(
    const Eigen::QuaternionBase<Derived>& a,
    const Eigen::QuaternionBase<Derived>& b, typename Derived::Scalar alpha) {
  return a.slerp(alpha, b);
}

//
// Pixar Revised ONB
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
//
// n is aligned to the z-axis
//
template <typename Derived>
Matrix<typename Derived::Scalar, 3, 3> onb(
    const Eigen::MatrixBase<Derived>& n) {
  static_assert(
      Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1,
      "Input must be 3x1");

  using T = typename Derived::Scalar;
  using Mat3 = Matrix<T, 3, 3>;
  using Vec3 = Matrix<T, 3, 1>;

  T sign = n(2) > 0 ? 1.0f : -1.0f;  // emulate copysign
  T a = -1.0f / (sign + n[2]);
  T b = n[0] * n[1] * a;
  Mat3 v;
  v.col(2) = n;
  v.col(1) = Vec3(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
  v.col(0) = Vec3(b, sign + n[1] * n[1] * a, -n[1]);
  return v;
}

/**
 * Simple ONB from a direction and an up vector.
 */
template <typename Derived1, typename Derived2>
Matrix<typename Derived1::Scalar, 3, 3> onb(
    const Eigen::MatrixBase<Derived1>& dir,
    const Eigen::MatrixBase<Derived2>& up) {
  using T = typename Derived1::Scalar;
  using Mat3 = Matrix<T, 3, 3>;

  Mat3 R;
  R.col(2) = dir.normalized();
  R.col(1) = up.normalized();
  R.col(0) = R.col(1).cross(R.col(2)).normalized();
  // make sure it works even if dir and up are not orthogonal
  R.col(1) = R.col(2).cross(R.col(0));
  return R;
}
template <typename T>
constexpr T epsilon() {
  static_assert(std::is_floating_point<T>::value,
                "Only allowed for floating point types.");
  return std::numeric_limits<T>::epsilon();
}

template <typename T>
constexpr T pi() {
  static_assert(std::is_floating_point<T>::value,
                "Only allowed for floating point types.");
  return T(3.14159265358979323846);
}

template <typename T>
constexpr T two_pi() {
  static_assert(std::is_floating_point<T>::value,
                "Only allowed for floating point types.");
  return pi<T>() * T(2);
}

inline float clamp(float v, float mi, float ma) {
  return std::min(ma, std::max(mi, v));
}

inline float smoothstep(float edge0, float edge1, float x) {
  float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

inline float fract(float a) { return a - std::floor(a); }

inline double fract(double a) { return a - std::floor(a); }

template <typename Derived>
constexpr typename Derived::PlainObject fract(
    const Eigen::MatrixBase<Derived>& v) {
  return (v.array() - v.array().floor());
}

template <typename T>
constexpr T degrees(T a) {
  static_assert(std::is_floating_point<T>::value,
                "Only allowed for floating point types.");
  return a * T(180.0) / pi<T>();
}

template <typename T>
constexpr T radians(T a) {
  static_assert(std::is_floating_point<T>::value,
                "Only allowed for floating point types.");
  return a / T(180.0) * pi<T>();
}

inline float mix(const float& a, const float& b, float alpha) {
  return (1 - alpha) * a + alpha * b;
}

// Maybe use more advanced implementation from boost?
// https://www.boost.org/doc/libs/1_51_0/boost/math/special_functions/sinc.hpp
template <typename T>
inline T sinc(const T x) {
  if (abs(x) >= std::numeric_limits<T>::epsilon()) {
    return (sin(x) / x);
  } else {
    return T(1);
  }
}

template <typename T>
Eigen::Matrix<T, 4, 4> translate(const Eigen::Vector<T, 3>& t) {
  Eigen::Matrix<T, 4, 4> m = Eigen::Matrix<T, 4, 4>::Identity();
  m.template block<3, 1>(0, 3) = t;
  return m;
}
template <typename T>
Eigen::Matrix<T, 4, 4> scale(T t) {
  Eigen::Matrix<T, 4, 4> m2 = Eigen::Matrix<T, 4, 4>::Identity();
  m2(0, 0) = t;
  m2(1, 1) = t;
  m2(2, 2) = t;
  return m2;
}

extern mat4 scale(const vec3& t);

// angle given in radians
extern mat4 rotate(float angle, const vec3& axis);

extern mat4 scale(const mat4& m, const vec3& t);
extern mat4 translate(const mat4& m, const vec3& t);
extern mat4 rotate(const mat4& m, float angle, const vec3& axis);
extern vec3 rotate(const vec3& v, float angle, const vec3& axis);

extern Eigen::Quaternionf rotate(const Eigen::Quaternionf& q, float angle,
                                 const vec3& axis);

extern Eigen::Quaternionf angleAxis(float angle, const vec3& axis);
extern Eigen::Quaternionf mix(const Eigen::Quaternionf& a,
                              const Eigen::Quaternionf& b, float alpha);
extern Eigen::Quaterniond mix(const Eigen::Quaterniond& a,
                              const Eigen::Quaterniond& b, double alpha);
extern Eigen::Quaternionf quat_cast(const mat3& m);
extern Eigen::Quaternionf quat_cast(const mat4& m);
extern Eigen::Quaternionf inverse(const Eigen::Quaternionf& q);

extern Eigen::Quaternionf rotation(const vec3& a, const vec3& b);

extern vec4 make_vec4(float x, float y, float z, float w);
extern vec4 make_vec4(const vec3& v, float a);
extern vec4 make_vec4(const vec2& v, const vec2& v2);
extern vec4 make_vec4(float a);
extern vec4 make_vec4(const float* a);

extern vec3 make_vec3(const vec2& v, float a);
extern vec3 make_vec3(float a);
extern vec3 make_vec3(const vec4& a);
extern vec3 make_vec3(const float* a);

template <typename Derived>
Derived min(const Eigen::MatrixBase<Derived>& a,
            const Eigen::MatrixBase<Derived>& b) {
  return a.cwiseMin(b);
}

template <typename Derived>
Derived max(const Eigen::MatrixBase<Derived>& a,
            const Eigen::MatrixBase<Derived>& b) {
  return a.cwiseMax(b);
}

// template <typename Derived>
// Derived operator/(const Eigen::MatrixBase<Derived>& a,
//                   const Eigen::MatrixBase<Derived>& b) {
//   return a.cwiseQuotient(b);
// }

template <typename Derived>
Derived operator/(const Eigen::MatrixBase<Derived>& a,
                  const Eigen::MatrixBase<Derived>& b) {
  return a.cwiseQuotient(b);
}

template <typename Op, typename Derived>
Derived operator/(const Eigen::CwiseBinaryOp<Op, Derived, Derived>& a,
                  const Eigen::CwiseBinaryOp<Op, Derived, Derived>& b) {
  return a.cwiseQuotient(b);
}
// template <typename Derived>
// Derived operator*(const Eigen::MatrixBase<Derived>& a,
//                   const Eigen::MatrixBase<Derived>& b);

extern vec2 make_vec2(float a);
extern vec2 make_vec2(const float* a);
extern vec2 make_vec2(const vec3& a);
extern vec2 make_vec2(float a, float b);
extern vec2 make_vec2(const ivec2& a);

extern ivec2 make_ivec2(int a, int b);
extern ucvec4 make_ucvec4(const ucvec3& v, unsigned char a);
extern mat4 make_mat4(float a00, float a01, float a02, float a03, float a10,
                      float a11, float a12, float a13, float a20, float a21,
                      float a22, float a23, float a30, float a31, float a32,
                      float a33);
extern mat4 make_mat4_row_major(float a00, float a01, float a02, float a03,
                                float a10, float a11, float a12, float a13,
                                float a20, float a21, float a22, float a23,
                                float a30, float a31, float a32, float a33);
extern mat4 make_mat4(const mat3& m);
extern mat3 make_mat3(const mat4& m);
extern mat3 make_mat3(float a00, float a01, float a02, float a10, float a11,
                      float a12, float a20, float a21, float a22);

extern mat4 make_mat4(const Eigen::Quaternionf& q);
extern mat3 make_mat3(const Eigen::Quaternionf& q);
extern Eigen::Quaternionf make_quat(float x, float y, float z, float w);
extern vec4 quat_to_vec4(const Eigen::Quaternionf& q);
extern Eigen::Quaternionf make_quat(const mat3& m);
extern Eigen::Quaternionf make_quat(const mat4& m);

extern mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up);
extern mat4 perspective(float fovy, float aspect, float zNear, float zFar);
extern mat4 ortho(float left, float right, float bottom, float top, float zNear,
                  float zFar);

extern mat4 createTRSmatrix(const vec3& t, const Eigen::Quaternionf& r,
                            const vec3& s);
extern mat4 createTRSmatrix(const vec4& t, const Eigen::Quaternionf& r,
                            const vec4& s);

}  // namespace reclib

#endif
