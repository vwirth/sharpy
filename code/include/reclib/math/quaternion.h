#ifndef QUATERNION_H
#define QUATERNION_H

#if WITH_CUDA
  #include <cuda_runtime.h>
  #include <thrust/device_vector.h>
  #include <thrust/host_vector.h>
#endif

#if HAS_OPENCV_MODULE
  #include <opencv2/core.hpp>
  #include <opencv2/core/affine.hpp>
  #include <utility>
#endif

#include "reclib/common.h"
#include "reclib/data_types.h"
#include "reclib/math/math_ops.h"
#include "reclib/platform.h"

namespace reclib {
class _API Quaternion {
 public:
  RECLIB_HD Quaternion() : coeff(vec4(0.f, 0.f, 0.f, 0.f)){};
  RECLIB_HD Quaternion(const vec4& coeff) : coeff(coeff){};
  RECLIB_HD Quaternion(const vec3& coeff)
      : coeff(coeff.x(), coeff.y(), coeff.z(), 0){};
  RECLIB_HD Quaternion(float i, float j, float k, float w)
      : coeff(vec4(i, j, k, w)){};

  // Generate a quaternion from rotation of a Rt Matrix.
  RECLIB_HOST Quaternion(const mat4& r) {
    // Compute trace of Matrix
    float tr = (float)r(0, 0) + r(1, 1) + r(2, 2);

    if (tr > 0) {
      float S = sqrt(tr + 1) * 2;
      coeff[0] = (r(2, 1) - r(1, 2)) / S;
      coeff[1] = (r(0, 2) - r(2, 0)) / S;
      coeff[2] = (r(1, 0) - r(0, 1)) / S;
      coeff[3] = 0.25 * S;
    } else if (r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2)) {
      float S = sqrt(1 + r(0, 0) - r(1, 1) - r(2, 2)) * 2;
      coeff[3] = (r(2, 1) - r(1, 2)) / S;
      coeff[2] = (r(0, 2) + r(2, 0)) / S;
      coeff[1] = (r(1, 0) + r(0, 1)) / S;
      coeff[0] = 0.25 * S;
    } else if (r(1, 1) > r(2, 2)) {
      float S = sqrt(1 - r(0, 0) + r(1, 1) - r(2, 2)) * 2;
      coeff[2] = (r(2, 1) + r(1, 2)) / S;
      coeff[3] = (r(0, 2) - r(2, 0)) / S;
      coeff[0] = (r(1, 0) + r(0, 1)) / S;
      coeff[1] = 0.25 * S;
    } else {
      float S = sqrt(1 - r(0, 0) - r(1, 1) + r(2, 2)) * 2;
      coeff[1] = (r(2, 1) + r(1, 2)) / S;
      coeff[0] = (r(0, 2) + r(2, 0)) / S;
      coeff[3] = (r(1, 0) - r(0, 1)) / S;
      coeff[2] = 0.25 * S;
    }
  }

  RECLIB_HD Quaternion conjugate() const {
    return Quaternion(vec4(-coeff.x(), -coeff.y(), -coeff.z(), coeff.w()));
  }

  RECLIB_HD Quaternion inverse() const {
    float norm = coeff.norm();
    return (norm > 0) * conjugate() / MAX(norm, 1e-8f);
  }

  RECLIB_HD float normalize() {
    float n = (float)coeff.norm();
    coeff = (n == 0) * coeff + (n != 0) * coeff * (1 / fmax(n, 1e-6f));
    return n;
  }

  RECLIB_HD Quaternion normalized() const {
    float n = (float)coeff.norm();

    return Quaternion(
        (vec4)((n == 0) * coeff + (n != 0) * coeff * (1 / fmax(n, 1e-6f))));
  }

  RECLIB_HD float norm() { return (float)coeff.norm(); }

  RECLIB_HD mat4 toMatrix() const {
    float W = coeff[3], X = -coeff[0], Y = -coeff[1], Z = -coeff[2];
    float xx = X * X, xy = X * Y, xz = X * Z, xw = X * W;
    float yy = Y * Y, yz = Y * Z, yw = Y * W, zz = Z * Z;
    float zw = Z * W;

    mat4 rot;
    rot << 1.f - 2.f * (yy + zz), 2.f * (xy + zw), 2.f * (xz - yw), 0,
        2.f * (xy - zw), 1.f - 2.f * (xx + zz), 2.f * (yz + xw), 0,
        2.f * (xz + yw), 2.f * (yz - xw), 1.f - 2.f * (xx + yy), 0, 0, 0, 0, 1;

    return rot;
  }

  template <typename T>
  static RECLIB_HOST Eigen::Matrix<T, 4, 4> toMatrix(T const* data) {
    Eigen::Map<const Eigen::Vector<T, 4>> coeff(data);

    T W = coeff[3], X = -coeff[0], Y = -coeff[1], Z = -coeff[2];
    T xx = X * X, xy = X * Y, xz = X * Z, xw = X * W;
    T yy = Y * Y, yz = Y * Z, yw = Y * W, zz = Z * Z;
    T zw = Z * W;

    Eigen::Matrix<T, 4, 4> rot;
    rot << T(1.f) - T(2.f) * (yy + zz), T(2.f) * (xy + zw), T(2.f) * (xz - yw),
        T(0), T(2.f) * (xy - zw), T(1.f) - T(2.f) * (xx + zz),
        T(2.f) * (yz + xw), T(0), T(2.f) * (xz + yw), T(2.f) * (yz - xw),
        T(1.f) - T(2.f) * (xx + yy), T(0), T(0), T(0), T(0), T(1.f);

    return rot;
  }

  RECLIB_HD float w() const { return coeff[3]; }
  RECLIB_HD float i() const { return coeff[0]; }
  RECLIB_HD float j() const { return coeff[1]; }
  RECLIB_HD float k() const { return coeff[2]; }
  RECLIB_HD float x() const { return coeff[0]; }
  RECLIB_HD float y() const { return coeff[1]; }
  RECLIB_HD float z() const { return coeff[2]; }

  RECLIB_HD float norm() const { return (float)coeff.norm(); }

  RECLIB_HD vec3 imag() const { return coeff.block<3, 1>(0, 0); }

  RECLIB_HD float real() const { return coeff[3]; }

  RECLIB_HD vec4 data() const { return coeff; }

  RECLIB_HD friend Quaternion operator*(float a,
                                                  const Quaternion& q) {
    vec4 newQ = a * q.coeff;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
  }
  RECLIB_HD friend Quaternion operator*(const Quaternion& q,
                                                  float a) {
    return a * q;
  }

  RECLIB_HD friend Quaternion operator*(const Quaternion& p,
                                                  const Quaternion& q) {
    vec4 r;
    // r[0] = (p.w() * q.x() + p.x() * q.w() + p.y() * q.z() - p.z() * q.y());
    // r[1] = (p.w() * q.y() - p.x() * q.z() + p.y() * q.w() + p.z() * q.x());
    // r[2] = (p.w() * q.z() + p.x() * q.y() - p.y() * q.x() + p.z() * q.w());
    // r[3] = (p.w() * q.w() - p.x() * q.x() - p.y() * q.y() - p.z() * q.z());

    r[3] = (p.w() * q.w() - p.imag().dot(q.imag()));
    r.head(3) =
        p.real() * q.imag() + q.real() * p.imag() + p.imag().cross(q.imag());

    return Quaternion(r);
  }

  RECLIB_HD friend vec3 operator*(const Quaternion& p,
                                            const vec3& q) {
    Quaternion point(vec4(q.x(), q.y(), q.z(), 0));
    Quaternion result = p * point * p.inverse();

    return result.coeff.head(3);
  }

  RECLIB_HD friend Quaternion operator/(const Quaternion& q,
                                                  float a) {
    vec4 newQ = q.coeff / a;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
  }

  RECLIB_HD friend Quaternion operator+(const Quaternion& q1,
                                                  const Quaternion& q2) {
    vec4 newQ = q1.coeff + q2.coeff;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
  }

  RECLIB_HD friend Quaternion operator~(const Quaternion& q) {
    return Quaternion(-q[0], -q[1], -q[2], q[3]);
  }

  RECLIB_HD friend Quaternion operator-(const Quaternion& q1,
                                                  const Quaternion& q2) {
    vec4 newQ = q1.coeff - q2.coeff;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
  }

  RECLIB_HD friend Quaternion& operator+=(Quaternion& q1,
                                                    const Quaternion& q2) {
    q1.coeff += q2.coeff;
    return q1;
  }
  RECLIB_HD friend Quaternion& operator-=(Quaternion& q1,
                                                    const Quaternion& q2) {
    q1.coeff -= q2.coeff;
    return q1;
  }

  RECLIB_HD friend Quaternion& operator/=(Quaternion& q, float a) {
    q.coeff /= a;
    return q;
  }

  RECLIB_HD float operator[](const int i) const { return coeff[i]; }

  friend std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
    os << "imag: (" << q.coeff[0] << "," << q.coeff[1] << "," << q.coeff[2]
       << ") real: " << q.coeff[3];
    return os;
  }

 private:
  // w, i, j, k coefficients
  vec4 coeff;
};

template <typename T>
RECLIB_HD Eigen::Quaternion<T> operator*(
    T scalar, const Eigen::Quaternion<T>& q) {
  Eigen::Quaternion<T> quat(q);
  quat.coeffs() *= scalar;
  return quat;
}

template <typename T>
RECLIB_HD Eigen::Quaternion<T> operator/(
    T scalar, const Eigen::Quaternion<T>& q) {
  Eigen::Quaternion<T> quat(q);
  quat.coeffs() /= scalar;
  return quat;
}

template <typename T>
RECLIB_HD Eigen::Quaternion<T> operator/(
    const Eigen::Quaternion<T>& q, T scalar) {
  Eigen::Quaternion<T> quat(q);
  quat.coeffs() /= scalar;
  return quat;
}

template <typename T>
RECLIB_HD Eigen::Quaternion<T> operator*(
    const Eigen::Quaternion<T>& q, T scalar) {
  Eigen::Quaternion<T> quat(q);
  quat.coeffs() *= scalar;
  return quat;
}

template <typename T>
RECLIB_HD Eigen::Quaternion<T> operator+(
    const Eigen::Quaternion<T>& p, const Eigen::Quaternion<T>& q) {
  Eigen::Quaternion<T> quat(p);
  quat.coeffs() += q.coeffs();
  return quat;
}

template <typename T>
RECLIB_HD Eigen::Quaternion<T> operator-(
    const Eigen::Quaternion<T>& p, const Eigen::Quaternion<T>& q) {
  Eigen::Quaternion<T> quat(p);
  quat.coeffs() -= q.coeffs();
  return quat;
}

template <typename T>
class _API DualQuaternionBase {
 public:
  RECLIB_HD DualQuaternionBase()
      : q0(T(1), T(0), T(0), T(0)),
        qe(T(0), T(0), T(0), T(0)){

        };

  RECLIB_HD DualQuaternionBase(T q0x, T q0y, T q0z, T q0w, T qex,
                                         T qey, T qez, T qew)
      : q0(q0w, q0x, q0y, q0z),
        qe(qew, qex, qey, qez){

        };

  RECLIB_HD DualQuaternionBase(const Eigen::Quaternion<T>& q0,
                                         const Eigen::Quaternion<T>& qe)
      : q0(q0),
        qe(qe){

        };

  RECLIB_HD DualQuaternionBase(const Eigen::Quaternion<T>& q0,
                                         const Eigen::Vector<T, 3>& t)
      : q0(q0) {
    qe = (T(1.f) / T(2.f)) * Eigen::Quaternion<T>(T(0), t[0], t[1], t[2]) * q0;
  };

  RECLIB_HD DualQuaternionBase(const Eigen::Vector<T, 4>& q0,
                                         const Eigen::Vector<T, 3>& t)
      : q0(q0) {
    qe = (T(1.f) / T(2.f)) * Eigen::Quaternion<T>(T(0), t[0], t[1], t[2]) * this->q0;
  };

  RECLIB_HD DualQuaternionBase(const Eigen::Vector<T, 4>& q0,
                                         const Eigen::Vector<T, 4>& qe)
      : q0(q0),
        qe(qe){

        };

  RECLIB_HD DualQuaternionBase(const Eigen::Vector<T, 3>& point)
      : q0(T(1), T(0), T(0), T(0)),
        qe(T(0), point.x(), point.y(), point.z()){

        };

  RECLIB_HD DualQuaternionBase(const Eigen::Matrix<T, 4, 4>& Rt) {
    q0 = Eigen::Quaternion<T>(
        (Eigen::Matrix<T, 3, 3>)(Rt.template block<3, 3>(0, 0)));

    if (Rt.cols() > 3) {
      Eigen::Vector<T, 3> t = Rt.template block<3, 1>(0, 3);
      qe =
          (T(1.f) / T(2.f)) * Eigen::Quaternion<T>(T(0), t[0], t[1], t[2]) * q0;
    } else {
      qe = Eigen::Quaternion<T>(T(0), T(0), T(0), T(0));
    }
  };

  RECLIB_HD DualQuaternionBase(Eigen::Quaternion<T>& q0,
                                         Eigen::Quaternion<T>& qe)
      : q0(q0), qe(qe){};

  static RECLIB_HD inline DualQuaternionBase<T> Zero() {
    return DualQuaternionBase<T>(T(0), T(0), T(0), T(0), T(0), T(0), T(0),
                                 T(0));
  }

  static RECLIB_HD inline DualQuaternionBase<T> Identity() {
    return DualQuaternionBase<T>(T(0), T(0), T(0), T(1), T(0), T(0), T(0),
                                 T(0));
  }

  RECLIB_HD Eigen::Quaternion<T> real() const { return q0; }

  RECLIB_HD Eigen::Quaternion<T> dual() const { return qe; }

  RECLIB_HD void norm(T& n1, T& n2) const {
    n1 = q0.norm();
    // normalization of dual part extracted from:
    // courses.cms.caltech.edu/cs174/projects/Cale%20Scholl%20CS174%20Dual%20Quaternion%20Blending.pdf
    n2 = q0.coeffs().dot(qe.coeffs());
  }

  RECLIB_HD DualQuaternionBase<T> normalized() const {
    // https://maxime-tournier.github.io/notes/dual-quaternions.html

    // quaternion inverse: q / |q|^2 (squared l2 norm!)
    Eigen::Quaternion<T> im = qe * q0.inverse();
    im.coeffs()[3] = T(0);
    return DualQuaternionBase<T>(q0.normalized(), im * q0.normalized());
  }

  RECLIB_HD void normalize() {
    // https://maxime-tournier.github.io/notes/dual-quaternions.html
    *this = normalized();
  }

  // Return the dual quaternion inverse
  // For unit dual quaternions dq.inverse = dq.quaternion_conjugate()
  RECLIB_HD DualQuaternionBase<T> inverse() const {
    DualQuaternionBase<T> result;
    result.q0 = q0.inverse();
    result.qe = (T(-1) * q0.inverse()) * qe * q0.inverse();
    return result;
  }

  //  Return the individual quaternion conjugates (q0, qe)* = (q0*, qe*)
  //  This is equivalent to inverse of a homogeneous matrix. It is used in
  //  applying a transformation to a line expressed in Plucker coordinates.
  RECLIB_HD DualQuaternionBase<T> quaternion_conjugate() const {
    DualQuaternionBase<T> result;
    result.q0 = q0.conjugate();
    result.qe = qe.conjugate();
    return result;
  }

  // Return the combination of the quaternion conjugate and dual number
  // conjugate
  // (q0, qe)* = (q0*, -qe*)
  //  This form is commonly used to transform a point
  RECLIB_HD DualQuaternionBase<T> conjugate() const {
    DualQuaternionBase<T> result;
    result.q0 = q0.conjugate();
    result.qe = T(-1) * qe.conjugate();
    return result;
  }

  template <typename S>
  RECLIB_HD DualQuaternionBase<S> cast() const {
    DualQuaternionBase<S> result(q0.template cast<S>(), qe.template cast<S>());
    return result;
  }

  RECLIB_HD bool isNormalized() const {
    T n1, n2;
    norm(n1, n2);
    if (n1 == T(1) && abs(n2) < T(1e-8f)) {
      return true;
    }
    return false;
  }

  RECLIB_HD Eigen::Vector<T, 8> data() const {
    return Eigen::Vector<T, 8>(q0.coeffs()[0], q0.coeffs()[1], q0.coeffs()[2],
                               q0.coeffs()[3], qe.coeffs()[0], qe.coeffs()[1],
                               qe.coeffs()[2], qe.coeffs()[3]);
  }

  RECLIB_HD friend DualQuaternionBase<T>& operator+=(
      DualQuaternionBase<T>& q1, const DualQuaternionBase<T>& q2) {
    q1.q0 = q1.q0 + q2.q0;
    q1.qe = q1.qe + q2.qe;
    return q1;
  }

  RECLIB_HD friend DualQuaternionBase<T> operator+(
      const DualQuaternionBase<T>& q1, const DualQuaternionBase<T>& q2) {
    DualQuaternionBase<T> q3 = q1;
    q3 += q2;
    return q3;
  }
  RECLIB_HD friend DualQuaternionBase<T> operator*(
      T a, const DualQuaternionBase<T>& q) {
    Eigen::Quaternion<T> newQ0 = a * q.q0;
    Eigen::Quaternion<T> newQe = a * q.qe;
    return DualQuaternionBase<T>(newQ0, newQe);
  }

  RECLIB_HD friend DualQuaternionBase<T> operator*(
      const DualQuaternionBase<T>& p, const DualQuaternionBase<T>& q) {
    DualQuaternionBase<T> result;
    result.q0 = p.q0 * q.q0;
    result.qe = p.q0 * q.qe + p.qe * q.q0;
    return result;
  }

  RECLIB_HD friend Eigen::Vector<T, 3> operator*(
      const DualQuaternionBase<T>& p, const Eigen::Vector<T, 3> q) {
    DualQuaternionBase<T> point(q);
    DualQuaternionBase<T> result = p * point * p.conjugate();
    return result.qe.coeffs().template head<3>();
  }

  RECLIB_HD Eigen::Matrix<T, 4, 4> matrix() const {
    DualQuaternionBase<T> q = *this;
    if (!isNormalized()) {
      q = normalized();
    }
    Eigen::Quaternion<T> q0 = q.real();
    Eigen::Quaternion<T> qe = q.dual();

    Eigen::Matrix<T, 4, 4> Rt = Eigen::Matrix<T, 4, 4>::Identity();

    Rt.template block<3, 3>(0, 0) = (q0).toRotationMatrix();
    Rt.template block<3, 1>(0, 3) =
        (T(2) * qe * q0.inverse()).coeffs().template head<3>();
    return Rt;
  }

  RECLIB_HD Eigen::Vector<T, 7> toQuatTrans() const {
    Eigen::Vector<T, 7> quat_trans;
    quat_trans.template head<4>() = q0.coeffs();
    quat_trans.template tail<3>() =
        (T(2) * qe * q0.inverse()).coeffs().template head<3>();
    return quat_trans;
  }

  RECLIB_HD Eigen::Vector<T, 3> translation() const {
    return (T(2) * qe * q0.inverse()).coeffs().template head<3>();
  }

  // partial derivative of se3(dqb) by lie components
  RECLIB_HD Eigen::Matrix<T, 18, 4> toMatrixJacobian(
      const Eigen::Matrix<T, 8, 6>& dqb_jacobi) const {
    Eigen::Matrix<T, 18, 4> J = Eigen::Matrix<T, 18, 4>::Zero(
        18, 4);  // 6 stacked [3x4] CpuMatrices, since we have 6 partial
    // derivatives of a [3x4] Matrix

    DualQuaternionBase<T> q = *this;
    for (unsigned int deriv_axis = 0; deriv_axis < 6; deriv_axis++) {
      J(deriv_axis * 3 + 0, 0) =
          -T(2) * (T(2) * q[1] * dqb_jacobi(1, deriv_axis) +
                   T(2) * q[2] * dqb_jacobi(2, deriv_axis));
      J(deriv_axis * 3 + 1, 0) =
          T(2) *
          (dqb_jacobi(0, deriv_axis) * q[1] + q[0] * dqb_jacobi(1, deriv_axis) +
           q[3] * dqb_jacobi(2, deriv_axis) + q[2] * dqb_jacobi(3, deriv_axis));
      J(deriv_axis * 3 + 2, 0) =
          T(2) *
          (dqb_jacobi(0, deriv_axis) * q[2] + q[0] * dqb_jacobi(2, deriv_axis) -
           q[3] * dqb_jacobi(1, deriv_axis) - q[1] * dqb_jacobi(3, deriv_axis));

      J(deriv_axis * 3 + 0, 1) =
          T(2) *
          (dqb_jacobi(0, deriv_axis) * q[1] + q[0] * dqb_jacobi(1, deriv_axis) -
           q[3] * dqb_jacobi(2, deriv_axis) - q[2] * dqb_jacobi(3, deriv_axis));
      J(deriv_axis * 3 + 1, 1) =
          -T(2) * (T(2) * q[0] * dqb_jacobi(0, deriv_axis) +
                   T(2) * q[2] * dqb_jacobi(2, deriv_axis));
      J(deriv_axis * 3 + 2, 1) =
          T(2) *
          (q[1] * dqb_jacobi(2, deriv_axis) + q[2] * dqb_jacobi(1, deriv_axis) +
           q[3] * dqb_jacobi(0, deriv_axis) + q[0] * dqb_jacobi(3, deriv_axis));

      J(deriv_axis * 3 + 0, 2) =
          T(2) *
          (dqb_jacobi(0, deriv_axis) * q[2] + q[0] * dqb_jacobi(2, deriv_axis) +
           q[3] * dqb_jacobi(1, deriv_axis) + q[1] * dqb_jacobi(3, deriv_axis));
      J(deriv_axis * 3 + 1, 2) =
          T(2) *
          (q[1] * dqb_jacobi(2, deriv_axis) + q[2] * dqb_jacobi(1, deriv_axis) -
           q[3] * dqb_jacobi(0, deriv_axis) - q[0] * dqb_jacobi(3, deriv_axis));
      J(deriv_axis * 3 + 2, 2) =
          -T(2) * (T(2) * q[0] * dqb_jacobi(0, deriv_axis) +
                   T(2) * q[1] * dqb_jacobi(1, deriv_axis));

      J(deriv_axis * 3 + 0, 3) =
          T(2) *
          (-q[7] * dqb_jacobi(0, deriv_axis) -
           dqb_jacobi(7, deriv_axis) * q[0] + q[4] * dqb_jacobi(3, deriv_axis) +
           q[3] * dqb_jacobi(4, deriv_axis) - q[5] * dqb_jacobi(2, deriv_axis) -
           q[2] * dqb_jacobi(5, deriv_axis) + q[6] * dqb_jacobi(1, deriv_axis) +
           q[1] * dqb_jacobi(6, deriv_axis));
      J(deriv_axis * 3 + 1, 3) =
          T(2) *
          (-q[7] * dqb_jacobi(1, deriv_axis) -
           dqb_jacobi(7, deriv_axis) * q[1] + q[4] * dqb_jacobi(2, deriv_axis) +
           q[2] * dqb_jacobi(4, deriv_axis) + q[5] * dqb_jacobi(3, deriv_axis) +
           q[3] * dqb_jacobi(5, deriv_axis) - q[6] * dqb_jacobi(0, deriv_axis) -
           q[0] * dqb_jacobi(6, deriv_axis));
      J(deriv_axis * 3 + 2, 3) =
          T(2) *
          (-q[7] * dqb_jacobi(2, deriv_axis) -
           dqb_jacobi(7, deriv_axis) * q[2] - q[4] * dqb_jacobi(1, deriv_axis) -
           q[1] * dqb_jacobi(4, deriv_axis) + q[5] * dqb_jacobi(0, deriv_axis) +
           q[0] * dqb_jacobi(5, deriv_axis) + q[6] * dqb_jacobi(3, deriv_axis) +
           q[3] * dqb_jacobi(6, deriv_axis));
    }

    return J;
  }

  RECLIB_HD T operator[](const int i) const {
    if (i < 4) {
      return q0.coeffs()[i];
    } else {
      return qe.coeffs()[i - 4];
    }
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const DualQuaternionBase<T>& q) {
    os << q.q0.coeffs() << "," << q.qe.coeffs() << std::endl;
    return os;
  }

 private:
  // Quaternion q0;  // rotation quaternion
  // Quaternion qe;  // translation quaternion
  Eigen::Quaternion<T> q0;
  Eigen::Quaternion<T> qe;
};
using DualQuaternion = DualQuaternionBase<float>;

// instantiation
template class _API reclib::DualQuaternionBase<float>;

RECLIB_HOST DualQuaternion DQB(const std::vector<float>& weights,
                            const std::vector<reclib::DualQuaternion>& quats);

template <typename T>
RECLIB_HOST void DQB(const T* weights, const T (*quats)[8], const int n,
                  T* output) {
  for (int i = 0; i < n; i++) {
    for (int dim = 0; dim < 8; dim++) {
      output[dim] += weights[i] * quats[i][dim];
      // std::cout << "output " << dim << ":  " << output[dim] << std::endl;
    }
  }
  Eigen::Map<Eigen::Vector<T, 4>> q0(output);
  Eigen::Map<Eigen::Vector<T, 4>> qe(output + 4);
  T norm_rot = q0.norm();
  if (norm_rot != T(0)) {
    q0 = q0 / norm_rot;
    qe = qe / norm_rot;
  }
  T real_dot_dual =
      (q0[0] * qe[0] + q0[1] * qe[1] + q0[2] * qe[2] + q0[3] * qe[3]);
  qe -= q0 * real_dot_dual;
}

// partial derivative of DQB w.r.t. lie algebra (lie_jacobi)
// to compute partial derivatives only w.r.t. quaternion itself,
// replace lie_jacobi by zero Matrix
RECLIB_HOST Eigen::Matrix<float, 8, 6> DQBJacobian(
    const std::vector<float>& weights,
    const std::vector<reclib::DualQuaternion>& quats,
    const unsigned int derivative_axis,
    const Eigen::Matrix<float, 8, 6>& lie_jacobi =
        Eigen::Matrix<float, 8, 6>::Zero(8, 6));

RECLIB_DEVICE Eigen::Matrix<float, 8, 6> DQBJacobian(
    const float* weights, const reclib::DualQuaternion* quats,
    const unsigned int size, const unsigned int derivative_axis,
    const Eigen::Matrix<float, 8, 6>& lie_jacobi =
        Eigen::Matrix<float, 8, 6>::Zero(8, 6));

RECLIB_HOST mat4 DQB(const std::vector<float>& weights,
                  const std::vector<mat4>& transforms);

RECLIB_DEVICE mat4 DQB(const float* weights, const mat4* transforms,
                    const size_t size);

RECLIB_DEVICE reclib::DualQuaternion DQB(const float* weights,
                                      const reclib::DualQuaternion* transforms,
                                      const size_t size);

}  // namespace reclib

#endif
