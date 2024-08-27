#ifndef EXP_COORDS_H
#define EXP_COORDS_H

#if WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#if HAS_OPENCV_MODULE
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>
#endif

#include <utility>

#include "reclib/data_types.h"
#include "reclib/math/math_ops.h"
#include "reclib/math/quaternion.h"
#include "reclib/utils/eigen_utils.h"

namespace reclib {

// representation for rotations
template <typename T>
class _API AxisAngleBase {
  template <typename U>
  friend class TwistBase;

 public:
  inline RECLIB_HD AxisAngleBase() : w_(T(0), T(0), T(0)){};
  inline RECLIB_HD AxisAngleBase(const Eigen::Vector<T, 3>& w) : w_(w){};
  inline RECLIB_HD AxisAngleBase(const Eigen::Map<Eigen::Vector<T, 3>>& w)
      : w_(w){};

  inline RECLIB_HD AxisAngleBase(const Eigen::Matrix<T, 4, 4>& mat)
      : w_(T(0), T(0), T(0)) {
    Eigen::Quaternion<T> quat(mat.template block<3, 3>(0, 0));
    *this = AxisAngleBase<T>(quat);

    //     std::cout << "trace: " << mat.trace() << std::endl;
    //     std::cout << "val: " << (mat.trace() - T(1.f)) / T(2.f) << std::endl;

    //     T theta = acos((mat.trace() - T(1.f)) / T(2.f));
    //     if (abs(theta - T(M_PI)) < T(1e-2f)) {
    //       w_[0] = sqrt(-(T(1.0) / T(4.0)) *
    //                    (mat(1, 1) - mat(0, 0) + mat(2, 2) - T(1.f)));
    //       w_[1] = sqrt(-(T(1.0) / T(4.0)) *
    //                    (-mat(1, 1) + mat(0, 0) + mat(2, 2) - T(1.f)));
    //       w_[2] = sqrt(-(T(1.0) / T(4.0)) *
    //                    (mat(1, 1) + mat(0, 0) - mat(2, 2) - T(1.f)));

    //       unsigned int nonzero_entries = abs(reclib::sign(w_[0])) +
    //                                      abs(reclib::sign(w_[1])) +
    //                                      abs(reclib::sign(w_[2]));

    //       if (nonzero_entries > 1) {
    //         if (nonzero_entries == 2) {
    //           if (w_[0] == T(0)) {
    //             w_[1] *= reclib::sign(mat(1, 2));
    //           } else if (w_[1] == T(0)) {
    //             w_[0] *= reclib::sign(mat(0, 2));
    //           } else if (w_[2] == T(0)) {
    //             w_[1] *= reclib::sign(mat(0, 1));
    //           }
    //         } else if (nonzero_entries == 3) {
    //           if (w_[0] > T(0)) {
    //             w_[1] *= reclib::sign(mat(0, 1) / w_[0]);
    //             w_[2] *= reclib::sign(mat(1, 2) * (w_[0] / mat(0, 1)));
    //             if (mat(0, 2) > T(0)) {
    //               w_[0] *= reclib::sign(w_[2]);
    //             } else {
    //               w_[0] *= -reclib::sign(w_[2]);
    //             }
    //           } else if (w_[1] > T(0)) {
    //             w_[0] *= reclib::sign(mat(0, 1) / w_[1]);
    //             w_[2] *= reclib::sign(mat(0, 2) * (w_[1] / mat(0, 1)));
    //             if (mat(1, 2) > T(0)) {
    //               w_[1] *= reclib::sign(w_[2]);
    //             } else {
    //               w_[1] *= -reclib::sign(w_[2]);
    //             }
    //           } else if (w_[2] > T(0)) {
    //             w_[0] *= reclib::sign(mat(0, 2) / w_[2]);
    //             w_[1] *= reclib::sign(mat(0, 1) * (w_[2] / mat(0, 2)));
    //             if (mat(1, 2) > T(0)) {
    //               w_[2] *= reclib::sign(w_[0]);
    //             } else {
    //               w_[2] *= -reclib::sign(w_[0]);
    //             }
    //           }
    //         }
    //       }
    //       w_ *= theta;
    //     } else {
    //       std::cout << "theta: " << theta << std::endl;
    //       Eigen::Matrix<T, 4, 4> log_r = Eigen::Matrix<T, 4, 4>::Zero(4, 4);
    //       if (theta != T(0)) {
    //         log_r = (theta / (T(2) * sin(theta))) * (mat - mat.transpose());
    //       }
    //       w_[0] = log_r(2, 1);
    //       w_[1] = log_r(0, 2);
    //       w_[2] = log_r(1, 0);
    //       std::cout << "W_: " << w_ << std::endl;
    //     }
  };

  inline RECLIB_HD T Angle() const {
    if (w_.cwiseAbs().sum() > T(0)) {
      return w_.norm();
    } else {
      return T(0);
    }
  }

  static inline RECLIB_HOST T Angle(T const* data) {
    Eigen::Map<const Eigen::Vector<T, 3>> w(data);
    if (w.norm() == T(0)) {
      return T(0);
    }
    return w.norm();
  }

  inline RECLIB_HD Eigen::Matrix<T, 3, 3> ToMatrix() const {
    Eigen::Matrix<T, 3, 3> CpuMat = Eigen::Matrix<T, 3, 3>::Identity(3, 3);
    T theta = Angle();
    CpuMat += sin(theta) * SkewMatrix() +
              (1 - cos(theta)) * SkewMatrix() * SkewMatrix();

    return CpuMat;
  }

  static inline RECLIB_HOST Eigen::Matrix<T, 3, 3> ToMatrix(T const* data) {
    Eigen::Map<const Eigen::Vector<T, 3>> w(data);
    Eigen::Matrix<T, 3, 3> CpuMat = Eigen::Matrix<T, 3, 3>::Identity(3, 3);
    T theta = Angle(data);

    // CpuMat += SkewMatrix(data);
    CpuMat += sin(theta) * SkewMatrix(data) +
              (T(1) - cos(theta)) * SkewMatrix(data) * SkewMatrix(data);
    return CpuMat;
  }

  // return stacked [(3*3) x 3] matrix
  // where each unknown has one [3 x 3] matrix in it
  inline RECLIB_HD Eigen::MatrixXf ToMatrixJacobi() const {
    Eigen::MatrixXf jacobi = Eigen::MatrixXf::Zero(9, 3);
    T theta = Angle();
    for (unsigned int i = 0; i < 3; i++) {
      jacobi.block<3, 3>(i * 3, 0) = PartialDerivRot(i);
    }

    return jacobi;
  }

  inline RECLIB_HD Eigen::Quaternion<T> ToQuaternion() const {
    T theta = Angle();
    Eigen::Vector<T, 3> w = Axis();
    Eigen::Vector<T, 4> q;
    int is_zero = theta == 0;
    q[0] = (1 - is_zero) * w[0] * sin(theta / 2.f);
    q[1] = (1 - is_zero) * w[1] * sin(theta / 2.f);
    q[2] = (1 - is_zero) * w[2] * sin(theta / 2.f);
    q[3] = cos(theta / 2.f);
    return Eigen::Quaternion<T>(q);
  }

  static inline void RECLIB_HOST ToQuaternion(T const* data, T* output) {
    T theta = Angle(data);
    Eigen::Vector<T, 3> w = Axis(data);
    Eigen::Map<Eigen::Vector<T, 4>> q(output);
    int is_zero = theta == T(0);
    if (!is_zero) {
      q[0] = w[0] * sin(theta / T(2.f));
      q[1] = w[1] * sin(theta / T(2.f));
      q[2] = w[2] * sin(theta / T(2.f));
      q[3] = cos(theta / T(2.f));
    } else {
      q[3] = cos(theta / T(2.f));
    }
  }

  inline RECLIB_HD Eigen::Matrix<float, 4, 3> ToQuaternionJacobi() const {
    Eigen::Matrix<float, 4, 3> J = Eigen::Matrix<float, 4, 3>::Zero(4, 3);

    T norm = Angle();
    Eigen::Vector<T, 3> w = w_;
    int is_zero = norm < 1e-8f;

    for (unsigned int axis = 0; axis < 3; axis++) {
      for (unsigned int deriv_axis = 0; deriv_axis < 3; deriv_axis++) {
        T dN = PartialDerivAngle(deriv_axis);
        J(axis, deriv_axis) =
            (1 - is_zero) *
            (((((axis) == deriv_axis) * norm - w_[axis] * dN) /
              MAX(pow(norm, 2), 1e-8f)) *
                 sin(norm / 2.f) +
             (w[axis] / MAX(norm, 1e-8f)) * cos(norm / 2.f) * ((2 * dN) / 4.f));
      }

      J(3, 0) = -sin(norm / 2.f) * ((2 * PartialDerivAngle(0)) / 4.f);
      J(3, 1) = -sin(norm / 2.f) * ((2 * PartialDerivAngle(1)) / 4.f);
      J(3, 2) = -sin(norm / 2.f) * ((2 * PartialDerivAngle(2)) / 4.f);
    }
    return J;
  }

  inline RECLIB_HD Eigen::Vector<T, 3> Axis() const {
    T theta = T(0);
    if (w_.cwiseAbs().sum() > T(0)) {
      theta = w_.norm();
    }
    return (T(1) - T(theta == T(0))) * (w_ / T(MAX(theta, T(1e-8f))));
  }

  static inline RECLIB_HOST Eigen::Vector<T, 3> Axis(T const* data) {
    Eigen::Map<const Eigen::Vector<T, 3>> w(data);
    bool is_zero = w.norm() < T(1e-8f);
    return T(1 - is_zero) * (w / MAX(w.norm(), T(1e-8f)));
    // return w / w.norm();
  }

  inline RECLIB_HD Eigen::Vector<T, 3> Data() const { return w_; }

  inline RECLIB_HD Eigen::Matrix<T, 3, 3> SkewMatrix() const {
    Eigen::Matrix<T, 3, 3> m = Eigen::Matrix<T, 3, 3>::Zero(3, 3);
    T norm = Angle();
    Eigen::Vector<T, 3> axis = Axis();

    m(0, 0) = T(0);
    m(1, 0) = axis[2];
    m(2, 0) = -axis[1];

    m(0, 1) = -axis[2];
    m(1, 1) = T(0);
    m(2, 1) = axis[0];

    m(0, 2) = axis[1];
    m(1, 2) = -axis[0];
    m(2, 2) = T(0);
    return m;
  }

  static inline RECLIB_HOST Eigen::Matrix<T, 3, 3> SkewMatrix(T const* data) {
    Eigen::Map<const Eigen::Vector<T, 3>> w(data);

    Eigen::Matrix<T, 3, 3> m = Eigen::Matrix<T, 3, 3>::Zero(3, 3);
    Eigen::Vector<T, 3> axis = Axis(data);

    m(0, 0) = T(0);
    m(1, 0) = axis[2];
    m(2, 0) = -axis[1];

    m(0, 1) = -axis[2];
    m(1, 1) = T(0);
    m(2, 1) = axis[0];

    m(0, 2) = axis[1];
    m(1, 2) = -axis[0];
    m(2, 2) = T(0);
    return m;
  }

  friend std::ostream& operator<<(std::ostream& os, const AxisAngleBase& q) {
    os << "Axis: (" << q.Axis()[0] << "," << q.Axis()[1] << "," << q.Axis()[2]
       << ") Angle: " << q.Angle();
    return os;
  }

  RECLIB_HD T operator[](const int i) const { return w_[i]; }

  RECLIB_HD friend AxisAngleBase operator+(const AxisAngleBase& q1,
                                           const AxisAngleBase& q2) {
    Eigen::Vector<T, 3> w = q1.w_ + q2.w_;
    return AxisAngleBase(w);
  }

  RECLIB_HD friend AxisAngleBase& operator+=(AxisAngleBase& q1,
                                             const AxisAngleBase& q2) {
    q1.w_ += q2.w_;
    return q1;
  }

  RECLIB_HD friend AxisAngleBase operator-(const AxisAngleBase& q1,
                                           const AxisAngleBase& q2) {
    Eigen::Vector<T, 3> w = q1.w_ - q2.w_;
    return AxisAngleBase(w);
  }

  RECLIB_HD friend AxisAngleBase& operator-=(AxisAngleBase& q1,
                                             const AxisAngleBase& q2) {
    q1.w_ -= q2.w_;
    return q1;
  }

  // public:
  Eigen::Vector<T, 3> w_;
  // Eigen::Vector<T, 3> data_w_;
  // Eigen::Map<Eigen::Vector<T, 3>> w_;

  // partial derivative of cos(||w||) by an axis of w itself
  inline RECLIB_HD T PartialDerivCos(unsigned int derivative_axis) const {
    T norm = Angle();
    return -sin(norm) * PartialDerivAngle(derivative_axis);
  }

  // partial derivative of sin(||w||) by an axis of w itself
  inline RECLIB_HD T PartialDerivSin(unsigned int derivative_axis) const {
    T norm = Angle();
    return cos(norm) * PartialDerivAngle(derivative_axis);
  }

  // partial derivative of normalized w, i.a. w / ||w|| by an axis of w itself
  inline RECLIB_HD Eigen::Vector<T, 3> PartialDerivW(
      unsigned int derivative_axis) const {
    Eigen::Vector<T, 3> dw;
    for (unsigned int i = 0; i < 3; i++) {
      T norm = Angle();
      int is_zero = norm < T(1e-8f);
      if (i == derivative_axis) {
        dw[i] = (1 - is_zero) *
                (norm - w_[i] * (1.f / (2.f * MAX(norm, 1e-8f))) * 2.f *
                            w_[derivative_axis]) /
                MAX(pow(norm, 2), 1e-8f);
      } else {
        dw[i] = (1 - is_zero) *
                (-w_[i] * (1.f / (2.f * MAX(norm, 1e-8f))) * 2.f *
                 w_[derivative_axis]) /
                MAX(pow(norm, 2), 1e-8f);
      }
    }
    return dw;
  }

  // partial derivative of normalized skew symmetric Matrix w, by an axis of
  // w itself
  inline RECLIB_HD Eigen::Matrix<T, 3, 3> PartialDerivSkewW(
      unsigned int derivative_axis) const {
    Eigen::Matrix<T, 3, 3> dSkew;
    Eigen::Vector<T, 3> dw = PartialDerivW(derivative_axis);
    dSkew << 0, -dw[2], dw[1], dw[2], 0, -dw[0], -dw[1], dw[0], 0;

    return dSkew;
  }

  // partial derivative of normalized skew symmetric Matrix product (skew *
  // skew), by an axis of w itself
  // notice that it's really skew*skew and not skew^T * skew (otherwise we would
  // have a reclib::sign twist)
  inline RECLIB_HD Eigen::Matrix<T, 3, 3> PartialDerivSkew2W(
      unsigned int derivative_axis) const {
    return PartialDerivSkewW(derivative_axis) * SkewMatrix() +
           SkewMatrix() * PartialDerivSkewW(derivative_axis);
  }

  // partial derivative of outer product of w, i.a. wwt,
  // by an axis of w itself
  inline RECLIB_HD Eigen::Matrix<T, 3, 3> PartialDerivWwt(
      unsigned int derivative_axis) const {
    Eigen::Matrix<T, 3, 3> dWwt;
    Eigen::Vector<T, 3> w = Axis();
    Eigen::Vector<T, 3> dw = PartialDerivW(derivative_axis);

    dWwt << 2.f * w[0] * dw[0], dw[0] * w[1] + dw[1] * w[0],
        dw[0] * w[2] + dw[2] * w[0], dw[0] * w[1] + dw[1] * w[0],
        2.f * w[1] * dw[1], w[1] * dw[2] + w[2] * dw[1],
        dw[0] * w[2] + dw[2] * w[0], w[1] * dw[2] + w[2] * dw[1],
        2.f * w[2] * dw[2];
    return dWwt;
  }

  // partial derivative of ||w|| by an axis of w itself
  inline RECLIB_HD T PartialDerivAngle(unsigned int derivative_axis) const {
    T norm = Angle();
    int is_zero = norm < 1e-8f;
    return (1 - is_zero) * (1.f / (2.f * MAX(norm, 1e-8f))) * 2 *
           w_[derivative_axis];
  }

  inline RECLIB_HD Eigen::Matrix<T, 3, 3> PartialDerivRot(
      unsigned int derivative_axis) const {
    T theta = Angle();
    Eigen::Matrix<T, 3, 3> res =
        PartialDerivSin(derivative_axis) * SkewMatrix() +
        PartialDerivSkewW(derivative_axis) * sin(theta) +
        -PartialDerivCos(derivative_axis) * SkewMatrix() * SkewMatrix() +
        (1 - cos(theta)) * PartialDerivSkew2W(derivative_axis);

    // Eigen::Matrix<T, 3, 3> res =
    //     PartialDerivSin(derivative_axis) * SkewMatrix() +
    //     PartialDerivSkewW(derivative_axis) * sin(theta);

    return res;
  }
};
using AxisAngle = AxisAngleBase<float>;

// representation for affine transformations

template <typename T>
class _API TwistBase {
 public:
  inline RECLIB_HD TwistBase(bool pseudo = false) : v_(0, 0, 0){};
  inline RECLIB_HD TwistBase(const Eigen::Vector<T, 3> w,
                             const Eigen::Vector<T, 3> v, bool pseudo = false)
      : w_(w), v_(v), pseudo_(pseudo){};
  inline RECLIB_HD TwistBase(const AxisAngleBase<T> w,
                             const Eigen::Vector<T, 3> v, bool pseudo = false)
      : w_(w), v_(v), pseudo_(pseudo){};

  // TODO: Matrix inverse is no RECLIB_DEVICE function?
  RECLIB_HOST TwistBase(const Eigen::Matrix<T, 4, 4>& CpuMat,
                        bool pseudo = false)
      : w_(CpuMat), v_(0, 0, 0), pseudo_(pseudo) {
    T theta = w_.Angle();

    Eigen::Vector<T, 3> t;
    t << CpuMat(0, 3), CpuMat(1, 3), CpuMat(2, 3);

    if (IsPureTrans()) {
      // no rotational part, just pure translation
      v_ = t;
    } else {
      // contains rotation
      Eigen::Matrix<T, 3, 3> G = Exp2transInverse();
      if (G.squaredNorm() >= 1e-8f) {
        v_ = G.inverse() * t;
      }
    }
  };

  RECLIB_HD TwistBase(const DualQuaternion& q, bool pseudo = false)
      : w_(q.real()), v_(0, 0, 0), pseudo_(pseudo) {
    T theta = w_.Angle();

    // compute translation vector t
    Eigen::Vector<T, 3> t = q.translation();
    if (IsPureTrans()) {
      // pure translation
      v_ = t;
    } else {
      Eigen::Matrix<T, 3, 3> G = Exp2transInverse();
      if (G.squaredNorm() >= 1e-8f) {
        v_ = G.inverse() * t;
      }
    }
  };

  inline RECLIB_HD Eigen::Matrix<T, 4, 4> ToMatrix() const {
    Eigen::Matrix<T, 4, 4> CpuMat = Eigen::Matrix<T, 4, 4>::Identity(4, 4);
    T theta = w_.Angle();
    // pure translation
    int pure_trans = IsPureTrans();

    Eigen::Matrix<T, 3, 3> rot = w_.ToMatrix();
    for (unsigned int row = 0; row < 3; row++) {
      for (unsigned int col = 0; col < 3; col++) {
        CpuMat(row, col) = rot(row, col);
      }
    }
    Eigen::Vector<T, 3> t =
        (pure_trans)*v_ +
        (1 - pure_trans) *
            ((Eigen::Matrix<T, 3, 3>::Identity(3, 3) - w_.ToMatrix()) *
                 w_.SkewMatrix() * v_ +
             w_.Axis() * w_.Axis().transpose() * theta * v_);
    for (unsigned int row = 0; row < 3; row++) {
      CpuMat(row, 3) = t(row);
    }

    return CpuMat;
  }

  static inline RECLIB_HOST Eigen::Matrix<T, 4, 4> ToMatrix(
      T const* data, bool pseudo = false) {
    Eigen::Map<const Eigen::Vector<T, 3>> w(data);
    Eigen::Map<const Eigen::Vector<T, 3>> v(data + 3);

    Eigen::Matrix<T, 4, 4> CpuMat = Eigen::Matrix<T, 4, 4>::Identity(4, 4);
    T theta = AxisAngleBase<T>::Angle(data);
    // pure translation
    int pure_trans = (int)(pseudo && w.squaredNorm() == T(0));

    Eigen::Matrix<T, 3, 3> rot = AxisAngleBase<T>::ToMatrix(data);

    Eigen::Vector<T, 3> t;
    if (pure_trans) {
      t = v;
    } else {
      for (unsigned int row = 0; row < 3; row++) {
        for (unsigned int col = 0; col < 3; col++) {
          CpuMat(row, col) = rot(row, col);
        }
      }

      Eigen::Matrix<T, 3, 3> IMinusR =
          (Eigen::Matrix<T, 3, 3>::Identity(3, 3) - rot);
      Eigen::Matrix<T, 3, 3> wwt = AxisAngleBase<T>::Axis(data) *
                                   AxisAngleBase<T>::Axis(data).transpose();
      t = (IMinusR * AxisAngleBase<T>::SkewMatrix(data) * v + wwt * theta * v);
    }
    for (unsigned int row = 0; row < 3; row++) {
      CpuMat(row, 3) = t(row);
    }

    return CpuMat;
  }

  // return stacked [(6*3) x 4] matrix
  // where each unknown has one [3 x 4] matrix in it
  inline RECLIB_HD Eigen::MatrixXf ToMatrixJacobi() const {
    Eigen::MatrixXf jacobi = Eigen::MatrixXf::Zero(18, 4);
    Eigen::MatrixXf jacobi_rotation = w_.ToMatrixJacobi();
    jacobi.block<9, 3>(0, 0) = jacobi_rotation;

    T theta = w_.Angle();
    // pure translation
    int pure_trans = IsPureTrans();
    for (unsigned int i = 0; i < 6; i++) {
      jacobi.block<3, 1>(i * 3, 3) = PartialDerivT(i);
    }

    return jacobi;
  }

  inline RECLIB_HD Eigen::Vector<T, 3> TranslationVector() const {
    T theta = w_.Angle();
    return (Eigen::Matrix<T, 3, 3>::Identity(3, 3) - w_.ToMatrix()) *
               w_.SkewMatrix() * v_ +
           w_.Axis() * w_.Axis().transpose() * theta * v_;
  }

  static inline RECLIB_HOST Eigen::Vector<T, 3> TranslationVector(
      T const* data) {
    Eigen::Map<const Eigen::Vector<T, 3>> v(data + 3);
    T theta = AxisAngleBase<T>::Angle(data);
    return (Eigen::Matrix<T, 3, 3>::Identity(3, 3) -
            AxisAngleBase<T>::ToMatrix(data)) *
               AxisAngleBase<T>::SkewMatrix(data) * v +
           AxisAngleBase<T>::Axis(data) *
               AxisAngleBase<T>::Axis(data).transpose() * theta * v;
  }

  inline RECLIB_HD DualQuaternionBase<T> ToDualQuaternion() const {
    T theta = w_.Angle();
    Eigen::Quaternion<T> p = w_.ToQuaternion();

    Eigen::Vector<T, 3> t;
    // pure translation
    int pure_trans = IsPureTrans();

    t = (pure_trans)*v_ + (1 - pure_trans) * TranslationVector();
    Eigen::Quaternion<T> t_as_quat(0, t[0], t[1], t[2]);
    Eigen::Quaternion<T> q = (1.f / 2.f) * t_as_quat * p;
    return DualQuaternionBase<T>(p, q);
  }

  static inline RECLIB_HOST void ToDualQuaternion(T const* data, T* output,
                                                  bool pseudo = false) {
    AxisAngleBase<T>::ToQuaternion(data, output);

    Eigen::Map<const Eigen::Vector<T, 3>> w(data);
    Eigen::Map<const Eigen::Vector<T, 3>> v(data + 3);
    Eigen::Map<const Eigen::Vector<T, 4>> p(output);

    Eigen::Vector<T, 3> t;
    // pure translation
    int pure_trans = pseudo && w.squaredNorm() == T(0);
    if (pure_trans) {
      output[4] =
          T(1.f / 2.f) * (v.x() * p.w() + v.y() * p.z() - v.z() * p.y());
      output[5] =
          T(1.f / 2.f) * (-v.x() * p.z() + v.y() * p.w() + v.z() * p.x());
      output[6] =
          T(1.f / 2.f) * (v.x() * p.y() - v.y() * p.x() + v.z() * p.w());
      output[7] =
          T(1.f / 2.f) * (-v.x() * p.x() - v.y() * p.y() - v.z() * p.z());
    } else {
      t = TranslationVector(data);
      output[4] =
          T(1.f / 2.f) * (t.x() * p.w() + t.y() * p.z() - t.z() * p.y());
      output[5] =
          T(1.f / 2.f) * (-t.x() * p.z() + t.y() * p.w() + t.z() * p.x());
      output[6] =
          T(1.f / 2.f) * (t.x() * p.y() - t.y() * p.x() + t.z() * p.w());
      output[7] =
          T(1.f / 2.f) * (-t.x() * p.x() - t.y() * p.y() - t.z() * p.z());
    }
  }

  inline RECLIB_HD Eigen::Matrix<float, 8, 6> ToDualQuaternionJacobi() const {
    Eigen::Matrix<float, 8, 6> J = Eigen::Matrix<float, 8, 6>::Zero(8, 6);
    DualQuaternion p = ToDualQuaternion();
    T norm = w_.Angle();

    J.block<4, 3>(0, 0) = w_.ToQuaternionJacobi();
    // pure translation
    int pure_trans = IsPureTrans();

    Eigen::Vector<T, 3> t;
    t = (pure_trans)*v_ + (1 - pure_trans) * TranslationVector();

    for (unsigned int axis = 4; axis < 8; axis++) {
      Eigen::Vector3i ind(3, 2, 1);
      if (axis == 5) {
        ind = Eigen::Vector3i(2, 3, 0);
      } else if (axis == 6) {
        ind = Eigen::Vector3i(1, 0, 3);
      } else if (axis == 7) {
        ind = Eigen::Vector3i(0, 1, 2);
      }
      for (unsigned int deriv_axis = 0; deriv_axis < 6; deriv_axis++) {
        int sign_x = 1 - (axis % 2) * 2;
        int sign_y = 1 - (axis > 5) * 2;
        int sign_z = 1 - (axis == 4 || axis == 7) * 2;

        Eigen::Vector<T, 3> dt = PartialDerivT(deriv_axis);
        // if(axis == 4)
        //   std::cout << "axis: " << deriv_axis << " dt: " << dt <<
        //   std::endl;
        J(axis, deriv_axis) =
            (1.f / 2.f) *
            ((sign_x) * (dt[0] * p[ind[0]] + t[0] * J(ind[0], deriv_axis)) +
             (sign_y) * (dt[1] * p[ind[1]] + t[1] * J(ind[1], deriv_axis)) +
             (sign_z) * (dt[2] * p[ind[2]] + t[2] * J(ind[2], deriv_axis)));
      }
    }

    return J;
  }

  inline RECLIB_HD Eigen::Vector<float, 6> Data() const {
    Eigen::Vector<float, 6> d;
    d.block<3, 1>(0, 0) = w_.Data();
    d.block<3, 1>(3, 0) = v_;
    return d;
  }

  inline RECLIB_HD AxisAngleBase<T> Rot() const { return w_; }

  inline RECLIB_HD Eigen::Vector<T, 3> Trans() const { return v_; }

  friend std::ostream& operator<<(std::ostream& os, const TwistBase<T>& q) {
    os << q.w_ << " Nu: (" << q.v_[0] << "," << q.v_[1] << "," << q.v_[2]
       << ")";
    return os;
  }

  RECLIB_HD T operator[](const int i) const {
    if (i <= 2) {
      return w_[i];
    } else {
      return v_[i - 3];
    }
  }

  RECLIB_HD friend TwistBase<T> operator+(const TwistBase<T>& q1,
                                          const TwistBase<T>& q2) {
    AxisAngle w = q1.w_ + q2.w_;
    Eigen::Vector<T, 3> v = q1.v_ + q2.v_;
    return TwistBase<T>(w.Data(), v);
  }

  RECLIB_HD friend TwistBase<T>& operator+=(TwistBase<T>& q1,
                                            const TwistBase<T>& q2) {
    q1.w_ += q2.w_;
    q1.v_ += q2.v_;
    return q1;
  }

  RECLIB_HD friend TwistBase<T> operator-(const TwistBase<T>& q1,
                                          const TwistBase<T>& q2) {
    AxisAngle w = q1.w_ - q2.w_;
    Eigen::Vector<T, 3> v = q1.v_ - q2.v_;
    return TwistBase<T>(w.Data(), v);
  }

  RECLIB_HD friend TwistBase<T>& operator-=(TwistBase<T>& q1,
                                            const TwistBase<T>& q2) {
    q1.w_ -= q2.w_;
    q1.v_ -= q2.v_;
    return q1;
  }

 private:
  AxisAngleBase<T> w_;
  Eigen::Vector<T, 3> v_;
  bool pseudo_;

  RECLIB_HD bool IsPureTrans() const {
    return pseudo_ && (w_.Data().squaredNorm() == 0);
  }

  Eigen::Matrix<T, 3, 3> Exp2transInverse() const {
    T theta = w_.Angle();
    Eigen::Matrix<T, 3, 3> G =
        (Eigen::Matrix<T, 3, 3>::Identity(3, 3) - w_.ToMatrix()) *
        w_.SkewMatrix();
    G += w_.Axis() * w_.Axis().transpose() * theta;
    return G;
  }

  // partial derivative of translation vector t, by an axis of w or v
  inline RECLIB_HD Eigen::Vector<T, 3> PartialDerivT(
      unsigned int derivative_axis) const {
    T norm = w_.Angle();
    int pure_trans = IsPureTrans();
    if (derivative_axis <= 2) {
      Eigen::Matrix<T, 3, 3> dR = w_.PartialDerivRot(derivative_axis);
      Eigen::Vector<T, 3> res(0, 0, 0);
      res = (1 - pure_trans) *
            (-dR * w_.SkewMatrix() * v_ +
             (Eigen::Matrix<T, 3, 3>::Identity() - w_.ToMatrix()) *
                 w_.PartialDerivSkewW(derivative_axis) * v_ +
             w_.PartialDerivWwt(derivative_axis) * norm * v_ +
             w_.Axis() * w_.Axis().transpose() *
                 w_.PartialDerivAngle(derivative_axis) * v_);
      return res;

    } else {
      Eigen::Vector<T, 3> unit_vec(0, 0, 0);
      unit_vec[derivative_axis - 3] = 1;
      return (pure_trans)*unit_vec +
             (1 - pure_trans) *
                 ((-sin(norm) * w_.SkewMatrix() -
                   (1 - cos(norm)) * w_.SkewMatrix() * w_.SkewMatrix()) *
                      w_.SkewMatrix() * unit_vec +
                  norm * w_.Axis() * w_.Axis().transpose() * unit_vec);
    }
  }
};
using Twist = TwistBase<float>;

}  // namespace reclib

#endif
