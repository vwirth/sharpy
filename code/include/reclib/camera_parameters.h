#ifndef CAMERA_PARAMETERS_H
#define CAMERA_PARAMETERS_H
#include <cmath>
#include <cstdint>
#include <iostream>

#if HAS_OPENCV_MODULE
#include <opencv2/core/mat.hpp>
#include <opencv2/core/softfloat.hpp>
#endif

#include "reclib/data_types.h"
#include "reclib/math/math_ops.h"
#include "reclib/platform.h"

namespace reclib {
/**
 *
 * \brief The camera intrinsics
 *
 * This structure stores the intrinsic camera parameters.
 *
 * Consists of:
 * 1) Image width and height,
 * 2) focal length in x and y direction and
 * 3) The principal point in x and y direction
 *
 */
struct IntrinsicParameters {
  int image_width_, image_height_{0};
  float focal_x_, focal_y_{0};
  float principal_x_, principal_y_{0};

  IntrinsicParameters() {}

  IntrinsicParameters(int image_width, int image_height, float focal_x,
                      float focal_y, float principal_x, float principal_y)
      : image_width_(image_width),
        image_height_(image_height),
        focal_x_(focal_x),
        focal_y_((focal_y)),
        principal_x_(principal_x),
        principal_y_(principal_y) {}

  inline ivec2da Size() const { return ivec2da(image_height_, image_width_); }

  inline vec2da Focal() const { return vec2da(focal_x_, focal_y_); }

  inline vec2da Principal() const { return vec2da(principal_x_, principal_y_); }

  Eigen::Matrix4f Matrix() const {
    Eigen::Matrix<float, 4, 4> a;
    a << focal_x_, 0, principal_x_, 0, 0, focal_y_, principal_y_, 0, 0, 0, 1, 0,
        0, 0, 0, 1;
    return a;
  }

#if HAS_OPENCV_MODULE
  cv::Mat MatrixCV() const {
    cv::Mat a = (cv::Mat_<float>(4, 4) << focal_x_, 0, principal_x_, 0, 0,
                 focal_y_, principal_y_, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return a;
  }
#endif

  /**
   * Returns camera parameters for a specified pyramid level; each level
   * corresponds to a scaling of pow(.5, level)
   * @param level The pyramid level to get the parameters for with 0 being the
   * non-scaled version, higher levels correspond to smaller spatial size
   * @return A IntrinsicParameters structure containing the scaled values
   */
  inline IntrinsicParameters Level(const uint32_t level) const {
    if (level == 0) {
      {
        return *this;
      }
    }

    const float scale_factor = powf(0.5F, static_cast<float>(level));
    return IntrinsicParameters{image_width_ >> level,
                               image_height_ >> level,
                               focal_x_ * scale_factor,
                               focal_y_ * scale_factor,
                               (principal_x_ + 0.5F) * scale_factor - 0.5F,
                               (principal_y_ + 0.5F) * scale_factor - 0.5F};
  }

  inline IntrinsicParameters Scale(const float scale_factor) const {
    if (scale_factor == 1.f) {
      {
        return *this;
      }
    }

    return IntrinsicParameters{(int)std::floor(image_width_ * scale_factor),
                               (int)std::floor(image_height_ * scale_factor),
                               focal_x_ * scale_factor,
                               focal_y_ * scale_factor,
                               (principal_x_ + 0.5F) * scale_factor - 0.5F,
                               (principal_y_ + 0.5F) * scale_factor - 0.5F};
  }

  friend inline std::ostream& operator<<(std::ostream& os,
                                         const IntrinsicParameters& obj) {
    os << "[IntrinsicParameters]" << std::endl;
    os << "image_width = " << obj.image_width_ << std::endl;
    os << "image_height = " << obj.image_height_ << std::endl;
    os << "focal_x = " << obj.focal_x_ << std::endl;
    os << "focal_y = " << obj.focal_y_ << std::endl;
    os << "principal_x = " << obj.principal_x_ << std::endl;
    os << "principal_y = " << obj.principal_y_ << std::endl;
    return os;
  }
};

struct ExtendedIntrinsicParameters : public IntrinsicParameters {
  float fovx_, fovy_;
  std::vector<float> k_;  // radial distortion coefficients
  std::vector<float> p_;  // tangential distortion coeffieicnets

  friend inline std::ostream& operator<<(
      std::ostream& os, const ExtendedIntrinsicParameters& obj) {
    os << "[ExtendedIntrinsicParameters]" << std::endl;
    os << "image_width = " << obj.image_width_ << std::endl;
    os << "image_height = " << obj.image_height_ << std::endl;
    os << "focal_x = " << obj.focal_x_ << std::endl;
    os << "focal_y = " << obj.focal_y_ << std::endl;
    os << "principal_x = " << obj.principal_x_ << std::endl;
    os << "principal_y = " << obj.principal_y_ << std::endl;
    os << "fovx_ = " << obj.fovx_ << std::endl;
    os << "fovy_ = " << obj.fovy_ << std::endl;
    os << "k_ = " << obj.k_ << std::endl;
    os << "p_ = " << obj.p_ << std::endl;
    return os;
  }
};

/**
 *
 * \brief The camera extrinsics
 *
 * This structure stores the extrinsic camera parameters.
 *
 * Consists of:
 * 1) Camera right vector
 * 2) Camera up vector
 * 3) Camera direction vector (equivalent to target - eye, or vice versa,
 * depending on sign)
 * 4) Camera position
 *
 */
class _API ExtrinsicParameters {
 public:
  vec3 right_{1.f, 0.f, 0.f};
  vec3 up_{0.f, 1.f, 0.f};
  vec3 dir_{0.f, 0.f, 1.f};
  vec3 eye_{0.f, 0.f, 0.f};  // aka camera position

  ExtrinsicParameters() = default;

  void forward(float by);
  void backward(float by);
  void leftward(float by);
  void rightward(float by);
  void upward(float by);
  void downward(float by);

  void yaw(float angle);
  void pitch(float angle);
  void roll(float angle);
  void rotate(const Eigen::Vector3f& axis, float angle);

  static ExtrinsicParameters from_matrix(const mat4& mat) {
    ExtrinsicParameters ext;
    ext.right_ = mat.col(0).head<3>();
    ext.up_ = mat.col(1).head<3>();
    ext.dir_ = mat.col(2).head<3>();
    ext.eye_ = mat.col(3).head<3>();
    return ext;
  }

  Eigen::Matrix4f Matrix() const {
    Eigen::Matrix<float, 4, 4> a;
    a << right_.x(), up_.x(), dir_.x(), eye_.x(), right_.y(), up_.y(), dir_.y(),
        eye_.y(), right_.z(), up_.z(), dir_.z(), eye_.z(), 0.F, 0.F, 0.F, 1.F;
    return a;
  }

#if HAS_OPENCV_MODULE
  cv::Mat MatrixCV() const {
    cv::Mat a = (cv::Mat_<float>(4, 4) << right_.x(), up_.x(), dir_.x(),
                 eye_.x(), right_.y(), up_.y(), dir_.y(), eye_.y(), right_.z(),
                 up_.z(), dir_.z(), eye_.z(), 0.F, 0.F, 0.F, 1.F);
    return a;
  }
#endif

  friend std::ostream& operator<<(std::ostream& os,
                                  const ExtrinsicParameters& obj) {
    os << "[ExtrinsicParameters]" << std::endl;
    os << "right = [" << obj.right_[0] << ", " << obj.right_[1] << ", "
       << obj.right_[2] << "]" << std::endl;
    os << "up = [" << obj.up_[0] << ", " << obj.up_[1] << ", " << obj.up_[2]
       << "]" << std::endl;
    os << "dir = [" << obj.dir_[0] << ", " << obj.dir_[1] << ", " << obj.dir_[2]
       << "]" << std::endl;
    os << "eye = [" << obj.eye_[0] << ", " << obj.eye_[1] << ", " << obj.eye_[2]
       << "]" << std::endl;

    return os;
  }
};

inline mat4 vision2graphics(const mat4& intrinsic, float near, float far,
                            unsigned int width, unsigned int height) {
  double fx = intrinsic(0, 0);
  double fy = intrinsic(1, 1);
  double cx = intrinsic(0, 2);
  double cy = intrinsic(1, 2);

  mat4 proj;
  // proj << fx / cx, 0, 0, 0, 0, fy / cy, 0, 0, 0, 0,
  //     -(far + near) / (far - near), -2 * far * near / (far - near), 0, 0, -1,
  //     0;

  proj << 2 * fx / (float)width, 0, 1.f - 2.f * cx / (float)width, 0, 0,
      2.f * fy / (float)height, 2.f * cy / (float)height - 1.f, 0, 0, 0,
      -(far + near) / (far - near), -2 * far * near / (far - near), 0, 0, -1, 0;
  return proj;
}

}  // namespace reclib

#endif
