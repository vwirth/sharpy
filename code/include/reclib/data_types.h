#ifndef DATA_TYPES
#define DATA_TYPES

#if WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <reclib/platform.h>

#include <Eigen/Eigen>
#include <iomanip>
#include <ios>
#include <iostream>

#if HAS_OPENCV_MODULE

#include <opencv2/core.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

using CpuMat = cv::Mat;
using GpuMat = cv::cuda::GpuMat;

template <typename Type>
using CpuMat_ = cv::Mat_<Type>;

template <typename Type>
using GpuMat_ = cv::cudev::GpuMat_<Type>;

#endif

#if WITH_CUDA
template <typename Type>
using GpuVec = thrust::device_vector<Type>;

template <typename Type>
using CpuVec = thrust::host_vector<Type>;
#else
template <typename Type>
using CpuVec = std::vector<Type>;
#endif

// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/darglein/saiga
// ---------------------------------------------------------------------------

// All vector types are formed from this typedef.
// -> A vector is a Nx1 Eigen::Matrix.
template <typename Scalar, int Size>
using Vector = Eigen::Matrix<Scalar, Size, 1, Eigen::ColMajor>;

// All 2D fixed size CpuMatrices are formed from this typedef.
// They are all stored in column major order.
template <typename Scalar, int Rows, int Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Eigen::ColMajor>;

// ===== Double Precision (Capital Letter) ======
using Vec2 = Vector<double, 2>;
using Vec3 = Vector<double, 3>;
using Vec4 = Vector<double, 4>;
using Vec5 = Vector<double, 5>;
using Vec6 = Vector<double, 6>;
using Vec7 = Vector<double, 7>;
using Vec8 = Vector<double, 8>;
using Vec9 = Vector<double, 9>;

using Mat2 = Matrix<double, 2, 2>;
using Mat3 = Matrix<double, 3, 3>;
using Mat4 = Matrix<double, 4, 4>;

// ===== Single Precision  ======
using vec1 = Vector<float, 1>;
using vec2 = Vector<float, 2>;
using vec3 = Vector<float, 3>;
using vec4 = Vector<float, 4>;
using vec5 = Vector<float, 5>;
using vec6 = Vector<float, 6>;
using vec7 = Vector<float, 7>;
using vec8 = Vector<float, 8>;
using vec9 = Vector<float, 9>;

using mat2 = Matrix<float, 2, 2>;
using mat3 = Matrix<float, 3, 3>;
using mat4 = Matrix<float, 4, 4>;

// ===== Non-floating point types. Used for example in image processing  ======

using uvec2 = Vector<unsigned int, 2>;
using uvec3 = Vector<unsigned int, 3>;
using uvec4 = Vector<unsigned int, 4>;

using ivec2 = Vector<int, 2>;
using ivec3 = Vector<int, 3>;
using ivec4 = Vector<int, 4>;

using cvec2 = Vector<char, 2>;
using cvec3 = Vector<char, 3>;
using cvec4 = Vector<char, 4>;

using ucvec2 = Vector<unsigned char, 2>;
using ucvec3 = Vector<unsigned char, 3>;
using ucvec4 = Vector<unsigned char, 4>;

using svec2 = Vector<short, 2>;
using svec3 = Vector<short, 3>;
using svec4 = Vector<short, 4>;

using usvec2 = Vector<unsigned short, 2>;
using usvec3 = Vector<unsigned short, 3>;
using usvec4 = Vector<unsigned short, 4>;
// ---------------------------------------------------------------------------
// End of adapted code
// ---------------------------------------------------------------------------

using imat2 = Matrix<int, 2, 2>;
using imat3 = Matrix<int, 3, 3>;
using imat4 = Matrix<int, 4, 4>;

using umat2 = Matrix<unsigned int, 2, 2>;
using umat3 = Matrix<unsigned int, 3, 3>;
using umat4 = Matrix<unsigned int, 4, 4>;

using vec2da = Eigen::Matrix<float, 2, 1, Eigen::DontAlign>;
using vec3da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using vec4da = Eigen::Matrix<float, 4, 1, Eigen::DontAlign>;

using ivec2da = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using ivec3da = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;
using ivec4da = Eigen::Matrix<int, 4, 1, Eigen::DontAlign>;

using mat31da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using mat3da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

using quat = Eigen::Quaternionf;
using Quat = Eigen::Quaterniond;

#if HAD_OPENCV_MODULE
template <typename T>
struct PtrStep1d : public cv::cuda::DevPtr<T> {
  RECLIB_HD PtrStep1d(T* data_) : cv::cuda::DevPtr<T>(data_){};

  RECLIB_HD T* ptr(int x = 0) {
    return ((T*)(((cv::cuda::DevPtr<T>*)this)->data)) + x;
  }
  RECLIB_HD const T* ptr(int x = 0) const {
    return ((const T*)(((cv::cuda::DevPtr<T>*)this)->data)) + x;
  }

  RECLIB_HD T& operator()(int x) { return ptr(x)[0]; }
  RECLIB_HD const T& operator()(int x) const { return ptr(x)[0]; }
};

class _API GpuMat1d : public GpuMat {
  GpuMat1d(uint32_t size, int32_t type) : GpuMat(1, size, type){};
  size_t size() { return this->cols; }

  template <typename _Tp>
  operator PtrStep1d<_Tp>() const;
};

class _API CpuMat1d : public CpuMat {
  CpuMat1d(uint32_t size, int32_t type) : CpuMat(1, size, type){};
  size_t size() { return this->cols; }
};
#endif

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const std::vector<Scalar>& v) {
  os.setf(std::ios::fixed);
  os << std::setprecision(4);

  int width = 6;

  os << std::setw(width) << "[";
  unsigned int thresh = 10;
  bool is_big = v.size() > thresh;
  for (unsigned int i = 0; i < v.size(); i++) {
    os << v[i];
    if (i < v.size() - 1) {
      os << "," << std::setw(width);
    }
    if (is_big && i % thresh == 1 && i > thresh && i < v.size() - 1) {
      os << " ..";
      os << std::endl;
    }
  }
  os << "]";

  os.unsetf(std::ios::fixed);

  return os;
}

template <typename Scalar, int Size>
std::ostream& operator<<(std::ostream& os, const Vector<Scalar, Size>& v) {
  os.setf(std::ios::fixed);
  os << std::setprecision(4);

  int width = 6;

  os << std::setw(width) << "(";
  unsigned int thresh = 10;
  bool is_big = v.rows() > thresh || v.cols() > thresh;
  bool row_major = v.rows() > v.cols();
  for (unsigned int i = 0; i < v.rows(); i++) {
    for (unsigned int j = 0; j < v.cols(); j++) {
      os << v(i, j);
      if ((!row_major && j < v.cols() - 1) || (row_major && i < v.rows() - 1)) {
        os << "," << std::setw(width);
      }
      if (is_big && ((row_major && i % thresh == 1 && i > thresh) ||
                     (!row_major && j % thresh == 1 && j > thresh))) {
        os << " ..";
        os << std::endl;
      }
    }
  }
  os << ")";

  os.unsetf(std::ios::fixed);

  return os;
}

template <typename Scalar, int Rows, int Cols>
std::ostream& operator<<(std::ostream& os,
                         const Matrix<Scalar, Rows, Cols>& v) {
  unsigned int thresh = 10;
  bool is_big = v.rows() > thresh || v.cols() > thresh;
  bool row_major = v.rows() > v.cols();

  os.setf(std::ios::fixed);
  os << std::setprecision(4);
  int width = 6;

  if ((v.cols() == 1 && v.rows() > 1) || (v.rows() == 1 && v.cols() > 1)) {
    os << std::setw(width) << "[";
    for (unsigned int i = 0; i < v.rows(); i++) {
      for (unsigned int j = 0; j < v.cols(); j++) {
        os << v(i, j);
        if ((!row_major && j < v.cols() - 1) ||
            (row_major && i < v.rows() - 1)) {
          os << "," << std::setw(width);
        }
        if (is_big && ((row_major && i % thresh == 1 && i > thresh) ||
                       (!row_major && j % thresh == 1 && j > thresh))) {
          os << " ..";
          os << std::endl;
          os << std::setw(width);
        }
      }
    }
    os << "]";
  } else {
    os << std::endl;
    os << "[";
    for (unsigned int i = 0; i < v.rows(); i++) {
      os << std::setw(width) << "[";
      for (unsigned int j = 0; j < v.cols(); j++) {
        os << v(i, j);
        if (j < v.cols() - 1) {
          os << "," << std::setw(width);
        }
        if (j % thresh == 1 && j > thresh) {
          os << " ..";
          os << std::endl;
          os << std::setw(width);
        }
      }
      os << "]" << std::endl;
    }
    os << "]";
    os << std::endl;
  }

  os.unsetf(std::ios::fixed);

  return os;
}

#endif