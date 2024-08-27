#ifndef RECLIB_OPTIM_REGISTRATION_H
#define RECLIB_OPTIM_REGISTRATION_H

#include <sophus/se3.hpp>

#include "reclib/configuration.h"
#include "reclib/data_types.h"
#include "reclib/opengl/geometry.h"
#include "reclib/optim/correspondences.h"

#if HAS_DNN_MODULE
#include "reclib/dnn/dnn_utils.h"
#endif

namespace reclib {

namespace optim {

struct RegistrationConfiguration : public reclib::Configuration {
  static const std::vector<std::string> YAML_PREFIXES;  // = {"Registration"}
  static const unsigned int METHOD_DIRECT_POINT2POINT = 0;
  static const unsigned int METHOD_ITERATIVE_POINT2POINT = 1;
  static const unsigned int METHOD_ITERATIVE_POINT2PLANE = 2;
};

template <typename T>
Eigen::Vector<T, 3> MeanTranslation(const T* source_vertices,
                                    uint32_t source_size,
                                    const T* target_vertices,
                                    uint32_t target_size,
                                    const Eigen::Matrix<T, 4, 4>& source_trans =
                                        Eigen::Matrix<T, 4, 4>::Identity(),
                                    const Eigen::Matrix<T, 4, 4>& target_trans =
                                        Eigen::Matrix<T, 4, 4>::Identity());
template <typename T>
Eigen::Vector<T, 3> MeanTranslation(
    const std::vector<CorrespondenceBase<T, 3>>& corrs);

template <typename T>
Eigen::Vector<T, 3> MeanTranslation(
    const T* source_vertices, uint32_t source_size, const T* target_vertices,
    uint32_t target_size, const std::vector<unsigned int>& source_indices,
    const std::vector<unsigned int>& target_indices);

vec3 MeanTranslation(const reclib::opengl::GeometryBase& source,
                     const reclib::opengl::GeometryBase& target,
                     const mat4& source_trans = mat4::Identity(),
                     const mat4& target_trans = mat4::Identity());

template <typename T>
Eigen::Quaternion<T> orientationFromMixedMatrixUQ(
    const Eigen::Matrix<T, 3, 3>& M);

template <typename T>
Eigen::Quaternion<T> orientationFromMixedMatrixSVD(
    const Eigen::Matrix<T, 3, 3>& M);

/**
 * Analytical solution to the point cloud registration problem.
 * The function is solved using the polar decomposition.
 * See also "Orthonormal Procrustes Problem".
 *
 * If scale != nullptr a scaling between the point clouds is also computed
 *
 *
 */
template <typename T>
Sophus::SE3<T> pointToPointDirect(
    const std::vector<CorrespondenceBase<T, 3>>& corrs, T* scale = nullptr);

template <typename T>
Sophus::SE3<T> pointToPointDirect(
    const T* source_vertices, uint32_t source_size, const T* target_vertices,
    uint32_t target_size, const std::vector<unsigned int>& source_indices,
    const std::vector<unsigned int>& target_indices, T* scale = nullptr);

#if HAS_DNN_MODULE
torch::Tensor pointToPointDirect(torch::Tensor source, torch::Tensor target);
torch::Tensor MeanTranslation(torch::Tensor source, torch::Tensor target);
#endif

/**
 * The basic point to point registration algorithm which minimized the function
 * above. Each correspondence only needs the 'refPoint' and 'srcPoint'. The
 * function is minized iterativley using the Gauss-Newton algorithm.
 */
template <typename T>
Sophus::SE3<T> pointToPointIterative(
    const std::vector<CorrespondenceBase<T, 3>>& corrs,
    const Sophus::SE3<T>& guess = Sophus::SE3<T>(), int innerIterations = 5);

/**
 * Minimized the distance between the source point to the surface plane at the
 * reference point:
 *
 * argmin_T || (ref_i - T*src_i)*ref_normal_i ||^2
 *
 * Each correspondnce additional needs the 'refNormal' attribute.
 */
template <typename T>
Sophus::SE3<T> pointToPlane(const std::vector<CorrespondenceBase<T, 3>>& corrs,
                            const Sophus::SE3<T>& ref = Sophus::SE3<T>(),
                            const Sophus::SE3<T>& src = Sophus::SE3<T>(),
                            int innerIterations = 1);

}  // namespace optim
}  // namespace reclib

#endif