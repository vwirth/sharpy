#include <ATen/ops/matmul.h>
#include <reclib/math/eigen_util_funcs.h>
#include <reclib/optim/registration.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>

#include "reclib/dnn/dnn_utils.h"

namespace reclib {
namespace optim {

// ------------------------------------------------------------
// ------------------------------------------------------------
// RegistrationConfiguration
// ------------------------------------------------------------
// ------------------------------------------------------------

const std::vector<std::string>
    reclib::optim::RegistrationConfiguration::YAML_PREFIXES = {"Registration"};

// ------------------------------------------------------------
// ------------------------------------------------------------

template <typename T>
Eigen::Vector<T, 3> MeanTranslation(
    const std::vector<CorrespondenceBase<T, 3>>& corrs) {
  Eigen::Vector<T, 3> meanTarget = Eigen::Vector<T, 3>::Zero();
  Eigen::Vector<T, 3> meanSrc = Eigen::Vector<T, 3>::Zero();
  for (unsigned int i = 0; i < corrs.size(); i++) {
    meanTarget += corrs[0].refPoint;
    meanSrc += corrs[0].srcPoint;
  }
  meanTarget /= corrs.size();
  meanSrc /= corrs.size();
  return (meanTarget - meanSrc).template head<3>();
}

template <typename T>
Eigen::Vector<T, 3> MeanTranslation(
    const T* source_vertices, uint32_t source_size, const T* target_vertices,
    uint32_t target_size, const std::vector<unsigned int>& source_indices,
    const std::vector<unsigned int>& target_indices) {
  _RECLIB_ASSERT_EQ(source_indices.size(), target_indices.size());

  Eigen::Vector<T, 3> meanTarget = Eigen::Vector<T, 3>::Zero();
  Eigen::Vector<T, 3> meanSrc = Eigen::Vector<T, 3>::Zero();
  for (unsigned int i = 0; i < source_indices.size(); i++) {
    vec3 s(&(source_vertices[source_indices[i] * 3]));
    vec3 t(&(target_vertices[target_indices[i] * 3]));

    meanTarget += t;
    meanSrc += s;
  }
  meanTarget /= source_indices.size();
  meanSrc /= source_indices.size();
  return (meanTarget - meanSrc).template head<3>();
}

template <typename T>
Eigen::Vector<T, 3> MeanTranslation(
    const T* source_vertices, uint32_t source_size, const T* target_vertices,
    uint32_t target_size, const Eigen::Matrix<T, 4, 4>& source_trans,
    const Eigen::Matrix<T, 4, 4>& target_trans) {
  Eigen::Vector<T, 3> meanTarget = Eigen::Vector<T, 3>::Zero();
  Eigen::Vector<T, 3> meanSrc = Eigen::Vector<T, 3>::Zero();
  for (unsigned int i = 0; i < source_size; i++) {
    meanSrc += (source_trans *
                reclib::make_Vector3(&(source_vertices[i * 3])).homogeneous())
                   .template head<3>();
  }
  for (unsigned int i = 0; i < target_size; i++) {
    meanTarget +=
        (target_trans *
         reclib::make_Vector3(&(target_vertices[i * 3])).homogeneous())
            .template head<3>();
  }

  meanTarget /= target_size;
  meanSrc /= source_size;

  return (meanTarget - meanSrc).template head<3>();
}

vec3 MeanTranslation(const reclib::opengl::GeometryBase& source,
                     const reclib::opengl::GeometryBase& target,
                     const mat4& source_trans, const mat4& target_trans) {
  if (0) {
    vec4 source_center =
        source_trans *
        ((source->bb_max - source->bb_min) + source->bb_min).homogeneous();
    vec4 target_center =
        target_trans *
        ((target->bb_max - target->bb_min) + target->bb_min).homogeneous();
    return (target_center - source_center).head<3>();
  } else {
    vec3 meanTarget(0, 0, 0);
    vec3 meanSrc(0, 0, 0);
    for (unsigned int i = 0; i < source->positions_size(); i++) {
      meanSrc +=
          (source_trans * source->get_position(i).homogeneous()).head<3>();
    }
    for (unsigned int i = 0; i < target->positions_size(); i++) {
      meanTarget +=
          (target_trans * target->get_position(i).homogeneous()).head<3>();
    }

    meanTarget /= target->positions_size();
    meanSrc /= source->positions_size();

    return (meanTarget - meanSrc).head<3>();
  }
}

template <typename T>
Sophus::SE3<T> pointToPointDirect(
    const std::vector<CorrespondenceBase<T, 3>>& corrs, T* scale) {
  auto cpy = corrs;

  // Compute center
  Eigen::Vector<T, 3> meanRef(0, 0, 0);
  Eigen::Vector<T, 3> meanSrc(0, 0, 0);
  for (auto c : corrs) {
    meanRef += c.refPoint;
    meanSrc += c.srcPoint;
  }
  meanRef /= corrs.size();
  meanSrc /= corrs.size();

  // Translate src to target and computed squared distance sum
  double refSumSq = 0;
  double srcSumSq = 0;
  for (auto& c : cpy) {
    c.refPoint -= meanRef;
    c.srcPoint -= meanSrc;
    refSumSq += c.refPoint.squaredNorm();
    srcSumSq += c.srcPoint.squaredNorm();
  }

  double S = 1;

  if (scale) {
    S = sqrt(refSumSq / srcSumSq);
    *scale = S;
  }

  Eigen::Vector<T, 3> t = meanRef - meanSrc;
  Eigen::Matrix<T, 3, 3> M;
  M.setZero();
  for (auto c : cpy) {
    M += (c.refPoint) * (c.srcPoint).transpose();
  }

  Eigen::Quaternion<T> R;

  if (0)
    R = orientationFromMixedMatrixUQ(M);
  else
    R = orientationFromMixedMatrixSVD(M);

  t = meanRef - S * (R * meanSrc);
  return Sophus::SE3<T>(R, t);
}

template <typename T>
Sophus::SE3<T> pointToPointDirect(
    const T* source_vertices, uint32_t source_size, const T* target_vertices,
    uint32_t target_size, const std::vector<unsigned int>& source_indices,
    const std::vector<unsigned int>& target_indices, T* scale) {
  _RECLIB_ASSERT_EQ(source_indices.size(), target_indices.size());
  // Compute center
  Eigen::Vector<T, 3> meanTarget(0, 0, 0);
  Eigen::Vector<T, 3> meanSrc(0, 0, 0);

  for (unsigned int i = 0; i < source_indices.size(); i++) {
    vec3 s(&(source_vertices[source_indices[i] * 3]));
    vec3 t(&(target_vertices[target_indices[i] * 3]));

    meanTarget += t;
    meanSrc += s;
  }
  meanTarget /= source_indices.size();
  meanSrc /= source_indices.size();

  std::vector<vec3> centered_src;
  std::vector<vec3> centered_target;

  // Translate src to target and computed squared distance sum
  double targetSumSq = 0;
  double srcSumSq = 0;
  for (unsigned int i = 0; i < source_indices.size(); i++) {
    vec3 s(&(source_vertices[source_indices[i] * 3]));
    vec3 t(&(target_vertices[target_indices[i] * 3]));

    s = (s - meanSrc);
    t = (t - meanTarget);

    centered_src.push_back(s);
    centered_target.push_back(t);

    targetSumSq += t.squaredNorm();
    srcSumSq += s.squaredNorm();
  }

  double S = 1;

  if (scale) {
    S = sqrt(targetSumSq / srcSumSq);
    *scale = S;
  }

  Eigen::Vector<T, 3> t = meanTarget - meanSrc;
  Eigen::Matrix<T, 3, 3> M;
  M.setZero();

  for (unsigned int i = 0; i < source_indices.size(); i++) {
    M += centered_target[i] * centered_src[i].transpose();
  }
  Eigen::Quaternion<T> R;

  if (0)
    R = orientationFromMixedMatrixUQ(M);
  else
    R = orientationFromMixedMatrixSVD(M);

  t = meanTarget - S * (R * meanSrc);
  return Sophus::SE3<T>(Eigen::Quaternion<T>(R), t);
}

#if HAS_DNN_MODULE
torch::Tensor pointToPointDirect(torch::Tensor source, torch::Tensor target) {
  _RECLIB_ASSERT_EQ(source.sizes()[0], target.sizes()[0]);
  _RECLIB_ASSERT_EQ(source.sizes()[1], 3);  // assume linearized tensor
  _RECLIB_ASSERT_EQ(target.sizes()[1], 3);  // assume linearized tensor
  torch::Device dev = source.device();

  // Compute center
  torch::Tensor mean_target = target.mean(0);
  torch::Tensor mean_source = source.mean(0);

  torch::Tensor centered_target = target - mean_target.unsqueeze(0);
  torch::Tensor centered_source = source - mean_source.unsqueeze(0);

  torch::Tensor translation = mean_target - mean_source;
  // [3xN] * [Nx3] = [3x3]
  torch::Tensor cov =
      torch::matmul(centered_target.transpose(1, 0), centered_source);

  if (cov.is_cuda()) {
    cov = cov.cpu();
  }
  mat3 M = reclib::dnn::torch2eigen<float>(cov);
  Eigen::Quaternion<float> Q;

  if (0)
    Q = orientationFromMixedMatrixUQ(M);
  else
    Q = orientationFromMixedMatrixSVD(M);

  mat3 R = Q.matrix();
  torch::Tensor rotation = reclib::dnn::eigen2torch(R).to(dev);

  translation = mean_target -
                torch::matmul(rotation, mean_source.unsqueeze(1)).squeeze(1);
  torch::Tensor T = torch::eye(4);
  T.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)},
               rotation);
  T.index_put_({torch::indexing::Slice(0, 3), 3}, translation);

  return T;
}

torch::Tensor MeanTranslation(torch::Tensor source, torch::Tensor target) {
  _RECLIB_ASSERT_EQ(source.sizes()[0], target.sizes()[0]);
  _RECLIB_ASSERT_EQ(source.sizes()[1], 3);  // assume linearized tensor
  _RECLIB_ASSERT_EQ(target.sizes()[1], 3);  // assume linearized tensor

  // Compute center
  torch::Tensor mean_target = target.mean(0);
  torch::Tensor mean_source = source.mean(0);

  return mean_target - mean_source;
}
#endif  // HAS_DNN_MODULE

template <typename T>
Eigen::Quaternion<T> orientationFromMixedMatrixUQ(
    const Eigen::Matrix<T, 3, 3>& M) {
  // Closed-form solution of absolute orientation using unit quaternions
  // https://pdfs.semanticscholar.org/3120/a0e44d325c477397afcf94ea7f285a29684a.pdf

  // Lower triangle
  Eigen::Matrix<T, 4, 4> N;
  N(0, 0) = M(0, 0) + M(1, 1) + M(2, 2);
  N(1, 0) = M(1, 2) - M(2, 1);
  N(2, 0) = M(2, 0) - M(0, 2);
  N(3, 0) = M(0, 1) - M(1, 0);

  N(1, 1) = M(0, 0) - M(1, 1) - M(2, 2);
  N(2, 1) = M(0, 1) + M(1, 0);
  N(3, 1) = M(2, 0) + M(0, 2);

  N(2, 2) = -M(0, 0) + M(1, 1) - M(2, 2);
  N(3, 2) = M(1, 2) + M(2, 1);

  N(3, 3) = -M(0, 0) - M(1, 1) + M(2, 2);
  //    N       = N.selfadjointView<Eigen::Upper>();

  //    Eigen::EigenSolver<Eigen::Matrix<T,4,4>> eigenSolver(N, true);

  // Only the lower triangular part of the input matrix is referenced.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 4, 4>> eigenSolver(N);

  int largestEV = 0;
  for (auto i = 1; i < 4; ++i) {
    if (eigenSolver.eigenvalues()(i) > eigenSolver.eigenvalues()(largestEV)) {
      largestEV = i;
    }
  }

  int largestEV2 = 0;
  eigenSolver.eigenvalues().maxCoeff(&largestEV2);
  _RECLIB_ASSERT_EQ(largestEV, largestEV2);

  Eigen::Vector<T, 4> E = eigenSolver.eigenvectors().col(largestEV);

  Eigen::Quaternion<T> R(E(0), E(1), E(2), E(3));
  R = R.conjugate();
  R.normalize();
  if (R.w() < 0) R.coeffs() *= -1;
  return R;
  //    return R;
}

template <typename T>
inline Eigen::Quaternion<T> orientationFromMixedMatrixSVD(
    const Eigen::Matrix<T, 3, 3>& M) {
  // polar decomp
  // M = USV^T
  // R = UV^T
  Eigen::JacobiSVD svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Vector<T, 3> S = Eigen::Vector<T, 3>::Ones(3);
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(2) = -1;

  Eigen::Matrix<T, 3, 3> R =
      svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

  Eigen::Quaternion<T> q = Eigen::Quaternion<T>(R).normalized();
  if (q.coeffs().array().isNaN().any()) {
    q = Eigen::Quaternion<T>::Identity();
  }
  return q;
}

template <typename T>
Sophus::SE3<T> pointToPointIterative(
    const std::vector<CorrespondenceBase<T, 3>>& corrs,
    const Sophus::SE3<T>& guess, int innerIterations) {
  _RECLIB_ASSERT_GE(corrs.size(), 6);

  float last_error = std::numeric_limits<float>::max();
  float initial_error = std::numeric_limits<float>::max();

  Sophus::SE3<T> trans = guess;
  Eigen::Matrix<T, 6, 6> JtJ;
  Eigen::Matrix<T, 6, 1> Jtb;

  for (int k = 0; k < innerIterations; ++k) {
    JtJ.setZero();
    Jtb.setZero();

    for (size_t i = 0; i < corrs.size(); ++i) {
      auto& corr = corrs[i];

      Eigen::Vector<T, 3> sp = trans * corr.srcPoint;

      Eigen::Matrix<T, 3, 6> Jrow;
      Jrow.template block<3, 3>(0, 0) = Eigen::Matrix<T, 3, 3>::Identity();
      Jrow.template block<3, 3>(0, 3) = -skew<T>(sp);

      Eigen::Vector<T, 3> res = corr.refPoint - sp;

      // use weight
      Jrow *= corr.weight;
      res *= corr.weight;

      JtJ += Jrow.transpose() * Jrow;
      Jtb += Jrow.transpose() * res;
    }
    Eigen::Matrix<T, 6, 1> x = JtJ.ldlt().solve(Jtb);
    trans = Sophus::SE3<T>::exp(x) * trans;

    if (1) {
      float mean_error_p2p = 0;
      float mean_error_p2pl = 0;
      for (unsigned int j = 0; j < corrs.size(); j++) {
        Eigen::Vector<T, 3> refPoint = corrs[j].refPoint;
        Eigen::Vector<T, 3> refNormal = corrs[j].refNormal;
        Eigen::Vector<T, 3> srcPoint = corrs[j].refPoint;

        srcPoint = (trans.matrix() * srcPoint.homogeneous()).template head<3>();

        mean_error_p2p += (refPoint - srcPoint).squaredNorm();
        mean_error_p2pl += refNormal.dot(refPoint - srcPoint);
      }
      mean_error_p2p /= corrs.size();
      mean_error_p2pl /= corrs.size();
      std::cout << "[p2p] Inner iteration " << k << ": p2p: " << mean_error_p2p
                << " p2pl: " << mean_error_p2pl << std::endl;

      if (k == 0) initial_error = mean_error_p2p;

      if (abs(mean_error_p2p - last_error) / abs(last_error) < 1e-4f) {
        break;
      }
      if ((mean_error_p2p - initial_error) / initial_error > 0.5f) {
        break;
      }
      last_error = mean_error_p2p;
    }
  }
  return trans;
}

template <typename T>
Sophus::SE3<T> pointToPlane(const std::vector<CorrespondenceBase<T, 3>>& corrs,
                            const Sophus::SE3<T>& ref,
                            const Sophus::SE3<T>& _src, int innerIterations) {
  _RECLIB_ASSERT_GE(corrs.size(), 6);
  auto src = _src;
  Eigen::Matrix<T, 6, 6> JtJ;
  Eigen::Matrix<T, 6, 1> Jtb;

  float last_error = std::numeric_limits<float>::max();
  float initial_error = std::numeric_limits<float>::max();

  for (int k = 0; k < innerIterations; ++k) {
    // Make use of symmetry
    JtJ.template triangularView<Eigen::Upper>().setZero();
    Jtb.setZero();

    for (size_t i = 0; i < corrs.size(); ++i) {
      auto& corr = corrs[i];

      Eigen::Vector<T, 3> rp = ref * corr.refPoint;
      Eigen::Vector<T, 3> rn = ref.so3() * corr.refNormal;
      Eigen::Vector<T, 3> sp = src * corr.srcPoint;

      Eigen::Matrix<T, 6, 1> row;
      row.template head<3>() = rn;
      // This is actually equal to:
      //      row.tail<3>() = -skew(sp).transpose() * rn;
      row.template tail<3>() = sp.cross(rn);
      Eigen::Vector<T, 3> di = rp - sp;
      double res = rn.dot(di);

      // use weight
      row *= corr.weight;
      res *= corr.weight;

      //            JtJ += row * row.transpose();
      JtJ += (row * row.transpose()).template triangularView<Eigen::Upper>();
      Jtb += row * res;
    }

    //        Eigen::Matrix<double, 6, 1> x = JtJ.ldlt().solve(Jtb);
    Eigen::Matrix<T, 6, 1> x =
        JtJ.template selfadjointView<Eigen::Upper>().ldlt().solve(Jtb);
    _RECLIB_ASSERT(!Eigen::isnan(x.array()).any());
    src = Sophus::SE3<T>::exp(x) * src;

    if (1) {
      float mean_error_p2p = 0;
      float mean_error_p2pl = 0;
      for (unsigned int j = 0; j < corrs.size(); j++) {
        Eigen::Vector<T, 3> refPoint = corrs[j].refPoint;
        Eigen::Vector<T, 3> refNormal = corrs[j].refNormal;
        Eigen::Vector<T, 3> srcPoint = corrs[j].refPoint;

        srcPoint = (src.matrix() * srcPoint.homogeneous()).template head<3>();

        mean_error_p2p += (refPoint - srcPoint).squaredNorm();
        mean_error_p2pl += refNormal.dot(refPoint - srcPoint);
      }
      mean_error_p2p /= corrs.size();
      mean_error_p2pl /= corrs.size();
      std::cout << "[p2pl] Inner iteration " << k << ": p2p: " << mean_error_p2p
                << " p2pl: " << mean_error_p2pl << std::endl;

      if (k == 0) initial_error = mean_error_p2p;

      if (abs(mean_error_p2p - last_error) / abs(last_error) < 1e-4f) {
        break;
      }
      if ((mean_error_p2p - initial_error) / initial_error > 0.5f) {
        break;
      }
      last_error = mean_error_p2p;
    }
  }
  return src;
}

// ------------------------------------------------------------
// ------------------------------------------------------------
// Template instantiation
// ------------------------------------------------------------
// ------------------------------------------------------------

template Eigen::Vector<float, 3> MeanTranslation(
    const float* source_vertices, uint32_t source_size,
    const float* target_vertices, uint32_t target_size,
    const std::vector<unsigned int>& source_indices,
    const std::vector<unsigned int>& target_indices);

template Eigen::Vector<float, 3> MeanTranslation(
    const std::vector<CorrespondenceBase<float, 3>>& corrs);
template Eigen::Vector<double, 3> MeanTranslation(
    const std::vector<CorrespondenceBase<double, 3>>& corrs);

template Eigen::Vector<float, 3> MeanTranslation(const float* source_vertices,
                                                 uint32_t source_size,
                                                 const float* target_vertices,
                                                 uint32_t target_size,
                                                 const mat4& source_trans,
                                                 const mat4& target_trans);
template Eigen::Vector<double, 3> MeanTranslation(const double* source_vertices,
                                                  uint32_t source_size,
                                                  const double* target_vertices,
                                                  uint32_t target_size,
                                                  const Mat4& source_trans,
                                                  const Mat4& target_trans);

template Sophus::SE3<float> pointToPointDirect(
    const float* source_vertices, uint32_t source_size,
    const float* target_vertices, uint32_t target_size,
    const std::vector<unsigned int>& source_indices,
    const std::vector<unsigned int>& target_indices, float* scale);

template Sophus::SE3<float> pointToPointDirect(
    const std::vector<CorrespondenceBase<float, 3>>& corrs, float* scale);
template Sophus::SE3<double> pointToPointDirect(
    const std::vector<CorrespondenceBase<double, 3>>& corrs, double* scale);

template Eigen::Quaternion<float> orientationFromMixedMatrixUQ(
    const Eigen::Matrix<float, 3, 3>& M);
template Eigen::Quaternion<double> orientationFromMixedMatrixUQ(
    const Eigen::Matrix<double, 3, 3>& M);

template Eigen::Quaternion<float> orientationFromMixedMatrixSVD(
    const Eigen::Matrix<float, 3, 3>& M);
template Eigen::Quaternion<double> orientationFromMixedMatrixSVD(
    const Eigen::Matrix<double, 3, 3>& M);

template Sophus::SE3<float> pointToPointIterative(
    const std::vector<CorrespondenceBase<float, 3>>& corrs,
    const Sophus::SE3<float>& guess, int innerIterations);
template Sophus::SE3<double> pointToPointIterative(
    const std::vector<CorrespondenceBase<double, 3>>& corrs,
    const Sophus::SE3<double>& guess, int innerIterations);

template Sophus::SE3<float> pointToPlane(
    const std::vector<CorrespondenceBase<float, 3>>& corrs,
    const Sophus::SE3<float>& ref, const Sophus::SE3<float>& _src,
    int innerIterations);

template Sophus::SE3<double> pointToPlane(
    const std::vector<CorrespondenceBase<double, 3>>& corrs,
    const Sophus::SE3<double>& ref, const Sophus::SE3<double>& _src,
    int innerIterations);

}  // namespace optim
}  // namespace reclib
