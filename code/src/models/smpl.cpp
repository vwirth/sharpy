
#include <limits>
#if __unix__
#include <reclib/models/smpl.h>

#include <fstream>
#include <iostream>
#include <string>

#include "reclib/assert.h"
#include "reclib/python/npy2eigen.h"
#include "reclib/utils/colors.h"

namespace reclib {
namespace models {

template <typename T>
Eigen::Matrix<float, Eigen::Dynamic, 6> compute_visible_part_bbs(
    reclib::models::ModelInstance<T>& mano, reclib::opengl::Camera cam) {
  Eigen::Matrix<float, Eigen::Dynamic, 6> bbs(
      mano.model.n_joints(), 6);  // [bbminx,bbminy,bbminz,bbmax,bbmaxy,bbmaxz]

  for (unsigned int i = 0; i < bbs.rows(); i++) {
    bbs(i, 0) = std::numeric_limits<float>::infinity();
    bbs(i, 1) = std::numeric_limits<float>::infinity();
    bbs(i, 2) = std::numeric_limits<float>::infinity();
    bbs(i, 3) = -std::numeric_limits<float>::infinity();
    bbs(i, 4) = -std::numeric_limits<float>::infinity();
    bbs(i, 5) = -std::numeric_limits<float>::infinity();
  }

  mat4 view = cam->view_normal;
  for (unsigned int i = 0; i < mano.model.n_verts(); i++) {
    vec4 n_cam =
        view * mano.gl_instance->mesh->geometry->get_normal(i).homogeneous();
    if (n_cam.z() > 0) {
      vec3 v = mano.gl_instance->mesh->geometry->get_position(i);
      int J = mano.model.verts2joints[i];

      for (unsigned int j = 0; j < 3; j++) {
        if (bbs(J, j) > v[j]) {
          bbs(J, j) = v[j];
        }
        if (bbs(J, 3 + j) < v[j]) {
          bbs(J, 3 + j) = v[j];
        }
      }
    }
  }

  return bbs;
}

template Eigen::Matrix<float, Eigen::Dynamic, 6> compute_visible_part_bbs(
    reclib::models::ModelInstance<reclib::models::MANOConfig>& mano,
    reclib::opengl::Camera cam);
template Eigen::Matrix<float, Eigen::Dynamic, 6> compute_visible_part_bbs(
    reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& mano,
    reclib::opengl::Camera cam);
template Eigen::Matrix<float, Eigen::Dynamic, 6> compute_visible_part_bbs(
    reclib::models::ModelInstance<
        reclib::models::MANOConfigAnglePCAGeneric<45>>& mano,
    reclib::opengl::Camera cam);
template Eigen::Matrix<float, Eigen::Dynamic, 6> compute_visible_part_bbs(
    reclib::models::ModelInstance<reclib::models::MANOConfigAnglePCA>& mano,
    reclib::opengl::Camera cam);

template <typename T>
Eigen::Vector<float, reclib::models::MANOConfigPCA::n_hand_pca_joints() * 3>
mano_angle2pca(reclib::models::ModelInstance<T>& mano) {
  Eigen::Vector<float, reclib::models::MANOConfigPCA::n_hand_pca_joints()* 3>
      pca = mano.pose().template tail<45>();

  if (mano.add_pose_mean_) {
    // add mean pose to all joints except the root joint
    pca.noalias() += mano.model.hand_mean;
  }

  if (mano.model.hand_type == reclib::models::HandType::left) {
    reclib::models::Model<reclib::models::MANOConfigPCAGeneric<45>> mano_pca(
        reclib::models::HandType::left);
    return (mano_pca.hand_comps.inverse() * pca);
  } else {
    reclib::models::Model<reclib::models::MANOConfigPCAGeneric<45>> mano_pca(
        reclib::models::HandType::right);
    return (mano_pca.hand_comps.inverse() * pca);
  }
}

// template instantiation
template Eigen::Vector<float,
                       reclib::models::MANOConfigPCA::n_hand_pca_joints() * 3>
mano_angle2pca(reclib::models::ModelInstance<reclib::models::MANOConfig>& mano);
template Eigen::Vector<float,
                       reclib::models::MANOConfigPCA::n_hand_pca_joints() * 3>
mano_angle2pca(
    reclib::models::ModelInstance<reclib::models::MANOConfigAnglePCA>& mano);
template Eigen::Vector<float,
                       reclib::models::MANOConfigPCA::n_hand_pca_joints() * 3>
mano_angle2pca(reclib::models::ModelInstance<
               reclib::models::MANOConfigAnglePCAGeneric<45>>& mano);

template <typename T>
std::vector<vec4> mano_canonical_space_colors(
    reclib::models::ModelInstance<T>& mano, bool transform_mano,
    bool use_cylinder) {
  reclib::models::ModelInstance<T> mano_copy(mano.model);
  mano_copy.add_pose_mean_ = mano.add_pose_mean_;
  mano_copy.params.setConstant(0);
  mano_copy.update(false, true);

  int sign = 1;
  mat4 model = mat4::Identity();
  if (mano_copy.model.hand_type == reclib::models::HandType::left) {
    sign = -1;
    model(0, 0) = -1;
  }

  vec3 bbmin(std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max());
  vec3 bbmax(std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min());

  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    vec3 v = mano_copy.verts().row(i).transpose();
    vec3 v_trans = (model * v.homogeneous()).hnormalized();
    bbmin = v_trans.cwiseMin(bbmin);
    bbmax = v_trans.cwiseMax(bbmax);
  }

  vec3 whd = bbmax - bbmin;

  std::vector<vec4> colors;
  if (!use_cylinder) {
    mat4 to_canonical = mat4::Identity();

    vec3 scale = vec3::Ones().cwiseQuotient(whd);
    to_canonical = reclib::scale(scale);
    to_canonical *= reclib::translate(vec3(-bbmin));
    to_canonical = to_canonical * model;

    for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
      vec3 v = mano_copy.verts().row(i).transpose();
      v = (to_canonical * v.homogeneous()).hnormalized();

      if (mano_copy.model.hand_type == reclib::models::HandType::left) {
        colors.push_back(vec4(v.x(), v.y(), v.z(), 0.5));
      } else {
        colors.push_back(vec4(v.x(), v.y(), v.z(), 1.0));
      }
    }

    if (0 && !reclib::opengl::Drawelement::valid("test_bb")) {
      reclib::opengl::Material mat("test_mat");
      mat->vec4_map["color"] = vec4(1, 0, 0, 1);
      reclib::opengl::Shader shad("test_bb", "MVP.vs", "color4Uniform.fs");
      reclib::opengl::BoundingBox bb("test_bb", bbmin, bbmax, mat);
      reclib::opengl::Drawelement d("test_bb", shad, bb);
    }
  } else {
    float min_val = std::numeric_limits<float>::max();
    unsigned int min_axis = 0;

    for (unsigned int i = 0; i < 3; i++) {
      if (whd[i] < min_val) {
        min_val = whd[i];
        min_axis = i;
      }
    }
    vec3 cyl_center = bbmin + (bbmax - bbmin) * 0.5;

    float value_min = bbmin.y();
    int value_axis = 1;

    vec2 hue_axis = vec2(1, 0);

    float saturation_max = 0;
    for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
      vec3 v = mano_copy.verts().row(i).transpose();
      v = (model * v.homogeneous()).hnormalized();
      vec3 normed = v - cyl_center;
      vec2 planar = vec2::Zero();
      unsigned int j = 0;
      for (unsigned int i = 0; i < 3; i++) {
        if (i != value_axis) {
          planar[j] = normed[i];
          j++;
        }
      }
      if (planar.norm() > saturation_max) {
        saturation_max = planar.norm();
      }
    }

    if (0 && !reclib::opengl::Drawelement::valid("test_bb")) {
      reclib::opengl::Material mat("test_mat");
      mat->vec4_map["color"] = vec4(1, 0, 0, 1);
      reclib::opengl::Shader shad("test_bb", "MVP.vs", "color4Uniform.fs");
      reclib::opengl::Cylinder bb("test_bb", saturation_max, whd.y(),
                                  vec3(0, 1, 0), cyl_center, mat);
      reclib::opengl::Drawelement d("test_bb", shad, bb);
    }

    // hue axis: x-z plane
    // value axis: y

    auto xyz2hsv = [&](const vec3& xyz) {
      vec3 normed = xyz - cyl_center;
      // float value = normed[value_axis] / value_range;
      float value = (xyz.y() - value_min) / whd.y();

      vec2 planar = vec2::Zero();
      unsigned int j = 0;
      for (unsigned int i = 0; i < 3; i++) {
        if (i != value_axis) {
          planar[j] = normed[i];
          j++;
        }
      }

      float saturation = ((planar.norm()) / (saturation_max));
      planar = planar.normalized();

      // calculate the hue, the angle between saturation axis and current planar
      // vector
      float dot = planar.dot(hue_axis);
      mat2 vec2mat = mat2::Zero();
      vec2mat.col(0) = hue_axis;
      vec2mat.col(1) = planar;
      float det = vec2mat.determinant();
      float hue = atan2(det, dot) * (180.f / M_PI);
      // atan yields values in [-pi,pi] range
      // if (hue < 0) {
      //   hue = 360.f + hue;
      // }

      _RECLIB_ASSERT_GE(value, -0.01f);
      _RECLIB_ASSERT_GE(saturation, -0.01f);
      _RECLIB_ASSERT_GE(hue, -180);
      _RECLIB_ASSERT_LE(value, 1);
      _RECLIB_ASSERT_LE(saturation, 1);
      _RECLIB_ASSERT_LE(hue, 180);

      value = reclib::clamp(value, 0, 1);
      saturation = reclib::clamp(saturation, 0, 1);
      hue = reclib::clamp(hue, -180, 180);

      return vec3(hue, saturation, value);
      // return vec3(hue, value, saturation);
    };

    for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
      vec3 v = mano_copy.verts().row(i).transpose();
      v = (model * v.homogeneous()).hnormalized();
      vec3 c = (xyz2hsv(v));
      c[0] = (c[0] + 180.f);

      c = hsv2rgb(c);

      if (mano_copy.model.hand_type == reclib::models::HandType::left) {
        colors.push_back(vec4(c.x(), c.y(), c.z(), 0.5));
      } else {
        colors.push_back(vec4(c.x(), c.y(), c.z(), 1.0));
      }
    }
  }

  if (transform_mano) {
    mano.params = mano_copy.params;
    mano.update(false, true);
  }

  return colors;
}

// template instantiation
template std::vector<vec4> mano_canonical_space_colors(
    reclib::models::ModelInstance<reclib::models::MANOConfig>& mano,
    bool transform_mano, bool use_cylinder);
// template instantiation
template std::vector<vec4> mano_canonical_space_colors(
    reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& mano,
    bool transform_mano, bool use_cylinder);
// template instantiation
template std::vector<vec4> mano_canonical_space_colors(
    reclib::models::ModelInstance<reclib::models::MANOConfigAnglePCA>& mano,
    bool transform_mano, bool use_cylinder);

template <typename T>
std::vector<vec4> mano_canonical_space_colors_hsv(
    reclib::models::ModelInstance<T>& mano, bool transform_mano,
    bool convert_to_rgb) {
  bool DEBUG = false;

  reclib::models::ModelInstance<T> mano_copy(mano.model);
  mano_copy.add_pose_mean_ = false;
  mano_copy.params.setConstant(0);
  mano_copy.update(false, true);
  mano_copy.generate_gl_drawelem();
  float* mano_normals = mano_copy.gl_instance->mesh->geometry->normals_ptr();
  // mano_copy is now transformed to the pose in the canonical space!

  vec3 bbmin(std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max());
  vec3 bbmax(std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min());

  mat4 model = mat4::Identity();
  if (mano_copy.model.hand_type == reclib::models::HandType::left) {
    model(0, 0) = -1;
  }

  // --------------------------------------------------
  // Calculate AABB
  // --------------------------------------------------
  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    vec3 v = mano_copy.verts().row(i).transpose();
    vec3 v_trans = (model * v.homogeneous()).hnormalized();
    bbmin = v_trans.cwiseMin(bbmin);
    bbmax = v_trans.cwiseMax(bbmax);
  }
  vec3 whd = bbmax - bbmin;

  // --------------------------------------------------
  // Calculate Value Axis
  // --------------------------------------------------
  float min_val = std::numeric_limits<float>::max();
  unsigned int min_axis = 0;

  for (unsigned int i = 0; i < 3; i++) {
    if (whd[i] < min_val) {
      min_val = whd[i];
      min_axis = i;
    }
  }
  unsigned int value_axis = min_axis;
  float value_range = min_val;

  vec3 root_joint = mano_copy.joints().row(0).transpose();
  vec3 cyl_center = root_joint + vec3(0.1, 0, 0);

  // --------------------------------------------------
  // Calculate Cylinder Radius
  // --------------------------------------------------
  float radius = 0;
  cyl_center[min_axis] = bbmin[min_axis] + value_range / 2.f;
  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    vec3 v = mano_copy.verts().row(i).transpose();
    vec3 v_trans = (model * v.homogeneous()).hnormalized();
    vec3 v_norm = v_trans - cyl_center;

    if (v_norm.norm() > radius) {
      radius = v_norm.norm();
    }
  }

  // --------------------------------------------------
  // Calculate HSV Origin and Saturation Root
  // --------------------------------------------------
  vec2 hue_axis = (vec2(-1, 0) + vec2(0, 0.0)).normalized();
  vec3 hsv_origin = cyl_center;
  hsv_origin[value_axis] -= (value_range / 2.f);

  vec3 saturation_root = (root_joint + vec3(0.03, 0, 0));
  vec2 planar_sat_root = vec2::Zero();
  unsigned int j = 0;
  for (unsigned int i = 0; i < 3; i++) {
    if (i != value_axis) {
      planar_sat_root[j] = saturation_root[i] - cyl_center[i];
      j++;
    }
  }

  // --------------------------------------------------
  // Construct Saturation Space dependent on the Kinematic Tree of the Mano
  // Model
  // --------------------------------------------------
  float max_saturation_thumb = 0;
  float max_saturation_index = 0;
  float max_saturation_middle = 0;
  float max_saturation_ring = 0;
  float max_saturation_pinky = 0;
  std::vector<unsigned int> max_vertices(5);
  std::vector<vec2> planar_saturation_axes(5);

  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    vec3 v = mano_copy.verts().row(i).transpose();
    v = (model * v.homogeneous()).hnormalized();

    int major_joint = -1;
    mano_copy.model.weights.row(i).toDense().maxCoeff(&major_joint);

    switch (major_joint) {
      case (MANOConfigExtra::THUMB_IDX + 2): {
        vec3 normed = v - saturation_root;
        vec2 planar = vec2::Zero();
        unsigned int j = 0;
        for (unsigned int p = 0; p < 3; p++) {
          if (p != value_axis) {
            planar[j] = normed[p];
            j++;
          }
        }
        if (planar.norm() > max_saturation_thumb) {
          max_saturation_thumb = planar.norm();
          max_vertices[0] = i;
          planar_saturation_axes[0] = planar;
        }
        break;
      }
      case (MANOConfigExtra::INDEX_IDX + 2): {
        vec3 normed = v - saturation_root;
        vec2 planar = vec2::Zero();
        unsigned int j = 0;
        for (unsigned int p = 0; p < 3; p++) {
          if (p != value_axis) {
            planar[j] = normed[p];
            j++;
          }
        }
        if (planar.norm() > max_saturation_index) {
          max_saturation_index = planar.norm();
          max_vertices[1] = i;
          planar_saturation_axes[1] = planar;
        }
        break;
      }
      case (MANOConfigExtra::MIDDLE_IDX + 2): {
        vec3 normed = v - saturation_root;
        vec2 planar = vec2::Zero();
        unsigned int j = 0;
        for (unsigned int p = 0; p < 3; p++) {
          if (p != value_axis) {
            planar[j] = normed[p];
            j++;
          }
        }
        if (planar.norm() > max_saturation_middle) {
          max_saturation_middle = planar.norm();
          max_vertices[2] = i;
          planar_saturation_axes[2] = planar;
        }
        break;
      }
      case (MANOConfigExtra::RING_IDX + 2): {
        vec3 normed = v - saturation_root;
        vec2 planar = vec2::Zero();
        unsigned int j = 0;
        for (unsigned int p = 0; p < 3; p++) {
          if (p != value_axis) {
            planar[j] = normed[p];
            j++;
          }
        }
        if (planar.norm() > max_saturation_ring) {
          max_saturation_ring = planar.norm();
          max_vertices[3] = i;
          planar_saturation_axes[3] = planar;
        }
        break;
      }
      case (MANOConfigExtra::PINKY_IDX + 2): {
        vec3 normed = v - saturation_root;
        vec2 planar = vec2::Zero();
        unsigned int j = 0;
        for (unsigned int p = 0; p < 3; p++) {
          if (p != value_axis) {
            planar[j] = normed[p];
            j++;
          }
        }
        if (planar.norm() > max_saturation_pinky) {
          max_saturation_pinky = planar.norm();
          max_vertices[4] = i;
          planar_saturation_axes[4] = planar;
        }
        break;
      }
      default: {
      }
    }
  }

  std::vector<vec3> thumb_vertices;
  std::vector<vec3> index_vertices;
  std::vector<vec3> middle_vertices;
  std::vector<vec3> ring_vertices;
  std::vector<vec3> pinky_vertices;

  float value_min = 0.3;
  float value_scale = (1.f - value_min) / 2.f;
  float value_shift = value_min + (1.f * value_scale);

  // --------------------------------------------------
  // XYZ -> HSV Function
  // --------------------------------------------------
  auto xyz2hsv = [&](const vec3& xyz, const vec3& normal) {
    vec3 normed = xyz - hsv_origin;
    // float value = normed[value_axis] / value_range;
    float value = (normal.y() * value_scale) + value_shift;

    vec2 planar = vec2::Zero();
    vec2 planar_saturation = vec2::Zero();
    unsigned int j = 0;
    for (unsigned int i = 0; i < 3; i++) {
      if (i != value_axis) {
        planar[j] = normed[i];
        planar_saturation[j] = (xyz - saturation_root)[i];
        j++;
      }
    }

    unsigned int hand_idx = 0;
    float idx_dist = 1000;
    for (unsigned int i = 0; i < planar_saturation_axes.size(); i++) {
      float dot = planar_saturation.dot(planar_saturation_axes[i].normalized());
      vec2 scaled_axis = dot * planar_saturation_axes[i].normalized();
      vec2 dist = scaled_axis - planar_saturation;

      if (dist.norm() < idx_dist) {
        idx_dist = dist.norm();
        hand_idx = i;
      }
    }

    if (DEBUG) {
      switch (hand_idx) {
        case (0): {
          thumb_vertices.push_back(xyz);
          break;
        }
        case (1): {
          index_vertices.push_back(xyz);
          break;
        }
        case (2): {
          middle_vertices.push_back(xyz);
          break;
        }
        case (3): {
          ring_vertices.push_back(xyz);
          break;
        }
        case (4): {
          pinky_vertices.push_back(xyz);
          break;
        }
      }
    }

    float saturation_range = planar_saturation_axes[hand_idx].norm();
    float saturation = ((planar_saturation.norm()) / (saturation_range));
    planar = planar.normalized();

    // calculate the hue, the angle between saturation axis and current planar
    // vector
    float dot = planar.dot(hue_axis);
    mat2 vec2mat = mat2::Zero();
    vec2mat.col(0) = hue_axis;
    vec2mat.col(1) = planar;
    float det = vec2mat.determinant();
    float hue = atan2(det, dot) * (180.f / M_PI);
    // atan yields values in [-pi,pi] range
    // if (hue < 0) {
    //   hue = 360.f + hue;
    // }

    _RECLIB_ASSERT_GE(value, -0.01f);
    _RECLIB_ASSERT_GE(saturation, -0.01f);
    _RECLIB_ASSERT_GE(hue, -180);
    _RECLIB_ASSERT_LE(value, 1);
    _RECLIB_ASSERT_LE(saturation, 1);
    _RECLIB_ASSERT_LE(hue, 180);

    value = reclib::clamp(value, 0, 1);
    saturation = reclib::clamp(saturation, 0, 1);
    hue = reclib::clamp(hue, -180, 180);

    return vec3(hue, saturation, value);
    // return vec3(hue, value, saturation);
  };

  // --------------------------------------------------
  // HSV -> RGB Function
  // --------------------------------------------------
  auto hsv2rgb = [&](const vec3& hsv) {
    float hue = hsv[0];
    if (hue < 0) {
      hue = 360.f + hue;
    }

    float C = hsv[2] * hsv[1];
    float X = C * (1 - abs(fmod(hue / 60.f, 2.f) - 1));
    float m = hsv[2] - C;

    vec3 rgb_prime = vec3::Zero();

    if (hue >= 0 && hue < 60) {
      rgb_prime = vec3(C, X, 0);
    } else if (hue >= 60 && hue < 120) {
      rgb_prime = vec3(X, C, 0);
    } else if (hue >= 120 && hue < 180) {
      rgb_prime = vec3(0, C, X);
    } else if (hue >= 180 && hue < 240) {
      rgb_prime = vec3(0, X, C);
    } else if (hue >= 240 && hue < 300) {
      rgb_prime = vec3(X, 0, C);
    } else if (hue >= 300 && hue < 360) {
      rgb_prime = vec3(C, 0, X);
    }

    vec3 rgb = (rgb_prime + reclib::make_vec3(m));
    return rgb;
  };

  // --------------------------------------------------
  // Min/Max Search for Hue Range
  // --------------------------------------------------
  float max_hue = -370;
  float min_hue = 370;
  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    vec3 v = mano_copy.verts().row(i).transpose();
    vec3 n(&(mano_normals[i * 3]));
    v = (model * v.homogeneous()).hnormalized();
    vec3 c = (xyz2hsv(v, n));

    if (c[0] > max_hue) {
      max_hue = c[0];
    }
    if (c[0] < min_hue) {
      min_hue = c[0];
    }
  }

  // min_hue -= 5.f;
  max_hue += 1.f;

  if (DEBUG) {
    thumb_vertices.clear();
    index_vertices.clear();
    middle_vertices.clear();
    ring_vertices.clear();
    pinky_vertices.clear();
  }

  // --------------------------------------------------
  // Compute Final Colors
  // --------------------------------------------------
  std::vector<vec4> colors;
  std::vector<vec3> colors3;
  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    vec3 v = mano_copy.verts().row(i).transpose();
    vec3 n(&(mano_normals[i * 3]));
    v = (model * v.homogeneous()).hnormalized();
    vec3 c = (xyz2hsv(v, n));
    c[0] = ((c[0] - min_hue) / (max_hue - min_hue)) * 360.f;

    if (convert_to_rgb) {
      c = hsv2rgb(c);
    }

    if (mano_copy.model.hand_type == reclib::models::HandType::left) {
      colors.push_back(vec4(c.x(), c.y(), c.z(), 0.5));
    } else {
      colors.push_back(vec4(c.x(), c.y(), c.z(), 1.0));
    }

    if (DEBUG) {
      colors3.push_back(c);
    }
  }

  // --------------------------------------------------
  // Lots of Debug stuff...
  // --------------------------------------------------
  if (DEBUG) {
    if (!reclib::opengl::Shader::valid("hsv2rgb")) {
      reclib::opengl::Shader s("hsv2rgb", "MVP_color3.vs", "color3PV.fs");
    }
    if (!reclib::opengl::Drawelement::valid("hsv2rgb")) {
      reclib::opengl::Drawelement d =
          reclib::opengl::DrawelementImpl::from_geometry(
              "hsv2rgb", reclib::opengl::Shader::find("hsv2rgb"), false,
              colors3);
      d->set_model_translation(vec3(1, 1, 0));
      d->mesh->primitive_type = GL_POINTS;
      d->mesh->geometry->add_attribute_vec3("colors", colors3);
      d->mesh->geometry->update_meshes();
    }
    if (!reclib::opengl::Drawelement::valid("rgb_bb")) {
      reclib::opengl::Material bb_mat("rgb_bb");
      bb_mat->vec4_map["color"] = vec4(0, 0, 0, 1);
      reclib::opengl::BoundingBox bb("rgb_bb", vec3(0, 0, 0), vec3(1, 1, 1),
                                     bb_mat);
      reclib::opengl::Shader s("rgb_bb", "MVP.vs", "color4Uniform.fs");
      reclib::opengl::Drawelement bb_d("rgb_bb", s, bb);
      bb_d->set_model_translation(vec3(1, 1, 0));
    }
    if (!reclib::opengl::Drawelement::valid("saturation")) {
      reclib::opengl::Shader s("saturation", "MVP.vs", "color4Uniform.fs");
      reclib::opengl::Material mat("saturation");
      mat->vec4_map["color"] = vec4(0, 0, 1, 1);

      std::vector<vec3> positions;
      std::vector<uint32_t> indices;
      positions.push_back(saturation_root);
      for (unsigned int i = 0; i < max_vertices.size(); i++) {
        positions.push_back(mano_copy.verts().row(max_vertices[i]).transpose());
        indices.push_back(0);
        indices.push_back(positions.size() - 1);
        std::cout << "max_vertices: " << max_vertices[i] << std::endl;
      }

      reclib::opengl::Drawelement d =
          reclib::opengl::DrawelementImpl::from_geometry(
              "saturation", reclib::opengl::Shader::find("saturation"), mat,
              false, positions, indices);
      d->mesh->primitive_type = GL_LINES;
      d->mesh->geometry->update_meshes();
    }
    if (!reclib::opengl::Drawelement::valid("mano_bb")) {
      reclib::opengl::Material bb_mat("bb_mat");
      bb_mat->vec4_map["color"] = vec4(1, 0, 0, 1);
      reclib::opengl::BoundingBox bb("mano_bb", bbmin, bbmax, bb_mat);
      reclib::opengl::Shader s("canonical", "MVP.vs", "color4Uniform.fs");
      reclib::opengl::Drawelement("mano_bb", s, bb);
    }
    if (!reclib::opengl::Drawelement::valid("mano_cyl")) {
      reclib::opengl::Material cyl_mat("cyl_mat");
      if (!reclib::opengl::Shader::valid("canonical")) {
        reclib::opengl::Shader s("canonical", "MVP.vs", "color4Uniform.fs");
      }

      cyl_mat->vec4_map["color"] = vec4(1, 1, 0, 1);
      vec3 axis = vec3::Zero();
      axis[value_axis] = 1;

      reclib::opengl::Cylinder cyl("mano_cyl", radius, value_range, axis,
                                   cyl_center, cyl_mat);
      reclib::opengl::Drawelement d(
          "mano_cyl", reclib::opengl::Shader::find("canonical"), cyl);
      d->set_wireframe_mode(true);
    }
    if (!reclib::opengl::Drawelement::valid("thumb_points")) {
      {
        reclib::opengl::Material mat("thumb_points");
        mat->vec4_map["color"] = vec4(1, 0, 0, 1);

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "thumb_points", reclib::opengl::Shader::find("canonical"), mat,
                false, thumb_vertices);

        d->mesh->primitive_type = GL_POINTS;
        d->mesh->geometry->update_meshes();
      }
      {
        reclib::opengl::Material mat("index_points");
        mat->vec4_map["color"] = vec4(0, 1, 0, 1);

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "index_points", reclib::opengl::Shader::find("canonical"), mat,
                false, index_vertices);

        d->mesh->primitive_type = GL_POINTS;
        d->mesh->geometry->update_meshes();
      }
      {
        reclib::opengl::Material mat("middle_points");
        mat->vec4_map["color"] = vec4(0, 1, 1, 1);

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "middle_points", reclib::opengl::Shader::find("canonical"), mat,
                false, middle_vertices);

        d->mesh->primitive_type = GL_POINTS;
        d->mesh->geometry->update_meshes();
      }
      {
        reclib::opengl::Material mat("ring_points");
        mat->vec4_map["color"] = vec4(0, 0, 1, 1);

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "ring_points", reclib::opengl::Shader::find("canonical"), mat,
                false, ring_vertices);

        d->mesh->primitive_type = GL_POINTS;
        d->mesh->geometry->update_meshes();
      }
      {
        reclib::opengl::Material mat("pinky_points");
        mat->vec4_map["color"] = vec4(1, 0, 1, 1);

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "pinky_points", reclib::opengl::Shader::find("canonical"), mat,
                false, pinky_vertices);

        d->mesh->primitive_type = GL_POINTS;
        d->mesh->geometry->update_meshes();
      }
    }
  }

  if (transform_mano) {
    mano.params = mano_copy.params;
    mano.update(false, true);
  }

  return colors;
}

// template instantiation
template std::vector<vec4> mano_canonical_space_colors_hsv(
    reclib::models::ModelInstance<reclib::models::MANOConfig>& mano,
    bool transform_mano, bool convert_to_rgb);
template std::vector<vec4> mano_canonical_space_colors_hsv(
    reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& mano,
    bool transform_mano, bool convert_to_rgb);
template std::vector<vec4> mano_canonical_space_colors_hsv(
    reclib::models::ModelInstance<reclib::models::MANOConfigAnglePCA>& mano,
    bool transform_mano, bool convert_to_rgb);

template <typename T>
std::pair<std::vector<vec4>, std::vector<vec4>>
mano_canonical_joint_mean_colors(reclib::models::ModelInstance<T>& mano,
                                 bool transform_mano) {
  reclib::models::ModelInstance<T> mano_copy(mano.model);
  mano_copy.add_pose_mean_ = mano.add_pose_mean_;
  mano_copy.params.setConstant(0);
  mano_copy.update(false, true);

  mat4 to_canonical = mano_canonical_space(mano_copy);
  // mano_copy is now transformed to the pose in the canonical space!

  std::vector<vec4> colors;
  std::vector<std::vector<vec4>> jointwise_colors(T::n_joints());
  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    unsigned int index = 0;
    Eigen::VectorXf weights = mano_copy.model.weights.row(i).toDense();
    weights.maxCoeff(&index);

    vec3 v = mano_copy.verts().row(i).transpose();
    v = (to_canonical * v.homogeneous()).hnormalized();

    if (mano_copy.model.hand_type == reclib::models::HandType::left) {
      jointwise_colors[index].push_back(vec4(v.x(), v.y(), v.z(), 0.5));
    } else {
      jointwise_colors[index].push_back(vec4(v.x(), v.y(), v.z(), 1.0));
    }
  }
  std::vector<vec4> jointwise_mean_colors(T::n_joints());
  for (unsigned int i = 0; i < jointwise_mean_colors.size(); i++) {
    vec4 mean_color(0, 0, 0, 0);
    for (unsigned int j = 0; j < jointwise_colors[i].size(); j++) {
      mean_color += jointwise_colors[i][j];
    }
    mean_color /= jointwise_colors[i].size();
    jointwise_mean_colors[i] = (mean_color);
  }

  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    unsigned int index = 0;
    Eigen::VectorXf weights = mano_copy.model.weights.row(i).toDense();
    weights.maxCoeff(&index);
    colors.push_back(jointwise_mean_colors[index]);
  }
  return std::make_pair(colors, jointwise_mean_colors);
}

// template instantiation
template std::pair<std::vector<vec4>, std::vector<vec4>>
mano_canonical_joint_mean_colors(
    reclib::models::ModelInstance<reclib::models::MANOConfig>& mano,
    bool transform_mano);
template std::pair<std::vector<vec4>, std::vector<vec4>>
mano_canonical_joint_mean_colors(
    reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& mano,
    bool transform_mano);
template std::pair<std::vector<vec4>, std::vector<vec4>>
mano_canonical_joint_mean_colors(
    reclib::models::ModelInstance<reclib::models::MANOConfigAnglePCA>& mano,
    bool transform_mano);

template <typename T>
mat4 mano_canonical_space(reclib::models::ModelInstance<T>& mano_copy) {
  const unsigned int index = MANOConfigExtra::INDEX_IDX;
  const unsigned int middle = MANOConfigExtra::MIDDLE_IDX;
  const unsigned int pinky = MANOConfigExtra::PINKY_IDX;
  const unsigned int ring = MANOConfigExtra::RING_IDX;
  const unsigned int thumb = MANOConfigExtra::THUMB_IDX;
  const unsigned int thickness = 9;

  reclib::models::ModelInstance<T> mano_copy_copy(mano_copy.model);
  mano_copy_copy.params.setConstant(0);
  mano_copy_copy.update(false, true);

  int sign = 1;
  mat4 model = mat4::Identity();
  if (mano_copy.model.hand_type == reclib::models::HandType::left) {
    sign = -1;
    model(0, 0) = -1;
  }

  mano_copy.update(false, true);

  vec3 bbmin(std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max());
  vec3 bbmax(std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min());

  for (unsigned int i = 0; i < mano_copy.verts().rows(); i++) {
    vec3 v = mano_copy.verts().row(i).transpose();
    vec3 v_trans = (model * v.homogeneous()).hnormalized();
    bbmin = v_trans.cwiseMin(bbmin);
    bbmax = v_trans.cwiseMax(bbmax);
  }

  vec3 whd = bbmax - bbmin;

  float max = 0;
  unsigned int ax = 0;
  float second_max = 0;
  for (unsigned int i = 0; i < 3; i++) {
    if (whd[i] > max) {
      second_max = max;
      max = whd[i];
      ax = i;
    } else if (second_max == 0 || whd[i] > second_max) {
      second_max = whd[i];
    }
  }
  vec3 axis = vec3::Zero();
  axis[ax] = 1;

  mat4 to_canonical = mat4::Identity();

  vec3 scale = vec3::Ones().cwiseQuotient(whd);
  to_canonical = reclib::scale(scale);
  to_canonical *= reclib::translate(vec3(-bbmin));

  if (0 && !reclib::opengl::Drawelement::valid("test_bb")) {
    reclib::opengl::Material mat("test_mat");
    mat->vec4_map["color"] = vec4(1, 0, 0, 1);
    reclib::opengl::Shader shad("test_bb", "MVP.vs", "color4Uniform.fs");
    reclib::opengl::BoundingBox bb("test_bb", bbmin, bbmax, mat);
    reclib::opengl::Drawelement d("test_bb", shad, bb);
  }

  return to_canonical * model;
}

// template instantiation
template mat4 mano_canonical_space(
    reclib::models::ModelInstance<reclib::models::MANOConfig>& mano_copy);
template mat4 mano_canonical_space(
    reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& mano_copy);
template mat4 mano_canonical_space(
    reclib::models::ModelInstance<reclib::models::MANOConfigAnglePCA>&
        mano_copy);

Eigen::Vector3f auto_color(size_t color_index) {
  static const Eigen::Vector3f palette[] = {
      Eigen::Vector3f{1.f, 0.2f, 0.3f},   Eigen::Vector3f{0.3f, 0.2f, 1.f},
      Eigen::Vector3f{0.3f, 1.2f, 0.2f},  Eigen::Vector3f{0.8f, 0.2f, 1.f},
      Eigen::Vector3f{0.7f, 0.7f, 0.7f},  Eigen::Vector3f{1.f, 0.45f, 0.f},
      Eigen::Vector3f{1.f, 0.17f, 0.54f}, Eigen::Vector3f{0.133f, 1.f, 0.37f},
      Eigen::Vector3f{1.f, 0.25, 0.21},   Eigen::Vector3f{1.f, 1.f, 0.25},
      Eigen::Vector3f{0.f, 0.45, 0.9},    Eigen::Vector3f{0.105, 0.522, 1.f},
      Eigen::Vector3f{0.9f, 0.5f, 0.7f},  Eigen::Vector3f{1.f, 0.522, 0.7f},
      Eigen::Vector3f{0.f, 1.0f, 0.8f},   Eigen::Vector3f{0.9f, 0.7f, 0.9f},
  };
  return palette[color_index % (sizeof palette / sizeof palette[0])]
      .normalized();
}

Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> auto_color_table(
    size_t num_colors) {
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> colors(num_colors,
                                                                  3);
  for (size_t i = 0; i < num_colors; ++i) {
    colors.row(i) = auto_color(i).transpose();
  }
  return colors;
}

// -------------------------------------------------------------
// Model
// -------------------------------------------------------------

template <class ModelConfig>
Model<ModelConfig>::Model(Gender gender) {
  load(gender);
  generate_verts2joints();
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::Model(const std::string& path, const std::string& uv_path,
                          Gender gender) {
  load(path, uv_path, gender);
  generate_verts2joints();
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::Model(HandType type) {
  load(type);
  generate_verts2joints();
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::Model(const std::string& path, const std::string& uv_path,
                          HandType type) {
  load(path, uv_path, type);
  generate_verts2joints();
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::~Model() {}

template <class ModelConfig>
void Model<ModelConfig>::load(Gender gender) {
  load(find_data_file(std::string(ModelConfig::default_path_prefix) +
                      gender_to_str(gender) + ".npz"),
       find_data_file(ModelConfig::default_uv_path), gender);
}

template <class ModelConfig>
void Model<ModelConfig>::load(HandType type) {
  load(find_data_file(std::string(ModelConfig::default_path_prefix) +
                      hand_type_to_str(type) + ".npz"),
       find_data_file(ModelConfig::default_uv_path), type);
}

template <class ModelConfig>
void Model<ModelConfig>::load(const std::string& path,
                              const std::string& uv_path, HandType type) {
  hand_type = type;
  return load(path, uv_path, Gender::unknown);
}

template <class ModelConfig>
void Model<ModelConfig>::load(const std::string& path,
                              const std::string& uv_path, Gender new_gender) {
  if (!std::ifstream(path)) {
    std::cerr << "ERROR: Model '" << path
              << "' does not exist, "
                 "did you download the model following instructions in "
                 "reconstruction-lib/data/models/"
                 "README.md?\n";
    return;
  }
  gender = new_gender;
  cnpy::npz_t npz = cnpy::npz_load(path);

  // Load kintree
  children.resize(n_joints());
  for (size_t i = 1; i < n_joints(); ++i) {
    children[ModelConfig::parent[i]].push_back(i);
  }

  // Load base template
  const auto& verts_raw = npz.at("v_template");
  assert_shape(verts_raw, {n_verts(), 3});
  verts = reclib::python::load_float_matrix_rm(verts_raw, n_verts(), 3);
  verts_load = verts;

  // Load triangle mesh
  const auto& faces_raw = npz.at("f");
  assert_shape(faces_raw, {n_faces(), 3});
  faces = reclib::python::load_uint_matrix_rm(faces_raw, n_faces(), 3);

  // Load joint regressor
  const auto& jreg_raw = npz.at("J_regressor");
  assert_shape(jreg_raw, {n_joints(), n_verts()});
  joint_reg.resize(n_joints(), n_verts());
  joint_reg =
      reclib::python::load_float_matrix_rm(jreg_raw, n_joints(), n_verts())
          .sparseView();
  joints = joint_reg * verts;
  joint_reg.makeCompressed();

  // Load LBS weights
  const auto& wt_raw = npz.at("weights");
  assert_shape(wt_raw, {n_verts(), n_joints()});
  weights.resize(n_verts(), n_joints());
  weights = reclib::python::load_float_matrix_rm(wt_raw, n_verts(), n_joints())
                .sparseView();
  weights.makeCompressed();

  blend_shapes.resize(3 * n_verts(), n_blend_shapes());
  // Load shape-dep blend shapes
  const auto& sb_raw = npz.at("shapedirs");
  assert_shape(sb_raw, {n_verts(), 3, n_shape_blends()});
  blend_shapes.template leftCols<n_shape_blends()>().noalias() =
      reclib::python::load_float_matrix_rm(sb_raw, 3 * n_verts(),
                                           n_shape_blends());

  // Load pose-dep blend shapes
  const auto& pb_raw = npz.at("posedirs");
  assert_shape(pb_raw, {n_verts(), 3, n_pose_blends()});
  blend_shapes.template rightCols<n_pose_blends()>().noalias() =
      reclib::python::load_float_matrix_rm(pb_raw, 3 * n_verts(),
                                           n_pose_blends());

  if (npz.count("hands_meanl") && npz.count("hands_meanr")) {
    // Model has hand PCA (e.g. SMPLXpca), load hand PCA
    const auto& hml_raw = npz.at("hands_meanl");
    const auto& hmr_raw = npz.at("hands_meanr");
    const auto& hcl_raw = npz.at("hands_componentsl");
    const auto& hcr_raw = npz.at("hands_componentsr");

    assert_shape(hml_raw, {ANY_SHAPE});
    assert_shape(hmr_raw, {hml_raw.shape[0]});

    size_t n_hand_params = hml_raw.shape[0];
    _RECLIB_ASSERT_EQ(n_hand_params, n_hand_pca_joints() * 3);

    assert_shape(hcl_raw, {n_hand_params, n_hand_params});
    assert_shape(hcr_raw, {n_hand_params, n_hand_params});

    if (hand_type == reclib::models::HandType::left) {
      hand_mean =
          reclib::python::load_float_matrix_rm(hml_raw, n_hand_params, 1);

      hand_comps = reclib::python::load_float_matrix_rm(hcl_raw, n_hand_params,
                                                        n_hand_params)
                       .topRows(n_hand_pca())
                       .transpose();
    } else {
      hand_mean =
          reclib::python::load_float_matrix_rm(hmr_raw, n_hand_params, 1);

      hand_comps = reclib::python::load_float_matrix_rm(hcr_raw, n_hand_params,
                                                        n_hand_params)
                       .topRows(n_hand_pca())
                       .transpose();
    }

  } else if (npz.count("hands_mean")) {
    assert(hand_type != HandType::unknown);
    // Model has hand PCA, e.g. MANO
    const auto& hm_raw = npz.at("hands_mean");
    const auto& hc_raw = npz.at("hands_components");

    assert_shape(hm_raw, {ANY_SHAPE});

    size_t n_hand_params = hm_raw.shape[0];

    if (n_hand_pca()) _RECLIB_ASSERT_EQ(n_hand_params, n_hand_pca_joints() * 3);

    assert_shape(hc_raw, {n_hand_params, n_hand_params});

    if (hand_type == HandType::left) {
      hand_mean =
          reclib::python::load_float_matrix_rm(hm_raw, n_hand_params, 1);
      hand_comps = reclib::python::load_float_matrix_rm(hc_raw, n_hand_params,
                                                        n_hand_params)
                       .topRows(n_hand_pca())
                       .transpose();
    } else {
      hand_mean =
          reclib::python::load_float_matrix_rm(hm_raw, n_hand_params, 1);
      auto tmp = reclib::python::load_float_matrix_rm(hc_raw, n_hand_params,
                                                      n_hand_params);
      hand_comps = tmp.topRows(n_hand_pca()).transpose();
    }
  }

  // Maybe load UV (UV mapping WIP)
  if (uv_path.size()) {
    std::ifstream ifs(uv_path);
    ifs >> _n_uv_verts;
    if (_n_uv_verts) {
      if (ifs) {
        // _SMPLX_ASSERT_LE(n_verts(), _n_uv_verts);
        // Load the uv data
        uv.resize(_n_uv_verts, 2);
        for (size_t i = 0; i < _n_uv_verts; ++i) ifs >> uv(i, 0) >> uv(i, 1);
        assert(ifs);
        uv_faces.resize(n_faces(), 3);
        for (size_t i = 0; i < n_faces(); ++i) {
          assert(ifs);
          for (size_t j = 0; j < 3; ++j) {
            ifs >> uv_faces(i, j);
            // Make indices 0-based
            --uv_faces(i, j);
            _RECLIB_ASSERT_LT(uv_faces(i, j), _n_uv_verts);
          }
        }
      }
    }
  }
}

template <class ModelConfig>
void Model<ModelConfig>::generate_verts2joints() {
  for (unsigned int i = 0; i < verts.rows(); i++) {
    vec3 v = verts.row(i).transpose();

    int major_joint = -1;
    weights.row(i).toDense().maxCoeff(&major_joint);

    verts2joints[i] = major_joint;
  }
}

template <class ModelConfig>
void Model<ModelConfig>::generate_vertex_ring() {
  vertex_ring = -1 * Eigen::MatrixXi::Ones(ModelConfig::n_verts(), 6);

  auto insert_fun = [&](int src, int dest) {
    for (unsigned int j = 0; j < 6; j++) {
      // free place -> store
      if (vertex_ring(src, j) < 0) {
        vertex_ring(src, j) = dest;
        break;
      }
      // neighbor already included
      if (vertex_ring(src, j) == dest) break;
    }
  };

  for (unsigned int f = 0; f < faces.rows(); f++) {
    int i0 = faces(f, 0);
    int i1 = faces(f, 1);
    int i2 = faces(f, 2);

    insert_fun(i0, i1);
    insert_fun(i0, i2);

    insert_fun(i1, i0);
    insert_fun(i1, i2);

    insert_fun(i2, i1);
    insert_fun(i2, i0);
  }
}

template <class ModelConfig>
void Model<ModelConfig>::set_deformations(
    const Eigen::Ref<
        const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>& d) {
  verts.noalias() = verts_load + d;
}

template <class ModelConfig>
void Model<ModelConfig>::set_template(
    const Eigen::Ref<
        const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>& t) {
  verts.noalias() = t;
}

// Instantiations
template class Model<SMPLConfig>;
template class Model<MANOConfig>;
template class Model<MANOConfigPCA>;
template class Model<MANOConfigPCAGeneric<45>>;
template class Model<MANOConfigPCAGeneric<23>>;
template class Model<MANOConfigAnglePCA>;
template class Model<MANOConfigAnglePCAGeneric<45>>;
template class Model<MANOConfigAnglePCAGeneric<23>>;

// -------------------------------------------------------------
// ModelInstance
// -------------------------------------------------------------

template <class ModelConfig>
ModelInstance<ModelConfig>::ModelInstance(const Model<ModelConfig>& model,
                                          bool set_zero)
    : model(model),
      use_anatomic_pose_(false),
      use_anatomic_pca_(false),
      params(model.n_params()),
      has_opengl_mesh(false) {
  if (set_zero) this->set_zero();

  if (model.hand_type != HandType::unknown) {
    anatomic_params =
        Eigen::Matrix<float, Eigen::Dynamic, 1>(MANOConfigExtra::anatomic_dofs);
    anatomic_params.setZero();
  }
  if (ModelConfig::n_hand_pca() > 0) {
    add_pose_mean_ = true;
  } else {
    add_pose_mean_ = false;
  }

  // Point cloud after applying shape keys but before lbs (num points,
  // 3)
  _verts_shaped.resize(model.n_verts(), 3);

  // Joints after applying shape keys but before lbs (num joints, 3)
  _joints_shaped.resize(model.n_joints(), 3);

  // Final deformed point cloud
  _verts.resize(model.n_verts(), 3);

  // Affine joint transformation, as 3x4 matrices stacked horizontally
  // (bottom row omitted) NOTE: col major
  _joint_transforms.resize(model.n_joints(), 12);
  _local_joint_transforms.resize(model.n_joints(), 12);
}

template <class ModelConfig>
ModelInstance<ModelConfig>::~ModelInstance() {
  if (has_opengl_mesh) {
    gl_instance->destroy_handles();
    gl_instance.free();
    gl_joint_lines->destroy_handles();
    gl_joint_lines.free();
    gl_joints->destroy_handles();
    gl_joints.free();
  }
}

template <class ModelConfig>
const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>&
ModelInstance<ModelConfig>::verts() const {
  return _verts;
}

template <class ModelConfig>
Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>&
ModelInstance<ModelConfig>::verts() {
  return _verts;
}

template <class ModelConfig>
const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>&
ModelInstance<ModelConfig>::verts_shaped() const {
  return _verts_shaped;
}

template <class ModelConfig>
const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>&
ModelInstance<ModelConfig>::joints() const {
  return _joints;
}

template <class ModelConfig>
const Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor>&
ModelInstance<ModelConfig>::joint_transforms() const {
  return _joint_transforms;
}

template <class ModelConfig>
Eigen::Vector<float, 48> ModelInstance<ModelConfig>::apose2pose(
    const ModelInstance<ModelConfig>& inst) {
  Eigen::Vector<float, 48> P;

  using TransformMap =
      Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>;
  P.setZero();
  if (inst.use_anatomic_pose_) {
    P.template head<3>() = inst.apose().template head<3>();
    for (unsigned int i = 1; i < ModelConfig::n_joints(); ++i) {
      TransformMap transform(inst._local_joint_transforms.row(i).data());
      mat3 T = transform.block<3, 3>(0, 0);
      Eigen::AngleAxisf aa(T);
      P.template segment<3>(i * 3) = aa.angle() * aa.axis();
    }
  }
  return P;
}

template <class ModelConfig>
Eigen::Vector<float, MANOConfigExtra::anatomic_dofs>
ModelInstance<ModelConfig>::pose2apose(const ModelInstance<ModelConfig>& inst,
                                       bool anatomical_limits) {
  Eigen::Vector<float, MANOConfigExtra::anatomic_dofs> P;

  using TransformMap =
      Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>;
  P.setZero();
  P.template head<3>() = inst.pose().template head<3>();
  int c = 3;
  for (unsigned int i = 1; i < ModelConfig::n_joints(); ++i) {
    mat3 T;

    TransformMap transform(inst._local_joint_transforms.row(i).data());
    T = transform.block<3, 3>(0, 0);

    const auto p = ModelConfig::parent[i];

    vec3 twist;
    if (i % 3 > 0) {
      unsigned int next = i + 1;
      twist = (inst._joints_shaped.row(next) - inst._joints_shaped.row(i))
                  .normalized();

    } else {
      int idx = ((i) / 3) - 1;
      int next = reclib::models::MANOConfigExtra::tips[idx];
      twist = (inst._verts_shaped.row(next) - inst._joints_shaped.row(i))
                  .normalized();
    }

    vec3 up_vec;
    if (i < 13) {
      up_vec = vec3(0, 1, 0);
    } else {
      if (inst.model.hand_type == reclib::models::HandType::right) {
        up_vec = vec3(1, 1, 1);
      } else {
        up_vec = vec3(-1, 1, 1);
      }
    }

    vec3 bend = twist.cross(up_vec).normalized();
    vec3 splay = bend.cross(twist).normalized();

    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> cross_matrix =
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Identity();
    cross_matrix.block<1, 3>(0, 0) = bend;
    cross_matrix.block<1, 3>(1, 0) = splay;
    cross_matrix.block<1, 3>(2, 0) = twist;

    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> cross_matrix_inv =
        cross_matrix.inverse();

    T = cross_matrix.block<3, 3>(0, 0) * T * cross_matrix_inv.block<3, 3>(0, 0);

    Eigen::AngleAxisf aa(T);
    Eigen::Quaternionf quat(T);
    vec3 AA = aa.angle() * aa.axis();

    int dof = MANOConfigExtra::dof_per_anatomic_joint[i];

    for (unsigned int j = 0; j < dof; j++) {
      if (anatomical_limits) {
        vec2 limits = MANOConfigExtra::anatomic_limits[c + j];
        P[c + j] = fmin(fmax(AA[j], limits[0]), limits[1]);
      } else {
        P[c + j] = AA[j];
      }
    }

    // if (dof == 3) {
    //   vec3 AA_A = P.segment<3>(c);
    //   std::cout << "i: " << i << " AA: " << AA << " ANATOMIC: " << AA_A
    //             << std::endl;
    // } else if (dof == 2) {
    //   vec2 AA_A = P.segment<2>(c);
    //   std::cout << "i: " << i << " AA: " << AA << " ANATOMIC: " << AA_A
    //             << std::endl;
    // } else {
    //   float AA_A = P[c];
    //   std::cout << "i: " << i << " AA: " << AA << " ANATOMIC: " << AA_A
    //             << std::endl;
    // }

    c += dof;
  }

  return P;
}

template <class ModelConfig>
const Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor>&
ModelInstance<ModelConfig>::vert_transforms() const {
  if (_vert_transforms.rows() == 0) {
    _vert_transforms.noalias() = model.weights * _joint_transforms;
  }
  return _vert_transforms;
}

// Main LBS routine
template <class ModelConfig>
void ModelInstance<ModelConfig>::update(bool force_cpu,
                                        bool enable_pose_blendshapes,
                                        bool anatomical_limits) {
  // Will store full pose params (angle-axis), including hand
  Eigen::Matrix<float, Eigen::Dynamic, 1> full_pose(3 * model.n_joints());

  // First rows: Shape params
  // Succeeding rows: linear joint transformations as flattened 3x3
  // rotation matrices rowmajor, only for blend shapes
  Eigen::Matrix<float, Eigen::Dynamic, 1> blendshape_params(
      model.n_blend_shapes());

  using TransformMap = Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>;
  using TransformTransposedMap = Eigen::Map<Eigen::Matrix<float, 4, 3>>;
  using RotationMap = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>;

  if (use_anatomic_pose_) {
    full_pose.setZero();

    Eigen::Vector<float, reclib::models::MANOConfigExtra::anatomic_dofs>
        full_apose = apose();
    if (use_anatomic_pca_) {
      unsigned int pcas = hand_pca().rows();
      full_apose.tail<reclib::models::MANOConfigExtra::anatomic_dofs - 3>() +=
          model.hand_comps.leftCols(pcas) * hand_pca() + model.hand_mean;
    }

    int c = 0;
    for (unsigned int i = 0; i < model.n_explicit_joints(); i++) {
      int dof = MANOConfigExtra::dof_per_anatomic_joint[i];

      for (unsigned int j = 0; j < dof; j++) {
        vec2 limits = MANOConfigExtra::anatomic_limits[c + j];

        if (anatomical_limits) {
          full_pose[3 * i + j] +=
              fmin(fmax(full_apose[c + j], limits[0]), limits[1]);
        } else {
          full_pose[3 * i + j] += full_apose[c + j];
        }
      }
      c += dof;
    }

  } else {
    // Copy body pose onto full pose
    // the pose consists of all joints in axis-angle representation
    // XXX: beware: the pose of joint i does not influence
    // joint i itself, but instead its children!!
    full_pose.head(3 * model.n_explicit_joints()).noalias() = pose();

    const unsigned int explicit_without_pca =
        ModelConfig::n_explicit_joints() - ModelConfig::n_duplicate_joints();
    if (model.n_hand_pca_joints() > 0) {
      full_pose
          .segment<3 * ModelConfig::n_hand_pca_joints()>(3 *
                                                         explicit_without_pca)
          .noalias() = model.hand_comps * hand_pca();

      if (ModelConfig::n_duplicate_joints() > 0) {
        full_pose.segment<3 * ModelConfig::n_hand_pca_joints()>(
            3 * explicit_without_pca) +=
            pose().segment(3 * explicit_without_pca,
                           3 * model.n_hand_pca_joints());
      }
    }
    if (add_pose_mean_) {
      // add mean pose to all joints except the root joint
      full_pose.segment<3 * (ModelConfig::n_joints() - 1)>(3).noalias() +=
          model.hand_mean;
    }
  }

  // Copy shape params to FIRST ROWS of blendshape params
  blendshape_params.head<ModelConfig::n_shape_blends()>() = shape();

  // Convert angle-axis to rotation matrix using rodrigues
  // First load joint_transforms into 3x4 matrix map
  // then fill with rodriguez converted axis-angle
  // TLDR: map axis-angle -> 3x4 matrix
  TransformMap(_local_joint_transforms.topRows<1>().data())
      .template leftCols<3>()
      .noalias() = rodrigues<float>(full_pose.head<3>());
  for (size_t i = 1; i < model.n_joints(); ++i) {
    // TLDR: map axis-angle -> 3x4 matrix
    TransformMap joint_trans(_local_joint_transforms.row(i).data());
    joint_trans.template leftCols<3>().noalias() =
        rodrigues<float>(full_pose.segment<3>(3 * i));

    // Store 3x3 joint-wise rotation matrix in blendshape_params
    // XXX: first rows are not overwritten! The shape parameters beta
    // are inside the first forws of blendshape_params
    RotationMap mp(blendshape_params.data() + 9 * i +
                   (model.n_shape_blends() - 9));
    mp.noalias() = joint_trans.template leftCols<3>();
    mp.diagonal().array() -= 1.f;
  }

  // Apply blend shapes
  {
    // load map of deformed vertices (only shape applied)
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> verts_shaped_flat(
        _verts_shaped.data(), model.n_verts() * 3);
    // load vertices in rest pose
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> verts_init_flat(
        model.verts.data(), model.n_verts() * 3);

    // Add shape blend shapes
    // model.blend_shapes (3*#verts, #shape blends)
    // (first rows of) blendshape_params: (#shape blends)
    // apply equation:
    // sum_i beta_i * S_i with S_i
    verts_shaped_flat.noalias() =
        verts_init_flat +
        model.blend_shapes.template leftCols<ModelConfig::n_shape_blends()>() *
            blendshape_params.head<ModelConfig::n_shape_blends()>();
  }

  // Apply joint regressor
  _joints_shaped = model.joint_reg * _verts_shaped;

  if (use_anatomic_pose_) {
    for (unsigned int i = 1; i < ModelConfig::n_joints(); ++i) {
      TransformMap transform(_local_joint_transforms.row(i).data());

      const auto p = ModelConfig::parent[i];

      vec3 twist;
      if (i % 3 > 0) {
        unsigned int next = i + 1;
        twist = (_joints_shaped.row(next) - _joints_shaped.row(i)).normalized();

      } else {
        int idx = ((i) / 3) - 1;
        int next = reclib::models::MANOConfigExtra::tips[idx];
        twist = (_verts_shaped.row(next) - _joints_shaped.row(i)).normalized();
      }

      vec3 up_vec;
      if (i < 13) {
        up_vec = vec3(0, 1, 0);
      } else {
        if (model.hand_type == reclib::models::HandType::right) {
          up_vec = vec3(1, 1, 1);
        } else {
          up_vec = vec3(-1, 1, 1);
        }
      }

      vec3 bend = twist.cross(up_vec).normalized();
      vec3 splay = bend.cross(twist).normalized();

      Eigen::Matrix<float, 4, 4, Eigen::RowMajor> cross_matrix =
          Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Identity();
      cross_matrix.block<1, 3>(0, 0) = bend;
      cross_matrix.block<1, 3>(1, 0) = splay;
      cross_matrix.block<1, 3>(2, 0) = twist;

      Eigen::Matrix<float, 4, 4, Eigen::RowMajor> cross_matrix_inv =
          cross_matrix.inverse();

      transform.block<3, 3>(0, 0) = cross_matrix_inv.block<3, 3>(0, 0) *
                                    transform.block<3, 3>(0, 0) *
                                    cross_matrix.block<3, 3>(0, 0);

      RotationMap mp(blendshape_params.data() + 9 * i +
                     (model.n_shape_blends() - 9));
      mp.noalias() = transform.template leftCols<3>();
      mp.diagonal().array() -= 1.f;
    }
  }

  if (enable_pose_blendshapes) {
    // HORRIBLY SLOW, like 95% of the time is spent here yikes
    // Add pose blend shapes

    // apply equation: sum_j (rot_j - rot_rest_pose_j)  * P_j
    // = sum_j rot_j * P_j (since rotation in rest pose is 0)
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> verts_shaped_flat(
        _verts_shaped.data(), model.n_verts() * 3);
    verts_shaped_flat +=
        model.blend_shapes.template rightCols<ModelConfig::n_pose_blends()>() *
        blendshape_params.tail<ModelConfig::n_pose_blends()>();
  }

  _joint_transforms = _local_joint_transforms;
  // Inputs: trans(), _joints_shaped
  // Outputs: _joints
  // Input/output: _joint_transforms
  //   (input: left 3x3 should be local rotation mat for joint
  //    output: completed joint local space transform rel global)
  _local_to_global();

  // * LBS *
  // Construct a transform for each vertex
  _vert_transforms.noalias() = model.weights * _joint_transforms;
  // _SMPLX_PROFILE(lbs weight computation);

  // Apply affine transform to each vertex and store to output
  // #pragma omp parallel for // Seems to only make it slower??
  for (size_t i = 0; i < model.n_verts(); ++i) {
    TransformTransposedMap transform_tr(_vert_transforms.row(i).data());
    _verts.row(i).noalias() = _verts_shaped.row(i).homogeneous() * transform_tr;
  }

  if (has_opengl_mesh) {
    gl_joints->mesh->geometry->update_meshes();
    gl_joint_lines->mesh->geometry->update_meshes();
    gl_instance->mesh->geometry->update_meshes();
  }
}

template <class ModelConfig>
void ModelInstance<ModelConfig>::_local_to_global() {
  _joints.resize(ModelConfig::n_joints(), 3);
  using TransformMap = Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>;
  using TransformTransposedMap = Eigen::Map<Eigen::Matrix<float, 4, 3>>;
  // Handle root joint transforms
  TransformTransposedMap root_transform_tr(
      _joint_transforms.topRows<1>().data());
  // compute root translation (in bottom row, as is transposed)
  root_transform_tr.bottomRows<1>().noalias() =
      _joints_shaped.topRows<1>() + trans().transpose();
  // apply to root joints
  _joints.topRows<1>().noalias() = root_transform_tr.bottomRows<1>();

  // Complete the affine transforms for all other joint by adding
  // translation components and composing with parent
  for (unsigned int i = 1; i < ModelConfig::n_joints(); ++i) {
    // XXX: beware: the rotation of joint i does not influence
    // joint i itself, but instead its children!!
    TransformMap transform(_joint_transforms.row(i).data());

    const auto p = ModelConfig::parent[i];

    // Set relative translation
    // in case we start at parent node, which is the relative origin
    // (0,0,0) in what direction do we have to move to get to child
    // joint
    transform.rightCols<1>().noalias() =
        (_joints_shaped.row(i) - _joints_shaped.row(p)).transpose();

    // Compose rotation with parent
    // 1. RELATIVE transform: apply 'transform' to rotate around the
    // parent and then translate to get to joint position
    // 2. ABSOLUTE transform: apply 'joint_transforms' of parent to
    // rotate joint position around global center
    mul_affine<float, Eigen::RowMajor>(
        TransformMap(_joint_transforms.row(p).data()), transform);
    // if (i == 1) {
    //   std::cout << "transform: " << transform << std::endl;
    // }

    // Grab the joint position in case the user wants it
    // since the transformation describes the whole translation from
    // the global origin (0,0,0) to the joint position, its translation
    // is the joint's position because the rotation part
    // multiplied with (0,0,0) would be zero anyway
    _joints.row(i).noalias() = transform.rightCols<1>().transpose();
  }

  for (unsigned int i = 0; i < ModelConfig::n_joints(); ++i) {
    TransformTransposedMap transform_tr(_joint_transforms.row(i).data());

    // This is a correction necessary for the vertices:
    // Let's assume we have a shaped vertex in the rest pose coordinate system
    // Analogously we would have to describe the rotation of that vertex around
    // a joint X through a relative coordinate system where X is the origin. for
    // the joint case we achieved this via the translation JX - JP where JP is
    // the parent joint. If we now multiply the current un-modified
    // transformation with a vertex that is not expressed in relative
    // coordinates the transformation will be wrong. what we would therefore
    // want is a transformation T = [R|t] such as:
    //
    // vertex_transformed = R * (vertex - joint_rest) + t
    //
    // This unwraps to: R * vertex - R * joint_rest + t
    //
    // So what we need to subtract here is the (R * joint_rest) to make the
    // transformation correct, i.a. relative to the coordinate system
    transform_tr.bottomRows<1>().noalias() -=
        _joints_shaped.row(i) * transform_tr.topRows<3>();
  }
}

template <class ModelConfig>
void ModelInstance<ModelConfig>::generate_gl_drawelem(
    reclib::opengl::Shader s, reclib::opengl::Material m) {
  //   bool has_opengl_mesh;
  // reclib::opengl::Drawelement gl_instance;
  // std::vector<reclib::opengl::Drawelement> gl_joints;
  // std::vector<reclib::opengl::Drawelement> gl_joint_lines;

  _RECLIB_ASSERT(!gl_instance.initialized());

  reclib::opengl::Material mat;
  if (!m.initialized() &&
      !reclib::opengl::Material::valid("modelinst_default")) {
    reclib::opengl::Material def("modelinst_default");
    def->vec4_map["color"] = vec4(0.2, 0.2, 0.2, 1);
    def->vec3_map["ambient_color"] = vec3(0.25, 0.2, 0.2).normalized();
    def->vec3_map["diffuse_color"] = vec3(0.25, 0.2, 0.2).normalized();
    def->vec3_map["specular_color"] = vec3(0.25, 0.2, 0.2).normalized();
    mat = def;

  } else if (reclib::opengl::Material::valid("modelinst_default")) {
    mat = reclib::opengl::Material::find("modelinst_default");
  } else {
    mat = m;
  }

  reclib::opengl::Shader shader;
  if (!s.initialized() && !reclib::opengl::Shader::valid("modelinst_default")) {
    reclib::opengl::Shader def("modelinst_default", "MVP_norm_color3.vs",
                               "color3PV.fs");
    shader = def;
  } else if (reclib::opengl::Shader::valid("modelinst_default")) {
    shader = reclib::opengl::Shader::find("modelinst_default");
  } else {
    shader = s;
  }

  unsigned int id = 0;
  for (; id < std::numeric_limits<unsigned int>::max(); id++) {
    if (!reclib::opengl::Drawelement::valid("modelinst_" + std::to_string(id)))
      break;
  }

  gl_model_indices = model.faces;
  reclib::opengl::Drawelement d =
      reclib::opengl::DrawelementImpl::from_geometry_wrapper(
          "modelinst_" + std::to_string(id), shader, mat, true, verts().data(),
          verts().rows(), gl_model_indices.data(), gl_model_indices.rows() * 3);
  gl_instance = d;

  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> per_vertex_colors =
      model.weights * reclib::models::auto_color_table(model.n_joints());

  gl_instance->mesh->geometry->add_attribute_float(
      "color", per_vertex_colors.data(), per_vertex_colors.rows(), 3, false);
  gl_instance->mesh->geometry->update_meshes();

  if (!reclib::opengl::Material::valid("modelinst_joints_material")) {
    reclib::opengl::Material sphere_material("modelinst_joints_material");
    sphere_material->vec4_map["color"] = vec4(1.f, 0.5f, 0.0f, 1.f);
  }
  if (!reclib::opengl::Material::valid("modelinst_joint_lines_material")) {
    reclib::opengl::Material joint_lines_material(
        "modelinst_joint_lines_material");
    joint_lines_material->vec4_map["color"] = vec4(0.5f, 0.7f, 0.8f, 1.f);
  }
  if (!reclib::opengl::Shader::valid("modelinst_joint_shader")) {
    reclib::opengl::Shader joint_shader("modelinst_joint_shader", "MVP.vs",
                                        "color4Uniform.fs");
  }

  reclib::opengl::Shader joint_shader =
      reclib::opengl::Shader::find("modelinst_joint_shader");
  reclib::opengl::Material joint_lines_material =
      reclib::opengl::Material::find("modelinst_joint_lines_material");
  reclib::opengl::Material sphere_material =
      reclib::opengl::Material::find("modelinst_joints_material");

  {
    reclib::opengl::Drawelement d =
        reclib::opengl::DrawelementImpl::from_geometry_wrapper(
            "modelinst_joints_" + std::to_string(id), joint_shader,
            sphere_material, false, (float*)joints().data(), joints().rows());
    d->add_pre_draw_func("size", [&]() { glPointSize(10.f); });
    d->add_post_draw_func("size", [&]() { glPointSize(1.f); });
    d->mesh->primitive_type = GL_POINTS;
    d->mesh->geometry->update_meshes();
    gl_joints = d;
  }

  gl_joint_line_indices.clear();
  for (int i = 0; i < (int)model.n_joints(); ++i) {
    unsigned int parent = model.parent(i);
    gl_joint_line_indices.push_back(i);
    gl_joint_line_indices.push_back(parent);
  }

  {
    reclib::opengl::Drawelement d =
        reclib::opengl::DrawelementImpl::from_geometry_wrapper(
            "modelinst_lines_" + std::to_string(id), joint_shader,
            joint_lines_material, false, (float*)joints().data(),
            joints().rows(), gl_joint_line_indices.data(),
            gl_joint_line_indices.size());
    d->add_pre_draw_func("size", [&]() { glLineWidth(2.f); });
    d->add_post_draw_func("size", [&]() { glLineWidth(1.f); });
    d->mesh->primitive_type = GL_LINES;
    d->mesh->geometry->update_meshes();
    gl_joint_lines = d;
  }

  // for (size_t i = 0; i < model.n_joints(); ++i) {
  //   auto joint_pos = joints().row(i).transpose();

  //   std::string sphere_name =
  //       std::to_string(id) + "_" + sphere->name + "_" + std::to_string(i);
  //   reclib::opengl::Drawelement elem(sphere_name, joint_shader, sphere);
  //   elem->set_model_translation(joint_pos);
  //   gl_joints.push_back(elem);

  //   if (i) {
  //     auto parent_pos = joints().row(model.parent(i)).transpose();

  //     reclib::opengl::Line line(
  //         std::to_string(id) + "_modelinst_joint_line_" +
  //         std::to_string(i), joint_pos, parent_pos, joint_lines_material);
  //     line->geometry->register_mesh(line);
  //     std::string line_name = line->name + "_drawelement";
  //     reclib::opengl::Drawelement elem(line->name + "_drawelement",
  //                                      joint_shader, line);
  //     gl_joint_lines.push_back(elem);
  //   }
  // }

  has_opengl_mesh = true;
}

template <class ModelConfig>
void ModelInstance<ModelConfig>::save_obj(const std::string& path) const {
  const auto& cur_verts = verts();
  if (cur_verts.rows() == 0) return;
  std::ofstream ofs(path);
  ofs << "# Generated by SMPL-X_cpp"
      << "\n";
  ofs << std::fixed << std::setprecision(6) << "o smplx\n";
  for (unsigned int i = 0; i < model.n_verts(); ++i) {
    ofs << "v " << cur_verts(i, 0) << " " << cur_verts(i, 1) << " "
        << cur_verts(i, 2) << "\n";
  }
  ofs << "s 1\n";
  for (unsigned int i = 0; i < model.n_faces(); ++i) {
    ofs << "f " << model.faces(i, 0) + 1 << " " << model.faces(i, 1) + 1 << " "
        << model.faces(i, 2) + 1 << "\n";
  }
  ofs.close();
}

// Instantiation
template class ModelInstance<SMPLConfig>;
template class ModelInstance<MANOConfig>;
template class ModelInstance<MANOConfigPCA>;
template class ModelInstance<MANOConfigAnglePCA>;
template class ModelInstance<MANOConfigPCAGeneric<45>>;
template class ModelInstance<MANOConfigPCAGeneric<23>>;
template class ModelInstance<MANOConfigAnglePCAGeneric<45>>;
template class ModelInstance<MANOConfigAnglePCAGeneric<23>>;

}  // namespace models
}  // namespace reclib

#endif  //__unix__