
#if __unix__
#ifndef RECLIB_MODEL_MODEL_CONFIG_H
#define RECLIB_MODEL_MODEL_CONFIG_H

#include <cnpy.h>

#include "reclib/data_types.h"

namespace reclib {

// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/sxyu/smplxpp
// Copyright 2019 Alex Yu

// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

// ---------------------------------------------------------------------------

namespace models {

enum class _API Gender { unknown, neutral, male, female };
enum class _API HandType { unknown, left, right };
const char* gender_to_str(Gender gender);
const char* hand_type_to_str(HandType type);

template <class Derived>
struct ModelConfigBase {
  static constexpr size_t n_joints() {
    return Derived::n_explicit_joints() +
           Derived::n_hand_pca_joints() * Derived::n_hands() -
           Derived::n_duplicate_joints();
  }
  static constexpr size_t n_params() {
    return 3 + Derived::n_explicit_joints() * 3 +
           Derived::n_hand_pca() * Derived::n_hands() +
           Derived::n_shape_blends();
  }
  static constexpr size_t n_pose_blends() { return 9 * (n_joints() - 1); }
  static constexpr size_t n_blend_shapes() {
    return Derived::n_shape_blends() + n_pose_blends();
  }
  static constexpr size_t n_hand_pca_joints() { return 0; }
  static constexpr size_t n_hand_pca() { return 0; }
  static constexpr size_t n_hands() { return 0; }
};

void assert_shape(const cnpy::NpyArray& m, std::initializer_list<size_t> shape);
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
load_float_matrix(const cnpy::NpyArray& raw, size_t r, size_t c);
Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
load_uint_matrix(const cnpy::NpyArray& raw, size_t r, size_t c);
std::string find_data_file(const std::string& data_path);
const size_t ANY_SHAPE = (size_t)-1;

// Classic SMPL model
struct SMPLConfig : public ModelConfigBase<SMPLConfig> {
  static constexpr size_t n_verts() { return 6890; }
  static constexpr size_t n_faces() { return 13776; }
  static constexpr size_t n_explicit_joints() { return 24; }
  static constexpr size_t n_duplicate_joints() { return 0; }
  static constexpr size_t n_shape_blends() { return 10; }
  static constexpr int tips[] = {};
  static constexpr size_t parent[] = {0,  0,  0,  0,  1,  2,  3,  4,
                                      5,  6,  7,  8,  9,  9,  9,  12,
                                      13, 14, 16, 17, 18, 19, 20, 21};
  static constexpr size_t n_hands() { return 2; }
  static constexpr const char* joint_name[] = {
      "pelvis",        "left_hip",       "right_hip",    "spine1",
      "left_knee",     "right_knee",     "spine2",       "left_ankle",
      "right_ankle",   "spine3",         "left_foot",    "right_foot",
      "neck",          "left_collar",    "right_collar", "head",
      "left_shoulder", "right_shoulder", "left_elbow",   "right_elbow",
      "left_wrist",    "right_wrist",    "left_hand",    "right_hand"};
  static constexpr const char* model_name = "SMPL";
  static constexpr const char* default_path_prefix = "models/smpl/SMPL_";
  static constexpr const char* default_uv_path = "models/smpl/uv.txt";
};

struct MANOConfigExtra {
  static const unsigned int INDEX_IDX = 1;
  static const unsigned int MIDDLE_IDX = 4;
  static const unsigned int PINKY_IDX = 7;
  static const unsigned int RING_IDX = 10;
  static const unsigned int THUMB_IDX = 13;
  // left: 445
  // right: 444
  // order: index, middle, pinky, ring, thumb

  // original index: 317
  // original ring: 556
  // original pinky: 673
  static constexpr int tips[] = {320, 444, 672, 555, 745};
  static constexpr int anatomic_dofs = 26;
  static constexpr int dof_per_anatomic_joint[] = {3, 2, 1, 1, 2, 1, 1, 2,
                                                   1, 1, 2, 1, 1, 3, 3, 1};
  static const vec2 anatomic_limits[];
};

// Classic MANO model
struct MANOConfig : public ModelConfigBase<MANOConfig> {
  static constexpr size_t n_verts() { return 778; }
  static constexpr size_t n_faces() { return 1538; }
  static constexpr size_t n_explicit_joints() { return 16; }
  static constexpr size_t n_duplicate_joints() { return 0; }
  static constexpr size_t n_shape_blends() { return 10; }

  static constexpr size_t parent[] = {0, 0, 1, 2,  0,  4, 5,  0,
                                      7, 8, 0, 10, 11, 0, 13, 14};
  static constexpr size_t n_hands() { return 1; }
  static constexpr const char* joint_name[] = {
      "wrist",   "index1", "index2", "index3", "middle1", "middle2",
      "middle3", "pinky1", "pinky2", "pinky3", "ring1",   "ring2",
      "ring3",   "thumb1", "thumb2", "thumb3"};
  static constexpr const char* model_name = "MANO";
  static constexpr const char* default_path_prefix = "models/mano/MANO_";
  static constexpr const char* default_uv_path =
      "models/mano/uv.txt";  // not available
};

// MANO model with PCA pose parameters

template <int PCA = 6>
struct MANOConfigPCAGeneric
    : public ModelConfigBase<MANOConfigPCAGeneric<PCA>> {
  static const unsigned int INDEX_IDX = 1;
  static const unsigned int MIDDLE_IDX = 4;
  static const unsigned int PINKY_IDX = 7;
  static const unsigned int RING_IDX = 10;
  static const unsigned int THUMB_IDX = 13;

  static constexpr size_t n_verts() { return 778; }
  static constexpr size_t n_faces() { return 1538; }
  static constexpr size_t n_explicit_joints() { return 1; }
  static constexpr size_t n_duplicate_joints() { return 0; }
  static constexpr size_t n_shape_blends() { return 10; }
  static constexpr int tips[] = {317, 444, 673, 556, 745};
  static constexpr size_t parent[] = {0, 0, 1, 2,  0,  4, 5,  0,
                                      7, 8, 0, 10, 11, 0, 13, 14};
  static constexpr size_t n_hand_pca_joints() { return 15; }
  static constexpr size_t n_hand_pca() { return PCA; }
  static constexpr size_t n_hands() { return 1; }
  static constexpr const char* joint_name[] = {
      "wrist",   "index1", "index2", "index3", "middle1", "middle2",
      "middle3", "pinky1", "pinky2", "pinky3", "ring1",   "ring2",
      "ring3",   "thumb1", "thumb2", "thumb3"};
  static constexpr const char* model_name = "MANO";
  static constexpr const char* default_path_prefix = "models/mano/MANO_";
  static constexpr const char* default_uv_path =
      "models/mano/uv.txt";  // not available
};
using MANOConfigPCA = MANOConfigPCAGeneric<>;

// MANO model with PCA pose parameters
template <int PCA = 6>
struct MANOConfigAnglePCAGeneric
    : public ModelConfigBase<MANOConfigAnglePCAGeneric<PCA>> {
  static const unsigned int INDEX_IDX = 1;
  static const unsigned int MIDDLE_IDX = 4;
  static const unsigned int PINKY_IDX = 7;
  static const unsigned int RING_IDX = 10;
  static const unsigned int THUMB_IDX = 13;

  static constexpr size_t n_verts() { return 778; }
  static constexpr size_t n_faces() { return 1538; }
  static constexpr size_t n_explicit_joints() { return 16; }
  static constexpr size_t n_duplicate_joints() { return 15; }
  static constexpr size_t n_shape_blends() { return 10; }
  static constexpr int tips[] = {317, 444, 673, 556, 745};
  static constexpr size_t parent[] = {0, 0, 1, 2,  0,  4, 5,  0,
                                      7, 8, 0, 10, 11, 0, 13, 14};
  static constexpr size_t n_hand_pca_joints() { return 15; }
  static constexpr size_t n_hand_pca() { return PCA; }
  static constexpr size_t n_hands() { return 1; }
  static constexpr const char* joint_name[] = {
      "wrist",   "index1", "index2", "index3", "middle1", "middle2",
      "middle3", "pinky1", "pinky2", "pinky3", "ring1",   "ring2",
      "ring3",   "thumb1", "thumb2", "thumb3"};
  static constexpr const char* model_name = "MANO";
  static constexpr const char* default_path_prefix = "models/mano/MANO_";
  static constexpr const char* default_uv_path =
      "models/mano/uv.txt";  // not available
};
using MANOConfigAnglePCA = MANOConfigAnglePCAGeneric<>;

}  // namespace models
}  // namespace reclib

#endif
#endif  // __unix__