#if __unix__

#ifndef RECLIB_MODELS_SMPL_TORCH
#define RECLIB_MODELS_SMPL_TORCH

#if HAS_DNN_MODULE

#include <ATen/SparseTensorImpl.h>
#include <ATen/TensorIndexing.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/torch.h>

#include <cstdio>
#include <sophus/so3.hpp>

#include "reclib/assert.h"
#include "reclib/data_types.h"
#include "reclib/models/model_config.h"
#include "reclib/opengl/drawelement.h"
#include "reclib/opengl/material.h"
#include "reclib/opengl/shader.h"
#include "reclib/utils/torch_utils.h"

namespace reclib {

namespace modelstorch {
// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/sxyu/smplxpp
// Copyright 2019 Alex Yu

// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of
// the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under
// the License.

// ---------------------------------------------------------------------------

/** Represents a generic SMPL-like human model.
 *  This contains the base shape/mesh/LBS weights of a SMPL-type
 *  model and handles loading from a standard SMPL-X npz file.
 *
 *  The loaded model can be passed to the Body<ModelConfig> class constructor.
 *  The Body class  can then generate a skinned human mesh from parameters using
 *  the model's data.
 *
 *  template arg ModelConfig is the static 'model configuration', which you
 *  should pick from smplx::model_config::SMPL/SMPLH/SMPLX */
template <class ModelConfig>
class _API Model {
 public:
  // Construct from .npz at default path for given gender, in
  // data/models/modelname/MODELNAME_GENDER.npz
  explicit Model(
      reclib::models::Gender gender = reclib::models::Gender::neutral);

  // Construct from .npz at default path for given hand tpe, in
  // data/models/mano/MANO_HANDTYPE.npz
  explicit Model(
      reclib::models::HandType type = reclib::models::HandType::right);

  // Construct from .npz at path (standard SMPL-X npz format)
  // path: .npz model path, e.g. data/models/smplx/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  explicit Model(
      const std::string& path, const std::string& uv_path = "",
      reclib::models::Gender gender = reclib::models::Gender::unknown);

  // Construct from .npz at path (standard MANO npz format)
  // path: .npz model path, e.g. data/models/mano/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  explicit Model(
      const std::string& path, const std::string& uv_path = "",
      reclib::models::HandType type = reclib::models::HandType::unknown);
  // Destructor
  ~Model();

  // Disable copy/move assignment
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;

  /*** MODEL NPZ LOADING ***/
  // Load from .npz at default path for given gender
  // useful for dynamically switching genders
  void load(reclib::models::Gender gender = reclib::models::Gender::neutral);

  /*** MODEL NPZ LOADING ***/
  // Load from .npz at default path for given hand type
  // useful for dynamically switching hand types
  void load(reclib::models::HandType type = reclib::models::HandType::right);

  // Load from .npz at path (standard SMPL-X npz format)
  // path: .npz model path, in data/models/smplx/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  void load(
      const std::string& path, const std::string& uv_path = "",
      reclib::models::Gender new_gender = reclib::models::Gender::unknown);

  // Load from .npz at path (standard SMPL-X npz format)
  // path: .npz model path, in data/models/smplx/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  void load(const std::string& path, const std::string& uv_path = "",
            reclib::models::HandType type = reclib::models::HandType::unknown);

  void cpu();
  void gpu();
  bool is_cpu() const;
  bool is_gpu() const;

  /*** MODEL MANIPULATION ***/
  // Set model deformations: verts := verts_load + d
  void set_deformations(const torch::Tensor& d);

  // Set model template: verts := t
  void set_template(const torch::Tensor& t);

  using Config = ModelConfig;

  /*** STATIC DATA SHAPE INFO SHORTHANDS,
   *   mostly forwarding to ModelConfig ***/

  // Number of vertices in model
  static constexpr size_t n_verts() { return Config::n_verts(); }
  // Number of faces in model
  static constexpr size_t n_faces() { return Config::n_faces(); }

  // Total number of joints = n_explicit_joints + n_hand_pca_joints * 2
  static constexpr size_t n_joints() { return Config::n_joints(); }
  // Number of explicit joint parameters stored as angle-axis
  static constexpr size_t n_explicit_joints() {
    return Config::n_explicit_joints();
  }
  // Number of joints per hand implicit computed from PCA
  static constexpr size_t n_hand_pca_joints() {
    return Config::n_hand_pca_joints();
  }

  // Total number of blend shapes = n_shape_blends + n_pose_blends
  static constexpr size_t n_blend_shapes() { return Config::n_blend_shapes(); }
  // Number of shape-dep blend shapes, including body and face
  static constexpr size_t n_shape_blends() { return Config::n_shape_blends(); }
  // Number of pose-dep blend shapes = 9 * (n_joints - 1)
  static constexpr size_t n_pose_blends() { return Config::n_pose_blends(); }

  // Number of PCA components for each hand
  static constexpr size_t n_hand_pca() { return Config::n_hand_pca(); }

  // Total number of params = 3 + 3 * n_body_joints + 2 * n_hand_pca +
  static constexpr size_t n_params() { return Config::n_params(); }

  // Number UV vertices (may be more than n_verts due to seams)
  // 0 if UV not available
  // NOTE: not static or a constexpr
  inline size_t n_uv_verts() const { return _n_uv_verts; }

  /*** ADDITIONAL MODEL INFORMATION ***/
  // Model name
  static constexpr const char* name() { return Config::model_name; }
  // Joint names name
  static constexpr const char* joint_name(size_t joint) {
    return Config::joint_name[joint];
  }
  // Parent joint
  static constexpr size_t parent(size_t joint) { return Config::parent[joint]; }

  // Model gender, may be unknown.
  reclib::models::Gender gender;

  // Hand type, may be unknown
  reclib::models::HandType hand_type;

  // Returns true if has UV map.
  // Note: not static, since we allow UV map variation among model instances.
  inline bool has_uv_map() const { return _n_uv_verts > 0; }

  /*** MODEL DATA ***/
  // Kinematic tree: joint children
  std::vector<std::vector<size_t>> children;

  // Vertices in the unskinned mesh, (#verts, 3).
  // This is verts_load with deformations (set with set_deformations).
  torch::Tensor verts;
  const torch::Tensor& verts_rest() const { return verts; }

  // Vertices in the initial loaded mesh, (#verts, 3)
  torch::Tensor verts_load;

  //  torch::Tensor in the mesh, (#faces, 3)
  torch::Tensor faces;

  // Initial joint positions
  torch::Tensor joints;

  torch::Tensor verts2joints;

  // Shape- and pose-dependent blend shapes,
  // (3*#verts, #shape blends + #pose blends)
  // each col represents a point cloud (#verts, 3) in row-major order
  torch::Tensor blend_shapes;  //  Eigen::Matrix<float, Eigen::Dynamic,
                               //  Model::n_blend_shapes()>

  // Joint regressor: verts -> joints, (#joints, #verts)
  // SPARSE
  torch::Tensor joint_reg;

  Eigen::Matrix<int, Eigen::Dynamic, 6> vertex_ring;

  // LBS weights, (#verts, #joints).
  // NOTE: this is ColMajor because I notice a speedup while profiling
  // SPARSE
  torch::Tensor weights;

  /*** Hand PCA data ***/
  // Hand PCA comps: pca -> joint pos delta
  // 3*#hand joints (=45) * #hand pca
  // columns are PC's
  torch::Tensor hand_comps;
  // Hand PCA means: mean pos of 3x15 hand joints
  torch::Tensor hand_mean;

  /*** UV Data , available if has_uv_map() ***/
  // UV coordinates, size (n_uv_verts, 2)
  torch::Tensor uv;
  // UV triangles (indices in uv), size (n_faces, 3)
  torch::Tensor uv_faces;

 private:
  // Number UV vertices (may be more than n_verts due to seams)
  // 0 if UV not available
  size_t _n_uv_verts;
  void generate_vertex_ring();
};

/** A particular SMPL instance constructed from a Model<ModelConfig>,
 *  storing pose/shape/hand parameters and a skinned point cloud generated
 *  from the parameters (via calling the update method).
 *  Implements linear blend skinning with GPU and CPU (Eigen) support. */
template <class ModelConfig>
class _API ModelInstance {
 public:
  // Construct body from model
  // set_zero: set to false to leave parameter array uninitialized
  explicit ModelInstance(const Model<ModelConfig>& model, bool set_zero = true);
  ModelInstance();
  ~ModelInstance();

  // Perform LBS and output verts
  // enable_pose_blendshapes: if false, disables pose blendshapes;
  //                          this provides a significant speedup at the cost
  //                          of worse accuracy
  void update(bool force_cpu = false, bool enable_pose_blendshapes = true,
              bool anatomical_limits = false, bool update_meshes = true);

  // Save as obj file
  void save_obj(const std::string& path) const;

  using Config = ModelConfig;

  void cpu();
  void gpu();
  void requires_grad(bool val);

  // Parameter accessors (maps to parts of params)
  // Base position (translation)
  inline auto trans() { return torch::head(params, 3); }
  inline auto trans() const { return torch::head(params, 3); }
  inline void set_trans(torch::Tensor t) {
    params.index_put_({torch::indexing::Slice(0, 3)}, t);
  }
  inline torch::Tensor apose() { return anatomic_params; }
  inline auto apose() const { return anatomic_params; }
  inline void set_apose(torch::Tensor t) { anatomic_params = t; }

  inline auto pose() {
    return torch::segment(params, 3, ModelConfig::n_explicit_joints() * 3);
  }
  inline auto pose() const {
    return torch::segment(params, 3, ModelConfig::n_explicit_joints() * 3);
  }

  inline auto rot() {
    if (use_anatomic_pose_) {
      return torch::segment(anatomic_params, 0, 3);
    } else {
      return torch::segment(params, 3, 3);
    }
  }
  inline auto rot() const {
    if (use_anatomic_pose_) {
      return torch::segment(anatomic_params, 0, 3);
    } else {
      return torch::segment(params, 3, 3);
    }
  }
  inline void set_rot(torch::Tensor t) {
    if (use_anatomic_pose_) {
      anatomic_params.index_put_({torch::indexing::Slice(0, 3)}, t);
    } else {
      params.index_put_({torch::indexing::Slice(3, 6)}, t);
    }
  }

  inline void set_pose(torch::Tensor t) {
    if (use_anatomic_pose_) {
      anatomic_params.index_put_({torch::indexing::Slice(3, 26)}, t);
    } else {
      params.index_put_(
          {torch::indexing::Slice(3, 3 + ModelConfig::n_explicit_joints() * 3)},
          t);
    }
  }

  inline auto hand_pca() {
    return torch::segment(params, 3 + 3 * model.n_explicit_joints(),
                          ModelConfig::n_hand_pca() * ModelConfig::n_hands());
  }
  inline auto hand_pca() const {
    return torch::segment(params, 3 + 3 * model.n_explicit_joints(),
                          ModelConfig::n_hand_pca() * ModelConfig::n_hands());
  }
  inline void set_hand_pca(torch::Tensor t) {
    if (use_anatomic_pca_) {
      params.index_put_(
          {torch::indexing::Slice(3 + 3 * model.n_explicit_joints(),
                                  3 + 3 * model.n_explicit_joints() + 23)},
          t);
    } else {
      params.index_put_(
          {torch::indexing::Slice(
              3 + 3 * model.n_explicit_joints(),
              3 + 3 * model.n_explicit_joints() + ModelConfig::n_hand_pca())},
          t);
    }
  }

  inline auto hand_pca_l() {
    assert(ModelConfig::n_hands());
    return torch::segment(params, 3 + 3 * model.n_explicit_joints(),
                          ModelConfig::n_hand_pca());
  }
  inline auto hand_pca_l() const {
    assert(ModelConfig::n_hands());
    return torch::segment(params, 3 + 3 * model.n_explicit_joints(),
                          ModelConfig::n_hand_pca());
  }
  inline auto hand_pca_r() {
    assert(ModelConfig::n_hands());
    if (ModelConfig::n_hands() == 1) {
      return hand_pca();
    }
    return torch::segment(
        params, 3 + 3 * model.n_explicit_joints() + model.n_hand_pca(),
        ModelConfig::n_hand_pca());
  }
  inline auto hand_pca_r() const {
    assert(ModelConfig::n_hands());
    if (ModelConfig::n_hands() == 1) {
      return hand_pca();
    }
    return torch::segment(
        params, 3 + 3 * model.n_explicit_joints() + model.n_hand_pca(),
        ModelConfig::n_hand_pca());
  }
  inline auto shape() {
    return torch::tail(params, ModelConfig::n_shape_blends());
  }
  inline auto shape() const {
    return torch::tail(params, ModelConfig::n_shape_blends());
  }
  inline void set_shape(torch::Tensor t) {
    params.index_put_({torch::indexing::Slice(
                          params.sizes()[0] - ModelConfig::n_shape_blends(),
                          params.sizes()[0])},
                      t);
  }

  // * OUTPUTS accessors
  // Get shaped + posed body vertices, in same order as model.verts;
  // must call update() before this is available
  const torch::Tensor& verts() const;
  torch::Tensor& verts();

  // Get shaped (but not posed) body vertices, in same order as model.verts;
  // must call update() before this is available
  const torch::Tensor& verts_shaped() const;

  // Get deformed body joints, in same order as model.joints;
  // must call update() before this is available
  const torch::Tensor& joints() const;

  // Get homogeneous transforms at each joint. (n_joints, 12).
  // Each row is a row-major (3, 4) rigid body transform matrix,
  // canonical -> posed space.
  const torch::Tensor& joint_transforms()
      const;  // Eigen::Matrix<float, Eigen::Dynamic, 12,
              // Eigen::RowMajor>

  // Get homogeneous transforms at each vertex. (n_verts, 12).
  // Each row is a row-major (3, 4) rigid body transform matrix,
  // canonical -> posed space.
  const torch::Tensor& vert_transforms()
      const;  // Eigen::Matrix<float, Eigen::Dynamic, 12,
              // Eigen::RowMajor>

  // Set parameters to zero
  inline void set_zero() {
    params.index({torch::All}) = 0;

    if (anatomic_params.sizes()[0] > 0) {
      anatomic_params.zero_();
    }
  }

  // The SMPL model used
  const Model<ModelConfig>& model;
  bool add_pose_mean_;
  bool use_anatomic_pose_;
  bool use_anatomic_pca_;

  // * INPUTS
  // Parameters vector
  torch::Tensor params;
  torch::Tensor anatomic_params;

  bool has_opengl_mesh;
  torch::Tensor gl_model_indices;
  torch::Tensor gl_model_vertices;
  torch::Tensor gl_model_joints;
  reclib::opengl::Drawelement gl_instance;
  reclib::opengl::Drawelement gl_joints;
  reclib::opengl::Drawelement gl_joint_lines;
  std::vector<uint32_t> gl_joint_line_indices;

  void generate_gl_drawelem(
      reclib::opengl::Shader s = reclib::opengl::Shader(),
      reclib::opengl::Material m = reclib::opengl::Material());

 private:
  // * OUTPUTS generated by update
  // Deformed vertices (only shape applied); not available in case of GPU
  // (only device.verts_shaped)
  mutable torch::Tensor _verts_shaped;

  // Deformed vertices (shape and pose applied)
  mutable torch::Tensor _verts;

  // Deformed joints (only shape applied)
  torch::Tensor _joints_shaped;

  // Homogeneous transforms at each joint (bottom row omitted)
  torch::Tensor _joint_rotations;

  torch::Tensor _joint_translations;

  // Homogeneous transforms at each vertex (bottom row omitted)
  mutable torch::Tensor _vert_rotations;

  mutable torch::Tensor _vert_translations;

  // Deformed joints (shape and pose applied)
  mutable torch::Tensor _joints;

  // Transform local to global coordinates
  // Inputs: trans(), _joints_shaped
  // Outputs: _joints
  // Input/output: _joint_transforms
  void _local_to_global();
};

// Angle-axis to rotation matrix using custom implementation
inline torch::Tensor rodrigues(const torch::Tensor& vec) {
  _RECLIB_ASSERT_EQ(vec.sizes().size(), 1);
  _RECLIB_ASSERT_EQ(vec.sizes()[0], 3);

  const torch::Tensor theta = vec.clone().norm();
  const torch::Tensor eye = torch::eye(3).to(vec.device());

  if ((torch::abs(theta) < 1e-5f).item<bool>()) {
    torch::NoGradGuard no_grad;
    return eye;
  } else {
    const torch::Tensor c = torch::cos(theta);
    const torch::Tensor s = torch::sin(theta);
    const torch::Tensor r = (vec / theta).to(vec.device());
    torch::Tensor skew = torch::zeros({3, 3}).to(vec.device());
    skew.index({0, 0}) = 0;
    skew.index({0, 1}) = -r.index({2});
    skew.index({0, 2}) = r.index({1});
    skew.index({1, 0}) = r.index({2});
    skew.index({1, 1}) = 0;
    skew.index({1, 2}) = -r.index({0});
    skew.index({2, 0}) = -r.index({1});
    skew.index({2, 1}) = r.index({0});
    skew.index({2, 2}) = 0;

    return c * eye + (1 - c) * torch::matmul(r.unsqueeze(1), r.unsqueeze(0)) +
           s * skew;
  }
}

inline torch::Tensor batch_rodrigues(torch::Tensor& poses) {
  auto batch_size = poses.sizes()[0];
  auto rot_vecs = poses.view({-1, 3});
  auto angle = torch::norm(rot_vecs + 1e-8, 2, 1, true);
  auto rot_dir = rot_vecs / angle;
  auto cos = torch::unsqueeze(torch::cos(angle), 1);
  auto sin = torch::unsqueeze(torch::sin(angle), 1);

  auto dirs = torch::split(rot_dir, 1, 1);
  auto K = torch::zeros({batch_size, 3, 3}, rot_vecs.options());
  auto zeros = torch::zeros({batch_size, 1}, rot_vecs.options());
  K = torch::cat({zeros, -dirs[2], dirs[1], dirs[2], zeros, -dirs[0], -dirs[1],
                  dirs[0], zeros},
                 1)
          .view({batch_size, 3, 3});

  auto ident = torch::eye(3, rot_vecs.options());
  return (ident + sin * K + (1 - cos) * torch::bmm(K, K))
      .view({batch_size, 3, 3});
}

// Affine transformation matrix (hopefully) faster multiplication
// bottom row omitted
inline void mul_affine(const torch::Tensor& a, torch::Tensor& b) {
  b.index({torch::All, torch::indexing::Slice({0, 3})}) =
      torch::matmul(a.index({torch::All, torch::indexing::Slice({0, 3})}),
                    b.index({torch::All, torch::indexing::Slice({0, 3})}));

  b.index({torch::All, 3}) =
      a.index({torch::All, 3}) +
      torch::matmul(a.index({torch::All, torch::indexing::Slice({0, 3})}),
                    b.index({torch::All, 3}));
}

Eigen::Vector3f auto_color(size_t color_index);
torch::Tensor auto_color_table(size_t num_colors);

// //
// ---------------------------------------------------------------------------
// // End of adapted code
// //
// ---------------------------------------------------------------------------

}  // namespace modelstorch
}  // namespace reclib

#endif  // HAS_DNN_MODULE

#endif

#endif  // __unix__