#if __unix__

#ifndef RECLIB_MODELS_SMPL
#define RECLIB_MODELS_SMPL

#include <cstdio>
#include <sophus/so3.hpp>

#include "reclib/assert.h"
#include "reclib/data_types.h"
#include "reclib/models/model_config.h"
#include "reclib/opengl/camera.h"
#include "reclib/opengl/drawelement.h"
#include "reclib/opengl/material.h"
#include "reclib/opengl/shader.h"

namespace reclib {

namespace models {
// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/sxyu/smplxpp
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
  explicit Model(Gender gender = Gender::neutral);

  // Construct from .npz at default path for given hand tpe, in
  // data/models/mano/MANO_HANDTYPE.npz
  explicit Model(HandType type = HandType::right);

  // Construct from .npz at path (standard SMPL-X npz format)
  // path: .npz model path, e.g. data/models/smplx/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  explicit Model(const std::string& path, const std::string& uv_path = "",
                 Gender gender = Gender::unknown);

  // Construct from .npz at path (standard MANO npz format)
  // path: .npz model path, e.g. data/models/mano/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  explicit Model(const std::string& path, const std::string& uv_path = "",
                 HandType type = HandType::unknown);
  // Destructor
  ~Model();

  // Disable copy/move assignment
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;

  /*** MODEL NPZ LOADING ***/
  // Load from .npz at default path for given gender
  // useful for dynamically switching genders
  void load(Gender gender = Gender::neutral);

  /*** MODEL NPZ LOADING ***/
  // Load from .npz at default path for given hand type
  // useful for dynamically switching hand types
  void load(HandType type = HandType::right);

  // Load from .npz at path (standard SMPL-X npz format)
  // path: .npz model path, in data/models/smplx/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  void load(const std::string& path, const std::string& uv_path = "",
            Gender new_gender = Gender::unknown);

  // Load from .npz at path (standard SMPL-X npz format)
  // path: .npz model path, in data/models/smplx/*.npz
  // uv_path: UV map information path, see data/models/smplx/uv.txt for an
  // example gender: records gender of model. For informational purposes only.
  void load(const std::string& path, const std::string& uv_path = "",
            HandType type = HandType::unknown);

  /*** MODEL MANIPULATION ***/
  // Set model deformations: verts := verts_load + d
  void set_deformations(
      const Eigen::Ref<
          const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>& d);

  // Set model template: verts := t
  void set_template(
      const Eigen::Ref<
          const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>& t);

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
  Gender gender;

  // Hand type, may be unknown
  HandType hand_type;

  // Returns true if has UV map.
  // Note: not static, since we allow UV map variation among model instances.
  inline bool has_uv_map() const { return _n_uv_verts > 0; }

  /*** MODEL DATA ***/
  // Kinematic tree: joint children
  std::vector<std::vector<size_t>> children;

  // Vertices in the unskinned mesh, (#verts, 3).
  // This is verts_load with deformations (set with set_deformations).
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> verts;
  const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& verts_rest()
      const {
    return verts;
  }

  // Vertices in the initial loaded mesh, (#verts, 3)
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> verts_load;

  // Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor> in the mesh,
  // (#faces, 3)
  Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor> faces;

  // Initial joint positions
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> joints;

  Eigen::Vector<int, Model::n_verts()> verts2joints;

  // Shape- and pose-dependent blend shapes,
  // (3*#verts, #shape blends + #pose blends)
  // each col represents a point cloud (#verts, 3) in row-major order
  Eigen::Matrix<float, Eigen::Dynamic, Model::n_blend_shapes()> blend_shapes;

  // Joint regressor: verts -> joints, (#joints, #verts)
  Eigen::SparseMatrix<float, Eigen::RowMajor> joint_reg;

  Eigen::Matrix<int, Eigen::Dynamic, 6> vertex_ring;

  // LBS weights, (#verts, #joints).
  // NOTE: this is ColMajor because I notice a speedup while profiling
  Eigen::SparseMatrix<float> weights;

  /*** Hand PCA data ***/
  // Hand PCA comps: pca -> joint pos delta
  // 3*#hand joints (=45) * #hand pca
  // columns are PC's
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hand_comps;
  // Hand PCA means: mean pos of 3x15 hand joints
  Eigen::Matrix<float, Eigen::Dynamic, 1> hand_mean;

  /*** UV Data , available if has_uv_map() ***/
  // UV coordinates, size (n_uv_verts, 2)
  Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> uv;
  // UV triangles (indices in uv), size (n_faces, 3)
  Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor> uv_faces;

 private:
  // Number UV vertices (may be more than n_verts due to seams)
  // 0 if UV not available
  size_t _n_uv_verts;
  void generate_vertex_ring();
  void generate_verts2joints();
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
              bool anatomical_limits = false);

  // Save as obj file
  void save_obj(const std::string& path) const;

  using Config = ModelConfig;

  // Parameter accessors (maps to parts of params)
  // Base position (translation)
  inline auto trans() { return params.template head<3>(); }
  inline auto trans() const { return params.template head<3>(); }
  inline Eigen::Matrix<float, Eigen::Dynamic, 1>& apose() {
    return anatomic_params;
  }
  inline auto apose() const { return anatomic_params; }
  static Eigen::Vector<float, 48> apose2pose(
      const ModelInstance<ModelConfig>& model);
  static Eigen::Vector<float, MANOConfigExtra::anatomic_dofs> pose2apose(
      const ModelInstance<ModelConfig>& mode, bool anatomical_limits = false);
  inline auto pose() {
    return params.template segment<ModelConfig::n_explicit_joints() * 3>(3);
  }
  inline auto pose() const {
    return params.template segment<ModelConfig::n_explicit_joints() * 3>(3);
  }
  inline auto hand_pca() {
    return params.segment<ModelConfig::n_hand_pca() * ModelConfig::n_hands()>(
        3 + 3 * model.n_explicit_joints());
  }
  inline auto hand_pca() const {
    return params.segment<ModelConfig::n_hand_pca() * ModelConfig::n_hands()>(
        3 + 3 * model.n_explicit_joints());
  }
  inline auto shape() {
    return params.template tail<ModelConfig::n_shape_blends()>();
  }
  inline auto shape() const {
    return params.template tail<ModelConfig::n_shape_blends()>();
  }

  // * OUTPUTS accessors
  // Get shaped + posed body vertices, in same order as model.verts;
  // must call update() before this is available
  const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& verts() const;
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& verts();

  // Get shaped (but not posed) body vertices, in same order as model.verts;
  // must call update() before this is available
  const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& verts_shaped()
      const;

  // Get deformed body joints, in same order as model.joints;
  // must call update() before this is available
  const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& joints()
      const;

  // Get homogeneous transforms at each joint. (n_joints, 12).
  // Each row is a row-major (3, 4) rigid body transform matrix,
  // canonical -> posed space.
  const Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor>&
  joint_transforms() const;

  // Get homogeneous transforms at each vertex. (n_verts, 12).
  // Each row is a row-major (3, 4) rigid body transform matrix,
  // canonical -> posed space.
  const Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor>&
  vert_transforms() const;

  // Set parameters to zero
  inline void set_zero() {
    params.setZero();
    if (anatomic_params.rows() > 0) {
      anatomic_params.setZero();
    }
  }

  // Set parameters uar in [-0.25, 0.25]
  inline void set_random() { params.setRandom() * 0.25; }

  // The SMPL model used
  const Model<ModelConfig>& model;
  bool add_pose_mean_;
  bool use_anatomic_pose_;
  bool use_anatomic_pca_;

  // * INPUTS
  // Parameters vector
  Eigen::Matrix<float, Eigen::Dynamic, 1> params;
  Eigen::Matrix<float, Eigen::Dynamic, 1> anatomic_params;

  bool has_opengl_mesh;
  Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor> gl_model_indices;
  reclib::opengl::Drawelement gl_instance;
  // std::vector<reclib::opengl::Drawelement> gl_joints;
  // std::vector<reclib::opengl::Drawelement> gl_joint_lines;
  reclib::opengl::Drawelement gl_joints;
  reclib::opengl::Drawelement gl_joint_lines;
  std::vector<uint32_t> gl_joint_line_indices;

  void generate_gl_drawelem(
      reclib::opengl::Shader s = reclib::opengl::Shader(),
      reclib::opengl::Material m = reclib::opengl::Material());

  // * OUTPUTS generated by update
  // Deformed vertices (only shape applied); not available in case of GPU
  // (only device.verts_shaped)
  mutable Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>
      _verts_shaped;

  // Deformed vertices (shape and pose applied)
  mutable Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> _verts;

  // Deformed joints (only shape applied)
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> _joints_shaped;

  // Homogeneous transforms at each joint (bottom row omitted)
  Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor> _joint_transforms,
      _local_joint_transforms;

  // Homogeneous transforms at each vertex (bottom row omitted)
  mutable Eigen::Matrix<float, Eigen::Dynamic, 12, Eigen::RowMajor>
      _vert_transforms;

  // Deformed joints (shape and pose applied)
  mutable Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> _joints;

 private:
  // Transform local to global coordinates
  // Inputs: trans(), _joints_shaped
  // Outputs: _joints
  // Input/output: _joint_transforms
  void _local_to_global();

#ifdef SMPLX_CUDA_ENABLED
 public:
  struct {
    float* verts = nullptr;
    float* verts_shaped = nullptr;
    float* joints_shaped = nullptr;
    // Internal temp
    float* verts_tmp = nullptr;
    // Internal (#total blend shapes) rm
    float* blendshape_params = nullptr;
    // Internal (#joints, 12) rm
    float* joint_transforms = nullptr;
  } device;

 private:
  // True if latest posed vertices constructed by update()
  // have been retrieved to main memory
  mutable bool _verts_retrieved;
  // True if latest shaped, unposed vertices constructed by update()
  // have been retrieved to main memory
  mutable bool _verts_shaped_retrieved;
  // True if last update made use of the GPU
  bool _last_update_used_gpu;
  // Cuda helpers
  void _cuda_load();
  void _cuda_free();
  void _cuda_maybe_retrieve_verts() const;
  void _cuda_maybe_retrieve_verts_shaped() const;
  void _cuda_update(float* h_blendshape_params, float* h_joint_transforms,
                    bool enable_pose_blendshapes = true);
#endif
};

// Angle-axis to rotation matrix using custom implementation
template <class T, int Option = Eigen::ColMajor>
inline Eigen::Matrix<T, 3, 3, Option> rodrigues(
    const Eigen::Ref<const Eigen::Matrix<T, 3, 1>>& vec) {
  const T theta = vec.norm();
  const Eigen::Matrix<T, 3, 3, Option> eye =
      Eigen::Matrix<T, 3, 3, Option>::Identity();

  if (std::fabs(theta) < 1e-5f)
    return eye;
  else {
    const T c = std::cos(theta);
    const T s = std::sin(theta);
    const Eigen::Matrix<T, 3, 1> r = vec / theta;
    Eigen::Matrix<T, 3, 3, Option> skew;
    skew << T(0), -r.z(), r.y(), r.z(), T(0), -r.x(), -r.y(), r.x(), T(0);
    return c * eye + (T(1) - c) * r * r.transpose() + s * skew;
  }
}

// Affine transformation matrix (hopefully) faster multiplication
// bottom row omitted
template <class T, int Option = Eigen::ColMajor>
inline void mul_affine(
    const Eigen::Ref<const Eigen::Matrix<T, 3, 4, Option>>& a,
    Eigen::Ref<Eigen::Matrix<T, 3, 4, Option>> b) {
  b.template leftCols<3>() =
      a.template leftCols<3>() * b.template leftCols<3>();
  b.template rightCols<1>() =
      a.template rightCols<1>() +
      a.template leftCols<3>() * b.template rightCols<1>();
}

template <class T, int Option = Eigen::ColMajor>
inline Eigen::Matrix<T, 3, 4, Option> mul_affine_ret(
    const Eigen::Ref<const Eigen::Matrix<T, 3, 4, Option>>& a,
    Eigen::Ref<Eigen::Matrix<T, 3, 4, Option>> b) {
  Eigen::Matrix<T, 3, 4, Option> c;
  c.template leftCols<3>() =
      a.template leftCols<3>() * b.template leftCols<3>();
  c.template rightCols<1>() =
      a.template rightCols<1>() +
      a.template leftCols<3>() * b.template rightCols<1>();
  return c;
}

Eigen::Vector3f auto_color(size_t color_index);
Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> auto_color_table(
    size_t num_colors);

// ---------------------------------------------------------------------------
// End of adapted code
// ---------------------------------------------------------------------------
template <typename T>
Eigen::Vector<float, reclib::models::MANOConfigPCA::n_hand_pca_joints() * 3>
mano_angle2pca(reclib::models::ModelInstance<T>& mano);

template <typename T>
mat4 mano_canonical_space(reclib::models::ModelInstance<T>& mano);

template <typename T>
Eigen::Matrix<float, Eigen::Dynamic, 6> compute_visible_part_bbs(
    reclib::models::ModelInstance<T>& mano, reclib::opengl::Camera cam);

template <typename T>
std::vector<vec4> mano_canonical_space_colors(
    reclib::models::ModelInstance<T>& mano, bool transform_mano = false,
    bool use_cylinder = false);

template <typename T>
std::vector<vec4> mano_canonical_space_colors_hsv(
    reclib::models::ModelInstance<T>& mano, bool transform_mano = false,
    bool convert_to_rgb = true);

template <typename T>
std::pair<std::vector<vec4>, std::vector<vec4>>
mano_canonical_joint_mean_colors(reclib::models::ModelInstance<T>& mano,
                                 bool transform_mano = false);

// template <typename T, class ModelConfig>
// void LBS(
//     const reclib::models::Model<ModelConfig>& model,
//     const Eigen::Ref<const Eigen::Vector<T, ModelConfig::n_shape_blends()>>&
//         shape,
//     const Eigen::Ref<const Eigen::Vector<T, ModelConfig::n_explicit_joints()
//     *
//                                                 3>>& incremental_pose,
//     const Eigen::Ref<
//         const Eigen::Vector<T, ModelConfig::n_explicit_joints() * 3>>& pose,
//     Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& _verts,
//     Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& _joints, bool
//     enable_pose_blendshapes, int vertex_index = -1, const Eigen::Ref<const
//     Eigen::Vector<
//         T, ModelConfig::n_hand_pca() * ModelConfig::n_hands()>>& hand_pca =
//         Eigen::Vector<T, ModelConfig::n_hand_pca() *
//                              ModelConfig::n_hands()>::Zero(),
//     const Eigen::Ref<const Eigen::Vector<T, 3>>& global_trans =
//         Eigen::Vector<T, 3>(T(0), T(0), T(0))) {
//   // Will store full pose params (angle-axis), including hand
//   Eigen::Vector<T, ModelConfig::n_joints() * 3> full_pose;

//   // Homogeneous transforms at each joint (bottom row omitted)
//   Eigen::Matrix<T, ModelConfig::n_joints(), 12, Eigen::RowMajor>
//       _joint_transforms;
//   Eigen::Matrix<T, ModelConfig::n_joints(), 3, Eigen::RowMajor>
//   _joints_shaped; Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>
//   _verts_shaped(
//       ModelConfig::n_verts(), 3);

//   // First rows: Shape params
//   // Succeeding rows: linear joint transformations as flattened 3x3 rotation
//   // matrices rowmajor, only for blend shapes
//   Eigen::Vector<T, ModelConfig::n_blend_shapes()> blendshape_params;

//   using TransformMap = Eigen::Map<Eigen::Matrix<T, 3, 4, Eigen::RowMajor>>;
//   using TransformTransposedMap = Eigen::Map<Eigen::Matrix<T, 4, 3>>;
//   using RotationMap = Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>>;

//   // Copy body pose onto full pose
//   // the pose consists of all joints in axis-angle representation
//   // XXX: beware: the pose of joint i does not influence
//   // joint i itself, but instead its children!!
//   full_pose.head(3 * ModelConfig::n_explicit_joints()).noalias() = pose;

//   if (ModelConfig::n_hand_pca_joints() > 0) {
//     // Use hand PCA weights to fill in hand pose within full pose
//     if (model.hand_type == HandType::left) {
//       full_pose
//           .segment(3 * ModelConfig::n_explicit_joints(),
//                    3 * ModelConfig::n_hand_pca_joints())
//           .noalias() =
//           model.hand_mean_l.template cast<T>() +
//           model.hand_comps_l.template cast<T>() *
//               hand_pca.template segment<ModelConfig::n_hand_pca()>(0);
//     } else if (model.hand_type == HandType::right) {
//       full_pose
//           .segment(3 * ModelConfig::n_explicit_joints(),
//                    3 * ModelConfig::n_hand_pca_joints())
//           .noalias() =
//           model.hand_mean_r.template cast<T>() +
//           model.hand_comps_r.template cast<T>() *
//               hand_pca.template segment<ModelConfig::n_hand_pca()>(0);
//     } else {
//       full_pose
//           .segment(3 * ModelConfig::n_explicit_joints(),
//                    3 * ModelConfig::n_hand_pca_joints())
//           .noalias() =
//           model.hand_mean_l.template cast<T>() +
//           model.hand_comps_l.template cast<T>() *
//               hand_pca.template segment<ModelConfig::n_hand_pca()>(0);
//       full_pose.tail(3 * ModelConfig::n_hand_pca_joints()).noalias() =
//           model.hand_mean_r.template cast<T>() +
//           model.hand_comps_r.template cast<T>() *
//               hand_pca.template segment<ModelConfig::n_hand_pca()>(
//                   ModelConfig::n_hand_pca());
//     }
//   }

//   // Copy shape params to FIRST ROWS of blendshape params
//   blendshape_params.template head<ModelConfig::n_shape_blends()>() = shape;

//   // Convert angle-axis to rotation matrix using rodrigues
//   // First load joint_transforms into 3x4 matrix map
//   // then fill with rodriguez converted axis-angle
//   // TLDR: map axis-angle -> 3x4 matrix
//   TransformMap(_joint_transforms.template topRows<1>().data())
//       .template leftCols<3>()
//       .noalias() = incrementalAA2Mat<T>(incremental_pose.template head<3>(),
//                                         pose.template head<3>());
//   for (size_t i = 1; i < ModelConfig::n_joints(); ++i) {
//     // TLDR: map axis-angle -> 3x4 matrix
//     TransformMap joint_trans(_joint_transforms.row(i).data());

//     if (i < ModelConfig::n_explicit_joints()) {
//       joint_trans.template leftCols<3>().noalias() =
//           incrementalAA2Mat<T>(incremental_pose.template segment<3>(3 * i),
//                                full_pose.template segment<3>(3 * i));
//     } else {
//       joint_trans.template leftCols<3>().noalias() =
//           Sophus::SO3<T>::exp(full_pose.template segment<3>(3 * i)).matrix();
//     }

//     // Store 3x3 joint-wise rotation matrix in blendshape_params
//     // XXX: first rows are not overwritten! The shape parameters beta are
//     // inside the first forws of blendshape_params
//     RotationMap mp(blendshape_params.data() + 9 * i +
//                    (ModelConfig::n_shape_blends() - 9));
//     mp.noalias() = joint_trans.template leftCols<3>();
//     mp.diagonal().array() -= T(1.f);
//   }

//   // Apply blend shapes
//   {
//     // load map of deformed vertices (only shape applied)
//     Eigen::Map<Eigen::Matrix<T, ModelConfig::n_verts() * 3, 1>>
//         verts_shaped_flat(_verts_shaped.data());
//     // load vertices in rest pose
//     // Eigen::Matrix<T, ModelConfig::n_verts(), 3, Eigen::RowMajor>
//     model_verts
//     // =
//     //     model.verts.template cast<T>();
//     Eigen::Map<const Eigen::Matrix<float, ModelConfig::n_verts() * 3, 1>>
//         verts_init_flat(model.verts.data());

//     // Add shape blend shapes
//     // model.blend_shapes (3*#verts, #shape blends)
//     // (first rows of) blendshape_params: (#shape blends)
//     // apply equation:
//     // sum_i beta_i * S_i with S_i
//     verts_shaped_flat.noalias() =
//         verts_init_flat.template cast<T>() +
//         model.blend_shapes.template leftCols<ModelConfig::n_shape_blends()>()
//                 .template cast<T>() *
//             blendshape_params.template head<ModelConfig::n_shape_blends()>();
//   }
//   // Apply joint regressor

//   _joints_shaped = model.joint_reg.template cast<T>() * _verts_shaped;

//   if (enable_pose_blendshapes) {
//     // HORRIBLY SLOW, like 95% of the time is spent here yikes
//     // Add pose blend shapes

//     // apply equation: sum_j (rot_j - rot_rest_pose_j)  * P_j
//     // = sum_j rot_j * P_j (since rotation in rest pose is 0)
//     Eigen::Map<Eigen::Matrix<T, ModelConfig::n_verts() * 3, 1>>
//         verts_shaped_flat(_verts_shaped.data());
//     verts_shaped_flat +=
//         model.blend_shapes.template rightCols<ModelConfig::n_pose_blends()>()
//             .template cast<T>() *
//         blendshape_params.template tail<ModelConfig::n_pose_blends()>();
//   }

//   Local2Global<T, ModelConfig>(_joint_transforms, _joints_shaped,
//   global_trans); _joints = _joints_shaped;

//   // * LBS *
//   if (vertex_index < 0) {
//     _RECLIB_ASSERT_EQ(_verts.rows(), ModelConfig::n_verts());

//     // Construct a transform for each vertex
//     Eigen::Matrix<T, Eigen::Dynamic, 12, Eigen::RowMajor> _vert_transforms(
//         ModelConfig::n_verts(), 12);
//     _vert_transforms.noalias() =
//         model.weights.template cast<T>() * _joint_transforms;

//     // Apply affine transform to each vertex and store to output
//     for (size_t i = 0; i < ModelConfig::n_verts(); ++i) {
//       TransformTransposedMap transform_tr(_vert_transforms.row(i).data());
//       _verts.row(i).noalias() =
//           _verts_shaped.row(i).homogeneous() * transform_tr;
//     }
//   } else {
//     _RECLIB_ASSERT_EQ(_verts.rows(), 1);

//     Eigen::Matrix<T, 1, 12, Eigen::RowMajor> vertex_transform;
//     Eigen::Matrix<T, 1, ModelConfig::n_joints()> weights =
//         model.weights.row(vertex_index).template cast<T>();
//     vertex_transform.noalias() = weights * _joint_transforms;

//     TransformTransposedMap transform_tr(vertex_transform.data());
//     _verts.row(0).noalias() =
//         _verts_shaped.row(vertex_index).homogeneous() * transform_tr;
//   }
// }

}  // namespace models
}  // namespace reclib

#endif

#endif  // __unix__