
#if __unix__
#if HAS_DNN_MODULE
#include <reclib/models/smpl_torch.h>

#include <fstream>
#include <iostream>
#include <string>

#include "reclib/assert.h"
#include "reclib/dnn/dnn_utils.h"
#include "reclib/opengl/query.h"
#include "reclib/python/npy2eigen.h"

namespace reclib {
namespace modelstorch {

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

torch::Tensor auto_color_table(size_t num_colors) {
  torch::Tensor colors = torch::zeros({(int)num_colors, 3});
  for (int i = 0; i < (int)num_colors; ++i) {
    vec3 c = auto_color(i);
    colors.index({i, 0}) = c[0];
    colors.index({i, 1}) = c[1];
    colors.index({i, 2}) = c[2];
  }
  return colors;
}

// -------------------------------------------------------------
// Model
// -------------------------------------------------------------

template <class ModelConfig>
reclib::modelstorch::Model<ModelConfig>::Model(reclib::models::Gender gender) {
  load(gender);
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::Model(const std::string& path, const std::string& uv_path,
                          reclib::models::Gender gender) {
  load(path, uv_path, gender);
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::Model(reclib::models::HandType type) {
  load(type);
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::Model(const std::string& path, const std::string& uv_path,
                          reclib::models::HandType type) {
  load(path, uv_path, type);
  generate_vertex_ring();
}

template <class ModelConfig>
Model<ModelConfig>::~Model() {}

template <class ModelConfig>
void Model<ModelConfig>::load(reclib::models::Gender gender) {
  load(reclib::models::find_data_file(
           std::string(ModelConfig::default_path_prefix) +
           gender_to_str(gender) + ".npz"),
       reclib::models::find_data_file(ModelConfig::default_uv_path), gender);
}

template <class ModelConfig>
void Model<ModelConfig>::load(reclib::models::HandType type) {
  load(reclib::models::find_data_file(
           std::string(ModelConfig::default_path_prefix) +
           hand_type_to_str(type) + ".npz"),
       reclib::models::find_data_file(ModelConfig::default_uv_path), type);
}

template <class ModelConfig>
void Model<ModelConfig>::load(const std::string& path,
                              const std::string& uv_path,
                              reclib::models::HandType type) {
  hand_type = type;
  return load(path, uv_path, reclib::models::Gender::unknown);
}

template <class ModelConfig>
void Model<ModelConfig>::cpu() {
  if (verts.sizes()[0] > 0) verts = verts.cpu();
  if (verts_load.sizes()[0] > 0) verts_load = verts_load.cpu();
  if (faces.sizes()[0] > 0) faces = faces.cpu();
  if (joints.sizes()[0] > 0) joints = joints.cpu();
  if (blend_shapes.sizes()[0] > 0) blend_shapes = blend_shapes.cpu();
  if (joint_reg.sizes()[0] > 0) joint_reg = joint_reg.cpu();
  if (weights.sizes()[0] > 0) weights = weights.cpu();
  if (hand_comps.sizes()[0] > 0) hand_comps = hand_comps.cpu();
  if (hand_mean.sizes()[0] > 0) hand_mean = hand_mean.cpu();
  if (uv.sizes()[0] > 0) uv = uv.cpu();
  if (uv_faces.sizes()[0] > 0) uv_faces = uv_faces.cpu();
}

template <class ModelConfig>
bool Model<ModelConfig>::is_cpu() const {
  return verts.is_cpu();
}

template <class ModelConfig>
void Model<ModelConfig>::gpu() {
  if (verts.sizes()[0] > 0) verts = verts.cuda();
  if (verts_load.sizes()[0] > 0) verts_load = verts_load.cuda();
  if (faces.sizes()[0] > 0) faces = faces.cuda();
  if (joints.sizes()[0] > 0) joints = joints.cuda();
  if (blend_shapes.sizes()[0] > 0) blend_shapes = blend_shapes.cuda();
  if (joint_reg.sizes()[0] > 0) joint_reg = joint_reg.cuda();
  if (weights.sizes()[0] > 0) weights = weights.cuda();
  if (hand_comps.sizes()[0] > 0) hand_comps = hand_comps.cuda();
  if (hand_mean.sizes()[0] > 0) hand_mean = hand_mean.cuda();
  if (uv.sizes()[0] > 0) uv = uv.cuda();
  if (uv_faces.sizes()[0] > 0) uv_faces = uv_faces.cuda();
}

template <class ModelConfig>
bool Model<ModelConfig>::is_gpu() const {
  return verts.is_cuda();
}

template <class ModelConfig>
void Model<ModelConfig>::load(const std::string& path,
                              const std::string& uv_path,
                              reclib::models::Gender new_gender) {
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
  auto& verts_raw = npz.at("v_template");
  reclib::models::assert_shape(verts_raw, {n_verts(), 3});
  // row-major
  verts = reclib::python::tensor::load_float_matrix(verts_raw, n_verts(), 3);
  verts_load = verts;
  verts.set_requires_grad(false);
  verts_load.set_requires_grad(false);

  // Load triangle mesh
  auto& faces_raw = npz.at("f");
  reclib::models::assert_shape(faces_raw, {n_faces(), 3});
  faces = reclib::python::tensor::load_uint_matrix(/*row-major*/ faces_raw,
                                                   n_faces(), 3);
  faces.set_requires_grad(false);

  // Load joint regressor
  auto& jreg_raw = npz.at("J_regressor");
  reclib::models::assert_shape(jreg_raw, {n_joints(), n_verts()});
  joint_reg = reclib::python::tensor::load_float_matrix(/*row-major*/ jreg_raw,
                                                        n_joints(), n_verts());
  joint_reg = joint_reg.to_sparse();
  joints = torch::matmul(joint_reg, verts);
  joint_reg.set_requires_grad(false);
  joints.set_requires_grad(false);

  // Load LBS weights
  auto& wt_raw = npz.at("weights");
  reclib::models::assert_shape(wt_raw, {n_verts(), n_joints()});
  weights = reclib::python::tensor::load_float_matrix(/*row-major*/ wt_raw,
                                                      n_verts(), n_joints());
  verts2joints = weights.argmax(1);

  weights = weights.to_sparse();
  weights.set_requires_grad(false);

  // Load shape-dep
  // blend shapes
  auto& sb_raw = npz.at("shapedirs");
  reclib::models::assert_shape(sb_raw, {n_verts(), 3, n_shape_blends()});
  blend_shapes = torch::zeros({n_verts() * 3, n_blend_shapes()});
  blend_shapes.set_requires_grad(false);
  torch::leftCols<n_shape_blends()>(blend_shapes) =
      reclib::python::tensor::load_float_matrix(
          /*row-major*/ sb_raw, 3 * n_verts(), n_shape_blends());

  // Load pose-dep blend
  // shapes
  auto& pb_raw = npz.at("posedirs");
  reclib::models::assert_shape(pb_raw, {n_verts(), 3, n_pose_blends()});
  torch::rightCols<n_pose_blends()>(blend_shapes) =
      reclib::python::tensor::load_float_matrix(/*row-major*/ pb_raw,
                                                3 * n_verts(), n_pose_blends());

  if (npz.count("hands_meanl") && npz.count("hands_meanr")) {
    // Model has hand
    // PCA (e.g.
    // SMPLXpca), load
    // hand PCA
    auto& hml_raw = npz.at("hands_meanl");
    auto& hmr_raw = npz.at("hands_meanr");
    auto& hcl_raw = npz.at("hands_componentsl");
    auto& hcr_raw = npz.at("hands_componentsr");

    reclib::models::assert_shape(hml_raw, {reclib::models::ANY_SHAPE});
    reclib::models::assert_shape(hmr_raw, {hml_raw.shape[0]});

    size_t n_hand_params = hml_raw.shape[0];
    _RECLIB_ASSERT_EQ(n_hand_params, n_hand_pca_joints() * 3);

    reclib::models::assert_shape(hcl_raw, {n_hand_params, n_hand_params});
    reclib::models::assert_shape(hcr_raw, {n_hand_params, n_hand_params});

    hand_mean = torch::zeros({(int)n_hand_params, 1});
    hand_comps = torch::zeros({(int)n_hand_params, (int)n_hand_params});
    hand_mean.set_requires_grad(false);
    hand_comps.set_requires_grad(false);

    if (hand_type == reclib::models::HandType::left) {
      hand_mean = reclib::python::tensor::load_float_matrix(
          /*row-major*/ hml_raw, n_hand_params, 1);

      hand_comps =
          reclib::python::tensor::load_float_matrix(/*row-major*/
                                                    hcl_raw, n_hand_params,
                                                    n_hand_params)
              .index({torch::indexing::Slice({0, n_hand_pca()})})
              .transpose(0, 1);
    } else {
      hand_mean = reclib::python::tensor::load_float_matrix(
          /*row-major*/ hmr_raw, n_hand_params, 1);
      hand_comps =
          reclib::python::tensor::load_float_matrix(/*row-major*/
                                                    hcr_raw, n_hand_params,
                                                    n_hand_params)
              .index({torch::indexing::Slice({0, n_hand_pca()})})
              .transpose(0, 1);
    }

  } else if (npz.count("hands_mean")) {
    assert(hand_type != reclib::models::HandType::unknown);
    // Model has hand
    // PCA, e.g. MANO
    auto& hm_raw = npz.at("hands_mean");
    auto& hc_raw = npz.at("hands_components");

    reclib::models::assert_shape(hm_raw, {reclib::models::ANY_SHAPE});

    size_t n_hand_params = hm_raw.shape[0];

    if (n_hand_pca()) _RECLIB_ASSERT_EQ(n_hand_params, n_hand_pca_joints() * 3);

    reclib::models::assert_shape(hc_raw, {n_hand_params, n_hand_params});

    hand_mean = torch::zeros({(int)n_hand_params, 1});
    hand_comps = torch::zeros({(int)n_hand_params, (int)n_hand_params});

    hand_mean.set_requires_grad(false);
    hand_comps.set_requires_grad(false);

    hand_mean = reclib::python::tensor::load_float_matrix(
        /*row-major*/ hm_raw, n_hand_params, 1);
    hand_comps =
        reclib::python::tensor::load_float_matrix(/*row-major*/
                                                  hc_raw, n_hand_params,
                                                  n_hand_params)
            .index({torch::indexing::Slice({0, n_hand_pca()})})
            .transpose(0, 1);
  }

  // Maybe load UV (UV
  // mapping WIP)
  if (uv_path.size()) {
    std::ifstream ifs(uv_path);
    ifs >> _n_uv_verts;

    if (_n_uv_verts) {
      if (ifs) {
        // _SMPLX_ASSERT_LE(n_verts(),
        // _n_uv_verts);
        // Load the uv
        // data
        uv = torch::zeros({(int)_n_uv_verts, 2});
        uv_faces = torch::zeros({n_faces(), 3});
        uv_faces.set_requires_grad(false);
        uv.set_requires_grad(false);

        for (unsigned int i = 0; i < _n_uv_verts; ++i) {
          int val;
          ifs >> val;
          uv.index({(int)i, 0}) = val;
          ifs >> val;
          uv.index({(int)i, 1}) = val;
        }

        assert(ifs);
        for (unsigned int i = 0; i < n_faces(); ++i) {
          assert(ifs);
          for (unsigned int j = 0; j < 3; ++j) {
            int val;
            ifs >> val;
            uv_faces.index({(int)i, (int)j}) = val;
            // Make
            // indices
            // 0-based
            uv_faces.index({(int)i, (int)j}) -= 1;
            _RECLIB_ASSERT_LT(uv_faces.index({(int)i, (int)j}).item().toInt(),
                              (int)_n_uv_verts);
          }
        }
      }
    }
  }

  std::cout << "loading done" << std::endl;
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

  for (unsigned int f = 0; f < faces.sizes()[0]; f++) {
    int i0 = faces.index({(int)f, 0}).item().toInt();
    int i1 = faces.index({(int)f, 1}).item().toInt();
    int i2 = faces.index({(int)f, 2}).item().toInt();

    insert_fun(i0, i1);
    insert_fun(i0, i2);

    insert_fun(i1, i0);
    insert_fun(i1, i2);

    insert_fun(i2, i1);
    insert_fun(i2, i0);
  }
}

template <class ModelConfig>
void Model<ModelConfig>::set_deformations(const torch::Tensor& d) {
  verts = verts_load + d;
}

template <class ModelConfig>
void Model<ModelConfig>::set_template(const torch::Tensor& t) {
  verts = t;
}

// Instantiations
template class Model<reclib::models::SMPLConfig>;
template class Model<reclib::models::MANOConfig>;
template class Model<reclib::models::MANOConfigPCA>;
template class Model<reclib::models::MANOConfigPCAGeneric<45>>;
template class Model<reclib::models::MANOConfigPCAGeneric<23>>;
template class Model<reclib::models::MANOConfigAnglePCA>;
template class Model<reclib::models::MANOConfigAnglePCAGeneric<45>>;
template class Model<reclib::models::MANOConfigAnglePCAGeneric<23>>;

// -------------------------------------------------------------
// ModelInstance
// -------------------------------------------------------------

template <class ModelConfig>
ModelInstance<ModelConfig>::ModelInstance(const Model<ModelConfig>& model,
                                          bool set_zero)
    : model(model),
      use_anatomic_pose_(false),
      use_anatomic_pca_(false),
      params(torch::zeros({(int)model.n_params()})),
      has_opengl_mesh(false) {
  if (set_zero) this->set_zero();

  if (model.hand_type != reclib::models::HandType::unknown) {
    anatomic_params =
        torch::zeros({reclib::models::MANOConfigExtra::anatomic_dofs});
  }
  if (ModelConfig::n_hand_pca() > 0) {
    add_pose_mean_ = true;
  } else {
    add_pose_mean_ = false;
  }

  params.set_requires_grad(false);
  // Point cloud after applying shape keys but before lbs (num points,
  // 3)
  _verts_shaped = torch::zeros({ModelConfig::n_verts(), 3});
  _verts_shaped.set_requires_grad(false);

  // Joints after applying shape keys but before lbs (num joints, 3)
  _joints_shaped = torch::zeros({ModelConfig::n_joints(), 3});
  _joints_shaped.set_requires_grad(false);

  // Final deformed point cloud
  _verts = torch::zeros({ModelConfig::n_verts(), 3});
  _verts.set_requires_grad(false);

  _joints = torch::zeros({ModelConfig::n_joints(), 3});
  _joints.set_requires_grad(false);

  // Affine joint transformation, as 3x4 matrices stacked horizontally
  // (bottom row omitted) NOTE: col major
  _joint_rotations = torch::zeros({ModelConfig::n_joints(), 3, 3});
  _joint_rotations.set_requires_grad(false);

  _joint_translations = torch::zeros({ModelConfig::n_joints(), 3});
  _joint_translations.set_requires_grad(false);
}

template <class ModelConfig>
ModelInstance<ModelConfig>::~ModelInstance() {
  if (has_opengl_mesh) {
    gl_instance->destroy_handles();
    gl_instance.free();

    gl_joints->destroy_handles();
    gl_joints.free();

    gl_joint_lines->destroy_handles();
    gl_joint_lines.free();
    // for (unsigned int i = 0; i < gl_joint_lines.size(); i++) {
    //   gl_joint_lines[i]->destroy_handles();
    //   gl_joint_lines[i].free();
    // }
    // for (unsigned int i = 0; i < gl_joints.size(); i++) {
    //   gl_joints[i]->destroy_handles();
    //   gl_joints[i].free();
    // }
  }
}

template <class ModelConfig>
const torch::Tensor& ModelInstance<ModelConfig>::verts() const {
  return _verts;
}

template <class ModelConfig>
torch::Tensor& ModelInstance<ModelConfig>::verts() {
  return _verts;
}

template <class ModelConfig>
const torch::Tensor& ModelInstance<ModelConfig>::verts_shaped() const {
  return _verts_shaped;
}

template <class ModelConfig>
const torch::Tensor& ModelInstance<ModelConfig>::joints() const {
  return _joints;
}

template <class ModelConfig>
void ModelInstance<ModelConfig>::cpu() {
  _RECLIB_ASSERT(model.is_cpu());
  if (params.sizes()[0] > 0) params = params.cpu();
  if (_verts_shaped.sizes()[0] > 0) _verts_shaped = _verts_shaped.cpu();
  if (_joints_shaped.sizes()[0] > 0) _joints_shaped = _joints_shaped.cpu();
  if (_verts.sizes()[0] > 0) _verts = _verts.cpu();
  if (_joint_rotations.sizes()[0] > 0)
    _joint_rotations = _joint_rotations.cpu();
  if (_joint_translations.sizes()[0] > 0)
    _joint_translations = _joint_translations.cpu();
  if (anatomic_params.sizes()[0] > 0) anatomic_params = anatomic_params.cpu();
}

template <class ModelConfig>
void ModelInstance<ModelConfig>::gpu() {
  _RECLIB_ASSERT(model.is_gpu());

  if (params.sizes()[0] > 0) params = params.cuda();
  if (_verts_shaped.sizes()[0] > 0) _verts_shaped = _verts_shaped.cuda();
  if (_joints_shaped.sizes()[0] > 0) _joints_shaped = _joints_shaped.cuda();
  if (_verts.sizes()[0] > 0) _verts = _verts.cuda();
  if (_joint_rotations.sizes()[0] > 0)
    _joint_rotations = _joint_rotations.cuda();
  if (_joint_translations.sizes()[0] > 0)
    _joint_translations = _joint_translations.cuda();
  if (anatomic_params.sizes()[0] > 0) anatomic_params = anatomic_params.cuda();
}

template <class ModelConfig>
void ModelInstance<ModelConfig>::requires_grad(bool val) {
  if (params.sizes()[0] > 0) params.set_requires_grad(val);
  if (_verts_shaped.sizes()[0] > 0) _verts_shaped.set_requires_grad(val);
  if (_joints_shaped.sizes()[0] > 0) _joints_shaped.set_requires_grad(val);
  if (_verts.sizes()[0] > 0) _verts.set_requires_grad(val);
  if (_joint_rotations.sizes()[0] > 0) _joint_rotations.set_requires_grad(val);
  if (_joint_translations.sizes()[0] > 0)
    _joint_translations.set_requires_grad(val);
  if (anatomic_params.sizes()[0] > 0)
    anatomic_params = anatomic_params.set_requires_grad(val);
}

// Main LBS routine
template <class ModelConfig>
void ModelInstance<ModelConfig>::update(bool force_cpu,
                                        bool enable_pose_blendshapes,
                                        bool anatomical_limits,
                                        bool update_meshes) {
  reclib::opengl::Timer t;
  const bool DEBUG = false;
  if (DEBUG) {
    t.begin();
  }

  // Will store full pose params (angle-axis), including hand
  torch::Tensor full_pose = torch::zeros(
      3 * model.n_joints(), torch::TensorOptions().device(params.device()));

  // First rows: Shape params
  // Succeeding rows: linear joint transformations as flattened 3x3
  // rotation matrices rowmajor, only for blend shapes
  torch::Tensor blendshapes_pose = torch::zeros(
      model.n_shape_blends(), torch::TensorOptions().device(params.device()));
  torch::Tensor blendshapes_shape = torch::zeros(
      model.n_pose_blends(), torch::TensorOptions().device(params.device()));

  if (DEBUG) {
    std::cout << "creating tensors: " << t.look_and_reset() << " ms"
              << std::endl;
  }

  if (use_anatomic_pose_) {
    full_pose.zero_();

    torch::Tensor full_apose = apose().clone();
    if (use_anatomic_pca_) {
      unsigned int pcas = hand_pca().sizes()[0];
      torch::tail<reclib::models::MANOConfigExtra::anatomic_dofs - 3>(
          full_apose) += (torch::matmul(torch::leftCols(model.hand_comps, pcas),
                                        hand_pca().clone().unsqueeze(1)))
                             .reshape({-1});
    }
    if (add_pose_mean_) {
      // add mean pose to all joints except the root joint
      full_apose.index_put_(
          {torch::indexing::Slice({3, 3 + model.hand_mean.sizes()[0]})},
          full_apose.index(
              {{torch::indexing::Slice({3, 3 + model.hand_mean.sizes()[0]})}}) +
              model.hand_mean.reshape({-1}));
    }

    int c = 0;

    torch::Tensor limits =
        torch::zeros({reclib::models::MANOConfigExtra::anatomic_dofs, 2},
                     torch::TensorOptions().device(params.device()));
    std::vector<long> apose_indices_;
    for (unsigned int i = 0; i < model.n_explicit_joints(); i++) {
      int dof = reclib::models::MANOConfigExtra::dof_per_anatomic_joint[i];
      for (unsigned int j = 0; j < dof; j++) {
        vec2 l = reclib::models::MANOConfigExtra::anatomic_limits[c + j];
        apose_indices_.push_back(3 * i + j);
        limits.index({c + (int)j, 0}) = l.x();
        limits.index({c + (int)j, 1}) = l.y();
      }
      c += dof;
    }
    torch::Tensor apose_indices =
        torch::from_blob(apose_indices_.data(), {(int)apose_indices_.size()},
                         torch::TensorOptions().dtype(torch::kLong))
            .clone()
            .to(params.device());

    if (anatomical_limits) {
      full_pose.index_put_(
          {apose_indices},
          torch::min(torch::max(full_apose, limits.index({torch::All, 0})),
                     limits.index({torch::All, 1})));
    } else {
      full_pose.index_put_({apose_indices}, full_apose);
    }

  } else {
    // Copy body pose onto full pose
    // the pose consists of all joints in axis-angle representation
    // XXX: beware: the pose of joint i does not influence
    // joint i itself, but instead its children!!
    torch::head(full_pose, 3 * model.n_explicit_joints()) = pose().clone();

    if (DEBUG) {
      std::cout << "assigning pose: " << t.look_and_reset() << " ms"
                << std::endl;
    }

    unsigned int explicit_without_pca =
        model.n_explicit_joints() - ModelConfig::n_duplicate_joints();
    if (model.n_hand_pca_joints() > 0) {
      torch::segment(full_pose, 3 * explicit_without_pca,
                     3 * model.n_hand_pca_joints()) =
          torch::matmul(model.hand_comps, hand_pca().clone().unsqueeze(1))
              .reshape({-1});

      if (ModelConfig::n_duplicate_joints() > 0) {
        torch::segment(full_pose, 3 * explicit_without_pca,
                       3 * model.n_hand_pca_joints()) +=
            torch::segment(pose().clone(), 3 * explicit_without_pca,
                           3 * model.n_hand_pca_joints());
      }
    }
    if (add_pose_mean_) {
      // add mean pose to all joints except the root joint

      full_pose.index_put_(
          {torch::indexing::Slice({3, 3 + model.hand_mean.sizes()[0]})},
          full_pose.index(
              {torch::indexing::Slice({3, 3 + model.hand_mean.sizes()[0]})}) +
              model.hand_mean.reshape({-1}));
    }
  }

  if (DEBUG) {
    std::cout << "PCA: " << t.look_and_reset() << " ms" << std::endl;
  }

  // Copy shape params to FIRST ROWS of blendshape params
  blendshapes_shape = shape().clone();

  if (DEBUG) {
    std::cout << "assigning shape: " << t.look_and_reset() << " ms"
              << std::endl;
  }

  // Convert angle-axis to rotation matrix using rodrigues
  // First load joint_transforms into 3x4 matrix map
  // then fill with rodriguez converted axis-angle
  // TLDR: map axis-angle -> 3x4 matrix

  torch::Tensor reshaped_pose = full_pose.reshape({-1, 3});
  _joint_rotations = batch_rodrigues(reshaped_pose);
  blendshapes_pose = _joint_rotations.index({torch::indexing::Slice(1)}) -
                     torch::eye(3, _joint_rotations.options())
                         .unsqueeze(0)
                         .repeat({_joint_rotations.sizes()[0] - 1, 1, 1});

  if (DEBUG) {
    std::cout << "AA -> Mat: " << t.look_and_reset() << " ms" << std::endl;
  }

  // Apply blend shapes
  {
    // Add shape blend shapes
    // model.blend_shapes (3*#verts, #shape blends)
    // (first rows of) blendshape_params: (#shape blends)
    // apply equation:
    // sum_i beta_i * S_i with S_i
    _verts_shaped =
        model.verts +
        torch::matmul(
            torch::leftCols<ModelConfig::n_shape_blends()>(model.blend_shapes),
            blendshapes_shape.unsqueeze(1))
            .reshape({ModelConfig::n_verts(), 3});
  }

  if (DEBUG) {
    std::cout << "Apply blend shapes: " << t.look_and_reset() << " ms"
              << std::endl;
  }

  // Apply joint regressor
  _joints_shaped = torch::matmul(model.joint_reg, _verts_shaped);

  if (DEBUG) {
    std::cout << "Generate joints: " << t.look_and_reset() << " ms"
              << std::endl;
  }

  if (use_anatomic_pose_) {
    std::vector<torch::Tensor> next;
    for (unsigned int i = 1; i < ModelConfig::n_joints(); ++i) {
      if (i % 3 == 0) {
        next.push_back(
            _verts_shaped
                .index({reclib::models::MANOConfigExtra::tips[(i / 3) - 1]})
                .clone());

      } else {
        next.push_back(_joints_shaped.index({(int)i + 1}).clone());
      }
    }
    torch::Tensor next_v = torch::stack(next).to(params.device());
    torch::Tensor twist =
        next_v - _joints_shaped.index(
                     {torch::indexing::Slice({1, _joints_shaped.sizes()[0]})});

    twist = torch::nn::functional::normalize(
        twist, torch::nn::functional::NormalizeFuncOptions().dim(1));

    torch::Tensor up_vec =
        torch::ones({ModelConfig::n_joints() - 1, 3},
                    torch::TensorOptions().device(params.device()));
    if (model.hand_type == reclib::models::HandType::left) {
      up_vec.index({torch::All, 0}) = -1;
    }
    up_vec.index({torch::indexing::Slice(0, 12), 0}) = 0;
    up_vec.index({torch::indexing::Slice(0, 12), 2}) = 0;

    torch::Tensor bend = torch::cross(twist, up_vec, 1);
    torch::Tensor splay = torch::cross(bend, twist, 1);

    bend = torch::nn::functional::normalize(
        bend, torch::nn::functional::NormalizeFuncOptions().dim(1));
    splay = torch::nn::functional::normalize(
        splay, torch::nn::functional::NormalizeFuncOptions().dim(1));

    // torch::Tensor cross_matrix =
    //     torch::cat({twist, twist, twist}, 1).reshape({bend.sizes()[0], 3,
    //     3});
    torch::Tensor cross_matrix =
        torch::cat({bend, splay, twist}, 1).reshape({bend.sizes()[0], 3, 3});

    // _joint_rotations.index_put_(
    //     {torch::indexing::Slice(1, _joint_rotations.sizes()[0])},
    //     cross_matrix);

    _joint_rotations.index_put_(
        {torch::indexing::Slice(1, _joint_rotations.sizes()[0])},
        torch::bmm(torch::inverse(cross_matrix),
                   torch::bmm(_joint_rotations
                                  .index({torch::indexing::Slice(
                                      1, _joint_rotations.sizes()[0])})
                                  .clone(),
                              cross_matrix)));
    blendshapes_pose = _joint_rotations.index({torch::indexing::Slice(1)}) -
                       torch::eye(3, _joint_rotations.options())
                           .unsqueeze(0)
                           .repeat({_joint_rotations.sizes()[0] - 1, 1, 1});
  }

  if (enable_pose_blendshapes) {
    // HORRIBLY SLOW, like 95% of the time is spent here yikes
    // Add pose blend shapes

    // apply equation: sum_j (rot_j - rot_rest_pose_j)  * P_j
    // = sum_j rot_j * P_j (since rotation in rest pose is 0)
    _verts_shaped.view({ModelConfig::n_verts() * 3, 1}) += torch::matmul(
        torch::rightCols<ModelConfig::n_pose_blends()>(model.blend_shapes),
        blendshapes_pose.reshape({-1, 1}));
  }

  if (DEBUG) {
    std::cout << "Apply pose blend shapes: " << t.look_and_reset() << " ms"
              << std::endl;
  }

  // Inputs: trans(), _joints_shaped
  // Outputs: _joints
  // Input/output: _joint_transforms
  //   (input: left 3x3 should be local rotation mat for joint
  //    output: completed joint local space transform rel global)
  _local_to_global();

  if (DEBUG) {
    std::cout << "Local2global: " << t.look_and_reset() << " ms" << std::endl;
  }

  // Construct a transform for each vertex

  _vert_rotations =
      torch::_sparse_mm(model.weights,
                        _joint_rotations.to(params.device()).reshape({-1, 9}))
          .reshape({-1, 3, 3});
  _vert_translations =
      torch::_sparse_mm(model.weights, _joint_translations.to(params.device()));

  if (DEBUG) {
    std::cout << "creating vert transforms: " << t.look_and_reset() << " ms"
              << std::endl;
  }
  // _SMPLX_PROFILE(lbs weight computation);

  _verts = torch::bmm(_vert_rotations, _verts_shaped.reshape({-1, 3, 1}))
               .reshape({-1, 3}) +
           _vert_translations;

  if (DEBUG) {
    std::cout << "creating verts: " << t.look_and_reset() << " ms" << std::endl;
  }

  if (has_opengl_mesh) {
    gl_model_vertices = verts().clone().cpu().contiguous();
    gl_instance->mesh->geometry.cast<reclib::opengl::GeometryWrapperImpl>()
        ->set(gl_model_vertices.data<float>(), gl_model_vertices.sizes()[0],
              (uint32_t*)gl_model_indices.template data<int32_t>(),
              gl_model_indices.sizes()[0] * 3);
    if (update_meshes) {
      gl_instance->mesh->geometry->update_meshes();
    }

    gl_model_joints = _joints.clone().cpu().contiguous();
    torch::Tensor tmp =
        torch::from_blob((void*)&reclib::models::MANOConfigExtra::tips[0], {5},
                         torch::TensorOptions().dtype(torch::kInt32))
            .toType(torch::kLong);
    if (model.hand_type == reclib::models::HandType::left) {
      tmp.index({1}) = 445;
    }
    torch::Tensor fingertip_vertices =
        _verts.clone().cpu().index({tmp}).contiguous();
    gl_model_joints = torch::cat({gl_model_joints, fingertip_vertices});

    gl_joints->mesh->geometry.cast<reclib::opengl::GeometryWrapperImpl>()->set(
        gl_model_joints.data<float>(), gl_model_joints.sizes()[0]);
    if (update_meshes) {
      gl_joints->mesh->geometry->update_meshes();
    }

    gl_joint_lines->mesh->geometry.cast<reclib::opengl::GeometryWrapperImpl>()
        ->set(gl_model_joints.data<float>(), gl_model_joints.sizes()[0],
              gl_joint_line_indices.data(), gl_joint_line_indices.size());
    if (update_meshes) {
      gl_joint_lines->mesh->geometry->update_meshes();
    }
  }
  if (DEBUG) {
    std::cout << "openGL: " << t.look_and_reset() << " ms" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
  }
}  // namespace modelstorch

template <class ModelConfig>
void ModelInstance<ModelConfig>::_local_to_global() {
  torch::Tensor parents = torch::zeros({ModelConfig::n_joints()},
                                       torch::TensorOptions(torch::kInt64));
  for (int i = 1; i < (int)ModelConfig::n_joints(); ++i) {
    parents.index({i}) = (int)ModelConfig::parent[i];
  }

  torch::Tensor rel_joints = _joints_shaped.clone();
  rel_joints.index({0}) += trans().clone();
  rel_joints.index({torch::indexing::Slice(1)}) -=
      _joints_shaped.index({parents.index({torch::indexing::Slice(1)})});

  torch::Tensor transforms_mat = torch::cat(
      {torch::nn::functional::pad(
           _joint_rotations,
           torch::nn::functional::PadFuncOptions({0, 0, 0, 1})),
       torch::nn::functional::pad(
           rel_joints.reshape({-1, 3, 1}),
           torch::nn::functional::PadFuncOptions({0, 0, 0, 1}).value(1))},
      2);

  std::vector<torch::Tensor> transform_chain;
  transform_chain.push_back(transforms_mat.index({0}));
  for (int i = 1; i < (int)ModelConfig::n_joints(); ++i) {
    torch::Tensor cur_res = torch::matmul(
        transform_chain[ModelConfig::parent[i]], transforms_mat.index({i}));
    transform_chain.push_back(cur_res);
  }

  torch::Tensor transforms = torch::stack(transform_chain, 0);

  _joint_rotations = transforms.index(
      {torch::All, torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});
  _joint_translations =
      transforms.index({torch::All, torch::indexing::Slice(0, 3), 3});

  _joints = _joint_translations.clone();
  _joint_translations -=
      torch::bmm(_joint_rotations.clone(), _joints_shaped.reshape({-1, 3, 1}))
          .squeeze(-1);
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

  } else if (!m.initialized() &&
             reclib::opengl::Material::valid("modelinst_default")) {
    mat = reclib::opengl::Material::find("modelinst_default");
  } else {
    mat = m;
  }

  reclib::opengl::Shader shader;
  if (!s.initialized() && !reclib::opengl::Shader::valid("modelinst_default")) {
    reclib::opengl::Shader def("modelinst_default", "MVP_norm_color3.vs",
                               "color3PV.fs");
    shader = def;
  } else if (!s.initialized() &&
             reclib::opengl::Shader::valid("modelinst_default")) {
    shader = reclib::opengl::Shader::find("modelinst_default");
  } else {
    shader = s;
  }

  unsigned int id = 0;
  for (; id < std::numeric_limits<unsigned int>::max(); id++) {
    if (!reclib::opengl::Drawelement::valid("modelinst_" + std::to_string(id)))
      break;
  }

  gl_model_indices = model.faces.clone().cpu();
  gl_model_vertices = verts().clone().cpu().contiguous();

  reclib::opengl::Drawelement d =
      reclib::opengl::DrawelementImpl::from_geometry_wrapper(
          "modelinst_" + std::to_string(id), shader, mat, true,
          gl_model_vertices.data<float>(), verts().sizes()[0],
          (uint32_t*)gl_model_indices.template data<int32_t>(),
          gl_model_indices.sizes()[0] * 3);
  gl_instance = d;

  torch::Tensor per_vertex_colors =
      torch::matmul(model.weights.clone().cpu(),
                    reclib::modelstorch::auto_color_table(model.n_joints()));

  gl_instance->mesh->geometry->add_attribute_float(
      "color", per_vertex_colors.data<float>(), per_vertex_colors.sizes()[0], 3,
      false);
  gl_instance->mesh->geometry->update_meshes();

  if (!reclib::opengl::Material::valid("modelinst_joints_material")) {
    reclib::opengl::Material sphere_material("modelinst_joints_material");
    sphere_material->vec4_map["color"] = vec4(1.f, 0.5f, 0.0f, 1.f);
  }
  if (!reclib::opengl::Material::valid("modelinst_joint_lines_material")) {
    reclib::opengl::Material joint_lines_material(
        "modelinst_joint_lines_material");
    // joint_lines_material->vec4_map["color"] = vec4(0.5f, 0.7f, 0.8f, 1.f);
    joint_lines_material->vec4_map["color"] = vec4(0.0f, 0.2f, 0.8f, 1.f);
  }
  if (!reclib::opengl::Shader::valid("modelinst_joint_shader")) {
    reclib::opengl::Shader joint_shader("modelinst_joint_shader", "MVP.vs",
                                        "color4Uniform.fs");
  }

  gl_model_joints = _joints.clone().cpu().contiguous();
  torch::Tensor tmp =
      torch::from_blob((void*)&reclib::models::MANOConfigExtra::tips[0], {5},
                       torch::TensorOptions().dtype(torch::kInt32))
          .toType(torch::kLong);
  if (model.hand_type == reclib::models::HandType::left) {
    tmp.index({1}) = 445;
  }
  torch::Tensor fingertip_vertices =
      _verts.clone().cpu().index({tmp}).contiguous();
  gl_model_joints = torch::cat({gl_model_joints, fingertip_vertices});

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
            sphere_material, false, gl_model_joints.data<float>(),
            gl_model_joints.sizes()[0]);
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
  gl_joint_line_indices.push_back(3);
  gl_joint_line_indices.push_back(model.n_joints());
  gl_joint_line_indices.push_back(6);
  gl_joint_line_indices.push_back(model.n_joints() + 1);
  gl_joint_line_indices.push_back(9);
  gl_joint_line_indices.push_back(model.n_joints() + 2);
  gl_joint_line_indices.push_back(12);
  gl_joint_line_indices.push_back(model.n_joints() + 3);
  gl_joint_line_indices.push_back(15);
  gl_joint_line_indices.push_back(model.n_joints() + 4);

  {
    reclib::opengl::Drawelement d =
        reclib::opengl::DrawelementImpl::from_geometry_wrapper(
            "modelinst_lines_" + std::to_string(id), joint_shader,
            joint_lines_material, false, gl_model_joints.data<float>(),
            gl_model_joints.sizes()[0], gl_joint_line_indices.data(),
            gl_joint_line_indices.size());
    d->add_pre_draw_func("size", [&]() { glLineWidth(6.f); });
    d->add_post_draw_func("size", [&]() { glLineWidth(1.f); });
    d->mesh->primitive_type = GL_LINES;
    d->mesh->geometry->update_meshes();
    gl_joint_lines = d;
  }

  reclib::opengl::GroupedDrawelements(
      "mano_" + std::to_string(id),
      std::vector<reclib::opengl::Drawelement>(
          {gl_joint_lines, gl_joints, gl_instance}));

  has_opengl_mesh = true;
}

template <class ModelConfig>
void ModelInstance<ModelConfig>::save_obj(const std::string& path) const {
  auto& cur_verts = verts();
  if (cur_verts.sizes()[0] == 0) return;
  std::ofstream ofs(path);
  ofs << "# Generated by SMPL-X_cpp"
      << "\n";
  ofs << std::fixed << std::setprecision(6) << "o smplx\n";
  for (int i = 0; i < (int)model.n_verts(); ++i) {
    ofs << "v " << cur_verts.index({i, 0}).item().toFloat() << " "
        << cur_verts.index({i, 1}).item().toFloat() << " "
        << cur_verts.index({i, 2}).item().toFloat() << "\n";
  }
  ofs << "s 1\n";
  for (int i = 0; i < (int)model.n_faces(); ++i) {
    ofs << "f " << model.faces.index({i, 0}).item().toInt() + 1 << " "
        << model.faces.index({i, 1}).item().toInt() + 1 << " "
        << model.faces.index({i, 2}).item().toInt() + 1 << "\n";
  }
  ofs.close();
}

// Instantiation
template class ModelInstance<reclib::models::SMPLConfig>;
template class ModelInstance<reclib::models::MANOConfig>;
template class ModelInstance<reclib::models::MANOConfigPCA>;
template class ModelInstance<reclib::models::MANOConfigAnglePCA>;
template class ModelInstance<reclib::models::MANOConfigPCAGeneric<45>>;
template class ModelInstance<reclib::models::MANOConfigPCAGeneric<23>>;
template class ModelInstance<reclib::models::MANOConfigAnglePCAGeneric<45>>;
template class ModelInstance<reclib::models::MANOConfigAnglePCAGeneric<23>>;
}  // namespace modelstorch
}  // namespace reclib

#endif  // HAS_DNN_MODULE
#endif  //__unix__