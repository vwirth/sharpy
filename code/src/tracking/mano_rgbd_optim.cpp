#include <reclib/tracking/mano_rgbd_optim.h>

#if HAS_DNN_MODULE

#include <ATen/TensorIndexing.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <math.h>
#include <torch/torch.h>

std::vector<long> reclib::tracking::apose_indices_ = {
    3,  4,  6,  9,  12, 13, 15, 18, 21, 22, 24, 27,
    30, 31, 33, 36, 39, 40, 41, 42, 43, 44, 45};

std::vector<long> reclib::tracking::apose_zero_indices_ = {
    5,  7,  8,  10, 11, 14, 16, 17, 19, 20,
    23, 25, 26, 28, 29, 32, 34, 35, 37, 38};

static void local_to_global(torch::Tensor trans, torch::Tensor& joints,
                            torch::Tensor& joint_rotations,
                            torch::Tensor& joint_translations) {
  torch::Tensor parents = torch::zeros({reclib::models::MANOConfig::n_joints()},
                                       torch::TensorOptions(torch::kInt64));
  for (int i = 1; i < (int)reclib::models::MANOConfig::n_joints(); ++i) {
    parents.index({i}) = (int)reclib::models::MANOConfig::parent[i];
  }

  torch::Tensor rel_joints = joints.clone();
  rel_joints.index({0}) += trans.clone();
  rel_joints.index({torch::indexing::Slice(1)}) -=
      joints.index({parents.index({torch::indexing::Slice(1)})});

  torch::Tensor transforms_mat = torch::cat(
      {torch::nn::functional::pad(
           joint_rotations,
           torch::nn::functional::PadFuncOptions({0, 0, 0, 1})),
       torch::nn::functional::pad(
           rel_joints.reshape({-1, 3, 1}),
           torch::nn::functional::PadFuncOptions({0, 0, 0, 1}).value(1))},
      2);

  std::vector<torch::Tensor> transform_chain;
  transform_chain.push_back(transforms_mat.index({0}));
  for (int i = 1; i < (int)reclib::models::MANOConfig::n_joints(); ++i) {
    torch::Tensor cur_res =
        torch::matmul(transform_chain[reclib::models::MANOConfig::parent[i]],
                      transforms_mat.index({i}));
    transform_chain.push_back(cur_res);
  }

  torch::Tensor transforms = torch::stack(transform_chain, 0);

  joint_rotations = transforms.index(
      {torch::All, torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});
  joint_translations =
      transforms.index({torch::All, torch::indexing::Slice(0, 3), 3});

  torch::Tensor tmp = joints.clone();
  joints = joint_translations.clone();
  joint_translations -=
      torch::bmm(joint_rotations.clone(), tmp.reshape({-1, 3, 1})).squeeze(-1);
}

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor> reclib::tracking::torch_lbs(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor pose,
    bool add_pose_mean) {
  // Will store full pose params (angle-axis), including hand
  torch::Tensor full_pose = torch::zeros(
      3 * model.n_joints(), torch::TensorOptions().device(trans.device()));

  // First rows: Shape params
  // Succeeding rows: linear joint transformations as flattened 3x3
  // rotation matrices rowmajor, only for blend shapes
  torch::Tensor blendshapes_pose = torch::zeros(
      model.n_shape_blends(), torch::TensorOptions().device(trans.device()));
  torch::Tensor blendshapes_shape = torch::zeros(
      model.n_pose_blends(), torch::TensorOptions().device(trans.device()));

  // Copy body pose onto full pose
  // the pose consists of all joints in axis-angle representation
  // XXX: beware: the pose of joint i does not influence
  // joint i itself, but instead its children!!
  torch::head(full_pose, 3 * model.n_explicit_joints()) =
      torch::cat({rot, pose});

  if (add_pose_mean) {
    // add mean pose to all joints except the root joint
    torch::segment(full_pose, 3, 3 * (model.n_explicit_joints() - 1)) +=
        model.hand_mean.reshape({-1});
  }

  // Copy shape params to FIRST ROWS of blendshape params
  blendshapes_shape = shape.clone();
  torch::Tensor reshaped_pose = full_pose.reshape({-1, 3});
  torch::Tensor joint_rotations =
      reclib::modelstorch::batch_rodrigues(reshaped_pose);
  torch::Tensor joint_translations =
      torch::zeros({joint_rotations.sizes()[0], 3});
  blendshapes_pose = joint_rotations.index({torch::indexing::Slice(1)}) -
                     torch::eye(3, joint_rotations.options())
                         .unsqueeze(0)
                         .repeat({joint_rotations.sizes()[0] - 1, 1, 1});

  torch::Tensor verts;
  // Apply blend shapes
  {
    verts = model.verts +
            torch::matmul(
                torch::leftCols<MODEL::n_shape_blends()>(model.blend_shapes),
                blendshapes_shape.unsqueeze(1))
                .reshape({MODEL::n_verts(), 3});
  }

  torch::Tensor joints = torch::matmul(model.joint_reg, verts);

  {
    verts.view({MODEL::n_verts() * 3, 1}) += torch::matmul(
        torch::rightCols<MODEL::n_pose_blends()>(model.blend_shapes),
        blendshapes_pose.reshape({-1, 1}));
  }

  local_to_global(trans, joints, joint_rotations, joint_translations);

  torch::Tensor vert_rotations =
      torch::_sparse_mm(model.weights,
                        joint_rotations.to(trans.device()).reshape({-1, 9}))
          .reshape({-1, 3, 3});
  torch::Tensor vert_translations =
      torch::_sparse_mm(model.weights, joint_translations.to(trans.device()));

  verts =
      torch::bmm(vert_rotations, verts.reshape({-1, 3, 1})).reshape({-1, 3}) +
      vert_translations;
  return std::make_pair(verts, joints);
}

// template instantiation
template std::pair<torch::Tensor, torch::Tensor> reclib::tracking::torch_lbs(
    const reclib::modelstorch::Model<reclib::models::MANOConfig>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor> reclib::tracking::torch_lbs(
    const reclib::modelstorch::Model<reclib::models::MANOConfigAnglePCA>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor> reclib::tracking::torch_lbs(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<45>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor> reclib::tracking::torch_lbs(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<23>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor> reclib::tracking::torch_lbs_pca(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor pca,
    bool add_pose_mean) {
  // Will store full pose params (angle-axis), including hand
  torch::Tensor full_pose = torch::zeros(
      3 * model.n_joints(), torch::TensorOptions().device(trans.device()));

  // First rows: Shape params
  // Succeeding rows: linear joint transformations as flattened 3x3
  // rotation matrices rowmajor, only for blend shapes
  torch::Tensor blendshapes_pose = torch::zeros(
      model.n_shape_blends(), torch::TensorOptions().device(trans.device()));
  torch::Tensor blendshapes_shape = torch::zeros(
      model.n_pose_blends(), torch::TensorOptions().device(trans.device()));

  // Copy body pose onto full pose
  // the pose consists of all joints in axis-angle representation
  // XXX: beware: the pose of joint i does not influence
  // joint i itself, but instead its children!!

  full_pose.index_put_({torch::indexing::Slice(0, 3)}, rot);
  int n_pca_comps = pca.sizes()[0];
  torch::Tensor pca2pose =
      torch::matmul(model.hand_comps.index(
                        {torch::All, torch::indexing::Slice(0, n_pca_comps)}),
                    pca.clone().unsqueeze(1));
  full_pose.index_put_(
      {torch::indexing::Slice(3, model.n_explicit_joints() * 3)},
      pca2pose.reshape({-1}));

  if (add_pose_mean) {
    // add mean pose to all joints except the root joint
    torch::segment(full_pose, 3, 3 * (model.n_explicit_joints() - 1)) +=
        model.hand_mean.reshape({-1});
  }

  // Copy shape params to FIRST ROWS of blendshape params
  blendshapes_shape = shape.clone();
  torch::Tensor reshaped_pose = full_pose.reshape({-1, 3});
  torch::Tensor joint_rotations =
      reclib::modelstorch::batch_rodrigues(reshaped_pose);
  torch::Tensor joint_translations =
      torch::zeros({joint_rotations.sizes()[0], 3});
  blendshapes_pose = joint_rotations.index({torch::indexing::Slice(1)}) -
                     torch::eye(3, joint_rotations.options())
                         .unsqueeze(0)
                         .repeat({joint_rotations.sizes()[0] - 1, 1, 1});

  torch::Tensor verts;
  // Apply blend shapes
  {
    verts = model.verts +
            torch::matmul(
                torch::leftCols<MODEL::n_shape_blends()>(model.blend_shapes),
                blendshapes_shape.unsqueeze(1))
                .reshape({MODEL::n_verts(), 3});
  }

  torch::Tensor joints = torch::matmul(model.joint_reg, verts);

  {
    verts.view({MODEL::n_verts() * 3, 1}) += torch::matmul(
        torch::rightCols<MODEL::n_pose_blends()>(model.blend_shapes),
        blendshapes_pose.reshape({-1, 1}));
  }

  local_to_global(trans, joints, joint_rotations, joint_translations);

  torch::Tensor vert_rotations =
      torch::_sparse_mm(model.weights,
                        joint_rotations.to(trans.device()).reshape({-1, 9}))
          .reshape({-1, 3, 3});
  torch::Tensor vert_translations =
      torch::_sparse_mm(model.weights, joint_translations.to(trans.device()));

  verts =
      torch::bmm(vert_rotations, verts.reshape({-1, 3, 1})).reshape({-1, 3}) +
      vert_translations;
  return std::make_pair(verts, joints);
}

// template instantiation
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca(
    const reclib::modelstorch::Model<reclib::models::MANOConfig>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca(
    const reclib::modelstorch::Model<reclib::models::MANOConfigAnglePCA>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<45>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<23>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, bool add_pose_mean);

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca_anatomic(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor pca,
    torch::Tensor cross_matrix, bool add_pose_mean) {
  // Will store full pose params (angle-axis), including hand
  torch::Tensor full_pose = torch::zeros(
      3 * model.n_joints(), torch::TensorOptions().device(trans.device()));

  // First rows: Shape params
  // Succeeding rows: linear joint transformations as flattened 3x3
  // rotation matrices rowmajor, only for blend shapes
  torch::Tensor blendshapes_pose = torch::zeros(
      model.n_shape_blends(), torch::TensorOptions().device(trans.device()));
  torch::Tensor blendshapes_shape = torch::zeros(
      model.n_pose_blends(), torch::TensorOptions().device(trans.device()));

  // Copy body pose onto full pose
  // the pose consists of all joints in axis-angle representation
  // XXX: beware: the pose of joint i does not influence
  // joint i itself, but instead its children!!

  full_pose.zero_();
  full_pose.index_put_({torch::indexing::Slice(0, 3)}, rot);

  torch::Tensor apose =
      torch::zeros(23, torch::TensorOptions().device(trans.device()));
  unsigned int pcas = pca.sizes()[0];
  torch::tail<reclib::models::MANOConfigExtra::anatomic_dofs - 3>(apose) +=
      (torch::matmul(torch::leftCols(model.hand_comps, pcas),
                     pca.clone().unsqueeze(1)))
          .reshape({-1});

  if (add_pose_mean) {
    // add mean pose to all joints except the root joint
    apose.index_put_(
        {torch::indexing::Slice({3, 3 + model.hand_mean.sizes()[0]})},
        apose.index(
            {{torch::indexing::Slice({3, 3 + model.hand_mean.sizes()[0]})}}) +
            model.hand_mean.reshape({-1}));
  }

  torch::Tensor apose_indices =
      torch::from_blob(apose_indices_.data(), {(int)apose_indices_.size()},
                       torch::TensorOptions().dtype(torch::kLong))
          .clone()
          .to(trans.device());

  full_pose.index_put_({apose_indices}, apose);

  // Copy shape params to FIRST ROWS of blendshape params
  blendshapes_shape = shape.clone();
  torch::Tensor reshaped_pose = full_pose.reshape({-1, 3});
  torch::Tensor joint_rotations =
      reclib::modelstorch::batch_rodrigues(reshaped_pose);
  torch::Tensor joint_translations =
      torch::zeros({joint_rotations.sizes()[0], 3});

  torch::Tensor verts;
  // Apply blend shapes
  {
    verts = model.verts +
            torch::matmul(
                torch::leftCols<MODEL::n_shape_blends()>(model.blend_shapes),
                blendshapes_shape.unsqueeze(1))
                .reshape({MODEL::n_verts(), 3});
  }

  torch::Tensor joints = torch::matmul(model.joint_reg, verts);

  if (cross_matrix.sizes().size() == 0 || cross_matrix.sizes()[0] == 0) {
    cross_matrix = reclib::tracking::compute_apose_matrix(model, shape);
  }

  joint_rotations.index_put_(
      {torch::indexing::Slice(1, joint_rotations.sizes()[0])},
      torch::bmm(torch::inverse(cross_matrix),
                 torch::bmm(joint_rotations
                                .index({torch::indexing::Slice(
                                    1, joint_rotations.sizes()[0])})
                                .clone(),
                            cross_matrix)));
  blendshapes_pose = joint_rotations.index({torch::indexing::Slice(1)}) -
                     torch::eye(3, joint_rotations.options())
                         .unsqueeze(0)
                         .repeat({joint_rotations.sizes()[0] - 1, 1, 1});

  {
    verts.view({MODEL::n_verts() * 3, 1}) += torch::matmul(
        torch::rightCols<MODEL::n_pose_blends()>(model.blend_shapes),
        blendshapes_pose.reshape({-1, 1}));
  }

  local_to_global(trans, joints, joint_rotations, joint_translations);

  torch::Tensor vert_rotations =
      torch::_sparse_mm(model.weights,
                        joint_rotations.to(trans.device()).reshape({-1, 9}))
          .reshape({-1, 3, 3});
  torch::Tensor vert_translations =
      torch::_sparse_mm(model.weights, joint_translations.to(trans.device()));

  verts =
      torch::bmm(vert_rotations, verts.reshape({-1, 3, 1})).reshape({-1, 3}) +
      vert_translations;
  return std::make_pair(verts, joints);
}

template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca_anatomic(
    const reclib::modelstorch::Model<reclib::models::MANOConfig>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pca, torch::Tensor cross_matrix, bool add_pose_mean);

template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca_anatomic(
    const reclib::modelstorch::Model<reclib::models::MANOConfigAnglePCA>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pca, torch::Tensor cross_matrix, bool add_pose_mean);

template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca_anatomic(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<45>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pca, torch::Tensor cross_matrix, bool add_pose_mean);

template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_pca_anatomic(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<23>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pca, torch::Tensor cross_matrix, bool add_pose_mean);

template <typename MODEL>
torch::Tensor reclib::tracking::compute_apose_matrix(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor shape) {
  torch::Tensor blendshapes_shape = torch::zeros(
      model.n_pose_blends(), torch::TensorOptions().device(shape.device()));

  // Copy shape params to FIRST ROWS of blendshape params
  blendshapes_shape = shape.clone();

  torch::Tensor verts;
  // Apply blend shapes
  {
    verts = model.verts +
            torch::matmul(
                torch::leftCols<MODEL::n_shape_blends()>(model.blend_shapes),
                blendshapes_shape.unsqueeze(1))
                .reshape({MODEL::n_verts(), 3});
  }

  torch::Tensor joints = torch::matmul(model.joint_reg, verts);

  torch::Tensor cross_matrix;
  {
    std::vector<torch::Tensor> next;
    for (unsigned int i = 1; i < MODEL::n_joints(); ++i) {
      if (i % 3 == 0) {
        next.push_back(
            verts.index({reclib::models::MANOConfigExtra::tips[(i / 3) - 1]})
                .clone());

      } else {
        next.push_back(joints.index({(int)i + 1}).clone());
      }
    }
    torch::Tensor next_v = torch::stack(next).to(shape.device());
    torch::Tensor twist =
        next_v - joints.index({torch::indexing::Slice({1, joints.sizes()[0]})});

    twist = torch::nn::functional::normalize(
        twist, torch::nn::functional::NormalizeFuncOptions().dim(1));

    torch::Tensor up_vec =
        torch::ones({MODEL::n_joints() - 1, 3},
                    torch::TensorOptions().device(shape.device()));
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

    cross_matrix =
        torch::cat({bend, splay, twist}, 1).reshape({bend.sizes()[0], 3, 3});
  }
  return cross_matrix;
}

// template instantiation
template torch::Tensor reclib::tracking::compute_apose_matrix(
    const reclib::modelstorch::Model<reclib::models::MANOConfig>& model,
    torch::Tensor shape);
template torch::Tensor reclib::tracking::compute_apose_matrix(
    const reclib::modelstorch::Model<reclib::models::MANOConfigAnglePCA>& model,
    torch::Tensor shape);
template torch::Tensor reclib::tracking::compute_apose_matrix(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<45>>& model,
    torch::Tensor shape);
template torch::Tensor reclib::tracking::compute_apose_matrix(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<23>>& model,
    torch::Tensor shape);

template <typename MODEL>
torch::Tensor reclib::tracking::apose2pose(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor shape,
    torch::Tensor rot_and_pose) {
  torch::Tensor cross_matrix = compute_apose_matrix(model, shape);

  torch::Tensor full_pose =
      torch::zeros(3 * model.n_joints(),
                   torch::TensorOptions().device(rot_and_pose.device()));
  full_pose.index_put_({torch::indexing::Slice(0, 3)},
                       rot_and_pose.index({torch::indexing::Slice(0, 3)}));

  torch::Tensor apose_indices =
      torch::from_blob(apose_indices_.data(), {(int)apose_indices_.size()},
                       torch::TensorOptions().dtype(torch::kLong))
          .clone()
          .to(rot_and_pose.device());
  full_pose.index_put_({apose_indices},
                       rot_and_pose.index({torch::indexing::Slice(3, 26)}));

  torch::Tensor reshaped_pose = full_pose.reshape({-1, 3});
  torch::Tensor joint_rotations =
      reclib::modelstorch::batch_rodrigues(reshaped_pose);

  joint_rotations.index_put_(
      {torch::indexing::Slice(1, joint_rotations.sizes()[0])},
      torch::bmm(torch::inverse(cross_matrix),
                 torch::bmm(joint_rotations
                                .index({torch::indexing::Slice(
                                    1, joint_rotations.sizes()[0])})
                                .clone(),
                            cross_matrix)));

  torch::Tensor pose = batch_mat2aa(joint_rotations);
  return pose;
}

template torch::Tensor reclib::tracking::apose2pose(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<45>>& model,
    torch::Tensor shape, torch::Tensor rot_and_pose);
template torch::Tensor reclib::tracking::apose2pose(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<23>>& model,
    torch::Tensor shape, torch::Tensor rot_and_pose);

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor> reclib::tracking::torch_lbs_anatomic(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor apose,
    torch::Tensor cross_matrix, bool add_pose_mean) {
  // Will store full pose params (angle-axis), including hand
  torch::Tensor full_pose = torch::zeros(
      3 * model.n_joints(), torch::TensorOptions().device(trans.device()));

  // First rows: Shape params
  // Succeeding rows: linear joint transformations as flattened 3x3
  // rotation matrices rowmajor, only for blend shapes
  torch::Tensor blendshapes_pose = torch::zeros(
      model.n_shape_blends(), torch::TensorOptions().device(trans.device()));
  torch::Tensor blendshapes_shape = torch::zeros(
      model.n_pose_blends(), torch::TensorOptions().device(trans.device()));

  // Copy body pose onto full pose
  // the pose consists of all joints in axis-angle representation
  // XXX: beware: the pose of joint i does not influence
  // joint i itself, but instead its children!!
  //   torch::head(full_pose, 3 * model.n_explicit_joints()) =
  //       torch::cat({rot, pose});

  //   if (use_anatomic_pca_) {
  //     unsigned int pcas = hand_pca().sizes()[0];
  //     torch::tail<reclib::models::MANOConfigExtra::anatomic_dofs - 3>(
  //         full_apose) =
  //         torch::matmul(torch::leftCols(model.hand_comps, pcas), hand_pca())
  //         + model.hand_mean;
  //   }

  full_pose.index_put_({torch::indexing::Slice(0, 3)}, rot);
  //   std::vector<long> apose_indices_;
  //   for (unsigned int i = 1; i < model.n_explicit_joints(); i++) {
  //     int dof = reclib::models::MANOConfigExtra::dof_per_anatomic_joint[i];
  //     for (unsigned int j = 0; j < dof; j++) {
  //       apose_indices_.push_back(3 * i + j);
  //     }
  //   }
  torch::Tensor apose_indices =
      torch::from_blob(apose_indices_.data(), {(int)apose_indices_.size()},
                       torch::TensorOptions().dtype(torch::kLong))
          .clone()
          .to(apose.device());

  full_pose.index_put_({apose_indices}, apose);
  // std::cout << "full_pose: " << full_pose << std::endl;

  if (add_pose_mean) {
    // add mean pose to all joints except the root joint
    torch::segment(full_pose, 3, 3 * (model.n_explicit_joints() - 1)) +=
        model.hand_mean.reshape({-1});
  }

  // Copy shape params to FIRST ROWS of blendshape params
  blendshapes_shape = shape.clone();
  torch::Tensor reshaped_pose = full_pose.reshape({-1, 3});
  torch::Tensor joint_rotations =
      reclib::modelstorch::batch_rodrigues(reshaped_pose);
  torch::Tensor joint_translations =
      torch::zeros({joint_rotations.sizes()[0], 3});

  torch::Tensor verts;
  // Apply blend shapes
  {
    verts = model.verts +
            torch::matmul(
                torch::leftCols<MODEL::n_shape_blends()>(model.blend_shapes),
                blendshapes_shape.unsqueeze(1))
                .reshape({MODEL::n_verts(), 3});
  }

  torch::Tensor joints = torch::matmul(model.joint_reg, verts);

  if (cross_matrix.sizes().size() == 0 || cross_matrix.sizes()[0] == 0) {
    cross_matrix = reclib::tracking::compute_apose_matrix(model, shape);
  }

  joint_rotations.index_put_(
      {torch::indexing::Slice(1, joint_rotations.sizes()[0])},
      torch::bmm(torch::inverse(cross_matrix),
                 torch::bmm(joint_rotations
                                .index({torch::indexing::Slice(
                                    1, joint_rotations.sizes()[0])})
                                .clone(),
                            cross_matrix)));
  blendshapes_pose = joint_rotations.index({torch::indexing::Slice(1)}) -
                     torch::eye(3, joint_rotations.options())
                         .unsqueeze(0)
                         .repeat({joint_rotations.sizes()[0] - 1, 1, 1});

  {
    verts.view({MODEL::n_verts() * 3, 1}) += torch::matmul(
        torch::rightCols<MODEL::n_pose_blends()>(model.blend_shapes),
        blendshapes_pose.reshape({-1, 1}));
  }

  local_to_global(trans, joints, joint_rotations, joint_translations);

  torch::Tensor vert_rotations =
      torch::_sparse_mm(model.weights,
                        joint_rotations.to(trans.device()).reshape({-1, 9}))
          .reshape({-1, 3, 3});
  torch::Tensor vert_translations =
      torch::_sparse_mm(model.weights, joint_translations.to(trans.device()));

  verts =
      torch::bmm(vert_rotations, verts.reshape({-1, 3, 1})).reshape({-1, 3}) +
      vert_translations;
  return std::make_pair(verts, joints);
}

// template instantiation
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_anatomic(
    const reclib::modelstorch::Model<reclib::models::MANOConfig>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, torch::Tensor cross_matrix, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_anatomic(
    const reclib::modelstorch::Model<reclib::models::MANOConfigAnglePCA>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, torch::Tensor cross_matrix, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_anatomic(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<45>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, torch::Tensor cross_matrix, bool add_pose_mean);
template std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::torch_lbs_anatomic(
    const reclib::modelstorch::Model<
        reclib::models::MANOConfigAnglePCAGeneric<23>>& model,
    torch::Tensor trans, torch::Tensor rot, torch::Tensor shape,
    torch::Tensor pose, torch::Tensor cross_matrix, bool add_pose_mean);

torch::Tensor reclib::tracking::batch_mat2aa(torch::Tensor batch_mat) {
  return reclib::tracking::batch_quat2aa(
      reclib::tracking::batch_mat2quat(batch_mat));
}

torch::Tensor reclib::tracking::batch_quat2aa(torch::Tensor quat) {
  torch::Tensor norms = torch::norm(
      quat.index({"...", torch::indexing::Slice({1, quat.sizes()[1]})}), 2, -1);

  torch::Tensor half_angles = torch::atan2(norms, quat.index({"...", 0}).abs());

  torch::Tensor angles = 2 * half_angles;
  float eps = 1e-6f;
  torch::Tensor small_norm = norms.abs() < eps;
  torch::Tensor axis_angle = torch::empty(
      {quat.sizes()[0], 3}, torch::TensorOptions().device(quat.device()));
  axis_angle.index_put_(
      {small_norm}, torch::zeros({small_norm.sum().item<int>(), 3},
                                 torch::TensorOptions().device(quat.device())));

  torch::Tensor negative_real = quat.index({"...", 0}) < 0;
  norms.index_put_({negative_real}, norms.index({negative_real}) * -1);

  torch::Tensor not_small_norm = torch::logical_not(small_norm);

  torch::Tensor normalized_quat =
      quat.index({"...", torch::indexing::Slice({1, quat.sizes()[1]})}) /
      norms.unsqueeze(1);

  torch::Tensor tmp = angles.unsqueeze(1) * normalized_quat;
  axis_angle.index_put_({not_small_norm}, tmp.index({not_small_norm}));

  //   for (int i = 0; i < quat.sizes()[0]; i++) {
  //     std::cout << "[TORCH] i: " << i << " quat: " << quat.index({i})
  //               << std::endl;
  //     std::cout << "[TORCH] i: " << i << " norm: " << norms.index({i})
  //               << std::endl;
  //     std::cout << "[TORCH] i: " << i
  //               << " half_angles: " << half_angles.index({i}) << std::endl;
  //     std::cout << "[TORCH] i: " << i << " AA: " << axis_angle.index({i})
  //               << std::endl;
  //   }

  //   torch::Tensor q = quat.index({0}).clone().contiguous();
  //   vec4 q_vec = reclib::dnn::torch2eigen<float>(q);
  //   Eigen::Quaternionf quat_eigen;
  //   quat_eigen.coeffs() = q_vec;
  //   Eigen::AngleAxisf eigen_aa(quat_eigen);
  //   std::cout << "q: " << q << std::endl;
  //   std::cout << "quat_eigen: " << quat_eigen.coeffs() << std::endl;
  //   std::cout << "axis_angle: " << axis_angle << std::endl;
  //   std::cout << "eigen aa: " << eigen_aa.angle() * eigen_aa.axis() <<
  //   std::endl;

  return axis_angle;

  //   torch::Tensor small_angles = angles.abs() < eps;

  //   small_angles = torch::logical_and(small_angles, small_norm);
  //   torch::Tensor not_small_angles = torch::logical_or(small_angles);
  //   std::cout << "small_angles: " << small_angles << std::endl;
  //   std::cout << "!small_angles: " << torch::logical_not(small_angles)
  //             << std::endl;
  //   std::cout << "hae" << std::endl;

  //   torch::Tensor sin_half_angles_over_angles = torch::empty_like(angles);
  //   sin_half_angles_over_angles.index_put_(
  //       {not_small_angles},
  //       (torch::sin(half_angles.index({not_small_angles})) /
  //                            angles.index({not_small_angles})));

  //   sin_half_angles_over_angles.index_put_(
  //       {small_angles},
  //       (0.5 -
  //        (angles.index({small_angles}) * angles.index({small_angles}))
  //        / 48.f));
  //   return quat.index({"...", torch::indexing::Slice({1, quat.sizes()[1]})})
  //   /
  //          sin_half_angles_over_angles;
}

torch::Tensor reclib::tracking::batch_mat2quat(torch::Tensor batch_mat) {
  _RECLIB_ASSERT_EQ(batch_mat.sizes().size(), 3);
  _RECLIB_ASSERT_EQ(batch_mat.sizes()[1], 3);
  _RECLIB_ASSERT_EQ(batch_mat.sizes()[2], 3);
  int batch_size = batch_mat.sizes()[0];
  std::vector<torch::Tensor> ret =
      torch::unbind(batch_mat.reshape({batch_size, 9}), -1);

  torch::Tensor m00 = ret[0];
  torch::Tensor m01 = ret[1];
  torch::Tensor m02 = ret[2];
  torch::Tensor m10 = ret[3];
  torch::Tensor m11 = ret[4];
  torch::Tensor m12 = ret[5];
  torch::Tensor m20 = ret[6];
  torch::Tensor m21 = ret[7];
  torch::Tensor m22 = ret[8];

  torch::Tensor tmp = torch::stack(
      {
          1 + m00 + m11 + m22,
          1 + m00 - m11 - m22,
          1 - m00 + m11 - m22,
          1 - m00 - m11 + m22,
      },
      -1);
  torch::Tensor q_abs = torch::zeros_like(tmp);
  torch::Tensor pos_mask = tmp > 0;
  q_abs.index_put_({pos_mask}, torch::sqrt(tmp.index({pos_mask})));

  torch::Tensor quat_by_rijk = torch::stack(
      {torch::stack({torch::pow(q_abs.index({"...", 0}), 2), m21 - m12,
                     m02 - m20, m10 - m01},
                    -1),

       torch::stack({m21 - m12, torch::pow(q_abs.index({"...", 1}), 2),
                     m10 + m01, m02 + m20},
                    -1),

       torch::stack({m02 - m20, m10 + m01,
                     torch::pow(q_abs.index({"...", 2}), 2), m12 + m21},
                    -1),

       torch::stack({m10 - m01, m20 + m02, m21 + m12,
                     torch::pow(q_abs.index({"...", 3}), 2)},
                    -1)

      },
      -2);

  torch::Tensor flr =
      torch::tensor(0.1f).toType(torch::kFloat).to(q_abs.device());
  torch::Tensor quat_candidates =
      quat_by_rijk / (2.0f * q_abs.index({"...", torch::None}).max(flr));

  torch::Tensor result =
      quat_candidates
          .index({torch::nn::functional::one_hot(q_abs.argmax(-1), 4) > 0.5,
                  torch::All})
          .reshape({batch_size, 4});

  //   torch::Tensor R = batch_mat.index({0}).clone().contiguous();
  //   mat3 eigen_R = reclib::dnn::torch2eigen<float>(R);
  //   std::cout << "R: " << R << std::endl;
  //   std::cout << "eigen_R: " << eigen_R << std::endl;
  //   Eigen::Quaternionf quat(eigen_R);

  //   std::cout << "Quat result: " << result << std::endl;
  //   std::cout << "Eigen quat: " << quat.coeffs() << std::endl;

  return result;
}

torch::Tensor reclib::tracking::compute_gmm_likelihood(
    const torch::Tensor weights, const torch::Tensor means,
    const torch::Tensor inv_covs, const torch::Tensor cov_det,
    torch::Tensor pose) {
  torch::Tensor repeated_pose =
      pose.repeat(weights.sizes()[0])
          .reshape({weights.sizes()[0], pose.sizes()[0]});
  torch::Tensor normalized = (repeated_pose - means).unsqueeze(2);  // [B,23,1]
  std::cout << "normalized: " << normalized << std::endl;
  std::cout << "inv cov variance: : " << inv_covs.index({0}).diag()
            << std::endl;
  torch::Tensor sum = torch::bmm(normalized.permute({0, 2, 1}),
                                 torch::bmm(inv_covs, normalized))
                          .reshape(-1);  // [B]

  sum = torch::exp(-0.5 * sum);
  torch::Tensor scalar =
      (1.0 / (pow(2 * 3.1415926535897931, pose.sizes()[0] / 2.0) *
              torch::sqrt(cov_det)));  // [B]

  torch::Tensor weighted_likelihood = (weights * scalar * sum).sum();

  if (torch::isinf(weighted_likelihood).any().item<bool>()) {
    std::cout << "sum: " << sum << std::endl;
    std::cout << "scalar: " << scalar << std::endl;
    exit(0);
  }
  return weighted_likelihood;
}

#endif  // HAS_DNN_MODULE