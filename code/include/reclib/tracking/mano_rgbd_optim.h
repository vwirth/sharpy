#ifndef RECLIB_TRACKING_MANO_RGBD_OPTIM_H
#define RECLIB_TRACKING_MANO_RGBD_OPTIM_H

#if HAS_DNN_MODULE

#include <ATen/TensorIndexing.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/torch.h>

#include "reclib/models/smpl_torch.h"

namespace reclib {
namespace tracking {

extern std::vector<long> apose_indices_;
extern std::vector<long> apose_zero_indices_;

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor> torch_lbs(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor pose,
    bool add_pose_mean = false);

template <typename MODEL>
torch::Tensor compute_apose_matrix(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor shape);

template <typename MODEL>
torch::Tensor apose2pose(const reclib::modelstorch::Model<MODEL>& model,
                         torch::Tensor shape, torch::Tensor rot_and_pose);

torch::Tensor batch_mat2aa(torch::Tensor batch_mat);
torch::Tensor batch_quat2aa(torch::Tensor quat);
torch::Tensor batch_mat2quat(torch::Tensor batch_mat);

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor> torch_lbs_anatomic(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor pose,
    torch::Tensor cross_matrix = torch::empty({0}), bool add_pose_mean = false);

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor> torch_lbs_pca(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor pca,
    bool add_pose_mean = false);

template <class MODEL>
std::pair<torch::Tensor, torch::Tensor> torch_lbs_pca_anatomic(
    const reclib::modelstorch::Model<MODEL>& model, torch::Tensor trans,
    torch::Tensor rot, torch::Tensor shape, torch::Tensor pca,
    torch::Tensor cross_matrix, bool add_pose_mean);

torch::Tensor compute_gmm_likelihood(const torch::Tensor weights,
                                     const torch::Tensor means,
                                     const torch::Tensor inv_covs,
                                     const torch::Tensor cov_det,
                                     torch::Tensor pose);

}  // namespace tracking
}  // namespace reclib

#endif  // HAS_DNN_MODULE

#endif