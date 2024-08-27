#if HAS_DNN_MODULE
#if HAS_NVDIFFRAST_MODULE

#include <nvdiffrast/torch/torch_funcs.h>
#include <reclib/assert.h>
#include <reclib/dnn/nvdiffrast_autograd.h>

std::vector<torch::Tensor> reclib::dnn::RasterizeFunc::forward(
    torch::autograd::AutogradContext* ctx, RasterizeCRStateWrapper& context,
    torch::Tensor positions, torch::Tensor triangles,
    std::tuple<int, int> resolution, torch::Tensor ranges, bool grad_db,
    int peeling_idx) {
  std::tuple<torch::Tensor, torch::Tensor> res = nvdiffrast::rasterize_fwd_cuda(
      context, positions, triangles, resolution, ranges, peeling_idx);
  // std::get<0>(res).reset();
  // std::get<1>(res).reset();

  // std::cout << "rasterize res0: " << std::get<0>(res).sizes() << std::endl;
  // std::cout << "rasterize res1: " << std::get<1>(res).sizes() << std::endl;

  ctx->save_for_backward({positions, triangles, std::get<0>(res)});
  ctx->saved_data["grad_db"] = grad_db;

  std::vector<torch::Tensor> ret;
  ret.push_back(std::get<0>(res));
  ret.push_back(std::get<1>(res));
  return ret;
}

std::vector<torch::Tensor> reclib::dnn::rasterize(
    RasterizeCRStateWrapper& context, torch::Tensor positions,
    torch::Tensor triangles, std::tuple<int, int> resolution,
    torch::Tensor ranges, bool grad_db, int peeling_idx) {
  return reclib::dnn::RasterizeFunc::apply(
      context, positions, triangles, resolution, ranges, grad_db, peeling_idx);
}

std::vector<torch::Tensor> reclib::dnn::RasterizeFunc::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> backward_inputs) {
  torch::NoGradGuard g;

  torch::Tensor dy = backward_inputs[0];
  torch::Tensor ddb = backward_inputs[1];
  std::vector<torch::Tensor> saved = ctx->get_saved_variables();
  torch::Tensor positions = saved[0];
  torch::Tensor triangles = saved[1];
  torch::Tensor out = saved[2];
  torch::Tensor g_pos;
  if (ctx->saved_data["grad_db"].toBool()) {
    g_pos = nvdiffrast::rasterize_grad_db(positions, triangles, out, dy, ddb);
  } else {
    g_pos = nvdiffrast::rasterize_grad(positions, triangles, out, dy);
  }
  // cudaFree(out.storage().data());
  // out.reset();

  std::vector<torch::Tensor> deriv;
  torch::Tensor uninitialized = torch::empty({0});
  uninitialized.reset();

  deriv.push_back(uninitialized);
  deriv.push_back(g_pos);
  deriv.push_back(uninitialized);
  deriv.push_back(uninitialized);
  deriv.push_back(uninitialized);
  deriv.push_back(uninitialized);
  deriv.push_back(uninitialized);

  ctx->save_for_backward({});
  ctx->saved_data.clear();

  return deriv;
}

std::vector<torch::Tensor> reclib::dnn::InterpolateFunc::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor attr,
    torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db,
    bool diff_attrs_all, std::vector<int> diff_attrs) {
  std::tuple<torch::Tensor, torch::Tensor> res;

  if (diff_attrs.size() > 0 || diff_attrs_all) {
    _RECLIB_ASSERT_GT(rast_db.sizes().size(), 0);
    _RECLIB_ASSERT(rast_db.defined());

    res = nvdiffrast::interpolate_fwd_da(attr, rast, tri, rast_db,
                                         diff_attrs_all, diff_attrs);
    ctx->save_for_backward({attr, rast, tri, rast_db});
    ctx->saved_data["diff_attrs_all"] = diff_attrs_all;
    ctx->saved_data["diff_attrs"] = diff_attrs;

  } else {
    res = nvdiffrast::interpolate_fwd(attr, rast, tri);
    ctx->save_for_backward({attr, rast, tri});
  }

  std::vector<torch::Tensor> ret;
  ret.push_back(std::get<0>(res));
  ret.push_back(std::get<1>(res));

  return ret;
}

std::vector<torch::Tensor> reclib::dnn::interpolate(
    torch::Tensor attr, torch::Tensor rast, torch::Tensor tri,
    torch::Tensor rast_db, bool diff_attrs_all, std::vector<int> diff_attrs) {
  return reclib::dnn::InterpolateFunc::apply(attr, rast, tri, rast_db,
                                             diff_attrs_all, diff_attrs);
}

std::vector<torch::Tensor> reclib::dnn::InterpolateFunc::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> backward_inputs) {
  torch::NoGradGuard g;

  torch::Tensor dy = backward_inputs[0];
  torch::Tensor dda = backward_inputs[1];

  std::vector<torch::Tensor> saved = ctx->get_saved_variables();
  torch::Tensor attr = saved[0];
  torch::Tensor rast = saved[1];
  torch::Tensor tri = saved[2];

  std::vector<torch::Tensor> ret;

  torch::Tensor uninitialized = torch::empty({0});
  uninitialized.reset();

  if (ctx->saved_data.find("diff_attrs") != ctx->saved_data.end()) {
    std::vector<int> diff_attrs =
        ctx->saved_data["diff_attrs"].to<std::vector<int>>();
    bool diff_attrs_all = ctx->saved_data["diff_attrs_all"].toBool();

    torch::Tensor rast_db = saved[3];

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> res =
        nvdiffrast::interpolate_grad_da(attr, rast, tri, dy, rast_db, dda,
                                        diff_attrs_all, diff_attrs);

    ret.push_back(std::get<0>(res));
    ret.push_back(std::get<1>(res));
    ret.push_back(uninitialized);
    ret.push_back(std::get<2>(res));
    ret.push_back(uninitialized);
    ret.push_back(uninitialized);
  } else {
    std::tuple<torch::Tensor, torch::Tensor> res =
        nvdiffrast::interpolate_grad(attr, rast, tri, dy);
    ret.push_back(std::get<0>(res));
    ret.push_back(std::get<1>(res));
    ret.push_back(uninitialized);
    ret.push_back(uninitialized);
    ret.push_back(uninitialized);
    ret.push_back(uninitialized);
  }

  ctx->save_for_backward({});
  ctx->saved_data.clear();

  return ret;
}

std::vector<torch::Tensor> reclib::dnn::AntialiasFunc::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor color,
    torch::Tensor rast, torch::Tensor pos, torch::Tensor tri,
    TopologyHashWrapper topology_hash_wrap, float pos_gradient_boost) {
  if (topology_hash_wrap.ev_hash.sizes()[0] == 0) {
    topology_hash_wrap = nvdiffrast::antialias_construct_topology_hash(tri);
    // topology_hash_wrap.ev_hash.set_requires_grad(color.requires_grad());
  }

  std::tuple<torch::Tensor, torch::Tensor> res =
      nvdiffrast::antialias_fwd(color, rast, pos, tri, topology_hash_wrap);
  // std::get<0>(res).reset();
  // std::get<1>(res).reset();

  // std::cout << "antialias res0: " << std::get<0>(res).sizes() << std::endl;
  // std::cout << "antialias res1: " << std::get<1>(res).sizes() << std::endl;

  ctx->save_for_backward({color, rast, pos, tri});
  ctx->saved_data["pos_gradient_boost"] = pos_gradient_boost;
  ctx->saved_data["work_buffer"] = std::get<1>(res);

  std::vector<torch::Tensor> ret;
  ret.push_back(std::get<0>(res));
  ret.push_back(std::get<1>(res));

  return ret;
}

std::vector<torch::Tensor> reclib::dnn::antialias(
    torch::Tensor color, torch::Tensor rast, torch::Tensor pos,
    torch::Tensor tri, TopologyHashWrapper topology_hash_wrap,
    float pos_gradient_boost) {
  return reclib::dnn::AntialiasFunc::apply(
      color, rast, pos, tri, topology_hash_wrap, pos_gradient_boost);
}

std::vector<torch::Tensor> reclib::dnn::AntialiasFunc::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> backward_inputs) {
  torch::NoGradGuard g;

  torch::Tensor dy = backward_inputs[0];
  float pos_gradient_boost = ctx->saved_data["pos_gradient_boost"].to<float>();
  torch::Tensor work_buffer =
      ctx->saved_data["work_buffer"].to<torch::Tensor>();
  std::vector<torch::Tensor> saved = ctx->get_saved_variables();

  torch::Tensor color = saved[0];
  torch::Tensor rast = saved[1];
  torch::Tensor pos = saved[2];
  torch::Tensor tri = saved[3];

  std::tuple<torch::Tensor, torch::Tensor> res =
      nvdiffrast::antialias_grad(color, rast, pos, tri, dy, work_buffer);

  work_buffer = work_buffer.cpu();
  // cudaFree(work_buffer.storage().data());
  // work_buffer.reset();

  torch::Tensor uninitialized = torch::empty({0});
  uninitialized.reset();
  std::vector<torch::Tensor> ret;
  ret.push_back(std::get<0>(res));
  ret.push_back(uninitialized);
  if (pos_gradient_boost != 1.f) {
    ret.push_back(std::get<1>(res) * pos_gradient_boost);
  } else {
    ret.push_back(std::get<1>(res));
  }
  ret.push_back(uninitialized);
  ret.push_back(uninitialized);
  ret.push_back(uninitialized);

  ctx->save_for_backward({});
  ctx->saved_data.clear();

  return ret;
}

#endif  // HAS_NVDIFFRAST_MODULE
#endif  // HAS_DNN_MODULE
