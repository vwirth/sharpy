#include <reclib/depth_processing.h>
#include <reclib/dnn/nvdiffrast_autograd.h>
#include <reclib/internal/filesystem_ops.h>
#include <reclib/optim/model_registration.h>
#include <reclib/tracking/mano_rgbd_optim.h>
#include <reclib/tracking/sharpy_tracker.h>

#include <nlohmann/json.hpp>
#include <reclib/cuda/device_info.cuh>

#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

static const int OPENPOSE_KEYPOINT_PAIRS[20][2] = {
    {0, 1},  {1, 2},   {2, 3},   {3, 4},    // thumb
    {0, 5},  {5, 6},   {6, 7},   {7, 8},    // pinkie
    {0, 9},  {9, 10},  {10, 11}, {11, 12},  // middle
    {0, 13}, {13, 14}, {14, 15}, {15, 16},  // ring
    {0, 17}, {17, 18}, {18, 19}, {19, 20}   // small
};

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

std::vector<torch::Tensor> reclib::tracking::HandTracker::reduce(
    std::vector<torch::Tensor>& in) {
  torch::NoGradGuard guard;

  std::vector<int> hands = config_["Input"]["hands"].as<std::vector<int>>();
  bool initialize = false;
  if (hand_states_.size() == 0) {
    // initialize the hand state
    initialize = true;
  }

  // keep a list of network indices that have been assigned
  // to a hand state
  std::vector<int> assigned_indices;

  int num_samples = in[0].sizes()[0];
  for (unsigned int i = 0; i < hands.size(); i++) {
    int type = hands[i];

    // loop over the network outputs
    // the outputs are ranked by the confidence score of the class
    // e.g. index j = 0 corresponds to the prediction where the
    // network is most certain
    for (int j = 0; j < num_samples; j++) {
      int cls = in[0].index({j}).to(torch::kCPU).item().toInt();
      if (std::find(assigned_indices.begin(), assigned_indices.end(), j) !=
          assigned_indices.end()) {
        // index has already been assigned, skip
        continue;
      }

      // continue if config hand label does not match the predicted label
      if ((reclib::tracking::HandType)type != reclib::tracking::HandType::ANY &&
          cls != type)
        continue;

      if (initialize) {
        if ((reclib::tracking::HandType)cls ==
            reclib::tracking::HandType::LEFT) {
          hand_states_.push_back(
              reclib::tracking::HandState(config_, mano_model_left_));
        } else if ((reclib::tracking::HandType)cls ==
                   reclib::tracking::HandType::RIGHT) {
          hand_states_.push_back(
              reclib::tracking::HandState(config_, mano_model_right_));
        } else {
          throw std::runtime_error("Unknown label.");
        }
        assigned_indices.push_back(j);
        break;
      } else {
        HandState& state = hand_states_.at(assigned_indices.size());

        if ((reclib::tracking::HandType)cls ==
                reclib::tracking::HandType::LEFT &&
            state.instance_->model.hand_type ==
                reclib::models::HandType::left) {
          assigned_indices.push_back(j);
          break;
        }

        if ((reclib::tracking::HandType)cls ==
                reclib::tracking::HandType::RIGHT &&
            state.instance_->model.hand_type ==
                reclib::models::HandType::right) {
          assigned_indices.push_back(j);
          break;
        }
      }
    }
  }

  // reduce the network output only to the relevant predictions
  // that were chosen as indices
  std::vector<torch::Tensor> out;
  torch::Tensor ind = torch::tensor(assigned_indices);
  for (unsigned int i = 0; i < in.size(); i++) {
    // some network outputs are predicted for the whole image
    // and not instance wise -> duplicate them for each hand instance
    if (in[i].sizes()[0] == 1) {
      std::vector<torch::Tensor> stacked;
      for (unsigned int j = 0; j < ind.sizes()[0]; j++) {
        stacked.push_back(in[i]);
      }
      torch::Tensor t = torch::cat(stacked, 0);
      out.push_back(t);
    } else {
      out.push_back(in[i].index({ind}));
    }
  }
  return out;
}

// -----------------------------------------------------------------------

cv::Rect reclib::tracking::HandTracker::compute_crop_factor(int width,
                                                            int height) {
  torch::NoGradGuard guard;

  torch::Tensor box = network_output_[2].to(torch::kCPU);
  torch::Tensor union_box = torch::zeros(4);
  union_box.index({0}) = torch::min(box.index({torch::All, 0}));
  union_box.index({1}) = torch::min(box.index({torch::All, 1}));
  union_box.index({2}) = torch::max(box.index({torch::All, 2}));
  union_box.index({3}) = torch::max(box.index({torch::All, 3}));

  cv::Rect crop_factor_new;
  crop_factor_new.x =
      std::fmax(0, union_box.index({0}).item<float>() -
                       config_["Input"]["prev_bb_pad_x"].as<float>());
  crop_factor_new.y =
      std::fmax(0, union_box.index({1}).item<float>() -
                       config_["Input"]["prev_bb_pad_y"].as<float>());
  crop_factor_new.width =
      std::fmin(width - 1, union_box.index({2}).item<float>() +
                               config_["Input"]["prev_bb_pad_x"].as<float>()) -
      crop_factor_new.x;
  crop_factor_new.height =
      std::fmin(height - 1, union_box.index({3}).item<float>() +
                                config_["Input"]["prev_bb_pad_y"].as<float>()) -
      crop_factor_new.y;

  return crop_factor_new;
}

// -----------------------------------------------------------------------

void reclib::tracking::HandTracker::detect_openpose(unsigned int index,
                                                    const CpuMat& rgb,
                                                    cv::Rect crop_factor) {
  torch::NoGradGuard guard;

  int cls_label =
      network_output_[0].index({(int)index}).to(torch::kCPU).item().toInt();

  // compute cropped RGB image
  CpuMat rgb_cropped = rgb;
  if (crop_factor.x >= 0) {
    CpuMat img_float_tmp(crop_factor.height, crop_factor.width, rgb.type());

    img_float_tmp = rgb(crop_factor).clone();
    rgb_cropped = img_float_tmp;
  }

  // transform RGB input into network-appropriate format
  CpuMat input_blob = cv::dnn::blobFromImage(
      rgb_cropped, config_["OpenPose"]["scale"].as<double>(),
      cv::Size(368, 368));
  openpose_.setInput(input_blob);
  // network forward
  CpuMat result = openpose_.forward();

  int H = result.size[2];
  int W = result.size[3];
  // scale factor
  float SX = float(rgb_cropped.cols) / W;
  float SY = float(rgb_cropped.rows) / H;

  float thresh = config_["OpenPose"]["threshold"].as<float>();
  std::vector<cv::Point> points(22);
  for (int n = 0; n < 22; n++) {
    // Slice heatmap of corresponding body's part.
    CpuMat heatMap(H, W, CV_32F, result.ptr(0, n));

    // 1 maximum per heatmap
    cv::Point p(-1, -1), pm;
    double conf;
    cv::minMaxLoc(heatMap, 0, &conf, 0, &pm);
    if (conf > thresh) p = pm;

    if (p.x > -1 && p.y > -1) {
      // scale to image size
      p.x = p.x * SX + crop_factor.x;
      p.y = p.y * SY + crop_factor.y;
    }
    points[n] = p;
  }

  if (config_["Debug"]["show_keypoints"].as<bool>()) {
    for (int n = 0; n < 20; n++) {
      // lookup 2 connected body/hand parts
      cv::Point2f a = points[OPENPOSE_KEYPOINT_PAIRS[n][0]];
      cv::Point2f b = points[OPENPOSE_KEYPOINT_PAIRS[n][1]];
      // we did not find enough confidence before
      if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0) continue;

      int finger_index = n / 4;
      vec3 c =
          reclib::tracking::HandState::color_mano_joints_[finger_index * 3 +
                                                          1] *
          255;

      cv::line(rgb, a, b, cv::Scalar(c.z(), c.y(), c.x()), 5);
      cv::circle(rgb, a, 3, cv::Scalar(255, 255, 255), 5);
      cv::circle(rgb, b, 3, cv::Scalar(255, 255, 255), 5);
    }

    if (config_["Pipeline"]["save_OpenPose"].as<bool>()) {
      fs::path output_dir =
          config_["Pipeline"]["data_output_folder"].as<std::string>();
      fs::path openpose_path = "openpose";
      std::stringstream filename;
      filename << std::setw(6) << std::setfill('0') << global_frame_counter_
               << ".png";
      if (!fs::exists(output_dir / openpose_path)) {
        reclib::utils::CreateDirectories({output_dir / openpose_path});
      }

      cv::imwrite(output_dir / openpose_path / fs::path(filename.str()), rgb);
    } else if (config_["Debug"]["show_keypoints"].as<bool>()) {
      cv::imshow("OpenPose", rgb);
      cv::waitKey(0);
    }
  }
}

void reclib::tracking::HandTracker::detect(cv::Rect crop_factor) {
  torch::NoGradGuard guard;

  // Prepare input data for network forward pass
  CpuMat img_float = rgb_.clone();

  unsigned int orig_image_h = img_float.rows;
  unsigned int orig_image_w = img_float.cols;
  if (crop_factor.height > 0) {
    CpuMat img_float_tmp(crop_factor.height, crop_factor.width,
                         img_float.type());

    img_float_tmp = img_float(crop_factor).clone();
    img_float = img_float_tmp;
  }
  torch::Tensor input_float = reclib::dnn::cv2torch(img_float);
  unsigned int image_h = input_float.sizes()[0];
  unsigned int image_w = input_float.sizes()[1];

  std::vector<torch::Tensor> output = transform_.transform({input_float});
  torch::Tensor net_input = output[0].permute({2, 0, 1});
  unsigned int transformed_h = output[0].sizes()[0];
  unsigned int transformed_w = output[0].sizes()[1];

  if (debug_) {
    std::cout << "---- Transforming input: " << timer_.look_and_reset()
              << " ms." << std::endl;
  }

  if (debug_ && config_["Debug"]["show_input"].as<bool>() == true) {
    torch::Tensor output_img = output[0].abs().contiguous();
    CpuMat output_mat = reclib::dnn::torch2cv(output_img, true);
    cv::imshow("Input (transformed)", output_mat);
    cv::waitKey(0);
    timer_.look_and_reset();
  }

  // Network forward pass
  std::vector<torch::Tensor> tmp =
      network_.forward(net_input.to(torch::kCUDA), false);

  if (debug_) {
    std::cout << "---- Network forward: " << timer_.look_and_reset() << " ms."
              << std::endl;
  }

  // untransform and reduce network output to relevant information
  std::vector<std::vector<torch::Tensor>> out_detection = network_.detect(tmp);
  std::vector<torch::Tensor> out_processed = network_.postprocess(
      out_detection, transformed_w, transformed_h, 0, false, false);
  out_processed = reduce(out_processed);
  out_processed = transform_.untransform(out_processed, image_w, image_h);

  // if network input was cropped, rescale output appropriately
  if (crop_factor.width > 0) {
    // rescale box
    out_processed[2].index({torch::All, 0}) += crop_factor.x;
    out_processed[2].index({torch::All, 1}) += crop_factor.y;
    out_processed[2].index({torch::All, 2}) += crop_factor.x;
    out_processed[2].index({torch::All, 3}) += crop_factor.y;

    if (config_["Network"]["bb_pad_x"].as<int>() > 0) {
      int pad = config_["Network"]["bb_pad_x"].as<int>();
      out_processed[2].index({torch::All, 0}) = torch::clamp(
          out_processed[2].index({torch::All, 0}) - pad, 0, (int)orig_image_w);
      out_processed[2].index({torch::All, 2}) = torch::clamp(
          out_processed[2].index({torch::All, 2}) + pad, 0, (int)orig_image_w);
    }
    if (config_["Network"]["bb_pad_y"].as<int>() > 0) {
      int pad = config_["Network"]["bb_pad_y"].as<int>();
      out_processed[2].index({torch::All, 1}) = torch::clamp(
          out_processed[2].index({torch::All, 1}) - pad, 0, (int)orig_image_h);
      out_processed[2].index({torch::All, 3}) = torch::clamp(
          out_processed[2].index({torch::All, 3}) + pad, 0, (int)orig_image_h);
    }

    // rescale mask
    torch::Tensor expanded_masks =
        torch::zeros({out_processed[3].sizes()[0], rgb_.rows, rgb_.cols},
                     torch::TensorOptions().device(out_processed[3].device()));

    expanded_masks.index_put_(
        {torch::All,
         torch::indexing::Slice(crop_factor.y,
                                crop_factor.y + crop_factor.height),
         torch::indexing::Slice(crop_factor.x,
                                crop_factor.x + crop_factor.width)},
        out_processed[3]);
    out_processed[3] = expanded_masks;

    // rescale corrs
    torch::Tensor expanded_corrs =
        torch::zeros({out_processed[4].sizes()[0], 3, rgb_.rows, rgb_.cols},
                     torch::TensorOptions().device(out_processed[4].device()));
    expanded_corrs.index_put_(
        {torch::All, torch::All,
         torch::indexing::Slice(crop_factor.y,
                                crop_factor.y + crop_factor.height),
         torch::indexing::Slice(crop_factor.x,
                                crop_factor.x + crop_factor.width)},
        out_processed[4]);
    out_processed[4] = expanded_corrs;
  }
  // store network output
  network_output_ = out_processed;

  if (debug_) {
    std::cout << "---- Untransforming network output: "
              << timer_.look_and_reset() << " ms." << std::endl;
  }
}

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE