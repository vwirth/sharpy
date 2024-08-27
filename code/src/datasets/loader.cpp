#include "reclib/datasets/loader.h"

#include <fstream>

#include "reclib/assert.h"
#include "reclib/models/smpl.h"
#include "reclib/python/python.h"

#if HAS_OPENCV_MODULE

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ----------------------------------------------------
// HandDatasetLoader
// ----------------------------------------------------

void reclib::datasets::HandDatasetLoader::load_from_file(fs::path filepath) {
  std::fstream str;
  str.open(filepath);

  std::string line;
  while (std::getline(str, line)) {
    if (line.length() == 0) continue;
    file_data_.push_back(line);
  }
}

void reclib::datasets::HandDatasetLoader::load_from_path(fs::path path) {
  file_data_.push_back(path);
}

void reclib::datasets::HandDatasetLoader::initialize_from_image_data() {
  int iter = 0;
  for (unsigned int i = 0; i < file_data_.size(); i++) {
    std::string line = file_data_[i];
    if (line.length() == 0) continue;
    MetaData data = compute_metadata_per_image(line);
    if (valid_images_.size() == 0 ||
        std::find(valid_images_.begin(), valid_images_.end(), data.file_) !=
            valid_images_.end()) {
      // skip every nth frame
      if (iter % (skip_frames_ + 1) == 0) files_.push_back(data);
      iter++;
    }
  }
}

void reclib::datasets::HandDatasetLoader::initialize_from_sequence_data() {
  for (unsigned int i = 0; i < file_data_.size(); i++) {
    std::string line = file_data_[i];
    fs::path seq_path = fs::path(line);
    fs::path base_path = root_path_ / fs::path(seq_path) / fs::path("rgb");
    fs::directory_iterator iter(base_path);

    std::set<fs::path> files;
    for (; iter != fs::directory_iterator(); iter++) {
      files.insert(base_path / iter->path());
    }
    int iteration = 0;
    for (auto f : files) {
      if (f.string().length() == 0) continue;
      if (f.extension().string().length() == 0) continue;
      if (f.extension().compare(".png") != 0 &&
          f.extension().compare(".jpg") != 0)
        continue;

      MetaData data = MetaData{f, seq_path, f.stem()};
      if (valid_images_.size() == 0 ||
          std::find(valid_images_.begin(), valid_images_.end(), data.file_) !=
              valid_images_.end()) {
        // skip every nth frame within a sequence
        if (iteration % (skip_frames_ + 1) == 0) files_.push_back(data);
        iteration++;
      }
    }
  }
}

void reclib::datasets::HandDatasetLoader::initialize_valid_images(
    fs::path filepath) {
  std::fstream str;
  str.open(filepath);
  std::string line;
  while (std::getline(str, line)) {
    if (line.length() == 0) continue;
    fs::path f = root_path_ / fs::path(line);
    valid_images_.push_back(f);
  }
  str.close();
}

reclib::datasets::MetaData
reclib::datasets::HandDatasetLoader::compute_metadata_per_image(
    fs::path line_in_image_file) {
  fs::path f;
  fs::path seq;
  fs::path fid;

  f = root_path_ / line_in_image_file;
  seq = line_in_image_file.parent_path().parent_path();
  fid = line_in_image_file.stem();
  return MetaData{f, seq, fid};
}

void reclib::datasets::HandDatasetLoader::get_input_(CpuMat& out_rgb,
                                                     CpuMat& out_depth) {
  fs::path img_path = files_[index_].file_;
  if (!fs::exists(img_path)) {
    std::cout << "Could not find img_path: " << img_path << std::endl;
  }
  _RECLIB_ASSERT(fs::exists(img_path));

  out_rgb = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
  if (out_rgb.elemSize() > 3) {
    cv::cvtColor(out_rgb, out_rgb, cv::COLOR_BGRA2BGR);
  }
  fs::path filename = img_path.stem();
  fs::path depth_path = root_path_ / files_[index_].seq_path_ /
                        fs::path("depth") /
                        fs::path(filename.string() + ".png");
  _RECLIB_ASSERT(fs::exists(depth_path));
  out_depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
}

bool reclib::datasets::HandDatasetLoader::get_input(CpuMat& out_rgb,
                                                    CpuMat& out_depth,
                                                    bool& new_sequence) {
  if (index_ >= files_.size()) return false;

  // check whether we start a new video sequence
  if (index_ > 0) {
    fs::path last_seq = files_[index_ - 1].seq_path_;
    fs::path seq = files_[index_].seq_path_;

    if (last_seq.string().compare(seq.string()) > 0) {
      // new sequence detected
      new_sequence = true;
    } else {
      int last_id = std::stoi(files_[index_ - 1].id_.string());
      int id = std::stoi(files_[index_].id_.string());

      int next_index = last_id + 1;
      if (config_["skip_frames"].as<int>() > 0) {
        next_index = last_id + 1 + config_["skip_frames"].as<int>();
      }

      if (id != next_index) {
        // new sub-sequence within current sequence detected
        new_sequence = true;
      } else {
        new_sequence = false;
      }
    }
  } else if (max_frame_per_sequence_ > 0) {
    int max = max_frame_per_sequence_;
    int seq = std::stoi(files_[index_].id_.string());
    fs::path main = files_[index_].seq_path_;

    if (seq > max) {
      while (true) {
        index_++;
        if (index_ >= files_.size()) return false;
        fs::path main_ = files_[index_].seq_path_;

        if (main_.string().compare(main.string()) > 0) {
          break;
        }
      }
      new_sequence = true;
    }
  } else {
    new_sequence = false;
  }
  new_sequence_last_request_ = new_sequence;

  // check single image case
  if (config_["debug_single_image"] &&
      config_["debug_single_image"].as<bool>()) {
    fs::path img_path = config_["image_path"].as<std::string>();
    // try to find image path within loaded files
    bool found = false;
    for (unsigned int i = 0; i < files_.size(); i++) {
      if (img_path.compare(files_[i].file_) == 0) {
        index_ = i;
        found = true;
        break;
      }
    }

    if (!found) {
      _RECLIB_ASSERT(fs::exists(img_path));
      // manually load image
      out_rgb = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
      if (out_rgb.elemSize() > 3) {
        cv::cvtColor(out_rgb, out_rgb, cv::COLOR_BGRA2BGR);
      }
      out_depth.create(out_rgb.size(), CV_32FC1);
      out_depth.setTo(0);

      return true;
    }
  }

  // invoke class-specific input loading function
  get_input_(out_rgb, out_depth);

  index_++;
  return out_rgb.cols * out_rgb.rows > 0 && out_depth.cols * out_depth.rows > 0;
}

bool reclib::datasets::HandDatasetLoader::get_gt(
    std::vector<HandData>& params) {
  return false;
}

reclib::IntrinsicParameters
reclib::datasets::HandDatasetLoader::get_intrinsics() {
  return reclib::IntrinsicParameters();
}

// ----------------------------------------------------
// H2OLoader
// ----------------------------------------------------

bool reclib::datasets::H2OLoader::get_gt(
    std::vector<reclib::datasets::HandData>& params) {
  fs::path img_path = files_[index_].file_;
  fs::path filename = img_path.stem();
  _RECLIB_ASSERT(fs::exists(img_path));

  fs::path mano_path = img_path.parent_path().parent_path() /
                       fs::path("hand_pose_mano") /
                       fs::path(filename.string() + ".txt");
  _RECLIB_ASSERT(fs::exists(mano_path));
  std::fstream str;
  std::vector<std::string> tokenized_data;
  str.open(mano_path.string(), std::ios::in);
  if (str.is_open()) {
    std::string data;
    while (std::getline(str, data, ' ')) {
      tokenized_data.push_back(data);
    }
    str.close();
  }

  // left
  {
    HandData data;
    for (unsigned int i = 1; i < data.params_.rows() + 1; i++) {
      float val = std::stof(tokenized_data[i]);
      data.params_(i - 1) = val;
    }
    params.push_back(data);
  }
  // right
  {
    HandData data;
    data.is_left_ = false;
    for (unsigned int i = data.params_.rows() + 2;
         i < data.params_.rows() * 2 + 2; i++) {
      data.params_(i - data.params_.rows() - 2) = std::stof(tokenized_data[i]);
    }
    params.push_back(data);
  }
  return true;
}

reclib::IntrinsicParameters reclib::datasets::H2OLoader::get_intrinsics() {
  fs::path img_path = files_[index_].file_;
  fs::path filename = img_path.stem();
  _RECLIB_ASSERT(fs::exists(img_path));
  reclib::IntrinsicParameters intrinsics;
  fs::path intrinsics_path =
      img_path.parent_path().parent_path() / "cam_intrinsics.txt";
  _RECLIB_ASSERT(fs::exists(intrinsics_path));

  std::fstream str;
  str.open(intrinsics_path, std::ios::in);
  if (str.is_open()) {
    std::string data;
    std::vector<std::string> tokenized_data;
    while (std::getline(str, data, ' ')) {
      tokenized_data.push_back(data);
    }
    str.close();

    intrinsics.focal_x_ = std::stof(tokenized_data[0]);
    intrinsics.focal_y_ = std::stof(tokenized_data[1]);
    intrinsics.principal_x_ = std::stof(tokenized_data[2]);
    intrinsics.principal_y_ = std::stof(tokenized_data[3]);
    intrinsics.image_width_ = std::stoi(tokenized_data[4]);
    intrinsics.image_height_ = std::stoi(tokenized_data[5]);
  }
  return intrinsics;
}

// ----------------------------------------------------
// H2O3DLoader
// ----------------------------------------------------

void reclib::datasets::H2O3DLoader::initialize_valid_images(fs::path filepath) {
  std::fstream str;
  str.open(filepath);
  std::string line;
  while (std::getline(str, line)) {
    if (line.length() == 0) continue;
    std::string main = line.substr(0, line.find("/"));
    std::string img = line.substr(line.find("/") + 1);

    std::string extension = ".jpg";
    if (use_ho3d_) {
      extension = ".png";
    }

    fs::path f = root_path_ / fs::path(main) / fs::path("rgb") /
                 fs::path(img + extension);
    valid_images_.push_back(f);
  }
  str.close();
}

reclib::datasets::MetaData
reclib::datasets::H2O3DLoader::compute_metadata_per_image(
    fs::path line_in_image_file) {
  fs::path f;
  fs::path seq;
  fs::path fid;

  std::string extension = ".jpg";
  if (use_ho3d_) {
    extension = ".png";
  }

  std::string line = line_in_image_file.string();
  std::string main = line.substr(0, line.find("/"));
  std::string img = line.substr(line.find("/") + 1);
  f = root_path_ / fs::path(main) / fs::path("rgb") / fs::path(img + extension);
  seq = main;
  fid = img;

  return MetaData{f, seq, fid};
}

void reclib::datasets::H2O3DLoader::get_input_(CpuMat& out_rgb,
                                               CpuMat& out_depth) {
  fs::path img_path = files_[index_].file_;
  if (!fs::exists(img_path)) {
    std::cout << "Could not find img_path: " << img_path << std::endl;
  }
  _RECLIB_ASSERT(fs::exists(img_path));

  out_rgb = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
  if (out_rgb.elemSize() > 3) {
    cv::cvtColor(out_rgb, out_rgb, cv::COLOR_BGRA2BGR);
  }
  fs::path filename = img_path.stem();
  fs::path depth_path = files_[index_].seq_path_ / fs::path("depth") /
                        fs::path(filename.string() + ".png");
  _RECLIB_ASSERT(fs::exists(depth_path));
  CpuMat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

  double depth_scale = 0.00012498664727900177;
  depth.convertTo(depth, CV_32FC1);
  cv::Mat depth_r, depth_g;
  cv::extractChannel(depth, depth_r, 2);
  cv::extractChannel(depth, depth_g, 1);
  out_depth = (depth_r + depth_g * 256) * depth_scale * 1000.f;
}

bool reclib::datasets::H2O3DLoader::get_gt(
    std::vector<reclib::datasets::HandData>& params) {
  fs::path img_path = files_[index_].file_;
  fs::path filename = img_path.stem();
  _RECLIB_ASSERT(fs::exists(img_path));

  fs::path mano_path = img_path.parent_path().parent_path() / fs::path("meta") /
                       fs::path("npz") / fs::path(filename.string() + ".npz");

  _RECLIB_ASSERT(fs::exists(mano_path));
  cnpy::npz_t npz = cnpy::npz_load(mano_path);
  if (npz.find("handBeta") == npz.end()) return false;

  // params are delivered in OpenGL space (negative z, positive y)
  // however, we expect them to be in Pinhole space
  // (positive z, negative y)
  mat3 mirror_mat = mat3::Identity();
  mirror_mat(1, 1) *= -1;
  mirror_mat(2, 2) *= -1;
  mat3 rot_mat;
  rot_mat << 1, 0, 0, 0, -1, 0, 0, 0, -1;

  // left
  if (!use_ho3d_) {
    HandData data;

    const auto& leftHandPose_raw = npz.at("leftHandPose");
    data.params_.segment<48>(3) =
        reclib::python::load_float_matrix_cm(leftHandPose_raw, 48, 1);

    const auto& leftHandTrans_raw = npz.at("leftHandTrans");
    data.params_.head<3>() =
        reclib::python::load_float_matrix_cm(leftHandTrans_raw, 3, 1);

    const auto& handBeta_raw = npz.at("handBeta");

    data.params_.segment<10>(3 + 48) =
        reclib::python::load_float_matrix_cm(handBeta_raw, 10, 1);

    for (unsigned int i = 0; i < 1; i++) {
      data.params_.segment<3>(i * 3) =
          mirror_mat * data.params_.segment<3>(i * 3);
    }

    for (unsigned int i = 1; i < 2; i++) {
      mat3 rot =
          reclib::models::rodrigues<float>(data.params_.segment<3>(i * 3));
      mat3 modified_rot = rot_mat * rot;
      Eigen::AngleAxisf aa(modified_rot);
      data.params_.segment<3>(i * 3) = aa.angle() * aa.axis();
    }

    params.push_back(data);
  }
  // right
  {
    std::string poseKey = "rightHandPose";
    std::string transKey = "rightHandTrans";
    if (use_ho3d_) {
      poseKey = "handPose";
      transKey = "handTrans";
    }

    HandData data;
    data.is_left_ = false;

    const auto& rightHandPose_raw = npz.at(poseKey);
    data.params_.segment<48>(3) =
        reclib::python::load_float_matrix_cm(rightHandPose_raw, 48, 1);

    const auto& rightHandTrans_raw = npz.at(transKey);
    data.params_.head<3>() =
        reclib::python::load_float_matrix_cm(rightHandTrans_raw, 3, 1);

    const auto& handBeta_raw = npz.at("handBeta");

    data.params_.segment<10>(3 + 48) =
        reclib::python::load_float_matrix_cm(handBeta_raw, 10, 1);

    for (unsigned int i = 0; i < 1; i++) {
      data.params_.segment<3>(i * 3) =
          mirror_mat * data.params_.segment<3>(i * 3);
    }

    for (unsigned int i = 1; i < 2; i++) {
      mat3 rot =
          reclib::models::rodrigues<float>(data.params_.segment<3>(i * 3));
      mat3 modified_rot = rot_mat * rot;
      Eigen::AngleAxisf aa(modified_rot);

      data.params_.segment<3>(i * 3) = aa.angle() * aa.axis();
    }

    params.push_back(data);
  }
  return true;
}

reclib::IntrinsicParameters reclib::datasets::H2O3DLoader::get_intrinsics() {
  fs::path img_path = files_[index_].file_;
  fs::path filename = img_path.stem();
  _RECLIB_ASSERT(fs::exists(img_path));
  reclib::IntrinsicParameters intrinsics;
  fs::path intrinsics_path = img_path.parent_path().parent_path() /
                             fs::path("meta") / fs::path("npz") /
                             fs::path(filename.string() + ".npz");

  std::string fallback_sequence = "MBC1";
  if (use_ho3d_) {
    fallback_sequence = "ABF10";
  }

  if (!fs::exists(intrinsics_path)) {
    std::cout << "loading camera intrinsics from training data.." << std::endl;
    intrinsics_path = root_path_ / fs::path("train") /
                      fs::path(fallback_sequence) / fs::path("meta") /
                      fs::path("npz") / fs::path(filename.string() + ".npz");
  }
  _RECLIB_ASSERT(fs::exists(intrinsics_path));
  cnpy::npz_t npz = cnpy::npz_load(intrinsics_path);
  const auto& camMat_raw = npz.at("camMat");
  if (camMat_raw.word_size == 0 && index_ > 0 && !new_sequence_last_request_) {
    return last_intrinsics_;
  }

  _RECLIB_ASSERT_GT(camMat_raw.word_size, 0);
  mat3 I = reclib::python::load_float_matrix_cm(camMat_raw, 3, 3);
  intrinsics.focal_x_ = I(0, 0);
  intrinsics.focal_y_ = I(1, 1);
  intrinsics.principal_x_ = I(0, 2);
  intrinsics.principal_y_ = I(1, 2);
  intrinsics.image_width_ = 640;
  intrinsics.image_height_ = 480;

  last_intrinsics_ = intrinsics;
  return intrinsics;
}

// ----------------------------------------------------
// AzureLoader
// ----------------------------------------------------

#if HAS_K4A
void reclib::datasets::AzureLoader::get_input_(CpuMat& out_rgb,
                                               CpuMat& out_depth) {
  fs::path img_path = files_[index_].file_;
  if (!fs::exists(img_path)) {
    std::cout << "Could not find img_path: " << img_path << std::endl;
  }
  _RECLIB_ASSERT(fs::exists(img_path));

  out_rgb = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
  if (out_rgb.elemSize() > 3) {
    cv::cvtColor(out_rgb, out_rgb, cv::COLOR_BGRA2BGR);
  }
  fs::path filename = img_path.stem();
  fs::path depth_path = files_[index_].seq_path_ / fs::path("depth") /
                        fs::path(filename.string() + ".png");
  _RECLIB_ASSERT(fs::exists(depth_path));
  CpuMat depth = cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
  depth.convertTo(depth, CV_32FC1);
  out_depth = reclib::KinectAzureCamera::depth2color(depth, out_rgb, calib_);
}

reclib::IntrinsicParameters reclib::datasets::AzureLoader::get_intrinsics() {
  return real_cam_->GetIntrinsics();
}
#endif

#endif  // HAS_OPENCV_MODULE