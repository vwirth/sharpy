#ifndef RECLIB_DATASETS_LOADER_H
#define RECLIB_DATASETS_LOADER_H
#include <filesystem>
#include <optional>

#include "reclib/camera_parameters.h"
#include "reclib/configuration.h"
#include "reclib/data_types.h"
#include "reclib/depth_camera.h"

#if HAS_OPENCV_MODULE

namespace reclib {
namespace datasets {

struct HandData {
  Eigen::VectorXf params_;
  bool is_left_;

  HandData() : params_(3 + 10 + 3 + 45), is_left_(true){};
};

struct MetaData {
  fs::path file_;      // full path to file
  fs::path seq_path_;  // base path of the sequence the file belongs to
  fs::path id_;        // id of the file (without extension)
};

class HandDatasetLoader {
 protected:
  reclib::Configuration config_;

  std::string root_path_;
  std::vector<std::string> file_data_;

  // contains full filepath
  std::vector<MetaData> files_;

  // contains a list of valid (full) image paths
  // the term valid either means there is no restriction to the data (size = 0)
  // or a file is given, which contains image indices of valid images
  std::vector<fs::path> valid_images_;

  unsigned int index_;
  unsigned int skip_frames_;
  int max_frame_per_sequence_;
  bool new_sequence_last_request_;
  reclib::IntrinsicParameters last_intrinsics_;

  void initialize_from_image_data();
  void initialize_from_sequence_data();
  void load_from_file(fs::path filepath);
  void load_from_path(fs::path path);

  virtual void initialize_valid_images(fs::path filepath);
  virtual MetaData compute_metadata_per_image(fs::path line_in_image_file);
  virtual void get_input_(CpuMat& out_rgb, CpuMat& out_depth);

 public:
  HandDatasetLoader(reclib::Configuration config)
      : config_(config),
        root_path_(config["path"].as<std::string>()),
        index_(0),
        skip_frames_(0),
        max_frame_per_sequence_(-1),
        new_sequence_last_request_(false) {
    if (config_["start_frame"]) {
      index_ = std::max(0, config_["start_frame"].as<int>());
    }
    if (config_["skip_frames"]) {
      skip_frames_ = std::max(0, config_["skip_frames"].as<int>());
    }

    if (config_["valid_file"]) {
      initialize_valid_images(config_["valid_file"].as<std::string>());
    }
    if (config_["image_file"]) {
      load_from_file(config_["image_file"].as<std::string>());
      initialize_from_image_data();
    } else if (config_["sequence_file"]) {
      load_from_file(config_["sequence_file"].as<std::string>());
      initialize_from_sequence_data();
    } else {
      load_from_path(root_path_);
      initialize_from_sequence_data();
    }

    if (config_["max_frame_per_sequence"]) {
      max_frame_per_sequence_ = std::min(
          (int)files_.size(), config_["max_frame_per_sequence"].as<int>());
    }

    const fs::path root_path{root_path_};
    if (!fs::exists(root_path)) {
      std::cerr << "Make sure `path` and other variables in sharpy.yaml are set" << std::endl;
      std::exit(1);
    }
  };

  virtual ~HandDatasetLoader(){};
  virtual bool get_gt(std::vector<HandData>& params);
  virtual reclib::IntrinsicParameters get_intrinsics();

  bool get_input(CpuMat& out_rgb, CpuMat& out_depth, bool& new_sequence);
  int index() const { return index_; }
  void set_index(int ind) { index_ = ind; }
  std::optional<MetaData> get_metadata() {
    if (index_ >= files_.size()) return {};
    return {files_[index_]};
  }
};

class H2OLoader : public HandDatasetLoader {
 public:
  H2OLoader(reclib::Configuration config) : HandDatasetLoader(config){};
  ~H2OLoader(){};

  bool get_gt(std::vector<HandData>& params) override;
  reclib::IntrinsicParameters get_intrinsics() override;
};

class H2O3DLoader : public HandDatasetLoader {
  void initialize_valid_images(fs::path filepath) override;
  MetaData compute_metadata_per_image(fs::path line_in_image_file) override;
  void get_input_(CpuMat& out_rgb, CpuMat& out_depth) override;

  bool use_ho3d_;  // ho3d is the previous version that only contains a single
                   // right hand

 public:
  H2O3DLoader(reclib::Configuration config, bool use_ho3d = false)
      : HandDatasetLoader(config), use_ho3d_(use_ho3d){};
  ~H2O3DLoader(){};

  bool get_gt(std::vector<HandData>& params) override;
  reclib::IntrinsicParameters get_intrinsics() override;
};

#if HAS_K4A
class AzureLoader : public HandDatasetLoader {
  void get_input_(CpuMat& out_rgb, CpuMat& out_depth) override;

  std::shared_ptr<reclib::DefaultCamera> real_cam_;
  k4a::calibration calib_;

 public:
  AzureLoader(reclib::Configuration config) : HandDatasetLoader(config) {
    real_cam_ = std::make_shared<reclib::DefaultCamera>(root_path_, true, 1.f);
    calib_ = reclib::KinectAzureCamera::CalibrationFromFile(root_path_);
  };
  ~AzureLoader(){};

  reclib::IntrinsicParameters get_intrinsics() override;
};
#endif  // HAS_K4A

}  // namespace datasets
}  // namespace reclib

#endif  // HAS_OPENCV_MODULE

#endif