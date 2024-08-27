#include <math.h>
#include <reclib/depth_camera.h>
#include <reclib/internal/filesystem_ops.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#if HAS_OPENCV_MODULE
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <system_error>
#include <thread>
#include <utility>

#include "reclib/assert.h"

#ifdef HAS_FREENECT2
#include <libfreenect2/logger.h>
#include <libfreenect2/packet_pipeline.h>
#endif

#ifdef HAS_K4A
#include <k4a/k4a.h>
#endif

#if HAS_OPENCV_MODULE
// ------------------------------------------------------------
// ------------------------------------------------------------
// DepthCamera
// ------------------------------------------------------------
// ------------------------------------------------------------

/**
 * Minimum depth of points (in meters). Points under this depth are presumed to
 * be noise. (0.0 to disable)
 */
const float reclib::DepthCamera::kNoiseFilterLow = 0.1;

const fs::path reclib::DepthCamera::depth_dir = "depth/";
const fs::path reclib::DepthCamera::xyz_dir = "xyz/";
const fs::path reclib::DepthCamera::amp_dir = "amp/";
const fs::path reclib::DepthCamera::rgb_dir = "rgb/";
const fs::path reclib::DepthCamera::ir_dir = "ir/";
const fs::path reclib::DepthCamera::flag_dir = "flag/";

reclib::DepthCamera::DepthCamera(float depth_scaling, float image_scaling,
                                 const fs::path &file_base_path)
    : bad_input_flag_(false),
      device_open_flag_(true),
      data_requested_(false),
      has_xyz_map_(true),
      has_rgb_map_(false),
      has_ir_map_(false),
      has_amp_map_(false),
      has_flag_map_(false),
      has_raw_depth_map_(false),
      depth_scaling_(depth_scaling),
      image_scaling_(image_scaling),
      file_base_path_(file_base_path),
      frame_counter_(0),
      capture_interrupt_(true),
      recording_(false) {}

reclib::DepthCamera::~DepthCamera() {
  bad_input_flag_ = true;
  EndCapture();
}

void reclib::DepthCamera::BeginCapture(int fps_cap, bool remove_noise) {
  if (!device_open_flag_) {
    std::cerr << "WARNING: beginCapture called on non-opened camera.\n";
    return;
  }
  assert(capture_interrupt_);

  capture_interrupt_ = false;
  capture_thd_ = std::make_unique<std::thread>([this, fps_cap, remove_noise] {
    using namespace std::chrono;
    steady_clock::time_point last_time;

    steady_clock::time_point curr_time;
    float time_per_frame = NAN;
    if (fps_cap > 0) {
      time_per_frame = 1e9f / fps_cap;
      last_time = steady_clock::now();
    }

    while (!capture_interrupt_) {
      if (NextFrame(remove_noise, curr_time.time_since_epoch().count() == 0)) {
        frame_counter_++;
      }

      if (curr_time.time_since_epoch().count() == 0) {
        curr_time = steady_clock::now();
      }
      // Cap FPS
      if (fps_cap > 0) {
        curr_time = steady_clock::now();
        steady_clock::duration delta =
            duration_cast<microseconds>(curr_time - last_time);

        if (delta.count() < time_per_frame) {
          long long ms =
              static_cast<long long>(time_per_frame - delta.count()) / 1e6f;
          std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
        last_time = curr_time;
      }
    }
  });
}

void reclib::DepthCamera::EndCapture() {
  // std::lock_guard<std::mutex> lock(image_mutex_);

  capture_interrupt_ = true;
  if (capture_thd_) {
    capture_thd_->join();
    capture_thd_.reset();
  }
}

void reclib::DepthCamera::StartRecording(const fs::path &file_base_path) {
  file_base_path_ = file_base_path;
  recording_ = true;
}

void reclib::DepthCamera::StopRecording() { recording_ = false; }

bool reclib::DepthCamera::IsRecording() { return recording_; }

bool MatIsZero(const CpuMat &CpuMat) { return cv::norm(cv::sum(CpuMat)) == 0; }

bool reclib::DepthCamera::HasNextFrame() {
  std::lock_guard<std::mutex> lock(image_mutex_);

  bool result =
      !data_requested_ &&
      ((!HasXYZMap() || (!xyz_map_.empty() && !MatIsZero(xyz_map_))) &&
       (!HasRGBMap() || (!rgb_map_.empty() && !MatIsZero(rgb_map_))) &&
       (!HasIRMap() || (!ir_map_.empty() && !MatIsZero(ir_map_))) &&
       (!HasAmpMap() || (!amp_map_.empty() && !MatIsZero(amp_map_))) &&
       (!HasFlagMap() || (!flag_map_.empty() && !MatIsZero(flag_map_))) &&
       (!HasRawDepthMap() ||
        (!raw_depth_map_.empty() && !MatIsZero(raw_depth_map_))));

  return result;
}

auto reclib::DepthCamera::NextFrame(bool remove_noise, bool first_frame)
    -> bool {
  bool initialize = ((HasXYZMap() && xyz_map_buf_.empty()) ||
                     (HasRGBMap() && rgb_map_buf_.empty()) ||
                     (HasIRMap() && ir_map_buf_.empty()) ||
                     (HasAmpMap() && amp_map_buf_.empty()) ||
                     (HasFlagMap() && flag_map_buf_.empty()) ||
                     (HasRawDepthMap() && raw_depth_map_buf_.empty()));
  // initialize back buffers
  if (frame_counter_ == 0) {
    InitializeImages();
  }

  // call update with back buffer images (to allow continued operation on
  // front end)
  Update(xyz_map_buf_, rgb_map_buf_, ir_map_buf_, amp_map_buf_, flag_map_buf_,
         raw_depth_map_buf_, first_frame);

  if (!BadInput() && !xyz_map_buf_.empty() && (xyz_map_buf_.data != nullptr) &&
      HasXYZMap()) {
    if (remove_noise) {
      reclib::DepthCamera::RemoveNoise(xyz_map_buf_, amp_map_buf_,
                                       FlagMapConfidenceThreshold());
    }
  }
  // lock all buffers while swapping
  std::lock_guard<std::mutex> lock(image_mutex_);

  bool need_update = !BadInput() && (data_requested_ || frame_counter_ == 0);
  if (need_update) {
    // when update is done, swap buffers to front
    SwapBuffers();
    if (IsRecording()) {
      WriteImage(file_base_path_);
    }
    data_requested_ = false;

    // call callbacks
    for (const auto &callback : update_callbacks_) {
      callback.second(*this);
    }
    InitializeImages();
  }

  return need_update;
}

/** Returns true on bad input */
auto reclib::DepthCamera::BadInput() -> bool { return bad_input_flag_; }

/**
Remove noise on zMap and xyz_map_
*/
void reclib::DepthCamera::RemoveNoise(CpuMat &xyz_map, CpuMat &amp_map,
                                      float confidence_thresh) {
  for (int r = 0; r < xyz_map.rows; ++r) {
    auto *ptr = xyz_map.ptr<vec3>(r);

    const float *ampptr = nullptr;
    if (amp_map.data != nullptr) {
      ampptr = amp_map.ptr<float>(r);
    }

    for (int c = 0; c < xyz_map.cols; ++c) {
      if (ptr[c][2] < kNoiseFilterLow ||
          (ampptr != nullptr && ampptr[c] < confidence_thresh)) {
        ptr[c][0] = ptr[c][1] = ptr[c][2] = 0.0F;
      }
    }
  }
}

auto reclib::DepthCamera::IsCapturing() -> bool { return !capture_interrupt_; }

auto reclib::DepthCamera::AddUpdateCallback(
    std::function<void(reclib::DepthCamera &)> func) -> int {
  int id = 0;
  if (update_callbacks_.empty()) {
    id = 0;
  } else {
    id = update_callbacks_.rbegin()->first + 1;
  }

  update_callbacks_[id] = std::move(func);
  return id;
}

void reclib::DepthCamera::RemoveUpdateCallback(int id) {
  update_callbacks_.erase(id);
}

auto reclib::DepthCamera::GetImageSize() const -> cv::Size {
  return {GetWidth(), GetHeight()};
}

auto reclib::DepthCamera::GetDepthImageSize() const -> cv::Size {
  return {GetDepthWidth(), GetDepthHeight()};
}

unsigned long int reclib::DepthCamera::GetFrameCounter() const {
  return frame_counter_;
}

auto reclib::DepthCamera::GetModelName() const -> const std::string {
  return "DepthCamera";
}

void reclib::DepthCamera::InitializeImages() {
  cv::Size sz = GetImageSize();
  cv::Size szd = GetDepthImageSize();

  // initialize back buffers, if necessary
  if (HasXYZMap() && xyz_map_buf_.size().area() != sz.area()) {
    xyz_map_buf_.release();
    xyz_map_buf_.create(sz, CV_32FC3);
    xyz_map_buf_.setTo(0);
  }

  if (HasRGBMap() && rgb_map_buf_.size().area() != sz.area()) {
    rgb_map_buf_.release();
    rgb_map_buf_.create(sz, CV_8UC3);
    rgb_map_buf_.setTo(0);
  }

  if (HasIRMap() && ir_map_buf_.size().area() != sz.area()) {
    ir_map_buf_.release();
    ir_map_buf_.create(sz, CV_32FC1);
    ir_map_buf_.setTo(0);
  }

  if (HasAmpMap() && amp_map_buf_.size().area() != sz.area()) {
    amp_map_buf_.release();
    amp_map_buf_.create(sz, CV_32F);
    amp_map_buf_.setTo(0);
  }

  if (HasFlagMap() && flag_map_buf_.size().area() != sz.area()) {
    flag_map_buf_.release();
    flag_map_buf_.create(sz, CV_8U);
    flag_map_buf_.setTo(0);
  }

  if (HasRawDepthMap() && raw_depth_map_buf_.size().area() != szd.area()) {
    raw_depth_map_buf_.release();
    raw_depth_map_buf_.create(szd, CV_32FC1);
    raw_depth_map_buf_.setTo(0);
  }
}

/** swap a single buffer */
void reclib::DepthCamera::SwapBuffer(bool (reclib::DepthCamera::*check_func)()
                                         const,
                                     CpuMat &img, CpuMat &buf) {
  if ((this->*check_func)()) {
    cv::swap(img, buf);
  } else {
    img.data = nullptr;
  }
}

/** swap all buffers */
void reclib::DepthCamera::SwapBuffers() {
  SwapBuffer(&reclib::DepthCamera::HasXYZMap, xyz_map_, xyz_map_buf_);
  SwapBuffer(&reclib::DepthCamera::HasRGBMap, rgb_map_, rgb_map_buf_);
  SwapBuffer(&reclib::DepthCamera::HasIRMap, ir_map_, ir_map_buf_);
  SwapBuffer(&reclib::DepthCamera::HasAmpMap, amp_map_, amp_map_buf_);
  SwapBuffer(&reclib::DepthCamera::HasFlagMap, flag_map_, flag_map_buf_);
  SwapBuffer(&reclib::DepthCamera::HasRawDepthMap, raw_depth_map_,
             raw_depth_map_buf_);
}

bool reclib::DepthCamera::WriteIntrinsics(const fs::path &destination,
                                          fs::path filename) const {
  if (filename.string().length() == 0) {
    filename = fs::path("cam_intrinsics.yaml");
  }

  reclib::utils::CreateDirectories(destination / filename);

  cv::FileStorage fs((destination / filename).string(), cv::FileStorage::WRITE);

  ExtendedIntrinsicParameters intrinsics = GetExtIntrinsics();
  ExtendedIntrinsicParameters depth_intrinsics = GetExtDepthIntrinsics();

  fs << "camera_model" << GetModelName();

  fs << "rgb-image_width" << intrinsics.image_width_;
  fs << "rgb-image_height" << intrinsics.image_height_;
  fs << "rgb-focal_x" << intrinsics.focal_x_;
  fs << "rgb-focal_y" << intrinsics.focal_y_;
  fs << "rgb-principal_x" << intrinsics.principal_x_;
  fs << "rgb-principal_y" << intrinsics.principal_y_;
  fs << "rgb-fovx" << fovx_;
  fs << "rgb-fovy" << fovy_;

  std::string distortion = "rgb-k";
  for (unsigned int i = 0; i < intrinsics.k_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << intrinsics.k_[i];
  }
  distortion = "rgb-p";
  for (unsigned int i = 0; i < intrinsics.p_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << intrinsics.p_[i];
  }

  fs << "depth-image_width" << depth_intrinsics.image_width_;
  fs << "depth-image_height" << depth_intrinsics.image_height_;
  fs << "depth-focal_x" << depth_intrinsics.focal_x_;
  fs << "depth-focal_y" << depth_intrinsics.focal_y_;
  fs << "depth-principal_x" << depth_intrinsics.principal_x_;
  fs << "depth-principal_y" << depth_intrinsics.principal_y_;
  fs << "depth-fovx" << dfovx_;
  fs << "depth-fovy" << dfovy_;

  distortion = "depth-k";
  for (unsigned int i = 0; i < depth_intrinsics.k_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << depth_intrinsics.k_[i];
  }
  distortion = "depth-p";
  for (unsigned int i = 0; i < depth_intrinsics.p_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << depth_intrinsics.p_[i];
  }

  fs.release();

  return true;
}

bool reclib::DepthCamera::WriteExtrinsics(const fs::path &destination,
                                          fs::path filename) const {
  if (filename.string().length() == 0) {
    filename = fs::path("cam_extrinsics.yaml");
  }

  reclib::utils::CreateDirectories(destination / filename);

  cv::FileStorage fs((destination / filename).string(), cv::FileStorage::WRITE);
  ExtrinsicParameters extrinsics_rgb2d = GetExtrinsicsRGBToD();

  fs << "camera_model" << GetModelName();

  fs << "col2d_right" << extrinsics_rgb2d.right_[0] << ","
     << extrinsics_rgb2d.right_[1] << "," << extrinsics_rgb2d.right_[2];
  fs << "col2d_up" << extrinsics_rgb2d.up_[0] << "," << extrinsics_rgb2d.up_[1]
     << "," << extrinsics_rgb2d.up_[2];
  fs << "col2d_dir" << extrinsics_rgb2d.dir_[0] << ","
     << extrinsics_rgb2d.dir_[1] << "," << extrinsics_rgb2d.dir_[2];
  fs << "col2d_eye" << extrinsics_rgb2d.eye_[0] << ","
     << extrinsics_rgb2d.eye_[1] << "," << extrinsics_rgb2d.eye_[2];

  fs.release();

  return true;
}

/**
write a frame into file located at "destination"
*/
auto reclib::DepthCamera::WriteImageYAML(const fs::path &destination,
                                         fs::path filename) -> bool {
  if (filename.string().length() == 0) {
    std::string fc = std::to_string(frame_counter_);
    unsigned int filling_zeros = 6;
    filename =
        fs::path(std::string(filling_zeros - fc.length(), '0') + fc + ".yaml");
  }

  cv::FileStorage fs((destination / filename).string(), cv::FileStorage::WRITE);

  std::lock_guard<std::mutex> lock(image_mutex_);

  fs << "xyz_map_" << xyz_map_;
  fs << "amp_map_" << amp_map_;
  fs << "flag_map_" << flag_map_;
  fs << "rgb_map_" << rgb_map_;
  fs << "ir_map_" << ir_map_;
  fs << "raw_depth_map_" << raw_depth_map_;

  fs.release();
  return true;
}

/**
Reads a frame from file located at "source"
*/
auto reclib::DepthCamera::ReadImageYAML(const fs::path &source,
                                        long unsigned int frame_counter)
    -> bool {
  cv::FileStorage fs;

  fs::path source_path = source;
  if (frame_counter >= 0) {
    std::string fc = std::to_string(frame_counter);
    unsigned int filling_zeros = 6;
    fs::path filename =
        fs::path(std::string(filling_zeros - fc.length(), '0') + fc + ".yaml");
    source_path = source / filename;
  }
  fs.open(source_path.string(), cv::FileStorage::READ);

  std::lock_guard<std::mutex> lock(image_mutex_);

  fs["xyz_map_"] >> xyz_map_;
  fs["amp_map_"] >> amp_map_;
  fs["flag_map_"] >> flag_map_;
  fs["rgb_map_"] >> rgb_map_;
  fs["ir_map_"] >> ir_map_;
  fs["raw_depth_map_"] >> raw_depth_map_;
  fs.release();

  // call callbacks
  for (const auto &callback : update_callbacks_) {
    callback.second(*this);
  }

  return !(xyz_map_.rows == 0 || amp_map_.rows == 0 || flag_map_.rows == 0);
}

/**
write a frame into file located at "destination"
*/
auto reclib::DepthCamera::WriteImage(const fs::path &destination,
                                     fs::path filename) const -> bool {
  bool return_val = false;
  if (filename.string().length() == 0) {
    std::string fc = std::to_string(frame_counter_);
    unsigned int filling_zeros = 6;
    filename = fs::path(std::string(filling_zeros - fc.length(), '0') + fc);
  }

  // translate negative values by half the float range
  float half_float_range = std::floor((pow(2, 16) - 1) / 2);
  float full_float_range = (pow(2, 16) - 1);

  WriteIntrinsics(destination);
  WriteExtrinsics(destination);

  // std::lock_guard<std::mutex>
  // lock(image_mutex_);

  // int filled_arrays = ((int)!xyz_map_.empty()) + ((int)!rgb_map_.empty()) +
  //                     ((int)!ir_map_.empty()) + ((int)!amp_map_.empty()) +
  //                     ((int)!flag_map_.empty()) +
  //                     ((int)!raw_depth_map_.empty());
  //
  // int should_be_filled = ((int)HasXYZMap()) + ((int)HasRGBMap()) +
  //                        ((int)HasIRMap()) + ((int)HasAmpMap()) +
  //                        ((int)HasFlagMap()) + ((int)HasRawDepthMap());
  //
  // if (filled_arrays != 0 && filled_arrays != should_be_filled) {
  //   std::cout << "xyz_map_: " << xyz_map_.size() << std::endl;
  //   std::cout << "rgb_map_: " << rgb_map_.size() << std::endl;
  //   std::cout << "ir_map_: " << ir_map_.size() << std::endl;
  //   std::cout << "amp_map_: " << amp_map_.size() << std::endl;
  //   std::cout << "flag_map_: " << flag_map_.size() << std::endl;
  //   std::cout << "raw_depth_map_: " << raw_depth_map_.size() << std::endl;
  //
  //   std::cout << "should_be_filled: " << should_be_filled << std::endl;
  //   throw std::runtime_error("whack.");
  // }
  //
  // if (frame_counter_ == 0) {
  //   std::cout << "xyz_map_: " << xyz_map_.size() << std::endl;
  //   std::cout << "rgb_map_: " << rgb_map_.size() << std::endl;
  //   std::cout << "ir_map_: " << ir_map_.size() << std::endl;
  //   std::cout << "amp_map_: " << amp_map_.size() << std::endl;
  //   std::cout << "flag_map_: " << flag_map_.size() << std::endl;
  //   std::cout << "raw_depth_map_: " << raw_depth_map_.size() << std::endl;
  // }

  if (HasXYZMap() && !xyz_map_.empty()) {
    reclib::utils::CreateDirectories(destination / xyz_dir);

    cv::FileStorage fsin(
        (destination / xyz_dir / fs::path("_info.yaml")).string(),
        cv::FileStorage::READ);

    int n_frames[3] = {0, 0, 0};
    fsin["number_frames_c0"] >> n_frames[0];
    fsin["number_frames_c1"] >> n_frames[1];
    fsin["number_frames_c2"] >> n_frames[2];
    int start_frame = 0;
    fsin["number_first_frame"] >> start_frame;

    fsin.release();

    for (unsigned int i = 0; i < 3; i++) {
      CpuMat xyz_channel;
      CpuMat xyz_channel_16bit;
      cv::extractChannel(xyz_map_, xyz_channel, i);

      CpuMat max;
      CpuMat min;
      cv::reduce(xyz_channel, max, 0, cv::REDUCE_MAX, -1);
      cv::reduce(max, max, 1, cv::REDUCE_MAX, -1);
      cv::reduce(xyz_channel, min, 0, cv::REDUCE_MIN, -1);
      cv::reduce(min, min, 1, cv::REDUCE_MIN, -1);
      if (max.at<float>(0, 0) > half_float_range + 1 ||
          min.at<float>(0, 0) < -half_float_range) {
        std::cout << "Could not save XYZ channel " << i
                  << ": Values outside of 16-bit half range "
                  << half_float_range << ": " << min << ", " << max
                  << std::endl;
        continue;
      }

      if (i < 2) {
        xyz_channel += half_float_range;
      }

      xyz_channel.convertTo(xyz_channel_16bit, CV_16UC1);

      bool success = cv::imwrite((destination / xyz_dir / filename).string() +
                                     "_c" + std::to_string(i) + ".png",
                                 xyz_channel_16bit);
      if (!success) {
        return false;
      }

      n_frames[i]++;

      return_val = true;
    }

    if (n_frames[0] != 0 || n_frames[1] != 0 || n_frames[2] != 0) {
      cv::FileStorage fsout(
          (destination / xyz_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::WRITE);
      fsout << "number_frames_c0" << n_frames[0];
      fsout << "number_frames_c1" << n_frames[1];
      fsout << "number_frames_c2" << n_frames[2];
      fsout << "number_first_frame" << (int)start_frame;

      fsout.release();
    } else {
      cv::FileStorage fsout(
          (destination / xyz_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::WRITE);
      fsout << "number_first_frame" << (int)frame_counter_;
      fsout.release();
    }
  }

  if (HasAmpMap() && !amp_map_.empty()) {
    reclib::utils::CreateDirectories(destination / amp_dir);

    CpuMat max;
    cv::reduce(amp_map_, max, 0, cv::REDUCE_MAX, -1);
    cv::reduce(max, max, 1, cv::REDUCE_MAX, -1);
    if (max.at<float>(0, 0) > full_float_range) {
      std::cout << "Could not save Amp map"
                << ": Values outside of 16-bit range " << full_float_range
                << ": " << max << std::endl;
    } else {
      CpuMat amp_16bit;
      amp_map_.convertTo(amp_16bit, CV_16UC1);
      cv::imwrite((destination / amp_dir / filename).string() + ".png",
                  amp_16bit);
      cv::FileStorage fsin(
          (destination / amp_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::READ);
      int n_frames = 0;
      fsin["number_frames"] >> n_frames;
      int start_frame = 0;
      fsin["number_first_frame"] >> start_frame;
      fsin.release();
      cv::FileStorage fsout(
          (destination / amp_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::WRITE);
      fsout << "number_frames" << n_frames + 1;
      if (n_frames == 0) {
        fsout << "number_first_frame" << (int)frame_counter_;
      } else {
        fsout << "number_first_frame" << (int)start_frame;
      }
      fsout.release();
      return_val = true;
    }
  }

  if (HasFlagMap() && !flag_map_.empty()) {
    reclib::utils::CreateDirectories(destination / flag_dir);

    cv::imwrite((destination / flag_dir / filename).string() + ".png",
                flag_map_);
    cv::FileStorage fsin(
        (destination / flag_dir / fs::path("_info.yaml")).string(),
        cv::FileStorage::READ);
    int n_frames = 0;
    fsin["number_frames"] >> n_frames;
    int start_frame = 0;
    fsin["number_first_frame"] >> start_frame;
    fsin.release();
    cv::FileStorage fsout(
        (destination / flag_dir / fs::path("_info.yaml")).string(),
        cv::FileStorage::WRITE);
    fsout << "number_frames" << n_frames + 1;
    if (n_frames == 0) {
      fsout << "number_first_frame" << (int)frame_counter_;
    } else {
      fsout << "number_first_frame" << (int)start_frame;
    }
    fsout.release();
    return_val = true;
  }

  if (HasRGBMap() && !rgb_map_.empty()) {
    reclib::utils::CreateDirectories(destination / rgb_dir);

    cv::imwrite((destination / rgb_dir / filename).string() + ".png", rgb_map_);
    cv::FileStorage fsin(
        (destination / rgb_dir / fs::path("_info.yaml")).string(),
        cv::FileStorage::READ);
    int n_frames = 0;
    fsin["number_frames"] >> n_frames;
    fsin.release();
    cv::FileStorage fsout(
        (destination / rgb_dir / fs::path("_info.yaml")).string(),
        cv::FileStorage::WRITE);
    fsout << "number_frames" << n_frames + 1;
    int start_frame = 0;
    fsin["number_first_frame"] >> start_frame;
    if (n_frames == 0) {
      fsout << "number_first_frame" << (int)frame_counter_;
    } else {
      fsout << "number_first_frame" << (int)start_frame;
    }
    fsout.release();
    return_val = true;
  }

  if (HasIRMap() && !ir_map_.empty()) {
    reclib::utils::CreateDirectories(destination / ir_dir);

    CpuMat max;
    cv::reduce(ir_map_, max, 0, cv::REDUCE_MAX, -1);
    cv::reduce(max, max, 1, cv::REDUCE_MAX, -1);
    if (max.at<float>(0, 0) > full_float_range) {
      std::cout << "Could not save Depth map"
                << ": Values outside of 16-bit range " << full_float_range
                << ": " << max << std::endl;
    } else {
      CpuMat ir_16bit;
      ir_map_.convertTo(ir_16bit, CV_16UC1);
      cv::imwrite((destination / ir_dir / filename).string() + ".png",
                  ir_16bit);
      cv::FileStorage fsin(
          (destination / ir_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::READ);
      int n_frames = 0;
      fsin["number_frames"] >> n_frames;
      int start_frame = 0;
      fsin["number_first_frame"] >> start_frame;
      fsin.release();
      cv::FileStorage fsout(
          (destination / depth_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::WRITE);
      fsout << "number_frames" << n_frames + 1;
      if (n_frames == 0) {
        fsout << "number_first_frame" << (int)frame_counter_;
      } else {
        fsout << "number_first_frame" << (int)start_frame;
      }
      fsout.release();
      return_val = true;
    }
  }

  if (HasRawDepthMap() && !raw_depth_map_.empty()) {
    reclib::utils::CreateDirectories(destination / depth_dir);

    CpuMat max;
    cv::reduce(raw_depth_map_, max, 0, cv::REDUCE_MAX, -1);
    cv::reduce(max, max, 1, cv::REDUCE_MAX, -1);
    if (max.at<float>(0, 0) > full_float_range) {
      std::cout << "Could not save Depth map"
                << ": Values outside of 16-bit range " << full_float_range
                << ": " << max << std::endl;
    } else {
      CpuMat depth_16bit;
      raw_depth_map_.convertTo(depth_16bit, CV_16UC1);
      cv::imwrite((destination / depth_dir / filename).string() + ".png",
                  depth_16bit);
      cv::FileStorage fsin(
          (destination / depth_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::READ);
      int n_frames = 0;
      fsin["number_frames"] >> n_frames;
      int start_frame = 0;
      fsin["number_first_frame"] >> start_frame;
      fsin.release();
      cv::FileStorage fsout(
          (destination / depth_dir / fs::path("_info.yaml")).string(),
          cv::FileStorage::WRITE);
      fsout << "number_frames" << n_frames + 1;
      if (n_frames == 0) {
        fsout << "number_first_frame" << (int)frame_counter_;
      } else {
        fsout << "number_first_frame" << (int)start_frame;
      }
      fsout.release();
      return_val = true;
    }
  }

  return return_val;
}

/**
Reads a frame from file located at "source"
*/
auto reclib::DepthCamera::ReadImage(CpuMat &xyz_map, CpuMat &rgb_map,
                                    CpuMat &ir_map, CpuMat &amp_map,
                                    CpuMat &flag_map, CpuMat &raw_depth_map,
                                    const fs::path &source,
                                    long unsigned int frame_counter) -> bool {
  bool return_val = false;
  fs::path filename = "";
  if (frame_counter >= 0) {
    std::string fc = std::to_string(frame_counter);
    unsigned int filling_zeros = 6;
    filename = fs::path(std::string(filling_zeros - fc.length(), '0') + fc);
  }

  std::lock_guard<std::mutex> lock(image_mutex_);
  float half_float_range = std::floor((pow(2, 16) - 1) / 2);

  if (HasXYZMap()) {
    if (xyz_map.empty()) {
      xyz_map.create(GetHeight(), GetWidth(), CV_32FC3);
    }

    for (unsigned int i = 0; i < 3; i++) {
      std::string filepath = (source / xyz_dir / filename).string() + "_c" +
                             std::to_string(i) + ".png";

      if (fs::exists(fs::path(filepath))) {
        CpuMat input = cv::imread(filepath, cv::IMREAD_UNCHANGED);
        CpuMat xyz_channel;
        input.convertTo(xyz_channel, CV_32FC1);
        if (i < 2) {
          xyz_channel -= half_float_range;
        }
        cv::insertChannel(xyz_channel, xyz_map, i);

        return_val = true;
      }
    }
  }

  if (HasRGBMap()) {
    std::string filepath = (source / rgb_dir / filename).string() + ".png";
    if (fs::exists(fs::path(filepath))) {
      rgb_map = cv::imread(filepath, cv::IMREAD_UNCHANGED);
      return_val = true;
    }
  }

  if (HasIRMap()) {
    std::string filepath = (source / ir_dir / filename).string() + ".png";
    if (fs::exists(fs::path(filepath))) {
      ir_map = cv::imread(filepath, cv::IMREAD_UNCHANGED);
      return_val = true;
    }
  }

  if (HasAmpMap()) {
    std::string filepath = (source / amp_dir / filename).string() + ".png";
    if (fs::exists(fs::path(filepath))) {
      CpuMat input = cv::imread(filepath, cv::IMREAD_UNCHANGED);
      input.convertTo(amp_map, CV_32FC1);
      return_val = true;
    }
  }

  if (HasFlagMap()) {
    std::string filepath = (source / flag_dir / filename).string() + ".png";
    if (fs::exists(fs::path(filepath))) {
      flag_map = cv::imread(filepath, cv::IMREAD_UNCHANGED);
      return_val = true;
    }
  }

  if (HasRawDepthMap()) {
    std::string filepath = (source / depth_dir / filename).string() + ".png";
    if (fs::exists(fs::path(filepath))) {
      CpuMat input = cv::imread(filepath, cv::IMREAD_ANYDEPTH);

      input.convertTo(raw_depth_map, CV_32FC1);
      raw_depth_map /= depth_scaling_;
      return_val = true;
    }
  }

  // call callbacks
  for (const auto &callback : update_callbacks_) {
    callback.second(*this);
  }

  return return_val;
}

auto reclib::DepthCamera::GetXYZMapAndUpdateBufs() -> const CpuMat {
  if (!HasXYZMap()) {
    throw;
  }
  if (!HasNextFrame()) {
    throw std::runtime_error("No new frame available");
  }

  std::lock_guard<std::mutex> lock(image_mutex_);

  CpuMat tmp;
  cv::swap(tmp, xyz_map_);
  data_requested_ = true;
  std::cout << "GetXYZ: xyz: " << xyz_map_.size() << std::endl;
  return tmp;
}

auto reclib::DepthCamera::GetAmpMapAndUpdateBufs() -> const CpuMat {
  if (!HasAmpMap()) {
    throw;
  }
  if (!HasNextFrame()) {
    throw std::runtime_error("No new frame available");
  }

  std::lock_guard<std::mutex> lock(image_mutex_);

  CpuMat tmp;
  cv::swap(tmp, amp_map_);
  data_requested_ = true;
  return tmp;
}

auto reclib::DepthCamera::GetFlagMapAndUpdateBufs() -> const CpuMat {
  if (!HasFlagMap()) {
    throw;
  }
  if (!HasNextFrame()) {
    throw std::runtime_error("No new frame available");
  }

  std::lock_guard<std::mutex> lock(image_mutex_);

  CpuMat tmp;
  cv::swap(tmp, flag_map_);
  data_requested_ = true;
  return tmp;
}

auto reclib::DepthCamera::GetRGBMapAndUpdateBufs() -> const CpuMat {
  if (!HasRGBMap()) {
    throw;
  }
  if (!HasNextFrame()) {
    throw std::runtime_error("No new frame available");
  }

  std::lock_guard<std::mutex> lock(image_mutex_);

  CpuMat tmp;
  cv::swap(tmp, rgb_map_);
  data_requested_ = true;
  return tmp;
}

auto reclib::DepthCamera::GetIRMapAndUpdateBufs() -> const CpuMat {
  if (!HasIRMap()) {
    throw;
  }
  if (!HasNextFrame()) {
    throw std::runtime_error("No new frame available");
  }

  std::lock_guard<std::mutex> lock(image_mutex_);

  CpuMat tmp;
  cv::swap(tmp, ir_map_);
  data_requested_ = true;
  return tmp;
}

auto reclib::DepthCamera::GetRawDepthMapAndUpdateBufs() -> const CpuMat {
  if (!HasRawDepthMap()) {
    throw;
  }
  if (!HasNextFrame()) {
    throw std::runtime_error("No new frame available");
  }

  std::lock_guard<std::mutex> lock(image_mutex_);

  CpuMat tmp;
  cv::swap(tmp, raw_depth_map_);
  data_requested_ = true;
  return tmp;
}

void reclib::DepthCamera::GetMaps(CpuMat &depth, CpuMat &rgb, CpuMat &xyz,
                                  CpuMat &ir, CpuMat &amp, CpuMat &flag) {
  if (!HasNextFrame()) {
    throw std::runtime_error("No new frame available");
  }

  std::lock_guard<std::mutex> lock(image_mutex_);

  if (HasRawDepthMap()) {
    cv::swap(depth, raw_depth_map_);
    data_requested_ = true;
  }
  if (HasRGBMap()) {
    cv::swap(rgb, rgb_map_);
    data_requested_ = true;
  }
  if (HasXYZMap()) {
    cv::swap(xyz, xyz_map_);
    data_requested_ = true;
  }
  if (HasIRMap()) {
    cv::swap(ir, ir_map_);
    data_requested_ = true;
  }
  if (HasAmpMap()) {
    cv::swap(amp, amp_map_);
    data_requested_ = true;
  }
  if (HasFlagMap()) {
    cv::swap(flag, flag_map_);
    data_requested_ = true;
  }
}

void reclib::DepthCamera::GetMaps(CpuMat &depth, CpuMat &rgb, CpuMat &xyz) {
  std::lock_guard<std::mutex> lock(image_mutex_);

  if (HasRawDepthMap()) {
    cv::swap(depth, raw_depth_map_);
    data_requested_ = true;
  }
  if (HasRGBMap()) {
    cv::swap(rgb, rgb_map_);
    data_requested_ = true;
  }
  if (HasXYZMap()) {
    cv::swap(xyz, xyz_map_);
    data_requested_ = true;
  }
}

void reclib::DepthCamera::GetMaps(CpuMat &depth, CpuMat &rgb, CpuMat &xyz,
                                  CpuMat &ir) {
  std::lock_guard<std::mutex> lock(image_mutex_);

  if (HasRawDepthMap()) {
    cv::swap(depth, raw_depth_map_);
    data_requested_ = true;
  }
  if (HasRGBMap()) {
    cv::swap(rgb, rgb_map_);
    data_requested_ = true;
  }
  if (HasXYZMap()) {
    cv::swap(xyz, xyz_map_);
    data_requested_ = true;
  }
  if (HasIRMap()) {
    cv::swap(ir, ir_map_);
    data_requested_ = true;
  }
}

auto reclib::DepthCamera::GetTimestamp() const -> uint64_t {
  return timestamp_;
}

auto reclib::DepthCamera::GetIntrinsics() const -> reclib::IntrinsicParameters {
  IntrinsicParameters params{};
  params.image_width_ = GetWidth();
  params.image_height_ = GetHeight();
  params.focal_x_ = fx_ * image_scaling_;
  params.focal_y_ = fy_ * image_scaling_;
  params.principal_x_ = (cx_ + 0.5f) * image_scaling_ - 0.5f;
  params.principal_y_ = (cy_ + 0.5f) * image_scaling_ - 0.5f;
  return params;
}

auto reclib::DepthCamera::GetExtrinsicsRGBToD() const
    -> reclib::ExtrinsicParameters {
  return extrinsics_rgbtod_;
}

auto reclib::DepthCamera::GetDepthIntrinsics() const
    -> reclib::IntrinsicParameters {
  IntrinsicParameters params{};
  params.image_width_ = GetDepthWidth();
  params.image_height_ = GetDepthHeight();
  params.focal_x_ = dfx_ * image_scaling_;
  params.focal_y_ = dfy_ * image_scaling_;
  params.principal_x_ = (dcx_ + 0.5f) * image_scaling_ - 0.5f;
  params.principal_y_ = (dcy_ + 0.5f) * image_scaling_ - 0.5f;
  return params;
}

auto reclib::DepthCamera::GetExtIntrinsics() const
    -> reclib::ExtendedIntrinsicParameters {
  ExtendedIntrinsicParameters params{};
  params.image_width_ = GetWidth();
  params.image_height_ = GetHeight();
  params.focal_x_ = fx_ * image_scaling_;
  params.focal_y_ = fy_ * image_scaling_;
  params.principal_x_ = (cx_ + 0.5f) * image_scaling_ - 0.5f;
  params.principal_y_ = (cy_ + 0.5f) * image_scaling_ - 0.5f;
  params.fovx_ = fovx_;
  params.fovy_ = fovy_;

  for (unsigned int i = 0; i < rad_distortion_.size(); i++) {
    params.k_.push_back(rad_distortion_[i]);
  }
  for (unsigned int i = 0; i < tang_distortion.size(); i++) {
    params.p_.push_back(rad_distortion_[i]);
  }

  return params;
}

auto reclib::DepthCamera::GetExtDepthIntrinsics() const
    -> reclib::ExtendedIntrinsicParameters {
  ExtendedIntrinsicParameters params{};
  params.image_width_ = GetDepthWidth();
  params.image_height_ = GetDepthHeight();
  params.focal_x_ = dfx_ * image_scaling_;
  params.focal_y_ = dfy_ * image_scaling_;
  params.principal_x_ = (dcx_ + 0.5f) * image_scaling_ - 0.5f;
  params.principal_y_ = (dcy_ + 0.5f) * image_scaling_ - 0.5f;
  params.fovx_ = dfovx_;
  params.fovy_ = dfovy_;

  for (unsigned int i = 0; i < drad_distortion_.size(); i++) {
    params.k_.push_back(drad_distortion_[i]);
  }
  for (unsigned int i = 0; i < dtang_distortion.size(); i++) {
    params.p_.push_back(drad_distortion_[i]);
  }
  return params;
}

float reclib::DepthCamera::GetFovX() const { return fovx_; }

float reclib::DepthCamera::GetFovY() const { return fovy_; }

float reclib::DepthCamera::GetDepthFovX() const { return dfovx_; }

float reclib::DepthCamera::GetDepthFovY() const { return dfovy_; }

auto reclib::DepthCamera::HasXYZMap() const -> bool {
  // Assume no xyz map, unless overridden
  return has_xyz_map_;
}

auto reclib::DepthCamera::HasAmpMap() const -> bool {
  // Assume no amp map, unless overridden
  return has_amp_map_;
}

auto reclib::DepthCamera::HasFlagMap() const -> bool {
  // Assume no flag map, unless overridden
  return has_flag_map_;
}

auto reclib::DepthCamera::HasRGBMap() const -> bool {
  // Assume no RGB image, unless overridden
  return has_rgb_map_;
}

auto reclib::DepthCamera::HasIRMap() const -> bool {
  // Assume no IR image, unless overridden
  return has_ir_map_;
}

auto reclib::DepthCamera::HasRawDepthMap() const -> bool {
  // Assume no depth map image, unless overridden
  return has_raw_depth_map_;
}

void reclib::DepthCamera::DisableXYZMap() { has_xyz_map_ = false; }
void reclib::DepthCamera::DisableRGBMap() { has_rgb_map_ = false; }
void reclib::DepthCamera::DisableIRMap() { has_ir_map_ = false; }
void reclib::DepthCamera::DisableAmpMap() { has_amp_map_ = false; }
void reclib::DepthCamera::DisableFlagMap() { has_flag_map_ = false; }
void reclib::DepthCamera::DisableRawDepthMap() { has_raw_depth_map_ = false; }

void reclib::DepthCamera::EnableXYZMap() { has_xyz_map_ = true; }
void reclib::DepthCamera::EnableRGBMap() { has_rgb_map_ = true; }
void reclib::DepthCamera::EnableIRMap() { has_ir_map_ = true; }
void reclib::DepthCamera::EnableAmpMap() { has_amp_map_ = true; }
void reclib::DepthCamera::EnableFlagMap() { has_flag_map_ = true; }
void reclib::DepthCamera::EnableRawDepthMap() { has_raw_depth_map_ = true; }

// note: depth camera must have XYZ map

auto reclib::DepthCamera::AmpMapInvalidFlagValue() const -> int { return -1; }

auto reclib::DepthCamera::FlagMapConfidenceThreshold() const -> float {
  return 0.5;
}

// ------------------------------------------------------------
// ------------------------------------------------------------
// DefaultCamera
// ------------------------------------------------------------
// ------------------------------------------------------------

reclib::DefaultCamera::DefaultCamera(const fs::path &base_path,
                                     bool repeat_sequence, float depth_scaling,
                                     unsigned int start_frame)
    : reclib::DepthCamera(depth_scaling, 1.f, base_path),
      repeat_sequence_(repeat_sequence),
      filesystem_frame_counter_(0),
      start_filesystem_frame_counter_(start_frame),
      width_(0),
      height_(0),
      depth_width_(0),
      depth_height_(0) {
  if (!fs::is_directory(base_path)) {
    throw std::runtime_error(
        "Error in DefaultCamera: base_path must be a valid directory.");
  }
  if (!fs::exists(base_path / fs::path("cam_intrinsics.yaml"))) {
    if (fs::exists(base_path / fs::path("calibration.json"))) {
#if HAS_K4A
      k4a::calibration calib =
          reclib::KinectAzureCamera::CalibrationFromFile(base_path);

      fx_ = calib.color_camera_calibration.intrinsics.parameters.param.fx;
      fy_ = calib.color_camera_calibration.intrinsics.parameters.param.fy;
      cx_ = calib.color_camera_calibration.intrinsics.parameters.param.cx;
      cy_ = calib.color_camera_calibration.intrinsics.parameters.param.cy;

      dfx_ = calib.depth_camera_calibration.intrinsics.parameters.param.fx;
      dfy_ = calib.depth_camera_calibration.intrinsics.parameters.param.fy;
      dcx_ = calib.depth_camera_calibration.intrinsics.parameters.param.cx;
      dcy_ = calib.depth_camera_calibration.intrinsics.parameters.param.cy;

      _RECLIB_ASSERT(fs::exists(base_path / "tags.xml"));
      std::ifstream str((base_path / "tags.xml"));
      std::string line;
      while (std::getline(str, line)) {
        if (line.find("K4A_DEPTH_MODE") != std::string::npos) {
          std::getline(str, line);
          int start_pos = line.find("<String>") + 8;
          int end_pos = line.find("</String>");
          int len = end_pos - start_pos;
          std::string mode = line.substr(start_pos, len);

          if (mode.compare("OFF") == 0) {
            dfovx_ = 0;
            dfovy_ = 0;
            depth_width_ = 0;
            depth_height_ = 0;
          } else if (mode.compare("NFOV_2X2BINNED") == 0) {
            dfovx_ = 75;
            dfovy_ = 65;
            depth_width_ = 320;
            depth_height_ = 288;
          } else if (mode.compare("NFOV_UNBINNED") == 0) {
            dfovx_ = 75;
            dfovy_ = 65;

            depth_width_ = 640;
            depth_height_ = 576;
          } else if (mode.compare("WFOV_2X2BINNED") == 0) {
            dfovx_ = 120;
            dfovy_ = 120;
            depth_width_ = 512;
            depth_height_ = 512;
          } else if (mode.compare("WFOV_UNBINNED ") == 0) {
            dfovx_ = 120;
            dfovy_ = 120;
            depth_width_ = 1024;
            depth_height_ = 1024;
          } else if (mode.compare("PASSIVE_IR") == 0) {
            dfovx_ = 0;
            dfovy_ = 0;
            depth_width_ = 0;
            depth_height_ = 0;

          } else {
            throw std::runtime_error("Unknown depth mode.");
          }
        }
        if (line.find("K4A_COLOR_MODE") != std::string::npos) {
          std::getline(str, line);
          int start_pos = line.find("<String>") + 8;
          int end_pos = line.find("</String>");
          int len = end_pos - start_pos;
          int split = line.substr(start_pos, len).find("_") + start_pos;
          std::string image_format = line.substr(start_pos, split - start_pos);
          std::string color_resolution =
              line.substr(split + 1, end_pos - split - 1);

          if (color_resolution.compare("OFF") == 0) {
            fovx_ = 0;
            fovy_ = 0;
            width_ = 0;
            height_ = 0;
          } else if (color_resolution.compare("720P") == 0) {
            fovx_ = 90;
            fovy_ = 59;
            width_ = 1280;
            height_ = 720;
          } else if (color_resolution.compare("1080P") == 0) {
            fovx_ = 90;
            fovy_ = 59;
            width_ = 1920;
            height_ = 1080;
          } else if (color_resolution.compare("1440P") == 0) {
            fovx_ = 90;
            fovy_ = 59;
            width_ = 2560;
            height_ = 1440;
          } else if (color_resolution.compare("1536P") == 0) {
            fovx_ = 90;
            fovy_ = 74.3;
            width_ = 2048;
            height_ = 1536;
          } else if (color_resolution.compare("2160P") == 0) {
            fovx_ = 90;
            fovy_ = 59;
            width_ = 3840;
            height_ = 2160;
          } else if (color_resolution.compare("3072P") == 0) {
            fovx_ = 90;
            fovy_ = 74.3;
            width_ = 4096;
            height_ = 3072;
          } else {
            throw std::runtime_error("Unknown color resolution mode.");
          }
        }
      }
#endif
    } else {
      throw std::runtime_error(
          "Error in DefaultCamera: Cannot find intrinsic parameters.");
    }

  } else {
    cv::FileStorage fsin((base_path / fs::path("cam_intrinsics.yaml")).string(),
                         cv::FileStorage::READ);

    std::string model_name;

    float fx, fy, cx, cy, dfx, dfy, dcx, dcy, fovx, fovy, dfovx, dfovy;

    fsin["rgb-image_width"] >> width_;
    fsin["rgb-image_height"] >> height_;
    fsin["rgb-focal_x"] >> fx;
    fsin["rgb-focal_y"] >> fy;
    fsin["rgb-principal_x"] >> cx;
    fsin["rgb-principal_y"] >> cy;
    fsin["rgb-fovx"] >> fovx;
    fsin["rgb-fovy"] >> fovy;

    fsin["depth-image_width"] >> depth_width_;
    fsin["depth-image_height"] >> depth_height_;
    fsin["depth-focal_x"] >> dfx;
    fsin["depth-focal_y"] >> dfy;
    fsin["depth-principal_x"] >> dcx;
    fsin["depth-principal_y"] >> dcy;
    fsin["depth-fovx"] >> dfovx;
    fsin["depth-fovy"] >> dfovy;

    fsin.release();

    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    dfx_ = dfx;
    dfy_ = dfy;
    dcx_ = dcx;
    dcy_ = dcy;
    dfovx_ = dfovx;
    dfovy_ = dfovy;
    fovx_ = fovx;
    fovy_ = fovy;
  }

  has_raw_depth_map_ = fs::exists(base_path / depth_dir);
  has_xyz_map_ = fs::exists(base_path / xyz_dir);
  has_amp_map_ = fs::exists(base_path / amp_dir);
  has_rgb_map_ = fs::exists(base_path / rgb_dir);
  has_ir_map_ = fs::exists(base_path / ir_dir);
  has_flag_map_ = fs::exists(base_path / flag_dir);

  UpdateFilesystemFrameCounter();
  // start_filesystem_frame_counter_ = filesystem_frame_counter_;
  std::cout << "DefaultCamera: Start reading at frame "
            << start_filesystem_frame_counter_ << std::endl;

  std::string fc = std::to_string(filesystem_frame_counter_);
  unsigned int filling_zeros = 6;
  fs::path filename =
      fs::path(std::string(filling_zeros - fc.length(), '0') + fc);

  if (HasRawDepthMap()) {
    CpuMat depth =
        cv::imread((file_base_path_ / depth_dir / filename).string() + ".png",
                   cv::IMREAD_UNCHANGED);
    _RECLIB_ASSERT_EQ(depth.cols, (int)depth_width_);
    _RECLIB_ASSERT_EQ(depth.rows, (int)depth_height_);
  }

  if (HasRGBMap()) {
    CpuMat img =
        cv::imread((file_base_path_ / rgb_dir / filename).string() + ".png",
                   cv::IMREAD_UNCHANGED);
    _RECLIB_ASSERT_EQ(img.cols, (int)width_);
    _RECLIB_ASSERT_EQ(img.rows, (int)height_);
  } else if (HasXYZMap()) {
    CpuMat img =
        cv::imread((file_base_path_ / xyz_dir / filename).string() + ".png",
                   cv::IMREAD_UNCHANGED);
    _RECLIB_ASSERT_EQ(img.cols, (int)width_);
    _RECLIB_ASSERT_EQ(img.rows, (int)height_);
  } else if (HasIRMap()) {
    CpuMat img =
        cv::imread((file_base_path_ / ir_dir / filename).string() + ".png",
                   cv::IMREAD_UNCHANGED);
    _RECLIB_ASSERT_EQ(img.cols, (int)depth_width_);
    _RECLIB_ASSERT_EQ(img.rows, (int)depth_height_);
  } else if (HasAmpMap()) {
    CpuMat img =
        cv::imread((file_base_path_ / amp_dir / filename).string() + ".png",
                   cv::IMREAD_UNCHANGED);
    _RECLIB_ASSERT_EQ(img.cols, (int)width_);
    _RECLIB_ASSERT_EQ(img.rows, (int)height_);
  } else if (HasFlagMap()) {
    CpuMat img =
        cv::imread((file_base_path_ / flag_dir / filename).string() + ".png",
                   cv::IMREAD_UNCHANGED);
    _RECLIB_ASSERT_EQ(img.cols, (int)width_);
    _RECLIB_ASSERT_EQ(img.rows, (int)height_);
  } else if (depth_width_ != 0 && depth_height_ != 0) {
    width_ = depth_width_;
    height_ = depth_height_;
  } else {
    throw std::runtime_error(
        "Could not set width and height parameters, since the specified path "
        "contains no files");
  }

  filesystem_frame_counter_--;
}

reclib::DefaultCamera::~DefaultCamera() {}

const std::string reclib::DefaultCamera::GetModelName() const {
  return "DefaultCamera";
}

int reclib::DefaultCamera::GetWidth() const { return width_; }

int reclib::DefaultCamera::GetHeight() const { return height_; }

int reclib::DefaultCamera::GetDepthWidth() const { return depth_width_; }

int reclib::DefaultCamera::GetDepthHeight() const { return depth_height_; }

long int NextValidFrameId(const fs::path &base_path,
                          const std::string &extension,
                          unsigned long int counter, unsigned long int offset,
                          unsigned long int max_count) {
  fs::path filename = "";
  do {
    if (counter >= max_count + offset) {
      return -1;
    }
    std::string fc = std::to_string(counter);
    unsigned int filling_zeros = 6;
    filename = fs::path(std::string(filling_zeros - fc.length(), '0') + fc +
                        extension);
    counter++;

  } while (!fs::exists(base_path / filename));
  return counter - 1;
}

long int reclib::DefaultCamera::GetFilesystemCounter() {
  return filesystem_frame_counter_;
}

fs::path reclib::DefaultCamera::GetFilename() {
  fs::path filename = "";

  std::string fc = std::to_string(filesystem_frame_counter_);
  unsigned int filling_zeros = 6;
  filename = fs::path(std::string(filling_zeros - fc.length(), '0') + fc);

  return filename;
}

void reclib::DefaultCamera::UpdateFilesystemFrameCounter() {
  if (filesystem_frame_counter_ < 0) {
    bad_input_flag_ = true;
    return;
  }

  int n_frames = 0;
  fs::path dir_path = "";
  if (HasAmpMap()) {
    dir_path = file_base_path_ / amp_dir;
  } else if (HasFlagMap()) {
    dir_path = file_base_path_ / flag_dir;
  } else if (HasRGBMap()) {
    dir_path = file_base_path_ / rgb_dir;
  } else if (HasIRMap()) {
    dir_path = file_base_path_ / ir_dir;
  } else if (HasRawDepthMap()) {
    dir_path = file_base_path_ / depth_dir;
  } else {
    bad_input_flag_ = true;
    return;
  }

  int start = 0;
  if (fs::exists((dir_path / fs::path("_info.yaml")))) {
    cv::FileStorage fs((dir_path / fs::path("_info.yaml")).string(),
                       cv::FileStorage::READ);
    fs["number_frames"] >> n_frames;

    fs["number_first_frame"] >> start;
    fs.release();
  } else {
    start = 1;
    fs::directory_iterator iter(dir_path);
    for (; iter != fs::directory_iterator(); iter++) {
      if (!iter->is_directory()) {
        n_frames++;
      }
    }
  }
  if (start_filesystem_frame_counter_ == 0)
    start_filesystem_frame_counter_ = start;

  filesystem_frame_counter_ =
      std::max(start_filesystem_frame_counter_, filesystem_frame_counter_);

  if (n_frames > 0) {
    long int counter =
        NextValidFrameId(dir_path, ".png", filesystem_frame_counter_,
                         start_filesystem_frame_counter_, n_frames);
    if (counter < 0) {
      if (repeat_sequence_) {
        // reset to first frame
        filesystem_frame_counter_ = start_filesystem_frame_counter_;
      } else {
        filesystem_frame_counter_ = -1;
        bad_input_flag_ = true;
        return;
      }
    } else {
      filesystem_frame_counter_ = counter;
    }
  } else {
    throw std::runtime_error(
        "DefaultCamera::UpdateFilesystemFrameCounter: n_frames is 0 but "
        "directory is not empty?");
  }
}

void reclib::DefaultCamera::Update(CpuMat &xyz_map, CpuMat &rgb_map,
                                   CpuMat &ir_map, CpuMat &amp_map,
                                   CpuMat &flag_map, CpuMat &raw_depth_map,
                                   bool first_frame) {
  if (filesystem_frame_counter_ < 0 && frame_counter_ > 0) {
    bad_input_flag_ = true;
    return;
  }
  // lock access to data_requested_ variable
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    // std::cout << "data_requested: " << data_requested_
    //           << " frame counter: " << frame_counter_
    //           << " filesystem: " << filesystem_frame_counter_
    //           << " start: " << start_filesystem_frame_counter_ <<
    //           std::endl;

    if (!(data_requested_) &&
        (frame_counter_ > 0 ||
         (frame_counter_ == 0 &&
          filesystem_frame_counter_ == start_filesystem_frame_counter_))) {
      bad_input_flag_ = true;
      // return;
    } else {
      filesystem_frame_counter_++;
      bad_input_flag_ = false;
      UpdateFilesystemFrameCounter();
    }
  }

  if (!bad_input_flag_) {
    if (!ReadImage(xyz_map, rgb_map, ir_map, amp_map, flag_map, raw_depth_map,
                   file_base_path_, filesystem_frame_counter_)) {
      bad_input_flag_ = true;
    }
    return;
  }
}

void reclib::DefaultCamera::StartRecording(const fs::path &file_base_path) {
  return;
}

// ------------------------------------------------------------
// ------------------------------------------------------------
// KinectV2FN2Camera
// ------------------------------------------------------------
// ------------------------------------------------------------

#if HAS_FREENECT2

const int reclib::KinectV2FN2Camera::kDefaultRgbWidth = 1920;
const int reclib::KinectV2FN2Camera::kDefaultRgbHeight = 1080;
const int reclib::KinectV2FN2Camera::kDefaultDepthWidth = 512;
const int reclib::KinectV2FN2Camera::kDefaultDepthHeight = 424;

reclib::KinectV2FN2Camera::KinectV2FN2Camera(std::string serial, bool use_kde,
                                             float image_scaling,
                                             float depth_scaling, bool verbose,
                                             bool compute_xy)
    : reclib::DepthCamera(depth_scaling, image_scaling),
      use_kde_(use_kde),
      serial_(std::move(serial)),
      verbose_(verbose),
      compute_xy_(compute_xy) {
  has_xyz_map_ = true;
  has_rgb_map_ = true;
  has_raw_depth_map_ = true;

  if (!verbose) {
    libfreenect2::setGlobalLogger(nullptr);
  }
  freenect2_ = std::make_unique<libfreenect2::Freenect2>();
  if (freenect2_->enumerateDevices() == 0) {
    std::cerr << "Fatal: No Freenect2 devices found" << std::endl;
    device_open_flag_ = false;
    return;
  }

#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
  if (use_kde_) {
    pipeline_ = new libfreenect2::CudaPacketPipeline(-1);
  } else {
    pipeline_ = new libfreenect2::CudaKdePacketPipeline(-1);
  }
#elif defined(LIBFREENECT2_WITH_OPENCL_SUPPORT)
  if (use_kde_)
    pipeline_ = new libfreenect2::OpenCLKdePacketPipeline(-1);
  else
    pipeline_ = new libfreenect2::OpenCLPacketPipeline(-1);
#elif defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
  pipeline_ = new libfreenect2::OpenGLPacketPipeline();
#else
  pipeline_ = new libfreenect2::CPUPacketPipeline();
#endif

  if (serial_.empty()) {
    serial_ = freenect2_->getDefaultDeviceSerialNumber();
  }

  if (pipeline_ == nullptr) {
    return;
  }
  device_ = std::unique_ptr<libfreenect2::Freenect2Device>(
      freenect2_->openDevice(serial_, pipeline_));

  listener_ = std::make_unique<libfreenect2::SyncMultiFrameListener>(
      libfreenect2::Frame::Color | libfreenect2::Frame::Depth);

  scaled_width_ = image_scaling_ * kDefaultRgbWidth;
  scaled_height_ = image_scaling_ * kDefaultRgbHeight;
  scaled_dwidth_ = image_scaling_ * kDefaultDepthWidth;
  scaled_dheight_ = image_scaling_ * kDefaultDepthHeight;

  rgb_map_.create(scaled_height_, scaled_width_, CV_8UC3);
  xyz_map_.create(scaled_height_, scaled_width_, CV_32FC3);
  raw_depth_map_.create(scaled_dheight_, scaled_dwidth_, CV_32FC1);

  device_->setColorFrameListener(listener_.get());
  device_->setIrAndDepthFrameListener(listener_.get());
  if (verbose) {
    std::cout << "Freenect device serial: " << device_->getSerialNumber()
              << std::endl;
    std::cout << "Freenect device firmware: " << device_->getFirmwareVersion()
              << std::endl;
  }

  if (!device_->start()) {
    pipeline_ = nullptr;
  }

  libfreenect2::Freenect2Device::ColorCameraParams rgb_cam_params =
      device_->getColorCameraParams();
  libfreenect2::Freenect2Device::IrCameraParams depth_cam_params =
      device_->getIrCameraParams();
  registration_ = std::make_unique<libfreenect2::Registration>(depth_cam_params,
                                                               rgb_cam_params);

  if (compute_xy_) {
    xy_table_cache_.resize(3, kDefaultRgbHeight * kDefaultRgbWidth);
    for (size_t i = 0; i < kDefaultRgbHeight; ++i) {
      for (size_t j = 0; j < kDefaultRgbWidth; ++j) {
        auto xyz = xy_table_cache_.col(i * kDefaultRgbWidth + j);
        xyz.x() = (j - rgb_cam_params.cx) / rgb_cam_params.fx * depth_scaling_;
        xyz.y() = (i - rgb_cam_params.cy) / rgb_cam_params.fy * depth_scaling_;
        xyz.z() = depth_scaling_;
      }
    }
  }

  fx_ = rgb_cam_params.fx;
  cx_ = rgb_cam_params.cx;
  fy_ = rgb_cam_params.fy;
  cy_ = rgb_cam_params.cy;

  dfx_ = depth_cam_params.fx;
  dcx_ = depth_cam_params.cx;
  dfy_ = depth_cam_params.fy;
  dcy_ = depth_cam_params.cy;

  fovx_ = 84.1;
  fovy_ = 53.8;
  dfovx_ = 70.6;
  dfovy_ = 60;
}

reclib::KinectV2FN2Camera::~KinectV2FN2Camera() {
  if (device_) {
    device_->stop();
    device_->close();
  }
}

auto reclib::KinectV2FN2Camera::GetModelName() const -> const std::string {
  return "Kinect V2 (Freenect2)";
}

auto reclib::KinectV2FN2Camera::GetWidth() const -> int {
  return scaled_width_;
}

auto reclib::KinectV2FN2Camera::GetHeight() const -> int {
  return scaled_height_;
}

auto reclib::KinectV2FN2Camera::GetDepthWidth() const -> int {
  return scaled_dwidth_;
}

auto reclib::KinectV2FN2Camera::GetDepthHeight() const -> int {
  return scaled_dheight_;
}

void reclib::KinectV2FN2Camera::DisableXYComputation() { compute_xy_ = false; }

void reclib::KinectV2FN2Camera::Update(
    CpuMat &xyz_map, CpuMat &rgb_map, CpuMat & /*ir_map*/, CpuMat & /*amp_map*/,
    CpuMat & /*flag_map*/, CpuMat &raw_depth_map, bool first_frame) {
  if (!listener_->waitForNewFrame(frames_, 10 * 1000)) {
    std::cout << "Freenect2 timeout!" << std::endl;
    bad_input_flag_ = true;
    return;
  }
  bad_input_flag_ = false;

  // depth comes in millimeters

  libfreenect2::Frame *rgb = frames_[libfreenect2::Frame::Color];
  libfreenect2::Frame *depth = frames_[libfreenect2::Frame::Depth];
  libfreenect2::Frame undistorted(depth->width, depth->height, 4);
  libfreenect2::Frame registered(depth->width, depth->height, 4);
  libfreenect2::Frame big_depth(rgb->width, rgb->height + 2, 4);
  std::vector<int> color_to_depth(depth->width * depth->height);
  registration_->apply(rgb, depth, &undistorted, &registered, true, &big_depth,
                       color_to_depth.data());

  const bool need_resize = image_scaling_ != 1.0F;
  const size_t input_size = rgb->height * rgb->width;
  const size_t depth_size = depth->height * depth->width;

  if (first_frame) {
    // check default resolution at first iteration
    _RECLIB_ASSERT_EQ(rgb->height, kDefaultRgbHeight);
    _RECLIB_ASSERT_EQ(rgb->width, kDefaultRgbWidth);
    _RECLIB_ASSERT_EQ(depth->height, kDefaultDepthHeight);
    _RECLIB_ASSERT_EQ(depth->width, kDefaultDepthWidth);
  }

  CpuMat rgb_tmp;
  CpuMat xyz_tmp;
  CpuMat raw_depth_tmp;

  if (need_resize) {
    rgb_tmp.create(rgb->height, rgb->width, CV_8UC3);
    xyz_tmp.create(rgb->height, rgb->width, CV_32FC3);
    raw_depth_tmp.create(depth->height, depth->width, CV_32FC1);
  }

  Eigen::Map<Eigen::Array<unsigned char, 3, Eigen::Dynamic> >(
      need_resize ? rgb_tmp.data : rgb_map.data, 3, input_size) =
      Eigen::Map<Eigen::Array<unsigned char, 4, Eigen::Dynamic> >(rgb->data, 4,
                                                                  input_size)
          .topRows<3>();

  // Depth
  Eigen::Map<Eigen::Array<float, 3, Eigen::Dynamic> > xyz_out(
      reinterpret_cast<float *>(need_resize ? xyz_tmp.data : xyz_map.data), 3,
      input_size);

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic> > depth_in(
      reinterpret_cast<float *>(depth->data), 1, depth_size);
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic> > depth_out(
      reinterpret_cast<float *>(need_resize ? raw_depth_tmp.data
                                            : raw_depth_map.data),
      1, depth_size);

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic> > big_depth_in(
      reinterpret_cast<float *>(big_depth.data) + rgb->width, 1, input_size);

  if (compute_xy_) {
    xyz_out = xy_table_cache_;
    xyz_out.rowwise() *= big_depth_in;
  } else {
    xyz_out.row(2) = big_depth_in * depth_scaling_;
  }

  depth_out = depth_in;
  depth_out *= depth_scaling_;

  for (unsigned int i = 0; i < input_size; ++i) {
    if (std::isinf(xyz_out(2, i))) {
      xyz_out.col(i).setZero();
    }
  }

  if (need_resize) {
    cv::resize(rgb_tmp, rgb_map, cv::Size(scaled_width_, scaled_height_));
    cv::resize(xyz_tmp, xyz_map, cv::Size(scaled_width_, scaled_height_));
    cv::resize(raw_depth_tmp, raw_depth_map,
               cv::Size(scaled_dwidth_, scaled_dheight_));
  }

  timestamp_ =
      static_cast<uint64_t>(std::max(rgb->timestamp, depth->timestamp)) *
      125000;

  listener_->release(frames_);
}
#endif  // HAS_FREENECT2

#if HAS_K4A

// color resolution 720P
const int reclib::KinectAzureCamera::kDefaultRgbWidth = 1280;
const int reclib::KinectAzureCamera::kDefaultRgbHeight = 720;
// depth in NFOV Unbinned mode
const int reclib::KinectAzureCamera::kDefaultDepthWidth = 640;
const int reclib::KinectAzureCamera::kDefaultDepthHeight = 576;

void reclib::KinectAzureCamera::InitCamera() {
  uint32_t device_count = k4a_device_get_installed_count();

  if (device_count == 0) {
    std::cerr << "No K4A devices found" << std::endl;
    device_open_flag_ = false;
    return;
  }

  device_ = k4a::device::open(K4A_DEVICE_DEFAULT);
  device_.start_cameras(&config_);
  device_.start_imu();
  calibration_ =
      device_.get_calibration(config_.depth_mode, config_.color_resolution);

  T_ = k4a::transformation(calibration_);

  std::cout << "Kinect Azure Opened. ID: " << device_.get_serialnum()
            << std::endl;
  std::cout << "   Color: "
            << calibration_.color_camera_calibration.resolution_width << " x "
            << calibration_.color_camera_calibration.resolution_height
            << std::endl;
  std::cout << "   Depth: "
            << calibration_.depth_camera_calibration.resolution_width << " x "
            << calibration_.depth_camera_calibration.resolution_height
            << std::endl;

  {
    extrinsics_rgbtod_.right_[0] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[0];
    extrinsics_rgbtod_.right_[1] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[3];
    extrinsics_rgbtod_.right_[2] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[6];
    extrinsics_rgbtod_.up_[0] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[1];
    extrinsics_rgbtod_.up_[1] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[4];
    extrinsics_rgbtod_.up_[2] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[7];
    extrinsics_rgbtod_.dir_[0] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[2];
    extrinsics_rgbtod_.dir_[1] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[5];
    extrinsics_rgbtod_.dir_[2] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .rotation[8];

    extrinsics_rgbtod_.eye_[0] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .translation[0];
    extrinsics_rgbtod_.eye_[1] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .translation[1];
    extrinsics_rgbtod_.eye_[2] =
        calibration_
            .extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH]
            .translation[2];
  }

  // Intrinsics from internal calibration
  {
    auto color = calibration_.color_camera_calibration;
    scaled_width_ = color.resolution_width;
    scaled_height_ = color.resolution_height;

    auto params = color.intrinsics.parameters.param;

    fx_ = params.fx;
    fy_ = params.fy;
    cx_ = params.cx;
    cy_ = params.cy;
    rad_distortion_.push_back(params.k1);
    rad_distortion_.push_back(params.k2);
    rad_distortion_.push_back(params.k3);
    rad_distortion_.push_back(params.k4);
    rad_distortion_.push_back(params.k5);
    rad_distortion_.push_back(params.k6);
    tang_distortion.push_back(params.p1);
    tang_distortion.push_back(params.p2);
  }
  {
    auto depth = calibration_.depth_camera_calibration;
    scaled_dwidth_ = depth.resolution_width;
    scaled_dheight_ = depth.resolution_height;

    auto params = depth.intrinsics.parameters.param;

    dfx_ = params.fx;
    dfy_ = params.fy;
    dcx_ = params.cx;
    dcy_ = params.cy;
    drad_distortion_.push_back(params.k1);
    drad_distortion_.push_back(params.k2);
    drad_distortion_.push_back(params.k3);
    drad_distortion_.push_back(params.k4);
    drad_distortion_.push_back(params.k5);
    drad_distortion_.push_back(params.k6);
    dtang_distortion.push_back(params.p1);
    dtang_distortion.push_back(params.p2);
  }

  switch (config_.depth_mode) {
    case (K4A_DEPTH_MODE_NFOV_UNBINNED): {
      dfovx_ = 75;
      dfovy_ = 65;
      break;
    }
    case (K4A_DEPTH_MODE_NFOV_2X2BINNED): {
      dfovx_ = 75;
      dfovy_ = 65;
      break;
    }
    case (K4A_DEPTH_MODE_WFOV_2X2BINNED): {
      dfovx_ = 120;
      dfovy_ = 120;
      break;
    }
    case (K4A_DEPTH_MODE_WFOV_UNBINNED): {
      dfovx_ = 120;
      dfovy_ = 120;
      break;
    }
    case (K4A_DEPTH_MODE_PASSIVE_IR): {
      dfovx_ = 0;
      dfovy_ = 0;
      break;
    }
    default: {
      dfovx_ = 0;
      dfovy_ = 0;
    }
  }

  switch (config_.color_resolution) {
    case (K4A_COLOR_RESOLUTION_3072P): {
      fovx_ = 90;
      fovy_ = 74.3;
      break;
    }
    case (K4A_COLOR_RESOLUTION_2160P): {
      fovx_ = 90;
      fovy_ = 59;
      break;
    }
    case (K4A_COLOR_RESOLUTION_1536P): {
      fovx_ = 90;
      fovy_ = 74.3;
      break;
    }
    case (K4A_COLOR_RESOLUTION_1440P): {
      fovx_ = 90;
      fovy_ = 59;
      break;
    }
    case (K4A_COLOR_RESOLUTION_1080P): {
      fovx_ = 90;
      fovy_ = 59;
      break;
    }
    case (K4A_COLOR_RESOLUTION_720P): {
      fovx_ = 90;
      fovy_ = 59;
      break;
    }
    default: {
      fovx_ = 0;
      fovy_ = 0;
    }
  }
}

reclib::KinectAzureCamera::KinectAzureCamera(std::string serial,
                                             float image_scaling,
                                             float depth_scaling, bool verbose,
                                             bool map_modalities_to_color)
    : KinectAzureCamera(serial, K4A_COLOR_RESOLUTION_720P,
                        K4A_DEPTH_MODE_NFOV_UNBINNED, image_scaling,
                        depth_scaling, verbose, map_modalities_to_color) {}

reclib::KinectAzureCamera::KinectAzureCamera(
    std::string serial, k4a_color_resolution_t color_resolution,
    k4a_depth_mode_t depth_mode, float image_scaling, float depth_scaling,
    bool verbose, bool map_modalities_to_color)
    : reclib::DepthCamera(depth_scaling, image_scaling),
      serial_(std::move(serial)),
      verbose_(verbose),
      map_modalities_to_color_(map_modalities_to_color) {
  has_rgb_map_ = true;
  has_raw_depth_map_ = true;
  has_xyz_map_ = false;

  config_.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  config_.depth_mode = depth_mode;
  config_.camera_fps = K4A_FRAMES_PER_SECOND_30;
  config_.color_resolution = color_resolution;
  // If set to true, k4a_capture_t objects will only be produced with
  // both color and depth images.
  config_.synchronized_images_only = true;

  scaled_width_ *= image_scaling_;
  scaled_height_ *= image_scaling_;
  scaled_dwidth_ *= image_scaling_;
  scaled_dheight_ *= image_scaling_;

  rgb_map_.create(scaled_height_, scaled_width_, CV_8UC3);
  raw_depth_map_.create(scaled_dheight_, scaled_dwidth_, CV_32FC1);
  // timer_.begin();

  InitCamera();
}

reclib::KinectAzureCamera::~KinectAzureCamera() {
  if (device_) {
    device_.close();
  }
}

auto reclib::KinectAzureCamera::GetModelName() const -> const std::string {
  return "Kinect V4 Azure (K4a)";
}

auto reclib::KinectAzureCamera::GetWidth() const -> int {
  return scaled_width_;
}

auto reclib::KinectAzureCamera::GetHeight() const -> int {
  return scaled_height_;
}

auto reclib::KinectAzureCamera::GetDepthWidth() const -> int {
  return scaled_dwidth_;
}

auto reclib::KinectAzureCamera::GetDepthHeight() const -> int {
  return scaled_dheight_;
}

// adapted from
// https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/master/src/transformation/intrinsic_transformation.c
vec3 unproject_undistort(const reclib::ExtendedIntrinsicParameters &params,
                         vec2 px, float d) {
  vec2 px_cam = px;
  px_cam.x() = (px_cam.x() - params.principal_x_) / params.focal_x_;
  px_cam.y() = (px_cam.y() - params.principal_y_) / params.focal_y_;

  float rs = px_cam.squaredNorm();
  float rss = rs * rs;
  float rsc = rss * rs;
  float a = 1.f + params.k_[0] * rs + params.k_[1] * rss + params.k_[2] * rsc;
  float b = 1.f + params.k_[3] * rs + params.k_[4] * rss + params.k_[5] * rsc;
  float ai;
  if (a != 0.f) {
    ai = 1.f / a;
  } else {
    ai = 1.f;
  }
  float di = ai * b;

  px_cam = px_cam * di;

  float two_xy = 2.f * px_cam[0] * px_cam[1];
  float xx = px_cam[0] * px_cam[0];
  float yy = px_cam[1] * px_cam[1];

  px_cam[0] -= (yy + 3.f * xx) * params.p_[1] + two_xy * params.p_[0];
  px_cam[1] -= (xx + 3.f * yy) * params.p_[0] + two_xy * params.p_[1];

  return vec3(px_cam.x() * d, px_cam.y() * d, d);
}

CpuMat reclib::KinectAzureCamera::Image2Mat(k4a::image &img, int channels) {
  const size_t height = img.get_height_pixels();
  const size_t width = img.get_width_pixels();
  switch (img.get_format()) {
    case (K4A_IMAGE_FORMAT_COLOR_BGRA32): {
      return CpuMat(height, width, CV_8UC4, img.get_buffer());
      break;
    }
    case (K4A_IMAGE_FORMAT_DEPTH16): {
      return CpuMat(height, width, CV_16UC1, img.get_buffer());
      break;
    }
    case (K4A_IMAGE_FORMAT_CUSTOM): {
      const size_t stride = img.get_stride_bytes();
      const size_t elem_size = stride / (channels * width);
      if (elem_size == 1) {
        return CpuMat(height, width, CV_8UC(channels), img.get_buffer());
      }
      if (elem_size == 2) {
        // CpuMat m(height, width, CV_16SC(channels));
        // m.setTo(0);
        // std::copy(img.get_buffer(),
        //           img.get_buffer() + height * width * sizeof(int16_t) * 3,
        //           m.data);
        // return m;

        return CpuMat(height, width, CV_16UC(channels), img.get_buffer());
      }
      if (elem_size == 4) {
        return CpuMat(height, width, CV_32FC(channels), img.get_buffer());
      } else {
        throw std::runtime_error("Unknown element size: " +
                                 std::to_string(elem_size));
        return cv::Mat(0, 0, CV_32FC1);
      }

      break;
    }
    case (K4A_IMAGE_FORMAT_CUSTOM8): {
      return CpuMat(height, width, CV_8UC(channels), img.get_buffer());
    }
    case (K4A_IMAGE_FORMAT_CUSTOM16): {
      return CpuMat(height, width, CV_16UC(channels), img.get_buffer());
    }
    default: {
      throw std::runtime_error("Unknown image type.");
      return cv::Mat(0, 0, CV_32FC1);
    }
  }
}

k4a::image reclib::KinectAzureCamera::Mat2Image(CpuMat &mat,
                                                bool use_depth_format) {
  const size_t height = mat.rows;
  const size_t width = mat.cols;

  int stride = mat.elemSize() * width;
  k4a_image_t image;
  switch (mat.type()) {
    case (CV_8UC1): {
      k4a_image_create_from_buffer(
          K4A_IMAGE_FORMAT_CUSTOM8, width, height, stride, mat.data,
          mat.elemSize() * width * height, nullptr, nullptr, &image);
      break;
    }
    case (CV_16UC1): {
      if (use_depth_format) {
        k4a_image_create_from_buffer(
            K4A_IMAGE_FORMAT_DEPTH16, width, height, stride, mat.data,
            mat.elemSize() * width * height, nullptr, nullptr, &image);
      } else {
        k4a_image_create_from_buffer(
            K4A_IMAGE_FORMAT_CUSTOM16, width, height, stride, mat.data,
            mat.elemSize() * width * height, nullptr, nullptr, &image);
      }

      break;
    }
    case (CV_8UC4): {
      k4a_image_create_from_buffer(
          K4A_IMAGE_FORMAT_COLOR_BGRA32, width, height, stride, mat.data,
          mat.elemSize() * width * height, nullptr, nullptr, &image);
      break;
    }
    default: {
      throw std::runtime_error("Unknown mat type: " +
                               std::to_string(mat.type()));
    }
  }
  return image;
}

cv::Mat reclib::KinectAzureCamera::custom2color(cv::Mat &custom, cv::Mat &depth,
                                                cv::Mat &color,
                                                k4a::calibration &calib) {
  k4a::transformation T(calib);

  cv::Mat tmp;
  k4a::image img_in;
  k4a::image depth_in;
  k4a::image img_out;
  if (depth.type() == CV_32FC1) {
    depth.convertTo(tmp, CV_16UC1);
    depth_in = reclib::KinectAzureCamera::Mat2Image(tmp);
  } else {
    depth_in = reclib::KinectAzureCamera::Mat2Image(depth);
  }
  if (custom.type() == CV_32FC1) {
    custom.convertTo(tmp, CV_16UC1);
    img_in = reclib::KinectAzureCamera::Mat2Image(tmp);
    img_out = k4a::image::create(K4A_IMAGE_FORMAT_CUSTOM16, color.cols,
                                 color.rows, color.cols * sizeof(uint16_t));
  } else {
    if (custom.depth() == CV_8U) {
      img_out = k4a::image::create(K4A_IMAGE_FORMAT_CUSTOM8, color.cols,
                                   color.rows, color.cols * sizeof(uint8_t));

    } else if (custom.depth() == CV_16U) {
      img_out = k4a::image::create(K4A_IMAGE_FORMAT_CUSTOM16, color.cols,
                                   color.rows, color.cols * sizeof(uint16_t));

    } else {
      throw std::runtime_error("Unknown mat type.");
    }
    img_in = reclib::KinectAzureCamera::Mat2Image(custom, false);
  }
  // k4a::image img_out = T.depth_image_to_color_camera(img_in);

  k4a::image transformed_depth =
      k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16, color.cols, color.rows,
                         color.cols * sizeof(uint16_t));
  T.depth_image_to_color_camera_custom(
      depth_in, img_in, &transformed_depth, &img_out,
      k4a_transformation_interpolation_type_t::
          K4A_TRANSFORMATION_INTERPOLATION_TYPE_LINEAR,
      0);

  cv::Mat out = reclib::KinectAzureCamera::Image2Mat(img_out);

  if (depth.type() == CV_32FC1) {
    out.convertTo(out, CV_32FC1);
  }

  return out;
}

cv::Mat reclib::KinectAzureCamera::depth2color(cv::Mat &depth, cv::Mat &color,
                                               k4a::calibration &calib) {
  k4a::transformation T(calib);

  cv::Mat tmp;
  k4a::image img_in;
  if (depth.type() == CV_32FC1) {
    depth.convertTo(tmp, CV_16UC1);
    img_in = reclib::KinectAzureCamera::Mat2Image(tmp);
  } else {
    img_in = reclib::KinectAzureCamera::Mat2Image(depth);
  }
  k4a::image img_out = T.depth_image_to_color_camera(img_in);

  cv::Mat out = reclib::KinectAzureCamera::Image2Mat(img_out);

  if (depth.type() == CV_32FC1) {
    out.convertTo(out, CV_32FC1);
  }

  return out;
}

cv::Mat reclib::KinectAzureCamera::depth2xyz(cv::Mat &depth,
                                             k4a::calibration &calib,
                                             k4a_calibration_type_t type) {
  k4a::transformation T(calib);

  cv::Mat tmp;
  k4a::image img_in;
  if (depth.type() == CV_32FC1) {
    depth.convertTo(tmp, CV_16UC1);
    img_in = reclib::KinectAzureCamera::Mat2Image(tmp);
  } else {
    img_in = reclib::KinectAzureCamera::Mat2Image(depth);
  }
  k4a::image img_out = T.depth_image_to_point_cloud(img_in, type);

  cv::Mat out = reclib::KinectAzureCamera::Image2Mat(img_out, 3);

  if (depth.type() == CV_32FC1) {
    out.convertTo(out, CV_32FC3);
  }

  return out;
}

cv::Mat reclib::KinectAzureCamera::color2depth(cv::Mat &depth, cv::Mat &color,
                                               k4a::calibration &calib) {
  k4a::transformation T(calib);
  k4a::image img_in_c;
  k4a::image img_in_d;

  cv::Mat tmp_d;
  cv::Mat tmp_c;
  if (depth.type() == CV_32FC1) {
    depth.convertTo(tmp_d, CV_16UC1);
    img_in_d = reclib::KinectAzureCamera::Mat2Image(tmp_d);
  } else {
    img_in_d = reclib::KinectAzureCamera::Mat2Image(depth);
  }

  if (color.type() == CV_32FC3) {
    color.convertTo(tmp_c, CV_8UC3);
    cv::cvtColor(tmp_c, tmp_c, cv::COLOR_RGB2RGBA);
    img_in_c = reclib::KinectAzureCamera::Mat2Image(tmp_c);
  } else if (color.type() == CV_8UC3) {
    tmp_c = color;
    cv::cvtColor(tmp_c, tmp_c, cv::COLOR_RGB2RGBA);
    img_in_c = reclib::KinectAzureCamera::Mat2Image(tmp_c);
  } else {
    img_in_c = reclib::KinectAzureCamera::Mat2Image(color);
  }

  k4a::image img_out = T.color_image_to_depth_camera(img_in_d, img_in_c);
  cv::Mat out = reclib::KinectAzureCamera::Image2Mat(img_out);

  if (color.type() == CV_32FC3) {
    cv::cvtColor(out, out, cv::COLOR_RGBA2RGB);
    out.convertTo(out, CV_32FC3);
  } else if (color.type() == CV_8UC3) {
    cv::cvtColor(out, out, cv::COLOR_RGBA2RGB);
  }
  return out;
}
vec3 reclib::KinectAzureCamera::depth3Dto2D(const vec3 &point,
                                            k4a::calibration &calib) {
  k4a_float3_t in;
  k4a_float2_t out;
  vec3 v_out = vec3::Ones() * -1;
  bool valid;
  for (unsigned int i = 0; i < 3; i++) {
    in.v[i] = point[i];
  }

  valid = calib.convert_3d_to_2d(in, K4A_CALIBRATION_TYPE_DEPTH,
                                 K4A_CALIBRATION_TYPE_DEPTH, &out);

  if (valid) {
    for (unsigned int i = 0; i < 2; i++) {
      v_out[i] = out.v[i];
    }
    v_out.z() = point.z();
  }

  return v_out;
}
vec3 reclib::KinectAzureCamera::depth2Dto3D(const vec3 &px_d,
                                            k4a::calibration &calib) {
  k4a_float2_t in;
  k4a_float3_t out;
  vec3 v_out(std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max());
  bool valid;
  for (unsigned int i = 0; i < 2; i++) {
    in.v[i] = px_d[i];
  }

  valid = calib.convert_2d_to_3d(in, px_d.z(), K4A_CALIBRATION_TYPE_DEPTH,
                                 K4A_CALIBRATION_TYPE_DEPTH, &out);

  if (valid) {
    for (unsigned int i = 0; i < 3; i++) {
      v_out[i] = out.v[i];
    }
  }

  return v_out;
}
vec3 reclib::KinectAzureCamera::depth2color3D(const vec3 &point_depth,
                                              k4a::calibration &calib) {
  k4a_float3_t in;
  k4a_float3_t out;
  vec3 v_out = vec3::Zero();
  for (unsigned int i = 0; i < 3; i++) {
    in.v[i] = point_depth[i];
  }

  out = calib.convert_3d_to_3d(in, K4A_CALIBRATION_TYPE_DEPTH,
                               K4A_CALIBRATION_TYPE_COLOR);

  for (unsigned int i = 0; i < 3; i++) {
    v_out[i] = out.v[i];
  }

  return v_out;
}
vec3 reclib::KinectAzureCamera::color2depth3D(const vec3 &point_color,
                                              k4a::calibration &calib) {
  k4a_float3_t in;
  k4a_float3_t out;
  vec3 v_out = vec3::Zero();
  for (unsigned int i = 0; i < 3; i++) {
    in.v[i] = point_color[i];
  }

  out = calib.convert_3d_to_3d(in, K4A_CALIBRATION_TYPE_COLOR,
                               K4A_CALIBRATION_TYPE_DEPTH);

  for (unsigned int i = 0; i < 3; i++) {
    v_out[i] = out.v[i];
  }

  return v_out;
}

vec3 reclib::KinectAzureCamera::color3Dto2D(const vec3 &point,
                                            k4a::calibration &calib) {
  k4a_float3_t in;
  k4a_float2_t out;
  vec3 v_out = vec3::Ones() * -1;
  bool valid;
  for (unsigned int i = 0; i < 3; i++) {
    in.v[i] = point[i];
  }

  valid = calib.convert_3d_to_2d(in, K4A_CALIBRATION_TYPE_COLOR,
                                 K4A_CALIBRATION_TYPE_COLOR, &out);

  if (valid) {
    for (unsigned int i = 0; i < 2; i++) {
      v_out[i] = out.v[i];
    }
    v_out.z() = point.z();
  }

  return v_out;
}

vec3 reclib::KinectAzureCamera::color2Dto3D(const vec3 &px_d,
                                            k4a::calibration &calib) {
  k4a_float2_t in;
  k4a_float3_t out;
  vec3 v_out(std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max());
  bool valid;
  for (unsigned int i = 0; i < 2; i++) {
    in.v[i] = px_d[i];
  }

  valid = calib.convert_2d_to_3d(in, px_d.z(), K4A_CALIBRATION_TYPE_COLOR,
                                 K4A_CALIBRATION_TYPE_COLOR, &out);

  if (valid) {
    for (unsigned int i = 0; i < 3; i++) {
      v_out[i] = out.v[i];
    }
  }

  return v_out;
}

mat4 reclib::KinectAzureCamera::calib2extrinsic(
    k4a_calibration_extrinsics_t &extr) {
  mat4 m;
  m << extr.rotation[0], extr.rotation[1], extr.rotation[2],
      extr.translation[0], extr.rotation[3], extr.rotation[4], extr.rotation[5],
      extr.translation[1], extr.rotation[6], extr.rotation[7], extr.rotation[8],
      extr.translation[2], 0, 0, 0, 1;
  return m;
}
mat4 reclib::KinectAzureCamera::calib2intrinsic(
    k4a_calibration_intrinsics_t &intr) {
  mat4 m;
  m << intr.parameters.param.fx, 0, intr.parameters.param.cx, 0, 0,
      intr.parameters.param.fy, intr.parameters.param.cy, 0, 0, 0, 1, 0, 0, 0,
      0, 1;
  return m;
}

void reclib::KinectAzureCamera::Update(CpuMat &xyz_map, CpuMat &rgb_map,
                                       CpuMat &ir_map, CpuMat & /*amp_map*/,
                                       CpuMat & /*flag_map*/,
                                       CpuMat &raw_depth_map,
                                       bool first_frame) {
  // std::cout << "Azure: time since Update: " << timer_.look() << " ms."
  //           << std::endl;

  k4a::capture capture;
  if (!device_.get_capture(&capture, std::chrono::milliseconds(1000))) {
    std::cout << "Capture Timeout" << std::endl;
    bad_input_flag_ = true;
    return;
  }

  bad_input_flag_ = false;
  const bool need_resize = image_scaling_ != 1.0F;

  // depth comes in millimeters
  k4a::image k4a_color;
  k4a::image k4a_depth;
  k4a::image k4a_ir;

  if (HasRGBMap()) {
    k4a_color = capture.get_color_image();
  }

  if (HasRawDepthMap()) {
    k4a_depth = capture.get_depth_image();
  }

  if (HasIRMap()) {
    if (!HasRGBMap()) {
      k4a_color = capture.get_color_image();
    }
    if (!HasRawDepthMap()) {
      k4a_depth = capture.get_depth_image();
    }

    k4a_ir = capture.get_ir_image();

    k4a::image tmp = k4a::image::create(
        K4A_IMAGE_FORMAT_DEPTH16, k4a_color.get_width_pixels(),
        k4a_color.get_height_pixels(),
        k4a_color.get_width_pixels() * sizeof(uint16_t));

    k4a::image ir_input = k4a::image::create_from_buffer(
        K4A_IMAGE_FORMAT_CUSTOM16, k4a_ir.get_width_pixels(),
        k4a_ir.get_height_pixels(), k4a_ir.get_stride_bytes(),
        k4a_ir.get_buffer(),
        k4a_ir.get_width_pixels() * k4a_ir.get_height_pixels() *
            sizeof(uint16_t),
        nullptr, nullptr);

    k4a::image mapped_ir = k4a::image::create(
        K4A_IMAGE_FORMAT_CUSTOM16, k4a_color.get_width_pixels(),
        k4a_color.get_height_pixels(),
        k4a_color.get_width_pixels() * sizeof(uint16_t));

    if (map_modalities_to_color_) {
      T_.depth_image_to_color_camera_custom(
          k4a_depth, ir_input, &tmp, &mapped_ir,
          k4a_transformation_interpolation_type_t::
              K4A_TRANSFORMATION_INTERPOLATION_TYPE_LINEAR,
          0);
    }
    k4a_ir = mapped_ir;
  }

  if (map_modalities_to_color_) {
    k4a_depth = T_.depth_image_to_color_camera(k4a_depth);
  }

  if (HasRGBMap()) {
    const size_t color_height = k4a_color.get_height_pixels();
    const size_t color_width = k4a_color.get_width_pixels();
    const size_t input_size = color_width * color_height;
    const size_t color_stride =
        k4a_color.get_stride_bytes() / (sizeof(uint8_t));
    _RECLIB_ASSERT_EQ(k4a_color.get_format(), K4A_IMAGE_FORMAT_COLOR_BGRA32);

    if (first_frame) {
      // check default resolution at first iteration
      _RECLIB_ASSERT_EQ(color_height, scaled_height_ / image_scaling_);
      _RECLIB_ASSERT_EQ(color_width, scaled_width_ / image_scaling_);
    }

    CpuMat rgb_tmp;

    CpuMat rgba8(color_height, color_width, CV_8UC4);
    std::copy(k4a_color.get_buffer(),
              k4a_color.get_buffer() +
                  color_height * color_width * sizeof(uint8_t) * 4,
              rgba8.data);

    if (need_resize) {
      rgb_tmp.create(color_height, color_width, CV_8UC3);
    }
    cv::cvtColor(rgba8, need_resize ? rgb_tmp : rgb_map, cv::COLOR_BGRA2BGR);
    if (need_resize) {
      cv::resize(rgb_tmp, rgb_map, cv::Size(scaled_width_, scaled_height_));
    }
  }

  if (HasRawDepthMap()) {
    const size_t depth_height = k4a_depth.get_height_pixels();
    const size_t depth_width = k4a_depth.get_width_pixels();
    const size_t depth_size = depth_width * depth_height;
    const size_t depth_stride = k4a_depth.get_stride_bytes() / sizeof(uint16_t);
    _RECLIB_ASSERT_EQ(k4a_depth.get_format(), K4A_IMAGE_FORMAT_DEPTH16);

    if (first_frame) {
      if (map_modalities_to_color_) {
        _RECLIB_ASSERT_EQ(depth_height, scaled_height_ / image_scaling_);
        _RECLIB_ASSERT_EQ(depth_width, scaled_width_ / image_scaling_);
      } else {
        _RECLIB_ASSERT_EQ(depth_height, scaled_dheight_ / image_scaling_);
        _RECLIB_ASSERT_EQ(depth_width, scaled_dwidth_ / image_scaling_);
      }
    }

    CpuMat raw_depth_tmp;
    CpuMat depth16(depth_height, depth_width, CV_16UC1);
    std::copy(
        k4a_depth.get_buffer(),
        k4a_depth.get_buffer() + depth_height * depth_width * sizeof(uint16_t),
        depth16.data);

    if (need_resize) {
      raw_depth_tmp.create(depth_height, depth_width, CV_32FC1);
    }
    depth16.convertTo(need_resize ? raw_depth_tmp : raw_depth_map, CV_32FC1);
    if (need_resize) {
      cv::resize(raw_depth_tmp, raw_depth_map,
                 cv::Size(scaled_dwidth_, scaled_dheight_));
    }
    raw_depth_map *= depth_scaling_;
  }

  if (HasIRMap()) {
    const size_t height = k4a_ir.get_height_pixels();
    const size_t width = k4a_ir.get_width_pixels();
    const size_t size = width * height;
    const size_t stride = k4a_ir.get_stride_bytes() / sizeof(uint16_t);
    _RECLIB_ASSERT(k4a_ir.get_format() == K4A_IMAGE_FORMAT_IR16 ||
                   k4a_ir.get_format() == K4A_IMAGE_FORMAT_CUSTOM16);

    if (first_frame) {
      if (map_modalities_to_color_) {
        _RECLIB_ASSERT_EQ(height, scaled_height_ / image_scaling_);
        _RECLIB_ASSERT_EQ(width, scaled_width_ / image_scaling_);
      } else {
        _RECLIB_ASSERT_EQ(height, scaled_dheight_ / image_scaling_);
        _RECLIB_ASSERT_EQ(width, scaled_dwidth_ / image_scaling_);
      }
    }

    CpuMat raw_ir_tmp;
    CpuMat ir16(height, width, CV_16UC1);
    std::copy(k4a_ir.get_buffer(),
              k4a_ir.get_buffer() + height * width * sizeof(uint16_t),
              ir16.data);

    if (need_resize) {
      raw_ir_tmp.create(height, width, CV_32FC1);
    }
    ir16.convertTo(need_resize ? raw_ir_tmp : ir_map, CV_32FC1);
    if (need_resize) {
      cv::resize(raw_ir_tmp, ir_map, cv::Size(scaled_dwidth_, scaled_dheight_));
    }
  }

  timestamp_ = static_cast<uint64_t>(
                   std::max(k4a_color.get_device_timestamp().count(),
                            k4a_depth.get_device_timestamp().count())) /
               (1000.0 * 1000.0);
}

void WriteAzureIntrinsics(cv::FileStorage &fs,
                          const k4a_calibration_intrinsics_t &intrinsics,
                          const std::string &prefix) {
  fs << prefix + "-type" << intrinsics.type;

  fs << prefix + "-parameter_count" << (int)intrinsics.parameter_count;

  fs << prefix + "-parameters-param-cx" << intrinsics.parameters.param.cx;

  fs << prefix + "-parameters-param-cy" << intrinsics.parameters.param.cy;

  fs << prefix + "-parameters-param-fx" << intrinsics.parameters.param.fx;

  fs << prefix + "-parameters-param-fy" << intrinsics.parameters.param.fy;

  fs << prefix + "-parameters-param-k1" << intrinsics.parameters.param.k1;

  fs << prefix + "-parameters-param-k2" << intrinsics.parameters.param.k2;

  fs << prefix + "-parameters-param-k3" << intrinsics.parameters.param.k3;

  fs << prefix + "-parameters-param-k4" << intrinsics.parameters.param.k4;

  fs << prefix + "-parameters-param-k5" << intrinsics.parameters.param.k5;

  fs << prefix + "-parameters-param-k6" << intrinsics.parameters.param.k6;

  fs << prefix + "-parameters-param-codx" << intrinsics.parameters.param.codx;

  fs << prefix + "-parameters-param-cody" << intrinsics.parameters.param.cody;

  fs << prefix + "-parameters-param-p2" << intrinsics.parameters.param.p2;

  fs << prefix + "-parameters-param-p1" << intrinsics.parameters.param.p1;

  fs << prefix + "-parameters-param-metric_radius"
     << intrinsics.parameters.param.metric_radius;
}

void WriteAzureExtrinsics(cv::FileStorage &fs,
                          const k4a_calibration_extrinsics_t &extrinsics,
                          const std::string &prefix) {
  fs << prefix + "-rotation0" << extrinsics.rotation[0];
  fs << prefix + "-rotation1" << extrinsics.rotation[1];
  fs << prefix + "-rotation2" << extrinsics.rotation[2];
  fs << prefix + "-rotation3" << extrinsics.rotation[3];
  fs << prefix + "-rotation4" << extrinsics.rotation[4];
  fs << prefix + "-rotation5" << extrinsics.rotation[5];
  fs << prefix + "-rotation6" << extrinsics.rotation[6];
  fs << prefix + "-rotation7" << extrinsics.rotation[7];
  fs << prefix + "-rotation8" << extrinsics.rotation[8];

  fs << prefix + "-translation0" << extrinsics.translation[0];
  fs << prefix + "-translation1" << extrinsics.translation[1];
  fs << prefix + "-translation2" << extrinsics.translation[2];
}

bool reclib::KinectAzureCamera::WriteIntrinsics(const fs::path &destination,
                                                fs::path filename) const {
  if (filename.string().length() == 0) {
    filename = fs::path("cam_intrinsics.yaml");
  }

  reclib::utils::CreateDirectories(destination / filename);

  cv::FileStorage fs((destination / filename).string(), cv::FileStorage::WRITE);

  ExtendedIntrinsicParameters intrinsics = GetExtIntrinsics();
  ExtendedIntrinsicParameters depth_intrinsics = GetExtDepthIntrinsics();

  fs << "camera_model" << GetModelName();

  fs << "rgb-image_width" << intrinsics.image_width_;
  fs << "rgb-image_height" << intrinsics.image_height_;
  fs << "rgb-focal_x" << intrinsics.focal_x_;
  fs << "rgb-focal_y" << intrinsics.focal_y_;
  fs << "rgb-principal_x" << intrinsics.principal_x_;
  fs << "rgb-principal_y" << intrinsics.principal_y_;
  fs << "rgb-fovx" << fovx_;
  fs << "rgb-fovy" << fovy_;

  std::string distortion = "rgb-k";
  for (unsigned int i = 0; i < intrinsics.k_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << intrinsics.k_[i];
  }
  distortion = "rgb-p";
  for (unsigned int i = 0; i < intrinsics.p_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << intrinsics.p_[i];
  }

  fs << "depth-image_width" << depth_intrinsics.image_width_;
  fs << "depth-image_height" << depth_intrinsics.image_height_;
  fs << "depth-focal_x" << depth_intrinsics.focal_x_;
  fs << "depth-focal_y" << depth_intrinsics.focal_y_;
  fs << "depth-principal_x" << depth_intrinsics.principal_x_;
  fs << "depth-principal_y" << depth_intrinsics.principal_y_;
  fs << "depth-fovx" << dfovx_;
  fs << "depth-fovy" << dfovy_;

  distortion = "depth-k";
  for (unsigned int i = 0; i < depth_intrinsics.k_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << depth_intrinsics.k_[i];
  }
  distortion = "depth-p";
  for (unsigned int i = 0; i < depth_intrinsics.p_.size(); i++) {
    std::string tmp = distortion + "_" + std::to_string(i);
    fs << tmp << depth_intrinsics.p_[i];
  }

  fs << "k4a-depth-mode" << config_.depth_mode;
  fs << "k4a-color-resolution" << config_.color_resolution;
  fs << "k4a-color-format" << config_.color_format;

  // WriteAzureExtrinsics(fs,
  // calibration_.depth_camera_calibration.extrinsics,
  //                      "depth_camera_calibration-extrinsics");
  // WriteAzureIntrinsics(fs,
  // calibration_.depth_camera_calibration.intrinsics,
  //                      "depth_camera_calibration-intrinsics");

  // fs << "depth_camera_calibration-resolution_width"
  //    << calibration_.depth_camera_calibration.resolution_width;

  // fs << "depth_camera_calibration-resolution_height"
  //    << calibration_.depth_camera_calibration.resolution_height;

  // fs << "depth_camera_calibration-metric_radius"
  //    << calibration_.depth_camera_calibration.metric_radius;

  // WriteAzureExtrinsics(fs,
  // calibration_.color_camera_calibration.extrinsics,
  //                      "color_camera_calibration-extrinsics");
  // WriteAzureIntrinsics(fs,
  // calibration_.color_camera_calibration.intrinsics,
  //                      "color_camera_calibration-intrinsics");

  // fs << "color_camera_calibration-resolution_width"
  //    << calibration_.color_camera_calibration.resolution_width;

  // fs << "color_camera_calibration-resolution_height"
  //    << calibration_.color_camera_calibration.resolution_height;

  // fs << "color_camera_calibration-metric_radius"
  //    << calibration_.color_camera_calibration.metric_radius;

  // for (unsigned int i = 0; i < K4A_CALIBRATION_TYPE_NUM; i++) {
  //   for (unsigned int j = 0; j < K4A_CALIBRATION_TYPE_NUM; j++) {
  //     WriteAzureExtrinsics(
  //         fs, calibration_.extrinsics[i][j],
  //         "extrinsics_" + std::to_string(i) + "_" + std::to_string(j));
  //   }
  // }

  // fs << "depth_mode" << calibration_.depth_mode;
  // fs << "color_resolution" << calibration_.color_resolution;

  fs.release();

  size_t calibration_size;

  k4a_device_get_raw_calibration(device_.handle(), nullptr, &calibration_size);
  std::vector<uint8_t> calib_data(calibration_size);
  k4a_device_get_raw_calibration(device_.handle(), calib_data.data(),
                                 &calibration_size);
  std::ofstream o;  // ofstream is the class for fstream package
  o.open(destination / "calibration.json");  // open is the method of ofstream
  for (unsigned int i = 0; i < calib_data.size(); i++) {
    o << calib_data[i];
  }
  o.close();

  return true;
}

void ReadAzureIntrinsics(cv::FileStorage &fs,
                         k4a_calibration_intrinsics_t &intrinsics,
                         const std::string &prefix) {
  fs[prefix + "-type"] >> intrinsics.type;

  int parameter_count;
  fs[prefix + "-parameter_count"] >> parameter_count;
  intrinsics.parameter_count = parameter_count;

  fs[prefix + "-parameters-param-cx"] >> intrinsics.parameters.param.cx;

  fs[prefix + "-parameters-param-cy"] >> intrinsics.parameters.param.cy;

  fs[prefix + "-parameters-param-fx"] >> intrinsics.parameters.param.fx;

  fs[prefix + "-parameters-param-fy"] >> intrinsics.parameters.param.fy;

  fs[prefix + "-parameters-param-k1"] >> intrinsics.parameters.param.k1;

  fs[prefix + "-parameters-param-k2"] >> intrinsics.parameters.param.k2;

  fs[prefix + "-parameters-param-k3"] >> intrinsics.parameters.param.k3;

  fs[prefix + "-parameters-param-k4"] >> intrinsics.parameters.param.k4;

  fs[prefix + "-parameters-param-k5"] >> intrinsics.parameters.param.k5;

  fs[prefix + "-parameters-param-k6"] >> intrinsics.parameters.param.k6;

  fs[prefix + "-parameters-param-codx"] >> intrinsics.parameters.param.codx;

  fs[prefix + "-parameters-param-cody"] >> intrinsics.parameters.param.cody;

  fs[prefix + "-parameters-param-p2"] >> intrinsics.parameters.param.p2;

  fs[prefix + "-parameters-param-p1"] >> intrinsics.parameters.param.p1;

  fs[prefix + "-parameters-param-metric_radius"] >>
      intrinsics.parameters.param.metric_radius;
}

void ReadAzureExtrinsics(cv::FileStorage &fs,
                         k4a_calibration_extrinsics_t &extrinsics,
                         const std::string &prefix) {
  fs[prefix + "-rotation0"] >> extrinsics.rotation[0];
  fs[prefix + "-rotation1"] >> extrinsics.rotation[1];
  fs[prefix + "-rotation2"] >> extrinsics.rotation[2];
  fs[prefix + "-rotation3"] >> extrinsics.rotation[3];
  fs[prefix + "-rotation4"] >> extrinsics.rotation[4];
  fs[prefix + "-rotation5"] >> extrinsics.rotation[5];
  fs[prefix + "-rotation6"] >> extrinsics.rotation[6];
  fs[prefix + "-rotation7"] >> extrinsics.rotation[7];
  fs[prefix + "-rotation8"] >> extrinsics.rotation[8];

  fs[prefix + "-translation0"] >> extrinsics.translation[0];
  fs[prefix + "-translation1"] >> extrinsics.translation[1];
  fs[prefix + "-translation2"] >> extrinsics.translation[2];
}

k4a::calibration reclib::KinectAzureCamera::CalibrationFromFile(
    const fs::path &destination, fs::path filename) {
  if (filename.string().length() == 0) {
    filename = fs::path("calibration.json");
  }
  std::ifstream in(destination / filename);
  std::vector<char> calib_data;
  while (!in.eof()) {
    char d;
    in >> d;
    calib_data.push_back(d);
  }
  calib_data.push_back('\0');
  in.close();

  k4a_image_format_t format;
  k4a_color_resolution_t resolution;
  k4a_depth_mode_t depth_mode;

  if (fs::exists(destination / "cam_intrinsics.yaml")) {
    cv::FileStorage fs(destination / "cam_intrinsics.yaml",
                       cv::FileStorage::READ);
    fs["k4a-depth-mode"] >> depth_mode;
    fs["k4a-color-resolution"] >> resolution;
    fs["k4a-color-format"] >> format;
    fs.release();
  } else {
    _RECLIB_ASSERT(fs::exists(destination / "tags.xml"));
    std::ifstream str((destination / "tags.xml"));
    std::string line;
    while (std::getline(str, line)) {
      if (line.find("K4A_DEPTH_MODE") != std::string::npos) {
        std::getline(str, line);
        int start_pos = line.find("<String>") + 8;
        int end_pos = line.find("</String>");
        int len = end_pos - start_pos;
        std::string mode = line.substr(start_pos, len);

        if (mode.compare("OFF") == 0) {
          depth_mode = k4a_depth_mode_t::K4A_DEPTH_MODE_OFF;
        } else if (mode.compare("NFOV_2X2BINNED") == 0) {
          depth_mode = k4a_depth_mode_t::K4A_DEPTH_MODE_NFOV_2X2BINNED;
        } else if (mode.compare("NFOV_UNBINNED") == 0) {
          depth_mode = k4a_depth_mode_t::K4A_DEPTH_MODE_NFOV_UNBINNED;
        } else if (mode.compare("WFOV_2X2BINNED") == 0) {
          depth_mode = k4a_depth_mode_t::K4A_DEPTH_MODE_WFOV_2X2BINNED;
        } else if (mode.compare("WFOV_UNBINNED ") == 0) {
          depth_mode = k4a_depth_mode_t::K4A_DEPTH_MODE_WFOV_UNBINNED;
        } else if (mode.compare("PASSIVE_IR") == 0) {
          depth_mode = k4a_depth_mode_t::K4A_DEPTH_MODE_PASSIVE_IR;
        } else {
          throw std::runtime_error("Unknown depth mode.");
        }
      }
      if (line.find("K4A_COLOR_MODE") != std::string::npos) {
        std::getline(str, line);
        int start_pos = line.find("<String>") + 8;
        int end_pos = line.find("</String>");
        int len = end_pos - start_pos;
        int split = line.substr(start_pos, len).find("_") + start_pos;
        std::string image_format = line.substr(start_pos, split - start_pos);
        std::string color_resolution =
            line.substr(split + 1, end_pos - split - 1);

        if (image_format.compare("MJPG") == 0) {
          format = k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_MJPG;
        } else if (image_format.compare("NV12") == 0) {
          format = k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_NV12;
        } else if (image_format.compare("YUY2") == 0) {
          format = k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_YUY2;
        } else if (image_format.compare("BGRA32") == 0) {
          format = k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_BGRA32;
        } else if (image_format.compare("DEPTH16") == 0) {
          format = k4a_image_format_t::K4A_IMAGE_FORMAT_DEPTH16;
        } else if (image_format.compare("IR16") == 0) {
          format = k4a_image_format_t::K4A_IMAGE_FORMAT_IR16;
        } else {
          throw std::runtime_error("Unknown image format mode.");
        }

        if (color_resolution.compare("OFF") == 0) {
          resolution = k4a_color_resolution_t::K4A_COLOR_RESOLUTION_OFF;
        } else if (color_resolution.compare("720P") == 0) {
          resolution = k4a_color_resolution_t::K4A_COLOR_RESOLUTION_720P;
        } else if (color_resolution.compare("1080P") == 0) {
          resolution = k4a_color_resolution_t::K4A_COLOR_RESOLUTION_1080P;
        } else if (color_resolution.compare("1440P") == 0) {
          resolution = k4a_color_resolution_t::K4A_COLOR_RESOLUTION_1440P;
        } else if (color_resolution.compare("1536P") == 0) {
          resolution = k4a_color_resolution_t::K4A_COLOR_RESOLUTION_1536P;
        } else if (color_resolution.compare("2160P") == 0) {
          resolution = k4a_color_resolution_t::K4A_COLOR_RESOLUTION_2160P;
        } else if (color_resolution.compare("3072P") == 0) {
          resolution = k4a_color_resolution_t::K4A_COLOR_RESOLUTION_3072P;
        } else {
          throw std::runtime_error("Unknown color resolution mode.");
        }
      }
    }
    str.close();
  }

  k4a::calibration calib = k4a::calibration::get_from_raw(
      (char *)calib_data.data(), calib_data.size(), depth_mode, resolution);

  // ReadAzureExtrinsics(fs, calib.depth_camera_calibration.extrinsics,
  //                     "depth_camera_calibration-extrinsics");
  // ReadAzureIntrinsics(fs, calib.depth_camera_calibration.intrinsics,
  //                     "depth_camera_calibration-intrinsics");

  // fs["depth_camera_calibration-resolution_width"] >>
  //     calib.depth_camera_calibration.resolution_width;

  // fs["depth_camera_calibration-resolution_height"] >>
  //     calib.depth_camera_calibration.resolution_height;

  // fs["depth_camera_calibration-metric_radius"] >>
  //     calib.depth_camera_calibration.metric_radius;

  // ReadAzureExtrinsics(fs, calib.color_camera_calibration.extrinsics,
  //                     "color_camera_calibration-extrinsics");
  // ReadAzureIntrinsics(fs, calib.color_camera_calibration.intrinsics,
  //                     "color_camera_calibration-intrinsics");

  // fs["color_camera_calibration-resolution_width"] >>
  //     calib.color_camera_calibration.resolution_width;

  // fs["color_camera_calibration-resolution_height"] >>
  //     calib.color_camera_calibration.resolution_height;

  // fs["color_camera_calibration-metric_radius"] >>
  //     calib.color_camera_calibration.metric_radius;

  // for (unsigned int i = 0; i < K4A_CALIBRATION_TYPE_NUM; i++) {
  //   for (unsigned int j = 0; j < K4A_CALIBRATION_TYPE_NUM; j++) {
  //     ReadAzureExtrinsics(
  //         fs, calib.extrinsics[i][j],
  //         "extrinsics_" + std::to_string(i) + "_" + std::to_string(j));
  //   }
  // }

  // fs["depth_mode"] >> calib.depth_mode;
  // fs["color_resolution"] >> calib.color_resolution;

  // fs.release();

  return calib;
}

// ---------------------------------------------------------
// Code adapted from:
// https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/examples/undistort/main.cpp
// --------------------------------------------------------

static void create_undistortion_lut(const k4a_calibration_t *calibration,
                                    const k4a_calibration_type_t camera,
                                    const reclib::pinhole_t *pinhole,
                                    k4a_image_t lut,
                                    reclib::interpolation_t type) {
  reclib::coordinate_t *lut_data =
      (reclib::coordinate_t *)(void *)k4a_image_get_buffer(lut);

  k4a_float3_t ray;
  ray.xyz.z = 1.f;

  int src_width = calibration->depth_camera_calibration.resolution_width;
  int src_height = calibration->depth_camera_calibration.resolution_height;
  if (camera == K4A_CALIBRATION_TYPE_COLOR) {
    src_width = calibration->color_camera_calibration.resolution_width;
    src_height = calibration->color_camera_calibration.resolution_height;
  }

  for (int y = 0; y < pinhole->height; y++) {
    ray.xyz.y = ((float)y - pinhole->py) / pinhole->fy;

    for (int x = 0; x < pinhole->width; x++) {
      ray.xyz.x = ((float)x - pinhole->px) / pinhole->fx;

      k4a_float2_t distorted;

      int valid;
      k4a_calibration_3d_to_2d(calibration, &ray, camera, camera, &distorted,
                               &valid);

      reclib::coordinate_t src;
      reclib::coordinate_t dest;
      dest.x = x;
      dest.y = y;
      if (type == reclib::INTERPOLATION_NEARESTNEIGHBOR) {
        // Remapping via nearest neighbor interpolation
        src.x = (int)floorf(distorted.xy.x + 0.5f);
        src.y = (int)floorf(distorted.xy.y + 0.5f);
      } else if (type == reclib::INTERPOLATION_BILINEAR ||
                 type == reclib::INTERPOLATION_BILINEAR_DEPTH) {
        // Remapping via bilinear interpolation
        src.x = (int)floorf(distorted.xy.x);
        src.y = (int)floorf(distorted.xy.y);
      } else {
        printf("Unexpected interpolation type!\n");
        exit(-1);
      }

      int idx = src.y * pinhole->width + src.x;

      if (valid && src.x >= 0 && src.x < src_width && src.y >= 0 &&
          src.y < src_height) {
        lut_data[idx] = dest;

        if (type == reclib::INTERPOLATION_BILINEAR ||
            type == reclib::INTERPOLATION_BILINEAR_DEPTH) {
          // Compute the floating point weights, using the distance from
          // projected point src to the image coordinate of the upper left
          // neighbor
          float w_x = distorted.xy.x - src.x;
          float w_y = distorted.xy.y - src.y;
          float w0 = (1.f - w_x) * (1.f - w_y);
          float w1 = w_x * (1.f - w_y);
          float w2 = (1.f - w_x) * w_y;
          float w3 = w_x * w_y;

          // Fill into lut
          lut_data[idx].weight[0] = w0;
          lut_data[idx].weight[1] = w1;
          lut_data[idx].weight[2] = w2;
          lut_data[idx].weight[3] = w3;
        }
      } else {
        lut_data[idx].x = reclib::INVALID;
        lut_data[idx].y = reclib::INVALID;
      }
    }
  }
}

static void remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst,
                  reclib::interpolation_t type, k4a_image_format_t format) {
  int src_width = k4a_image_get_width_pixels(src);
  int dst_width = k4a_image_get_width_pixels(dst);
  int dst_height = k4a_image_get_height_pixels(dst);

  void *src_data = (void *)k4a_image_get_buffer(src);
  void *dst_data = (void *)k4a_image_get_buffer(dst);

  int elem_size = 0;  // size in bytes
  int channels = 1;
  if (format == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
    elem_size = 1;
    channels = 4;

  } else if (format == K4A_IMAGE_FORMAT_DEPTH16) {
    elem_size = 2;
    channels = 1;

  } else {
    throw std::runtime_error("Unknown format.");
  }

  reclib::coordinate_t *lut_data =
      (reclib::coordinate_t *)(void *)k4a_image_get_buffer(lut);

  memset(dst_data, 0,
         (size_t)dst_width * (size_t)dst_height * elem_size * channels);

  for (int i = 0; i < dst_width * dst_height; i++) {
    if (lut_data[i].x != reclib::INVALID && lut_data[i].y != reclib::INVALID) {
      if (type == reclib::INTERPOLATION_NEARESTNEIGHBOR) {
        for (int ch = 0; ch < channels; ch++) {
          ((uint8_t *)dst_data)[(i * channels + ch) * elem_size] = ((
              uint8_t *)src_data)
              [((lut_data[i].y * src_width + lut_data[i].x) * channels + ch) *
               elem_size];
        }

      } else if (type == reclib::INTERPOLATION_BILINEAR ||
                 type == reclib::INTERPOLATION_BILINEAR_DEPTH) {
        const int neighbors[4]{
            lut_data[i].y * src_width + lut_data[i].x,
            lut_data[i].y * src_width + lut_data[i].x + 1,
            (lut_data[i].y + 1) * src_width + lut_data[i].x,
            (lut_data[i].y + 1) * src_width + lut_data[i].x + 1};

        if (type == reclib::INTERPOLATION_BILINEAR_DEPTH) {
          uint16_t v0 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[0] * elem_size * channels]);
          uint16_t v1 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[1] * elem_size * channels]);
          uint16_t v2 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[2] * elem_size * channels]);
          uint16_t v3 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[3] * elem_size * channels]);
          // If the image contains reclib::INVALID data, e.g. depth image
          // contains value 0, ignore the bilinear interpolation for current
          // target pixel if one of the neighbors contains reclib::INVALID
          // data to avoid introduce noise on the edge. If the image is color
          // or ir images, user should use reclib::INTERPOLATION_BILINEAR
          if (v0 == 0 || v1 == 0 || v2 == 0 || v3 == 0) {
            continue;
          }

          // Ignore interpolation at large depth discontinuity without
          // disrupting slanted surface Skip interpolation threshold is
          // estimated based on the following logic:
          // - angle between two pixels is: theta = 0.234375 degree (120
          // degree / 512) in binning resolution mode
          // - distance between two pixels at same depth approximately is: A
          // ~= sin(theta) * depth
          // - distance between two pixels at highly slanted surface (e.g.
          // alpha = 85 degree) is: B = A / cos(alpha)
          // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
          // We use B as the threshold that to skip interpolation if the depth
          // difference in the triangle is larger than B. This is a
          // conservative threshold to estimate largest distance on a highly
          // slanted surface at given depth, in reality, given distortion,
          // distance, resolution difference, B can be smaller
          const float skip_interpolation_ratio = 0.04693441759f;
          float depth_min = std::min(std::min(v0, v1), std::min(v2, v3));
          float depth_max = std::max(std::max(v0, v1), std::max(v2, v3));
          float depth_delta = depth_max - depth_min;
          float skip_interpolation_threshold =
              skip_interpolation_ratio * depth_min;
          if (depth_delta > skip_interpolation_threshold) {
            continue;
          }
        }

        if (format == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
          for (int ch = 0; ch < channels; ch++) {
            uint8_t v0 =
                (((uint8_t *)(src_data))[neighbors[0] * elem_size * channels +
                                         ch * elem_size]);
            uint8_t v1 =
                (((uint8_t *)(src_data))[neighbors[1] * elem_size * channels +
                                         ch * elem_size]);
            uint8_t v2 =
                (((uint8_t *)(src_data))[neighbors[2] * elem_size * channels +
                                         ch * elem_size]);
            uint8_t v3 =
                (((uint8_t *)(src_data))[neighbors[3] * elem_size * channels +
                                         ch * elem_size]);

            uint8_t *ptr =
                (uint8_t *)dst_data + i * elem_size * channels + ch * elem_size;
            ptr[i] = (uint8_t)(
                v0 * lut_data[i].weight[0] + v1 * lut_data[i].weight[1] +
                v2 * lut_data[i].weight[2] + v3 * lut_data[i].weight[3] + 0.5f);
          }

        } else if (format == K4A_IMAGE_FORMAT_DEPTH16) {
          uint16_t v0 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[0] * elem_size * channels]);
          uint16_t v1 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[1] * elem_size * channels]);
          uint16_t v2 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[2] * elem_size * channels]);
          uint16_t v3 = uint16_t(
              ((uint8_t *)(src_data))[neighbors[3] * elem_size * channels]);

          uint16_t *ptr =
              (uint16_t *)((uint8_t *)dst_data + i * elem_size * channels);
          ptr[i] = (uint16_t)(
              v0 * lut_data[i].weight[0] + v1 * lut_data[i].weight[1] +
              v2 * lut_data[i].weight[2] + v3 * lut_data[i].weight[3] + 0.5f);
        }

      } else {
        printf("Unexpected interpolation type!\n");
        exit(-1);
      }
    }
  }
}

// ---------------------------------------------------------
// End of Code adapted from:
// https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/examples/undistort/main.cpp
// --------------------------------------------------------

reclib::KinectAzureUndistortion reclib::KinectAzureCamera::create_undistortion(
    k4a_calibration_type_t type, reclib::interpolation_t interpolation_type,
    k4a::calibration &calibration) {
  KinectAzureUndistortion k;
  k.type = type;
  k.interpolation_type = interpolation_type;

  // Generate a pinhole model for depth camera
  // k.pinhole = create_pinhole_from_xy_range(&calibration, type);

  if (type == K4A_CALIBRATION_TYPE_DEPTH) {
    k.pinhole.fx =
        calibration.depth_camera_calibration.intrinsics.parameters.param.fx;
    k.pinhole.fy =
        calibration.depth_camera_calibration.intrinsics.parameters.param.fy;
    k.pinhole.px =
        calibration.depth_camera_calibration.intrinsics.parameters.param.cx;
    k.pinhole.py =
        calibration.depth_camera_calibration.intrinsics.parameters.param.cy;
    k.pinhole.width = calibration.depth_camera_calibration.resolution_width;
    k.pinhole.height = calibration.depth_camera_calibration.resolution_height;
  } else if (K4A_CALIBRATION_TYPE_COLOR) {
    k.pinhole.fx =
        calibration.color_camera_calibration.intrinsics.parameters.param.fx;
    k.pinhole.fy =
        calibration.color_camera_calibration.intrinsics.parameters.param.fy;
    k.pinhole.px =
        calibration.color_camera_calibration.intrinsics.parameters.param.cx;
    k.pinhole.py =
        calibration.color_camera_calibration.intrinsics.parameters.param.cy;
    k.pinhole.width = calibration.color_camera_calibration.resolution_width;
    k.pinhole.height = calibration.color_camera_calibration.resolution_height;
  } else {
    throw std::runtime_error("Unknown type");
  }

  k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, k.pinhole.width, k.pinhole.height,
                   k.pinhole.width * (int)sizeof(reclib::coordinate_t),
                   &(k.lut));

  create_undistortion_lut(&calibration, type, &(k.pinhole), k.lut,
                          interpolation_type);

  return k;
}

k4a::image reclib::KinectAzureCamera::undistort(
    k4a::image &distorted, reclib::KinectAzureUndistortion &undistortion) {
  int size = 0;
  if (distorted.get_format() == K4A_IMAGE_FORMAT_DEPTH16) {
    size = sizeof(uint16_t);

  } else if (distorted.get_format() == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
    size = sizeof(uint32_t);
  } else {
    throw std::runtime_error("Unknown image format.");
  }

  k4a_image_t undistorted;
  k4a_image_create(distorted.get_format(), undistortion.pinhole.width,
                   undistortion.pinhole.height,
                   undistortion.pinhole.width * size, &undistorted);

  remap(distorted.handle(), undistortion.lut, undistorted,
        undistortion.interpolation_type, distorted.get_format());

  return undistorted;
}

vec2 reclib::KinectAzureCamera::undistort(
    const vec2 &distorted, KinectAzureUndistortion &undistortion) {
  reclib::coordinate_t *lut_data =
      (reclib::coordinate_t *)(void *)k4a_image_get_buffer(undistortion.lut);
  int i = distorted.y() * undistortion.pinhole.width + distorted.x();
  vec2 out(std::numeric_limits<float>::max(),
           std::numeric_limits<float>::max());

  if (lut_data[i].x != reclib::INVALID && lut_data[i].y != reclib::INVALID) {
    if (undistortion.interpolation_type ==
        reclib::INTERPOLATION_NEARESTNEIGHBOR) {
      out.x() = lut_data[i].x;
      out.y() = lut_data[i].y;

    } else if (undistortion.interpolation_type ==
                   reclib::INTERPOLATION_BILINEAR ||
               undistortion.interpolation_type ==
                   reclib::INTERPOLATION_BILINEAR_DEPTH) {
      throw std::runtime_error(
          "Cannot map to a fixed pixel if interpolation mode is bilinear");

    } else {
      printf("Unexpected interpolation type!\n");
      exit(-1);
    }
  } else {
    return vec2(-1, -1);
  }

  return out;
}

#endif  // HAS_K4A

#endif  // HAS_OPENCV_MODULE
