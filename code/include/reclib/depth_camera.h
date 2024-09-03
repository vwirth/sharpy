#ifndef DEPTH_CAMERA_H
#define DEPTH_CAMERA_H

#if HAS_FREENECT2
#include <libfreenect2/config.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

#include <libfreenect2/libfreenect2.hpp>
#endif

#if HAS_K4A
#include <k4a/k4a.hpp>
#endif

#include <reclib/internal/filesystem.h>

#include <Eigen/Core>
#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#if HAS_OPENCV_MODULE
#include <opencv2/core/matx.hpp>
#endif
#include <thread>
#include <vector>

#include "reclib/camera_parameters.h"
#include "reclib/opengl/query.h"
#include "reclib/platform.h"

namespace fs = std::filesystem;

namespace reclib {

#if HAS_OPENCV_MODULE
// ---------------------------------------------------------------------------
// Code adapted from:
// https://github.com/sxyu/avatar
// Copyright 2019 Alex Yu

// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
// ---------------------------------------------------------------------------

/*
 *
 * Models the interface to a device that provides raw depth images
 */
class _API DepthCamera {
  // Section A: Methods that should be implemented in child camera classes
 public:
  /**
   * Get the camera's model name.
   */
  virtual const std::string GetModelName() const;

  /**
   * Returns the width of the RGB frame in pixels.
   */
  virtual int GetWidth() const = 0;

  /**
   * Returns the height of the RGB frame in pixels.
   */
  virtual int GetHeight() const = 0;

  /**
   * Returns the width of the depth frame in pixels.
   */
  virtual int GetDepthWidth() const = 0;

  /**
   * Returns the height of the depth frame in pixels.
   */
  virtual int GetDepthHeight() const = 0;

  /**
   * Constructor for the DepthCamera class
   */
  DepthCamera(float depth_scaling = 1.f, float image_scaling = 1.f,
              const fs::path &file_base_path = "./");

  unsigned long int GetFrameCounter() const;

  /**
   * Destructor for the DepthCamera class (automatically stops capturing)
   */
  virtual ~DepthCamera();

 protected:
  // Section A.1: Protected methods that must be implemented in child camera
  // classes

  /**
   * Helper for initializing images used by the generic depth camera.
   * Allocates memory for back buffers if required.
   */
  virtual void InitializeImages();

  /**
   * Retrieve the next frame from the camera, updating the xyz, rgb, ir, etc.
   * images. NOTE: Method is abstract and must be implemented in child
   * classes. Directly modify the images passed to this function in the update
   * method to update the camera's images. The images needed will already be
   * initialized to getHeight() * getWidth(). WARNING: if has***Map() is false
   * for the camera class, then the ***_map is not guarenteed to be
   * initialized. so for ex. if you plan to enable the RGB map, please
   * override hasRGBMap() to return true, etc.
   * @param [out] xyz_map XYZ map (projection point cloud). CV_32FC3
   * @param [out] rgb_map RGB image. CV_8UC3
   * @param [out] ir_map IR image. CV_32FC1
   * @param [out] amp_map amplitude map. CV_32FC1
   * @param [out] flag_map flag map. CV_8UC1
   * @param [out] raw_depth_map raw depth map map. CV_32FC1
   */
  virtual void Update(CpuMat &xyz_map, CpuMat &rgb_map, CpuMat &ir_map,
                      CpuMat &amp_map, CpuMat &flag_map, CpuMat &raw_depth_map,
                      bool first_frame = false) = 0;

 public:
  // Section B: Stuff that may be overridden but don't need to be

  /**
   * Returns true if an XYZ image is available from this camera.
   */
  virtual bool HasXYZMap() const;

  /**
   * Returns true if an RGB image is available from this camera.
   */
  virtual bool HasRGBMap() const;

  /**
   * Returns true if an IR image is available from this camera.
   */
  virtual bool HasIRMap() const;

  /**
   * Returns true if a flag map is available from this camera.
   */
  virtual bool HasAmpMap() const;

  /**
   * Returns true if a flag map is available from this camera.
   */
  virtual bool HasFlagMap() const;

  /**
   * Returns true if a raw depth map map is available from this camera.
   */
  virtual bool HasRawDepthMap() const;

  /**
   * Value that determines the validity of a point with respect to the
   * camera's ampMap.
   */
  virtual int AmpMapInvalidFlagValue() const;

  /**
   * Value that determines the validity of a point with respect to the
   * camera's flagMap.
   */
  virtual float FlagMapConfidenceThreshold() const;

  /**
   * Check if the camera input is invalid.
   * @return true on bad input (e.g. error or disconnection), false otherwise
   */
  virtual bool BadInput();

  /**
    Start or stop saving the captured frames
  **/
  virtual void StartRecording(const fs::path &file_base_path);

  // Section C: Generic methods/variables that may be used by all cameras

  /**
   * Retrieve the next frame from the depth camera.
   * Calls the update() function of the derived camera class and resets stored
   * information for the frame.
   * @param removeNoise if true, performs noise removal on the depth image
   * after retrieving it
   * @return true on success, false on bad input
   */
  bool NextFrame(bool remove_noise = true, bool first_frame = false);

  /**
   * Begin capturing frames continuously from this camera on a parallel
   * thread, capped at a certain maximum FPS. WARNING: throws an error if
   * capture already started.
   * @param fps_cap maximum FPS of capture (-1 to disable)
   * @param removeNoise if true, performs noise removal on the depth image
   * after retrieving it
   * @see endCapture
   * @see isCapturing
   */
  void BeginCapture(int fps_cap = -1, bool remove_noise = true);

  /**
   * Stop capturing from this camera.
   * You may use beginCapture() to start capturing again afterwards.
   * Note: this is performed automatically when this instance is destroyed.
   * @see beginCapture
   * @see isCapturing
   */
  void EndCapture();

  /**
   * Returns true if the camera is currently capturing, false otherwise.
   * @see beginCapture
   * @see endCapture
   */
  bool IsCapturing();

  void StopRecording();
  bool IsRecording();

  /** returns true in case update function has been called and
      new values are inserted to the frame buffers
      **/
  bool HasNextFrame();

  /**
   * Add a callback function to be called after each frame update.
   * WARNING: may be called from a different thread than the one where the
   * callback is added.
   * @param func the function. Must take exactly one argument--a reference to
   * the updated DepthCamera instance
   * @see removeUpdateCallBack
   * @return unique ID for this callback function, needed for
   * removeUpdateCallback.
   */
  int AddUpdateCallback(std::function<void(DepthCamera &)> func);

  /** Remove the update callback function with the specified unique ID.
   *  (The ID may be obtained from by addUpdateCallback when the callback is
   * added)
   * @see addUpdateCallBack
   */
  void RemoveUpdateCallback(int id);

  /**
   * Returns the size of the camera's frame (getWidth() * getHeight).
   */
  cv::Size GetImageSize() const;
  cv::Size GetDepthImageSize() const;

  /**
   * Returns the current XYZ map (ordered point cloud) of the camera.
   * Contains the XYZ position (in meters) of each pixel on the screen.
   * Type: CV_32FC3
   */
  const CpuMat GetXYZMapAndUpdateBufs();

  /**
   * Get the RGB Image from this camera, if available. Else, throws an error.
   * Type: CV_8UC3
   */
  const CpuMat GetRGBMapAndUpdateBufs();

  /**
   * Get the infrared (IR) Image from this camera, if available. Else, throws
   * an error. Type: CV_8UC1
   */
  const CpuMat GetIRMapAndUpdateBufs();

  /**
   * Returns the current AmpMap
   * Type: CV_32FC1
   */
  const CpuMat GetAmpMapAndUpdateBufs();

  /**
   * Returns the current FlagMap.
   * Type: CV_8UC1
   */
  const CpuMat GetFlagMapAndUpdateBufs();

  /**
   * Returns the current unprocessed depth map (in meters)
   * Type: CV_32FC1
   */
  const CpuMat GetRawDepthMapAndUpdateBufs();

  void GetMaps(CpuMat &depth, CpuMat &rgb, CpuMat &xyz);
  void GetMaps(CpuMat &depth, CpuMat &rgb, CpuMat &xyz, CpuMat &ir);
  void GetMaps(CpuMat &depth, CpuMat &rgb, CpuMat &xyz, CpuMat &ir, CpuMat &amp,
               CpuMat &flag);

  // disables generation of maps
  void DisableXYZMap();
  void DisableRGBMap();
  void DisableIRMap();
  void DisableAmpMap();
  void DisableFlagMap();
  void DisableRawDepthMap();
  void EnableXYZMap();
  void EnableRGBMap();
  void EnableIRMap();
  void EnableAmpMap();
  void EnableFlagMap();
  void EnableRawDepthMap();

  /** Get the timestamp of the last image in nanoseconds */
  uint64_t GetTimestamp() const;

  /** Get camera intrinsics */
  IntrinsicParameters GetIntrinsics() const;

  /** Get camera intrinsics */
  IntrinsicParameters GetDepthIntrinsics() const;

  /** Get camera extrinsics */
  ExtrinsicParameters GetExtrinsicsRGBToD() const;

  /** Get camera intrinsics */
  ExtendedIntrinsicParameters GetExtIntrinsics() const;

  /** Get camera intrinsics */
  ExtendedIntrinsicParameters GetExtDepthIntrinsics() const;

  // field of view (in degrees)
  float GetFovX() const;
  float GetFovY() const;
  float GetDepthFovX() const;
  float GetDepthFovY() const;

  /**
   * Reads a sample frame from file.
   * @param source the directory which the frame file is stored
   */
  bool ReadImageYAML(const fs::path &source,
                     long unsigned int frame_counter = -1);

  /**
   * Writes the current frame into file.
   * @param destination the directory which the frame should be written to
   */
  bool WriteImageYAML(const fs::path &destination, fs::path filename = "");

  /**
   * Reads a sample frame from file.
   * @param source the directory which the frame file is stored
   */
  bool ReadImage(CpuMat &xyz_map, CpuMat &rgb_map, CpuMat &ir_map,
                 CpuMat &amp_map, CpuMat &flag_map, CpuMat &raw_depth_map,
                 const fs::path &source, long unsigned int frame_counter = -1);

  /**
   * Writes the current frame into file.
   * @param destination the directory which the frame should be written to
   */
  bool WriteImage(const fs::path &destination, fs::path filename = "") const;

  virtual bool WriteIntrinsics(const fs::path &destination,
                               fs::path filename = "") const;

  bool WriteExtrinsics(const fs::path &destination,
                       fs::path filename = "") const;

  /** Shared pointer to depth camera instance */
  typedef std::shared_ptr<DepthCamera> Ptr;

 protected:
  /**
   * Matrix storing the (x,y,z) data of every point in the observable world.
   * Matrix type CV_32FC3
   */
  mutable CpuMat xyz_map_;

  /**
   * Matrix of confidence values of each corresponding point in the world.
   * Matrix type CV_32FC1
   */
  mutable CpuMat amp_map_;

  /**
   * Matrix representing additional information about the points in the
   * world. Matrix type CV_8UC1
   */
  mutable CpuMat flag_map_;

  /**
   * The RGB image from this camera, if available
   * Matrix type CV_8UC3
   */
  mutable CpuMat rgb_map_;

  /**
   * The infrared image from this camera, if available
   * Matrix type CV_8UC1
   */
  mutable CpuMat ir_map_;

  /**
   * The raw depth map from this camera, if available
   * Matrix type is CV_32FC1
   */
  mutable CpuMat raw_depth_map_;

  /** Back buffers for various images */
  CpuMat xyz_map_buf_;
  CpuMat rgb_map_buf_;
  CpuMat ir_map_buf_;
  CpuMat amp_map_buf_;
  CpuMat flag_map_buf_;
  CpuMat raw_depth_map_buf_;

  /**
   * True if input is invalid
   * By default, badInput() returns the value of badInputFlag.
   * badInput()'s behavior may be overridden.
   */
  std::atomic<bool> bad_input_flag_;

  /**
   * True if device is open (default true).
   * if false, camera will not be able to begin capturing
   */
  std::atomic<bool> device_open_flag_;

  /** Latest timestamp in ns */
  std::atomic<int64_t> timestamp_{};

  /** Pinhole intrinsics */
  std::atomic<float> fx_{}, cx_{}, fy_{}, cy_{};
  /** Pinhole intrinsics for IR sensor*/
  std::atomic<float> dfx_{}, dcx_{}, dfy_{}, dcy_{};
  /** Field of view parameters (in degrees) */
  std::atomic<float> fovx_{}, fovy_{};
  std::atomic<float> dfovx_{}, dfovy_{};
  /** Distorion parameters */
  std::vector<float> rad_distortion_;
  std::vector<float> drad_distortion_;
  std::vector<float> tang_distortion;
  std::vector<float> dtang_distortion;

  // whether data was requested by Get methods such as GetXYZ, etc.
  mutable std::atomic<bool> data_requested_;

  // used to enable and disable generation of maps
  bool has_xyz_map_;
  bool has_rgb_map_;
  bool has_ir_map_;
  bool has_amp_map_;
  bool has_flag_map_;
  bool has_raw_depth_map_;

  // scaling factor to multiply the raw depth values with
  float depth_scaling_;
  // scales all images (aka maps) down by this scaling factor
  float image_scaling_;

  /** Mutex to ensure thread safety while updating images
   *  (mutable = modificable even to const methods)
   */
  mutable std::mutex image_mutex_;

  static const fs::path depth_dir;
  static const fs::path xyz_dir;
  static const fs::path amp_dir;
  static const fs::path rgb_dir;
  static const fs::path ir_dir;
  static const fs::path flag_dir;

  // base path to automatically write files into
  fs::path file_base_path_;

  unsigned long int frame_counter_;
  ExtrinsicParameters extrinsics_rgbtod_;

 private:
  // Section D: implementation details

  /**
   * Helper for swapping a single back buffer to the foreground.
   * If the image is not available, creates a dummy CpuMat with null value.
   * @param check_func member function pointer to function that, if true on
   * call, buffers are swapped if false, a dummy CpuMat is created
   * @param img pointer to foreground image
   * @param buf pointer to back buffer
   */
  void SwapBuffer(bool (DepthCamera::*check_func)() const, CpuMat &img,
                  CpuMat &buf);

  /**
   * Helper for swapping all back buffers to the foreground.
   * If an image is not available, creates a dummy CpuMat with null value.
   */
  void SwapBuffers();

  /**
   * Removes noise from an XYZMap based on confidence provided in the AmpMap
   * and FlagMap.
   */
  static void RemoveNoise(CpuMat &xyz_map, CpuMat &amp_map,
                          float confidence_thresh);

  /** stores the callbacks functions to call after each update (ID, function)
   */
  std::map<int, std::function<void(DepthCamera &)>> update_callbacks_;

  /** interrupt for immediately terminating the capturing thread */
  std::atomic<bool> capture_interrupt_;

  /**
   * Minimum depth of points (in meters). Points under this depth are presumed
   * to be noise. (0.0 to disable) (Defined in DepthCamera.cpp)
   */
  static const float kNoiseFilterLow;

  /** Thread **/
  std::unique_ptr<std::thread> capture_thd_;

  bool recording_;
};

// ---------------------------------------------------------------------------
// End of adapted code
// ---------------------------------------------------------------------------

class _API DefaultCamera : public DepthCamera {
 public:
  explicit DefaultCamera(const fs::path &base_path, bool repeat_sequence = true,
                         float depth_scaling = 1.f,
                         unsigned int start_frame = 0);

  ~DefaultCamera() override;

  /**
   * Get the camera's model name.
   */
  const std::string GetModelName() const override;

  /**
   * Returns the width of the SR300 camera frame
   */
  int GetWidth() const override;

  /**
   * Returns the height of the SR300 camera frame
   */
  int GetHeight() const override;

  /**
   * Returns the width of the IR depth frame.
   */
  int GetDepthWidth() const override;

  /**
   * Returns the height of the IR depth frame.
   */
  int GetDepthHeight() const override;

  // disable recording mode since DefaultCamera is recorded anyway
  void StartRecording(const fs::path &file_base_path) override;

  long int GetFilesystemCounter();

  fs::path GetFilename();

 protected:
  /**
   * Gets the new frame from the sensor (implements functionality).
   * Updates xyzMap and ir_map.
   */
  void Update(CpuMat &xyz_map, CpuMat &rgb_map, CpuMat &ir_map, CpuMat &amp_map,
              CpuMat &flag_map, CpuMat &raw_depth_map,
              bool first_frame = false) override;

  bool repeat_sequence_;
  long int filesystem_frame_counter_;
  long int start_filesystem_frame_counter_;
  int width_;
  int height_;
  int depth_width_;
  int depth_height_;

  void UpdateFilesystemFrameCounter();
};
/*
 * Provides depth frames acquired by a Kinect V2 camera.
 */

#if HAS_FREENECT2
/**
 * Adapted from sxyu's Avatar project:
 * https://github.com/sxyu/avatar
 *
  // Copyright 2019 Alex Yu

  // Licensed under the Apache License, Version 2.0 (the "License"); you may not
  // use this file except in compliance with the License. You may obtain a copy
 of
  // the License at

  // http://www.apache.org/licenses/LICENSE-2.0

  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  // WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  // License for the specific language governing permissions and limitations
 under
  // the License.

 * Class defining the behavior of an Kinect V2 Camera using libfreenect2.
 * Example on how to read from sensor and visualize its output
 * @include SensorIO.cpp
 */
class _API KinectV2FN2Camera : public DepthCamera {
 public:
  /** Freenect2 KinectV2 constructor
   *  @param serial serial number of device to open. Leave empty to use
   * default.
   *  @param use_kde whether to use Kernel Density Estimation (KDE), Lawin et
   * al. ECCV16 only available if Freenect2 built with CUDA or OpenCL
   *  @param scale amount to scale down final image by
   *  @param verbose enable verbose output
   */
  explicit KinectV2FN2Camera(std::string serial = "", bool use_kde = false,
                             float image_scaling = 1.0F,
                             float depth_scaling = 1.0f, bool verbose = false,
                             bool compute_xy = true);

  /** Clean up method (capture thread) */
  ~KinectV2FN2Camera() override;

  /**
   * Get the camera's model name.
   */
  const std::string GetModelName() const override;

  /**
   * Returns the width of the SR300 camera frame
   */
  int GetWidth() const override;

  /**
   * Returns the height of the SR300 camera frame
   */
  int GetHeight() const override;

  /**
   * Returns the width of the IR depth frame.
   */
  int GetDepthWidth() const override;

  /**
   * Returns the height of the IR depth frame.
   */
  int GetDepthHeight() const override;

  void DisableXYComputation();

  /** Shared pointer to Freenect2Kinect camera instance */
  using Ptr = std::shared_ptr<KinectV2FN2Camera>;

 protected:
  /**
   * Gets the new frame from the sensor (implements functionality).
   * Updates xyzMap and ir_map.
   */
  void Update(CpuMat &xyz_map, CpuMat &rgb_map, CpuMat &ir_map, CpuMat &amp_map,
              CpuMat &flag_map, CpuMat &raw_depth_map,
              bool first_frame = false) override;

  /**
   * Initialize the camera, opening channels and resetting to initial
   * configurations
   */
  void InitCamera();

  // internal storage
  // * Device info
  // Whether to use Kernel Density Estimation (KDE), Lawin et al. ECCV16
  bool use_kde_;
  // Device serial number
  std::string serial_;
  // Verbose mode
  bool verbose_;

  // whether to compute the x and y components of the xyz map (more time
  // expensive)
  bool compute_xy_;

  // Scaled size
  double scaled_width_, scaled_height_;
  // Scaled depth size
  double scaled_dwidth_, scaled_dheight_;

  // * Context
  std::unique_ptr<libfreenect2::Freenect2> freenect2_;
  std::unique_ptr<libfreenect2::Freenect2Device> device_;
  libfreenect2::PacketPipeline *pipeline_;
  std::unique_ptr<libfreenect2::Registration> registration_;
  std::unique_ptr<libfreenect2::SyncMultiFrameListener> listener_;
  libfreenect2::FrameMap frames_;

  Eigen::Array<float, 3, Eigen::Dynamic> xy_table_cache_;

  const int32_t timeout_in_ms_ = 1000;

  static const int kDefaultRgbWidth, kDefaultRgbHeight;
  static const int kDefaultDepthWidth, kDefaultDepthHeight;
};
#endif  // HAS_FREENECT2

#if HAS_K4A

const uint32_t INVALID = std::numeric_limits<uint32_t>::min();

typedef struct _pinhole_t {
  float px;
  float py;
  float fx;
  float fy;

  int width;
  int height;
} pinhole_t;

typedef struct _coordinate_t {
  int x;
  int y;
  float weight[4];
} coordinate_t;

typedef enum {
  INTERPOLATION_NEARESTNEIGHBOR, /**< Nearest neighbor interpolation */
  INTERPOLATION_BILINEAR,        /**< Bilinear interpolation */
  INTERPOLATION_BILINEAR_DEPTH   /**< Bilinear interpolation with invalidation
                                    when neighbor contain invalid
                                               data with value 0 */
} interpolation_t;

struct KinectAzureUndistortion {
  k4a_calibration_type_t type;
  interpolation_t interpolation_type;
  pinhole_t pinhole;
  k4a_image_t lut;
};

/**
 * Adapted from sxyu's Avatar project:
 * https://github.com/sxyu/avatar
 *
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

 * Class defining the behavior of an Kinect V2 Camera using libfreenect2.
 * Example on how to read from sensor and visualize its output
 * @include SensorIO.cpp
 */
class _API KinectAzureCamera : public DepthCamera {
 public:
  /** Freenect2 KinectV2 constructor
   *  @param serial serial number of device to open. Leave empty to use
   * default.
   *  @param scale amount to scale down final image by
   *  @param verbose enable verbose output
   */
  explicit KinectAzureCamera(std::string serial = "",
                             float image_scaling = 1.0F,
                             float depth_scaling = 1 / 1000.0f,
                             bool verbose = false,
                             bool map_modalities_to_color = false);

  /** Freenect2 KinectV2 constructor
   *  @param serial serial number of device to open. Leave empty to use
   * default.
   *  @param color_format color format according to table in
   * https://unanancyowen.github.io/k4asdk_apireference/k4atypes_8h.html#abd9688eb20d5cb878fd22d36de882ddb
   *  @param depth_mode depth mode according to table in
   * https://unanancyowen.github.io/k4asdk_apireference/k4atypes_8h.html#a3507ee60c1ffe1909096e2080dd2a05d
   *  @param scale amount to scale down final image by
   *  @param verbose enable verbose output
   */
  explicit KinectAzureCamera(
      std::string serial = "",
      k4a_color_resolution_t color_resolution = K4A_COLOR_RESOLUTION_720P,
      k4a_depth_mode_t depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED,
      float image_scaling = 1.0F, float depth_scaling = 1 / 1000.0f,
      bool verbose = false, bool map_modalities_to_color = true);

  /** Clean up method (capture thread) */
  ~KinectAzureCamera() override;

  /**
   * Get the camera's model name.
   */
  const std::string GetModelName() const override;

  /**
   * Returns the width of the SR300 camera frame
   */
  int GetWidth() const override;

  /**
   * Returns the height of the SR300 camera frame
   */
  int GetHeight() const override;

  /**
   * Returns the width of the IR depth frame.
   */
  int GetDepthWidth() const override;

  /**
   * Returns the height of the IR depth frame.
   */
  int GetDepthHeight() const override;

  bool WriteIntrinsics(const fs::path &destination,
                       fs::path filename = "") const override;

  k4a::calibration GetCalibration() { return calibration_; }

  /** Shared pointer to KinectAzureCamera camera instance */
  using Ptr = std::shared_ptr<KinectAzureCamera>;

  static k4a::calibration CalibrationFromFile(const fs::path &destination,
                                              fs::path filename = "");

  static CpuMat Image2Mat(k4a::image &img, int channels = 1);
  static k4a::image Mat2Image(CpuMat &mat, bool use_depth_format = true);
  static cv::Mat depth2color(cv::Mat &depth, cv::Mat &color,
                             k4a::calibration &calib);
  static cv::Mat custom2color(cv::Mat &custom, cv::Mat &depth, cv::Mat &color,
                              k4a::calibration &calib);
  static cv::Mat depth2xyz(
      cv::Mat &depth, k4a::calibration &calib,
      k4a_calibration_type_t type = K4A_CALIBRATION_TYPE_DEPTH);
  static cv::Mat color2depth(cv::Mat &depth, cv::Mat &color,
                             k4a::calibration &calib);
  static vec3 depth3Dto2D(const vec3 &point, k4a::calibration &calib);
  static vec3 depth2Dto3D(const vec3 &px_d, k4a::calibration &calib);
  static vec3 depth2color3D(const vec3 &point_depth, k4a::calibration &calib);
  static vec3 color2depth3D(const vec3 &point_color, k4a::calibration &calib);
  static vec3 color3Dto2D(const vec3 &point, k4a::calibration &calib);
  static vec3 color2Dto3D(const vec3 &px_d, k4a::calibration &calib);
  static mat4 calib2extrinsic(k4a_calibration_extrinsics_t &extr);
  static mat4 calib2intrinsic(k4a_calibration_intrinsics_t &intr);

  static KinectAzureUndistortion create_undistortion(
      k4a_calibration_type_t type, interpolation_t interpolation_type,
      k4a::calibration &calibration);
  static k4a::image undistort(k4a::image &distorted,
                              KinectAzureUndistortion &undistortion);
  static vec2 undistort(const vec2 &distorted,
                        KinectAzureUndistortion &undistortion);

 protected:
  /**
   * Gets the new frame from the sensor (implements functionality).
   * Updates xyzMap and ir_map.
   */
  void Update(CpuMat &xyz_map, CpuMat &rgb_map, CpuMat &ir_map, CpuMat &amp_map,
              CpuMat &flag_map, CpuMat &raw_depth_map,
              bool first_frame = false) override;

  /**
   * Initialize the camera, opening channels and resetting to initial
   * configurations
   */
  void InitCamera();

  // Device serial number
  std::string serial_;
  // Verbose mode
  bool verbose_;
  bool map_modalities_to_color_;
  // Scaled size
  double scaled_width_, scaled_height_;
  // Scaled depth size
  double scaled_dwidth_, scaled_dheight_;

  // * Context
  k4a_device_configuration_t config_;
  k4a::device device_;
  k4a::transformation T_;
  k4a::calibration calibration_;

  Eigen::Array<float, 3, Eigen::Dynamic> xy_table_cache_;

  const int32_t timeout_in_ms_ = 1000;

  static const int kDefaultRgbWidth, kDefaultRgbHeight;
  static const int kDefaultDepthWidth, kDefaultDepthHeight;

  // reclib::opengl::Timer timer_;
};

#endif  // HAS_K4A

#endif  // HAS_OPENCV_MODULE

}  // namespace reclib

#endif
