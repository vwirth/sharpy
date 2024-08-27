#ifndef VOXEL_H
#define VOXEL_H

#if HAS_OPENCV_MODULE
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "reclib/platform.h"

namespace reclib {

template <typename T>
class _API VoxelGrid {
 public:
  VoxelGrid() : grid_size_(0, 0, 0), voxel_size_(0) {}
  VoxelGrid(ivec3 grid_size, float voxel_size)
      : grid_size_(grid_size), voxel_size_(voxel_size) {}

  ~VoxelGrid() = default;

  T& Voxel(const vec3& position) {
    ivec3 index = (position / voxel_size_);
    return Voxel(index);
  };

  virtual T& Voxel(const ivec3& index) = 0;
  const ivec3& GridSize() const { return grid_size_; }
  float VoxelSize() const { return voxel_size_; }

 protected:
  ivec3 grid_size_;   // the quantity is number of voxels per dimension
  float voxel_size_;  // the size of one voxel;
};

#if HAS_OPENCV_MODULE
// models the full voxel volume with an array
// without any space optimizations
template <typename T>
class _API ArrayVoxelGrid : public VoxelGrid<T> {
 public:
  ArrayVoxelGrid() = default;
  ArrayVoxelGrid(ivec3 grid_size, float voxel_size, int type)
      : VoxelGrid<T>(grid_size, voxel_size),
        storage_(cv::cuda::createContinuous(grid_size[1] * grid_size[2],
                                            grid_size[1], type)) {
    storage_.setTo(0);
  }

  ~ArrayVoxelGrid() = default;

  T& Voxel(const ivec3& index) override {
    return storage_.ptr<T>(index[2] * this->grid_size_[1] + index[1])[index[0]];
  };

  CpuMat Download() {
    CpuMat m(storage_.size(), storage_.type());
    storage_.download(m);
    return m;
  }

  GpuMat storage_;  // 3D space stored as a 2D array with dimensions [h * d, w];
};

#endif

}  // namespace reclib

#endif
