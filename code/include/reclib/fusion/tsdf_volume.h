#ifndef CUDA_TSDF_VOLUME_H
#define CUDA_TSDF_VOLUME_H

#if WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "reclib/data_types.h"

namespace reclib {
#define DIVSHORTMAX 0.0000305185f

#if HAS_OPENCV_MODULE
float interpolate_trilinearly(const vec3& point, const CpuMat_<short2>& volume,
                              const ivec3& volume_size,
                              const float voxel_scale);

template <typename T>
bool interpolate_trilinearly(const Eigen::Vector<T, 3>& point,
                             const CpuMat_<short2>& volume,
                             const ivec3& volume_size, T& result,
                             ivec3 grid_index) {
  if ((point.array() < Eigen::Vector<T, 3>::Zero().array()).any() ||
      (point.array() >= volume_size.cast<T>().array()).any()) {
    return false;
  }

  // In 2D: determine the 'upper left' corner of the voxel
  // to which 'point' belongs
  // this is the first point of the axis-wise lines, which
  // we want to interpolate
  // (it is not necessarily the starting point)
  Eigen::Vector<T, 3> point_in_grid = Eigen::ceil(point.array());

  // compute the center of current voxel
  const T vx = (point_in_grid.x() + T(0.5f));
  const T vy = (point_in_grid.y() + T(0.5f));
  const T vz = (point_in_grid.z() + T(0.5f));

  // determine, whether we want to interpolate between
  // preceeding neighbors ("point_in_grid-1") and current voxel or succeeding
  // neighbors ("point_in_grid+1") and current voxel
  // Example in x-direction: If point is smaller than the voxel center it
  // belongs to we want to interpolate with the left voxel neighbor. If the
  // point is larger than the voxel center, we want to interpolate with the
  // right voxel neighbor
  point_in_grid.x() = (point.x() < vx)
                          ? std::max(point_in_grid.x() - T(1), T(0))
                          : point_in_grid.x();
  point_in_grid.y() = (point.y() < vy)
                          ? std::max(point_in_grid.y() - T(1), T(0))
                          : point_in_grid.y();
  point_in_grid.z() = (point.z() < vz)
                          ? std::max(point_in_grid.z() - T(1), T(0))
                          : point_in_grid.z();

  const T a = (point.x() - (point_in_grid.x() + T(0.5f)));
  const T b = (point.y() - (point_in_grid.y() + T(0.5f)));
  const T c = (point.z() - (point_in_grid.z() + T(0.5f)));

  const ivec3 neighbor(std::min(grid_index.x() + 1, volume_size.x() - 1),
                       std::min(grid_index.y() + 1, volume_size.y() - 1),
                       std::min(grid_index.z() + 1, volume_size.z() - 1));

  result = T(0);
  short2 val = volume.ptr<short2>((grid_index.z()) * volume_size.y() +
                                  grid_index.y())[grid_index.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * (T(1) - a) * (T(1) - b) * (T(1) - c);

  val = volume.ptr<short2>(neighbor.z() * volume_size.y() +
                           grid_index.y())[grid_index.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * (T(1) - a) * (T(1) - b) * c;

  val = volume.ptr<short2>((grid_index.z()) * volume_size.y() +
                           neighbor.y())[grid_index.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * (T(1) - a) * b * (T(1) - c);

  val = volume.ptr<short2>(neighbor.z() * volume_size.y() +
                           neighbor.y())[grid_index.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * (T(1) - a) * b * c;

  val = volume.ptr<short2>((grid_index.z()) * volume_size.y() +
                           grid_index.y())[neighbor.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * a * (T(1) - b) * (T(1) - c);

  val = volume.ptr<short2>(neighbor.z() * volume_size.y() +
                           grid_index.y())[neighbor.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * a * (T(1) - b) * c;

  val = volume.ptr<short2>((grid_index.z()) * volume_size.y() +
                           neighbor.y())[neighbor.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * a * b * (T(1) - c);

  val = volume.ptr<short2>(neighbor.z() * volume_size.y() +
                           neighbor.y())[neighbor.x()];
  if (val.y == 0) return false;
  result += T(val.x * DIVSHORTMAX) * a * b * c;

  return true;
}

float get_tsdf_value(const vec3& point, CpuMat_<short2>& volume,
                     const ivec3& volume_size, const float voxel_scale);

#endif
}  // namespace reclib

#endif