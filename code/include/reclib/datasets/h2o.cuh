#ifndef RECLIB_DATASETS_H2O_CU
#define RECLIB_DATASETS_H2O_CU

#if HAS_OPENCV_MODULE
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#endif

namespace reclib {
namespace datasets {
namespace cuda {

#if HAS_OPENCV_MODULE
void RoundToNearestNeighbor(CpuMat& image, const CpuMat& neighbors);
void RoundToNearestNeighborIndex(CpuMat& image, const CpuMat& neighbors,
                                 CpuMat& ids);
#endif

}  // namespace cuda
}  // namespace datasets
}  // namespace reclib

#endif
