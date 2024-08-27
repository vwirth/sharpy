#ifndef RECLIB_UTILS_OPENCV_UTILS
#define RECLIB_UTILS_OPENCV_UTILS

#if HAS_OPENCV_MODULE

#include "reclib/data_types.h"

namespace reclib {
namespace utils {

/**
 * @brief Find unique elements of an OpenCV image
 *
 * @tparam type is the C++ type to access the image elements.
 * @param in is the OpenCV single-channel image to find the unique values. Note:
 * This modifies the image. Make a copy with .clone(), if you need the image
 * afterwards.
 *
 * @returns vector of unique elements
 */
template <typename type>
inline std::vector<type> unique(cv::Mat in) {
  assert(in.channels() == 1 &&
         "This implementation is only for single-channel images");
  auto begin = in.begin<type>(), end = in.end<type>();
  auto last =
      std::unique(begin, end);      // remove adjacent duplicates to reduce size
  std::sort(begin, last);           // sort remaining elements
  last = std::unique(begin, last);  // remove duplicates
  return std::vector<type>(begin, last);
}

}  // namespace utils
}  // namespace reclib

#endif  // HAS_OPENCV_MODULE

#endif