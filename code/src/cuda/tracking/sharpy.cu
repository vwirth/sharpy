#include <reclib/assert.h>
#include <reclib/data_types.h>

#include <reclib/cuda/common.cuh>
#include <reclib/cuda/data_types.cuh>
#include <reclib/cuda/debug.cuh>
#include <reclib/cuda/helper.cuh>
#include <reclib/cuda/thread_info.cuh>
#include <reclib/tracking/sharpy.cuh>

#if HAS_DNN_MODULE

__global__ void kernel_compute_segmentation(float* hsv_handspace,
                                            int32_t* segmentation, int width,
                                            int height, uint8_t* visible_ids) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || x >= width || y < 0 || y >= height) return;

  Eigen::Map<vec3> hsv(&hsv_handspace[(y * width + x) * 3]);
  if (hsv.x() == 0 && hsv.y() == 0 && hsv.z() == 0) {
    segmentation[(y * width + x)] = 30;
    return;
  }

  int hsv_finger_seg =
      max(0, int(((hsv.y() - 0.58) / (0.139)) + 1));  // yields 0, 1, 2 or 3
  int hsv_finger = 0;
  if (hsv.x() * 360 < 180) {
    hsv_finger =
        int(fmax(0, int(hsv.x() * 360 - 20)) / (80.0));  // yields 0 or 1
  } else {
    hsv_finger = int(fmax(0, int(hsv.x() * 360 - 180)) / (60.0)) +
                 2;  // yields 2,3, or 4
  }
  int id = hsv_finger * 4 + hsv_finger_seg;
  segmentation[(y * width + x)] = id;
  visible_ids[id] = 1;
}

__global__ void kernel_compute_segmentation(float* hsv_handspace,
                                            int32_t* segmentation, int width,
                                            int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || x >= width || y < 0 || y >= height) return;

  Eigen::Map<vec3> hsv(&hsv_handspace[(y * width + x) * 3]);
  if (hsv.x() == 0 && hsv.y() == 0 && hsv.z() == 0) {
    segmentation[(y * width + x)] = 30;
    return;
  }

  int hsv_finger_seg =
      max(0, int(((hsv.y() - 0.58) / (0.139)) + 1));  // yields 0, 1, 2 or 3
  int hsv_finger = 0;
  if (hsv.x() * 360 < 180) {
    hsv_finger =
        int(fmax(0, int(hsv.x() * 360 - 20)) / (80.0));  // yields 0 or 1
  } else {
    hsv_finger = int(fmax(0, int(hsv.x() * 360 - 180)) / (60.0)) +
                 2;  // yields 2,3, or 4
  }
  int id = hsv_finger * 4 + hsv_finger_seg;
  segmentation[(y * width + x)] = id;
}

void reclib::tracking::cuda::compute_segmentation_from_corrs(
    torch::Tensor& input, torch::Tensor& segmentation,
    std::vector<uint8_t>& visible_ids) {
  if (!segmentation.defined()) {
    segmentation = torch::zeros(
        {input.sizes()[0], input.sizes()[1]},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
  }
  int height = input.sizes()[0];
  int width = input.sizes()[1];

  _RECLIB_ASSERT_EQ(input.scalar_type(), torch::ScalarType::Float);
  _RECLIB_ASSERT_EQ(input.sizes()[2], 3);
  _RECLIB_ASSERT_EQ(segmentation.scalar_type(), torch::ScalarType::Int);
  _RECLIB_ASSERT_EQ(segmentation.sizes().size(), 2);
  _RECLIB_ASSERT_EQ(height, segmentation.sizes()[0]);
  _RECLIB_ASSERT_EQ(width, segmentation.sizes()[1]);

  _RECLIB_ASSERT(input.is_cuda());
  _RECLIB_ASSERT(segmentation.is_cuda());

  dim3 threads(32, 32);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

  uint8_t* visible_ids_cuda =
      upload_ptr<uint8_t>(visible_ids.data(), visible_ids.size());

  kernel_compute_segmentation<<<blocks, threads>>>(
      input.data_ptr<float>(), segmentation.data_ptr<int32_t>(), width, height,
      visible_ids_cuda);
  CUDA_SYNC_CHECK();

  download_ptr<uint8_t>(visible_ids_cuda, (uint8_t*)visible_ids.data(),
                        visible_ids.size());
}

void reclib::tracking::cuda::compute_segmentation_from_corrs(
    torch::Tensor& input, torch::Tensor& segmentation) {
  if (!segmentation.defined()) {
    segmentation = torch::zeros(
        {input.sizes()[0], input.sizes()[1]},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
  }
  int height = input.sizes()[0];
  int width = input.sizes()[1];

  _RECLIB_ASSERT_EQ(input.scalar_type(), torch::ScalarType::Float);
  _RECLIB_ASSERT_EQ(input.sizes()[2], 3);
  _RECLIB_ASSERT_EQ(segmentation.scalar_type(), torch::ScalarType::Int);
  _RECLIB_ASSERT_EQ(segmentation.sizes().size(), 2);
  _RECLIB_ASSERT_EQ(height, segmentation.sizes()[0]);
  _RECLIB_ASSERT_EQ(width, segmentation.sizes()[1]);

  _RECLIB_ASSERT(input.is_cuda());
  _RECLIB_ASSERT(segmentation.is_cuda());

  dim3 threads(32, 32);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

  kernel_compute_segmentation<<<blocks, threads>>>(
      input.data_ptr<float>(), segmentation.data_ptr<int32_t>(), width, height);
  CUDA_SYNC_CHECK();
}

#endif  // HAS_DNN_MODULE