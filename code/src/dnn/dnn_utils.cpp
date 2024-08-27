#if HAS_DNN_MODULE
#include <reclib/dnn/dnn_utils.h>

namespace torch {
torch::indexing::Slice All;
c10::nullopt_t None{0};

}  // namespace torch
#endif  // HAS_DNN_MODULE