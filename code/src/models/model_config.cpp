#if __unix__
#include <cnpy.h>
#include <reclib/models/model_config.h>

#include <fstream>
#include <iostream>
#include <string>

#include "reclib/assert.h"

namespace reclib {
namespace models {

const vec2 MANOConfigExtra::anatomic_limits[] = {
    // wrist
    vec2(-std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()),
    vec2(-std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()),
    vec2(-std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()),
    // index
    vec2(-1, 1.5), vec2(-0.5, 0.5), vec2(0.1, 2.5), vec2(-0.5, 1),
    // middle
    vec2(-1, 1.5), vec2(-0.5, 0.5), vec2(0.1, 2.5), vec2(-0.5, 1),
    // pinky
    vec2(-1, 1.5), vec2(-0.5, 0.5), vec2(0.1, 2.5), vec2(-0.5, 1),
    // ring
    vec2(-1, 1.5), vec2(-0.5, 0.5), vec2(0.1, 2.5), vec2(-0.5, 1),
    // thumb
    // vec2(-std::numeric_limits<float>::infinity(),
    //      std::numeric_limits<float>::infinity()),
    // vec2(-std::numeric_limits<float>::infinity(),
    //      std::numeric_limits<float>::infinity()),
    // vec2(-std::numeric_limits<float>::infinity(),
    //      std::numeric_limits<float>::infinity()),
    // vec2(-std::numeric_limits<float>::infinity(),
    //      std::numeric_limits<float>::infinity()),
    // vec2(-std::numeric_limits<float>::infinity(),
    //      std::numeric_limits<float>::infinity()),
    // vec2(-std::numeric_limits<float>::infinity(),
    //      std::numeric_limits<float>::infinity()),
    vec2(-1, 2), vec2(-1, 1), vec2(-1, 1), vec2(-1, 2), vec2(-1, 1),
    vec2(-1, 1), vec2(-0.5, 1)};

const char* gender_to_str(Gender gender) {
  switch (gender) {
    case Gender::neutral:
      return "NEUTRAL";
    case Gender::male:
      return "MALE";
    case Gender::female:
      return "FEMALE";
    default:
      return "UNKNOWN";
  }
}

const char* hand_type_to_str(HandType type) {
  switch (type) {
    case HandType::left:
      return "LEFT";
    case HandType::right:
      return "RIGHT";
    default:
      return "UNKNOWN";
  }
}

void assert_shape(const cnpy::NpyArray& m,
                  std::initializer_list<size_t> shape) {
  _RECLIB_ASSERT_EQ(m.shape.size(), shape.size());
  size_t idx = 0;
  for (auto& dim : shape) {
    if (dim != ANY_SHAPE) _RECLIB_ASSERT_EQ(m.shape[idx], dim);
    ++idx;
  }
}

std::string find_data_file(const std::string& data_path) {
  static const std::string TEST_PATH = std::string("data/models/smplx/uv.txt");
  static const int MAX_LEVELS = 6;
  static std::string data_dir_saved = "\n";
  if (data_dir_saved == "\n") {
    data_dir_saved.clear();
    const char* env = std::getenv("SMPLX_DIR");
    if (env) {
      // use environmental variable if exists and works
      data_dir_saved = env;

      // auto append slash
      if (!data_dir_saved.empty() && data_dir_saved.back() != '/' &&
          data_dir_saved.back() != '\\')
        data_dir_saved.push_back('/');

      std::ifstream test_ifs(data_dir_saved + TEST_PATH);
      if (!test_ifs) data_dir_saved.clear();
    }

    // else check current directory and parents
    if (data_dir_saved.empty()) {
      for (int i = 0; i < MAX_LEVELS; ++i) {
        std::ifstream test_ifs(data_dir_saved + TEST_PATH);
        if (test_ifs) break;
        data_dir_saved.append("../");
      }
    }

    data_dir_saved.append("data/");
  }
  return data_dir_saved + data_path;
}

}  // namespace models
}  // namespace reclib

#endif  //__unix__