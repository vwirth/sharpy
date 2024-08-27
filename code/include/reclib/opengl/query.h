#ifndef RECLIB_OPENGL_QUERY_H
#define RECLIB_OPENGL_QUERY_H

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <string>
#include <vector>

#include "reclib/opengl/named_handle.h"

namespace reclib {
namespace opengl {
// -------------------------------------------------------
// simple cpu timer struct

struct Timer {
  inline Timer() { begin(); }

  inline void begin() { start_time = std::chrono::system_clock::now(); }

  inline double look() {  // milliseconds
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::milli>>(
               std::chrono::system_clock::now() - start_time)
        .count();
  }

  inline double look_and_reset() {  // milliseconds
    double diff =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            std::chrono::system_clock::now() - start_time)
            .count();
    begin();
    return diff;
  }

  // data
  std::chrono::time_point<std::chrono::system_clock> start_time;
};

// -------------------------------------------------------
// Query interface (with ring buffer)

class _API Query {
 public:
  Query(const std::string& name, size_t N = 256)
      : name(name), N(N), curr(0), data(N, 0.f), exp_avg(0.f), last_val(0.f) {}
  virtual ~Query() {}

  virtual void begin() = 0;
  virtual void end() = 0;

  void put(float val) {
    data[curr] = val;
    curr = (curr + 1) % N;
    const float f = 0.1f;
    exp_avg = f * val + (1 - f) * exp_avg;
    last_val = val;
  }

  float last() { return last_val; }

  float min() const {
    float t = std::numeric_limits<float>::max();
    for (const auto& val : data) t = std::min(t, val);
    return t;
  }

  float max() const {
    float t = std::numeric_limits<float>::min();
    for (const auto& val : data) t = std::max(t, val);
    return t;
  }

  float avg() const {
    float t = 0;
    for (const auto& val : data) t += val;
    return t / N;
  }

  // data
  const std::string name;
  const size_t N;
  size_t curr;
  std::vector<float> data;
  float exp_avg, last_val;
};

// -------------------------------------------------------
// (CPU) TimerQuery (in ms)

class _API TimerQueryImpl : public Query {
 public:
  TimerQueryImpl(const std::string& name, size_t samples = 256);
  virtual ~TimerQueryImpl();

  void begin();
  void end();

  // data
  Timer timer;

  static inline std::string type_to_str() { return "TimerQueryImpl"; }
};

using TimerQuery = reclib::opengl::NamedHandle<TimerQueryImpl>;

// -------------------------------------------------------
// (GPU) TimerQueryGL (in ms)

class _API TimerQueryGLImpl : public Query {
 public:
  TimerQueryGLImpl(const std::string& name, size_t samples = 256);
  virtual ~TimerQueryGLImpl();

  // prevent copies and moves, since GL buffers aren't reference counted
  TimerQueryGLImpl(const TimerQueryGLImpl&) = delete;
  TimerQueryGLImpl& operator=(const TimerQueryGLImpl&) = delete;
  TimerQueryGLImpl& operator=(const TimerQueryGLImpl&&) = delete;

  void begin();
  void end();

  // data
  GLuint query_ids[2][2];
  GLuint64 start_time, stop_time;

  static inline std::string type_to_str() { return "TimerQueryGLImpl"; }
};

using TimerQueryGL = reclib::opengl::NamedHandle<TimerQueryGLImpl>;

// -------------------------------------------------------
// (GPU) PrimitiveQueryGL

class _API PrimitiveQueryGLImpl : public Query {
 public:
  PrimitiveQueryGLImpl(const std::string& name, size_t samples = 256);
  virtual ~PrimitiveQueryGLImpl();

  // prevent copies and moves, since GL buffers aren't reference counted
  PrimitiveQueryGLImpl(const PrimitiveQueryGLImpl&) = delete;
  PrimitiveQueryGLImpl& operator=(const PrimitiveQueryGLImpl&) = delete;
  PrimitiveQueryGLImpl& operator=(const PrimitiveQueryGLImpl&&) = delete;

  void begin();
  void end();

  // data
  GLuint query_ids[2];
  GLuint64 start_time, stop_time;

  static inline std::string type_to_str() { return "PrimitiveQueryGLImpl"; }
};

using PrimitiveQueryGL = reclib::opengl::NamedHandle<PrimitiveQueryGLImpl>;

// -------------------------------------------------------
// (GPU) FragmentQueryGL

class _API FragmentQueryGLImpl : public Query {
 public:
  FragmentQueryGLImpl(const std::string& name, size_t samples = 256);
  virtual ~FragmentQueryGLImpl();

  // prevent copies and moves, since GL buffers aren't reference counted
  FragmentQueryGLImpl(const FragmentQueryGLImpl&) = delete;
  FragmentQueryGLImpl& operator=(const FragmentQueryGLImpl&) = delete;
  FragmentQueryGLImpl& operator=(const FragmentQueryGLImpl&&) = delete;

  void begin();
  void end();

  // data
  GLuint query_ids[2];
  GLuint64 start_time, stop_time;

  static inline std::string type_to_str() { return "FragmentQueryGL"; }
};

using FragmentQueryGL = reclib::opengl::NamedHandle<FragmentQueryGLImpl>;

}  // namespace opengl
}  // namespace reclib

template class _API reclib::opengl::NamedHandle<
    reclib::opengl::FragmentQueryGLImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::PrimitiveQueryGLImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::TimerQueryGLImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::TimerQueryImpl>;  // needed for Windows DLL export

#endif