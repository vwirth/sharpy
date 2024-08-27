#include <reclib/opengl/query.h>

// -------------------------------------------------------
// (CPU) TimerQuery (in ms)

reclib::opengl::TimerQueryImpl::TimerQueryImpl(const std::string& name,
                                               size_t samples)
    : Query(name, samples) {}

reclib::opengl::TimerQueryImpl::~TimerQueryImpl() {}

void reclib::opengl::TimerQueryImpl::begin() { timer.begin(); }

void reclib::opengl::TimerQueryImpl::end() { put(float(timer.look())); }

// -------------------------------------------------------
// (GPU) TimerQueryGL (in ms)

reclib::opengl::TimerQueryGLImpl::TimerQueryGLImpl(const std::string& name,
                                                   size_t samples)
    : Query(name, samples), start_time(0), stop_time(0) {
  glGenQueries(2, query_ids[0]);
  glGenQueries(2, query_ids[1]);
  glQueryCounter(query_ids[1][0], GL_TIMESTAMP);
  glQueryCounter(query_ids[1][1], GL_TIMESTAMP);
}

reclib::opengl::TimerQueryGLImpl::~TimerQueryGLImpl() {
  glDeleteQueries(2, query_ids[0]);
  glDeleteQueries(2, query_ids[1]);
}

void reclib::opengl::TimerQueryGLImpl::begin() {
  glQueryCounter(query_ids[0][0], GL_TIMESTAMP);
}

void reclib::opengl::TimerQueryGLImpl::end() {
  glQueryCounter(query_ids[0][1], GL_TIMESTAMP);
  std::swap(query_ids[0], query_ids[1]);  // switch front/back buffer
  glGetQueryObjectui64v(query_ids[0][0], GL_QUERY_RESULT, &start_time);
  glGetQueryObjectui64v(query_ids[0][1], GL_QUERY_RESULT, &stop_time);
  put(float((stop_time - start_time) / 1000000.0));
}

// -------------------------------------------------------
// (GPU) PrimitiveQueryGL

reclib::opengl::PrimitiveQueryGLImpl::PrimitiveQueryGLImpl(
    const std::string& name, size_t samples)
    : Query(name, samples), start_time(0), stop_time(0) {
  glGenQueries(2, query_ids);
  // avoid error on first run
  glBeginQuery(GL_PRIMITIVES_GENERATED, query_ids[0]);
  glEndQuery(GL_PRIMITIVES_GENERATED);
  glBeginQuery(GL_PRIMITIVES_GENERATED, query_ids[1]);
  glEndQuery(GL_PRIMITIVES_GENERATED);
}

reclib::opengl::PrimitiveQueryGLImpl::~PrimitiveQueryGLImpl() {
  glDeleteQueries(2, query_ids);
}

void reclib::opengl::PrimitiveQueryGLImpl::begin() {
  glBeginQuery(GL_PRIMITIVES_GENERATED, query_ids[0]);
}

void reclib::opengl::PrimitiveQueryGLImpl::end() {
  glEndQuery(GL_PRIMITIVES_GENERATED);
  std::swap(query_ids[0], query_ids[1]);  // switch front/back buffer
  GLuint result;
  glGetQueryObjectuiv(query_ids[0], GL_QUERY_RESULT, &result);
  put(float(result));
}

// -------------------------------------------------------
// (GPU) FragmentQueryGL

reclib::opengl::FragmentQueryGLImpl::FragmentQueryGLImpl(
    const std::string& name, size_t samples)
    : Query(name, samples), start_time(0), stop_time(0) {
  glGenQueries(2, query_ids);
  // avoid error on first run
  glBeginQuery(GL_SAMPLES_PASSED, query_ids[0]);
  glEndQuery(GL_SAMPLES_PASSED);
  glBeginQuery(GL_SAMPLES_PASSED, query_ids[1]);
  glEndQuery(GL_SAMPLES_PASSED);
}

reclib::opengl::FragmentQueryGLImpl::~FragmentQueryGLImpl() {
  glDeleteQueries(2, query_ids);
}

void reclib::opengl::FragmentQueryGLImpl::begin() {
  glBeginQuery(GL_SAMPLES_PASSED, query_ids[0]);
}

void reclib::opengl::FragmentQueryGLImpl::end() {
  glEndQuery(GL_SAMPLES_PASSED);
  std::swap(query_ids[0], query_ids[1]);  // switch front/back buffer
  GLuint result;
  glGetQueryObjectuiv(query_ids[0], GL_QUERY_RESULT, &result);
  put(float(result));
}
