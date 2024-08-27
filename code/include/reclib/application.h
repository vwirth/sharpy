#ifndef RECLIB_APPLICATION_H
#define RECLIB_APPLICATION_H

#include <stdexcept>

#include "reclib/opengl/context.h"
#include "reclib/opengl/gui.h"

namespace reclib {
class GenericApplication {
 protected:
  inline static bool initialized_ = false;

  std::function<void()> loop_func_;

  GenericApplication()
      : loop_func_(loop),
        initialize_func_([]() {}),
        update_func_([]() {}),
        render_func_([]() {}){

        };
  GenericApplication(std::function<void()> initialize_func,
                     std::function<void()> update_func,
                     std::function<void()> render_func)
      : loop_func_(loop),
        initialize_func_(initialize_func),
        update_func_(update_func),
        render_func_(render_func){};
  ~GenericApplication(){};

  // start the application loop
  static void loop() {
    if (!initialized_) {
      throw std::runtime_error(
          "GenericApplication needs to be initialized first");
    }
    instance().update_func_();
    instance().render_func_();
  }

 public:
  // initialize application
  std::function<void()> initialize_func_;
  // update application state
  std::function<void()> update_func_;
  // render graphical output
  std::function<void()> render_func_;

  GenericApplication(const GenericApplication&) = delete;
  GenericApplication& operator=(const GenericApplication&) = delete;
  GenericApplication& operator=(const GenericApplication&&) = delete;

  void set_initialize(std::function<void()> f) { initialize_func_ = f; }
  void set_update(std::function<void()> f) { update_func_ = f; }
  void set_render(std::function<void()> f) { render_func_ = f; }

  static GenericApplication& init() {
    instance().initialize_func_();
    initialized_ = true;
    return instance();
  }

  static GenericApplication& instance() {
    static GenericApplication app;
    return app;
  }

  static void start() { instance().loop_func_(); }
};

class OpenGLApplication {
 protected:
  inline static bool initialized_ = false;

  std::function<void()> loop_func_;

  OpenGLApplication()
      : loop_func_(loop),
        initialize_func_([]() {}),
        update_func_([]() {}),
        render_func_([]() {}){

        };
  OpenGLApplication(std::function<void()> initialize_func,
                    std::function<void()> update_func,
                    std::function<void()> render_func)
      : loop_func_(loop),
        initialize_func_(initialize_func),
        update_func_(update_func),
        render_func_(render_func){};
  ~OpenGLApplication(){};

  // start the application loop
  static void loop() {
    if (!initialized_) {
      throw std::runtime_error(
          "OpenGLApplication needs to be initialized first");
    }
    while (reclib::opengl::Context::running()) {
      // finish frame
      reclib::opengl::Context::swap_buffers();
    }
  }

 public:
  // initialize application
  std::function<void()> initialize_func_;
  // update application state
  std::function<void()> update_func_;
  // render graphical output
  std::function<void()> render_func_;

  OpenGLApplication(const OpenGLApplication&) = delete;
  OpenGLApplication& operator=(const OpenGLApplication&) = delete;
  OpenGLApplication& operator=(const OpenGLApplication&&) = delete;

  void set_initialize(std::function<void()> f) { initialize_func_ = f; }
  void set_update(std::function<void()> f) { update_func_ = f; }
  void set_render(std::function<void()> f) { render_func_ = f; }

  static OpenGLApplication& init(reclib::opengl::ContextParameters params) {
    reclib::opengl::Context::init(params);
    instance().initialize_func_();
    reclib::opengl::MainWindow::init_single(
        [&]() {
          instance().update_func_();
          instance().render_func_();
          return true;
        },
        ImGuiWindowFlags_NoDocking);
    initialized_ = true;

    return instance();
  }
  static OpenGLApplication& instance() {
    static OpenGLApplication app;
    return app;
  }

  static void start() { instance().loop_func_(); }
};
}  // namespace reclib

#endif