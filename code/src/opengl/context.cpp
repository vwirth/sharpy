#include <reclib/internal/debug.h>
#include <reclib/opengl/camera.h>
#include <reclib/opengl/context.h>
#include <reclib/opengl/gui.h>
#include <reclib/opengl/texture.h>

#include <filesystem>
#include <iostream>

#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "imgui/imgui.h"
#include "stb/stb_image_write.h"

const fs::path reclib::opengl::IMGUI_DEFAULT_CONFIG =
    (fs::path(RECLIB_LIBRARY_SOURCE_PATH) / fs::path("include") /
     fs::path("reclib") / fs::path("opengl") / fs::path("imgui_default.ini"));

// -------------------------------------------
// helper funcs

static void glfw_error_func(int error, const char* description) {
  fprintf(stderr, "GLFW: Error %i: %s\n", error, description);
}

static void (*user_keyboard_callback)(int key, int scancode, int action,
                                      int mods) = 0;
static void (*user_mouse_callback)(double xpos, double ypos) = 0;
static void (*user_mouse_button_callback)(int button, int action, int mods) = 0;
static void (*user_mouse_scroll_callback)(double xoffset, double yoffset) = 0;
static void (*user_resize_callback)(int w, int h) = 0;

static void glfw_key_callback(GLFWwindow* window, int key, int scancode,
                              int action, int mods) {
  if (key == GLFW_KEY_F1 && action == GLFW_PRESS)
    reclib::opengl::Context::show_gui = !reclib::opengl::Context::show_gui;
  if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
    reclib::opengl::Context::show_gui = !reclib::opengl::Context::show_gui;

  if (ImGui::GetIO().WantCaptureKeyboard) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    return;
  }
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, 1);
  if (user_keyboard_callback)
    user_keyboard_callback(key, scancode, action, mods);
}

static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  if (user_mouse_callback) user_mouse_callback(xpos, ypos);
}

static void glfw_mouse_button_callback(GLFWwindow* window, int button,
                                       int action, int mods) {
  if (ImGui::GetIO().WantCaptureMouse) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    return;
  }
  if (user_mouse_button_callback)
    user_mouse_button_callback(button, action, mods);
}

static void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset,
                                       double yoffset) {
  if (ImGui::GetIO().WantCaptureMouse) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    return;
  }

  reclib::opengl::CameraImpl::default_camera_movement_speed +=
      reclib::opengl::CameraImpl::default_camera_movement_speed *
      float(yoffset * 0.1);
  reclib::opengl::CameraImpl::default_camera_movement_speed = std::max(
      0.00001f, reclib::opengl::CameraImpl::default_camera_movement_speed);
  if (user_mouse_scroll_callback) user_mouse_scroll_callback(xoffset, yoffset);
}

static void glfw_resize_callback(GLFWwindow* window, int w, int h) {
  reclib::opengl::Context::resize(w, h);
  if (user_resize_callback) user_resize_callback(w, h);
}

static void glfw_char_callback(GLFWwindow* window, unsigned int c) {
  ImGui_ImplGlfw_CharCallback(window, c);
}

// -------------------------------------------
// Context

bool reclib::opengl::Context::show_gui = false;
static reclib::opengl::ContextParameters parameters;
bool reclib::opengl::Context::initialized = false;

reclib::opengl::Context::Context() {
  if (!glfwInit()) throw std::runtime_error("glfwInit failed!");
  glfwSetErrorCallback(glfw_error_func);

  // some GL context settings
  if (parameters.gl_major > 0)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, parameters.gl_major);
  if (parameters.gl_minor > 0)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, parameters.gl_minor);
  glfwWindowHint(GLFW_RESIZABLE, parameters.resizable);
  glfwWindowHint(GLFW_VISIBLE, parameters.visible);
  glfwWindowHint(GLFW_DECORATED, parameters.decorated);
  glfwWindowHint(GLFW_FLOATING, parameters.floating);
  glfwWindowHint(GLFW_MAXIMIZED, parameters.maximised);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, parameters.gl_debug_context);
  glfwWindowHint(GLFW_DEPTH_BITS, parameters.depth_buffer_bytes * 8);

  // create window and context
  glfw_window = glfwCreateWindow(parameters.width, parameters.height,
                                 parameters.title.c_str(), 0, 0);
  if (!glfw_window) {
    glfwTerminate();
    throw std::runtime_error("glfwCreateContext failed!");
  }
  glfwMakeContextCurrent(glfw_window);
  glfwSwapInterval(parameters.swap_interval);

  if (parameters.resizable == GLFW_FALSE) {
    glfwSetWindowSizeLimits(glfw_window, parameters.width, parameters.height,
                            parameters.width, parameters.height);
  }

  glewExperimental = GL_TRUE;
  const GLenum err = glewInit();
  if (err != GLEW_OK) {
    glfwDestroyWindow(glfw_window);
    glfwTerminate();
    throw std::runtime_error(std::string("GLEWInit failed: ") +
                             (const char*)glewGetErrorString(err));
  }

  // output configuration
  std::cout << "GLFW: " << glfwGetVersionString() << std::endl;
  std::cout << "OpenGL: " << glGetString(GL_VERSION) << ", "
            << glGetString(GL_RENDERER) << std::endl;
  std::cout << "GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION)
            << std::endl;

  // enable debugging output
  debug::enable_stack_trace_on_crash();
  debug::enable_gl_debug_output();

  // setup user ptr
  glfwSetWindowUserPointer(glfw_window, this);

  // install callbacks
  glfwSetKeyCallback(glfw_window, glfw_key_callback);
  glfwSetCursorPosCallback(glfw_window, glfw_mouse_callback);
  glfwSetMouseButtonCallback(glfw_window, glfw_mouse_button_callback);
  glfwSetScrollCallback(glfw_window, glfw_mouse_scroll_callback);
  glfwSetFramebufferSizeCallback(glfw_window, glfw_resize_callback);
  glfwSetCharCallback(glfw_window, glfw_char_callback);

  // set input mode
  glfwSetInputMode(glfw_window, GLFW_STICKY_KEYS, 1);
  glfwSetInputMode(glfw_window, GLFW_STICKY_MOUSE_BUTTONS, 1);

  // init imgui
  ImGui::CreateContext();

  ImGui_ImplGlfw_InitForOpenGL(glfw_window, false);
  ImGui_ImplOpenGL3_Init("#version 130");

  // ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  // load custom font?
  if (fs::exists(parameters.font_ttf_filename)) {
    ImFontConfig config;
    config.OversampleH = 3;
    config.OversampleV = 3;
    std::cout << "Loading: " << parameters.font_ttf_filename << "..."
              << std::endl;
    ImGui::GetIO().FontDefault = ImGui::GetIO().Fonts->AddFontFromFileTTF(
        parameters.font_ttf_filename.string().c_str(),
        float(parameters.font_size_pixels), &config);
  }
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigDockingWithShift = true;
  io.IniFilename = NULL;
  io.WantSaveIniSettings = false;
  // load default settings
  fs::path current_path = fs::current_path();
  if (fs::exists(current_path / fs::path("imgui.ini"))) {
    std::cout << "Loading ImGui from current path: "
              << current_path / fs::path("imgui.ini") << std::endl;
    ImGui::LoadIniSettingsFromDisk(
        (current_path / fs::path("imgui.ini")).c_str());
  } else {
    ImGui::LoadIniSettingsFromDisk(IMGUI_DEFAULT_CONFIG.c_str());
  }

  reclib::opengl::set_imgui_theme();

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  reclib::opengl::create_default_windows(parameters.imgui_left_dock,
                                         parameters.imgui_bottom_dock,
                                         parameters.imgui_right_dock);

  // set some sane GL defaults
  glEnable(GL_DEPTH_TEST);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glClearColor(0.5, 0.5, 0.5, 1);
  glClearDepth(1);

  // setup timer
  last_t = curr_t = glfwGetTime();
  cpu_timer = TimerQuery("CPU-time");
  frame_timer = TimerQuery("Frame-time");
  gpu_timer = TimerQueryGL("GPU-time");
  prim_count = PrimitiveQueryGL("#Primitives");
  frag_count = FragmentQueryGL("#Fragments");
  cpu_timer->begin();
  frame_timer->begin();
  gpu_timer->begin();
  prim_count->begin();
  frag_count->begin();
}

reclib::opengl::Context::~Context() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  glfwDestroyWindow(glfw_window);
  glfwTerminate();
}

reclib::opengl::Context& reclib::opengl::Context::init(
    const ContextParameters& params) {
  parameters = params;
  reclib::opengl::Context::initialized = true;
  return instance();
}

reclib::opengl::Context& reclib::opengl::Context::instance() {
  static Context ctx;
  return ctx;
}

bool reclib::opengl::Context::running() {
  return !glfwWindowShouldClose(instance().glfw_window);
}

void reclib::opengl::Context::render_gui() {}

void reclib::opengl::Context::swap_buffers() {
  if (reclib::opengl::Context::show_gui) gui_draw_toggle();
  gui_draw();
  ImGui::Render();
  ImGui::UpdatePlatformWindows();
  ImGui::RenderPlatformWindowsDefault();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  instance().cpu_timer->end();
  instance().gpu_timer->end();
  instance().prim_count->end();
  instance().frag_count->end();
  glfwSwapBuffers(instance().glfw_window);
  instance().frame_timer->end();
  instance().frame_timer->begin();
  instance().cpu_timer->begin();
  instance().gpu_timer->begin();
  instance().prim_count->begin();
  instance().frag_count->begin();
  instance().last_t = instance().curr_t;
  instance().curr_t = glfwGetTime() * 1000;  // s to ms
  glfwPollEvents();
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

double reclib::opengl::Context::frame_time() {
  return instance().curr_t - instance().last_t;
}

std::vector<uint8_t> reclib::opengl::Context::screenshot(const fs::path& path) {
  stbi_flip_vertically_on_write(0);
  const ivec2 size = resolution();
  std::vector<uint8_t> pixels(size.x() * size.y() * 3);
  // glReadPixels can align the first pixel in each row at 1-, 2-, 4- and 8-byte
  // boundaries. We have allocated the exact size needed for the image so we
  // have to use 1-byte alignment (otherwise glReadPixels would write out of
  // bounds)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, size.x(), size.y(), GL_RGB, GL_UNSIGNED_BYTE,
               pixels.data());
  Texture2DImpl::flip_horizontally(pixels.data(), size.x(), size.y(), 3, 1);
  // check file extension
  if (path.extension() == ".png") {
    stbi_write_png(path.string().c_str(), size.x(), size.y(), 3, pixels.data(),
                   0);
    std::cout << path << " written." << std::endl;
  } else if (path.extension() == ".jpg") {
    stbi_write_jpg(path.string().c_str(), size.x(), size.y(), 3, pixels.data(),
                   0);
    std::cout << path << " written." << std::endl;
  } else {
    std::cerr
        << "WARN: reclib::opengl::Context::screenshot: unknown extension, "
           "changing to .png!"
        << std::endl;
    stbi_write_png(fs::path(path).replace_extension(".png").string().c_str(),
                   size.x(), size.y(), 3, pixels.data(), 0);
    std::cout << fs::path(path).replace_extension(".png") << " written."
              << std::endl;
  }
  return pixels;
}

void reclib::opengl::Context::show() { glfwShowWindow(instance().glfw_window); }

void reclib::opengl::Context::hide() { glfwHideWindow(instance().glfw_window); }

ivec2 reclib::opengl::Context::resolution() {
  int w, h;
  glfwGetFramebufferSize(instance().glfw_window, &w, &h);
  return ivec2(w, h);
}

void reclib::opengl::Context::resize(int w, int h) {
  glfwSetWindowSize(instance().glfw_window, w, h);
  glViewport(0, 0, w, h);
}

void reclib::opengl::Context::set_title(const std::string& name) {
  glfwSetWindowTitle(instance().glfw_window, name.c_str());
}

void reclib::opengl::Context::set_swap_interval(uint32_t interval) {
  glfwSwapInterval(interval);
}

void reclib::opengl::Context::capture_mouse(bool on) {
  glfwSetInputMode(instance().glfw_window, GLFW_CURSOR,
                   on ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

vec2 reclib::opengl::Context::mouse_pos() {
  double xpos, ypos;
  glfwGetCursorPos(instance().glfw_window, &xpos, &ypos);
  return vec2(xpos, ypos);
}

bool reclib::opengl::Context::mouse_button_pressed(int button) {
  return glfwGetMouseButton(instance().glfw_window, button) == GLFW_PRESS;
}

bool reclib::opengl::Context::key_pressed(int key) {
  return glfwGetKey(instance().glfw_window, key) == GLFW_PRESS;
}

void reclib::opengl::Context::set_keyboard_callback(
    void (*fn)(int key, int scancode, int action, int mods)) {
  user_keyboard_callback = fn;
}

void reclib::opengl::Context::set_mouse_callback(void (*fn)(double xpos,
                                                            double ypos)) {
  user_mouse_callback = fn;
}

void reclib::opengl::Context::set_mouse_button_callback(void (*fn)(int button,
                                                                   int action,
                                                                   int mods)) {
  user_mouse_button_callback = fn;
}

void reclib::opengl::Context::set_mouse_scroll_callback(
    void (*fn)(double xoffset, double yoffset)) {
  user_mouse_scroll_callback = fn;
}

void reclib::opengl::Context::set_resize_callback(void (*fn)(int w, int h)) {
  user_resize_callback = fn;
}
