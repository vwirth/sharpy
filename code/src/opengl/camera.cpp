#include <imgui/imgui.h>
#include <reclib/internal/filesystem_ops.h>
#include <reclib/opengl/context.h>
#include <reclib/opengl/camera.h>

#include <iostream>

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
// clang-format on

static reclib::opengl::Camera current_cam;

reclib::opengl::Camera reclib::opengl::current_camera() {
  static reclib::opengl::Camera default_cam("default");
  return current_cam ? current_cam : default_cam;
}

void reclib::opengl::make_camera_current(const reclib::opengl::Camera& cam) {
  current_cam = cam;
}

static mat4 get_projection_matrix(float left, float right, float top,
                                  float bottom, float n, float f) {
  mat4 proj = mat4::Zero();
  proj(0, 0) = (2.f * n) / (right - left);
  proj(1, 1) = (2.f * n) / (top - bottom);
  proj(0, 2) = (right + left) / (right - (left));
  proj(1, 2) = (bottom + top) / (top - bottom);
  proj(2, 2) = -(f + n) / (f - n);
  proj(3, 2) = -1.f;
  proj(2, 3) = (-2 * f * n) / (f - n);
  return proj;
}

vec3 reclib::opengl::convert_3d_to_2d(const Camera& cam, const vec3& world,
                                      const ivec2& resolution,
                                      bool linearize_depth) {
  unsigned int width = resolution.x();
  unsigned int height = resolution.y();
  float near_ = cam->near;
  float far_ = cam->far;

  vec4 cam_space = cam->view * world.homogeneous();
  vec4 clip_space = cam->proj * cam_space;

  // x and y between [-1,1], z between [0,1]
  vec3 ndc_space = clip_space.hnormalized();

  if (abs(ndc_space.x()) > 1 || abs(ndc_space.y()) > 1 ||
      abs(ndc_space.z()) > 1) {
    // vertex does not appear in view frustum
    return vec3(-1, -1, 0);
  }

  float d = 0;
  if (linearize_depth) {
    if (0) {
      // linear depth in [-near, -,far]
      float linear_depth =
          -(2 * near_ * far_) / (far_ + near_ - (far_ - near_) * ndc_space.z());
      d = -linear_depth;
      // // linear depth in [0,1]
      // float norm_depth = (linear_depth + near_) / (near_ - far_);
      // // linear depth in [near, near+far]
      // d = norm_depth * far_ + near_;
    } else {
      // thats practically the same as above, since the linear depth corresponds
      // to the distance to the camera in z-direction
      d = -cam_space.z();
    }
  } else {
    d = ndc_space.z();
  }

  // x in [0,1]
  float x = (ndc_space.x() + 1) / 2.f;
  // x in [0, width]
  x = x * width;

  // y in [0,1]
  float y = (ndc_space.y() + 1) / 2.f;
  // y in [0, height], needs to be flipped as opengl defines y-axis reversely
  // to a matrix storage layout
  y = height - (y * height);

  x = floor(x);
  y = floor(y);

  return vec3(x, y, d);
}

vec3 reclib::opengl::convert_2d_to_3d(const Camera& cam, const vec3& px_d,
                                      const ivec2& resolution) {
  unsigned int width = resolution.x();
  unsigned int height = resolution.y();
  float near_ = cam->near;
  float far_ = cam->far;

  if (px_d.z() > far_) {
    return vec3(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max());
  }
  if (px_d.x() < 0 || px_d.x() >= width) {
    return vec3(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max());
  }
  if (px_d.y() < 0 || px_d.y() >= height) {
    return vec3(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max());
  }

  // map from [near,near+far] to [0,1]
  float norm_depth = (px_d.z() - near_) / far_;
  // map from [0,1] to [-near, -far]
  float linear_depth = (norm_depth * (near_ - far_)) - near_;
  // map from [-near, -near] to [-1,1] but in nonlinear manner
  float nonlinear_depth = ((far_ + near_) / (far_ - near_)) +
                          (2 * far_ * near_) / ((far_ - near_) * linear_depth);

  // map from [0, width] to [0,1], add 0.5 to backproject the pixel center and
  // not the upper-left corner
  float x = (px_d.x() + 0.5) / width;
  // map from [0,1] to [-1,1]
  x = (x * 2) - 1;

  // map from [0, height] to [0,1], reverse y direction because of opengl,
  // add 0.5 to backproject the pixel center and
  // not the upper-left corner
  float y = (height + 0.5 - px_d.y()) / height;
  // map from [0,1] to [-1,1]
  y = (y * 2) - 1;

  // if we divide the camera coords by the .w() component then this
  // computation is practically the same as if we would set
  // the ndc coords to (x * nonlinear_depth, y * nonlinear_depth,
  // nonlinear_depth)
  vec3 ndc_coords(x, y, nonlinear_depth);
  vec4 camera_coords = cam->proj.inverse() * ndc_coords.homogeneous();
  camera_coords /= camera_coords.w();
  vec4 world_coords = cam->view.inverse() * camera_coords;
  return world_coords.hnormalized();
}

// ----------------------------------------------------
// reclib::opengl::CameraImpl

float reclib::opengl::CameraImpl::default_camera_movement_speed = 0.0005f;

reclib::opengl::CameraImpl::CameraImpl(const std::string& name)
    : name(name),
      pos(0, 0, 0),
      dir(1, 0, 0),
      up(0, 1, 0),
      fov_degree(70),
      near(0.01f),
      far(1000),
      left(-100),
      right(100),
      bottom(-100),
      top(100),
      perspective(true),
      skewed(false),
      fix_up_vector(true),
      fix_proj(false) {
  update();
  store();
}

void reclib::opengl::CameraImpl::draw_frustum(
    const reclib::opengl::Shader shader, const vec3& color) const {
  reclib::opengl::Shader draw_shader;
  if (!shader.initialized()) {
    if (!reclib::opengl::Shader::valid("frustum_shader")) {
      reclib::opengl::Shader("frustum_shader", "MVP.vs", "color4Uniform.fs");
    }
    draw_shader = reclib::opengl::Shader::find("frustum_shader");
  } else {
    draw_shader = shader;
  }

  if (!Drawelement::valid("Drawelement_Frustum_" + name)) {
    Material mat("Frustum_Material_" + name);
    mat->vec3_map["ambient_color"] = color;
    mat->vec4_map["color"] = vec4(color.x(), color.y(), color.z(), 1);

    Geometry geo = reclib::opengl::FrustumImpl::generate_geometry(
        "geometry_" + name, pos, view, proj);
    Mesh frustum("Frustum_" + name, geo, mat);
    frustum->set_primitive_type(GL_LINES);
    Drawelement("Drawelement_Frustum_" + name, draw_shader, frustum);
  } else {
    Drawelement elem = Drawelement::find("Drawelement_Frustum_" + name);
    elem->mesh->geometry = reclib::opengl::FrustumImpl::generate_geometry(
        "geometry_" + name, pos, view, proj);
    elem->mesh->upload_gpu();
    elem->mesh->material->vec3_map["ambient_color"] = color;
  }
  Drawelement elem = Drawelement::find("Drawelement_Frustum_" + name);
  reclib::opengl::Shader cur_shader = elem->shader;
  elem->shader = draw_shader;
  elem->bind();
  elem->draw();
  elem->unbind();
  elem->shader = cur_shader;
}

reclib::opengl::CameraImpl::~CameraImpl() {
  // TODO debug segfault on reset
  // if (current_camera().ptr.get() == this) // reset
  // current_cam = Camera();
}

void reclib::opengl::CameraImpl::update() {
  dir = normalize(dir);
  up = normalize(up);
  view = lookAt(pos, pos + dir, up);
  view_normal = transpose(inverse(view));
  if (!fix_proj) {
    proj = perspective
               ? (skewed ? get_projection_matrix(left, right, top, bottom, near,
                                                 far)
                         : reclib::perspective(fov_degree * float(M_PI / 180),
                                               aspect_ratio(), near, far))
               : ortho(left, right, bottom, top, near, far);
  }

  if (Drawelement::valid("Drawelement_Frustum_" + name)) {
    Drawelement elem = Drawelement::find("Drawelement_Frustum_" + name);
    elem->mesh->geometry = reclib::opengl::FrustumImpl::generate_geometry(
        "geometry_" + name, pos, view, proj);
  }
}

void reclib::opengl::CameraImpl::translate(float x, float y, float z) {
  pos.x() += x;
  pos.y() += y;
  pos.z() += z;
}

void reclib::opengl::CameraImpl::forward(float by) { pos += by * dir; }
void reclib::opengl::CameraImpl::backward(float by) { pos -= by * dir; }
void reclib::opengl::CameraImpl::leftward(float by) {
  pos -= by * normalize(cross(dir, up));
}
void reclib::opengl::CameraImpl::rightward(float by) {
  pos += by * normalize(cross(dir, up));
}
void reclib::opengl::CameraImpl::upward(float by) {
  pos += by * normalize(cross(cross(dir, up), dir));
}
void reclib::opengl::CameraImpl::downward(float by) {
  pos -= by * normalize(cross(cross(dir, up), dir));
}

void reclib::opengl::CameraImpl::yaw(float angle) {
  dir = normalize(rotate(dir, angle * float(M_PI) / 180.f, up));
}
void reclib::opengl::CameraImpl::pitch(float angle) {
  dir = normalize(
      rotate(dir, angle * float(M_PI) / 180.f, normalize(cross(dir, up))));
  if (!fix_up_vector) up = normalize(cross(cross(dir, up), dir));
}
void reclib::opengl::CameraImpl::roll(float angle) {
  up = normalize(rotate(up, angle * float(M_PI) / 180.f, dir));
}

void reclib::opengl::CameraImpl::store(vec3& pos, quat& rot) const {
  pos = this->pos;
  rot = quat_cast(view);
}

void reclib::opengl::CameraImpl::load(const vec3& pos, const quat& rot) {
  this->pos = pos;
  this->view = mat4::Zero();
  this->view.block<3, 3>(0, 0) = rot.matrix();
  this->dir = -vec3(view(2, 0), view(2, 1), view(2, 2));
  this->up = vec3(view(1, 0), view(1, 1), view(1, 2));
}

void reclib::opengl::CameraImpl::load(const vec3& pos, const mat4& rot) {
  this->pos = pos;
  this->view = mat4::Zero();
  this->view.block<3, 3>(0, 0) = rot.block<3, 3>(0, 0);
  this->dir = -vec3(view(2, 0), view(2, 1), view(2, 2));
  this->up = vec3(view(1, 0), view(1, 1), view(1, 2));
}

reclib::ExtrinsicParameters reclib::opengl::CameraImpl::to_extrinsics() const {
  reclib::ExtrinsicParameters extr;
  extr.up_ = this->up;
  extr.dir_ = this->dir;
  extr.eye_ = this->pos;
  extr.right_ = cross(this->dir, this->up);
  return extr;
}

void reclib::opengl::CameraImpl::from_intrinsics(
    const IntrinsicParameters& intr) {
  proj = reclib::vision2graphics(intr.Matrix(), 0.001, 1000, intr.image_width_,
                                 intr.image_height_);
  fix_proj = true;
}
void reclib::opengl::CameraImpl::from_intrinsics(const mat4& intr, int width,
                                                 int height) {
  proj = reclib::vision2graphics(intr, 0.001, 1000, width, height);
  fix_proj = true;
}

void reclib::opengl::CameraImpl::from_extrinsics(
    const reclib::ExtrinsicParameters& extr) {
  this->up = extr.up_;
  this->dir = extr.dir_;
  this->pos = extr.eye_;
}

void reclib::opengl::CameraImpl::from_extrinsics(const mat4& extr) {
  this->up = extr.col(1).head<3>().normalized();
  this->up.y() *= -1;
  // this->up.x() *= -1;
  this->dir = extr.col(2).head<3>().normalized();
  this->dir.y() *= -1;
  // this->dir.x() *= -1;
  this->pos = extr.col(3).head<3>();
  this->pos.y() *= -1;
  // this->pos.x() *= -1;
}

float reclib::opengl::CameraImpl::aspect_ratio() {
  GLint xywh[4];
  glGetIntegerv(GL_VIEWPORT, xywh);
  return xywh[2] / (float)xywh[3];
}

bool reclib::opengl::CameraImpl::default_input_handler(double dt_ms) {
  static bool imgui_last_click = false;
  static bool last_click = false;
  static bool current_click = false;

  bool moved = false;
  bool moved_mouse = false;
  if (!ImGui::GetIO().WantCaptureKeyboard && !ImGui::GetIO().WantCaptureMouse) {
    moved = default_keyboard_handler(dt_ms);
  }
  current_click =
      reclib::opengl::Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT) ||
      reclib::opengl::Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT);
  if (!last_click && current_click) {
    // std::cout << "Clicked! Capture: " << ImGui::GetIO().WantCaptureMouse
    //           << std::endl;
    // imgui_last_click = ImGui::GetIO().WantCaptureMouse;
  }

  if (!ImGui::GetIO().WantCaptureMouse) {
    moved_mouse = default_mouse_handler(dt_ms);
  }
  last_click = current_click;

  moved = moved || moved_mouse;

  return moved;
}

static bool did_screenshot = false;
fs::path reclib::opengl::CameraImpl::screenshot_location = "";

bool reclib::opengl::CameraImpl::default_keyboard_handler(double dt_ms) {
  bool moved = false;
  // keyboard
  if (Context::key_pressed(GLFW_KEY_W)) {
    current_camera()->forward(float(dt_ms * default_camera_movement_speed));
    moved = true;
  }
  if (Context::key_pressed(GLFW_KEY_S)) {
    current_camera()->backward(float(dt_ms * default_camera_movement_speed));
    moved = true;
  }
  if (Context::key_pressed(GLFW_KEY_A)) {
    current_camera()->leftward(float(dt_ms * default_camera_movement_speed));
    moved = true;
  }
  if (Context::key_pressed(GLFW_KEY_D)) {
    current_camera()->rightward(float(dt_ms * default_camera_movement_speed));
    moved = true;
  }
  if (Context::key_pressed(GLFW_KEY_R)) {
    current_camera()->upward(float(dt_ms * default_camera_movement_speed));
    moved = true;
  }
  if (Context::key_pressed(GLFW_KEY_F)) {
    current_camera()->downward(float(dt_ms * default_camera_movement_speed));
    moved = true;
  }
  if (Context::key_pressed(GLFW_KEY_Q)) {
    current_camera()->roll(float(dt_ms * -0.1));
    moved = true;
  }
  if (Context::key_pressed(GLFW_KEY_E)) {
    current_camera()->roll(float(dt_ms * 0.1));
    moved = true;
  }
  if (reclib::opengl::Context::key_pressed(GLFW_KEY_F10) && !did_screenshot) {
    if (!fs::exists(screenshot_location)) {
      reclib::utils::CreateDirectories(screenshot_location);
    }
    for (unsigned int i = 0; i < 100; i++) {
      std::stringstream str;
      str << "screenshot" << i << ".png";
      fs::path p = screenshot_location / fs::path(str.str());
      if (!fs::exists(p.string())) {
        reclib::opengl::Context::screenshot(p.string());
        did_screenshot = true;
        break;
      }
    }
  } else if (!reclib::opengl::Context::key_pressed(GLFW_KEY_F10)) {
    did_screenshot = false;
  }

  return moved;
}

bool reclib::opengl::CameraImpl::default_mouse_handler(double dt_ms) {
  bool moved = false;
  // mouse
  static float rot_speed = 0.05f;
  static vec2 last_pos(-1, -1);
  const vec2 curr_pos = Context::mouse_pos();
  if (last_pos == vec2(-1, -1)) last_pos = curr_pos;
  const vec2 diff = last_pos - curr_pos;
  if (Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT)) {
    if (length(diff) > 0.01) {
      current_camera()->pitch(diff.y() * rot_speed);
      current_camera()->yaw(diff.x() * rot_speed);
      moved = true;
    }
  }
  if (Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT)) {
    // pan mouse
    if (length(diff) > 0.01) {
      current_camera()->translate(diff.x() * rot_speed * 0.5,
                                  diff.y() * rot_speed * 0.5, 0);
      moved = true;
    }
  }
  if (Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_MIDDLE)) {
    // pan mouse
    if (length(diff) > 0.01) {
      if (diff.y() > 0) current_camera()->forward(diff.y() * rot_speed);
      if (diff.y() < 0) current_camera()->backward(-diff.y() * rot_speed);
      moved = true;
    }
  }
  last_pos = curr_pos;

  return moved;
}