#ifndef RECLIB_OPENGL_CAMERA_H
#define RECLIB_OPENGL_CAMERA_H

#include <memory>
#include <string>

#include "reclib/camera_parameters.h"
#include "reclib/data_types.h"
#include "reclib/opengl/drawelement.h"
#include "reclib/opengl/mesh_templates.h"
#include "reclib/opengl/named_handle.h"
#include "reclib/opengl/shader.h"

#undef far
#undef near
// ----------------------------------------------------
// Camera
namespace reclib {
namespace opengl {

struct CameraSnapshot {
  vec3 pos, dir, up;               // camera coordinate system
  float fov_degree, near, far;     // perspective projection
  float left, right, bottom, top;  // orthographic projection or skewed frustum
  bool perspective;    // switch between perspective and orthographic
                       // (default: perspective)
  bool skewed;         // switcg between normal perspective and skewed frustum
                       // (default: normal)
  bool fix_up_vector;  // keep up vector fixed to avoid camera drift
  bool fix_proj;  // avoid updating the projection matrix. This is useful in
                  // case you want to use a custom projection matrix
  mat4 view, view_normal,
      proj;  // camera matrices (computed via a call update())
};

class _API CameraImpl {
 public:
  CameraImpl(const std::string& name);
  virtual ~CameraImpl();

  static inline std::string type_to_str() { return "CameraImpl"; }

  void update();

  // move
  void forward(float by);
  void backward(float by);
  void leftward(float by);
  void rightward(float by);
  void upward(float by);
  void downward(float by);
  void translate(float x, float y, float z);

  // rotate
  void yaw(float angle);
  void pitch(float angle);
  void roll(float angle);

  // load/store
  void store(vec3& pos, quat& rot) const;
  void load(const vec3& pos, const quat& rot);
  void load(const vec3& pos, const mat4& rot);

  ExtrinsicParameters to_extrinsics() const;
  void from_extrinsics(const ExtrinsicParameters& extr);
  void from_extrinsics(const mat4& extr);
  void from_intrinsics(const IntrinsicParameters& intr);
  void from_intrinsics(const mat4& intr, int width, int height);

  void store() {
    snapshot.pos = pos;
    snapshot.dir = dir;
    snapshot.up = up;
    snapshot.fov_degree = fov_degree;
    snapshot.near = near;
    snapshot.far = far;
    snapshot.left = left;
    snapshot.right = right;
    snapshot.bottom = bottom;
    snapshot.top = top;
    snapshot.perspective = perspective;
    snapshot.skewed = skewed;
    snapshot.fix_up_vector = fix_up_vector;
    snapshot.fix_proj = fix_proj;
    snapshot.view = view;
    snapshot.view_normal = view_normal;
    snapshot.proj = proj;
  }

  void load_snapshot() {
    pos = snapshot.pos;
    dir = snapshot.dir;
    up = snapshot.up;
    fov_degree = snapshot.fov_degree;
    near = snapshot.near;
    far = snapshot.far;
    left = snapshot.left;
    right = snapshot.right;
    bottom = snapshot.bottom;
    top = snapshot.top;
    perspective = snapshot.perspective;
    skewed = snapshot.skewed;
    fix_up_vector = snapshot.fix_up_vector;
    fix_proj = snapshot.fix_proj;
    view = snapshot.view;
    view_normal = snapshot.view_normal;
    proj = snapshot.proj;
  }

  void draw_frustum(const Shader shader = Shader(),
                    const vec3& color = vec3(1, 1, 0)) const;

  // compute aspect ratio from current viewport
  static float aspect_ratio();

  // default camera keyboard/mouse handler for basic movement
  static float default_camera_movement_speed;
  static bool default_input_handler(double dt_ms);
  static bool default_keyboard_handler(double dt_ms);
  static bool default_mouse_handler(double dt_ms);
  static void set_screenshot_location(fs::path p) { screenshot_location = p; }

  // data
  static fs::path screenshot_location;

  const std::string name;
  vec3 pos, dir, up;               // camera coordinate system
  float fov_degree, near, far;     // perspective projection
  float left, right, bottom, top;  // orthographic projection or skewed frustum
  bool perspective;    // switch between perspective and orthographic
                       // (default: perspective)
  bool skewed;         // switcg between normal perspective and skewed frustum
                       // (default: normal)
  bool fix_up_vector;  // keep up vector fixed to avoid camera drift
  bool fix_proj;  // avoid updating the projection matrix. This is useful in
                  // case you want to use a custom projection matrix
  mat4 view, view_normal,
      proj;  // camera matrices (computed via a call update())

  CameraSnapshot snapshot;

  friend std::ostream& operator<<(std::ostream& os, const CameraImpl& cam) {
    os << "[Camera Parameters]" << std::endl;
    os << "pos: " << cam.pos << std::endl;
    os << "dir: " << cam.dir << std::endl;
    os << "up: " << cam.up << std::endl;

    if (cam.perspective) {
      os << "fov: " << cam.fov_degree << " near: " << cam.near
         << " far: " << cam.far << std::endl;
    } else {
      os << "left: " << cam.left << " right: " << cam.right
         << " bottom: " << cam.bottom << " top: " << cam.top << std::endl;
    }

    return os;
  }
};

using Camera = reclib::opengl::NamedHandle<CameraImpl>;

// TODO move to CameraImpl::current()
Camera current_camera();
void make_camera_current(const Camera& cam);

// converts world space 3d coordinates to 2d pixel coordinates as if they
// would be computed via extrinsic and intrinsic matrices
vec3 convert_3d_to_2d(const Camera& cam, const vec3& world,
                      const ivec2& resolution, bool linearize_depth = true);
// converts pixel coordinates + linear depth
// (computed via extrinsic and intrinsic parameters)
// back to 3d world coordinates
vec3 convert_2d_to_3d(const Camera& cam, const vec3& px_d,
                      const ivec2& resolution);

}  // namespace opengl
}  // namespace reclib

template class _API reclib::opengl::NamedHandle<
    reclib::opengl::CameraImpl>;  // needed for Windows DLL export

#endif
