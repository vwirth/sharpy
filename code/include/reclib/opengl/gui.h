#ifndef RECLIB_OPENGL_GUI
#define RECLIB_OPENGL_GUI

#include <functional>
#include <string>
#include <type_traits>

#include "imgui/imgui.h"
#include "reclib/configuration.h"
#include "reclib/opengl/camera.h"
#include "reclib/opengl/context.h"
#include "reclib/opengl/drawelement.h"
#include "reclib/opengl/framebuffer.h"
#include "reclib/opengl/geometry.h"
#include "reclib/opengl/material.h"
#include "reclib/opengl/mesh.h"
#include "reclib/opengl/shader.h"
#include "reclib/opengl/texture.h"

namespace reclib {
namespace opengl {

class ImguiWindowBase;

extern std::map<std::string, unsigned int> imgui_window_name_to_id;
// only visible once you 'show_gui' is set to true in OpenGL context
extern std::map<unsigned int, std::shared_ptr<ImguiWindowBase>>
    imgui_dockspace_windows;
extern std::map<unsigned int, std::shared_ptr<ImguiWindowBase>>
    imgui_toggle_windows;
extern std::map<unsigned int, std::shared_ptr<ImguiWindowBase>>
    imgui_visible_windows;

class DockspaceWindowBase;

template <typename T>
class ImguiWindowBaseWrapper {
  inline static unsigned int global_id = 0;

 public:
  ImguiWindowBaseWrapper(){};
  template <class... Args>
  ImguiWindowBaseWrapper(const std::string& name, bool visible, Args&&... args)
      : ptr(std::make_shared<T>(name, global_id, args...)) {
    if (std::is_base_of<DockspaceWindowBase, T>()) {
      imgui_dockspace_windows[global_id] = ptr;
    } else if (visible) {
      imgui_visible_windows[global_id] = ptr;
    } else {
      imgui_toggle_windows[global_id] = ptr;
    }
    imgui_window_name_to_id[name] = global_id;
    global_id++;
  }
  inline T* operator->() { return ptr.operator->(); }
  inline const T* operator->() const { return ptr.operator->(); }
  std::shared_ptr<T> ptr;
};

class ImguiWindowBase {
 protected:
  unsigned int internal_id_;
  ImVec2 pos_;
  ImVec2 size_;

  ImVec2 backup_pos_;
  ImVec2 backup_size_;

 public:
  bool initialized_;
  bool first_draw_;
  bool shrinked_;
  bool updated_size_pos_;
  bool show_window_;
  std::string name_;

  std::function<void(ImguiWindowBase&)> user_update_func_;

  ImGuiWindowFlags flags_;

  std::vector<std::shared_ptr<ImguiWindowBase>> child_windows_;

  ImGuiID id_;
  std::string dock_name_;
  bool is_docked_;
  bool is_hovered_;
  bool border_;

  ImguiWindowBase()
      : internal_id_(0),
        pos_(-1, -1),
        size_(-1, -1),
        backup_pos_(-1, -1),
        backup_size_(-1, -1),
        initialized_(false),
        first_draw_(true),
        shrinked_(false),
        updated_size_pos_(false),
        show_window_(true),
        name_("default"),
        user_update_func_([](ImguiWindowBase& window) {}),
        flags_(0),
        id_(ImGui::GetID("default")),
        is_docked_(false),
        is_hovered_(false),
        border_(true) {
    flags_ = flags_ | ImGuiWindowFlags_HorizontalScrollbar;
  };

  ImguiWindowBase(const std::string name, unsigned int id,
                  ImGuiWindowFlags flags = 0)
      : internal_id_(id),
        pos_(-1, -1),
        size_(-1, -1),
        backup_pos_(-1, -1),
        backup_size_(-1, -1),
        initialized_(false),
        first_draw_(true),
        shrinked_(false),
        updated_size_pos_(false),
        show_window_(true),
        name_(name),
        user_update_func_([](ImguiWindowBase& window) {}),
        flags_(flags),
        id_(ImGui::GetID(name.c_str())),
        is_docked_(false),
        is_hovered_(false),
        border_(true) {
    flags_ = flags_ | ImGuiWindowFlags_HorizontalScrollbar;
  };

  ImguiWindowBase(const std::string name, unsigned int id,
                  ImVec2 pos = ImVec2(-1, -1), ImVec2 size = ImVec2(0, 0),
                  ImGuiWindowFlags flags = 0)
      : internal_id_(id),
        pos_(pos),
        size_(size),
        backup_pos_(-1, -1),
        backup_size_(-1, -1),
        initialized_(false),
        first_draw_(true),
        shrinked_(false),
        updated_size_pos_(false),
        show_window_(true),
        name_(name),
        user_update_func_([](ImguiWindowBase& window) {}),
        flags_(flags),
        id_(ImGui::GetID(name.c_str())),
        is_docked_(false),
        is_hovered_(false),
        border_(true) {
    flags_ = flags_ | ImGuiWindowFlags_HorizontalScrollbar;
  };

  ImguiWindowBase(const std::string name, unsigned int id,
                  std::function<void(ImguiWindowBase&)> user_update_func,
                  ImGuiWindowFlags flags = 0)
      : internal_id_(id),
        pos_(-1, -1),
        size_(-1, -1),
        backup_pos_(-1, -1),
        backup_size_(-1, -1),
        initialized_(false),
        first_draw_(true),
        shrinked_(false),
        updated_size_pos_(false),
        show_window_(true),
        name_(name),
        user_update_func_(user_update_func),
        flags_(flags),
        id_(ImGui::GetID(name.c_str())),
        is_docked_(false),
        is_hovered_(false),
        border_(true) {
    flags_ = flags_ | ImGuiWindowFlags_HorizontalScrollbar;
  };

  ImguiWindowBase(const std::string name, unsigned int id,
                  std::function<void(ImguiWindowBase&)> user_update_func,
                  ImVec2 pos = ImVec2(-1, -1), ImVec2 size = ImVec2(0, 0),
                  ImGuiWindowFlags flags = 0)
      : internal_id_(id),
        pos_(pos),
        size_(size),
        backup_pos_(-1, -1),
        backup_size_(-1, -1),
        initialized_(false),
        first_draw_(true),
        shrinked_(false),
        updated_size_pos_(false),
        show_window_(true),
        name_(name),
        user_update_func_(user_update_func),
        flags_(flags),
        id_(ImGui::GetID(name.c_str())),
        is_docked_(false),
        is_hovered_(false),
        border_(true) {
    flags_ = flags_ | ImGuiWindowFlags_HorizontalScrollbar;
  };

  ImVec2 get_pos() { return pos_; }
  void set_pos(ImVec2 val) {
    // pos_.x = int(val.x);
    // pos_.y = int(val.y);
    pos_ = val;
    updated_size_pos_ = true;
  }
  ImVec2 get_size() { return size_; }
  void set_size(ImVec2 val) {
    size_ = val;
    // size_.x = int(val.x);
    // size_.y = int(val.x);
    updated_size_pos_ = true;
  }

  virtual void create();
  void draw();
  virtual void pre();
  virtual void draw_();
  virtual void draw_always();
  virtual void post();
  virtual void destroy();

  virtual ~ImguiWindowBase() = 0;
};
template class ImguiWindowBaseWrapper<ImguiWindowBase>;
typedef ImguiWindowBaseWrapper<ImguiWindowBase> ImguiWindow;

class OptionWindowBase : public ImguiWindowBase {
 private:
  ImGuiWindowFlags backup_flags_;

 public:
  OptionWindowBase() : ImguiWindowBase(){};

  OptionWindowBase(const std::string name, unsigned int id,
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, flags){

        };

  OptionWindowBase(const std::string name, unsigned int id,
                   ImVec2 pos = ImVec2(-1, -1), ImVec2 size = ImVec2(0, 0),
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, pos, size, flags){

        };

  OptionWindowBase(const std::string name, unsigned int id,
                   std::function<void(ImguiWindowBase&)> user_update_func,
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, user_update_func, flags){

        };

  OptionWindowBase(const std::string name, unsigned int id,
                   std::function<void(ImguiWindowBase&)> user_update_func,
                   ImVec2 pos, ImVec2 size, ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, user_update_func, pos, size, flags){

        };

  OptionWindowBase(const std::string name, unsigned int id,
                   std::function<void()> user_update_func,
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [user_update_func](ImguiWindowBase& window) { user_update_func(); },
            flags){

        };

  OptionWindowBase(const std::string name, unsigned int id,
                   std::function<void()> user_update_func, ImVec2 pos,
                   ImVec2 size, ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [user_update_func](ImguiWindowBase& window) { user_update_func(); },
            pos, size, flags){

        };

  ~OptionWindowBase(){};
  void create() override;
  void pre() override;
  void draw_() override;
  void post() override;
  void destroy() override;
};
template class ImguiWindowBaseWrapper<OptionWindowBase>;
typedef ImguiWindowBaseWrapper<OptionWindowBase> OptionWindow;

class DockspaceWindowBase : public ImguiWindowBase {
 public:
  ImGuiDockNodeFlags dockspace_flags_;
  ImGuiID dockspace_id_;

  DockspaceWindowBase() : ImguiWindowBase(){};

  DockspaceWindowBase(const std::string name, unsigned int id,
                      ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, flags),
        dockspace_flags_(ImGuiDockNodeFlags_None){

        };

  DockspaceWindowBase(const std::string name, unsigned int id,
                      ImVec2 pos = ImVec2(-1, -1), ImVec2 size = ImVec2(0, 0),
                      ImGuiWindowFlags flags = 0,
                      ImGuiDockNodeFlags dockspace_flags = 0)
      : ImguiWindowBase(name, id, pos, size, flags),
        dockspace_flags_(dockspace_flags){

        };

  DockspaceWindowBase(const std::string name, unsigned int id,
                      std::function<void(ImguiWindowBase&)> user_update_func,
                      ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, user_update_func, flags),
        dockspace_flags_(ImGuiDockNodeFlags_None){

        };

  DockspaceWindowBase(const std::string name, unsigned int id,
                      std::function<void(ImguiWindowBase&)> user_update_func,
                      ImVec2 pos = ImVec2(-1, -1), ImVec2 size = ImVec2(0, 0),
                      ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, user_update_func, pos, size, flags),
        dockspace_flags_(ImGuiDockNodeFlags_None){

        };

  DockspaceWindowBase(const std::string name, unsigned int id,
                      std::function<void()> user_update_func,
                      ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [user_update_func](ImguiWindowBase& window) { user_update_func(); },
            flags),
        dockspace_flags_(ImGuiDockNodeFlags_None){

        };

  DockspaceWindowBase(const std::string name, unsigned int id,
                      std::function<void()> user_update_func,
                      ImVec2 pos = ImVec2(-1, -1), ImVec2 size = ImVec2(0, 0),
                      ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [user_update_func](ImguiWindowBase& window) { user_update_func(); },
            pos, size, flags),
        dockspace_flags_(ImGuiDockNodeFlags_None){

        };

  ~DockspaceWindowBase(){};
  void create() override;
  void pre() override;
  void draw_always() override;
  void draw_() override;
  void post() override;
  void destroy() override;
};
template class ImguiWindowBaseWrapper<DockspaceWindowBase>;
typedef ImguiWindowBaseWrapper<DockspaceWindowBase> DockspaceWindow;

struct CameraHandlerState {
  bool clicked_;
  unsigned int click_id_;
};

class RenderWindowBase : public ImguiWindowBase {
 protected:
  ImGuiWindowFlags backup_flags_;
  reclib::opengl::Framebuffer fbo_;
  reclib::opengl::Camera backup_cam_;
  ImVec2 content_size_;

  bool gui_camera_handler(double dt_ms);
  static bool default_camera_handler(double dt_ms, CameraHandlerState state,
                                     const RenderWindowBase& window);

 public:
  std::function<bool(double, CameraHandlerState, const RenderWindowBase&)>
      camera_handler_;
  bool resizable_;
  bool preserve_aspect_ratio_;
  bool last_updated_;
  /** this is set to true in case the user update function never has returned
  'true' so far */
  bool first_render_;
  ivec2 image_size_;
  reclib::opengl::Camera cam_;

  RenderWindowBase() : ImguiWindowBase(){};

  RenderWindowBase(const std::string name, unsigned int id,
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id,
                        flags | ImGuiWindowFlags_NoScrollbar |
                            ImGuiWindowFlags_NoScrollWithMouse),
        camera_handler_(default_camera_handler),
        resizable_(true),
        preserve_aspect_ratio_(false),
        image_size_(0, 0) {
    border_ = false;
  };

  RenderWindowBase(const std::string name, unsigned int id,
                   ImVec2 pos = ImVec2(-1, -1), ImVec2 size = ImVec2(0, 0),
                   ivec2 image_size = ivec2(0, 0), ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(name, id, pos, size,
                        flags | ImGuiWindowFlags_NoScrollbar |
                            ImGuiWindowFlags_NoScrollWithMouse),
        camera_handler_(default_camera_handler),
        resizable_(true),
        preserve_aspect_ratio_(false),
        last_updated_(false),
        first_render_(true),
        image_size_(image_size) {
    border_ = false;
  };

  RenderWindowBase(const std::string name, unsigned int id,
                   std::function<bool(ImguiWindowBase&)> user_update_func,
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [this, user_update_func](ImguiWindowBase& window) {
              last_updated_ = user_update_func(window);
              if (last_updated_) {
                first_render_ = false;
              }
            },
            flags | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse),
        camera_handler_(default_camera_handler),
        resizable_(true),
        preserve_aspect_ratio_(false),
        last_updated_(false),
        first_render_(true),
        image_size_(0, 0) {
    border_ = false;
  };

  RenderWindowBase(const std::string name, unsigned int id,
                   std::function<bool(ImguiWindowBase&)> user_update_func,
                   ImVec2 pos, ImVec2 size, ivec2 image_size = ivec2(0, 0),
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [this, user_update_func](ImguiWindowBase& window) {
              last_updated_ = user_update_func(window);
              if (last_updated_) {
                first_render_ = false;
              }
            },
            pos, size,
            flags | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse),
        camera_handler_(default_camera_handler),
        resizable_(true),
        preserve_aspect_ratio_(false),
        last_updated_(false),
        first_render_(true),
        image_size_(image_size) {
    border_ = false;
  };

  RenderWindowBase(const std::string name, unsigned int id,
                   std::function<bool()> user_update_func,
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [this, user_update_func](ImguiWindowBase& window) {
              last_updated_ = user_update_func();
              if (last_updated_) {
                first_render_ = false;
              }
            },
            flags | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse),
        camera_handler_(default_camera_handler),
        resizable_(true),
        preserve_aspect_ratio_(false),
        last_updated_(false),
        first_render_(true),
        image_size_(0, 0) {
    border_ = false;
  };

  RenderWindowBase(const std::string name, unsigned int id,
                   std::function<bool()> user_update_func, ImVec2 pos,
                   ImVec2 size, ivec2 image_size = ivec2(0, 0),
                   ImGuiWindowFlags flags = 0)
      : ImguiWindowBase(
            name, id,
            [this, user_update_func](ImguiWindowBase& window) {
              last_updated_ = user_update_func();
              if (last_updated_) {
                first_render_ = false;
              }
            },
            pos, size,
            flags | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse),
        camera_handler_(default_camera_handler),
        resizable_(true),
        preserve_aspect_ratio_(false),
        last_updated_(false),
        first_render_(true),
        image_size_(image_size) {
    border_ = false;
  };

  ~RenderWindowBase(){};
  void create() override;
  void pre() override;
  void draw_() override;
  void post() override;
  void destroy() override;
};
template class ImguiWindowBaseWrapper<RenderWindowBase>;
typedef ImguiWindowBaseWrapper<RenderWindowBase> RenderWindow;

class MeshViewer {
  static bool gui_camera_handler(double dt_ms, CameraHandlerState state,
                                 const reclib::opengl::RenderWindowBase& window,
                                 const std::vector<std::string> names);

 private:
  RenderWindow window_;

 public:
  MeshViewer(const std::string name, const std::vector<std::string> elem_name,
             ImGuiWindowFlags flags = 0)
      : window_(RenderWindow(
            name, true, [elem_name]() { return view_drawelement(elem_name); },
            flags)) {
    window_->camera_handler_ =
        [elem_name](double dt_ms, CameraHandlerState state,
                    const reclib::opengl::RenderWindowBase& window) {
          _RECLIB_ASSERT_GE(elem_name.size(), 1);
          bool moved = gui_camera_handler(dt_ms, state, window, elem_name);
          return moved;
        };
  };
  ~MeshViewer(){};

  static bool view_drawelement(const std::vector<std::string> names);
};

template <typename T>
struct ConfigEntry {
  T* val_;
  T default_val_;
  float min_val_;
  float max_val_;
  float step_;
  std::function<void()> update_func_;
  bool is_color_;

  YAML::Node node_;
  bool has_node_;

  ConfigEntry() : val_(nullptr){};
  ConfigEntry(
      T* val, float min = -5, float max = 5,
      std::function<void()> func = []() {}, float step = 1,
      bool is_color = false)
      : val_(val),
        default_val_(*val),
        min_val_(min),
        max_val_(max),
        step_(step),
        update_func_(func),
        is_color_(is_color),
        has_node_(false){};

  ConfigEntry(
      YAML::Node node, float min = -5, float max = 5,
      std::function<void()> func = []() {}, float step = 1,
      bool is_color = false)
      : default_val_(node.as<T>()),
        min_val_(min),
        max_val_(max),
        step_(step),
        update_func_(func),
        is_color_(is_color),
        node_(node),
        has_node_(true){};

  ConfigEntry(const ConfigEntry& other)
      : val_(other.val_),
        default_val_(other.default_val_),
        min_val_(other.min_val_),
        max_val_(other.max_val_),
        step_(other.step_),
        update_func_(other.update_func_),
        is_color_(other.is_color_),
        node_(other.node_),
        has_node_(other.has_node_){};
};

struct GroupedConfig {
  std::map<std::string, ConfigEntry<bool>> bool_map;
  std::map<std::string, ConfigEntry<float>> float_map;
  std::map<std::string, ConfigEntry<double>> double_map;
  std::map<std::string, ConfigEntry<int>> int_map;

  std::map<std::string, ConfigEntry<vec2>> vec2_map;
  std::map<std::string, ConfigEntry<vec3>> vec3_map;
  std::map<std::string, ConfigEntry<vec4>> vec4_map;

  std::map<std::string, std::string> string_map;
};

class ConfigViewer {
 private:
  void display_yaml(std::string key, YAML::Node node) const;
  void display_config_group(const GroupedConfig& group) const;
  YAML::Node group_to_yaml(const GroupedConfig& group, YAML::Node node) const;
  YAML::Node to_yaml() const;

  OptionWindow window_;
  std::map<std::string, reclib::Configuration*> config_map;
  std::map<std::string, GroupedConfig> configs;

  ConfigViewer(const std::string name, ImGuiWindowFlags flags = 0)
      : window_(OptionWindow(name, false, flags)) {
    window_->user_update_func_ = [this](ImguiWindowBase& window) {
      this->display_config();
    };
  }
  ~ConfigViewer(){};

 public:
  static ConfigViewer& instance() {
    static ConfigViewer view("ConfigViewer##Default");
    return view;
  }

  // variable values
  void add_bool(
      const std::string& key, bool* v,
      std::function<void()> update_func = [] {}) {
    configs["default"].bool_map[key] = ConfigEntry(v, 0, 0, update_func, 0);
  }
  void add_bool(
      const std::string& group, const std::string& key, bool* v,
      std::function<void()> update_func = [] {}) {
    configs[group].bool_map[key] = ConfigEntry(v, 0, 0, update_func, 0);
  }

  void add_int(
      const std::string& key, int* v, std::function<void()> update_func = [] {},
      int min = -5, int max = 5, int step = 1) {
    configs["default"].int_map[key] =
        ConfigEntry(v, min, max, update_func, step);
  }
  void add_int(
      const std::string& key, YAML::Node v,
      std::function<void()> update_func = [] {}, int min = -5, int max = 5,
      int step = 1) {
    configs["default"].int_map[key] =
        ConfigEntry<int>(v, min, max, update_func, step);
  }
  void add_int(
      const std::string& group, const std::string& key, int* v,
      std::function<void()> update_func = [] {}, int min = -5, int max = 5,
      int step = 1) {
    configs[group].int_map[key] = ConfigEntry(v, min, max, update_func, step);
  }
  void add_int(
      const std::string& group, const std::string& key, YAML::Node v,
      std::function<void()> update_func = [] {}, int min = -5, int max = 5,
      int step = 1) {
    configs[group].int_map[key] =
        ConfigEntry<int>(v, min, max, update_func, step);
  }

  void add_float(
      const std::string& key, float* v,
      std::function<void()> update_func = [] {}, float min = -5, float max = 5,
      float step = 0.1) {
    configs["default"].float_map[key] =
        ConfigEntry(v, min, max, update_func, step);
  }
  void add_float(
      const std::string& group, const std::string& key, float* v,
      std::function<void()> update_func = [] {}, float min = -5, float max = 5,
      float step = 0.1) {
    configs[group].float_map[key] = ConfigEntry(v, min, max, update_func, step);
  }
  void add_double(
      const std::string& key, double* v,
      std::function<void()> update_func = [] {}, float min = -5, float max = 5,
      float step = 0.1) {
    configs["default"].double_map[key] =
        ConfigEntry(v, min, max, update_func, step);
  }
  void add_double(
      const std::string& group, const std::string& key, double* v,
      std::function<void()> update_func = [] {}, float min = -5, float max = 5,
      float step = 0.1) {
    configs[group].double_map[key] =
        ConfigEntry(v, min, max, update_func, step);
  }

  void add_vec2(
      const std::string& key, vec2* v,
      std::function<void()> update_func = [] {}, float min = -5, float max = 5,
      float step = 0.1) {
    configs["default"].vec2_map[key] =
        ConfigEntry(v, min, max, update_func, step);
  }
  void add_vec2(
      const std::string& group, const std::string& key, vec2* v,
      std::function<void()> update_func = [] {}, float min = -5, float max = 5,
      float step = 0.1) {
    configs[group].vec2_map[key] = ConfigEntry(v, min, max, update_func, step);
  }
  void add_vec3(
      const std::string& key, vec3* v,
      std::function<void()> update_func = [] {}, bool is_color = true,
      float min = -5, float max = 5, float step = 0.1) {
    configs["default"].vec3_map[key] =
        ConfigEntry(v, min, max, update_func, step, is_color);
  }
  void add_vec3(
      const std::string& group, const std::string& key, vec3* v,
      std::function<void()> update_func = [] {}, bool is_color = true,
      float min = -5, float max = 5, float step = 0.1) {
    configs[group].vec3_map[key] =
        ConfigEntry(v, min, max, update_func, step, is_color);
  }
  void add_vec4(
      const std::string& key, vec4* v,
      std::function<void()> update_func = [] {}, bool is_color = true,
      float min = -5, float max = 5, float step = 0.1) {
    configs["default"].vec4_map[key] =
        ConfigEntry(v, min, max, update_func, step, is_color);
  }
  void add_vec4(
      const std::string& group, const std::string& key, vec4* v,
      std::function<void()> update_func = [] {}, bool is_color = true,
      float min = -5, float max = 5, float step = 0.1) {
    configs[group].vec4_map[key] =
        ConfigEntry(v, min, max, update_func, step, is_color);
  }

  void add_config(const std::string& key, reclib::Configuration* v) {
    config_map[key] = v;
  }

  // constant values
  void add_bool(const std::string& key, bool v) {
    configs["default"].string_map[key] = v;
  }
  void add_bool(const std::string& group, const std::string& key, bool v) {
    configs[group].string_map[key] = v;
  }
  void add_int(const std::string& key, int v) {
    configs["default"].string_map[key] = v;
  }
  void add_int(const std::string& group, const std::string& key, int v) {
    configs[group].string_map[key] = v;
  }
  void add_float(const std::string& key, float v) {
    configs["default"].string_map[key] = v;
  }
  void add_float(const std::string& group, const std::string& key, float v) {
    configs[group].string_map[key] = v;
  }
  void add_vec2(const std::string& key, vec2 v) {
    std::stringstream str;
    str << v;
    configs["default"].string_map[key] = str.str();
  }
  void add_vec2(const std::string& group, const std::string& key, vec2 v) {
    std::stringstream str;
    str << v;
    configs[group].string_map[key] = str.str();
  }
  void add_vec3(const std::string& key, vec3 v) {
    std::stringstream str;
    str << v;
    configs["default"].string_map[key] = str.str();
  }
  void add_vec3(const std::string& group, const std::string& key, vec3 v) {
    std::stringstream str;
    str << v;
    configs[group].string_map[key] = str.str();
  }
  void add_vec4(const std::string& key, vec4 v) {
    std::stringstream str;
    str << v;
    configs["default"].string_map[key] = str.str();
  }
  void add_vec4(const std::string& group, const std::string& key, vec4 v) {
    std::stringstream str;
    str << v;
    configs[group].string_map[key] = str.str();
  }
  void add_string(const std::string& key, std::string v) {
    configs["default"].string_map[key] = v;
  }
  void add_string(const std::string& group, const std::string& key,
                  std::string v) {
    configs[group].string_map[key] = v;
  }

  void display_config() const;
};

template <typename T>
struct ImageEntry {
  T val_;
  ivec2 size_;
  bool changed_;
  ImageEntry(){};
  ImageEntry(T val, ivec2 size = ivec2(300, 300))
      : val_(val), size_(size), changed_(true){};
  ImageEntry(const ImageEntry& other)
      : val_(other.val_), size_(other.size_), changed_(other.changed_){};
};

class ImageViewer {
 private:
  OptionWindow window_;

#if HAS_OPENCV_MODULE
  std::map<std::string, ImageEntry<CpuMat>> mat_map;
#endif
  std::map<std::string, ImageEntry<reclib::opengl::Texture2D>> tex_map;

  ImageViewer(const std::string name, ImGuiWindowFlags flags = 0)
      : window_(OptionWindow(name, false, flags)) {
    window_->user_update_func_ = [this](ImguiWindowBase& window) {
      this->display_images();
    };
  }
  ~ImageViewer(){};

 public:
  static ImageViewer& instance() {
    static ImageViewer view("ImageViewer##Default");
    return view;
  }

#if HAS_OPENCV_MODULE
  void add_mat(std::string key, CpuMat i, ivec2 size = ivec2(300, 300)) {
    mat_map[key] = ImageEntry(i, size);
    mat_map[key].changed_ = true;
  }
#endif
  void add_tex(std::string key, reclib::opengl::Texture2D i,
               ivec2 size = ivec2(300, 300)) {
    tex_map[key] = ImageEntry(i, size);
  }

  void display_images();
};

enum TilingFormat {
  // colums x rows
  SINGLE = 0,
  VERTICAL_ONExONE = 1,
  VERTICAL_ONExTWO = 2,
  HORIZONTAL_ONExONE = 3,
  HORIZONTAL_ONExTWO = 4,
  TWOxTWO = 5,

};

class MainWindow {
 private:
  static bool initialized_;
  static TilingFormat format_;

  MainWindow(){};
  ~MainWindow(){};

 public:
  static RenderWindow window_lt_;
  static RenderWindow window_rt_;
  static RenderWindow window_lb_;
  static RenderWindow window_rb_;

  MainWindow(const MainWindow&) = delete;
  MainWindow& operator=(const MainWindow&) = delete;
  MainWindow& operator=(const MainWindow&&) = delete;

  static void resize();

  static void init_single(ImGuiWindowFlags flags = 0) {
    if (!initialized_) {
      window_lt_ = RenderWindow("Main##Default", true,
                                flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                    ImGuiWindowFlags_NoScrollbar |
                                    ImGuiWindowFlags_NoTitleBar);
      window_lt_->cam_ = reclib::opengl::current_camera();
      initialized_ = true;
      format_ = SINGLE;
    }
  };

  static void init_single(
      std::function<bool(ImguiWindowBase&)> user_update_func,
      ImGuiWindowFlags flags = 0) {
    if (!initialized_) {
      window_lt_ = RenderWindow("Main##Default", true, user_update_func,
                                flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                    ImGuiWindowFlags_NoTitleBar);
      window_lt_->cam_ = reclib::opengl::current_camera();
      initialized_ = true;
      format_ = SINGLE;
    }
  };

  static void init_single(std::function<bool()> user_update_func,
                          ImGuiWindowFlags flags = 0) {
    if (!initialized_) {
      window_lt_ = RenderWindow("Main##Default", true, user_update_func,
                                flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                    ImGuiWindowFlags_NoTitleBar);
      window_lt_->cam_ = reclib::opengl::current_camera();
      initialized_ = true;
      format_ = SINGLE;
    }
  };

  // funcs order: from top to bottom, then from left to right
  // (column first)
  static void init_multiple(
      TilingFormat format,
      std::vector<std::function<bool(ImguiWindowBase&)>> user_update_funcs,
      ImGuiWindowFlags flags = 0) {
    if (!initialized_) {
      format_ = format;
      if (format == SINGLE) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 1);
        window_lt_ =
            RenderWindow("Main##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);
        window_lt_->cam_ = reclib::opengl::current_camera();
      }
      if (format == HORIZONTAL_ONExONE || format == VERTICAL_ONExONE) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 2);
        window_lt_ =
            RenderWindow("Main_Left##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rt_ =
            RenderWindow("Main_Right##Default", true, user_update_funcs[1],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);
      }
      if (format == VERTICAL_ONExTWO) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 3);
        window_lt_ =
            RenderWindow("Main_Left##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rt_ =
            RenderWindow("Main_Right_Top##Default", true, user_update_funcs[1],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rb_ = RenderWindow(
            "Main_Right_Bottom##Default", true, user_update_funcs[2],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);
      }
      if (format == HORIZONTAL_ONExTWO) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 3);
        window_lt_ =
            RenderWindow("Main_Left##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_lb_ = RenderWindow(
            "Main_Left_Bottom##Default", true, user_update_funcs[1],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);

        window_rb_ = RenderWindow(
            "Main_Right_Bottom##Default", true, user_update_funcs[2],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);
      }
      if (format == TWOxTWO) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 4);
        window_lt_ =
            RenderWindow("Main_Left_Top##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_lb_ = RenderWindow(
            "Main_Left_Bottom##Default", true, user_update_funcs[1],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);

        window_rt_ =
            RenderWindow("Main_Right_Top##Default", true, user_update_funcs[2],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rb_ = RenderWindow(
            "Main_Right_Bottom##Default", true, user_update_funcs[3],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);
      }

      initialized_ = true;
    }
  }

  static void init_multiple(
      TilingFormat format, std::vector<std::function<bool()>> user_update_funcs,
      ImGuiWindowFlags flags = 0) {
    if (!initialized_) {
      format_ = format;
      if (format == SINGLE) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 1);
        window_lt_ =
            RenderWindow("Main##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);
        window_lt_->cam_ = reclib::opengl::current_camera();
      }
      if (format == HORIZONTAL_ONExONE || format == VERTICAL_ONExONE) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 2);
        window_lt_ =
            RenderWindow("Main_Left##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rt_ =
            RenderWindow("Main_Right##Default", true, user_update_funcs[1],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);
      }
      if (format == VERTICAL_ONExTWO) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 3);
        window_lt_ =
            RenderWindow("Main_Left##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rt_ =
            RenderWindow("Main_Right_Top##Default", true, user_update_funcs[1],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rb_ = RenderWindow(
            "Main_Right_Bottom##Default", true, user_update_funcs[2],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);
      }
      if (format == HORIZONTAL_ONExTWO) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 3);
        window_lt_ =
            RenderWindow("Main_Left##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_lb_ = RenderWindow(
            "Main_Left_Bottom##Default", true, user_update_funcs[1],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);

        window_rb_ = RenderWindow(
            "Main_Right_Bottom##Default", true, user_update_funcs[2],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);
      }
      if (format == TWOxTWO) {
        _RECLIB_ASSERT_EQ(user_update_funcs.size(), 4);
        window_lt_ =
            RenderWindow("Main_Left_Top##Default", true, user_update_funcs[0],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_lb_ = RenderWindow(
            "Main_Left_Bottom##Default", true, user_update_funcs[1],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);

        window_rt_ =
            RenderWindow("Main_Right_Top##Default", true, user_update_funcs[2],
                         flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

        window_rb_ = RenderWindow(
            "Main_Right_Bottom##Default", true, user_update_funcs[3],
            flags | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoTitleBar);
      }

      initialized_ = true;
    }
  }

  static bool initialized() { return initialized_; }
  static void draw() {
    if (initialized_) window_lt_->draw();
  }
};

void create_default_windows(bool left_dock, bool bottom_dock, bool right_dock);

// -------------------------------------------
// callbacks

void imgui_resize_callback();
void gui_add_callback(const std::string& name, std::function<void()> fn);
void gui_remove_callback(const std::string& name);

// ------------------------------------------
// theme
void set_imgui_theme();

// -------------------------------------------
// main draw call

void gui_draw_toggle();
void gui_draw();

// -------------------------------------------
// helper functions to display properties

void gui_display_mat4(mat4& mat);
void gui_display_camera(Camera& cam);
void gui_display_texture(const Texture2D& tex,
                         const ivec2& size = ivec2(300, 300));
void gui_display_shader(Shader& shader);
void gui_display_framebuffer(const Framebuffer& fbo);
void gui_display_material(Material& mat);
void gui_display_geometry(GeometryBase& geom);
void gui_display_mesh(Mesh& mesh);
void gui_display_drawelement(Drawelement elem);
void gui_display_grouped_drawelements(GroupedDrawelements elem);
void gui_display_query_timer(const Query& query, const char* label = "");
void gui_display_query_counter(const Query& query, const char* label = "");
// TODO add more

}  // namespace opengl
}  // namespace reclib

#endif