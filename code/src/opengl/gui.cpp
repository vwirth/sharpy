
#include <reclib/opengl/gui.h>

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <map>
#include <cmath>
#include <memory>
#include <sstream>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "imgui/imgui_internal.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

std::map<std::string, ImGuiID> reclib::opengl::imgui_window_name_to_id;
std::map<ImGuiID, std::shared_ptr<reclib::opengl::ImguiWindowBase>>
    reclib::opengl::imgui_toggle_windows;
std::map<ImGuiID, std::shared_ptr<reclib::opengl::ImguiWindowBase>>
    reclib::opengl::imgui_dockspace_windows;
std::map<ImGuiID, std::shared_ptr<reclib::opengl::ImguiWindowBase>>
    reclib::opengl::imgui_visible_windows;
// std::map<unsigned int, ImGuiID> reclib::opengl::imgui_to_internal_id;

static void print_window(ImGuiWindow* window) {
  if (window == nullptr) return;
  std::cout << "----- Window: " << window << std::endl;

  if (window != nullptr) {
    std::cout << "ID: " << window->ID << std::endl;
    std::cout << "DockId: " << window->DockId << std::endl;
    std::cout << "Name: " << window->Name << std::endl;
    if (window->RootWindow == window) {
      std::cout << "RootWindow: self" << std::endl;
    } else {
      std::cout << "Rootwindow: ";
      if (window->RootWindow != nullptr)
        std::cout << window->RootWindow->Name << "," << window->RootWindow->ID
                  << std::endl;
      // if (window->RootWindow != window) print_window(window->RootWindow);
    }

    if (window->ParentWindow == window) {
      std::cout << "ParentWindow: self" << std::endl;
    } else {
      std::cout << "ParentWindow: ";
      if (window->ParentWindow != nullptr)
        std::cout << window->ParentWindow->Name << ","
                  << window->ParentWindow->ID << std::endl;
      // if (window->ParentWindow != window) print_window(window->ParentWindow);
    }

    std::cout << "ID Stack: " << std::endl;
    for (unsigned int i = 0; i < window->IDStack.size(); i++) {
      std::cout << "-- " << window->IDStack[i] << std::endl;
    }
  }
  std::cout << "****************************************" << std::endl;
}

static void print_node(ImGuiDockNode* node) {
  if (node == nullptr) return;
  std::cout << "----- Node: " << node << std::endl;
  if (node != nullptr) {
    std::cout << "ID: " << node->ID << std::endl;
    std::cout << "IsDockspace: " << node->IsDockSpace() << std::endl;

    if (node->ParentNode == node) {
      std::cout << "ParentNode: self" << std::endl;
    } else {
      std::cout << "ParentNode: ";
      if (node->ParentNode != nullptr) {
        std::cout << node->ParentNode->ID << std::endl;
        // if (reclib::opengl::imgui_to_internal_id.find(node->ParentNode->ID)
        // !=
        //     reclib::opengl::imgui_to_internal_id.end()) {
        //   std::cout
        //       << "Found internal id: "
        //       << reclib::opengl::imgui_to_internal_id[node->ParentNode->ID]
        //       << std::endl;
        // }
      }
      // if (node->ParentNode != node) print_node(node->ParentNode);
    }

    std::cout << "HostWindow: ";
    print_window(node->HostWindow);

    std::cout << "#Windows: " << node->Windows.size() << std::endl;
    for (unsigned int i = 0; i < node->Windows.size(); i++) {
      print_window(node->Windows[i]);
    }
  }
  std::cout << "--------------------------------------------" << std::endl;
}

static bool shift_pressed() {
  glfwPollEvents();
  if (reclib::opengl::Context::key_pressed(GLFW_KEY_LEFT_SHIFT) ||
      reclib::opengl::Context::key_pressed(GLFW_KEY_RIGHT_SHIFT)) {
    return true;
  }
  return false;
}

// -------------------------------------------
// ImguiWindowBase
// -------------------------------------------
void reclib::opengl::ImguiWindowBase::create() {
  if (pos_.x >= 0 && pos_.y >= 0) {
    ImGui::SetNextWindowPos(pos_);
  }
  if (size_.x > 0 && size_.y > 0) {
    ImGui::SetNextWindowSize(size_);
  }

  initialized_ = true;
}

void reclib::opengl::ImguiWindowBase::pre() {}
void reclib::opengl::ImguiWindowBase::draw_always() {}

void reclib::opengl::ImguiWindowBase::draw_() {
  user_update_func_(*this);
  for (unsigned int i = 0; i < child_windows_.size(); i++) {
    child_windows_[i]->draw();
  }
}
void reclib::opengl::ImguiWindowBase::post() {}

void reclib::opengl::ImguiWindowBase::draw() {
  if (!initialized_) {
    create();
    if (!initialized_) {
      throw std::runtime_error("Window " + name_ +
                               " could not be initialized.");
    }
  }

  if (is_docked_) {
    if (imgui_window_name_to_id.find(dock_name_) !=
        imgui_window_name_to_id.end()) {
      int id = imgui_window_name_to_id[dock_name_];
      if (imgui_dockspace_windows.find(id) != imgui_dockspace_windows.end()) {
        show_window_ = imgui_dockspace_windows[id]->show_window_;
      } else if (imgui_visible_windows.find(id) !=
                 imgui_visible_windows.end()) {
        show_window_ = imgui_visible_windows[id]->show_window_;
      } else if (imgui_toggle_windows.find(id) != imgui_toggle_windows.end()) {
        show_window_ = imgui_toggle_windows[id]->show_window_;
      }
    } else {
      // std::cout << name_ << " [ERROR] Could not find dockid:  " << dock_name_
      //           << std::endl;
    }
    // if (show_window_)
    //   std::cout << "name: " << name_ << " docked: " << dock_name_
    //             << " show_window: " << show_window_ << std::endl;
  }

  pre();

  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  bool last_shrinked = shrinked_;
  if (!first_draw_) {
    shrinked_ = false;

    if (!is_docked_) {
      if ((pos_.x > viewport->WorkPos.x + viewport->WorkSize.x) || pos_.x < 0) {
        // clamp to viewport
        backup_pos_.x = pos_.x;
        pos_.x = std::fmax(
            0,
            std::fmin(viewport->WorkPos.x + viewport->WorkSize.x - 50, pos_.x));

        shrinked_ = true;
      }
      if ((pos_.y > viewport->WorkPos.y + viewport->WorkSize.y) || pos_.y < 0) {
        // clamp to viewport
        backup_pos_.y = pos_.y;
        pos_.y = std::fmax(
            0,
            std::fmin(viewport->WorkPos.y + viewport->WorkSize.y - 50, pos_.y));

        shrinked_ = true;
      }
      if ((pos_.x + size_.x > viewport->WorkPos.x + viewport->WorkSize.x)) {
        // clamp to viewport
        backup_size_.x = size_.x;
        size_.x = std::fmax(
            0, std::fmin(viewport->WorkPos.x + viewport->WorkSize.x - pos_.x,
                         size_.x));
        shrinked_ = true;
      }
      if ((pos_.y + size_.y > viewport->WorkPos.y + viewport->WorkSize.y)) {
        // clamp to viewport
        backup_size_.y = size_.y;
        size_.y = std::fmax(
            0, std::fmin(viewport->WorkPos.y + viewport->WorkSize.y - pos_.y,
                         size_.y));
        shrinked_ = true;
      }
    }

    if (shrinked_) {
      if (backup_pos_.x < 0) backup_pos_.x = pos_.x;
      if (backup_pos_.y < 0) backup_pos_.y = pos_.y;
      if (backup_size_.x < 0) backup_size_.x = size_.x;
      if (backup_size_.y < 0) backup_size_.y = size_.y;
    }
  }

  if (first_draw_ || updated_size_pos_) {
    if ((pos_.x >= 0 && pos_.y >= 0)) ImGui ::SetNextWindowPos(pos_);
    if (size_.x > 0 && size_.y > 0) ImGui::SetNextWindowSize(size_);

    updated_size_pos_ = false;
  }
  if (!first_draw_ && shrinked_ && !last_shrinked) {
    ImGui::SetNextWindowPos(backup_pos_);
    ImGui::SetNextWindowSize(backup_size_);
  }
  if (!first_draw_ && last_shrinked && !shrinked_) {
    backup_pos_.x = -1;
    backup_pos_.y = -1;
    backup_size_.x = -1;
    backup_size_.y = -1;
    if ((pos_.x >= 0 && pos_.y >= 0)) ImGui::SetNextWindowPos(pos_);
    if (size_.x > 0 && size_.y > 0) ImGui::SetNextWindowSize(size_);
  }
  // if (!first_draw_ && is_docked_ && !shift_pressed()) {
  //   ImGuiID dock_id;
  //   if (imgui_window_name_to_id.find(dock_name_) !=
  //       imgui_window_name_to_id.end()) {
  //     int id = imgui_window_name_to_id[dock_name_];
  //     if (imgui_dockspace_windows.find(id) != imgui_dockspace_windows.end())
  //     {
  //       dock_id =
  //       std::dynamic_pointer_cast<reclib::opengl::DockspaceWindowBase,
  //                                           reclib::opengl::ImguiWindowBase>(
  //                     imgui_dockspace_windows[id])
  //                     ->dockspace_id_;
  //       ImGui::SetNextWindowDockID(dock_id);
  //     } else {
  //       std::cout << "Could not find dock name: " << dock_name_ << std::endl;
  //     }
  //   }
  // }

  if (show_window_ || first_draw_) {
    if (ImGui::Begin(name_.c_str(), &show_window_, flags_)) {
      // if (first_draw_)
      //   std::cout << "name: " << name_ << " docked: " <<
      //   ImGui::IsWindowDocked()
      //             << std::endl;
      if (ImGui::IsWindowDocked()) {
        if (ImGui::IsWindowDocked()) {
          ImGuiDockNode* node = ImGui::GetWindowDockNode();
          std::string dock_name = node->HostWindow->RootWindow->Name;
          if (imgui_window_name_to_id.find(dock_name) !=
              imgui_window_name_to_id.end()) {
            dock_name_ = dock_name;
          } else {
            // std::cout << name_ << " could not find name: " << dock_name
            //           << " parent name: " << node->HostWindow->Name
            //           << " name: " <<
            //           ImGui::GetWindowDockNode()->HostWindow->Name
            //           << std::endl;
          }
        }
      }
      is_docked_ = ImGui::IsWindowDocked();

      if (size_.x <= 0 && size_.y <= 0) {
        size_ = ImGui::GetWindowSize();
      } else {
        // if (ImGui::GetWindowSize().x != size_.x ||
        //     ImGui::GetWindowSize().y != size_.y) {
        //   updated_size_pos_ = true;
        // }
      }

      if (first_draw_ && !is_docked_) {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        if ((pos_.x < 0 && pos_.y < 0)) {
          // place in the center
          pos_.x = viewport->WorkPos.x + viewport->WorkSize.x / 2.f -
                   std::fmax(0, size_.x * 0.5);
          pos_.y = viewport->WorkPos.y + viewport->WorkSize.y / 2.f -
                   std::fmax(0, size_.y * 0.5);
          // only set position if window has no position in the .ini file
          ImGui::SetWindowPos(pos_, ImGuiCond_FirstUseEver);
        }
      }

      if (pos_.x <= 0 && pos_.y <= 0) {
        pos_ = ImGui::GetWindowPos();
      } else {
        // if (ImGui::GetWindowPos().x != pos_.x ||
        //     ImGui::GetWindowPos().y != pos_.y) {
        //   updated_size_pos_ = true;
        // }
      }

      if (ImGui::IsWindowHovered()) {
        is_hovered_ = true;
      } else {
        is_hovered_ = false;
      }

      draw_always();
      if (show_window_) draw_();
    }
    ImGui::End();
  }

  post();

  if (first_draw_) first_draw_ = false;
}

void reclib::opengl::ImguiWindowBase::destroy() {}

reclib::opengl::ImguiWindowBase::~ImguiWindowBase(){};

// -------------------------------------------
// OptionWindowBase
// -------------------------------------------
void reclib::opengl::OptionWindowBase::create() {
  backup_flags_ = flags_;
  initialized_ = true;
}

void reclib::opengl::OptionWindowBase::pre() {
  flags_ = backup_flags_;
  if (!shift_pressed()) {
    flags_ |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
  }
}

void reclib::opengl::OptionWindowBase::post() {}

void reclib::opengl::OptionWindowBase::draw_() {
  if (!show_window_) return;

  user_update_func_(*this);
  for (unsigned int i = 0; i < child_windows_.size(); i++) {
    child_windows_[i]->draw();
  }
}

void reclib::opengl::OptionWindowBase::destroy() {}

// -------------------------------------------
// RenderWindowBase
// -------------------------------------------

bool reclib::opengl::RenderWindowBase::gui_camera_handler(double dt_ms) {
  static bool last_click = false;
  static bool current_click = false;
  static bool is_clicking_state = false;
  static unsigned int clicked_id = 0;

  bool shift = shift_pressed();
  current_click =
      (reclib::opengl::Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT) ||
       reclib::opengl::Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT) ||
       reclib::opengl::Context::mouse_button_pressed(
           GLFW_MOUSE_BUTTON_MIDDLE)) &&
      !shift;

  if (!last_click && current_click) {
    is_clicking_state = true;
    clicked_id = id_;
  }
  if (last_click && !current_click) {
    is_clicking_state = false;
  }
  CameraHandlerState state = {.clicked_ = is_clicking_state,
                              .click_id_ = clicked_id};
  bool moved = camera_handler_(dt_ms, state, *this);

  last_click = current_click;

  return moved;
}

bool reclib::opengl::RenderWindowBase::default_camera_handler(
    double dt_ms, CameraHandlerState state,
    const reclib::opengl::RenderWindowBase& window) {
  bool moved = false;
  if (ImGui::GetIO().WantCaptureMouse) {
    if (state.clicked_ && window.id_ != state.click_id_) return moved;

    moved = reclib::opengl::CameraImpl::default_mouse_handler(dt_ms);
    bool tmp = reclib::opengl::CameraImpl::default_keyboard_handler(dt_ms);
    moved = moved || tmp;
  }
  return moved;
}

void reclib::opengl::RenderWindowBase::create() {
  if (!reclib::opengl::Context::running()) return;

  if (image_size_.x() <= 0 || image_size_.y() <= 0) {
    image_size_ = ivec2(std::fmax(0, size_.x), std::fmax(0, size_.y));
  }
  if (image_size_.x() <= 0 || image_size_.y() <= 0) {
    // default size
    image_size_ = ivec2(500, 500);
  }

  if (!fbo_.initialized()) {
    fbo_ = reclib::opengl::Framebuffer("window_" + name_, image_size_.x(),
                                       image_size_.y());

    fbo_->attach_depthbuffer(reclib::opengl::Texture2D(
        name_ + "_preprocess_framebuffer/depth", image_size_.x(),
        image_size_.y(), GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT));
    fbo_->attach_colorbuffer(reclib::opengl::Texture2D(
        name_ + "_preprocess_framebuffer/col", image_size_.x(), image_size_.y(),
        GL_RGB32F, GL_RGB, GL_FLOAT));
    fbo_->check();
  }

  if (!cam_.initialized()) {
    cam_ = reclib::opengl::Camera("window_" + name_);
    cam_->pos = vec3(0, 0, 0);
    cam_->dir = vec3(0, 0, 1);
    cam_->up = vec3(0, 1, 0);
    cam_->fix_up_vector = false;
    cam_->far = 5000;
    cam_->update();
  }
  backup_flags_ = flags_;

  initialized_ = true;
}

void reclib::opengl::RenderWindowBase::pre() {
  if (resizable_ && !first_draw_ &&
      (image_size_.x() != content_size_.x ||
       image_size_.y() != content_size_.y)) {
    if (content_size_.x > 0 && content_size_.y > 0) {
      if (!preserve_aspect_ratio_) {
        image_size_.x() = content_size_.x;
        image_size_.y() = content_size_.y;
      } else if (preserve_aspect_ratio_ &&
                 (image_size_.x() != content_size_.x &&
                  image_size_.y() != content_size_.y)) {
        // std::cout << "[BEFORE]" << name_ << " image: " << image_size_
        //           << " content: " << content_size_.x << "," <<
        //           content_size_.y
        //           << std::endl;

        float content_aspect = content_size_.x / (float)content_size_.y;
        float image_aspect = image_size_.x() / (float)image_size_.y();

        // std::cout << "content aspect: " << content_aspect
        //           << " image aspect: " << image_aspect << std::endl;

        if (image_aspect > 1) {
          // width > height
          if (content_size_.x * (1.f / image_aspect) > content_size_.y) {
            image_size_.y() = content_size_.y;
            image_size_.x() = content_size_.y * image_aspect;
          } else {
            image_size_.x() = content_size_.x;
            image_size_.y() = content_size_.x * (1.f / image_aspect);
          }
        } else {
          if (content_size_.y * image_aspect > content_size_.x) {
            image_size_.x() = content_size_.x;
            image_size_.y() = content_size_.x * (1.f / image_aspect);
          } else {
            image_size_.y() = content_size_.y;
            image_size_.x() = content_size_.y * image_aspect;
          }
        }

        // std::cout << "[AFTER]" << name_ << " image: " << image_size_
        //           << " content: " << content_size_.x << "," <<
        //           content_size_.y
        //           << std::endl;

        // std::cout << "----------------------------" << std::endl;
      }

      fbo_->resize(image_size_.x(), image_size_.y());
    }
  }

  flags_ = backup_flags_;
  if (!shift_pressed()) {
    flags_ |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
  }
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  if (!show_window_) return;

  backup_cam_ = reclib::opengl::current_camera();
  reclib::opengl::make_camera_current(cam_);

  GLfloat clear_color_backup[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color_backup);

  if (is_hovered_ || first_draw_) {
    // gui_camera_handler(reclib::opengl::Context::frame_time());
    gui_camera_handler(reclib::opengl::Context::frame_time());
    cam_->update();
  }

  fbo_->bind();
  glClearColor(0.2, 0.2, 0.2, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  user_update_func_(*this);

  fbo_->unbind();

  glClearColor(clear_color_backup[0], clear_color_backup[1],
               clear_color_backup[2], clear_color_backup[3]);
  reclib::opengl::make_camera_current(backup_cam_);
}

void reclib::opengl::RenderWindowBase::post() { ImGui::PopStyleVar(); }

void reclib::opengl::RenderWindowBase::draw_() {
  ImVec2 vMin = ImGui::GetWindowContentRegionMin();
  ImVec2 vMax = ImGui::GetWindowContentRegionMax();
  content_size_.x = vMax.x - vMin.x;
  content_size_.y = vMax.y - vMin.y;

  if (!show_window_) return;
  if (!first_draw_ && (content_size_.x <= 0 || content_size_.y <= 0)) return;

  reclib::opengl::Texture2D tex = fbo_->color_textures[0];
  ImGui::Image((ImTextureID)tex->id, ImVec2(image_size_.x(), image_size_.y()),
               ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1),
               ImVec4(1, 1, 1, 0.5));
  for (unsigned int i = 0; i < child_windows_.size(); i++) {
    child_windows_[i]->draw();
  }
}

void reclib::opengl::RenderWindowBase::destroy() {}

// -------------------------------------------
// MainWindow
// -------------------------------------------
bool reclib::opengl::MainWindow::initialized_ = false;
reclib::opengl::TilingFormat reclib::opengl::MainWindow::format_ = SINGLE;
reclib::opengl::RenderWindow reclib::opengl::MainWindow::window_lt_;
reclib::opengl::RenderWindow reclib::opengl::MainWindow::window_lb_;
reclib::opengl::RenderWindow reclib::opengl::MainWindow::window_rt_;
reclib::opengl::RenderWindow reclib::opengl::MainWindow::window_rb_;

void reclib::opengl::MainWindow::resize() {
  if (!initialized_) return;

  // resize main window
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImVec2 main_size = viewport->WorkSize;
  ImVec2 main_pos = viewport->WorkPos;
  if (reclib::opengl::Context::show_gui) {
    if (imgui_window_name_to_id.find("Docking Window##Left") !=
        imgui_window_name_to_id.end()) {
      std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
          [imgui_window_name_to_id["Docking Window##Left"]];
      if (window->show_window_) {
        main_pos.x += window->get_pos().x + window->get_size().x;
      }
    }
    if (imgui_window_name_to_id.find("Docking Window##Right") !=
        imgui_window_name_to_id.end()) {
      std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
          [imgui_window_name_to_id["Docking Window##Right"]];
      if (window->show_window_) {
        main_size.x = window->get_pos().x - main_pos.x;
      }
    }
    if (imgui_window_name_to_id.find("Docking Window##Bottom") !=
        imgui_window_name_to_id.end()) {
      std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
          [imgui_window_name_to_id["Docking Window##Bottom"]];
      if (window->show_window_) {
        main_size.y = window->get_pos().y - main_pos.y;
      }
    }
  }
  if (format_ == SINGLE) {
    reclib::opengl::MainWindow::window_lt_->set_size(main_size);
    reclib::opengl::MainWindow::window_lt_->set_pos(main_pos);
  } else if (format_ == VERTICAL_ONExONE) {
    ImVec2 size_half(main_size.x * 0.5f, main_size.y);
    ImVec2 pos_left(main_pos.x, main_pos.y);
    ImVec2 pos_right(main_pos.x + size_half.x, main_pos.y);

    reclib::opengl::MainWindow::window_lt_->set_size(size_half);
    reclib::opengl::MainWindow::window_lt_->set_pos(pos_left);

    reclib::opengl::MainWindow::window_rt_->set_size(size_half);
    reclib::opengl::MainWindow::window_rt_->set_pos(pos_right);
  } else if (format_ == HORIZONTAL_ONExONE) {
    ImVec2 size_half(main_size.x, main_size.y * 0.5f);
    ImVec2 pos_left(main_pos.x, main_pos.y);
    ImVec2 pos_right(main_pos.x, main_pos.y + size_half.y);

    reclib::opengl::MainWindow::window_lt_->set_size(size_half);
    reclib::opengl::MainWindow::window_lt_->set_pos(pos_left);

    reclib::opengl::MainWindow::window_rt_->set_size(size_half);
    reclib::opengl::MainWindow::window_rt_->set_pos(pos_right);
  } else if (format_ == VERTICAL_ONExTWO) {
    ImVec2 size_half(main_size.x * 0.5f, main_size.y);
    ImVec2 size_half_half(main_size.x * 0.5f, main_size.y * 0.5);

    ImVec2 pos_left_top(main_pos.x, main_pos.y);
    ImVec2 pos_right_top(main_pos.x + size_half.x, main_pos.y);
    ImVec2 pos_right_bottom(main_pos.x + size_half.x,
                            main_pos.y + size_half_half.y);

    reclib::opengl::MainWindow::window_lt_->set_size(size_half);
    reclib::opengl::MainWindow::window_lt_->set_pos(pos_left_top);

    reclib::opengl::MainWindow::window_rt_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_rt_->set_pos(pos_right_top);

    reclib::opengl::MainWindow::window_rb_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_rb_->set_pos(pos_right_bottom);
  } else if (format_ == HORIZONTAL_ONExTWO) {
    ImVec2 size_half(main_size.x, main_size.y * 0.5f);
    ImVec2 size_half_half(main_size.x * 0.5f, main_size.y * 0.5);

    ImVec2 pos_top(main_pos.x, main_pos.y);
    ImVec2 pos_left_bottom(main_pos.x, main_pos.y + size_half.y);
    ImVec2 pos_right_bottom(main_pos.x + size_half_half.x,
                            main_pos.y + size_half.y);

    reclib::opengl::MainWindow::window_lt_->set_size(size_half);
    reclib::opengl::MainWindow::window_lt_->set_pos(pos_top);

    reclib::opengl::MainWindow::window_rb_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_rb_->set_pos(pos_right_bottom);

    reclib::opengl::MainWindow::window_lb_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_lb_->set_pos(pos_left_bottom);
  } else if (format_ == TWOxTWO) {
    ImVec2 size_half_half(main_size.x * 0.5f, main_size.y * 0.5);

    ImVec2 pos_left_top(main_pos.x, main_pos.y);
    ImVec2 pos_left_bottom(main_pos.x, main_pos.y + size_half_half.y);
    ImVec2 pos_right_top(main_pos.x + size_half_half.x, main_pos.y);
    ImVec2 pos_right_bottom(main_pos.x + size_half_half.x,
                            main_pos.y + size_half_half.y);

    reclib::opengl::MainWindow::window_lt_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_lt_->set_pos(pos_left_top);

    reclib::opengl::MainWindow::window_lb_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_lb_->set_pos(pos_left_bottom);

    reclib::opengl::MainWindow::window_rt_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_rt_->set_pos(pos_right_top);

    reclib::opengl::MainWindow::window_rb_->set_size(size_half_half);
    reclib::opengl::MainWindow::window_rb_->set_pos(pos_right_bottom);
  }
}

// -------------------------------------------
// MeshViewer
// -------------------------------------------

bool reclib::opengl::MeshViewer::gui_camera_handler(
    double dt_ms, CameraHandlerState state,
    const reclib::opengl::RenderWindowBase& window,
    const std::vector<std::string> names) {
  bool moved = false;
  if (names.size() == 0) return moved;

  vec3 bb_min(std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity());
  vec3 bb_max(-std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity());
  for (unsigned int i = 0; i < names.size(); i++) {
    if (names[i].length() == 0) continue;
    if (!reclib::opengl::Drawelement::valid(names[i])) continue;
    reclib::opengl::Drawelement elem_ =
        reclib::opengl::Drawelement::find(names[i]);

    bb_min = bb_min.cwiseMin(elem_->mesh->geometry->bb_min);
    bb_max = bb_max.cwiseMax(elem_->mesh->geometry->bb_max);
  }
  vec3 center = (bb_max - bb_min) * 0.5 + bb_min;

  vec3 cam_pos = reclib::opengl::current_camera()->pos;
  vec3 cam_dir = reclib::opengl::current_camera()->dir;
  vec3 cam_up = reclib::opengl::current_camera()->up;

  float required_length = (center - cam_pos).norm();

  if (window.first_draw_) {
    vec3 length = bb_max - center;
    // computer aabb radius
    float max = std::fmax(length.x(), std::fmax(length.y(), length.z()));

    // given the y-axis field of view of camera and the bounding box length in
    // y-direction we can use trigonometric functions to estimate the
    // minimum camera distance to the object
    required_length =
        (max + max) / std::tan(reclib::radians(
                          reclib::opengl::current_camera()->fov_degree * 0.5));
    cam_pos = center - cam_dir.normalized() * (required_length);
  }

  vec3 dist = center - cam_pos;

  static float rot_speed = 1.f;
  static vec2 last_pos(-1, -1);
  const vec2 curr_pos = Context::mouse_pos();
  if (last_pos == vec2(-1, -1)) last_pos = curr_pos;
  const vec2 diff = last_pos - curr_pos;
  if (ImGui::GetIO().WantCaptureMouse) {
    if (state.clicked_ && window.id_ != state.click_id_) return moved;

    if (Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_MIDDLE)) {
      float scale = std::fmin(
          1, std::fmax(0.1, (abs(diff.y()) / window.image_size_.y())));
      if (diff.y() < 0) {
        required_length -= scale * required_length;
      } else if (diff.y() > 0) {
        required_length += scale * required_length;
      }
      required_length = std::fmax(0.01, required_length);

      cam_pos = center - cam_dir.normalized() * (required_length);
    }

    if ((Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT)) &&
        diff.cwiseAbs().sum() > 0) {
      mat4 rot;
      if (abs(diff.x()) > abs(diff.y())) {
        rot = reclib::rotate(diff.x() * rot_speed * float(M_PI) / 180.f,
                             cam_up.normalized());
      } else {
        rot = reclib::rotate(diff.y() * rot_speed * float(M_PI) / 180.f,
                             cross(cam_dir, cam_up).normalized());
      }

      vec3 updated_dist = (rot * -dist.homogeneous()).head<3>();
      cam_pos = center + (updated_dist).normalized() * required_length;

      cam_dir = (center - cam_pos).normalized();
      if (!reclib::opengl::current_camera()->fix_up_vector)
        cam_up = normalize(cross(cross(cam_dir, cam_up), cam_dir));
    }

    if (Context::key_pressed(GLFW_KEY_R)) {
      vec3 length = bb_max - center;
      // computer aabb radius
      float max = std::fmax(length.x(), std::fmax(length.y(), length.z()));

      // given the y-axis field of view of camera and the bounding box length in
      // y-direction we can use trigonometric functions to estimate the
      // minimum camera distance to the object
      required_length =
          (max + max) /
          std::tan(reclib::radians(
              reclib::opengl::current_camera()->fov_degree * 0.5));
      cam_pos = center - cam_dir.normalized() * (required_length);
    }
  }
  last_pos = curr_pos;

  reclib::opengl::current_camera()->pos = cam_pos;
  reclib::opengl::current_camera()->dir = cam_dir;
  reclib::opengl::current_camera()->up = cam_up;

  return moved;
}

bool reclib::opengl::MeshViewer::view_drawelement(
    const std::vector<std::string> names) {
  if (names.size() == 0) return false;

  for (unsigned int i = 0; i < names.size(); i++) {
    if (names[i].length() == 0) continue;

    if (!reclib::opengl::Drawelement::valid(names[i])) continue;
    reclib::opengl::Drawelement elem_ =
        reclib::opengl::Drawelement::find(names[i]);

    elem_->bind();
    elem_->draw();
    elem_->unbind();
  }
  return true;
}

// -------------------------------------------
// ConfigViewer
// -------------------------------------------

void reclib::opengl::ConfigViewer::display_yaml(std::string key,
                                                YAML::Node node) const {
  switch (node.Type()) {
    case YAML::NodeType::Scalar: {
      ImGui::TextColored(ImVec4(73 / 255.f, 165 / 255.f, 227 / 255.f, 1),
                         "%s: ", key.c_str());
      ImGui::SameLine();
      ImGui::TextWrapped("%s", node.as<std::string>().c_str());

      break;
    }
    case YAML::NodeType::Sequence: {
      std::vector<std::string> elems = node.as<std::vector<std::string>>();
      std::stringstream str;
      for (unsigned int i = 0; i < elems.size(); i++) {
        if (i < elems.size() - 1) {
          str << elems[i] << ", ";
        } else {
          str << elems[i];
        }
      }
      ImGui::TextColored(ImVec4(73 / 255.f, 165 / 255.f, 227 / 255.f, 1),
                         "%s: ", key.c_str());
      ImGui::SameLine();
      ImGui::TextWrapped("%s", str.str().c_str());

      break;
    }
    case YAML::NodeType::Map: {
      if (ImGui::CollapsingHeader((key + "##" + window_->name_).c_str())) {
        for (auto it : node) {
          display_yaml(it.first.as<std::string>(), it.second);
        }
      }
      break;
    }
    default: {
    }
  }
}

YAML::Node reclib::opengl::ConfigViewer::to_yaml() const {
  YAML::Node node;

  for (auto it : configs) {
    node = group_to_yaml(it.second, node);
  }

  return node;
}

YAML::Node reclib::opengl::ConfigViewer::group_to_yaml(
    const GroupedConfig& group, YAML::Node node) const {
  for (auto it : group.bool_map) {
    std::string key = it.first;
    std::replace(key.begin(), key.end(), ' ', '_');
    if (it.second.has_node_) {
      node[key] = it.second.node_.as<bool>();
    } else {
      node[key] = *it.second.val_;
    }
  }
  for (auto it : group.float_map) {
    std::string key = it.first;
    std::replace(key.begin(), key.end(), ' ', '_');
    if (it.second.has_node_) {
      node[key] = it.second.node_.as<float>();
    } else {
      node[key] = *it.second.val_;
    }
  }
  for (auto it : group.double_map) {
    std::string key = it.first;
    std::replace(key.begin(), key.end(), ' ', '_');
    if (it.second.has_node_) {
      node[key] = it.second.node_.as<double>();
    } else {
      node[key] = *it.second.val_;
    }
  }
  for (auto it : group.int_map) {
    std::string key = it.first;
    std::replace(key.begin(), key.end(), ' ', '_');
    if (it.second.has_node_) {
      node[key] = it.second.node_.as<int>();
    } else {
      node[key] = *it.second.val_;
    }
  }
  for (auto it : group.string_map) {
    std::string key = it.first;
    std::replace(key.begin(), key.end(), ' ', '_');
    node[key] = it.second;
  }
  return node;
}

void reclib::opengl::ConfigViewer::display_config_group(
    const reclib::opengl::GroupedConfig& group) const {
  int slider_thresh = 10;

  for (auto it : group.bool_map) {
    if (it.second.has_node_) {
      bool val = it.second.node_.as<bool>();
      if (ImGui::Checkbox((it.first + "##" + window_->name_).c_str(), &val)) {
        it.second.node_ = val;
        it.second.update_func_();
      }
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        it.second.node_ = it.second.default_val_;
        it.second.update_func_();
      }
    } else {
      if (ImGui::Checkbox((it.first + "##" + window_->name_).c_str(),
                          it.second.val_))
        it.second.update_func_();
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        *it.second.val_ = it.second.default_val_;
        it.second.update_func_();
      }
    }
  }
  for (auto it : group.float_map) {
    ImGui::PushItemWidth(100);

    if (it.second.has_node_) {
      float val = it.second.node_.as<float>();

      bool updated = true;
      if ((it.second.max_val_ - it.second.min_val_) / it.second.step_ <
          slider_thresh) {
        updated = ImGui::SliderFloat(
            ("##" + it.first + "##" + window_->name_).c_str(), &val,
            it.second.step_, it.second.min_val_);

      } else {
        updated = ImGui::DragFloat(
            ("##" + it.first + "##" + window_->name_).c_str(), &val,
            it.second.step_, it.second.min_val_, it.second.max_val_);
      }
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        it.second.node_ = it.second.default_val_;
        updated = true;
      }

      if (updated) {
        it.second.node_ = val;
        it.second.update_func_();
      }
    } else {
      bool updated = true;
      if ((it.second.max_val_ - it.second.min_val_) / it.second.step_ <
          slider_thresh) {
        updated = ImGui::SliderFloat(
            ("##" + it.first + "##" + window_->name_).c_str(), it.second.val_,
            it.second.step_, it.second.min_val_);

      } else {
        updated = ImGui::DragFloat(
            ("##" + it.first + "##" + window_->name_).c_str(), it.second.val_,
            it.second.step_, it.second.min_val_, it.second.max_val_);
      }
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        *it.second.val_ = it.second.default_val_;
        updated = true;
      }

      if (updated) it.second.update_func_();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::TextWrapped("%s", (it.first).c_str());
  }
  for (auto it : group.double_map) {
    double min = it.second.min_val_;
    double max = it.second.max_val_;

    ImGui::PushItemWidth(100);

    if (it.second.has_node_) {
      double val = it.second.node_.as<double>();
      bool updated = true;
      if ((it.second.max_val_ - it.second.min_val_) / it.second.step_ <
          slider_thresh) {
        updated = ImGui::SliderScalar(
            ("##" + it.first + "##" + window_->name_).c_str(),
            ImGuiDataType_Double, &val, &min, &max);
      } else {
        updated = ImGui::DragScalar(
            ("##" + it.first + "##" + window_->name_).c_str(),
            ImGuiDataType_Double, &val, it.second.step_, &min, &max);
      }
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        it.second.node_ = it.second.default_val_;
        updated = true;
      }

      if (updated) {
        it.second.node_ = val;
        it.second.update_func_();
      }

    } else {
      bool updated = true;
      if ((it.second.max_val_ - it.second.min_val_) / it.second.step_ <
          slider_thresh) {
        updated = ImGui::SliderScalar(
            ("##" + it.first + "##" + window_->name_).c_str(),
            ImGuiDataType_Double, it.second.val_, &min, &max);
      } else {
        updated = ImGui::DragScalar(
            ("##" + it.first + "##" + window_->name_).c_str(),
            ImGuiDataType_Double, it.second.val_, it.second.step_, &min, &max);
      }
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        *it.second.val_ = it.second.default_val_;
        updated = true;
      }

      if (updated) it.second.update_func_();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::TextWrapped("%s", (it.first).c_str());
  }
  for (auto it : group.int_map) {
    ImGui::PushItemWidth(100);

    if (it.second.has_node_) {
      int val = it.second.node_.as<int>();

      bool updated = true;
      if (it.second.step_ == 1 &&
          (it.second.max_val_ - it.second.min_val_) / it.second.step_ <
              slider_thresh) {
        updated =
            ImGui::SliderInt(("##" + it.first + "##" + window_->name_).c_str(),
                             &val, it.second.min_val_, it.second.max_val_);
      } else {
        updated = ImGui::DragInt(
            ("##" + it.first + "##" + window_->name_).c_str(), &val,
            it.second.step_, it.second.min_val_, it.second.max_val_);
      }
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        it.second.node_ = it.second.default_val_;
        updated = true;
      }

      if (updated) {
        it.second.node_ = val;
        it.second.update_func_();
      }
    } else {
      bool updated = true;
      if ((it.second.max_val_ - it.second.min_val_) / it.second.step_ <
          slider_thresh) {
        updated = ImGui::SliderInt(
            ("##" + it.first + "##" + window_->name_).c_str(), it.second.val_,
            it.second.min_val_, it.second.max_val_);
      } else {
        updated = ImGui::DragInt(
            ("##" + it.first + "##" + window_->name_).c_str(), it.second.val_,
            it.second.step_, it.second.min_val_, it.second.max_val_);
      }
      ImGui::SameLine();
      if (ImGui::Button(("R##" + it.first).c_str())) {
        *it.second.val_ = it.second.default_val_;
        updated = true;
      }

      if (updated) it.second.update_func_();
    }

    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::TextWrapped("%s", (it.first).c_str());
  }
  for (auto it : group.vec2_map) {
    ImGui::PushItemWidth(100);

    if (ImGui::SliderFloat2(("##" + it.first + "##" + window_->name_).c_str(),
                            it.second.val_->data(), it.second.min_val_,
                            it.second.max_val_))
      it.second.update_func_();
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::TextWrapped("%s", (it.first).c_str());

    ImGui::SameLine();
    if (ImGui::Button(("R##" + it.first).c_str())) {
      *it.second.val_ = it.second.default_val_;
      it.second.update_func_();
    }
  }
  for (auto it : group.vec3_map) {
    ImGui::PushItemWidth(100);
    if (!it.second.is_color_) {
      if (ImGui::SliderFloat3(("##" + it.first + "##" + window_->name_).c_str(),
                              it.second.val_->data(), it.second.min_val_,
                              it.second.max_val_))
        it.second.update_func_();

    } else {
      if (ImGui::ColorEdit3(("##" + it.first + "##" + window_->name_).c_str(),
                            it.second.val_->data()))
        it.second.update_func_();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::TextWrapped("%s", (it.first).c_str());

    ImGui::SameLine();
    if (ImGui::Button(("R##" + it.first).c_str())) {
      *it.second.val_ = it.second.default_val_;
      it.second.update_func_();
    }
  }
  for (auto it : group.vec4_map) {
    ImGui::PushItemWidth(100);
    if (!it.second.is_color_) {
      if (ImGui::SliderFloat4(("##" + it.first + "##" + window_->name_).c_str(),
                              it.second.val_->data(), it.second.min_val_,
                              it.second.max_val_))
        it.second.update_func_();
    } else {
      if (ImGui::ColorEdit4(("##" + it.first + "##" + window_->name_).c_str(),
                            it.second.val_->data()))
        it.second.update_func_();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::TextWrapped("%s", (it.first).c_str());

    ImGui::SameLine();
    if (ImGui::Button(("R##" + it.first).c_str())) {
      *it.second.val_ = it.second.default_val_;
      it.second.update_func_();
    }
  }
  for (auto it : group.string_map) {
    ImGui::TextColored(ImVec4(73 / 255.f, 165 / 255.f, 227 / 255.f, 1),
                       "%s: ", it.first.c_str());
    ImGui::SameLine();
    ImGui::TextWrapped("%s", it.second.c_str());

    // ImGui::Text("%s: %s", it.first.c_str(), it.second.c_str());
  }
}

void reclib::opengl::ConfigViewer::display_config() const {
  if (ImGui::Button("Save to YAML")) {
    YAML::Node node = to_yaml();
    std::ofstream fout(fs::current_path() / fs::path("config.yaml"));
    fout << node;
    fout.close();
    std::cout << "Saved config to: "
              << fs::current_path() / fs::path("config.yaml") << std::endl;
  }

  for (auto it : configs) {
    if (it.first.compare("default") == 0) {
      display_config_group(it.second);

    } else {
      if (ImGui::CollapsingHeader((it.first + "##" + window_->name_).c_str())) {
        ImGui::Indent();
        display_config_group(it.second);
        ImGui::Unindent();
      }
    }
  }

  if (config_map.size() > 0) {
    if (ImGui::CollapsingHeader(("Config Files##" + window_->name_).c_str())) {
      for (auto it : config_map) {
        if (ImGui::Button(
                ("Reload##" + it.first + "##" + window_->name_).c_str())) {
          it.second->reload();
        }
        ImGui::SameLine();
        if (ImGui::Button(
                ("Save##" + it.first + "##" + window_->name_).c_str())) {
          it.second->to_file();
        }
        if (ImGui::CollapsingHeader(
                (it.first + "##" + window_->name_).c_str())) {
          for (auto it_yaml : it.second->yaml_) {
            display_yaml(it_yaml.first.as<std::string>(), it_yaml.second);
          }
        }
      }
    }
  }
}

// -------------------------------------------
// ImageViewer
// -------------------------------------------
void reclib::opengl::ImageViewer::display_images() {
  if (ImGui::CollapsingHeader(("Textures##" + window_->name_).c_str())) {
    for (auto it : tex_map) {
      reclib::opengl::Texture2D tex = it.second.val_;

      ImGui::Image((ImTextureID)tex->id,
                   ImVec2(it.second.size_.x(), it.second.size_.y()),
                   ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1),
                   ImVec4(1, 1, 1, 0.5));

      ImGui::Text("%s", it.first.c_str());
      ImGui::SameLine();
      if (ImGui::Button(("Save PNG##" + tex->name).c_str()))
        tex->save_png(fs::path(tex->name).replace_extension(".png"));

      ImGui::Separator();
    }
  }
#if HAS_OPENCV_MODULE
  if (ImGui::CollapsingHeader(("OpenCV Mats##" + window_->name_).c_str())) {
    for (auto it : mat_map) {
      CpuMat mat = it.second.val_.clone();
      if (mat.rows == 0 && mat.cols == 0) continue;
      if (mat.depth() == CV_8U) {
        mat.convertTo(mat, CV_32FC(mat.channels()));
      }
      _RECLIB_ASSERT_EQ(mat.depth(), CV_32F);

      if (!reclib::opengl::Texture2D::valid(it.first)) {
        if (mat.channels() == 1) {
          reclib::opengl::Texture2D t(it.first, mat.cols, mat.rows, GL_R32F,
                                      GL_RED, GL_FLOAT);
        } else if (mat.channels() == 3) {
          reclib::opengl::Texture2D t(it.first, mat.cols, mat.rows, GL_RGB32F,
                                      GL_RGB, GL_FLOAT);
        }
      }
      if (it.second.changed_) {
        reclib::opengl::Texture2D t = reclib::opengl::Texture2D::find(it.first);
        t->load(mat, false);
        it.second.changed_ = false;
      }
      reclib::opengl::Texture2D t = reclib::opengl::Texture2D::find(it.first);

      ImGui::Image(
          (ImTextureID)t->id, ImVec2(it.second.size_.x(), it.second.size_.y()),
          ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 0.5));
      ImGui::Text("%s", it.first.c_str());
      ImGui::SameLine();
      if (ImGui::Button(("Save PNG##" + t->name).c_str()))
        t->save_png(fs::path(t->name).replace_extension(".png"));

      ImGui::Separator();
    }
  }
#endif
}

// -------------------------------------------
// DockspaceWindowBase
// -------------------------------------------
void reclib::opengl::DockspaceWindowBase::create() {
  ImGuiIO& io = ImGui::GetIO();
  flags_ |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
  flags_ |=
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  flags_ |= ImGuiWindowFlags_NoScrollbar;
  flags_ |= ImGuiWindowFlags_NoDocking;

  initialized_ = true;
}
void reclib::opengl::DockspaceWindowBase::pre() {}

void reclib::opengl::DockspaceWindowBase::post() {}

void reclib::opengl::DockspaceWindowBase::draw_always() {
  ImGuiIO& io = ImGui::GetIO();
  if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
    dockspace_id_ = ImGui::GetID(("Docknode_" + name_).c_str());
    ImGui::DockSpace(dockspace_id_, size_, dockspace_flags_);
  }
}

void reclib::opengl::DockspaceWindowBase::draw_() {
  user_update_func_(*this);

  for (unsigned int i = 0; i < child_windows_.size(); i++) {
    child_windows_[i]->draw();
  }
}

void reclib::opengl::DockspaceWindowBase::destroy() {}

// -------------------------------------------
// Default window creation
// -------------------------------------------

static char filepath_buf[512];

void reclib::opengl::create_default_windows(bool left_dock, bool bottom_dock,
                                            bool right_dock) {
  const ImGuiViewport* viewport = ImGui::GetMainViewport();

  int window_size = 400;
  ImVec2 left_size = viewport->WorkSize;
  ImVec2 left_pos = viewport->WorkPos;
  left_pos.y = 20;
  left_size.x = window_size;
  left_size.y = viewport->WorkSize.y - window_size - 20;
  reclib::opengl::DockspaceWindow docking_window_left(
      "Docking Window##Left", false, left_pos, left_size);
  docking_window_left->show_window_ = left_dock;

  ImVec2 right_size = viewport->WorkSize;
  right_size.x = window_size - 20;
  right_size.y = viewport->WorkSize.y - window_size - 20;
  ImVec2 right_pos = viewport->WorkPos;
  right_pos.x += viewport->WorkSize.x - right_size.x;
  right_pos.y = 20;
  reclib::opengl::DockspaceWindow docking_window_right(
      "Docking Window##Right", false, right_pos, right_size);
  docking_window_right->show_window_ = right_dock;

  ImVec2 bottom_size = viewport->WorkSize;
  bottom_size.y = window_size;
  ImVec2 bottom_pos = viewport->WorkPos;
  bottom_pos.y = viewport->WorkSize.y - bottom_size.y;
  reclib::opengl::DockspaceWindow docking_window_bottom(
      "Docking Window##Bottom", false, bottom_pos, bottom_size);
  docking_window_bottom->show_window_ = bottom_dock;

  for (unsigned int i = 0;
       i < reclib::opengl::IMGUI_DEFAULT_CONFIG.parent_path().string().length();
       i++) {
    filepath_buf[i] = reclib::opengl::IMGUI_DEFAULT_CONFIG.string()[i];
  }

  {
    OptionWindow window(
        "CPUGPUTimer##Default", false,
        []() {
          ImGui::PushStyleVar(ImGuiStyleVar_Alpha, .9f);
          for (const auto& item : TimerQuery::map) {
            ImGui::Separator();
            gui_display_query_timer(*item.second, item.first.c_str());
          }
          for (const auto& item : TimerQueryGL::map) {
            ImGui::Separator();
            gui_display_query_timer(*item.second, item.first.c_str());
          }
          for (const auto& item : PrimitiveQueryGL::map) {
            ImGui::Separator();
            gui_display_query_counter(*item.second, item.first.c_str());
          }
          for (const auto& item : FragmentQueryGL::map) {
            ImGui::Separator();
            gui_display_query_counter(*item.second, item.first.c_str());
          }
          ImGui::PopStyleVar();
        },
        ImGuiWindowFlags_NoScrollbar);
  }
  {
    OptionWindow window("CurrentCamera##Default", false, []() {
      ImGui::Text("Current: %s", current_camera()->name.c_str());
      ImGui::Indent();
      ImGui::SliderFloat(
          "Movement Speed: ",
          &reclib::opengl::CameraImpl::default_camera_movement_speed, 0.01f,
          1.5f);
      auto cam = current_camera();
      gui_display_camera(cam);
    });
  }
  {
    OptionWindow window("Cameras##Default", false, []() {
      ImGui::Text("Current: %s", current_camera()->name.c_str());
      for (auto& item : Camera::map) {
        if (ImGui::CollapsingHeader(item.first.c_str()))
          gui_display_camera(item.second);
      }
    });
  }
  {
    OptionWindow window("Textures##Default", false, []() {
      for (const auto& item : Texture2D::map) {
        if (ImGui::CollapsingHeader(item.first.c_str()))
          gui_display_texture(item.second, ivec2(300, 300));
      }
    });
  }
  {
    OptionWindow window("Framebuffers##Default", false, []() {
      for (const auto& item : Framebuffer::map)
        if (ImGui::CollapsingHeader(item.first.c_str()))
          gui_display_framebuffer(item.second);
    });
  }
  {
    OptionWindow window("Meshes##Default", false, []() {
      for (auto& item : Mesh::map)
        if (ImGui::CollapsingHeader(item.first.c_str()))
          gui_display_mesh(item.second);
    });
  }
  {
    OptionWindow window("Shader##Default", false, []() {
      for (auto& item : Shader::map)
        if (ImGui::CollapsingHeader(item.first.c_str()))
          gui_display_shader(item.second);
      if (ImGui::Button("Reload modified")) reload_modified_shaders();
    });
  }
  {
    OptionWindow window(
        "Materials##Default", false,
        []() {
          for (auto& item : Material::map)
            if (ImGui::CollapsingHeader(item.first.c_str()))
              gui_display_material(item.second);
        },
        ImGuiWindowFlags_NoResize);
  }
  {
    OptionWindow window("Geometries##Default", false, []() {
      for (auto& item : GeometryBase::map)
        if (ImGui::CollapsingHeader(item.first.c_str()))
          gui_display_geometry(item.second);
    });
  }
  {
    OptionWindow window("Drawelements##Default", false, []() {
      for (auto& item : Drawelement::map) {
        if (item.second->is_grouped) continue;
        if (ImGui::CollapsingHeader(item.first.c_str())) {
          if (item.second->is_grouped) continue;
          gui_display_drawelement(item.second);
        }
      }
    });
  }

  {
    OptionWindow window("GroupedDrawelements##Default", false, []() {
      for (auto& item : GroupedDrawelements::map) {
        if (ImGui::CollapsingHeader(item.first.c_str())) {
          gui_display_grouped_drawelements(item.second);
        }
      }
    });
  }

  if (0) {
    RenderWindow window("GL Example##Default", false, []() {
      if (!reclib::opengl::Drawelement::valid("test_cube")) {
        reclib::opengl::Material m("test_cube");
        m->vec4_map["color"] = vec4(1, 0, 0, 1);
        reclib::opengl::Cuboid c("test_cube", 1, 1, 1, vec3(0.5, 0.5, 0.5), m);
        if (!reclib::opengl::Shader::valid("test_shader")) {
          reclib::opengl::Shader("test_shader", "MVP_norm.vs",
                                 "color4Uniform.fs");
        }
        reclib::opengl::Drawelement elem(
            "test_cube", reclib::opengl::Shader::find("test_shader"), c);
      }
      {
        reclib::opengl::Drawelement d =
            reclib::opengl::Drawelement::find("test_cube");
        d->bind();
        d->draw();
        d->unbind();
      }
      return true;
    });
  }
}

// -------------------------------------------
// callbacks

static std::map<std::string, std::function<void()>> gui_callbacks;

void reclib::opengl::imgui_resize_callback() {
  const ImGuiViewport* viewport = ImGui::GetMainViewport();

  int window_size = 400;
  ImVec2 left_size = viewport->WorkSize;
  ImVec2 left_pos = viewport->WorkPos;
  left_pos.y = 20;
  left_size.x = window_size;
  left_size.y = viewport->Size.y - window_size - 20;
  if (imgui_window_name_to_id.find("Docking Window##Bottom") !=
      imgui_window_name_to_id.end()) {
    std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
        [imgui_window_name_to_id["Docking Window##Bottom"]];
    if (!window->show_window_) {
      left_size.y = viewport->Size.y;
    }
  }

  if (imgui_window_name_to_id.find("Docking Window##Left") !=
      imgui_window_name_to_id.end()) {
    std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
        [imgui_window_name_to_id["Docking Window##Left"]];
    window->set_pos(left_pos);
    window->set_size(left_size);
  }

  ImVec2 bottom_size = viewport->WorkSize;
  bottom_size.y = window_size;
  ImVec2 bottom_pos = viewport->WorkPos;
  bottom_pos.y = viewport->Size.y - bottom_size.y;
  if (imgui_window_name_to_id.find("Docking Window##Bottom") !=
      imgui_window_name_to_id.end()) {
    std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
        [imgui_window_name_to_id["Docking Window##Bottom"]];
    window->set_pos(bottom_pos);
    window->set_size(bottom_size);
  }

  ImVec2 right_size = viewport->WorkSize;
  right_size.x = window_size - 20;
  right_size.y = viewport->Size.y - window_size - 20;
  ImVec2 right_pos = viewport->WorkPos;
  right_pos.x += viewport->WorkSize.x - right_size.x;
  right_pos.y = 20;
  if (imgui_window_name_to_id.find("Docking Window##Bottom") !=
      imgui_window_name_to_id.end()) {
    std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
        [imgui_window_name_to_id["Docking Window##Bottom"]];
    if (!window->show_window_) {
      right_size.y = viewport->Size.y;
    }
  }

  if (imgui_window_name_to_id.find("Docking Window##Right") !=
      imgui_window_name_to_id.end()) {
    std::shared_ptr<ImguiWindowBase> window = imgui_dockspace_windows
        [imgui_window_name_to_id["Docking Window##Right"]];
    window->set_pos(right_pos);
    window->set_size(right_size);
  }

  reclib::opengl::MainWindow::resize();
}

void reclib::opengl::gui_add_callback(const std::string& name,
                                      std::function<void()> fn) {
  gui_callbacks[name] = fn;
}

void reclib::opengl::gui_remove_callback(const std::string& name) {
  gui_callbacks.erase(name);
}

// -------------------------------------------
// main setup, draw and shutdown routines

void reclib::opengl::set_imgui_theme() {
  ImGui::StyleColorsDark();

  ImGuiStyle& style = ImGui::GetStyle();

  auto& colors = ImGui::GetStyle().Colors;
  colors[ImGuiCol_WindowBg] = ImVec4{0.1f, 0.105f, 0.11f, 1.0f};

  // Headers
  colors[ImGuiCol_Header] = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
  colors[ImGuiCol_HeaderHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
  colors[ImGuiCol_HeaderActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

  // Buttons
  colors[ImGuiCol_Button] = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
  colors[ImGuiCol_ButtonHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
  colors[ImGuiCol_ButtonActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

  // Frame BG
  colors[ImGuiCol_FrameBg] = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
  colors[ImGuiCol_FrameBgHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
  colors[ImGuiCol_FrameBgActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

  // Tabs
  colors[ImGuiCol_Tab] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
  colors[ImGuiCol_TabHovered] = ImVec4{0.38f, 0.3805f, 0.381f, 1.0f};
  colors[ImGuiCol_TabActive] = ImVec4{0.28f, 0.2805f, 0.281f, 1.0f};
  colors[ImGuiCol_TabUnfocused] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};

  // Title
  colors[ImGuiCol_TitleBg] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
  colors[ImGuiCol_TitleBgActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

  style.FrameBorderSize = 1.f;
  style.FramePadding = ImVec2(5.f, 2.f);
  style.FrameRounding = 6;
  style.GrabRounding = 6;
}

static bool show_config_window = false;

void reclib::opengl::gui_draw_toggle() {
  imgui_resize_callback();
  // ImGui::ShowDemoWindow();

  if (ImGui::BeginMainMenuBar()) {
    // camera menu
    ImGui::Checkbox("Left Dock",
                    &imgui_dockspace_windows
                         [imgui_window_name_to_id["Docking Window##Left"]]
                             ->show_window_);
    ImGui::Separator();
    ImGui::Checkbox("Bottom Dock",
                    &imgui_dockspace_windows
                         [imgui_window_name_to_id["Docking Window##Bottom"]]
                             ->show_window_);
    ImGui::Separator();
    ImGui::Checkbox("Right Dock",
                    &imgui_dockspace_windows
                         [imgui_window_name_to_id["Docking Window##Right"]]
                             ->show_window_);
    ImGui::Separator();

    if (ImGui::BeginMenu("Windows")) {
      for (auto it : imgui_toggle_windows) {
        ImGui::MenuItem(it.second->name_.c_str(), nullptr,
                        &it.second->show_window_);
      }
      ImGui::EndMenu();
    }
    if (ImGui::Button("Load Configs")) {
      show_config_window = !show_config_window;
    }
    if (show_config_window) {
      if (ImGui::Begin("Configs", &show_config_window)) {
        ImGui::SetWindowSize(ImVec2(600, 200));
        if (ImGui::Button("Default")) {
          ImGui::LoadIniSettingsFromDisk(
              (fs::path(RECLIB_LIBRARY_SOURCE_PATH) / fs::path("include") /
               fs::path("reclib") / fs::path("opengl") /
               fs::path("imgui_default.ini"))
                  .c_str());
        }
        ImGui::Separator();
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.9f);
        bool changed = ImGui::InputText(
            "Path", filepath_buf, 512,
            ImGuiInputTextFlags_CallbackCompletion |
                ImGuiInputTextFlags_EnterReturnsTrue,
            [](ImGuiInputTextCallbackData* data) -> int {
              if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
                fs::path path = filepath_buf;
                fs::path next;
                if (fs::exists(path)) {
                  fs::path parent = path.parent_path();
                  fs::directory_iterator iter(parent);
                  next = iter->path();

                  for (; iter != fs::directory_iterator(); iter++) {
                    if (!iter->is_directory()) continue;
                    if (iter->path().compare(path) == 0) {
                      next = (++iter)->path();
                      break;
                    }
                  }
                  data->DeleteChars(0, data->BufTextLen);
                  data->InsertChars(0, next.c_str());
                }
              }
              return 0;
            });

        if (ImGui::BeginChild("Files")) {
          fs::path path = filepath_buf;
          if (fs::exists(path)) {
            fs::directory_iterator iter(path);
            for (; iter != fs::directory_iterator(); iter++) {
              if (!iter->is_regular_file()) continue;
              if (!iter->path().has_extension()) continue;
              if (!(iter->path().extension().compare(".ini") == 0)) continue;
              if (ImGui::MenuItem(iter->path().stem().c_str())) {
                ImGui::LoadIniSettingsFromDisk(iter->path().c_str());
                imgui_resize_callback();
              }
            }
          }
          ImGui::EndChild();
        }
      }
      ImGui::End();
    }

    if (ImGui::Button("Screenshot")) Context::screenshot("screenshot.png");
    if (ImGui::Button("Resize")) imgui_resize_callback();
    if (ImGui::Button("Save Config")) {
      std::cout << "Saved to 'imgui.ini'." << std::endl;
      ImGui::SaveIniSettingsToDisk("imgui.ini");
    }
    ImGui::EndMainMenuBar();
  }

  // update main window size

  for (auto it : imgui_dockspace_windows) {
    it.second->draw();
  }
  for (auto it : imgui_toggle_windows) {
    it.second->draw();
  }

  // call callbacks
  for (const auto& item : gui_callbacks) item.second();
}

void reclib::opengl::gui_draw() {
  imgui_resize_callback();

  for (auto it : imgui_visible_windows) {
    it.second->draw();
  }

  // call callbacks
  for (const auto& item : gui_callbacks) item.second();
}

// -------------------------------------------
// helpers

void reclib::opengl::gui_display_camera(Camera& cam) {
  ImGui::Indent();
  ImGui::Text("name: %s", cam->name.c_str());
  ImGui::DragFloat3(("pos##" + cam->name).c_str(), &cam->pos.x(), 0.001f);
  ImGui::DragFloat3(("dir##" + cam->name).c_str(), &cam->dir.x(), 0.001f);
  ImGui::DragFloat3(("up##" + cam->name).c_str(), &cam->up.x(), 0.001f);
  ImGui::Checkbox(("fix_up_vector##" + cam->name).c_str(), &cam->fix_up_vector);
  ImGui::Checkbox(("perspective##" + cam->name).c_str(), &cam->perspective);
  if (cam->perspective) {
    ImGui::DragFloat(("fov##" + cam->name).c_str(), &cam->fov_degree, 0.01f);
    ImGui::DragFloat(("near##" + cam->name).c_str(), &cam->near, 0.001f);
    ImGui::DragFloat(("far##" + cam->name).c_str(), &cam->far, 0.001f);
  } else {
    ImGui::DragFloat(("left##" + cam->name).c_str(), &cam->left, 0.001f);
    ImGui::DragFloat(("right##" + cam->name).c_str(), &cam->right, 0.001f);
    ImGui::DragFloat(("top##" + cam->name).c_str(), &cam->top, 0.001f);
    ImGui::DragFloat(("bottom##" + cam->name).c_str(), &cam->bottom, 0.001f);
  }
  if (ImGui::Button(("Make current##" + cam->name).c_str()))
    make_camera_current(cam);
  if (ImGui::Button(("Store##" + cam->name).c_str())) cam->store();
  if (ImGui::Button(("Load##" + cam->name).c_str())) cam->load_snapshot();
  ImGui::Unindent();
}

void reclib::opengl::gui_display_texture(const Texture2D& tex,
                                         const ivec2& size) {
  ImGui::Indent();
  ImGui::Text("name: %s", tex->name.c_str());
  ImGui::Text("ID: %u, size: %ux%u", tex->id, tex->w, tex->h);
  ImGui::Text("internal_format: %u, format %u, type: %u", tex->internal_format,
              tex->format, tex->type);
  ImGui::Image((ImTextureID)tex->id, ImVec2(size.x(), size.y()), ImVec2(0, 1),
               ImVec2(1, 0), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 0.5));
  if (ImGui::Button(("Save PNG##" + tex->name).c_str()))
    tex->save_png(fs::path(tex->name).replace_extension(".png"));
  ImGui::SameLine();
  if (ImGui::Button(("Save JPEG##" + tex->name).c_str()))
    tex->save_jpg(fs::path(tex->name).replace_extension(".jpg"));
  ImGui::Unindent();
}

void reclib::opengl::gui_display_shader(Shader& shader) {
  ImGui::Indent();
  ImGui::Text("name: %s", shader->name.c_str());
  ImGui::Text("ID: %u", shader->id);
  if (shader->source_files.count(GL_VERTEX_SHADER))
    ImGui::Text("vertex source: %s",
                shader->source_files[GL_VERTEX_SHADER].string().c_str());
  if (shader->source_files.count(GL_TESS_CONTROL_SHADER))
    ImGui::Text("tess_control source: %s",
                shader->source_files[GL_TESS_CONTROL_SHADER].string().c_str());
  if (shader->source_files.count(GL_TESS_EVALUATION_SHADER))
    ImGui::Text(
        "tess_eval source: %s",
        shader->source_files[GL_TESS_EVALUATION_SHADER].string().c_str());
  if (shader->source_files.count(GL_GEOMETRY_SHADER))
    ImGui::Text("geometry source: %s",
                shader->source_files[GL_GEOMETRY_SHADER].string().c_str());
  if (shader->source_files.count(GL_FRAGMENT_SHADER))
    ImGui::Text("fragment source: %s",
                shader->source_files[GL_FRAGMENT_SHADER].string().c_str());
  if (shader->source_files.count(GL_COMPUTE_SHADER))
    ImGui::Text("compute source: %s",
                shader->source_files[GL_COMPUTE_SHADER].string().c_str());
  if (ImGui::Button("Compile")) shader->compile();
  ImGui::Unindent();
}

void reclib::opengl::gui_display_framebuffer(const Framebuffer& fbo) {
  ImGui::Indent();
  ImGui::Text("name: %s", fbo->name.c_str());
  ImGui::Text("ID: %u", fbo->id);
  ImGui::Text("size: %ux%u", fbo->w, fbo->h);
  if (ImGui::CollapsingHeader(("depth attachment##" + fbo->name).c_str()) &&
      fbo->depth_texture)
    gui_display_texture(fbo->depth_texture);
  for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
    if (ImGui::CollapsingHeader(std::string("color attachment " +
                                            std::to_string(i) + "##" +
                                            fbo->name)
                                    .c_str()))
      gui_display_texture(fbo->color_textures[i]);
  ImGui::Unindent();
}

void reclib::opengl::gui_display_material(Material& mat) {
  ImGui::Indent();
  ImGui::Text("name: %s", mat->name.c_str());

  ImGui::Text("int params: %lu", mat->int_map.size());
  ImGui::Indent();
  for (auto& entry : mat->int_map)
    ImGui::SliderInt((mat->name + "##" + entry.first).c_str(), &entry.second,
                     -1, 1);
  // ImGui::Text("%s: %i", entry.first.c_str(), entry.second);
  ImGui::Unindent();

  ImGui::Text("float params: %lu", mat->float_map.size());
  ImGui::Indent();
  for (auto& entry : mat->float_map)
    ImGui::SliderFloat((mat->name + "##" + entry.first).c_str(), &entry.second,
                       -1, 1);
  // ImGui::Text("%s: %f", entry.first.c_str(), entry.second);
  ImGui::Unindent();

  ImGui::Text("vec2 params: %lu", mat->vec2_map.size());
  ImGui::Indent();
  for (auto& entry : mat->vec2_map)
    ImGui::SliderFloat2((mat->name + "##" + entry.first).c_str(),
                        entry.second.data(), -1, 1);
  // ImGui::Text("%s: (%f, %f)", entry.first.c_str(), entry.second.x(),
  //            entry.second.y());
  ImGui::Unindent();

  ImGui::Text("vec3 params: %lu", mat->vec3_map.size());
  ImGui::Indent();
  for (auto& entry : mat->vec3_map) {
    ImGui::ColorEdit3((mat->name + "##" + entry.first).c_str(),
                      entry.second.data());
  }

  // ImGui::Text("%s: (%f, %f, %f)", entry.first.c_str(), entry.second.x(),
  //             entry.second.y(), entry.second.z());
  ImGui::Unindent();

  ImGui::Text("vec4 params: %lu", mat->vec4_map.size());
  ImGui::Indent();
  for (auto& entry : mat->vec4_map) {
    ImGui::ColorEdit4((mat->name + "##" + entry.first).c_str(),
                      entry.second.data());
  }
  // ImGui::Text("%s: (%f, %f, %f, %.f)", entry.first.c_str(), entry.second.x(),
  //             entry.second.y(), entry.second.z(), entry.second.w());
  ImGui::Unindent();

  ImGui::Text("textures: %lu", mat->texture_map.size());
  ImGui::Indent();
  for (const auto& entry : mat->texture_map) {
    ImGui::Text("%s:", entry.first.c_str());
    gui_display_texture(entry.second);
  }
  ImGui::Unindent();

  ImGui::Unindent();
}

void reclib::opengl::gui_display_geometry(GeometryBase& geom) {
  ImGui::Indent();
  ImGui::Text("name: %s", geom->name.c_str());
  ImGui::Text("AABB min: (%.3f, %.3f, %.3f)", geom->bb_min.x(),
              geom->bb_min.y(), geom->bb_min.z());
  ImGui::Text("AABB max: (%.3f, %.3f, %.3f)", geom->bb_max.x(),
              geom->bb_max.y(), geom->bb_max.z());
  ImGui::Text("#Vertices: %lu", geom->positions_size());
  ImGui::Text("#Indices: %lu", geom->indices_size());
  ImGui::Text("#Normals: %lu", geom->normals_size());
  ImGui::Text("#Texcoords: %zu", geom->texcoords_size());
  ImGui::Text("float params: %lu", geom->float_map.size());
  ImGui::Indent();
  for (auto& it : geom->float_map) {
    ImGui::Text(("#" + it.first + ": %lux%lu").c_str(), it.second.size,
                it.second.dim);
  }
  ImGui::Unindent();
  ImGui::Text("int params: %lu", geom->int_map.size());
  ImGui::Indent();
  for (auto& it : geom->int_map) {
    ImGui::Text(("#" + it.first + ": %lux%lu").c_str(), it.second.size,
                it.second.dim);
  }
  ImGui::Unindent();
  ImGui::Text("uint params: %lu", geom->uint_map.size());
  ImGui::Indent();
  for (auto& it : geom->uint_map) {
    ImGui::Text(("#" + it.first + ": %lux%lu").c_str(), it.second.size,
                it.second.dim);
  }
  ImGui::Unindent();
  ImGui::Text("vec2 params: %lu", geom->vec2_map.size());
  ImGui::Indent();
  for (auto& it : geom->vec2_map) {
    ImGui::Text(("#" + it.first + ": %lu").c_str(), it.second.size);
  }
  ImGui::Unindent();
  ImGui::Text("vec3 params: %lu", geom->vec3_map.size());
  ImGui::Indent();
  for (auto& it : geom->vec3_map) {
    ImGui::Text(("#" + it.first + ": %lu").c_str(), it.second.size);
  }
  ImGui::Unindent();
  ImGui::Text("vec4 params: %lu", geom->vec4_map.size());
  ImGui::Indent();
  for (auto& it : geom->vec4_map) {
    ImGui::Text(("#" + it.first + ": %lu").c_str(), it.second.size);
  }
  ImGui::Unindent();
  if (ImGui::Button("Update Meshes")) {
    geom->update_meshes();
  }
  ImGui::Unindent();
}

void reclib::opengl::gui_display_mesh(Mesh& mesh) {
  ImGui::Indent();
  ImGui::Text("name: %s", mesh->name.c_str());
  if (ImGui::CollapsingHeader(("geometry: " + mesh->geometry->name).c_str()))
    gui_display_geometry(mesh->geometry);
  if (mesh->material.initialized()) {
    if (ImGui::CollapsingHeader(("material: " + mesh->material->name).c_str()))
      gui_display_material(mesh->material);
  }

  ImGui::Unindent();
}

void reclib::opengl::gui_display_mat4(mat4& mat) {
  ImGui::Indent();
  mat4 row_maj = transpose(mat);
  bool modified = false;
  if (ImGui::DragFloat4("row0", &row_maj(0, 0), .01f)) modified = true;
  if (ImGui::DragFloat4("row1", &row_maj(0, 1), .01f)) modified = true;
  if (ImGui::DragFloat4("row2", &row_maj(0, 2), .01f)) modified = true;
  if (ImGui::DragFloat4("row3", &row_maj(0, 3), .01f)) modified = true;
  if (modified) mat = transpose(row_maj);
  ImGui::Unindent();
}

void reclib::opengl::gui_display_drawelement(Drawelement elem) {
  ImGui::Indent();
  ImGui::Text("name: %s", elem->name.c_str());
  if (ImGui::CollapsingHeader(("modelmatrix##" + elem->name).c_str()))
    gui_display_mat4(elem->model);
  if (ImGui::CollapsingHeader(("shader##" + elem->shader->name).c_str()))
    gui_display_shader(elem->shader);
  if (ImGui::CollapsingHeader(("mesh##" + elem->mesh->name).c_str()))
    gui_display_mesh(elem->mesh);

  if (ImGui::Button(("View Mesh##" + elem->name).c_str())) {
    if (imgui_window_name_to_id.find(elem->name) !=
        imgui_window_name_to_id.end()) {
      std::shared_ptr<ImguiWindowBase> window =
          imgui_visible_windows[imgui_window_name_to_id[elem->name]];
      window->show_window_ = true;
    } else {
      reclib::opengl::MeshViewer(elem->name, {elem->name});
    }
  }

  ImGui::SameLine();
  ImGui::Checkbox(("Show wireframe##" + elem->name).c_str(),
                  &elem->wireframe_mode);
  ImGui::SameLine();
  ImGui::Checkbox(("Disable##" + elem->name).c_str(), &elem->disable_render);

  ImGui::Unindent();
}

void reclib::opengl::gui_display_grouped_drawelements(
    GroupedDrawelements elem) {
  ImGui::Indent();
  ImGui::Text("name: %s", elem->name.c_str());
  if (ImGui::CollapsingHeader(("drawelements##" + elem->name).c_str())) {
    ImGui::Indent();
    for (auto e : elem->elems) {
      if (ImGui::CollapsingHeader(e->name.c_str())) {
        gui_display_drawelement(e);
      }
    }
    ImGui::Unindent();
  }

  if (ImGui::Button(("View Mesh##" + elem->name).c_str())) {
    if (imgui_window_name_to_id.find(elem->name) !=
        imgui_window_name_to_id.end()) {
      std::shared_ptr<ImguiWindowBase> window =
          imgui_visible_windows[imgui_window_name_to_id[elem->name]];
      window->show_window_ = true;
    } else {
      std::vector<std::string> elem_names;
      for (auto e : elem->elems) {
        elem_names.push_back(e->name);
      }
      reclib::opengl::MeshViewer(elem->name, elem_names);
    }
  }

  bool wf = elem->wireframe_mode;
  if (ImGui::Checkbox(("Show wireframe##" + elem->name).c_str(), &wf)) {
    elem->wireframe_mode = !elem->wireframe_mode;
    elem->set_wireframe_mode(elem->wireframe_mode);
  }
  bool dis = elem->disable_render;
  if (ImGui::Checkbox(("Disable##" + elem->name).c_str(), &dis)) {
    elem->disable_render = !elem->disable_render;
    elem->set_disable_render(elem->disable_render);
  }

  ImGui::Unindent();
}

void reclib::opengl::gui_display_query_timer(const Query& query,
                                             const char* label) {
  const float avg = query.exp_avg;
  const float lower = query.min();
  const float upper = query.max();
  ImGui::Text("avg: %.1fms, min: %.1fms, max: %.1fms", avg, lower, upper);
  ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(.7, .7, 0, 1));
  ImGui::PlotHistogram(label, query.data.data(), query.data.size(), query.curr,
                       0, 0.f, std::max(upper, 17.f), ImVec2(0, 30));
  ImGui::PopStyleColor();
}

void reclib::opengl::gui_display_query_counter(const Query& query,
                                               const char* label) {
  const float avg = query.exp_avg;
  const float lower = query.min();
  const float upper = query.max();
  ImGui::Text("avg: %uK, min: %uK, max: %uK", uint32_t(avg / 1000),
              uint32_t(lower / 1000), uint32_t(upper / 1000));
  ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0, .7, .7, 1));
  ImGui::PlotHistogram(label, query.data.data(), query.data.size(), query.curr,
                       0, 0.f, std::max(upper, 17.f), ImVec2(0, 30));
  ImGui::PopStyleColor();
}
