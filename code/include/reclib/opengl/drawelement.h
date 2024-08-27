#ifndef RECLIB_OPENGL_DRAWELEMENT_H
#define RECLIB_OPENGL_DRAWELEMENT_H

#include "reclib/data_types.h"
#include "reclib/opengl/mesh.h"
#include "reclib/opengl/named_handle.h"
#include "reclib/opengl/shader.h"

// -----------------------------------------------
// Drawelement (object instance for rendering)

namespace reclib {
namespace opengl {

class _API DrawelementImpl {
 public:
  DrawelementImpl(const std::string& name, const Shader& shader = Shader(),
                  const Mesh& mesh = Mesh(),
                  const mat4& model = mat4::Identity());
  virtual ~DrawelementImpl();

  static inline std::string type_to_str() { return "DrawelementImpl"; }

  template <class... Args>
  static NamedHandle<DrawelementImpl> from_mesh(const std::string& name,
                                                const Shader& shader = Shader(),
                                                Args&&... mesh_args) {
    Mesh mesh(name + std::string("_mesh"), mesh_args...);
    return NamedHandle<DrawelementImpl>(name, shader, mesh);
  }

  template <class... Args>
  static NamedHandle<DrawelementImpl> from_geometry(const std::string& name,
                                                    const Shader& shader,
                                                    const Material& material,
                                                    bool generate_normals,
                                                    Args&&... geometry_args) {
    Mesh mesh = MeshImpl::from_geometry(name + std::string("_mesh"), material,
                                        generate_normals, geometry_args...);
    return NamedHandle<DrawelementImpl>(name, shader, mesh);
  }

  template <class... Args>
  static NamedHandle<DrawelementImpl> from_geometry(const std::string& name,
                                                    const Shader& shader,
                                                    bool generate_normals,
                                                    Args&&... geometry_args) {
    Mesh mesh = MeshImpl::from_geometry(name + std::string("_mesh"),
                                        generate_normals, geometry_args...);
    return NamedHandle<DrawelementImpl>(name, shader, mesh);
  }

  template <class... Args>
  static NamedHandle<DrawelementImpl> from_geometry_wrapper(
      const std::string& name, const Shader& shader, const Material& material,
      bool generate_normals, Args&&... geometry_args) {
    Mesh mesh =
        MeshImpl::from_geometry_wrapper(name + std::string("_mesh"), material,
                                        generate_normals, geometry_args...);
    return NamedHandle<DrawelementImpl>(name, shader, mesh);
  }

  template <class... Args>
  static NamedHandle<DrawelementImpl> from_geometry_wrapper(
      const std::string& name, const Shader& shader, bool generate_normals,
      Args&&... geometry_args) {
    Mesh mesh = MeshImpl::from_geometry_wrapper(
        name + std::string("_mesh"), generate_normals, geometry_args...);
    return NamedHandle<DrawelementImpl>(name, shader, mesh);
  }

  void bind() const;
  void draw() const;
  void draw_wireframe() const;
  void draw_bb() const;
  void draw_index(uint32_t index) const;
  void unbind() const;
  void destroy_handles();

  void set_model_translation(const vec3& trans);
  void set_model_rotation(const mat3& rot);
  void set_wireframe_mode(bool mode);
  void set_disable_render(bool mode);

  std::map<std::string, std::function<void()>> pre_draw_funcs;
  std::map<std::string, std::function<void()>> post_draw_funcs;

  void add_pre_draw_func(const std::string& key, std::function<void()> func,
                         bool overwrite = false) {
    if (!overwrite) _RECLIB_ASSERT_EQ(pre_draw_funcs.count(key), 0);
    pre_draw_funcs[key] = func;
  }
  void add_post_draw_func(const std::string& key, std::function<void()> func,
                          bool overwrite = false) {
    if (!overwrite) _RECLIB_ASSERT_EQ(post_draw_funcs.count(key), 0);
    post_draw_funcs[key] = func;
  }

  // data
  const std::string name;
  mat4 model;
  Shader shader;
  Mesh mesh;
  bool wireframe_mode;
  bool disable_render;
  bool is_grouped;  // only used for imgui
};

using Drawelement = NamedHandle<DrawelementImpl>;

class _API GroupedDrawelementsImpl {
 private:
 public:
  GroupedDrawelementsImpl(
      const std::string& name,
      const std::vector<reclib::opengl::Drawelement>& elems);
  virtual ~GroupedDrawelementsImpl();

  static inline std::string type_to_str() { return "GroupedDrawelementsImpl"; }

  void bind_draw_unbind() const;
  void bind_draw_wireframe_unbind() const;
  void bind_draw_bb_unbind() const;
  void destroy_handles();

  void set_model_translation(const vec3& trans);
  void set_model_rotation(const mat3& rot);
  void set_wireframe_mode(bool mode);
  void set_disable_render(bool mode);

  std::map<std::string, std::function<void()>> pre_draw_funcs;
  std::map<std::string, std::function<void()>> post_draw_funcs;

  void add_pre_draw_func(const std::string& key, std::function<void()> func,
                         bool overwrite = false) {
    if (!overwrite) _RECLIB_ASSERT_EQ(pre_draw_funcs.count(key), 0);
    pre_draw_funcs[key] = func;
  }
  void add_post_draw_func(const std::string& key, std::function<void()> func,
                          bool overwrite = false) {
    if (!overwrite) _RECLIB_ASSERT_EQ(post_draw_funcs.count(key), 0);
    post_draw_funcs[key] = func;
  }

  const std::string name;
  std::vector<reclib::opengl::Drawelement> elems;
  bool wireframe_mode;
  bool disable_render;
};
using GroupedDrawelements = NamedHandle<GroupedDrawelementsImpl>;

}  // namespace opengl
}  // namespace reclib

template class _API reclib::opengl::NamedHandle<
    reclib::opengl::DrawelementImpl>;  // needed for Windows DLL export

#endif