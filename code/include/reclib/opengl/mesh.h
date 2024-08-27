#ifndef RECLIB_OPENGL_MESH_H
#define RECLIB_OPENGL_MESH_H

#include <memory>
#include <string>
#include <vector>

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on

#include "reclib/internal/filesystem.h"
#include "reclib/opengl/buffer.h"
#include "reclib/opengl/geometry.h"
#include "reclib/opengl/material.h"

namespace fs = std::filesystem;

// ------------------------------------------
// Mesh

namespace reclib {
namespace opengl {

class _API MeshImpl {
 public:
  // XXX: When you pass a geometry of type 'Geometry' or 'GeometryWrapper'
  // to this constructor, it will be converted to 'GeometryBase'
  // that also means, that the handle will be deleted from
  // 'Geometry'/'GeometryWrapper' and instead is now part of 'GeometryBase'
  MeshImpl(const std::string& name,
           const GeometryBase& geometry = GeometryBase(),
           const Material& material = Material());
  virtual ~MeshImpl();

  static inline std::string type_to_str() { return "MeshImpl"; }

  template <class... Args>
  static NamedHandle<MeshImpl> from_geometry(const std::string& name,
                                             const Material& material,
                                             bool generate_normals,
                                             Args&&... geometry_args) {
    Geometry geo(name + std::string("_geometry"), geometry_args...);
    if (generate_normals) geo->auto_generate_normals();
    NamedHandle<MeshImpl> m(name, geo, material);
    geo->register_mesh(m);
    return m;
  }

  template <class... Args>
  static NamedHandle<MeshImpl> from_geometry(const std::string& name,
                                             bool generate_normals,
                                             Args&&... geometry_args) {
    Geometry geo(name + std::string("_geometry"), geometry_args...);
    if (generate_normals) geo->auto_generate_normals();
    NamedHandle<MeshImpl> m(name, geo);
    geo->register_mesh(m);
    return m;
  }

  template <class... Args>
  static NamedHandle<MeshImpl> from_geometry_wrapper(const std::string& name,
                                                     const Material& material,
                                                     bool generate_normals,
                                                     Args&&... geometry_args) {
    GeometryWrapper geo(name + std::string("_geometrywrap"), geometry_args...);
    if (generate_normals) geo->auto_generate_normals();
    NamedHandle<MeshImpl> m(name, geo, material);
    geo->register_mesh(m);
    return m;
  }

  template <class... Args>
  static NamedHandle<MeshImpl> from_geometry_wrapper(const std::string& name,
                                                     bool generate_normals,
                                                     Args&&... geometry_args) {
    GeometryWrapper geo(name + std::string("_geometrywrap"), geometry_args...);
    if (generate_normals) geo->auto_generate_normals();
    NamedHandle<MeshImpl> m(name, geo);
    geo->register_mesh(m);
    return m;
  }

  // prevent copies and moves, since GL buffers aren't reference counted
  MeshImpl(const MeshImpl&) = delete;
  MeshImpl& operator=(const MeshImpl&) = delete;
  MeshImpl& operator=(const MeshImpl&&) = delete;

  void clear_gpu();   // free gpu resources
  void upload_gpu();  // cpu -> gpu transfer
  void destroy_handles();

  // call in this order to draw
  void bind(const Shader& shader) const;
  void draw() const;
  void draw_index(uint32_t index) const;
  void unbind() const;

  // GL vertex and index buffer operations
  uint32_t add_vertex_buffer(GLenum type, uint32_t element_dim,
                             uint32_t num_vertices, const void* data,
                             GLenum hint = GL_STATIC_DRAW);
  void add_index_buffer(uint32_t num_indices, const uint32_t* data,
                        GLenum hint = GL_STATIC_DRAW);
  void update_vertex_buffer(
      uint32_t buf_id, const void* data);  // assumes matching size for buffer
                                           // buf_id from add_vertex_buffer()
  void set_primitive_type(GLenum type);    // default: GL_TRIANGLES

  // map/unmap from GPU mem
  // (https://www.seas.upenn.edu/~pcozzi/OpenGLInsights/OpenGLInsights-AsynchronousBufferTransfers.pdf)
  void* map_vbo(uint32_t buf_id, GLenum access = GL_READ_WRITE) const;
  void unmap_vbo(uint32_t buf_id) const;
  void* map_ibo(GLenum access = GL_READ_WRITE) const;
  void unmap_ibo() const;

  // CPU data
  const std::string name;
  GeometryBase geometry;
  Material material;
  // GPU data
  GLuint vao;
  IBO ibo;
  uint32_t num_vertices;
  uint32_t num_indices;
  std::vector<VBO> vbos;
  std::vector<GLenum> vbo_types;
  std::vector<uint32_t> vbo_dims;
  GLenum primitive_type;
};

using Mesh = NamedHandle<MeshImpl>;

// ------------------------------------------
// Mesh loader (Ass-Imp)

std::vector<std::pair<Geometry, Material>> load_meshes_cpu(
    const fs::path& path, bool normalize = false,
    const std::string& mesh_name = "");
std::vector<Mesh> load_meshes_gpu(const fs::path& path, bool normalize = false);
// save meshes as .obj
void save_meshes_cpu(
    const std::vector<std::pair<GeometryBase, Material>>& meshes,
    const fs::path& path, const fs::path& texture_base_path,
    bool export_material = false);

void save_meshes_cpu(const std::vector<Mesh>& meshes, const fs::path& path,
                     const fs::path& texture_base_path,
                     bool export_material = false);

void save_mesh_cpu(const Mesh& meshes, const fs::path& path,
                   const fs::path& texture_base_path,
                   bool export_material = false);

}  // namespace opengl
}  // namespace reclib

template class _API reclib::opengl::NamedHandle<
    reclib::opengl::MeshImpl>;  // needed for Windows DLL export

#endif
