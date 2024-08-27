#ifndef RECLIB_OPENGL_SHADER_H
#define RECLIB_OPENGL_SHADER_H

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "reclib/data_types.h"
#include "reclib/internal/filesystem.h"
#include "reclib/opengl/named_handle.h"
#include "reclib/opengl/texture.h"

namespace fs = std::filesystem;

// ------------------------------------------
// Shader
namespace reclib {
namespace opengl {

class _API ShaderImpl {
 public:
  ShaderImpl(const std::string& name);
  ShaderImpl(const std::string& name, const fs::path& compute_source);
  ShaderImpl(const std::string& name, const fs::path& vertex_source,
             const fs::path& fragment_source);
  ShaderImpl(const std::string& name, const fs::path& vertex_source,
             const fs::path& geometry_source, const fs::path& fragment_source);
  virtual ~ShaderImpl();

  // prevent copies and moves, since GL buffers aren't reference counted
  ShaderImpl(const ShaderImpl&) = delete;
  ShaderImpl(const ShaderImpl&&) = delete;  // TODO allow moves?
  ShaderImpl& operator=(const ShaderImpl&) = delete;
  ShaderImpl& operator=(const ShaderImpl&&) = delete;  // TODO allow moves?

  static inline std::string type_to_str() { return "ShaderImpl"; }

  explicit inline operator bool() const { return glIsProgram(id); }
  inline operator GLuint() const { return id; }

  // bind/unbind to/from OpenGL
  void bind() const;
  void unbind() const;

  // set the path to the source file for the shader type
  void set_source(GLenum type, const fs::path& path);
  void set_vertex_source(const fs::path& path);
  void set_tesselation_control_source(const fs::path& path);
  void set_tesselation_evaluation_source(const fs::path& path);
  void set_geometry_source(const fs::path& path);
  void set_fragment_source(const fs::path& path);
  void set_compute_source(const fs::path& path);

  // compile and link shader from previously given source files
  void compile();

  // compute shader dispatch (call with actual amount of threads, will
  // internally divide by workgroup size)
  void dispatch_compute(uint32_t w, uint32_t h = 1, uint32_t d = 1) const;

  // uniform upload handling
  void uniform(const std::string& name, int val) const;
  void uniform(const std::string& name, int* val, uint32_t count) const;
  void uniform(const std::string& name, float val) const;
  void uniform(const std::string& name, float* val, uint32_t count) const;
  void uniform(const std::string& name, const vec2& val) const;
  void uniform(const std::string& name, const vec3& val) const;
  void uniform(const std::string& name, const vec4& val) const;
  void uniform(const std::string& name, const ivec2& val) const;
  void uniform(const std::string& name, const ivec3& val) const;
  void uniform(const std::string& name, const ivec4& val) const;
  void uniform(const std::string& name, const uvec2& val) const;
  void uniform(const std::string& name, const uvec3& val) const;
  void uniform(const std::string& name, const uvec4& val) const;
  void uniform(const std::string& name, const mat3& val) const;
  void uniform(const std::string& name, const mat4& val) const;
  void uniform(const std::string& name, const Texture2D& tex,
               uint32_t unit) const;
  void uniform(const std::string& name, const Texture3D& tex,
               uint32_t unit) const;

  // management/reload
  void clear();
  void reload_if_modified();

  static void add_shader_path(fs::path path);

  // data
  const std::string name;
  GLuint id;
  std::map<GLenum, fs::path> source_files;
  std::map<GLenum, fs::file_time_type> timestamps;

  // somewhat like <basepath>/reclib
  // const fs::path BASE_PATH = fs::path(__FILE__).parent_path().parent_path();
  // default paths where to search for shader files
  static std::vector<fs::path> SHADER_SEARCH_PATHS;
};

using Shader = NamedHandle<ShaderImpl>;

void reload_modified_shaders();

}  // namespace opengl
}  // namespace reclib
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::ShaderImpl>;  // needed for Windows DLL export

#endif