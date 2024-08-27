#ifndef RECLIB_OPENGL_MATERIAL_H
#define RECLIB_OPENGL_MATERIAL_H

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on
#include <assimp/material.h>

#include <map>
#include <memory>
#include <string>

#include "reclib/internal/filesystem.h"
#include "reclib/opengl/named_handle.h"
#include "reclib/opengl/shader.h"
#include "reclib/opengl/texture.h"

namespace fs = std::filesystem;

// ------------------------------------------
// Material
namespace reclib {
namespace opengl {

class _API MaterialImpl {
 public:
  MaterialImpl(const std::string& name);
  MaterialImpl(const std::string& name, const fs::path& base_path,
               const aiMaterial* mat_ai);
  virtual ~MaterialImpl();

  static inline std::string type_to_str() { return "MaterialImpl"; }

  void bind(const Shader& shader) const;
  void unbind() const;

  inline bool has_texture(const std::string& uniform_name) const {
    return texture_map.count(uniform_name);
  }
  inline Texture2D get_texture(const std::string& uniform_name) const {
    return texture_map.at(uniform_name);
  }
  inline void add_texture(const std::string& uniform_name,
                          const Texture2D& texture) {
    texture_map[uniform_name] = texture;
  }

  // data
  const std::string name;
  std::map<std::string, int> int_map;
  std::map<std::string, float> float_map;
  std::map<std::string, vec2> vec2_map;
  std::map<std::string, vec3> vec3_map;
  std::map<std::string, vec4> vec4_map;
  std::map<std::string, Texture2D> texture_map;
};

using Material = NamedHandle<MaterialImpl>;

}  // namespace opengl
}  // namespace reclib
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::MaterialImpl>;  // needed for Windows DLL export

#endif