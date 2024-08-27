#ifndef RECLIB_OPENGL_TEXTURE_H
#define RECLIB_OPENGL_TEXTURE_H

#include <memory>

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on

#include <vector>

#include "named_handle.h"
#include "reclib/data_types.h"
#include "reclib/internal/filesystem.h"

namespace fs = std::filesystem;

// ----------------------------------------------------
// Texture2D

namespace reclib {
namespace opengl {

void save_buffer_as_png(const fs::path& path, std::vector<uint8_t>& data,
                        uint32_t w, uint32_t h, uint32_t channels,
                        uint32_t precision, bool flip = false);

class _API Texture2DImpl {
 public:
  // construct from image on disk
  Texture2DImpl(const std::string& name, const fs::path& path,
                bool mipmap = true);
  // construct empty texture or from raw data
  Texture2DImpl(const std::string& name, uint32_t w, uint32_t h,
                GLint internal_format, GLenum format, GLenum type,
                const void* data = 0, bool mipmap = false);
  virtual ~Texture2DImpl();

  // prevent copies and moves, since GL buffers aren't reference counted
  Texture2DImpl(const Texture2DImpl&) = delete;
  Texture2DImpl& operator=(const Texture2DImpl&) = delete;
  Texture2DImpl& operator=(const Texture2DImpl&&) = delete;

  static inline std::string type_to_str() { return "Texture2DImpl"; }

  explicit inline operator bool() const {
    return w > 0 && h > 0 && glIsTexture(id);
  }
  inline operator GLuint() const { return id; }

  // resize (discards all data!)
  void resize(uint32_t w, uint32_t h);

  // bind/unbind to/from OpenGL
  void bind(uint32_t uint) const;
  void unbind() const;
  void bind_image(uint32_t unit, GLenum access, GLenum format) const;
  void unbind_image(uint32_t unit) const;
  void clear() const;
  void fill(const float* color) const;
  void load(const void* data, bool flip = true);
#if HAS_OPENCV_MODULE
  void load(CpuMat data, bool flip = true);
#endif

  // TODO CPU <-> GPU data transfers

  // save to disk
  void save_png(const fs::path& path, bool flip = true, int byte_precision = -1,
                int save_channels = -1) const;
  void save_tiff(const fs::path& path, bool flip = true,
                 int byte_precision = -1) const;
  void save_jpg(const fs::path& path, int quality = 100,
                bool flip = true) const;  // quality: [1, 100]

  template <typename T = uint8_t>
  std::vector<T> data(bool flip = true, unsigned int buffer_channels = 4);
  void data(void* container, uint32_t size_in_bytes, GLenum type,
            bool flip = true);

  uint32_t get_channels() const;
  uint32_t get_precision() const;

  static void flip_horizontally(uint8_t* pixels, unsigned int width,
                                unsigned int height, unsigned int channels,
                                unsigned int precision);

  // data
  const std::string name;
  const fs::path loaded_from_path;
  GLuint id;
  int w, h;
  GLint internal_format;
  GLenum format, type;
};

using Texture2D = NamedHandle<Texture2DImpl>;

#if HAS_OPENCV_MODULE
Texture2D tex_from_mat(const std::string& name, CpuMat mat, bool flip_y,
                       bool normalize = true);
#endif

// ----------------------------------------------------
// Texture3D

class _API Texture3DImpl {
 public:
  // construct empty texture or from raw data
  Texture3DImpl(const std::string& name, uint32_t w, uint32_t h, uint32_t d,
                GLint internal_format, GLenum format, GLenum type,
                const void* data = 0, bool mipmap = false);
  virtual ~Texture3DImpl();

  // prevent copies and moves, since GL buffers aren't reference counted
  Texture3DImpl(const Texture3DImpl&) = delete;
  Texture3DImpl& operator=(const Texture3DImpl&) = delete;
  Texture3DImpl& operator=(const Texture3DImpl&&) = delete;

  static inline std::string type_to_str() { return "Texture3DImpl"; }

  explicit inline operator bool() const {
    return w > 0 && h > 0 && d > 0 && glIsTexture(id);
  }
  inline operator GLuint() const { return id; }

  // resize (discards all data!)
  void resize(uint32_t w, uint32_t h, uint32_t d);

  // bind/unbind to/from OpenGL
  void bind(uint32_t uint) const;
  void unbind() const;
  void bind_image(uint32_t unit, GLenum access, GLenum format) const;
  void unbind_image(uint32_t unit) const;

  // TODO CPU <-> GPU data transfers

  // data
  const std::string name;
  GLuint id;
  int w, h, d;
  GLint internal_format;
  GLenum format, type;
};

using Texture3D = NamedHandle<Texture3DImpl>;

}  // namespace opengl
}  // namespace reclib

template class _API reclib::opengl::NamedHandle<
    reclib::opengl::Texture2DImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::Texture3DImpl>;  // needed for Windows DLL export

#endif