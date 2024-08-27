#include <opencv2/core/hal/interface.h>
#include <reclib/assert.h>
#include <reclib/opengl/texture.h>
#include <reclib/opengl/context.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <FreeImage.h>

#include "stb/stb_image_write.h"

// ----------------------------------------------------
// helper funcs

inline uint32_t format_to_channels(GLint format) {
  return format == GL_RGBA ? 4 : format == GL_RGB ? 3 : format == GL_RG ? 2 : 1;
}
inline GLint channels_to_format(uint32_t channels) {
  return channels == 4
             ? GL_RGBA
             : channels == 3 ? GL_RGB : channels == 2 ? GL_RG : GL_RED;
}
inline GLint channels_to_float_format(uint32_t channels) {
  return channels == 4
             ? GL_RGBA32F
             : channels == 3 ? GL_RGB32F : channels == 2 ? GL_RG32F : GL_R32F;
}
inline GLint channels_to_ubyte_format(uint32_t channels) {
  return channels == 4
             ? GL_RGBA8
             : channels == 3 ? GL_RGB8 : channels == 2 ? GL_RG8 : GL_R8;
}

inline uint32_t internal_formal_to_byte_precision(GLint format) {
  if (format == GL_R16 || format == GL_R16_SNORM || format == GL_RG16 ||
      format == GL_RG16_SNORM || format == GL_RGB16_SNORM ||
      format == GL_RGBA16 || format == GL_R16F || format == GL_RG16F ||
      format == GL_RGB16F || format == GL_RGBA16F || format == GL_R16I ||
      format == GL_R16UI || format == GL_RG16I || format == GL_RG16UI ||
      format == GL_RGB16I || format == GL_RGB16UI || format == GL_RGBA16I ||
      format == GL_RGBA16UI) {
    return 2;
  }
  if (format == GL_R32F || format == GL_RG32F || format == GL_RGB32F ||
      format == GL_RGBA32F || format == GL_R32I || format == GL_R32UI ||
      format == GL_RG32I || format == GL_RG32UI || format == GL_RGB32I ||
      format == GL_RGB32UI || format == GL_RGBA32I || format == GL_RGBA32UI) {
    return 4;
  }
  // round to next higher precision
  if (format == GL_RGB10 || format == GL_RGB12 || format == GL_RGB10_A2 ||
      format == GL_RGB10_A2UI || format == GL_RGBA12 ||
      format == GL_R11F_G11F_B10F || format == GL_RGB9_E5) {
    return 2;
  }
  // default
  return 1;
}
template <typename T>
inline T unpack_bytes_per_channel(std::vector<uint8_t>& data, uint32_t index,
                                  uint32_t byte_precision) {
  if (byte_precision == 1) {
    return data[index];
  } else if (byte_precision == 2) {
    return (data[index + 1] << 8) | data[index + 0];
  } else if (byte_precision == 4) {
    // return (data[index + 3] << 24) | (data[index + 2] << 16) | (data[index +
    // 1] << 8) | data[index + 0];
    return data[index];
  }
  throw std::runtime_error("Unsupported number of channels");
}

template <typename T>
inline std::vector<T> unpack_bytes(std::vector<uint8_t>& data, uint32_t index,
                                   uint32_t byte_precision, uint32_t channels) {
  std::vector<T> type_per_channel;
  if (channels == 1) {
    type_per_channel.push_back(
        unpack_bytes_per_channel<T>(data, index, byte_precision));
  } else if (channels == 2) {
    type_per_channel.push_back(
        unpack_bytes_per_channel<T>(data, index, byte_precision));
    type_per_channel.push_back(unpack_bytes_per_channel<T>(
        data, index + byte_precision, byte_precision));
  } else if (channels == 3) {
    type_per_channel.push_back(
        unpack_bytes_per_channel<T>(data, index, byte_precision));
    type_per_channel.push_back(unpack_bytes_per_channel<T>(
        data, index + byte_precision, byte_precision));
    type_per_channel.push_back(unpack_bytes_per_channel<T>(
        data, index + byte_precision * 2, byte_precision));
  } else {
    type_per_channel.push_back(
        unpack_bytes_per_channel<T>(data, index, byte_precision));
    type_per_channel.push_back(unpack_bytes_per_channel<T>(
        data, index + byte_precision, byte_precision));
    type_per_channel.push_back(unpack_bytes_per_channel<T>(
        data, index + byte_precision * 2, byte_precision));
    type_per_channel.push_back(unpack_bytes_per_channel<T>(
        data, index + byte_precision * 3, byte_precision));
  }
  return type_per_channel;
}

inline void freeimage_store(std::vector<uint8_t>& data, std::string path,
                            uint32_t width, uint32_t height, uint32_t channels,
                            uint32_t byte_precision) {
  FreeImage_Initialise();

  FREE_IMAGE_TYPE type = FIT_BITMAP;
  if (channels == 1 && byte_precision == 2)
    type = FIT_UINT16;
  else if (channels == 1 && byte_precision == 4)
    type = FIT_UINT32;
  else if (channels == 3 && byte_precision == 2)
    type = FIT_RGB16;
  else if (channels == 4 && byte_precision == 2)
    type = FIT_RGBA16;
  else if (channels == 3 && byte_precision == 4)
    type = FIT_RGB16;
  else if (channels == 4 && byte_precision == 4)
    type = FIT_RGBA16;
  FIBITMAP* dib = FreeImage_AllocateT(type, width, height, channels * 8);
  uint8_t* bits = FreeImage_GetBits(dib);
  uint32_t pitch = FreeImage_GetPitch(dib);

  if (channels == 1 && byte_precision == 1) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      uint8_t* bitsRowStartType = (uint8_t*)bitsRowStart;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * channels * byte_precision;
        std::vector<uint8_t> v =
            unpack_bytes<uint8_t>(data, base_index, byte_precision, channels);
        bitsRowStartType[x] = v[0];
      }
    }
  } else if (channels == 1 && byte_precision == 2) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      uint16_t* bitsRowStartType = (uint16_t*)bitsRowStart;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * channels * byte_precision;
        std::vector<uint16_t> v =
            unpack_bytes<uint16_t>(data, base_index, byte_precision, channels);
        bitsRowStartType[x] = v[0];
      }
    }
  } else if (channels == 1 && byte_precision == 4) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      uint32_t* bitsRowStartType = (uint32_t*)bitsRowStart;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * channels * byte_precision;
        std::vector<uint32_t> v =
            unpack_bytes<uint32_t>(data, base_index, byte_precision, channels);
        bitsRowStartType[x] = v[0];
      }
    }
  } else if ((channels == 2 && byte_precision == 1)) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * (channels + 1) * byte_precision;
        std::vector<uint8_t> v =
            unpack_bytes<uint8_t>(data, base_index, byte_precision, channels);
        bitsRowStart[x * channels + FI_RGBA_RED] = v[0];
        bitsRowStart[x * channels + FI_RGBA_GREEN] = v[1];
        bitsRowStart[x * channels + FI_RGBA_BLUE] = 0;
      }
    }
  } else if ((channels == 2 && byte_precision == 2)) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      uint16_t* bitsRowStartType = (uint16_t*)bitsRowStart;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * (channels + 1) * byte_precision;
        std::vector<uint16_t> v =
            unpack_bytes<uint16_t>(data, base_index, byte_precision, channels);
        bitsRowStartType[x * channels + FI_RGBA_RED] = v[0];
        bitsRowStartType[x * channels + FI_RGBA_GREEN] = v[1];
        bitsRowStartType[x * channels + FI_RGBA_BLUE] = 0;
      }
    }
  } else if ((channels == 3 && byte_precision == 1)) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * channels * byte_precision;
        std::vector<uint8_t> v =
            unpack_bytes<uint8_t>(data, base_index, byte_precision, channels);
        bitsRowStart[x * channels + FI_RGBA_RED] = v[0];
        bitsRowStart[x * channels + FI_RGBA_GREEN] = v[1];
        bitsRowStart[x * channels + FI_RGBA_BLUE] = v[2];
      }
    }
  } else if ((channels == 3 && byte_precision == 2)) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      uint16_t* bitsRowStartType = (uint16_t*)bitsRowStart;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * channels * byte_precision;
        std::vector<uint16_t> v =
            unpack_bytes<uint16_t>(data, base_index, byte_precision, channels);
        // For some reason, FreeImage expects the color in BGR order so we swap
        // 2nd and 0th index
        bitsRowStartType[x * channels + FI_RGBA_RED] = v[2];
        bitsRowStartType[x * channels + FI_RGBA_GREEN] = v[1];
        bitsRowStartType[x * channels + FI_RGBA_BLUE] = v[0];
      }
    }
  } else if ((channels == 4 && byte_precision == 1)) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * channels * byte_precision;
        std::vector<uint8_t> v =
            unpack_bytes<uint8_t>(data, base_index, byte_precision, channels);
        // For some reason, FreeImage expects the color in BGR order so we swap
        // 2nd and 0th index
        bitsRowStart[x * channels + FI_RGBA_RED] = v[2];
        bitsRowStart[x * channels + FI_RGBA_GREEN] = v[1];
        bitsRowStart[x * channels + FI_RGBA_BLUE] = v[0];
        bitsRowStart[x * channels + FI_RGBA_ALPHA] = v[3];
      }
    }
  } else if ((channels == 4 && byte_precision == 2)) {
    for (uint32_t y = 0; y < height; y++) {
      uint8_t* bitsRowStart = bits + (height - 1 - y) * pitch;
      uint16_t* bitsRowStartType = (uint16_t*)bitsRowStart;
      for (uint32_t x = 0; x < width; x++) {
        uint32_t base_index = (y * width + x) * channels * byte_precision;
        std::vector<uint16_t> v =
            unpack_bytes<uint16_t>(data, base_index, byte_precision, channels);
        // For some reason, FreeImage expects the color in BGR order so we swap
        // 2nd and 0th index
        bitsRowStartType[x * channels + FI_RGBA_RED] = v[2];
        bitsRowStartType[x * channels + FI_RGBA_GREEN] = v[1];
        bitsRowStartType[x * channels + FI_RGBA_BLUE] = v[0];
        bitsRowStartType[x * channels + FI_RGBA_ALPHA] = v[3];
      }
    }
  }

  if (channels == 1 && byte_precision == 4 && path.length() > 4 &&
      path.find(".tiff") != std::string::npos) {
    FreeImage_Save(FIF_TIFF, dib, path.c_str(), TIFF_DEFAULT);
  } else if (!(channels == 1 && byte_precision == 4) && path.length() > 4 &&
             path.find(".jpg") != std::string::npos) {
    FreeImage_Save(FIF_JPEG, dib, path.c_str(), JPEG_QUALITYSUPERB);
  } else if (!(channels == 1 && byte_precision == 4) && path.length() > 4 &&
             path.find(".png") != std::string::npos) {
    FreeImage_Save(FIF_PNG, dib, path.c_str());
  } else {
    throw std::runtime_error(
        "Unknown image configuration for given path: Wrong format?");
  }
  FreeImage_Unload(dib);

  FreeImage_DeInitialise();
}

// ----------------------------------------------------
// Texture2D

// OpenGL reads the given rectangle from bottom-left to top-right,
// instead, most image array are read from first entry (e.g. top left) to
// last entry (e.g. bottom right) so we must swap along the horizontal axis
void reclib::opengl::Texture2DImpl::flip_horizontally(uint8_t* pixels,
                                                      unsigned int width,
                                                      unsigned int height,
                                                      unsigned int channels,
                                                      unsigned int precision) {
  unsigned int bytesPerPixel = channels * precision;
  for (unsigned int y = 0; y < height / 2; y++) {
    const int swapY = height - y - 1;
    for (unsigned int x = 0; x < width; x++) {
      const int offset = int(bytesPerPixel) * (x + y * width);
      const int swapOffset = int(bytesPerPixel) * (x + swapY * width);
      // Swap R, G and B of the 2 pixels
      for (unsigned int c = 0; c < channels * precision; c++) {
        std::swap(pixels[offset + c], pixels[swapOffset + c]);
      }
    }
  }
}

reclib::opengl::Texture2DImpl::Texture2DImpl(const std::string& name,
                                             const fs::path& path, bool mipmap)
    : name(name), loaded_from_path(path), id(0) {
  _RECLIB_ASSERT(fs::exists(path));
  _RECLIB_ASSERT(reclib::opengl::Context::initialized);
  // load image from disk
  stbi_set_flip_vertically_on_load(1);
  int channels;
  uint8_t* data = 0;
  if (stbi_is_hdr(path.string().c_str())) {
    data = (uint8_t*)stbi_loadf(path.string().c_str(), &w, &h, &channels, 0);
    internal_format = channels_to_float_format(channels);
    format = channels_to_format(channels);
    type = GL_FLOAT;
  } else {
    data = stbi_load(path.string().c_str(), &w, &h, &channels, 0);
    internal_format = channels_to_ubyte_format(channels);
    format = channels_to_format(channels);
    type = GL_UNSIGNED_BYTE;
  }
  if (!data) {
    throw std::runtime_error("Failed to load image file: " + path.string());
    return;
  }

  // init GL texture

  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);

  // opengl by default needs 4 byte alignment after every row
  // stbi loaded data is not aligned that way -> pixelStore attributes need to
  // be set
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type,
               &data[0]);
  if (mipmap) glGenerateMipmap(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);

  // free data
  stbi_image_free(data);
}

reclib::opengl::Texture2DImpl::Texture2DImpl(const std::string& name,
                                             uint32_t w, uint32_t h,
                                             GLint internal_format,
                                             GLenum format, GLenum type,
                                             const void* data, bool mipmap)
    : name(name),
      id(0),
      w(w),
      h(h),
      internal_format(internal_format),
      format(format),
      type(type) {
  // init GL texture
  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                  (format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL)
                      ? GL_NEAREST
                      : GL_LINEAR);
  glTexParameteri(
      GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
      mipmap ? GL_LINEAR_MIPMAP_LINEAR
             : (format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL)
                   ? GL_NEAREST
                   : GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type, data);
  if (mipmap && data != 0) glGenerateMipmap(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);
}

reclib::opengl::Texture2DImpl::~Texture2DImpl() {
  if (glIsTexture(id)) glDeleteTextures(1, &id);
}

void reclib::opengl::Texture2DImpl::resize(uint32_t w, uint32_t h) {
  this->w = w;
  this->h = h;
  glBindTexture(GL_TEXTURE_2D, id);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void reclib::opengl::Texture2DImpl::bind(uint32_t unit) const {
  glActiveTexture(GL_TEXTURE0 + unit);
  glBindTexture(GL_TEXTURE_2D, id);
}

void reclib::opengl::Texture2DImpl::unbind() const {
  glBindTexture(GL_TEXTURE_2D, 0);
}

void reclib::opengl::Texture2DImpl::bind_image(uint32_t unit, GLenum access,
                                               GLenum format) const {
  glBindImageTexture(unit, id, 0, GL_FALSE, 0, access, format);
}

void reclib::opengl::Texture2DImpl::unbind_image(uint32_t unit) const {
  glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
}

void reclib::opengl::Texture2DImpl::fill(const float* color) const {
  glClearTexImage(id, 0, format, type, color);
}

void reclib::opengl::Texture2DImpl::clear() const {
  glClearTexImage(id, 0, format, type, 0);
}

void reclib::opengl::Texture2DImpl::load(const void* data, bool flip) {
  GLint red_prec = 0;
  GLint green_prec = 0;
  GLint blue_prec = 0;
  GLint alpha_prec = 0;
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
  unsigned int channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
                          uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);
  _RECLIB_ASSERT_GT(channels, 0);
  unsigned int precision =
      (red_prec + green_prec + blue_prec + alpha_prec) / (channels * 8);
  _RECLIB_ASSERT_GT(precision, 0);

  if (flip) {
    Texture2DImpl::flip_horizontally((uint8_t*)data, w, h, channels, precision);
  }
  glBindTexture(GL_TEXTURE_2D, id);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type, data);
  glBindTexture(GL_TEXTURE_2D, 0);
}

#if HAS_OPENCV_MODULE
void reclib::opengl::Texture2DImpl::load(CpuMat mat, bool flip) {
  if (mat.elemSize() == 0) {
    std::cout << "[WARNING] Texture2DImpl::load mat.elemSize() is empty"
              << std::endl;
    return;
  }

  GLint red_prec = 0;
  GLint green_prec = 0;
  GLint blue_prec = 0;
  GLint alpha_prec = 0;
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
  unsigned int channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
                          uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);
  _RECLIB_ASSERT_GT(channels, 0);
  unsigned int precision =
      (red_prec + green_prec + blue_prec + alpha_prec) / (channels * 8);
  _RECLIB_ASSERT_GT(precision, 0);

  void* mat_data = mat.ptr();
  _RECLIB_ASSERT_EQ(mat.channels(), channels);
  _RECLIB_ASSERT_EQ(mat.elemSize(), channels * precision);
  if (flip) {
    Texture2DImpl::flip_horizontally((uint8_t*)mat_data, w, h, channels,
                                     precision);
  }
  glBindTexture(GL_TEXTURE_2D, id);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type,
               mat_data);
  glBindTexture(GL_TEXTURE_2D, 0);
}
#endif  // HAS_OPENCV_MODULE

template <typename T>  // = uint8_t
std::vector<T> reclib::opengl::Texture2DImpl::data(
    bool flip, unsigned int buffer_channels) {
  unsigned int channels = 0;
  unsigned int precision = 0;
  if (format == GL_DEPTH_COMPONENT) {
    GLint depth_prec = 0;
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_DEPTH_SIZE, &depth_prec);
    channels = uint32_t(depth_prec > 0);
    precision = (depth_prec) / (channels * 8);
  } else {
    GLint red_prec = 0;
    GLint green_prec = 0;
    GLint blue_prec = 0;
    GLint alpha_prec = 0;
    GLint depth_prec = 0;
    GLint buffer_size = 0;
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_DEPTH_SIZE, &depth_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BUFFER_SIZE, &buffer_size);
    channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
               uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0) +
               uint32_t(depth_prec > 0);
    precision = (red_prec + green_prec + blue_prec + alpha_prec + depth_prec) /
                (channels * 8);
  }

  if (format == GL_DEPTH_COMPONENT) {
    buffer_channels = 4;
  }

  _RECLIB_ASSERT_GE(sizeof(T), precision);
  // std::cout << "channels: " << channels << std::endl;
  // std::cout << "precision: " << precision << std::endl;

  if (sizeof(T) > precision) {
    throw std::runtime_error("Not completely implemented yet. Needs testing.");
    std::vector<uint8_t> pixels(w * h * buffer_channels * precision);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTextureImage(id, 0, format, type, pixels.size(), &pixels[0]);
    std::vector<T> pixels_out(w * h * channels);
    for (int x = 0; x < w * h; x++) {
      unsigned int index = x * buffer_channels * precision;
      // std::cout << "index: " << x << ": " << int32_t(pixels[index + 0]) <<
      // ","
      //           << int32_t(pixels[index + 1]) << ","
      //           << int32_t(pixels[index + 2]) << ","
      //           << int32_t(pixels[index + 3]) << std::endl;
      for (int c = 0; c < channels; c++) {
        T val;
        if (sizeof(T) == 4 && precision == 3) {
          uint32_t tmp = (*(reinterpret_cast<uint32_t*>(pixels.data() + index +
                                                        c * precision)) &
                          0xffffff);
          std::memcpy(&val, &tmp, sizeof(tmp));
          // val = reinterpret_cast<T&>(tmp);
          std::cout << "tmp: " << tmp << " val: " << val << std::endl;
        } else {
          throw std::runtime_error(
              "Invalid combination of size and precision.");
        }
        pixels_out[x * channels + c] = val;
      }
    }
    if (flip)
      Texture2DImpl::flip_horizontally((uint8_t*)pixels_out.data(), w, h,
                                       channels, sizeof(T));
    return pixels_out;

  } else {
    std::vector<T> pixels(w * h * buffer_channels);

    // glReadPixels can align the first pixel in each row at 1-, 2-, 4- and
    // 8-byte boundaries. We have allocated the exact size needed for the image
    // so we have to use 1-byte alignment (otherwise glReadPixels would write
    // out of bounds)
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTextureImage(id, 0, format, type, pixels.size() * sizeof(T),
                      &pixels[0]);

    if (channels < buffer_channels) {
      std::vector<T> pixels_out;
      for (int x = 0; x < w * h; x++) {
        unsigned int index = x * buffer_channels;
        // std::cout << "index: " << x << ": " << int32_t(pixels[index + 0]) <<
        // ","
        //           << int32_t(pixels[index + 1]) << ","
        //           << int32_t(pixels[index + 2]) << ","
        //           << int32_t(pixels[index + 3]) << std::endl;
        for (int c = 0; c < channels; c++) {
          pixels_out.push_back(pixels[index + c]);
        }
      }
      pixels = pixels_out;
    }

    if (flip)
      Texture2DImpl::flip_horizontally((uint8_t*)pixels.data(), w, h, channels,
                                       precision);
    return pixels;
  }
}

// template instantiation
template std::vector<uint8_t> reclib::opengl::Texture2DImpl::data(
    bool flip, unsigned int buffer_channels);
template std::vector<uint32_t> reclib::opengl::Texture2DImpl::data(
    bool flip, unsigned int buffer_channels);
template std::vector<float> reclib::opengl::Texture2DImpl::data(
    bool flip, unsigned int buffer_channels);

void reclib::opengl::Texture2DImpl::data(void* container,
                                         uint32_t size_in_bytes, GLenum type,
                                         bool flip) {
  unsigned int channels = 0;
  unsigned int precision = 0;
  if (format == GL_DEPTH_COMPONENT) {
    GLint depth_prec = 0;
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_DEPTH_TYPE, &depth_prec);
    channels = uint32_t(depth_prec > 0);
    precision = (depth_prec) / (channels * 8);
  } else {
    GLint red_prec = 0;
    GLint green_prec = 0;
    GLint blue_prec = 0;
    GLint alpha_prec = 0;
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
    glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
    channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
               uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);
    precision =
        (red_prec + green_prec + blue_prec + alpha_prec) / (channels * 8);
  }

  _RECLIB_ASSERT_GE(size_in_bytes, w * h * channels * precision);

  // glReadPixels can align the first pixel in each row at 1-, 2-, 4- and 8-byte
  // boundaries. We have allocated the exact size needed for the image so we
  // have to use 1-byte alignment (otherwise glReadPixels would write out of
  // bounds)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTextureImage(id, 0, format, type, size_in_bytes, container);

  if (flip)
    Texture2DImpl::flip_horizontally((uint8_t*)container, w, h, channels,
                                     precision);
}

void reclib::opengl::save_buffer_as_png(const fs::path& path,
                                        std::vector<uint8_t>& data, uint32_t w,
                                        uint32_t h, uint32_t channels,
                                        uint32_t precision, bool flip) {
  if (flip)
    Texture2DImpl::flip_horizontally((uint8_t*)data.data(), w, h, channels,
                                     precision);

  freeimage_store(data, path.string(), w, h, channels, precision);
}

void reclib::opengl::Texture2DImpl::save_png(const fs::path& path, bool flip,
                                             int byte_precision,
                                             int save_channels) const {
  GLenum type = GL_UNSIGNED_BYTE;

  GLint red_prec = 0;
  GLint green_prec = 0;
  GLint blue_prec = 0;
  GLint alpha_prec = 0;
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
  int channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
                 uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);
  int precision =
      (red_prec + green_prec + blue_prec + alpha_prec) / (channels * 8);

  if (byte_precision > 0) {
    precision = (unsigned int)(byte_precision);
  }

  if (precision > 1 && channels == 1) {
    // freeimage can store a maximum of 2 bytes per channel
    type = GL_UNSIGNED_SHORT;
    precision = 2;
  } else if (precision > 1 && channels > 1) {
    precision = 1;
  }

  std::vector<uint8_t> pixels(w * h * channels * precision);

  // glReadPixels can align the first pixel in each row at 1-, 2-, 4- and 8-byte
  // boundaries. We have allocated the exact size needed for the image so we
  // have to use 1-byte alignment (otherwise glReadPixels would write out of
  // bounds)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glBindTexture(GL_TEXTURE_2D, id);
  glGetTexImage(GL_TEXTURE_2D, 0, format, type, &pixels[0]);
  glBindTexture(GL_TEXTURE_2D, 0);

  if ((save_channels > 0 && channels > save_channels)) {
    std::vector<uint8_t> pixels_reduced;
    for (int x = 0; x < w * h; x++) {
      unsigned int index = x * channels * precision;

      for (int c = 0; c < save_channels * precision; c++) {
        pixels_reduced.push_back(pixels[index + c]);
      }
    }
    pixels = pixels_reduced;
    channels = save_channels;
  }

  if (flip)
    Texture2DImpl::flip_horizontally((uint8_t*)pixels.data(), w, h, channels,
                                     precision);
  freeimage_store(pixels, path.string(), w, h, channels, precision);

  // std::cout << path << " written." << std::endl;
}

void reclib::opengl::Texture2DImpl::save_tiff(const fs::path& path, bool flip,
                                              int byte_precision) const {
  GLenum type = GL_UNSIGNED_BYTE;

  GLint red_prec = 0;
  GLint green_prec = 0;
  GLint blue_prec = 0;
  GLint alpha_prec = 0;
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
  unsigned int channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
                          uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);
  unsigned int precision =
      (red_prec + green_prec + blue_prec + alpha_prec) / (channels * 8);

  if (byte_precision > 0) {
    precision = (unsigned int)(byte_precision);
  }

  if (precision > 1) {
    if (channels == 1 && precision == 4) {
      type = GL_UNSIGNED_INT;
    } else {
      // freeimage can store a maximum of 2 bytes per channel
      // in case we have more than 1 channel
      type = GL_UNSIGNED_SHORT;
      precision = 2;
    }
  }

  std::vector<uint8_t> pixels(w * h * channels * precision);
  // glReadPixels can align the first pixel in each row at 1-, 2-, 4- and 8-byte
  // boundaries. We have allocated the exact size needed for the image so we
  // have to use 1-byte alignment (otherwise glReadPixels would write out of
  // bounds)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glBindTexture(GL_TEXTURE_2D, id);
  glGetTexImage(GL_TEXTURE_2D, 0, format, type, &pixels[0]);
  glBindTexture(GL_TEXTURE_2D, 0);

  if (flip)
    Texture2DImpl::flip_horizontally((uint8_t*)pixels.data(), w, h, channels,
                                     precision);
  freeimage_store(pixels, path.string(), w, h, channels, precision);

  // std::cout << path << " written." << std::endl;
}

void reclib::opengl::Texture2DImpl::save_jpg(const fs::path& path, int quality,
                                             bool flip) const {
  GLint red_prec = 0;
  GLint green_prec = 0;
  GLint blue_prec = 0;
  GLint alpha_prec = 0;
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
  unsigned int channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
                          uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);
  unsigned int precision =
      (red_prec + green_prec + blue_prec + alpha_prec) / (channels * 8);

  stbi_flip_vertically_on_write(flip);
  std::vector<uint8_t> pixels(w * h * channels);
  // glReadPixels can align the first pixel in each row at 1-, 2-, 4- and 8-byte
  // boundaries. We have allocated the exact size needed for the image so we
  // have to use 1-byte alignment (otherwise glReadPixels would write out of
  // bounds)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glBindTexture(GL_TEXTURE_2D, id);
  glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_BYTE, &pixels[0]);
  glBindTexture(GL_TEXTURE_2D, 0);
  Texture2DImpl::flip_horizontally((uint8_t*)pixels.data(), w, h, channels, 1);
  stbi_write_jpg(path.string().c_str(), w, h, channels, pixels.data(), quality);
  // std::cout << path << " written." << std::endl;
}

uint32_t reclib::opengl::Texture2DImpl::get_channels() const {
  GLint red_prec = 0;
  GLint green_prec = 0;
  GLint blue_prec = 0;
  GLint alpha_prec = 0;
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
  unsigned int channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
                          uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);

  return channels;
}

uint32_t reclib::opengl::Texture2DImpl::get_precision() const {
  GLint red_prec = 0;
  GLint green_prec = 0;
  GLint blue_prec = 0;
  GLint alpha_prec = 0;
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_RED_SIZE, &red_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_GREEN_SIZE, &green_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_BLUE_SIZE, &blue_prec);
  glGetTextureLevelParameteriv(id, 0, GL_TEXTURE_ALPHA_SIZE, &alpha_prec);
  unsigned int channels = uint32_t(red_prec > 0) + uint32_t(green_prec > 0) +
                          uint32_t(blue_prec > 0) + uint32_t(alpha_prec > 0);
  unsigned int precision =
      (red_prec + green_prec + blue_prec + alpha_prec) / (channels * 8);

  return precision;
}

#if HAS_OPENCV_MODULE
reclib::opengl::Texture2D reclib::opengl::tex_from_mat(const std::string& name,
                                                       CpuMat mat, bool flip_y,
                                                       bool normalize) {
  if (mat.rows == 0 && mat.cols == 0) {
    std::cout << "[WARNING] Mat " << name
              << " is empty. Aborting 'tex_from_mat'." << std::endl;
    return reclib::opengl::Texture2D();
  }

  CpuMat tmp;
  if (mat.depth() == CV_32F) {
    tmp = mat.clone();

    if (normalize) {
      double min, max;
      cv::minMaxIdx(mat, &min, &max);
      if (max > 1) {
        tmp = tmp / max;
      }
    }
  } else if (mat.depth() == CV_8U) {
    double min, max;
    cv::minMaxIdx(mat, &min, &max);
    mat.convertTo(tmp, CV_32FC(mat.channels()));

    if (normalize) {
      if (max > 1) {
        tmp = tmp / 255.f;
      }
    }
  } else {
    throw std::runtime_error("Matrix type not implemented for 'tex_from_mat'");
  }

  if (!reclib::opengl::Texture2D::valid(name)) {
    switch (mat.channels()) {
      case 1: {
        reclib::opengl::Texture2D(name, mat.cols, mat.rows, GL_R32F, GL_RED,
                                  GL_FLOAT);
        break;
      }
      case 3: {
        reclib::opengl::Texture2D(name, mat.cols, mat.rows, GL_RGB32F, GL_RGB,
                                  GL_FLOAT);
        break;
      }
      case 4: {
        reclib::opengl::Texture2D(name, mat.cols, mat.rows, GL_RGBA32F, GL_RGBA,
                                  GL_FLOAT);
        break;
      }
      default: {
        throw std::runtime_error(
            "Invalid number of channels in 'tex_from_mat': " +
            std::to_string(mat.channels()));
      }
    }
  } else {
    reclib::opengl::Texture2D::find(name)->resize(mat.cols, mat.rows);
  }
  reclib::opengl::Texture2D tex = reclib::opengl::Texture2D::find(name);
  tex->load(tmp, flip_y);

  return tex;
}
#endif  // HAS_OPENCV_MODULE

// ----------------------------------------------------
// Texture3D

reclib::opengl::Texture3DImpl::Texture3DImpl(const std::string& name,
                                             uint32_t w, uint32_t h, uint32_t d,
                                             GLint internal_format,
                                             GLenum format, GLenum type,
                                             const void* data, bool mipmap)
    : name(name),
      id(0),
      w(w),
      h(h),
      d(d),
      internal_format(internal_format),
      format(format),
      type(type) {
  // init GL texture
  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_3D, id);
  // default border color is (0, 0, 0, 0)
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER,
                  mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
  glTexImage3D(GL_TEXTURE_3D, 0, internal_format, w, h, d, 0, format, type,
               data);
  if (mipmap && data != 0) glGenerateMipmap(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);
}

reclib::opengl::Texture3DImpl::~Texture3DImpl() {
  if (glIsTexture(id)) glDeleteTextures(1, &id);
}

void reclib::opengl::Texture3DImpl::resize(uint32_t w, uint32_t h, uint32_t d) {
  this->w = w;
  this->h = h;
  this->d = d;
  glBindTexture(GL_TEXTURE_3D, id);
  glTexImage2D(GL_TEXTURE_3D, 0, internal_format, w, h, 0, format, type, 0);
  glBindTexture(GL_TEXTURE_3D, 0);
}

void reclib::opengl::Texture3DImpl::bind(uint32_t unit) const {
  glActiveTexture(GL_TEXTURE0 + unit);
  glBindTexture(GL_TEXTURE_3D, id);
}

void reclib::opengl::Texture3DImpl::unbind() const {
  glBindTexture(GL_TEXTURE_3D, 0);
}

void reclib::opengl::Texture3DImpl::bind_image(uint32_t unit, GLenum access,
                                               GLenum format) const {
  glBindImageTexture(unit, id, 0, GL_FALSE, 0, access, format);
}

void reclib::opengl::Texture3DImpl::unbind_image(uint32_t unit) const {
  glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
}
