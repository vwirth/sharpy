#ifndef RECLIB_OPENGL_GEOMETRY_H
#define RECLIB_OPENGL_GEOMETRY_H

#include <assimp/mesh.h>

#include <map>
#include <vector>

#include "reclib/assert.h"
#include "reclib/data_types.h"
#include "reclib/math/eigen_glm_interface.h"
#include "reclib/opengl/named_handle.h"
#include "reclib/platform.h"

// ------------------------------------------
// Geometry

namespace reclib {
namespace opengl {

class MeshImpl;

std::vector<vec3> face2vertex_normals(const std::vector<vec3>& face_normals,
                                      const uint32_t position_size,
                                      const std::vector<uint32_t>& indices);

std::vector<vec3> face2vertex_normals(const float* face_normals,
                                      const uint32_t normal_size,
                                      const uint32_t position_size,
                                      const uint32_t* indices,
                                      const uint32_t indices_size);

std::vector<vec3> face2vertex_normals(const std::vector<vec3>& face_normals,
                                      const uint32_t position_size,
                                      const uint32_t* indices,
                                      const uint32_t indices_size);

std::vector<vec3> generate_face_normals(const std::vector<vec3>& positions,
                                        const std::vector<uint32_t>& indices);

std::vector<vec3> generate_face_normals(const float* positions,
                                        const uint32_t* indices,
                                        const uint32_t indices_size);

std::vector<vec3> generate_vertex_normals(const std::vector<vec3>& positions,
                                          const std::vector<uint32_t>& indices);

std::vector<vec3> generate_vertex_normals(const float* positions,
                                          const uint32_t position_size,
                                          const uint32_t* indices,
                                          const uint32_t indices_size);

class GeometryImpl;
class GeometryWrapperImpl;

template <typename T>
struct AttributePtr {
  T* ptr;
  uint32_t size;
  uint32_t dim;
  bool external;
};

class _API GeometryBaseImpl {
 public:
  GeometryBaseImpl(const std::string& name);
  virtual ~GeometryBaseImpl();

  static inline std::string type_to_str() { return "GeometryBaseImpl"; }

  virtual float* positions_ptr();
  virtual float* normals_ptr();
  virtual float* texcoords_ptr();
  virtual uint32_t* indices_ptr();

  virtual size_t positions_size() const;
  virtual size_t normals_size() const;
  virtual size_t texcoords_size() const;
  virtual size_t indices_size() const;

  virtual vec3 get_position(unsigned int index) const;
  virtual vec3 get_normal(unsigned int index) const;
  virtual vec2 get_texcoord(unsigned int index) const;
  virtual uint32_t get_index(unsigned int index) const;

  virtual void set_position(unsigned int index, const vec3& position);
  virtual void set_normal(unsigned int index, const vec3& normal);
  virtual void set_texcoord(unsigned int index, const vec2& texcoord);
  virtual void set_index(unsigned int index, uint32_t index_value);

  virtual bool has_normals() const;
  virtual bool has_texcoords() const;

  // automatically generate normals from vertex positions
  virtual void auto_generate_normals(
      const mat4& initial_transform = mat4::Identity());
  // convert per-face normals to per-vertex normals
  virtual void convert_normals_perface2vertex();

  virtual void clear();

  // O(n) geometry operations
  virtual void recompute_aabb();
  virtual void fit_into_aabb(const vec3& aabb_min, const vec3& aabb_max);
  virtual void translate(const vec3& by);
  virtual void scale(const vec3& by);
  virtual void rotate(float angle_degrees, const vec3& axis);
  virtual void transform(const mat4& trans);

  void add_attribute_vec3(const std::string& name, vec3* attribute,
                          uint32_t size, bool external = false,
                          bool overwrite = false);
  void add_attribute_vec3(const std::string& name, std::vector<vec3>& attribute,
                          bool external = false, bool overwrite = false);

  void add_attribute_vec4(const std::string& name, vec4* attribute,
                          uint32_t size, bool external = false,
                          bool overwrite = false);
  void add_attribute_vec4(const std::string& name, std::vector<vec4>& attribute,
                          bool external = false, bool overwrite = false);

  void add_attribute_vec2(const std::string& name, vec2* attribute,
                          uint32_t size, bool external = false,
                          bool overwrite = false);
  void add_attribute_vec2(const std::string& name, std::vector<vec2>& attribute,
                          bool external = false, bool overwrite = false);

  void add_attribute_float(const std::string& name, float* attribute,
                           uint32_t size, uint32_t dim, bool external = false,
                           bool overwrite = false);
  void add_attribute_float(const std::string& name,
                           std::vector<float>& attribute, uint32_t dim,
                           bool external = false, bool overwrite = false);

  void add_attribute_uint(const std::string& name, uint32_t* attribute,
                          uint32_t size, uint32_t dim, bool external = false,
                          bool overwrite = false);
  void add_attribute_uint(const std::string& name,
                          std::vector<uint32_t>& attribute, uint32_t dim,
                          bool external = false, bool overwrite = false);

  void add_attribute_int(const std::string& name, int32_t* attribute,
                         uint32_t size, uint32_t dim, bool external = false,
                         bool overwrite = false);
  void add_attribute_int(const std::string& name,
                         std::vector<int32_t>& attribute, uint32_t dim,
                         bool external = false, bool overwrite = false);

  // data
  const std::string name;
  vec3 bb_min, bb_max;
  bool auto_normals_;

  std::vector<std::vector<vec4>> vec4_storage;
  std::vector<std::vector<vec3>> vec3_storage;
  std::vector<std::vector<vec2>> vec2_storage;
  std::vector<std::vector<float>> float_storage;
  std::vector<std::vector<uint32_t>> uint_storage;
  std::vector<std::vector<int32_t>> int_storage;

  std::map<std::string, AttributePtr<vec4>> vec4_map;
  std::map<std::string, AttributePtr<vec3>> vec3_map;
  std::map<std::string, AttributePtr<vec2>> vec2_map;
  std::map<std::string, AttributePtr<float>> float_map;
  std::map<std::string, AttributePtr<uint32_t>> uint_map;
  std::map<std::string, AttributePtr<int32_t>> int_map;

  // list of meshes that use this geometry
  // this is necessary to trigger updates to GPU memory in case
  // the underlying geometry changes
  std::map<std::string, NamedHandle<MeshImpl>> used_by;

  // register mesh
  // mesh is being updated if the underlying geometry changes
  // by calling 'update_meshes'
  void register_mesh(const NamedHandle<MeshImpl>& mesh);
  void unregister_mesh(const std::string& name);
  void clear_mesh_gpu_memory();
  void update_meshes();
};

class _API GeometryImpl : public GeometryBaseImpl {
 public:
  GeometryImpl(const std::string& name);
  GeometryImpl(const std::string& name, const aiMesh* mesh_ai);
  GeometryImpl(const std::string& name, const std::vector<vec3>& positions,
               const std::vector<uint32_t>& indices = std::vector<uint32_t>(),
               const std::vector<vec3>& normals = std::vector<vec3>(),
               const std::vector<vec2>& texcoords = std::vector<vec2>());

  GeometryImpl(const std::string& name, const std::vector<float>& positions,
               const std::vector<uint32_t>& indices = std::vector<uint32_t>(),
               const std::vector<float>& normals = std::vector<float>(),
               const std::vector<float>& texcoords = std::vector<float>());

  // pos_size is size of vertices, e.g. a tuple of (float,float,float), it is
  // NOT the size of floats! analogously for normals_size and texcoords_size
  GeometryImpl(const std::string& name, const float* positions, size_t pos_size,
               const uint32_t* indices = nullptr, size_t indices_size = 0,
               const float* normals = nullptr, size_t normals_size = 0,
               const float* texcoords = nullptr, size_t texcoords_size = 0);
  virtual ~GeometryImpl();

  explicit inline operator bool() const {
    return (positions.size() > 0) && (indices.size() > 0);
  }

  // automatically generate normals from vertex positions
  void auto_generate_normals(
      const mat4& initial_transform = mat4::Identity()) override;
  void convert_normals_perface2vertex() override;

  void add(const aiMesh* mesh_ai);
  void add(const GeometryImpl& other);
  void add(const std::vector<vec3>& positions,
           const std::vector<uint32_t>& indices,
           const std::vector<vec3>& normals = std::vector<vec3>(),
           const std::vector<vec2>& texcoords = std::vector<vec2>());
  void add(const std::vector<float>& positions,
           const std::vector<uint32_t>& indices,
           const std::vector<float>& normals = std::vector<float>(),
           const std::vector<float>& texcoords = std::vector<float>());

  // pos_size is size of vertices, e.g. a tuple of (float,float,float), it is
  // NOT the size of floats! analogously for normals_size and texcoords_size
  void add(const float* positions, size_t pos_size,
           const uint32_t* indices = nullptr, size_t indices_size = 0,
           const float* normals = nullptr, size_t normals_size = 0,
           const float* texcoords = nullptr, size_t texcoords_size = 0);

  // pos_size is size of vertices, e.g. a tuple of (float,float,float), it is
  // NOT the size of floats! analogously for normals_size and texcoords_size
  void set(const float* positions, size_t pos_size,
           const uint32_t* indices = nullptr, size_t indices_size = 0,
           const float* normals = nullptr, size_t normals_size = 0,
           const float* texcoords = nullptr, size_t texcoords_size = 0);
  void clear() override;

  bool has_normals() const override { return !normals.empty(); }
  bool has_texcoords() const override { return !texcoords.empty(); }

  // O(n) geometry operations
  void recompute_aabb() override;
  void fit_into_aabb(const vec3& aabb_min, const vec3& aabb_max) override;
  void translate(const vec3& by) override;
  void scale(const vec3& by) override;
  void rotate(float angle_degrees, const vec3& axis) override;
  void transform(const mat4& trans) override;

  vec3 get_position(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, positions.size());
    return this->positions[index];
  }

  vec3 get_normal(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, normals.size());
    return this->normals[index];
  }

  vec2 get_texcoord(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, texcoords.size());
    return this->texcoords[index];
  }

  uint32_t get_index(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, indices.size());
    return this->indices[index];
  }

  float* positions_ptr() override { return positions[0].data(); }
  float* normals_ptr() override { return normals[0].data(); }
  float* texcoords_ptr() override { return texcoords[0].data(); }
  uint32_t* indices_ptr() override { return indices.data(); }

  size_t positions_size() const override { return positions.size(); }
  size_t normals_size() const override { return normals.size(); }
  size_t texcoords_size() const override { return texcoords.size(); }
  size_t indices_size() const override { return indices.size(); }

  void set_position(unsigned int index, const vec3& position) override {
    _RECLIB_ASSERT_LT(index, positions.size());
    positions[index] = position;
  }
  void set_normal(unsigned int index, const vec3& normal) override {
    _RECLIB_ASSERT_LT(index, normals.size());
    normals[index] = normal;
  }
  void set_texcoord(unsigned int index, const vec2& texcoord) override {
    _RECLIB_ASSERT_LT(index, texcoords.size());
    texcoords[index] = texcoord;
  }
  void set_index(unsigned int index, uint32_t index_value) override {
    _RECLIB_ASSERT_LT(index, indices.size());
    indices[index] = index_value;
  }

 private:
  std::vector<vec3> positions;
  std::vector<uint32_t> indices;
  std::vector<vec3> normals;
  std::vector<vec2> texcoords;
};

class _API GeometryWrapperImpl : public GeometryBaseImpl {
 public:
  GeometryWrapperImpl(const std::string& name);
  GeometryWrapperImpl(const std::string& name, float* positions,
                      size_t pos_size, uint32_t* indices = nullptr,
                      size_t indices_size = 0, float* normals = nullptr,
                      size_t normals_size = 0, float* texcoords = nullptr,
                      size_t texcoords_size = 0);
  virtual ~GeometryWrapperImpl();

  explicit inline operator bool() const {
    return (positions_size_ > 0) && (indices_size_ > 0);
  }

  // the sizes must be the length of the pointer, e.g. if we have 4 positions
  // with (x,y,z) coordinates the size must be (3*4) = 12
  void set(float* positions, size_t pos_size, uint32_t* indices = nullptr,
           size_t indices_size = 0, float* normals = nullptr,
           size_t normals_size = 0, float* texcoords = nullptr,
           size_t texcoords_size = 0);
  void clear() override;

  bool has_normals() const override { return normals_size_ > 0; }
  bool has_texcoords() const override { return texcoords_size_ > 0; }

  // O(n) geometry operations
  void recompute_aabb() override;
  void fit_into_aabb(const vec3& aabb_min, const vec3& aabb_max) override;
  void translate(const vec3& by) override;
  void scale(const vec3& by) override;
  void rotate(float angle_degrees, const vec3& axis) override;
  void transform(const mat4& trans) override;

  vec3 get_position(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, positions_size_);
    return make_vec3(&positions_ptr_[index * 3]);
  }

  vec3 get_normal(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, normals_size_);
    return make_vec3(&normals_ptr_[index * 3]);
  }

  vec2 get_texcoord(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, texcoords_size_);
    return make_vec2(&texcoords_ptr_[index * 2]);
  }

  uint32_t get_index(unsigned int index) const override {
    _RECLIB_ASSERT_LT(index, indices_size_);
    return indices_ptr_[index];
  }

  float* positions_ptr() override { return positions_ptr_; }
  float* normals_ptr() override { return normals_ptr_; }
  float* texcoords_ptr() override { return texcoords_ptr_; }
  uint32_t* indices_ptr() override { return indices_ptr_; }

  size_t positions_size() const override { return positions_size_; }
  size_t normals_size() const override { return normals_size_; }
  size_t texcoords_size() const override { return texcoords_size_; }
  size_t indices_size() const override { return indices_size_; }

  void set_position(unsigned int index, const vec3& position) override {
    _RECLIB_ASSERT_LT(index, positions_size_);
    positions_ptr_[index * 3 + 0] = position.x();
    positions_ptr_[index * 3 + 1] = position.y();
    positions_ptr_[index * 3 + 2] = position.z();
  }
  void set_normal(unsigned int index, const vec3& normal) override {
    _RECLIB_ASSERT_LT(index, normals_size_);
    normals_ptr_[index * 3 + 0] = normal.x();
    normals_ptr_[index * 3 + 1] = normal.y();
    normals_ptr_[index * 3 + 2] = normal.z();
  }
  void set_texcoord(unsigned int index, const vec2& texcoord) override {
    _RECLIB_ASSERT_LT(index, texcoords_size_);
    texcoords_ptr_[index * 2 + 0] = texcoord.x();
    texcoords_ptr_[index * 2 + 1] = texcoord.y();
  }
  void set_index(unsigned int index, uint32_t index_value) override {
    _RECLIB_ASSERT_LT(index, indices_size_);
    indices_ptr_[index] = index_value;
  }

  void auto_generate_normals(
      const mat4& initial_transform = mat4::Identity()) override;
  void convert_normals_perface2vertex() override;

  // data
 private:
  std::vector<uint32_t> indices;
  std::vector<vec3> normals;

  float* positions_ptr_;
  uint32_t* indices_ptr_;
  float* normals_ptr_;
  float* texcoords_ptr_;

  uint32_t positions_size_{
      0};  // size in vertices, e.g. if we have 4 vertices (x,y,z) - the size is
           // NOT (3*4) but instead only 4!
  uint32_t indices_size_{0};
  uint32_t normals_size_{
      0};  // size in normals, e.g. if we have 4 normals (x,y,z) - the size is
           // NOT (3*4) but instead only 4!
  uint32_t texcoords_size_{
      0};  // size in texcoords, e.g. if we have 4 texcoords (x,y) - the size is
           // NOT (2*4) but instead only 4!

 private:
};

using GeometryBase = NamedHandle<GeometryBaseImpl>;
using Geometry = NamedHandle<GeometryImpl>;
using GeometryWrapper = NamedHandle<GeometryWrapperImpl>;

}  // namespace opengl
}  // namespace reclib

template class _API reclib::opengl::NamedHandle<
    reclib::opengl::GeometryBaseImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::GeometryImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::GeometryWrapperImpl>;  // needed for Windows DLL export

#endif