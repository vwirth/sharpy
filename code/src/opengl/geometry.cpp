#include <reclib/assert.h>
#include <reclib/opengl/geometry.h>
#include <reclib/opengl/mesh.h>

#include <iostream>
#include <numeric>

namespace reclib {
namespace opengl {

std::vector<vec3> face2vertex_normals(const std::vector<vec3>& face_normals,
                                      const uint32_t position_size,
                                      const std::vector<uint32_t>& indices) {
  _RECLIB_ASSERT_GE(indices.size(), position_size);
  _RECLIB_ASSERT_EQ(face_normals.size(), indices.size() / 3);

  std::vector<std::vector<uint32_t>> vertex_appearance(position_size);
  std::vector<vec3> vertex_normals;

  // std::cout << "Face normals" << std::endl;
  // for (unsigned int i = 0; i < face_normals.size(); i++) {
  //   std::cout << "[" << i << "," << i + 3 << "]: " << face_normals[i]
  //             << std::endl;
  // }

  // this assumes that the same vertex is referenced within multiple faces
  // it does not work in cases in which the same vertex is in the vertex buffer
  // multiple times
  for (unsigned int i = 0; i < indices.size(); i++) {
    vertex_appearance[indices[i]].push_back(i / 3);

    // std::cout << "vertex_appearance " << indices[i] << ": " << i / 3
    //           << " normal: " << face_normals[i / 3] << std::endl;
  }
  for (unsigned int i = 0; i < vertex_appearance.size(); i++) {
    vec3 n(0, 0, 0);
    unsigned int c = 0;
    for (unsigned int j = 0; j < vertex_appearance[i].size(); j++) {
      n += face_normals[vertex_appearance[i][j]];
      c++;
    }
    vertex_normals.push_back((n / c).normalized());
  }
  return vertex_normals;
}

std::vector<vec3> face2vertex_normals(const float* face_normals,
                                      const uint32_t normal_size,
                                      const uint32_t position_size,
                                      const uint32_t* indices,
                                      const uint32_t indices_size) {
  _RECLIB_ASSERT_GE(indices_size, position_size);
  _RECLIB_ASSERT_EQ(normal_size, indices_size / 3);

  std::vector<std::vector<uint32_t>> vertex_appearance(position_size);
  std::vector<vec3> vertex_normals;

  for (unsigned int i = 0; i < indices_size; i++) {
    vertex_appearance[indices[i]].push_back(i / 3);
  }

  for (unsigned int i = 0; i < vertex_appearance.size(); i++) {
    vec3 n(0, 0, 0);
    unsigned int c = 0;
    for (unsigned int j = 0; j < vertex_appearance[i].size(); j++) {
      n += vec3(face_normals[vertex_appearance[i][j] * 3 + 0],
                face_normals[vertex_appearance[i][j] * 3 + 1],
                face_normals[vertex_appearance[i][j] * 3 + 2]);
      c++;
    }
    vertex_normals.push_back((n / c).normalized());
  }
  return vertex_normals;
}

std::vector<vec3> face2vertex_normals(const std::vector<vec3>& face_normals,
                                      const uint32_t position_size,
                                      const uint32_t* indices,
                                      const uint32_t indices_size) {
  _RECLIB_ASSERT_GE(indices_size, position_size);
  _RECLIB_ASSERT_EQ(face_normals.size(), indices_size / 3);

  std::vector<std::vector<uint32_t>> vertex_appearance(position_size);
  std::vector<vec3> vertex_normals;

  for (unsigned int i = 0; i < indices_size; i++) {
    vertex_appearance[indices[i]].push_back(i / 3);
  }
  for (unsigned int i = 0; i < vertex_appearance.size(); i++) {
    vec3 n(0, 0, 0);
    unsigned int c = 0;
    for (unsigned int j = 0; j < vertex_appearance[i].size(); j++) {
      n += face_normals[vertex_appearance[i][j]];
      c++;
    }
    vertex_normals.push_back((n / c).normalized());
  }
  return vertex_normals;
}

std::vector<vec3> generate_face_normals(const std::vector<vec3>& positions,
                                        const std::vector<uint32_t>& indices) {
  std::vector<vec3> face_normals;

  for (unsigned int i = 0; i < indices.size(); i = i + 3) {
    vec3 e1 = positions[indices[i + 1]] - positions[indices[i + 0]];
    vec3 e2 = positions[indices[i + 2]] - positions[indices[i + 1]];
    vec3 n = (e1.cross(e2)).normalized();
    face_normals.push_back(n);
  }

  return face_normals;
}

std::vector<vec3> generate_face_normals(const float* positions,
                                        const uint32_t* indices,
                                        const uint32_t indices_size) {
  std::vector<vec3> face_normals;

  for (unsigned int i = 0; i < indices_size; i = i + 3) {
    vec3 p0(positions[indices[i + 0] * 3 + 0],
            positions[indices[i + 0] * 3 + 1],
            positions[indices[i + 0] * 3 + 2]);

    vec3 p1(positions[indices[i + 1] * 3 + 0],
            positions[indices[i + 1] * 3 + 1],
            positions[indices[i + 1] * 3 + 2]);

    vec3 p2(positions[indices[i + 2] * 3 + 0],
            positions[indices[i + 2] * 3 + 1],
            positions[indices[i + 2] * 3 + 2]);

    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p1;
    vec3 n = (e1.cross(e2)).normalized();
    face_normals.push_back(n);
  }

  return face_normals;
}

std::vector<vec3> generate_vertex_normals(
    const std::vector<vec3>& positions, const std::vector<uint32_t>& indices) {
  std::vector<vec3> face_normals = generate_face_normals(positions, indices);
  return face2vertex_normals(face_normals, positions.size(), indices);
}

std::vector<vec3> generate_vertex_normals(const float* positions,
                                          const uint32_t position_size,
                                          const uint32_t* indices,
                                          const uint32_t indices_size) {
  std::vector<vec3> face_normals =
      generate_face_normals(positions, indices, indices_size);
  return face2vertex_normals(face_normals, position_size, indices,
                             indices_size);
}

// -------------------------------------------------------------------
// GeometryBaseImpl
// -------------------------------------------------------------------

reclib::opengl::GeometryBaseImpl::GeometryBaseImpl(const std::string& name)
    : name(name),
      bb_min(std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()),
      bb_max(std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min(),
             std::numeric_limits<float>::min()),
      auto_normals_(false) {}

reclib::opengl::GeometryBaseImpl::~GeometryBaseImpl() {}

float* reclib::opengl::GeometryBaseImpl::positions_ptr() {
  assert(false);
  return nullptr;
};
float* reclib::opengl::GeometryBaseImpl::normals_ptr() {
  assert(false);
  return nullptr;
};
float* reclib::opengl::GeometryBaseImpl::texcoords_ptr() {
  assert(false);
  return nullptr;
};
uint32_t* reclib::opengl::GeometryBaseImpl::indices_ptr() {
  assert(false);
  return nullptr;
};

size_t reclib::opengl::GeometryBaseImpl::positions_size() const {
  assert(false);
  return 0;
};
size_t reclib::opengl::GeometryBaseImpl::normals_size() const {
  assert(false);
  return 0;
};
size_t reclib::opengl::GeometryBaseImpl::texcoords_size() const {
  assert(false);
  return 0;
};
size_t reclib::opengl::GeometryBaseImpl::indices_size() const {
  assert(false);
  return 0;
};

vec3 reclib::opengl::GeometryBaseImpl::get_position(unsigned int index) const {
  assert(false);
  return vec3(0, 0, 0);
};
vec3 reclib::opengl::GeometryBaseImpl::get_normal(unsigned int index) const {
  assert(false);
  return vec3(0, 0, 0);
};
vec2 reclib::opengl::GeometryBaseImpl::get_texcoord(unsigned int index) const {
  assert(false);
  return vec2(0, 0);
};
uint32_t reclib::opengl::GeometryBaseImpl::get_index(unsigned int index) const {
  assert(false);
  return 0;
};

void reclib::opengl::GeometryBaseImpl::set_position(unsigned int index,
                                                    const vec3& position) {
  assert(false);
}
void reclib::opengl::GeometryBaseImpl::set_normal(unsigned int index,
                                                  const vec3& normal) {
  assert(false);
}
void reclib::opengl::GeometryBaseImpl::set_texcoord(unsigned int index,
                                                    const vec2& texcoord) {
  assert(false);
}
void reclib::opengl::GeometryBaseImpl::set_index(unsigned int index,
                                                 uint32_t index_value) {
  assert(false);
}

bool reclib::opengl::GeometryBaseImpl::has_normals() const {
  assert(false);
  return false;
};
bool reclib::opengl::GeometryBaseImpl::has_texcoords() const {
  assert(false);
  return false;
};

void reclib::opengl::GeometryBaseImpl::auto_generate_normals(
    const mat4& initial_transform) {
  assert(false);
}

void reclib::opengl::GeometryBaseImpl::convert_normals_perface2vertex() {
  assert(false);
}

void reclib::opengl::GeometryBaseImpl::register_mesh(
    const NamedHandle<MeshImpl>& mesh) {
  used_by.emplace(mesh->name, mesh);
}
void reclib::opengl::GeometryBaseImpl::unregister_mesh(
    const std::string& name) {
  if (used_by.find(name) != used_by.end()) used_by.erase(name);
}

void reclib::opengl::GeometryBaseImpl::clear_mesh_gpu_memory() {
  for (auto iter = used_by.begin(); iter != used_by.end(); iter++) {
    iter->second->clear_gpu();
  }
}

void reclib::opengl::GeometryBaseImpl::update_meshes() {
  for (auto iter = used_by.begin(); iter != used_by.end(); iter++) {
    iter->second->upload_gpu();
  }
}

void reclib::opengl::GeometryBaseImpl::clear() { _RECLIB_ASSERT(false); }

// O(n) geometry operations
void reclib::opengl::GeometryBaseImpl::recompute_aabb() {
  _RECLIB_ASSERT(false);
}

void reclib::opengl::GeometryBaseImpl::fit_into_aabb(const vec3& aabb_min,
                                                     const vec3& aabb_max) {
  _RECLIB_ASSERT(false);
}

void reclib::opengl::GeometryBaseImpl::translate(const vec3& by) {
  _RECLIB_ASSERT(false);
}

void reclib::opengl::GeometryBaseImpl::scale(const vec3& by) {
  _RECLIB_ASSERT(false);
}

void reclib::opengl::GeometryBaseImpl::rotate(float angle_degrees,
                                              const vec3& axis) {
  _RECLIB_ASSERT(false);
}

void reclib::opengl::GeometryBaseImpl::transform(const mat4& trans) {
  _RECLIB_ASSERT(false);
}

void reclib::opengl::GeometryBaseImpl::add_attribute_vec3(
    const std::string& name, vec3* attribute, uint32_t size, bool external,
    bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(vec3_map.count(name), 0);

  AttributePtr<vec3> ptr;
  ptr.external = external;
  ptr.size = size;
  ptr.dim = 1;
  if (!external) {
    std::vector<vec3> v;
    for (unsigned int i = 0; i < size; i++) {
      v.push_back(attribute[i]);
    }
    vec3_storage.push_back(v);
    ptr.ptr = vec3_storage[vec3_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute;
  }
  vec3_map[name] = ptr;
}
void reclib::opengl::GeometryBaseImpl::add_attribute_vec3(
    const std::string& name, std::vector<vec3>& attribute, bool external,
    bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(vec3_map.count(name), 0);

  AttributePtr<vec3> ptr;
  ptr.external = external;
  ptr.size = attribute.size();
  ptr.dim = 1;
  if (!external) {
    std::vector<vec3> v(attribute);
    vec3_storage.push_back(v);
    ptr.ptr = vec3_storage[vec3_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute.data();
  }
  vec3_map[name] = ptr;
}

void reclib::opengl::GeometryBaseImpl::add_attribute_vec4(
    const std::string& name, vec4* attribute, uint32_t size, bool external,
    bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(vec4_map.count(name), 0);

  AttributePtr<vec4> ptr;
  ptr.external = external;
  ptr.size = size;
  ptr.dim = 1;
  if (!external) {
    std::vector<vec4> v;
    for (unsigned int i = 0; i < size; i++) {
      v.push_back(attribute[i]);
    }
    vec4_storage.push_back(v);
    ptr.ptr = vec4_storage[vec4_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute;
  }
  vec4_map[name] = ptr;
}
void reclib::opengl::GeometryBaseImpl::add_attribute_vec4(
    const std::string& name, std::vector<vec4>& attribute, bool external,
    bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(vec4_map.count(name), 0);

  AttributePtr<vec4> ptr;
  ptr.external = external;
  ptr.size = attribute.size();
  ptr.dim = 1;
  if (!external) {
    std::vector<vec4> v(attribute);
    vec4_storage.push_back(v);
    ptr.ptr = vec4_storage[vec4_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute.data();
  }
  vec4_map[name] = ptr;
}

void reclib::opengl::GeometryBaseImpl::add_attribute_vec2(
    const std::string& name, vec2* attribute, uint32_t size, bool external,
    bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(vec2_map.count(name), 0);

  AttributePtr<vec2> ptr;
  ptr.external = external;
  ptr.size = size;
  ptr.dim = 1;
  if (!external) {
    std::vector<vec2> v;
    for (unsigned int i = 0; i < size; i++) {
      v.push_back(attribute[i]);
    }
    vec2_storage.push_back(v);
    ptr.ptr = vec2_storage[vec2_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute;
  }
  vec2_map[name] = ptr;
}

void reclib::opengl::GeometryBaseImpl::add_attribute_vec2(
    const std::string& name, std::vector<vec2>& attribute, bool external,
    bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(vec2_map.count(name), 0);

  AttributePtr<vec2> ptr;
  ptr.external = external;
  ptr.size = attribute.size();
  ptr.dim = 1;
  if (!external) {
    std::vector<vec2> v(attribute);
    vec2_storage.push_back(v);
    ptr.ptr = vec2_storage[vec2_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute.data();
  }
  vec2_map[name] = ptr;
}

void reclib::opengl::GeometryBaseImpl::add_attribute_float(
    const std::string& name, float* attribute, uint32_t size, uint32_t dim,
    bool external, bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(float_map.count(name), 0);

  AttributePtr<float> ptr;
  ptr.external = external;
  ptr.size = size;
  ptr.dim = dim;
  if (!external) {
    std::vector<float> v;
    for (unsigned int i = 0; i < size * dim; i++) {
      v.push_back(attribute[i]);
    }
    _RECLIB_ASSERT_EQ(v.size(), ptr.size * ptr.dim);
    float_storage.push_back(v);
    ptr.ptr = float_storage[float_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute;
  }
  float_map[name] = ptr;
}
void reclib::opengl::GeometryBaseImpl::add_attribute_float(
    const std::string& name, std::vector<float>& attribute, uint32_t dim,
    bool external, bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(float_map.count(name), 0);

  AttributePtr<float> ptr;
  ptr.external = external;
  ptr.size = attribute.size() / dim;
  ptr.dim = dim;
  if (!external) {
    std::vector<float> v(attribute);
    _RECLIB_ASSERT_EQ(v.size(), ptr.size * ptr.dim);
    float_storage.push_back(v);
    ptr.ptr = float_storage[float_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute.data();
  }
  float_map[name] = ptr;
}

void reclib::opengl::GeometryBaseImpl::add_attribute_uint(
    const std::string& name, uint32_t* attribute, uint32_t size, uint32_t dim,
    bool external, bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(uint_map.count(name), 0);

  AttributePtr<uint32_t> ptr;
  ptr.external = external;
  ptr.size = size;
  ptr.dim = dim;
  if (!external) {
    std::vector<uint32_t> v;
    for (unsigned int i = 0; i < size * dim; i++) {
      v.push_back(attribute[i]);
    }
    _RECLIB_ASSERT_EQ(v.size(), ptr.size * ptr.dim);
    uint_storage.push_back(v);
    ptr.ptr = uint_storage[uint_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute;
  }
  uint_map[name] = ptr;
}
void reclib::opengl::GeometryBaseImpl::add_attribute_uint(
    const std::string& name, std::vector<uint32_t>& attribute, uint32_t dim,
    bool external, bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(uint_map.count(name), 0);

  AttributePtr<uint32_t> ptr;
  ptr.external = external;
  ptr.size = attribute.size() / dim;
  ptr.dim = dim;
  if (!external) {
    std::vector<uint32_t> v(attribute);
    uint_storage.push_back(v);
    _RECLIB_ASSERT_EQ(v.size(), ptr.size * ptr.dim);
    ptr.ptr = uint_storage[uint_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute.data();
  }
  uint_map[name] = ptr;
}

void reclib::opengl::GeometryBaseImpl::add_attribute_int(
    const std::string& name, int32_t* attribute, uint32_t size, uint32_t dim,
    bool external, bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(int_map.count(name), 0);

  AttributePtr<int32_t> ptr;
  ptr.external = external;
  ptr.size = size;
  ptr.dim = dim;
  if (!external) {
    std::vector<int32_t> v;
    for (unsigned int i = 0; i < size * dim; i++) {
      v.push_back(attribute[i]);
    }
    _RECLIB_ASSERT_EQ(v.size(), ptr.size * ptr.dim);
    int_storage.push_back(v);
    ptr.ptr = int_storage[int_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute;
  }
  int_map[name] = ptr;
}
void reclib::opengl::GeometryBaseImpl::add_attribute_int(
    const std::string& name, std::vector<int32_t>& attribute, uint32_t dim,
    bool external, bool overwrite) {
  if (!overwrite) _RECLIB_ASSERT_EQ(int_map.count(name), 0);

  AttributePtr<int32_t> ptr;
  ptr.external = external;
  ptr.size = attribute.size() / dim;
  ptr.dim = dim;
  if (!external) {
    std::vector<int32_t> v(attribute);
    int_storage.push_back(v);
    _RECLIB_ASSERT_EQ(v.size(), ptr.size * ptr.dim);
    ptr.ptr = int_storage[int_storage.size() - 1].data();
  } else {
    ptr.ptr = attribute.data();
  }
  int_map[name] = ptr;
}

// -------------------------------------------------------------------
// GeometryImpl
// -------------------------------------------------------------------

reclib::opengl::GeometryImpl::GeometryImpl(const std::string& name)
    : reclib::opengl::GeometryBaseImpl(name) {}

reclib::opengl::GeometryImpl::GeometryImpl(const std::string& name,
                                           const aiMesh* mesh_ai)
    : GeometryImpl(name) {
  add(mesh_ai);
}

reclib::opengl::GeometryImpl::GeometryImpl(const std::string& name,
                                           const std::vector<vec3>& positions,
                                           const std::vector<uint32_t>& indices,
                                           const std::vector<vec3>& normals,
                                           const std::vector<vec2>& texcoords)
    : GeometryImpl(name) {
  _RECLIB_ASSERT(normals.size() == 0 || positions.size() == normals.size());
  _RECLIB_ASSERT(texcoords.size() == 0 || positions.size() == texcoords.size());
  _RECLIB_ASSERT(indices.size() == 0 || indices.size() >= positions.size());

  if (indices.size() == 0) {
    std::vector<uint32_t> auto_indices(positions.size());
    std::iota(auto_indices.begin(), auto_indices.end(), 0);
    add(positions, auto_indices, normals, texcoords);
  } else {
    add(positions, indices, normals, texcoords);
  }
}

reclib::opengl::GeometryImpl::GeometryImpl(const std::string& name,
                                           const std::vector<float>& positions,
                                           const std::vector<uint32_t>& indices,
                                           const std::vector<float>& normals,
                                           const std::vector<float>& texcoords)
    : GeometryImpl(name) {
  _RECLIB_ASSERT(normals.size() == 0 || positions.size() == normals.size());
  _RECLIB_ASSERT(texcoords.size() == 0 ||
                 positions.size() / 3 == texcoords.size() / 2);
  _RECLIB_ASSERT_EQ(indices.size(),
                    0 || indices.size() >= positions.size() / 3);

  if (indices.size() == 0) {
    std::vector<uint32_t> auto_indices(positions.size());
    std::iota(auto_indices.begin(), auto_indices.end(), 0);
    add(positions, auto_indices, normals, texcoords);
  } else {
    add(positions, indices, normals, texcoords);
  }
}

reclib::opengl::GeometryImpl::GeometryImpl(
    const std::string& name, const float* positions, size_t pos_size,
    const uint32_t* indices, size_t indices_size, const float* normals,
    size_t normals_size_, const float* texcoords, size_t texcoords_size_)
    : GeometryImpl(name) {
  _RECLIB_ASSERT(normals_size_ == 0 || pos_size == normals_size_);
  _RECLIB_ASSERT(texcoords_size_ == 0 || pos_size == texcoords_size_);
  _RECLIB_ASSERT(indices_size == 0 || indices_size >= pos_size);

  if (indices_size == 0) {
    std::vector<uint32_t> auto_indices(pos_size);
    std::iota(auto_indices.begin(), auto_indices.end(), 0);
    add(positions, pos_size, auto_indices.data(), auto_indices.size(), normals,
        normals_size_, texcoords, texcoords_size_);
  } else {
    add(positions, pos_size, indices, indices_size, normals, normals_size_,
        texcoords, texcoords_size_);
  }
}

reclib::opengl::GeometryImpl::~GeometryImpl() {}

void reclib::opengl::GeometryImpl::add(const aiMesh* mesh_ai) {
  // conversion helper
  const auto to_eigen = [](const aiVector3D& v) { return vec3(v.x, v.y, v.z); };
  // extract vertices, normals and texture coords
  positions.reserve(positions.size() + mesh_ai->mNumVertices);
  normals.reserve(normals.size() +
                  (mesh_ai->HasNormals() ? mesh_ai->mNumVertices : 0));
  texcoords.reserve(texcoords.size() +
                    (mesh_ai->HasTextureCoords(0) ? mesh_ai->mNumVertices : 0));
  for (uint32_t i = 0; i < mesh_ai->mNumVertices; ++i) {
    positions.emplace_back(to_eigen(mesh_ai->mVertices[i]));
    if (mesh_ai->HasNormals())
      normals.emplace_back(to_eigen(mesh_ai->mNormals[i]));
    if (mesh_ai->HasTextureCoords(0))
      texcoords.emplace_back(
          reclib::make_vec2(to_eigen(mesh_ai->mTextureCoords[0][i])));
    // update AABB
    bb_min = min(bb_min, to_eigen(mesh_ai->mVertices[i]));
    bb_max = max(bb_max, to_eigen(mesh_ai->mVertices[i]));
  }
  unsigned int index_start_index = this->indices.size();
  // extract faces
  indices.reserve(indices.size() + mesh_ai->mNumFaces * 3);
  for (uint32_t i = 0; i < mesh_ai->mNumFaces; ++i) {
    const aiFace& face = mesh_ai->mFaces[i];
    if (face.mNumIndices == 3) {
      indices.emplace_back(face.mIndices[0]);
      indices.emplace_back(face.mIndices[1]);
      indices.emplace_back(face.mIndices[2]);
    } else
      std::cerr << "WARN: Geometry: skipping non-triangle face!" << std::endl;
  }

  if (!mesh_ai->HasNormals() && auto_normals_) {
    this->normals.reserve(this->normals.size() + positions.size());

    std::vector<uint32_t> new_indices(this->indices.begin() + index_start_index,
                                      this->indices.end());
    std::vector<vec3> vertex_normals =
        generate_vertex_normals(positions, new_indices);

    this->normals.insert(this->normals.end(), vertex_normals.begin(),
                         vertex_normals.end());
  }
}

void reclib::opengl::GeometryImpl::add(const GeometryImpl& other) {
  add(other.positions, other.indices, other.normals, other.texcoords);
}

void reclib::opengl::GeometryImpl::add(const std::vector<vec3>& positions,
                                       const std::vector<uint32_t>& indices,
                                       const std::vector<vec3>& normals,
                                       const std::vector<vec2>& texcoords) {
  _RECLIB_ASSERT(normals.size() == 0 || positions.size() == normals.size());
  _RECLIB_ASSERT(texcoords.size() == 0 || positions.size() == texcoords.size());
  _RECLIB_ASSERT(indices.size() == 0 || indices.size() >= positions.size());

  // add vertices, normals and texture coords
  this->positions.reserve(this->positions.size() + positions.size());
  this->normals.reserve(this->normals.size() + normals.size());
  this->texcoords.reserve(this->texcoords.size() + texcoords.size());
  for (uint32_t i = 0; i < positions.size(); ++i) {
    this->positions.emplace_back(positions[i]);
    if (i < normals.size()) this->normals.emplace_back(normals[i]);
    if (i < texcoords.size()) this->texcoords.emplace_back(texcoords[i]);
    // update AABB
    bb_min = min(bb_min, positions[i]);
    bb_max = max(bb_max, positions[i]);
  }
  unsigned int index_start_index = this->indices.size();
  // add indices
  this->indices.reserve(this->indices.size() + indices.size());
  for (uint32_t i = 0; i < indices.size(); ++i)
    this->indices.emplace_back(indices[i]);

  if (normals.size() == 0 && auto_normals_) {
    this->normals.reserve(this->normals.size() + positions.size());

    std::vector<uint32_t> new_indices(this->indices.begin() + index_start_index,
                                      this->indices.end());
    std::vector<vec3> vertex_normals =
        generate_vertex_normals(positions, new_indices);

    this->normals.insert(this->normals.end(), vertex_normals.begin(),
                         vertex_normals.end());
  }
}

void reclib::opengl::GeometryImpl::add(const std::vector<float>& positions,
                                       const std::vector<uint32_t>& indices,
                                       const std::vector<float>& normals,
                                       const std::vector<float>& texcoords) {
  _RECLIB_ASSERT(normals.size() == 0 || positions.size() == normals.size());
  _RECLIB_ASSERT(texcoords.size() == 0 ||
                 positions.size() / 3 == texcoords.size() / 2);
  _RECLIB_ASSERT(indices.size() == 0 || indices.size() >= positions.size() / 3);

  // add vertices, normals and texture coords
  this->positions.reserve(this->positions.size() + (positions.size() / 3));
  this->normals.reserve(this->normals.size() + (normals.size() / 3));
  this->texcoords.reserve(this->texcoords.size() + (texcoords.size() / 2));

  const uint32_t start_index = this->positions.size();
  for (uint32_t i = 0; i < (positions.size() / 3); ++i) {
    this->positions.emplace_back(
        vec3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
    if (i < (normals.size() / 3))
      this->normals.emplace_back(
          vec3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]));
    if (i < (texcoords.size() / 2))
      this->texcoords.emplace_back(
          vec2(texcoords[i * 2 + 0], texcoords[i * 2 + 1]));
    // update AABB
    bb_min = min(bb_min, this->positions[start_index + i]);
    bb_max = max(bb_max, this->positions[start_index + i]);
  }
  unsigned int index_start_index = this->indices.size();
  // add indices
  this->indices.reserve(this->indices.size() + indices.size());
  for (uint32_t i = 0; i < indices.size(); ++i)
    this->indices.emplace_back(indices[i]);

  if (normals.size() == 0 && auto_normals_) {
    this->normals.reserve(this->normals.size() + positions.size() / 3);

    std::vector<uint32_t> new_indices(this->indices.begin() + index_start_index,
                                      this->indices.end());
    std::vector<vec3> vertex_normals =
        generate_vertex_normals(positions.data(), positions.size() / 3,
                                new_indices.data(), new_indices.size());

    this->normals.insert(this->normals.end(), vertex_normals.begin(),
                         vertex_normals.end());
  }
}

void reclib::opengl::GeometryImpl::set(
    const float* positions, size_t pos_size, const uint32_t* indices,
    size_t indices_size, const float* normals, size_t normals_size_,
    const float* texcoords, size_t texcoords_size_) {
  _RECLIB_ASSERT(normals_size_ == 0 || pos_size == normals_size_);
  _RECLIB_ASSERT(texcoords_size_ == 0 || pos_size == texcoords_size_);
  _RECLIB_ASSERT(indices_size == 0 || indices_size >= pos_size);

  clear_mesh_gpu_memory();
  clear();

  // add vertices, normals and texture coords
  this->positions.reserve(this->positions.size() + pos_size);
  this->normals.reserve(this->normals.size() + (normals_size_));
  this->texcoords.reserve(this->texcoords.size() + (texcoords_size_));

  const uint32_t start_index = this->positions.size();
  for (uint32_t i = 0; i < (pos_size); ++i) {
    this->positions.emplace_back(
        vec3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
    if (i < (normals_size_))
      this->normals.emplace_back(
          vec3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]));
    if (i < (texcoords_size_))
      this->texcoords.emplace_back(
          vec2(texcoords[i * 2 + 0], texcoords[i * 2 + 1]));
    // update AABB
    bb_min = min(bb_min, this->positions[start_index + i]);
    bb_max = max(bb_max, this->positions[start_index + i]);
  }
  // add indices
  this->indices.reserve(this->indices.size() + indices_size);
  for (uint32_t i = 0; i < indices_size; ++i)
    this->indices.emplace_back(indices[i]);

  if (!indices) {
    this->indices.resize(pos_size);
    std::iota(this->indices.begin(), this->indices.end(), 0);
  }

  if (normals_size_ == 0 && auto_normals_) {
    this->normals.reserve(this->normals.size() + pos_size);

    std::vector<vec3> vertex_normals = generate_vertex_normals(
        positions, pos_size, this->indices.data(), this->indices.size());

    this->normals.insert(this->normals.end(), vertex_normals.begin(),
                         vertex_normals.end());
  }
}

void reclib::opengl::GeometryImpl::add(
    const float* positions, size_t pos_size, const uint32_t* indices,
    size_t indices_size, const float* normals, size_t normals_size_,
    const float* texcoords, size_t texcoords_size_) {
  _RECLIB_ASSERT(normals_size_ == 0 || pos_size == normals_size_);
  _RECLIB_ASSERT(texcoords_size_ == 0 || pos_size == texcoords_size_);
  _RECLIB_ASSERT(indices_size == 0 || indices_size >= pos_size);

  // add vertices, normals and texture coords
  this->positions.reserve(this->positions.size() + (pos_size));
  this->normals.reserve(this->normals.size() + (normals_size_));
  this->texcoords.reserve(this->texcoords.size() + (texcoords_size_));

  const uint32_t start_index = this->positions.size();
  for (uint32_t i = 0; i < (pos_size); ++i) {
    this->positions.emplace_back(
        vec3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
    if (i < (normals_size_))
      this->normals.emplace_back(
          vec3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]));
    if (i < (texcoords_size_))
      this->texcoords.emplace_back(
          vec2(texcoords[i * 2 + 0], texcoords[i * 2 + 1]));
    // update AABB
    bb_min = min(bb_min, this->positions[start_index + i]);
    bb_max = max(bb_max, this->positions[start_index + i]);
  }

  // add indices
  const uint32_t index_start_index = this->indices.size();
  this->indices.reserve(this->indices.size() + indices_size);
  if (!indices) {
    this->indices.resize(this->indices.size() + pos_size);
    std::iota(this->indices.begin() + index_start_index, this->indices.end(),
              index_start_index);
  } else {
    for (uint32_t i = 0; i < indices_size; ++i)
      this->indices.emplace_back(indices[i]);
  }

  if (normals_size_ == 0 && auto_normals_) {
    this->normals.reserve(this->normals.size() + pos_size);
    std::vector<vec3> face_normals;

    std::vector<uint32_t> new_indices(this->indices.begin() + index_start_index,
                                      this->indices.end());
    std::vector<vec3> vertex_normals = generate_vertex_normals(
        positions, pos_size, new_indices.data(), new_indices.size());

    this->normals.insert(this->normals.end(), vertex_normals.begin(),
                         vertex_normals.end());
  }
}

void reclib::opengl::GeometryImpl::auto_generate_normals(
    const mat4& initial_transform) {
  _RECLIB_ASSERT_EQ(normals.size(), 0);
  auto_normals_ = true;

  std::vector<vec3> vertex_normals =
      generate_vertex_normals(positions, indices);
  for (unsigned int i = 0; i < vertex_normals.size(); i++) {
    vertex_normals[i] =
        (initial_transform.block<3, 3>(0, 0) * vertex_normals[i]).normalized();
  }

  this->normals.insert(this->normals.end(), vertex_normals.begin(),
                       vertex_normals.end());
}

void reclib::opengl::GeometryImpl::convert_normals_perface2vertex() {
  _RECLIB_ASSERT_EQ(normals.size(), positions.size());
  this->normals.clear();
  auto_generate_normals();
  // auto_normals_ = true;

  // std::vector<vec3> updated_normals;
  // for (unsigned int i = 0; i < normals.size(); i = i + 3) {
  //   // updated_normals.push_back(normals[indices[i]]);
  //   updated_normals.push_back(normals[indices[i]]);
  // }

  // std::vector<vec3> vertex_normals =
  //     face2vertex_normals(updated_normals, positions.size(), indices);

  // this->normals.clear();
  // this->normals.insert(this->normals.end(), vertex_normals.begin(),
  //                      vertex_normals.end());
}

void reclib::opengl::GeometryImpl::clear() {
  bb_min =
      vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
           std::numeric_limits<float>::max());
  bb_max =
      vec3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
           std::numeric_limits<float>::min());
  if (positions.size()) {
    positions.clear();
  }
  if (indices.size()) {
    indices.clear();
  }
  if (normals.size()) {
    normals.clear();
  }
  if (texcoords.size()) {
    texcoords.clear();
  }
}

void reclib::opengl::GeometryImpl::recompute_aabb() {
  bb_min =
      vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
           std::numeric_limits<float>::max());
  bb_max =
      vec3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
           std::numeric_limits<float>::min());

  for (const auto& pos : positions) {
    bb_min = min(bb_min, pos);
    bb_max = max(bb_max, pos);
  }
}

void reclib::opengl::GeometryImpl::fit_into_aabb(const vec3& aabb_min,
                                                 const vec3& aabb_max) {
  // compute offset to origin and scale factor
  const vec3 center = (bb_min + bb_max) * .5f;
  const vec3 scale_v = (aabb_max - aabb_min) / (bb_max - bb_min);
  const float scale_f =
      std::min(scale_v.x(), std::min(scale_v.y(), scale_v.z()));
  // apply
  for (uint32_t i = 0; i < positions.size(); ++i)
    positions[i] = (positions[i] - center) * scale_f;
}

void reclib::opengl::GeometryImpl::translate(const vec3& by) {
  for (uint32_t i = 0; i < positions.size(); ++i) positions[i] += by;
}

void reclib::opengl::GeometryImpl::scale(const vec3& by) {
  const mat4 s = reclib::scale(mat4::Identity(), by);
  const mat4 s_inv = reclib::inverse(s);
  const mat4 mat_norm = reclib::transpose(s_inv);

  // scale positions
  for (uint32_t i = 0; i < positions.size(); ++i)
    positions[i] = positions[i].cwiseProduct(by);
  // scale normals
  for (uint32_t i = 0; i < normals.size(); ++i)
    normals[i] = reclib::normalize(
        reclib::make_vec3(mat_norm * reclib::make_vec4(normals[i], 0)));
}

void reclib::opengl::GeometryImpl::rotate(float angle_degrees,
                                          const vec3& axis) {
  const mat4 rot = reclib::rotate(reclib::radians(angle_degrees), axis);
  const mat4 rot_inv_tra = reclib::transpose(reclib::inverse(rot));

  // rotate positions
  for (uint32_t i = 0; i < positions.size(); ++i)
    positions[i] = reclib::make_vec3(rot * reclib::make_vec4(positions[i], 1));
  // rotate normals
  for (uint32_t i = 0; i < normals.size(); ++i)
    normals[i] = reclib::normalize(
        reclib::make_vec3(rot_inv_tra * reclib::make_vec4(normals[i], 0)));
}

void reclib::opengl::GeometryImpl::transform(const mat4& trans) {
  mat3 rot_inv_tra = trans.block<3, 3>(0, 0).inverse().transpose();
  // rotate positions
  for (uint32_t i = 0; i < positions.size(); ++i)
    positions[i] =
        reclib::make_vec3(trans * reclib::make_vec4(positions[i], 1));
  // rotate normals
  for (uint32_t i = 0; i < normals.size(); ++i)
    normals[i] = (rot_inv_tra * normals[i]).normalized();
}

// -------------------------------------------------------------------
// GeometryWrapperImpl
// -------------------------------------------------------------------

reclib::opengl::GeometryWrapperImpl::GeometryWrapperImpl(
    const std::string& name)
    : reclib::opengl::GeometryBaseImpl(name) {}

reclib::opengl::GeometryWrapperImpl::GeometryWrapperImpl(
    const std::string& name, float* positions, size_t pos_size,
    uint32_t* indices, size_t indices_size, float* normals,
    size_t normals_size_, float* texcoords, size_t texcoords_size_)
    : GeometryWrapperImpl(name) {
  _RECLIB_ASSERT(normals_size_ == 0 || pos_size == normals_size_);
  _RECLIB_ASSERT(texcoords_size_ == 0 || pos_size == texcoords_size_);
  _RECLIB_ASSERT(indices_size == 0 || indices_size >= pos_size);

  set(positions, pos_size, indices, indices_size, normals, normals_size_,
      texcoords, texcoords_size_);
}

reclib::opengl::GeometryWrapperImpl::~GeometryWrapperImpl() {}

void reclib::opengl::GeometryWrapperImpl::set(
    float* positions, size_t pos_size, uint32_t* indices, size_t indices_size,
    float* normals, size_t normals_size_, float* texcoords,
    size_t texcoords_size_) {
  _RECLIB_ASSERT(normals_size_ == 0 || pos_size == normals_size_);
  _RECLIB_ASSERT(texcoords_size_ == 0 || pos_size == texcoords_size_);
  _RECLIB_ASSERT(indices_size == 0 || indices_size >= pos_size);

  // clear_mesh_gpu_memory();
  clear();

  for (uint32_t i = 0; i < pos_size; ++i) {
    // update AABB
    bb_min = min(bb_min, vec3(positions[i * 3 + 0], positions[i * 3 + 1],
                              positions[i * 3 + 2]));
    bb_max = max(bb_max, vec3(positions[i * 3 + 0], positions[i * 3 + 1],
                              positions[i * 3 + 2]));
  }

  this->positions_ptr_ = positions;
  this->indices_ptr_ = indices;
  this->normals_ptr_ = normals;
  this->texcoords_ptr_ = texcoords;

  this->positions_size_ = pos_size;
  this->indices_size_ = indices_size;
  this->normals_size_ = normals_size_;
  this->texcoords_size_ = texcoords_size_;

  if (!indices) {
    this->indices.resize(pos_size);
    std::iota(this->indices.begin(), this->indices.end(), 0);
    this->indices_ptr_ = this->indices.data();
    this->indices_size_ = this->indices.size();
  }

  if (normals_size_ == 0 && auto_normals_) {
    this->normals.reserve(this->normals.size() + pos_size);

    std::vector<vec3> vertex_normals = generate_vertex_normals(
        positions_ptr_, positions_size_, indices_ptr_, indices_size_);

    this->normals.insert(this->normals.end(), vertex_normals.begin(),
                         vertex_normals.end());

    this->normals_ptr_ = this->normals[0].data();
    this->normals_size_ = this->normals.size();
  }
}

void reclib::opengl::GeometryWrapperImpl::auto_generate_normals(
    const mat4& initial_transform) {
  _RECLIB_ASSERT_EQ(normals.size(), 0);
  auto_normals_ = true;

  this->normals.reserve(this->normals.size() + this->positions_size_);

  std::vector<vec3> vertex_normals = generate_vertex_normals(
      positions_ptr_, positions_size_, indices_ptr_, indices_size_);
  for (unsigned int i = 0; i < vertex_normals.size(); i++) {
    vertex_normals[i] =
        (initial_transform.block<3, 3>(0, 0) * vertex_normals[i]).normalized();
  }

  this->normals.insert(this->normals.end(), vertex_normals.begin(),
                       vertex_normals.end());

  this->normals_ptr_ = this->normals[0].data();
  this->normals_size_ = this->normals.size();
}

void reclib::opengl::GeometryWrapperImpl::convert_normals_perface2vertex() {
  _RECLIB_ASSERT_EQ(normals_size_, positions_size_);
  this->normals_ptr_ = nullptr;
  this->normals.clear();
  this->normals_size_ = 0;
  auto_generate_normals();
  // auto_normals_ = true;

  // std::vector<vec3> updated_normals;
  // for (unsigned int i = 0; i < normals.size(); i = i + 3) {
  //   updated_normals.push_back(get_normal(get_index(i)));
  // }

  // std::vector<vec3> vertex_normals =
  //     face2vertex_normals(updated_normals[0].data(), updated_normals.size(),
  //                         positions_size_, indices_ptr_, indices_size_);

  // this->normals.clear();
  // this->normals.reserve(this->normals.size() + this->positions_size_);
  // this->normals.insert(this->normals.end(), vertex_normals.begin(),
  //                      vertex_normals.end());

  // this->normals_ptr_ = this->normals[0].data();
  // this->normals_size_ = this->normals.size();
}

void reclib::opengl::GeometryWrapperImpl::clear() {
  bb_min.x() = std::numeric_limits<float>::max();
  bb_min.y() = std::numeric_limits<float>::max();
  bb_min.z() = std::numeric_limits<float>::max();

  bb_max.x() = std::numeric_limits<float>::min();
  bb_max.y() = std::numeric_limits<float>::min();
  bb_max.z() = std::numeric_limits<float>::min();

  if (indices.size()) {
    indices.clear();
  }
  if (normals.size()) {
    normals.clear();
  }

  positions_size_ = 0;
  indices_size_ = 0;
  normals_size_ = 0;
  texcoords_size_ = 0;

  positions_ptr_ = nullptr;
  indices_ptr_ = nullptr;
  normals_ptr_ = nullptr;
  texcoords_ptr_ = nullptr;
}

void reclib::opengl::GeometryWrapperImpl::recompute_aabb() {
  bb_min =
      vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
           std::numeric_limits<float>::max());
  bb_max =
      vec3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
           std::numeric_limits<float>::min());

  for (unsigned int i = 0; i < positions_size_; i++) {
    bb_min =
        min(bb_min, vec3(positions_ptr_[i * 3 + 0], positions_ptr_[i * 3 + 1],
                         positions_ptr_[i * 3 + 2]));
    bb_max =
        max(bb_max, vec3(positions_ptr_[i * 3 + 0], positions_ptr_[i * 3 + 1],
                         positions_ptr_[i * 3 + 2]));
  }
}

void reclib::opengl::GeometryWrapperImpl::fit_into_aabb(const vec3& aabb_min,
                                                        const vec3& aabb_max) {
  // compute offset to origin and scale factor
  const vec3 center = (bb_min + bb_max) * .5f;
  const vec3 scale_v = (aabb_max - aabb_min) / (bb_max - bb_min);
  const float scale_f =
      std::min(scale_v.x(), std::min(scale_v.y(), scale_v.z()));
  // apply

  for (uint32_t i = 0; i < positions_size_; ++i) {
    positions_ptr_[i * 3 + 0] =
        (positions_ptr_[i * 3 + 0] - center.x()) * scale_f;
    positions_ptr_[i * 3 + 1] =
        (positions_ptr_[i * 3 + 1] - center.y()) * scale_f;
    positions_ptr_[i * 3 + 2] =
        (positions_ptr_[i * 3 + 2] - center.z()) * scale_f;
  }
}

void reclib::opengl::GeometryWrapperImpl::translate(const vec3& by) {
  for (uint32_t i = 0; i < positions_size_; ++i) {
    positions_ptr_[i * 3 + 0] = (positions_ptr_[i * 3 + 0] + by.x());
    positions_ptr_[i * 3 + 1] = (positions_ptr_[i * 3 + 1] + by.y());
    positions_ptr_[i * 3 + 2] = (positions_ptr_[i * 3 + 2] + by.z());
  }
}

void reclib::opengl::GeometryWrapperImpl::scale(const vec3& by) {
  const mat4 s = reclib::scale(mat4::Identity(), by);
  const mat4 s_inv = reclib::inverse(s);
  const mat4 mat_norm = reclib::transpose(s_inv);

  for (uint32_t i = 0; i < positions_size_; ++i) {
    positions_ptr_[i * 3 + 0] = (positions_ptr_[i * 3 + 0] * by.x());
    positions_ptr_[i * 3 + 1] = (positions_ptr_[i * 3 + 1] * by.y());
    positions_ptr_[i * 3 + 2] = (positions_ptr_[i * 3 + 2] * by.z());
  }

  for (uint32_t i = 0; i < normals_size_; ++i) {
    normals_ptr_[i * 3 + 0] = (mat_norm(0, 0) * normals_ptr_[i * 3 + 0] +
                               mat_norm(0, 1) * normals_ptr_[i * 3 + 1] +
                               mat_norm(0, 2) * normals_ptr_[i * 3 + 2]);
    normals_ptr_[i * 3 + 1] = (mat_norm(1, 0) * normals_ptr_[i * 3 + 0] +
                               mat_norm(1, 1) * normals_ptr_[i * 3 + 1] +
                               mat_norm(1, 2) * normals_ptr_[i * 3 + 2]);
    normals_ptr_[i * 3 + 2] = (mat_norm(2, 0) * normals_ptr_[i * 3 + 0] +
                               mat_norm(2, 1) * normals_ptr_[i * 3 + 1] +
                               mat_norm(2, 2) * normals_ptr_[i * 3 + 2]);

    float norm =
        sqrt(pow(normals_ptr_[i * 3 + 0], 2) + pow(normals_ptr_[i * 3 + 1], 2) +
             pow(normals_ptr_[i * 3 + 2], 2));

    normals_ptr_[i * 3 + 0] /= norm;
    normals_ptr_[i * 3 + 1] /= norm;
    normals_ptr_[i * 3 + 2] /= norm;
  }
}

void reclib::opengl::GeometryWrapperImpl::rotate(float angle_degrees,
                                                 const vec3& axis) {
  const mat4 rot = reclib::rotate(reclib::radians(angle_degrees), axis);
  const mat4 rot_inv_tra = reclib::transpose(reclib::inverse(rot));

  for (uint32_t i = 0; i < positions_size_; ++i) {
    vec3 pos_copy = reclib::make_vec3(&(positions_ptr_[i * 3 + 0]));
    Eigen::Map<vec3> pos(&(positions_ptr_[i * 3 + 0]));
    pos = (rot * pos_copy.homogeneous()).head<3>();
  }

  for (uint32_t i = 0; i < normals_size_; ++i) {
    vec3 norm_copy = reclib::make_vec3(&(normals_ptr_[i * 3 + 0]));
    Eigen::Map<vec3> norm(&(normals_ptr_[i * 3 + 0]));
    norm = (rot_inv_tra * norm_copy.homogeneous()).head<3>();
    norm.normalize();
  }
}

void reclib::opengl::GeometryWrapperImpl::transform(const mat4& trans) {
  mat3 rot_inv_tra = trans.block<3, 3>(0, 0).inverse().transpose();

  for (uint32_t i = 0; i < positions_size_; ++i) {
    vec3 pos_copy = reclib::make_vec3(&(positions_ptr_[i * 3 + 0]));
    Eigen::Map<vec3> pos(&(positions_ptr_[i * 3 + 0]));
    pos = (trans * pos_copy.homogeneous()).head<3>();
  }

  for (uint32_t i = 0; i < normals_size_; ++i) {
    vec3 norm_copy = reclib::make_vec3(&(normals_ptr_[i * 3 + 0]));
    Eigen::Map<vec3> norm(&(normals_ptr_[i * 3 + 0]));
    norm = rot_inv_tra * norm_copy;
    norm.normalize();
  }
}

}  // namespace opengl
}  // namespace reclib
