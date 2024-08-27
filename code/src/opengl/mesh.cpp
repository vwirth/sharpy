#include <assimp/material.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <reclib/opengl/context.h>
#include <reclib/opengl/mesh.h>
#include <reclib/platform.h>

#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <iostream>
#include <vector>

// ------------------------------------------
// helper funcs

inline uint32_t type_to_bytes(GLenum type) {
  switch (type) {
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
      return 1;
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
    case GL_HALF_FLOAT:
      return 2;
    case GL_FLOAT:
    case GL_FIXED:
    case GL_INT:
    case GL_UNSIGNED_INT:
      return 4;
    case GL_DOUBLE:
      return 8;
    default:
      throw std::runtime_error("Unknown GL type!");
  }
}

// ------------------------------------------
// MeshImpl

// XXX: When you pass a geometry of type 'Geometry' or 'GeometryWrapper'
// to this constructor, it will be converted to 'GeometryBase'
// that also means, that the handle will be deleted from
// 'Geometry'/'GeometryWrapper' and instead is now part of 'GeometryBase'
reclib::opengl::MeshImpl::MeshImpl(const std::string& name,
                                   const GeometryBase& geometry,
                                   const Material& material)
    : name(name),
      geometry(geometry),
      material(material),
      vao(0),
      num_vertices(0),
      num_indices(0),
      primitive_type(GL_TRIANGLES) {
  glGenVertexArrays(1, &vao);
  upload_gpu();
}

reclib::opengl::MeshImpl::~MeshImpl() {
  if (this->geometry.initialized()) {
    this->geometry->unregister_mesh(name);
  }

  clear_gpu();
  glDeleteVertexArrays(1, &vao);
}

void reclib::opengl::MeshImpl::destroy_handles() {
  if (this->geometry.initialized()) {
    this->geometry->unregister_mesh(name);
    if (reclib::opengl::Geometry::valid(geometry->name)) {
      reclib::opengl::Geometry elem =
          reclib::opengl::Geometry::find(geometry->name);
      this->geometry = reclib::opengl::Geometry();
      elem.free();
    }
  }
  if (this->material.initialized()) {
    if (reclib::opengl::Material::valid(material->name)) {
      reclib::opengl::Material elem =
          reclib::opengl::Material::find(material->name);
      this->material = reclib::opengl::Material();
      elem.free();
    }
  }
}

void reclib::opengl::MeshImpl::clear_gpu() {
  if (ibo.initialized()) {
    ibo.free(true);
  }
  for (unsigned int i = 0; i < vbos.size(); i++) {
    if (vbos[i].initialized()) {
      vbos[i].free(true);
    }
  }
  vbos.clear();
  vbo_types.clear();
  vbo_dims.clear();
  num_vertices = num_indices = 0;
}

void reclib::opengl::MeshImpl::upload_gpu() {
  if (!geometry) return;
  // free gpu resources
  clear_gpu();
  // (re-)upload data to GL

  // std::cout << "name: " << geometry->name
  //           << " positions: " << geometry->positions_size()
  //           << " normals: " << geometry->normals_size()
  //           << " texcoords: " << geometry->texcoords_size()
  //           << " indices: " << geometry->indices_size() << std::endl;
  add_vertex_buffer(GL_FLOAT, 3, geometry->positions_size(),
                    geometry->positions_ptr());

  if (geometry->has_normals())
    add_vertex_buffer(GL_FLOAT, 3, geometry->normals_size(),
                      geometry->normals_ptr());
  if (geometry->has_texcoords())
    add_vertex_buffer(GL_FLOAT, 2, geometry->texcoords_size(),
                      geometry->texcoords_ptr());

  for (auto iter = geometry->vec3_map.begin(); iter != geometry->vec3_map.end();
       iter++) {
    add_vertex_buffer(GL_FLOAT, 3, iter->second.size, iter->second.ptr->data());
  }
  for (auto iter = geometry->vec4_map.begin(); iter != geometry->vec4_map.end();
       iter++) {
    add_vertex_buffer(GL_FLOAT, 4, iter->second.size, iter->second.ptr->data());
  }
  for (auto iter = geometry->vec2_map.begin(); iter != geometry->vec2_map.end();
       iter++) {
    add_vertex_buffer(GL_FLOAT, 2, iter->second.size, iter->second.ptr->data());
  }
  for (auto iter = geometry->float_map.begin();
       iter != geometry->float_map.end(); iter++) {
    add_vertex_buffer(GL_FLOAT, iter->second.dim, iter->second.size,
                      iter->second.ptr);
  }
  for (auto iter = geometry->uint_map.begin(); iter != geometry->uint_map.end();
       iter++) {
    add_vertex_buffer(GL_UNSIGNED_INT, iter->second.dim, iter->second.size,
                      iter->second.ptr);
  }
  for (auto iter = geometry->int_map.begin(); iter != geometry->int_map.end();
       iter++) {
    add_vertex_buffer(GL_INT, iter->second.dim, iter->second.size,
                      iter->second.ptr);
  }

  add_index_buffer(geometry->indices_size(), geometry->indices_ptr());
}

void reclib::opengl::MeshImpl::bind(const Shader& shader) const {
  glBindVertexArray(vao);
  if (material) material->bind(shader);
}

void reclib::opengl::MeshImpl::draw() const {
  if (ibo)
    glDrawElements(primitive_type, num_indices, GL_UNSIGNED_INT, 0);
  else
    glDrawArrays(primitive_type, 0, num_vertices);
}

void reclib::opengl::MeshImpl::draw_index(uint32_t index) const {
  glDrawElements(primitive_type, 1, GL_UNSIGNED_INT,
                 (GLvoid*)(sizeof(uint32_t) * index));
}

void reclib::opengl::MeshImpl::unbind() const {
  glBindVertexArray(0);
  if (material) material->unbind();
}

uint32_t reclib::opengl::MeshImpl::add_vertex_buffer(GLenum type,
                                                     uint32_t element_dim,
                                                     uint32_t num_vertices,
                                                     const void* data,
                                                     GLenum hint) {
  if (this->num_vertices && this->num_vertices != num_vertices)
    throw std::runtime_error(
        name + " Mesh::add_vertex_buffer: vertex buffer size mismatch! this: " +
        std::to_string(this->num_vertices) +
        " num_vertices: " + std::to_string(num_vertices));
  // setup vbo
  this->num_vertices = num_vertices;

  const uint32_t buf_id = vbos.size();

  vbos.emplace_back(name + "_vertex_buffer_" + std::to_string(buf_id));
  vbos[buf_id]->upload_data(
      data, type_to_bytes(type) * element_dim * num_vertices, hint);
  vbo_types.push_back(type);
  vbo_dims.push_back(element_dim);
  // setup vertex attributes
  glBindVertexArray(vao);
  vbos[buf_id]->bind();
  ;
  glEnableVertexAttribArray(buf_id);
  if (type == GL_BYTE || type == GL_UNSIGNED_BYTE || type == GL_SHORT ||
      type == GL_UNSIGNED_SHORT || type == GL_INT || type == GL_UNSIGNED_INT)
    glVertexAttribIPointer(buf_id, element_dim, type, 0, 0);
  else if (type == GL_DOUBLE)
    glVertexAttribLPointer(buf_id, element_dim, type, 0, 0);
  else
    glVertexAttribPointer(buf_id, element_dim, type, GL_FALSE, 0, 0);
  glBindVertexArray(0);
  vbos[buf_id]->unbind();
  return buf_id;
}

void reclib::opengl::MeshImpl::add_index_buffer(uint32_t num_indices,
                                                const uint32_t* data,
                                                GLenum hint) {
  this->num_indices = num_indices;
  ibo = IBO(name + "_index_buffer");
  ibo->upload_data(data, sizeof(uint32_t) * num_indices, hint);
  // setup vao+ibo
  glBindVertexArray(vao);
  ibo->bind();
  glBindVertexArray(0);
  ibo->unbind();
}

void reclib::opengl::MeshImpl::update_vertex_buffer(uint32_t buf_id,
                                                    const void* data) {
  if (buf_id >= vbos.size())
    throw std::runtime_error(
        "Mesh::update_vertex_buffer: buffer id out of range!");
  vbos[buf_id]->upload_subdata(
      data, 0,
      type_to_bytes(vbo_types[buf_id]) * vbo_dims[buf_id] * num_vertices);
}

void reclib::opengl::MeshImpl::set_primitive_type(GLenum primitive_type) {
  this->primitive_type = primitive_type;
}

void* reclib::opengl::MeshImpl::map_vbo(uint32_t buf_id, GLenum access) const {
  if (buf_id >= vbos.size())
    throw std::runtime_error("Mesh::map_vbo: buffer id out of range!");
  return vbos[buf_id]->map(access);
}

void reclib::opengl::MeshImpl::unmap_vbo(uint32_t buf_id) const {
  if (buf_id >= vbos.size())
    throw std::runtime_error("Mesh::map_vbo: buffer id out of range!");
  vbos[buf_id]->unmap();
}

void* reclib::opengl::MeshImpl::map_ibo(GLenum access) const {
  if (!ibo) throw std::runtime_error("Mesh::map_ibo: no index buffer present!");
  return ibo->map(access);
}

void reclib::opengl::MeshImpl::unmap_ibo() const { ibo->unmap(); }

// ------------------------------------------
// Mesh loader (Ass-Imp)

std::vector<std::pair<reclib::opengl::Geometry, reclib::opengl::Material>>
reclib::opengl::load_meshes_cpu(const fs::path& path, bool normalize,
                                const std::string& mesh_name) {
  _RECLIB_ASSERT(reclib::opengl::Context::initialized);
  // load from disk
  Assimp::Importer importer;
  std::cout << "Loading: " << path << "..." << std::endl;
  const aiScene* scene_ai = importer.ReadFile(
      path.string(),
      aiProcess_Triangulate | aiProcess_GenNormals);  // | aiProcess_FlipUVs);
  if (!scene_ai)                                      // handle error
    throw std::runtime_error("ERROR: Failed to load file: " + path.string() +
                             "!");

  std::string base_name;
  if (mesh_name.length() == 0) {
    base_name = path.filename().replace_extension("").string();
  } else {
    base_name = mesh_name;
  }

  // load geometries
  std::vector<reclib::opengl::Geometry> geometries;
  for (uint32_t i = 0; i < scene_ai->mNumMeshes; ++i) {
    const aiMesh* ai_mesh = scene_ai->mMeshes[i];
    geometries.push_back(Geometry(
        base_name + "_" + ai_mesh->mName.C_Str() + "_" + std::to_string(i),
        ai_mesh));
  }
  // move and scale geometry to fit into [-1, 1]x3?
  if (normalize) {
    vec3 bb_min(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()),
        bb_max(std::numeric_limits<float>::min(),
               std::numeric_limits<float>::min(),
               std::numeric_limits<float>::min());
    for (const auto& geom : geometries) {
      bb_min = min(bb_min, geom->bb_min);
      bb_max = max(bb_max, geom->bb_max);
    }
    const vec3 center = (bb_min + bb_max) * 0.5f;
    const vec3 max = vec3(1), min = vec3(-1);
    const vec3 scale_v = (max - min) / (bb_max - bb_min);
    const float scale_f =
        std::min(scale_v.x(), std::min(scale_v.y(), scale_v.z()));
    for (auto& geom : geometries) {
      geom->translate(-center);
      geom->scale(vec3(scale_f, scale_f, scale_f));
    }
  }

  // load materials
  std::vector<Material> materials;
  for (uint32_t i = 0; i < scene_ai->mNumMaterials; ++i) {
    aiString name_ai;
    scene_ai->mMaterials[i]->Get(AI_MATKEY_NAME, name_ai);
    Material m(base_name + "_" + name_ai.C_Str(), path.parent_path(),
               scene_ai->mMaterials[i]);
    materials.push_back(m);
  }
  // link geometry <-> material
  std::vector<std::pair<Geometry, Material>> result;
  for (uint32_t i = 0; i < scene_ai->mNumMeshes; ++i)
    result.push_back(std::make_pair(
        geometries[i], materials[scene_ai->mMeshes[i]->mMaterialIndex]));
  return result;
}

fs::path subtract_paths(const fs::path& upper, const fs::path& lower) {
  fs::path diff_path;
  fs::path base = lower;

  while (base != upper) {
    diff_path = base.stem() / diff_path;
    base = base.parent_path();
  }

  return diff_path;
}

void reclib::opengl::save_meshes_cpu(
    const std::vector<std::pair<GeometryBase, Material>>& meshes,
    const fs::path& path, const fs::path& texture_base_path,
    bool export_material) {
  std::unique_ptr<aiScene> scene(new aiScene());

  std::unique_ptr<aiNode> node(new aiNode());

  scene->mRootNode = new aiNode();
  scene->mRootNode->mNumMeshes = 1;
  scene->mRootNode->mMeshes = new unsigned[1]{0};
  scene->mMetaData = new aiMetadata();  // workaround, issue #3781

  scene->mNumMeshes = meshes.size();
  scene->mMeshes = new aiMesh*[meshes.size()];

  if (!export_material) {
    scene->mMaterials = new aiMaterial* [1] { new aiMaterial() };
    scene->mNumMaterials = 1;
  } else {
    scene->mMaterials = new aiMaterial*[meshes.size() + 1];
    scene->mMaterials[0] = new aiMaterial();
    scene->mNumMaterials = meshes.size() + 1;
  }

  unsigned int mesh_counter = 0;
  for (auto it : meshes) {
    reclib::opengl::GeometryBase& geo = it.first;
    reclib::opengl::Material& mat = it.second;

    aiMesh* m = new aiMesh();
    if (geo.initialized()) {
      m->mName = geo->name;
      m->mMaterialIndex = 0;

      m->mNumVertices = geo->positions_size();
      m->mVertices = new aiVector3D[geo->positions_size()];
      if (geo->normals_size() == geo->positions_size()) {
        m->mNormals = new aiVector3D[geo->normals_size()];
      }
      if (geo->texcoords_size() == geo->positions_size()) {
        m->mTextureCoords[0] = new aiVector3D[geo->texcoords_size()];
      }

      if (geo->texcoords_size() > 0) {
        m->mNumUVComponents[0] = geo->texcoords_size();
      }
      for (unsigned int i = 0; i < geo->positions_size(); i++) {
        m->mVertices[i].x = geo->get_position(i).x();
        m->mVertices[i].y = geo->get_position(i).y();
        m->mVertices[i].z = geo->get_position(i).z();

        if (geo->normals_size() == geo->positions_size()) {
          m->mNormals[i].x = geo->get_normal(i).x();
          m->mNormals[i].y = geo->get_normal(i).y();
          m->mNormals[i].z = geo->get_normal(i).z();
        }
        if (geo->texcoords_size() == geo->positions_size()) {
          m->mTextureCoords[0][i].x = geo->get_texcoord(i).x();
          m->mTextureCoords[0][i].y = geo->get_texcoord(i).y();
          m->mTextureCoords[0][i].z = 0;
        }
      }

      m->mFaces = new aiFace[geo->indices_size() / 3];
      m->mNumFaces = geo->indices_size() / 3;

      for (unsigned int i = 0; i < geo->indices_size(); i = i + 3) {
        m->mFaces[i / 3].mIndices = new unsigned int[3];
        m->mFaces[i / 3].mNumIndices = 3;

        m->mFaces[i / 3].mIndices[0] = geo->get_index(i + 0);
        m->mFaces[i / 3].mIndices[1] = geo->get_index(i + 1);
        m->mFaces[i / 3].mIndices[2] = geo->get_index(i + 2);
      }
    }

    aiMaterial* material = new aiMaterial();
    if (mat.initialized() && export_material) {
      // int values
      if (mat->int_map.count("twosided") > 0) {
        int* val = new int();
        val[0] = (mat->int_map["twosided"]);
        material->AddProperty(val, 1, AI_MATKEY_TWOSIDED);
      }
      if (mat->int_map.count("blend_mode") > 0) {
        int* val = new int();
        val[0] = (mat->int_map["blend_mode"]);
        material->AddProperty(val, 1, AI_MATKEY_BLEND_FUNC);
      }

      // float values
      if (mat->float_map.count("opacity") > 0) {
        float* val = new float();
        val[0] = (mat->float_map["opacity"]);
        material->AddProperty(val, 1, AI_MATKEY_OPACITY);
      }
      if (mat->float_map.count("roughness") > 0) {
        float* val = new float();
        val[0] = (mat->float_map["roughness"] * 2.f) - 2.f;
        material->AddProperty(val, 1, AI_MATKEY_SHININESS);
      }
      if (mat->float_map.count("ior") > 0) {
        float* val = new float();
        val[0] = (mat->float_map["ior"]);
        material->AddProperty(val, 1, AI_MATKEY_REFRACTI);
      }

      // vec3 values
      if (mat->vec3_map.count("ambient_color") > 0) {
        aiColor3D* val = new aiColor3D();
        val[0].r = mat->vec3_map["ambient_color"].x();
        val[0].g = mat->vec3_map["ambient_color"].y();
        val[0].b = mat->vec3_map["ambient_color"].z();
        material->AddProperty(val, 1, AI_MATKEY_COLOR_AMBIENT);
      }
      if (mat->vec3_map.count("diffuse_color") > 0) {
        aiColor3D* val = new aiColor3D();
        val[0].r = mat->vec3_map["diffuse_color"].x();
        val[0].g = mat->vec3_map["diffuse_color"].y();
        val[0].b = mat->vec3_map["diffuse_color"].z();
        material->AddProperty(val, 1, AI_MATKEY_COLOR_DIFFUSE);
      }
      if (mat->vec3_map.count("specular_color") > 0) {
        aiColor3D* val = new aiColor3D();
        val[0].r = mat->vec3_map["specular_color"].x();
        val[0].g = mat->vec3_map["specular_color"].y();
        val[0].b = mat->vec3_map["specular_color"].z();
        material->AddProperty(val, 1, AI_MATKEY_COLOR_SPECULAR);
      }

      // textures

      unsigned int diffuse_counter = 0;
      unsigned int specular_counter = 0;
      unsigned int ambient_counter = 0;
      unsigned int emissive_counter = 0;
      unsigned int heightmap_counter = 0;
      unsigned int alphamap_counter = 0;
      unsigned int roughness_counter = 0;
      unsigned int displacement_counter = 0;
      unsigned int lightmap_counter = 0;
      for (auto tex : mat->texture_map) {
        if (!tex.second->loaded_from_path.empty()) {
          fs::path tex_path =
              fs::path("./") /
              subtract_paths(texture_base_path, tex.second->loaded_from_path);
          std::string* val = new std::string(tex_path.string());

          if (tex.first.compare("diffuse") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, diffuse_counter));
            diffuse_counter++;
          }

          if (tex.first.compare("specular") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_SPECULAR, specular_counter));
            specular_counter++;
          }

          if (tex.first.compare("ambient") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_AMBIENT, ambient_counter));
            ambient_counter++;
          }

          if (tex.first.compare("emissive") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_EMISSIVE, emissive_counter));
            emissive_counter++;
          }

          if (tex.first.compare("normalmap") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_HEIGHT, heightmap_counter));
            heightmap_counter++;
          }

          if (tex.first.compare("alphamap") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_OPACITY, alphamap_counter));
            alphamap_counter++;
          }

          if (tex.first.compare("roughness") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_SHININESS, roughness_counter));
            roughness_counter++;
          }

          if (tex.first.compare("displacement") == 0) {
            material->AddProperty(val, 1,
                                  AI_MATKEY_TEXTURE(aiTextureType_DISPLACEMENT,
                                                    displacement_counter));
            displacement_counter++;
          }

          if (tex.first.compare("lightmap") == 0) {
            material->AddProperty(
                val, 1,
                AI_MATKEY_TEXTURE(aiTextureType_LIGHTMAP, lightmap_counter));
            lightmap_counter++;
          }
        }
      }
      m->mMaterialIndex = mesh_counter + 1;
    }

    scene->mMeshes[mesh_counter] = m;
    scene->mMaterials[mesh_counter + 1] = material;

    mesh_counter++;
  }

  Assimp::Exporter exporter;
  exporter.Export(scene.get(), "obj", path.string().c_str());
  std::cout << "Saved mesh to " << path << std::endl;
}

void reclib::opengl::save_meshes_cpu(const std::vector<Mesh>& meshes,
                                     const fs::path& path,
                                     const fs::path& texture_base_path,
                                     bool export_material) {
  std::vector<std::pair<GeometryBase, Material>> meshes_split;
  for (unsigned int i = 0; i < meshes.size(); i++) {
    meshes_split.push_back(
        std::make_pair(meshes[i]->geometry, meshes[i]->material));
  }
  save_meshes_cpu(meshes_split, path, texture_base_path, export_material);
}

void reclib::opengl::save_mesh_cpu(const Mesh& mesh, const fs::path& path,
                                   const fs::path& texture_base_path,
                                   bool export_material) {
  std::vector<std::pair<GeometryBase, Material>> meshes_split;

  meshes_split.push_back(std::make_pair(mesh->geometry, mesh->material));
  save_meshes_cpu(meshes_split, path, texture_base_path, export_material);
}

std::vector<reclib::opengl::Mesh> reclib::opengl::load_meshes_gpu(
    const fs::path& path, bool normalize) {
  // build meshes from cpu data
  std::vector<Mesh> meshes;
  for (const auto& item : load_meshes_cpu(path, normalize))
    meshes.push_back(Mesh(item.first->name + "/" + item.second->name,
                          item.first, item.second));
  return meshes;
}
