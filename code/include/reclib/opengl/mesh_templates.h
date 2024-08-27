#ifndef RECLIB_OPENGL_MESH_TEMPLATES_H
#define RECLIB_OPENGL_MESH_TEMPLATES_H

#pragma once

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on

#include <vector>

#include "reclib/data_types.h"
#include "reclib/opengl/geometry.h"
#include "reclib/opengl/mesh.h"

// ----------------------------
// Definition
namespace reclib {

namespace opengl {

// ------------------------------------------
// LineImpl
class _API LineImpl : public MeshImpl {
 public:
  vec3 from;
  vec3 to;

  static Geometry generate_geometry(const std::string& name, const vec3& from,
                                    const vec3& to);

  LineImpl(const std::string& name, const vec3& from, const vec3& to,
           const Material& material = Material());
  virtual ~LineImpl();

  static inline std::string type_to_str() { return "LineImpl"; }
};
using Line = NamedHandle<LineImpl>;

// ------------------------------------------
// CuboidImpl
class _API CuboidImpl : public MeshImpl {
 public:
  float width;
  float height;
  float depth;
  vec3 center;

  // width = x-axis length
  // height = y-axis length
  // depth = z-axis length
  static Geometry generate_geometry(const std::string& name, float width,
                                    float height, float depth,
                                    const vec3& center = vec3(0, 0, 0));

  CuboidImpl(const std::string& name, float width, float height, float depth,
             const vec3& center = vec3(0, 0, 0),
             const Material& material = Material());
  virtual ~CuboidImpl();

  static inline std::string type_to_str() { return "CuboidImpl"; }
};
using Cuboid = NamedHandle<CuboidImpl>;

// ------------------------------------------
// VoxelGridImpl
class _API VoxelGridImpl : public MeshImpl {
 public:
  unsigned int size;
  float scale;
  vec3 center;

  // width = x-axis length
  // height = y-axis length
  // depth = z-axis length
  static Geometry generate_geometry(const std::string& name, unsigned int size,
                                    float scale);

  VoxelGridImpl(const std::string& name, unsigned int size, float scale,
                const Material& material = Material());
  virtual ~VoxelGridImpl();

  static inline std::string type_to_str() { return "VoxelGridImpl"; }
};
using VoxelGrid = NamedHandle<VoxelGridImpl>;

// ------------------------------------------
// VoxelGridXYZImpl
class _API VoxelGridXYZImpl : public MeshImpl {
 public:
  vec3 size;
  float scale;
  vec3 center;

  // width = x-axis length
  // height = y-axis length
  // depth = z-axis length
  static Geometry generate_geometry(const std::string& name, const vec3& size,
                                    float scale);

  VoxelGridXYZImpl(const std::string& name, const vec3& size, float scale,
                   const Material& material = Material());
  virtual ~VoxelGridXYZImpl();

  static inline std::string type_to_str() { return "VoxelGridXYZImpl"; }
};
using VoxelGridXYZ = NamedHandle<VoxelGridXYZImpl>;

// ------------------------------------------
// SphereImpl

class _API SphereImpl : public MeshImpl {
 public:
  float radius;
  vec3 center;
  float stack_count;
  float sector_count;

  static Geometry generate_geometry(const std::string& name, float radius,
                                    const vec3& center, float sector_count = 20,
                                    float stack_count = 20);

  SphereImpl(const std::string& name, float radius, const vec3& center,
             const Material& material = Material(), float sector_count = 20,
             float stack_count = 20);
  virtual ~SphereImpl();

  static inline std::string type_to_str() { return "SphereImpl"; }
};
using Sphere = NamedHandle<SphereImpl>;

// ------------------------------------------
// EllipsoidImpl

class _API EllipsoidImpl : public MeshImpl {
 public:
  vec3 pc1;
  vec3 pc2;
  vec3 pc3;
  vec3 center;
  float stack_count;
  float sector_count;

  static Geometry generate_geometry(const std::string& name, const vec3& pc1,
                                    const vec3& pc2, const vec3& pc3,
                                    const vec3& center, float sector_count = 20,
                                    float stack_count = 20);

  EllipsoidImpl(const std::string& name, const vec3& pc1, const vec3& pc2,
                const vec3& pc3, const vec3& center,
                const Material& material = Material(), float sector_count = 20,
                float stack_count = 20);
  virtual ~EllipsoidImpl();

  static inline std::string type_to_str() { return "EllipsoidImpl"; }
};
using Ellipsoid = NamedHandle<EllipsoidImpl>;

// ------------------------------------------
//  CylinderImpl

class _API CylinderImpl : public MeshImpl {
 public:
  float radius;
  float height;
  vec3 axis;
  vec3 center;
  float stack_count;
  float sector_count;

  static Geometry generate_geometry(const std::string& name, const float radius,
                                    const float height, const vec3& axis,
                                    const vec3& center, float sector_count = 20,
                                    float stack_count = 20);

  CylinderImpl(const std::string& name, const float radius, const float height,
               const vec3& axis, const vec3& center,
               const Material& material = Material(), float sector_count = 20,
               float stack_count = 20);
  virtual ~CylinderImpl();

  static inline std::string type_to_str() { return "CylinderImpl"; }
};
using Cylinder = NamedHandle<CylinderImpl>;

// ------------------------------------------
// FrustumImpl

class _API FrustumImpl : public MeshImpl {
 public:
  static Geometry generate_geometry(const std::string& name, const vec3& pos,
                                    const mat4& view, const mat4& proj);

  FrustumImpl(const std::string& name, const vec3& pos, const mat4& view,
              const mat4& proj, const Material& material = Material());
  virtual ~FrustumImpl();

  static inline std::string type_to_str() { return "FrustumImpl"; }
};
using Frustum = NamedHandle<FrustumImpl>;

// ------------------------------------------
// BoundingBoxImpl

class _API BoundingBoxImpl : public MeshImpl {
 public:
  static Geometry generate_geometry(const std::string& name, const vec3& bb_min,
                                    const vec3& bb_max);

  BoundingBoxImpl(const std::string& name, const vec3& bb_min,
                  const vec3& bb_max, const Material& material = Material());
  virtual ~BoundingBoxImpl();

  static inline std::string type_to_str() { return "BoundingBoxImpl"; }
};
using BoundingBox = NamedHandle<BoundingBoxImpl>;
}  // namespace opengl

}  // namespace reclib
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::SphereImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::CuboidImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::FrustumImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::LineImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::VoxelGridImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::EllipsoidImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::CylinderImpl>;  // needed for Windows DLL export
template class _API reclib::opengl::NamedHandle<
    reclib::opengl::BoundingBoxImpl>;  // needed for Windows DLL export

#endif