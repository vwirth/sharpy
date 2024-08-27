#include <reclib/opengl/mesh_templates.h>

namespace reclib {
// ------------------------------------------
// LineImpl

reclib::opengl::Geometry reclib::opengl::LineImpl::generate_geometry(
    const std::string& name, const vec3& from, const vec3& to) {
  std::vector<vec3> positions;
  std::vector<uint32_t> indices;

  positions.push_back(from);
  positions.push_back(to);

  indices.push_back(0);
  indices.push_back(1);

  return Geometry(name, positions, indices);
}

reclib::opengl::LineImpl::LineImpl(const std::string& name, const vec3& from,
                                   const vec3& to, const Material& material)
    : MeshImpl(name,
               reclib::opengl::LineImpl::generate_geometry("geometry_" + name,
                                                           from, to),
               material),
      from(from),
      to(to) {
  primitive_type = GL_LINES;
}

reclib::opengl::LineImpl::~LineImpl() {}

// ------------------------------------------
// CuboidImpl

void add_quad(const vec3& bbmin, const vec3& axis1, const vec3& axis2,
              std::vector<vec3>& positions, std::vector<vec3>& normals,
              std::vector<uint32_t>& indices) {
  unsigned int start_index = positions.size();
  vec3 a1(0, 0, 0);
  vec3 a2(0, 0, 0);

  vec3 axis_mask = sign(abs((vec3)(axis1 - axis2)));

  vec3 n =
      (cross(bbmin - axis1, bbmin + axis2).cwiseProduct(axis_mask)).cwiseSign();
  if ((n.x() + n.y() + n.z()) > 0) {
    a1 = axis2;
    a2 = axis1;
  } else {
    a1 = axis1;
    a2 = axis2;
  }

  positions.push_back(bbmin);
  positions.push_back(bbmin + a1);
  positions.push_back(bbmin + a1 + a2);

  indices.push_back(indices.size());
  indices.push_back(indices.size());
  indices.push_back(indices.size());

  positions.push_back(bbmin);
  positions.push_back(bbmin + a1 + a2);
  positions.push_back(bbmin + a2);

  indices.push_back(indices.size());
  indices.push_back(indices.size());
  indices.push_back(indices.size());

  vec3 d1 = positions[start_index + 1] - positions[start_index];
  vec3 d2 = positions[start_index + 5] - positions[start_index];

  vec3 normal = cross(d1.normalized(), d2.normalized());
  normals.push_back(normal);
  normals.push_back(normal);
  normals.push_back(normal);
  normals.push_back(normal);
  normals.push_back(normal);
  normals.push_back(normal);
}

// CuboidImpl
reclib::opengl::Geometry reclib::opengl::CuboidImpl::generate_geometry(
    const std::string& name, float width, float height, float depth,
    const vec3& center) {
  float half_x = width / 2;
  float half_y = height / 2;
  float half_z = depth / 2;

  std::vector<vec3> positions;
  std::vector<vec3> normals;
  std::vector<uint32_t> indices;

  // label-strategy: x first, then y, then z
  // front side
  add_quad(vec3(center.x() - half_x, center.y() - half_y, center.z() + half_z),
           vec3(width, 0, 0), vec3(0, height, 0), positions, normals, indices);

  // back side
  add_quad(vec3(center.x() - half_x, center.y() - half_y, center.z() - half_z),
           vec3(0, height, 0), vec3(width, 0, 0), positions, normals, indices);

  // left side
  add_quad(vec3(center.x() - half_x, center.y() - half_y, center.z() - half_z),
           vec3(0, 0, depth), vec3(0, height, 0), positions, normals, indices);

  // right side
  add_quad(vec3(center.x() + half_x, center.y() - half_y, center.z() - half_z),
           vec3(0, height, 0), vec3(0, 0, depth), positions, normals, indices);

  // top side
  add_quad(vec3(center.x() - half_x, center.y() + half_y, center.z() - half_z),
           vec3(0, 0, depth), vec3(width, 0, 0), positions, normals, indices);

  // bottom side
  add_quad(vec3(center.x() - half_x, center.y() - half_y, center.z() - half_z),
           vec3(width, 0, 0), vec3(0, 0, depth), positions, normals, indices);

  return Geometry(name, positions, indices, normals);
}

reclib::opengl::CuboidImpl::CuboidImpl(const std::string& name, float width,
                                       float height, float depth,
                                       const vec3& center,
                                       const reclib::opengl::Material& material)
    : MeshImpl(name,
               reclib::opengl::CuboidImpl::generate_geometry(
                   "geometry_" + name, width, height, depth, center),
               material),
      width(width),
      height(height),
      depth(depth),
      center(center) {}

reclib::opengl::CuboidImpl::~CuboidImpl() {}

// ------------------------------------------
// SphereImpl

reclib::opengl::Geometry reclib::opengl::SphereImpl::generate_geometry(
    const std::string& name, float radius, const vec3& center,
    float sector_count, float stack_count) {
  const float PI = 3.141592f;

  float lengthInv = 1.0f / radius;

  float sectorStep = 2 * PI / sector_count;
  float stackStep = PI / stack_count;
  float sectorAngle, stackAngle;

  std::vector<vec3> positions;
  std::vector<vec3> normals;
  std::vector<vec2> texcoords;
  std::vector<uint32_t> indices;

  for (int i = 0; i <= stack_count; ++i) {
    stackAngle = PI / 2 - i * stackStep;   // starting from pi/2 to -pi/2
    float xy = radius * cosf(stackAngle);  // r * cos(u)
    float z = radius * sinf(stackAngle);   // r * sin(u)

    // add (sector_count+1) vertices per stack
    // the first and last vertices have same position and normal, but different
    // tex coords
    for (int j = 0; j <= sector_count; ++j) {
      sectorAngle = j * sectorStep;  // starting from 0 to 2pi

      // vertex position
      float x = xy * cosf(sectorAngle);  // r * cos(u) * cos(v)
      float y = xy * sinf(sectorAngle);  // r * cos(u) * sin(v)
      positions.push_back(
          vec3(vec3(x + center.x(), y + center.y(), z + center.z())));

      // normalized vertex normal
      float nx = x * lengthInv;
      float ny = y * lengthInv;
      float nz = z * lengthInv;
      normals.push_back(vec3(nx, ny, nz));

      // vertex tex coord between [0, 1]
      float s = (float)j / sector_count;
      float t = (float)i / stack_count;
      texcoords.push_back(vec2(s, t));
    }
  }

  // indices
  //  k1--k1+1
  //  |  / |
  //  | /  |
  //  k2--k2+1
  unsigned int k1, k2;
  for (int i = 0; i < stack_count; ++i) {
    k1 = i * (sector_count + 1);  // beginning of current stack
    k2 = k1 + sector_count + 1;   // beginning of next stack

    for (int j = 0; j < sector_count; ++j, ++k1, ++k2) {
      // 2 triangles per sector excluding 1st and last stacks
      if (i != 0) {
        indices.push_back(k1);
        indices.push_back(k2);
        indices.push_back(k1 + 1);
      }

      if (i != (stack_count - 1)) {
        indices.push_back(k1 + 1);
        indices.push_back(k2);
        indices.push_back(k2 + 1);
      }
    }
  }
  reclib::opengl::Geometry geo(name, positions, indices, normals, texcoords);
  return geo;
}

reclib::opengl::SphereImpl::SphereImpl(const std::string& name, float radius,
                                       const vec3& center,
                                       const reclib::opengl::Material& material,
                                       float sector_count, float stack_count)
    : MeshImpl(
          name,
          reclib::opengl::SphereImpl::generate_geometry(
              "geometry_" + name, radius, center, sector_count, stack_count),
          material),
      radius(radius),
      center(center),
      stack_count(stack_count),
      sector_count(sector_count) {}

reclib::opengl::SphereImpl::~SphereImpl() {}

// ------------------------------------------
// EllipsoidImpl

reclib::opengl::Geometry reclib::opengl::EllipsoidImpl::generate_geometry(
    const std::string& name, const vec3& pc1, const vec3& pc2, const vec3& pc3,
    const vec3& center, float sector_count, float stack_count) {
  const float PI = 3.141592f;

  float tStep = (PI) / (float)sector_count;
  float sStep = (PI) / (float)stack_count;

  std::vector<vec3> positions;
  std::vector<uint32_t> indices;
  std::vector<vec3> normals;

  for (float t = -PI / 2; t <= (PI / 2) + .0001; t += tStep) {
    for (float s = -PI; s <= PI + .0001; s += sStep) {
      vec3 cur =
          center + pc1 * cos(t) * cos(s) + pc2 * cos(t) * sin(s) + pc3 * sin(t);
      vec3 next = center + pc1 * cos(t + tStep) * cos(s) +
                  pc2 * cos(t + tStep) * sin(s) + pc3 * sin(t + tStep);

      vec3 cur_n = pc2.cross(pc3) * cos(t) * cos(s) +
                   pc3.cross(pc1) * cos(t) * sin(s) + pc1.cross(pc2) * sin(t);
      vec3 next_n = pc2.cross(pc3) * cos(t + tStep) * cos(s) +
                    pc3.cross(pc1) * cos(t + tStep) * sin(s) +
                    pc1.cross(pc2) * sin(t + tStep);

      positions.push_back(cur);
      positions.push_back(next);
      normals.push_back(cur_n);
      normals.push_back(next_n);
      indices.push_back(indices.size());
      indices.push_back(indices.size());
    }
  }

  reclib::opengl::Geometry geo(name, positions, indices, normals);
  return geo;
}

reclib::opengl::EllipsoidImpl::EllipsoidImpl(
    const std::string& name, const vec3& pc1, const vec3& pc2, const vec3& pc3,
    const vec3& center, const reclib::opengl::Material& material,
    float sector_count, float stack_count)
    : MeshImpl(name,
               reclib::opengl::EllipsoidImpl::generate_geometry(
                   "geometry_" + name, pc1, pc2, pc3, center, sector_count,
                   stack_count),
               material),
      pc1(pc1),
      pc2(pc2),
      pc3(pc3),
      center(center),
      stack_count(stack_count),
      sector_count(sector_count) {
  primitive_type = GL_TRIANGLE_STRIP;
}

reclib::opengl::EllipsoidImpl::~EllipsoidImpl() {}

// ------------------------------------------
// CylinderImpl
// adapted from: http://www.songho.ca/opengl/gl_cylinder.html

///////////////////////////////////////////////////////////////////////////////
// generate 3D vertices of a unit circle on XY plance
///////////////////////////////////////////////////////////////////////////////
std::vector<float> buildUnitCircleVertices(float sector_count) {
  const float PI = acos(-1);
  float sectorStep = 2 * PI / sector_count;
  float sectorAngle;  // radian

  std::vector<float> unitCircleVertices;
  for (int i = 0; i <= sector_count; ++i) {
    sectorAngle = i * sectorStep;
    unitCircleVertices.push_back(cos(sectorAngle));  // x
    unitCircleVertices.push_back(sin(sectorAngle));  // y
    unitCircleVertices.push_back(0);                 // z
  }
  return unitCircleVertices;
}

///////////////////////////////////////////////////////////////////////////////
// generate shared normal vectors of the side of cylinder
///////////////////////////////////////////////////////////////////////////////
std::vector<float> getSideNormals(float height, float sector_count) {
  const float PI = acos(-1);
  float sectorStep = 2 * PI / sector_count;
  float sectorAngle;  // radian

  // compute the normal vector at 0 degree first
  // tanA = (baseRadius-topRadius) / height
  float zAngle = atan2(0, height);
  float x0 = cos(zAngle);  // nx
  float y0 = 0;            // ny
  float z0 = sin(zAngle);  // nz

  // rotate (x0,y0,z0) per sector angle
  std::vector<float> normals;
  for (int i = 0; i <= sector_count; ++i) {
    sectorAngle = i * sectorStep;
    normals.push_back(cos(sectorAngle) * x0 - sin(sectorAngle) * y0);  // nx
    normals.push_back(sin(sectorAngle) * x0 + cos(sectorAngle) * y0);  // ny
    normals.push_back(z0);                                             // nz
                                                                       /*
                                                                       //debug
                                                                       float nx = cos(sectorAngle)*x0 - sin(sectorAngle)*y0;
                                                                       float ny = sin(sectorAngle)*x0 + cos(sectorAngle)*y0;
                                                                       std::cout << "normal=(" << nx << ", " << ny << ", " << z0
                                                                                 << "), length=" << sqrtf(nx*nx + ny*ny + z0*z0) << std::endl;
                                                                       */
  }

  return normals;
}

///////////////////////////////////////////////////////////////////////////////
// build vertices of cylinder with smooth shading
// where v: sector angle (0 <= v <= 360)
///////////////////////////////////////////////////////////////////////////////
void buildVerticesSmooth(float radius, float height, const vec3& axis,
                         const vec3& center, float stack_count,
                         float sector_count, std::vector<vec3>& positions,
                         std::vector<uint32_t>& indices,
                         std::vector<vec3>& normals,
                         std::vector<vec2>& texcoords) {
  vec3 radius_axis1 = -axis.cross(axis + vec3(0.1, 0.2, 0.5)).normalized();
  vec3 radius_axis2 = -(radius_axis1.cross(axis));

  float x, y, z;  // vertex position
  // float s, t;                                     // texCoord

  // get normals for cylinder sides
  std::vector<float> sideNormals = getSideNormals(height, sector_count);

  std::vector<float> unitCircleVertices = buildUnitCircleVertices(sector_count);

  // put vertices of side cylinder to array by scaling unit circle
  for (int i = 0; i <= stack_count; ++i) {
    z = -(height * 0.5f) +
        (float)i / stack_count * height;      // vertex position z // lerp
    float t = 1.0f - (float)i / stack_count;  // top-to-bottom

    for (int j = 0, k = 0; j <= sector_count; ++j, k += 3) {
      x = unitCircleVertices[k];
      y = unitCircleVertices[k + 1];

      // positions.push_back(center + vec3(x * radius, y * radius, z));

      positions.push_back(center + x * radius * radius_axis1 +
                          y * radius * radius_axis2 + z * axis);
      normals.push_back(
          vec3(sideNormals[k], sideNormals[k + 1], sideNormals[k + 2]));
      texcoords.push_back(vec2(j / sector_count, t));
    }
  }

  // remember where the base.top vertices start
  unsigned int baseVertexIndex = (unsigned int)positions.size();

  // put vertices of base of cylinder
  z = -height * 0.5f;
  positions.push_back(center + z * axis);
  // positions.push_back(center + vec3(0, 0, z));
  normals.push_back(vec3(0, 0, -1));
  texcoords.push_back(vec2(0.5, 0.5));

  for (int i = 0, j = 0; i < sector_count; ++i, j += 3) {
    x = unitCircleVertices[j];
    y = unitCircleVertices[j + 1];

    positions.push_back(center + x * radius * radius_axis1 +
                        y * radius * radius_axis2 + z * axis);
    // positions.push_back(center + vec3(x * radius, y * radius, z));
    normals.push_back(vec3(0, 0, -1));
    texcoords.push_back(
        vec2(-x * 0.5f + 0.5f, -y * 0.5f + 0.5f));  // flip horizontal
  }

  // remember where the base vertices start
  unsigned int topVertexIndex = (unsigned int)positions.size();

  // put vertices of top of cylinder
  z = height * 0.5f;
  positions.push_back(center + z * axis);
  // positions.push_back(center + vec3(0, 0, z));
  normals.push_back(vec3(0, 0, 1));
  texcoords.push_back(vec2(0.5, 0.5));

  for (int i = 0, j = 0; i < sector_count; ++i, j += 3) {
    x = unitCircleVertices[j];
    y = unitCircleVertices[j + 1];

    positions.push_back(center + x * radius * radius_axis1 +
                        y * radius * radius_axis2 + z * axis);
    // positions.push_back(center + vec3(x * radius, y * radius, z));
    normals.push_back(vec3(0, 0, 1));
    texcoords.push_back(
        vec2(x * 0.5f + 0.5f, -y * 0.5f + 0.5f));  // flip horizontal
  }

  // put indices for sides
  unsigned int k1, k2;
  for (int i = 0; i < stack_count; ++i) {
    k1 = i * (sector_count + 1);  // bebinning of current stack
    k2 = k1 + sector_count + 1;   // beginning of next stack

    for (int j = 0; j < sector_count; ++j, ++k1, ++k2) {
      // 2 trianles per sector
      indices.push_back(k1);
      indices.push_back(k1 + 1);
      indices.push_back(k2);
      indices.push_back(k2);
      indices.push_back(k1 + 1);
      indices.push_back(k2 + 1);
    }
  }

  // remember where the base indices start
  unsigned int baseIndex = (unsigned int)indices.size();

  // put indices for base
  for (int i = 0, k = baseVertexIndex + 1; i < sector_count; ++i, ++k) {
    if (i < (sector_count - 1)) {
      indices.push_back(baseVertexIndex);
      indices.push_back(k + 1);
      indices.push_back(k);
    } else  // last triangle
    {
      indices.push_back(baseVertexIndex);
      indices.push_back(baseVertexIndex + 1);
      indices.push_back(k);
    }
  }

  // remember where the base indices start
  unsigned int topIndex = (unsigned int)indices.size();

  for (int i = 0, k = topVertexIndex + 1; i < sector_count; ++i, ++k) {
    if (i < (sector_count - 1)) {
      indices.push_back(topVertexIndex);
      indices.push_back(k);
      indices.push_back(k + 1);
    } else {
      indices.push_back(topVertexIndex);
      indices.push_back(k);
      indices.push_back(topVertexIndex + 1);
    }
  }
}

reclib::opengl::Geometry reclib::opengl::CylinderImpl::generate_geometry(
    const std::string& name, const float radius, const float height,
    const vec3& axis, const vec3& center, float sector_count,
    float stack_count) {
  const float PI = 3.141592f;

  float tStep = (PI) / (float)sector_count;
  float sStep = (PI) / (float)stack_count;

  std::vector<vec3> positions;
  std::vector<uint32_t> indices;
  std::vector<vec3> normals;
  std::vector<vec2> texcoords;
  buildVerticesSmooth(radius, height, axis, center, sector_count, stack_count,
                      positions, indices, normals, texcoords);

  reclib::opengl::Geometry geo(name, positions, indices, normals, texcoords);
  return geo;
}

reclib::opengl::CylinderImpl::CylinderImpl(
    const std::string& name, const float radius, const float height,
    const vec3& axis, const vec3& center,
    const reclib::opengl::Material& material, float sector_count,
    float stack_count)
    : MeshImpl(
          name,
          reclib::opengl::CylinderImpl::generate_geometry(
              "geometry_" + name, radius, height, axis, center, stack_count),
          material),
      radius(radius),
      height(height),
      axis(axis),
      center(center),
      stack_count(stack_count),
      sector_count(sector_count) {
  primitive_type = GL_TRIANGLES;
}

reclib::opengl::CylinderImpl::~CylinderImpl() {}

// ------------------------------------------
// FrustumImpl

reclib::opengl::Geometry reclib::opengl::FrustumImpl::generate_geometry(
    const std::string& name, const vec3& pos, const mat4& view,
    const mat4& proj) {
  std::vector<vec3> positions;
  std::vector<uint32_t> indices;

  mat4 ndcToWorld = inverse(proj * view);

  // construct camera frustum
  vec4 x1 = ndcToWorld * vec4(-1, -1, -1, 1);
  vec4 x2 = ndcToWorld * vec4(1, -1, -1, 1);
  vec4 x3 = ndcToWorld * vec4(1, 1, -1, 1);
  vec4 x4 = ndcToWorld * vec4(-1, 1, -1, 1);
  vec4 x5 = ndcToWorld * vec4(-1, -1, 1, 1);
  vec4 x6 = ndcToWorld * vec4(1, -1, 1, 1);
  vec4 x7 = ndcToWorld * vec4(1, 1, 1, 1);
  vec4 x8 = ndcToWorld * vec4(-1, 1, 1, 1);

  positions.push_back(vec3(x1.x() / x1.w(), x1.y() / x1.w(), x1.z() / x1.w()));
  positions.push_back(vec3(x2.x() / x2.w(), x2.y() / x2.w(), x2.z() / x2.w()));
  positions.push_back(vec3(x3.x() / x3.w(), x3.y() / x3.w(), x3.z() / x3.w()));
  positions.push_back(vec3(x4.x() / x4.w(), x4.y() / x4.w(), x4.z() / x4.w()));
  positions.push_back(vec3(x5.x() / x5.w(), x5.y() / x5.w(), x5.z() / x5.w()));
  positions.push_back(vec3(x6.x() / x6.w(), x6.y() / x6.w(), x6.z() / x6.w()));
  positions.push_back(vec3(x7.x() / x7.w(), x7.y() / x7.w(), x7.z() / x7.w()));
  positions.push_back(vec3(x8.x() / x8.w(), x8.y() / x8.w(), x8.z() / x8.w()));
  positions.push_back(pos);

  // construct lines that connect in the frustum
  indices.push_back(0);
  indices.push_back(1);

  indices.push_back(1);
  indices.push_back(2);

  indices.push_back(2);
  indices.push_back(3);

  indices.push_back(3);
  indices.push_back(0);

  indices.push_back(4);
  indices.push_back(5);

  indices.push_back(5);
  indices.push_back(6);

  indices.push_back(6);
  indices.push_back(7);

  indices.push_back(7);
  indices.push_back(4);

  indices.push_back(3);
  indices.push_back(7);

  indices.push_back(2);
  indices.push_back(6);

  indices.push_back(1);
  indices.push_back(5);

  indices.push_back(0);
  indices.push_back(4);

  indices.push_back(8);
  indices.push_back(4);

  indices.push_back(8);
  indices.push_back(5);

  indices.push_back(8);
  indices.push_back(6);

  indices.push_back(8);
  indices.push_back(7);

  return Geometry(name, positions, indices);
}

reclib::opengl::FrustumImpl::FrustumImpl(
    const std::string& name, const vec3& pos, const mat4& view,
    const mat4& proj, const reclib::opengl::Material& material)
    : MeshImpl(name,
               reclib::opengl::FrustumImpl::generate_geometry(
                   "geometry_" + name, pos, view, proj),
               material) {
  primitive_type = GL_LINES;
}

reclib::opengl::FrustumImpl::~FrustumImpl() {}

// ------------------------------------------
// VoxelGridImpl

void add_cube(const vec3& offset, float scale, std::vector<vec3>& positions,
              std::vector<uint32_t>& indices) {
  const int start_index = positions.size();

  // front bottom left
  positions.push_back(offset);
  // front bottom right
  positions.push_back(offset + vec3(scale, 0, 0));
  // front top left
  positions.push_back(offset + vec3(0, scale, 0));
  // front top right
  positions.push_back(offset + vec3(scale, scale, 0));
  // back bottom left
  positions.push_back(offset + vec3(scale, 0, scale));
  // back bottom right
  positions.push_back(offset + vec3(scale, 0, scale));
  // back top left
  positions.push_back(offset + vec3(0, scale, scale));
  // back top right
  positions.push_back(offset + vec3(scale, scale, scale));

  // edges from front bottom left
  // left -> right
  indices.push_back(start_index + 0);
  indices.push_back(start_index + 1);

  // bottom -> up
  indices.push_back(start_index + 0);
  indices.push_back(start_index + 2);

  // front -> back
  indices.push_back(start_index + 0);
  indices.push_back(start_index + 4);

  // edges from front bottom right
  // bottom -> up
  indices.push_back(start_index + 1);
  indices.push_back(start_index + 3);

  // fromt -> back
  indices.push_back(start_index + 1);
  indices.push_back(start_index + 5);

  // edges from front top left
  // left -> right
  indices.push_back(start_index + 2);
  indices.push_back(start_index + 3);

  // front -> back
  indices.push_back(start_index + 2);
  indices.push_back(start_index + 6);

  // edges from front top right
  // front -> back
  indices.push_back(start_index + 3);
  indices.push_back(start_index + 7);

  // edges from back bottom left
  // left -> right
  indices.push_back(start_index + 4);
  indices.push_back(start_index + 5);

  // bottom -> top
  indices.push_back(start_index + 4);
  indices.push_back(start_index + 6);

  // edges from back bottom right
  // bottom -> top
  indices.push_back(start_index + 5);
  indices.push_back(start_index + 7);

  // edges from back top left
  // left -> right
  indices.push_back(start_index + 6);
  indices.push_back(start_index + 7);
}

reclib::opengl::Geometry reclib::opengl::VoxelGridImpl::generate_geometry(
    const std::string& name, unsigned int size, float scale) {
  std::vector<vec3> positions;
  std::vector<uint32_t> indices;

  // for (unsigned int d = 0; d < size; d++) {
  //   for (unsigned int h = 0; h < size; h++) {
  //     for (unsigned int w = 0; w < size; w++) {
  //       add_cube(vec3(w * scale, h * scale, d * scale), scale,
  //       positions,
  //                indices);
  //     }
  //   }
  // }

  for (unsigned int d = 0; d <= size; d++) {
    for (unsigned int h = 0; h <= size; h++) {
      // push lines from left to right
      positions.push_back(vec3(0 * scale, h * scale, d * scale));
      positions.push_back(vec3((size)*scale, h * scale, d * scale));
      indices.push_back(indices.size());
      indices.push_back(indices.size());
    }
  }

  for (unsigned int d = 0; d <= size; d++) {
    for (unsigned int w = 0; w <= size; w++) {
      // push lines from top to bottom
      positions.push_back(vec3(w * scale, 0 * scale, d * scale));
      positions.push_back(vec3(w * scale, (size)*scale, d * scale));
      indices.push_back(indices.size());
      indices.push_back(indices.size());
    }
  }

  for (unsigned int h = 0; h <= size; h++) {
    for (unsigned int w = 0; w <= size; w++) {
      // push lines from front to back
      positions.push_back(vec3(w * scale, h * scale, 0 * scale));
      positions.push_back(vec3(w * scale, h * scale, (size)*scale));
      indices.push_back(indices.size());
      indices.push_back(indices.size());
    }
  }

  return Geometry(name, positions, indices);
}

reclib::opengl::VoxelGridImpl::VoxelGridImpl(
    const std::string& name, unsigned int size, float scale,
    const reclib::opengl::Material& material)
    : MeshImpl(name,
               reclib::opengl::VoxelGridImpl::generate_geometry(
                   "geometry_" + name, size, scale),
               material) {
  primitive_type = GL_LINES;
}

reclib::opengl::VoxelGridImpl::~VoxelGridImpl() {}

// ------------------------------------------
// VoxelGridXYZImpl

reclib::opengl::Geometry reclib::opengl::VoxelGridXYZImpl::generate_geometry(
    const std::string& name, const vec3& size, float scale) {
  std::vector<vec3> positions;
  std::vector<uint32_t> indices;

  for (unsigned int d = 0; d < size.z(); d++) {
    for (unsigned int h = 0; h < size.y(); h++) {
      for (unsigned int w = 0; w < size.x(); w++) {
        add_cube(vec3(w * scale, h * scale, d * scale), scale, positions,
                 indices);
      }
    }
  }

  // for (unsigned int d = 0; d <= size.z(); d++) {
  //   for (unsigned int h = 0; h <= size.y(); h++) {
  //     // push lines from left to right
  //     positions.push_back(vec3(0 * scale, h * scale, d * scale));
  //     positions.push_back(vec3((size.x()) * scale, h * scale, d * scale));
  //     indices.push_back(indices.size());
  //     indices.push_back(indices.size());
  //   }
  // }

  // for (unsigned int d = 0; d <= size.z(); d++) {
  //   for (unsigned int w = 0; w <= size.x(); w++) {
  //     // push lines from top to bottom
  //     positions.push_back(vec3(w * scale, 0 * scale, d * scale));
  //     positions.push_back(vec3(w * scale, (size.y()) * scale, d * scale));
  //     indices.push_back(indices.size());
  //     indices.push_back(indices.size());
  //   }
  // }

  // for (unsigned int h = 0; h <= size.y(); h++) {
  //   for (unsigned int w = 0; w <= size.x(); w++) {
  //     // push lines from front to back
  //     positions.push_back(vec3(w * scale, h * scale, 0 * scale));
  //     positions.push_back(vec3(w * scale, h * scale, (size.z()) * scale));
  //     indices.push_back(indices.size());
  //     indices.push_back(indices.size());
  //   }
  // }

  return Geometry(name, positions, indices);
}

reclib::opengl::VoxelGridXYZImpl::VoxelGridXYZImpl(
    const std::string& name, const vec3& size, float scale,
    const reclib::opengl::Material& material)
    : MeshImpl(name,
               reclib::opengl::VoxelGridXYZImpl::generate_geometry(
                   "geometry_" + name, size, scale),
               material) {
  primitive_type = GL_LINES;
}

reclib::opengl::VoxelGridXYZImpl::~VoxelGridXYZImpl() {}

// ------------------------------------------
// BoundingBoxImpl

reclib::opengl::Geometry reclib::opengl::BoundingBoxImpl::generate_geometry(
    const std::string& name, const vec3& bb_min, const vec3& bb_max) {
  std::vector<vec3> positions;
  std::vector<uint32_t> indices;

  vec3 whd = bb_max - bb_min;
  unsigned int start_index = 0;

  // front bottom left
  positions.push_back(bb_min);
  // front bottom right
  positions.push_back(bb_min + vec3(whd.x(), 0, 0));
  // front top left
  positions.push_back(bb_min + vec3(0, whd.y(), 0));
  // front top right
  positions.push_back(bb_min + vec3(whd.x(), whd.y(), 0));
  // back bottom left
  positions.push_back(bb_min + vec3(0, 0, whd.z()));
  // back bottom right
  positions.push_back(bb_min + vec3(whd.x(), 0, whd.z()));
  // back top left
  positions.push_back(bb_min + vec3(0, whd.y(), whd.z()));
  // back top right
  positions.push_back(bb_min + vec3(whd.x(), whd.y(), whd.z()));

  // edges from front bottom left
  // left -> right
  indices.push_back(start_index + 0);
  indices.push_back(start_index + 1);

  // bottom -> up
  indices.push_back(start_index + 0);
  indices.push_back(start_index + 2);

  // front -> back
  indices.push_back(start_index + 0);
  indices.push_back(start_index + 4);

  // edges from front bottom right
  // bottom -> up
  indices.push_back(start_index + 1);
  indices.push_back(start_index + 3);

  // fromt -> back
  indices.push_back(start_index + 1);
  indices.push_back(start_index + 5);

  // edges from front top left
  // left -> right
  indices.push_back(start_index + 2);
  indices.push_back(start_index + 3);

  // front -> back
  indices.push_back(start_index + 2);
  indices.push_back(start_index + 6);

  // edges from front top right
  // front -> back
  indices.push_back(start_index + 3);
  indices.push_back(start_index + 7);

  // edges from back bottom left
  // left -> right
  indices.push_back(start_index + 4);
  indices.push_back(start_index + 5);

  // bottom -> top
  indices.push_back(start_index + 4);
  indices.push_back(start_index + 6);

  // edges from back bottom right
  // bottom -> top
  indices.push_back(start_index + 5);
  indices.push_back(start_index + 7);

  // edges from back top left
  // left -> right
  indices.push_back(start_index + 6);
  indices.push_back(start_index + 7);
  return Geometry(name, positions, indices);
}

reclib::opengl::BoundingBoxImpl::BoundingBoxImpl(const std::string& name,
                                                 const vec3& bb_min,
                                                 const vec3& bb_max,
                                                 const Material& material)
    : MeshImpl(name,
               reclib::opengl::BoundingBoxImpl::generate_geometry(
                   "geometry_" + name, bb_min, bb_max),
               material) {
  primitive_type = GL_LINES;
}

reclib::opengl::BoundingBoxImpl::~BoundingBoxImpl() {}

}  // namespace reclib
