#include "reclib/opengl/rgbd_utils.h"

#include <reclib/optim/correspondences.h>

#include <opencv2/core/utility.hpp>

#include "reclib/opengl/rgbd_utils.h"

#if HAS_OPENCV_MODULE

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include "reclib/depth_processing.h"

CpuMat reclib::opengl::fill_depth_nearest_neighbor(CpuMat depth,
                                                   float random_noise_scale) {
  reclib::optim::PointCloud<float, 2> pc_wrapper;
  reclib::optim::KdTree<float, 2> kdtree(2, pc_wrapper, {32});
  std::vector<vec2> valid_indices;

  CpuMat filled_depth = depth.clone();

#if WITH_CUDA
  GpuMat depth_in(depth.size(), depth.type());
  depth_in.upload(depth);
  GpuMat depth_out(depth.size(), depth.type());
  depth_out.upload(depth);

  reclib::opengl::cuda::fill_depth_nearest_neighbor(depth_in, depth_out,
                                                    random_noise_scale);

  depth_out.download(filled_depth);

#else   // !WITH_CUDA
  for (unsigned int y = 0; y < depth.rows; y++) {
    for (unsigned int x = 0; x < depth.cols; x++) {
      if (depth.ptr<float>(0)[y * depth.cols + x] > 0) {
        valid_indices.push_back(vec2(x, y));
      }
    }
  }

  kdtree.buildIndex();

  for (unsigned int y = 0; y < depth.rows; y++) {
    for (unsigned int x = 0; x < depth.cols; x++) {
      if (depth.ptr<float>(0)[y * depth.cols + x] == 0) {
        std::vector<size_t> knn_indices(1);
        std::vector<float> knn_dist_l2(1);
        nanoflann::KNNResultSet<float> knn_resultSet(1);
        knn_resultSet.init(knn_indices.data(), knn_dist_l2.data());
        kdtree.findNeighbors(knn_resultSet, vec2(x, y).data(),
                             nanoflann::SearchParams());

        ivec2 neighbor_coords = valid_indices[knn_indices[0]].cast<int>();
        filled_depth.ptr<float>(0)[y * depth.cols + x] =
            depth.ptr<float>(
                0)[neighbor_coords.y() * depth.cols + neighbor_coords.x()] +
            random::sampleFloat(-random_noise_scale, random_noise_scale);
      }
    }
  }
#endif  // WITH_CUDA
  return filled_depth;
}

// generates point cloud from depth image with normal and color attribute
reclib::opengl::Mesh reclib::opengl::pointcloud_norm_color(
    const CpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    const CpuMat& rgb, bool pinhole2opengl_normals, const std::string& name,
    const CpuMat& xyz, const ivec2& xy_offset, const float scale) {
  _RECLIB_ASSERT(depth.type() == CV_32FC1);
  _RECLIB_ASSERT(rgb.depth() == CV_32F);
  unsigned int channels = rgb.channels();
  _RECLIB_ASSERT(channels == 3 || channels == 4);

  CpuMat vertex_map;
  if (xyz.cols == 0 && xyz.rows == 0) {
    vertex_map.create(depth.size(), CV_32FC3);
    reclib::ComputeVertexMap(depth, vertex_map, intrinsics.Level(0), -1,
                             xy_offset, scale);
  } else {
    vertex_map = xyz;
  }
  CpuMat normal_map(depth.size(), CV_32FC3);
  reclib::ComputeNormalMap(vertex_map, normal_map);

  std::vector<vec3> vertices;
  std::vector<vec3> normals;
  std::vector<vec3> colors;

  unsigned int counter = 0;
  for (int i = 0; i < vertex_map.rows * vertex_map.cols; i++) {
    if (depth.ptr<float>(0)[i] > 0) {
      vertices.push_back(vertex_map.ptr<vec3>(0)[i]);
      vec3 n = normal_map.ptr<vec3>(0)[i];
      if (pinhole2opengl_normals) {
        n.y() = -n.y();  // revert y-axis
        n.z() = -n.z();  // revert z-axis
      }
      normals.push_back(n);
      if (channels == 4) {
        colors.push_back(rgb.ptr<vec4>(0)[i].head<3>());
      } else {
        colors.push_back(rgb.ptr<vec3>(0)[i]);
      }
      counter++;
    }
  }

  if (!reclib::opengl::Mesh::valid(name)) {
    reclib::opengl::Mesh d = reclib::opengl::MeshImpl::from_geometry(
        name, false, vertices, std::vector<uint32_t>(), normals);
    d->geometry->add_attribute_vec3("colors", colors);
    d->primitive_type = GL_POINTS;
    d->geometry->update_meshes();
    return d;
  } else {
    reclib::opengl::Mesh d = reclib::opengl::Mesh::find(name);
    d->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        vertices[0].data(), vertices.size(), (uint32_t*)nullptr, 0,
        normals[0].data(), normals.size());
    d->geometry->add_attribute_vec3("colors", colors, false, true);
    d->geometry->update_meshes();
    return d;
  }
}

reclib::opengl::Mesh reclib::opengl::triangle_mesh_norm_color(
    const CpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    const CpuMat& rgb, bool pinhole2opengl_normals, const std::string& name,
    const CpuMat& xyz, const ivec2& xy_offset, const float scale) {
  _RECLIB_ASSERT(depth.type() == CV_32FC1);
  _RECLIB_ASSERT(rgb.depth() == CV_32F);
  unsigned int channels = rgb.channels();
  _RECLIB_ASSERT(channels == 3 || channels == 4);

  CpuMat vertex_map;
  if (xyz.cols == 0 && xyz.rows == 0) {
    vertex_map.create(depth.size(), CV_32FC3);
    reclib::ComputeVertexMap(depth, vertex_map, intrinsics.Level(0), -1,
                             xy_offset, scale);
  } else {
    vertex_map = xyz;
  }
  CpuMat normal_map(depth.size(), CV_32FC3);
  reclib::ComputeNormalMap(vertex_map, normal_map);

  std::vector<vec3> vertices;
  std::vector<vec3> normals;
  std::vector<vec3> colors;
  std::vector<uint32_t> indices;
  std::map<int, int> vindex2buffer;

  auto valid_func = [depth, vertex_map](int x, int y) {
    int index = y * vertex_map.cols + x;
    if (x < 0 || x >= depth.cols) return false;
    if (y < 0 || y >= depth.rows) return false;
    if (index < 0 || index >= depth.rows * depth.cols) return false;
    if (depth.ptr<float>(0)[index] <= 0 ||
        vertex_map.ptr<vec3>(0)[index].z() <= 0)
      return false;
    return true;
  };

  auto add_vertex = [pinhole2opengl_normals, channels, vertex_map, normal_map,
                     rgb, &vindex2buffer, &indices, &vertices, &normals,
                     &colors](int index) {
    vertices.push_back(vertex_map.ptr<vec3>(0)[index]);
    vec3 n = normal_map.ptr<vec3>(0)[index];
    if (pinhole2opengl_normals) {
      n.y() = -n.y();  // revert y-axis
      n.z() = -n.z();  // revert z-axis
    }
    normals.push_back(n);
    if (channels == 4) {
      colors.push_back(rgb.ptr<vec4>(0)[index].head<3>());
    } else {
      colors.push_back(rgb.ptr<vec3>(0)[index]);
    }
    indices.push_back(vertices.size() - 1);
    _RECLIB_ASSERT(vindex2buffer.find(index) == vindex2buffer.end());
    vindex2buffer[index] = indices[indices.size() - 1];
  };

  for (unsigned int y = 0; y < vertex_map.rows; y++) {
    for (unsigned int x = 0; x < vertex_map.cols; x++) {
      unsigned int i00 = y * vertex_map.cols + x;
      unsigned int i10 = (y + 1) * vertex_map.cols + x;
      unsigned int i01 = y * vertex_map.cols + x + 1;
      unsigned int i11 = (y + 1) * vertex_map.cols + x + 1;

      int num_vertices = valid_func(x, y) + valid_func(x + 1, y) +
                         valid_func(x + 1, y + 1) + valid_func(x, y + 1);

      if (num_vertices < 3) continue;
      // _RECLIB_ASSERT_EQ(num_vertices, 4);

      int indices_bef = indices.size();

      if (num_vertices == 4) {
        // try this configuration
        //  _
        // |/|
        // ---
        if (vindex2buffer.find(i00) == vindex2buffer.end()) {
          add_vertex(i00);
        } else {
          indices.push_back(vindex2buffer[i00]);
        }

        if (vindex2buffer.find(i10) == vindex2buffer.end()) {
          add_vertex(i10);
        } else {
          indices.push_back(vindex2buffer[i10]);
        }

        if (vindex2buffer.find(i01) == vindex2buffer.end()) {
          add_vertex(i01);
        } else {
          indices.push_back(vindex2buffer[i01]);
        }

        add_vertex(i11);
        indices.push_back(vindex2buffer[i01]);
        indices.push_back(vindex2buffer[i10]);

        // if (i00 < 100) {
        //   for (unsigned int i = 0; i < 6; i++) {
        //     std::cout << "i: " << indices[indices.size() - 6 + i] <<
        //     std::endl;
        //   }
        //   std::cout << "--------------------------------------------------"
        //             << std::endl;
        // }
      } else if (!valid_func(x, y)) {
        // try this configuration
        //
        //  /|
        // ---

        add_vertex(i11);
        if (vindex2buffer.find(i10) == vindex2buffer.end()) {
          add_vertex(i10);
        } else {
          indices.push_back(vindex2buffer[i10]);
        }

        if (vindex2buffer.find(i01) == vindex2buffer.end()) {
          add_vertex(i01);
        } else {
          indices.push_back(vindex2buffer[i01]);
        }
      } else if (!valid_func(x + 1, y)) {
        // try this configuration
        //
        // |\\
        // ---

        if (vindex2buffer.find(i00) == vindex2buffer.end()) {
          add_vertex(i00);
        } else {
          indices.push_back(vindex2buffer[i00]);
        }

        if (vindex2buffer.find(i10) == vindex2buffer.end()) {
          add_vertex(i10);
        } else {
          indices.push_back(vindex2buffer[i10]);
        }

        add_vertex(i11);

      } else if (!valid_func(x, y + 1)) {
        // try this configuration
        //  _
        //  \|
        //

        if (vindex2buffer.find(i00) == vindex2buffer.end()) {
          add_vertex(i00);
        } else {
          indices.push_back(vindex2buffer[i00]);
        }

        if (vindex2buffer.find(i01) == vindex2buffer.end()) {
          add_vertex(i01);
        } else {
          indices.push_back(vindex2buffer[i01]);
        }

        add_vertex(i11);

      } else if (!valid_func(x + 1, y + 1)) {
        // try this configuration
        //  _
        // |/
        //

        if (vindex2buffer.find(i00) == vindex2buffer.end()) {
          add_vertex(i00);
        } else {
          indices.push_back(vindex2buffer[i00]);
        }

        if (vindex2buffer.find(i10) == vindex2buffer.end()) {
          add_vertex(i10);
        } else {
          indices.push_back(vindex2buffer[i10]);
        }

        if (vindex2buffer.find(i01) == vindex2buffer.end()) {
          add_vertex(i01);
        } else {
          indices.push_back(vindex2buffer[i01]);
        }
      }

      if (num_vertices == 4) _RECLIB_ASSERT_EQ(indices.size() - indices_bef, 6);
      if (num_vertices == 3) _RECLIB_ASSERT_EQ(indices.size() - indices_bef, 3);

      // if (indices.size() > 12) break;
    }
    // if (indices.size() > 12) break;
  }

  if (!reclib::opengl::Mesh::valid(name)) {
    reclib::opengl::Mesh d = reclib::opengl::MeshImpl::from_geometry(
        name, false, vertices, indices, normals);
    d->geometry->add_attribute_vec3("colors", colors);
    d->primitive_type = GL_TRIANGLES;
    d->geometry->update_meshes();
    return d;
  } else {
    reclib::opengl::Mesh d = reclib::opengl::Mesh::find(name);
    d->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        vertices[0].data(), vertices.size(), indices.data(), indices.size(),
        normals[0].data(), normals.size());
    d->geometry->add_attribute_vec3("colors", colors, false, true);
    d->geometry->update_meshes();
    return d;
  }
}
// generates point cloud from depth image with normal attribute
reclib::opengl::Mesh reclib::opengl::pointcloud_norm(
    const CpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    bool pinhole2opengl_normals, const std::string& name, const CpuMat& xyz,
    const ivec2& xy_offset, const float scale) {
  _RECLIB_ASSERT(depth.type() == CV_32FC1);

  CpuMat vertex_map;
  if (xyz.cols == 0 && xyz.rows == 0) {
    vertex_map.create(depth.size(), CV_32FC3);
    reclib::ComputeVertexMap(depth, vertex_map, intrinsics.Level(0), -1,
                             xy_offset, scale);
  } else {
    vertex_map = xyz;
  }
  CpuMat normal_map(depth.size(), CV_32FC3);
  reclib::ComputeNormalMap(vertex_map, normal_map);

  std::vector<vec3> vertices;
  std::vector<vec3> normals;
  for (int i = 0; i < vertex_map.rows * vertex_map.cols; i++) {
    if (depth.ptr<float>(0)[i] > 0) {
      vertices.push_back(vertex_map.ptr<vec3>(0)[i]);
      vec3 n = normal_map.ptr<vec3>(0)[i];
      if (pinhole2opengl_normals) {
        n.y() = -n.y();  // revert y-axis
        n.z() = -n.z();  // revert z-axis
      }
      normals.push_back(n);
    }
  }

  if (!reclib::opengl::Mesh::valid(name)) {
    reclib::opengl::Mesh d = reclib::opengl::MeshImpl::from_geometry(
        name, false, vertices, std::vector<uint32_t>(), normals);
    d->primitive_type = GL_POINTS;
    return d;
  } else {
    reclib::opengl::Mesh d = reclib::opengl::Mesh::find(name);
    d->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        vertices[0].data(), vertices.size(), (uint32_t*)nullptr, 0,
        normals[0].data(), normals.size());
    d->geometry->update_meshes();
    return d;
  }
}

reclib::opengl::Mesh reclib::opengl::pointcloud_norm_color(
    const CpuMat& vertex_map, const CpuMat& normal_map,
    bool pinhole2opengl_normals, const std::string& name,
    const ivec2& xy_offset, const float scale) {
  std::vector<vec3> vertices;
  std::vector<vec3> normals;
  for (int i = 0; i < vertex_map.rows * vertex_map.cols; i++) {
    if (vertex_map.ptr<float>(0)[i * 3 + 2] > 0) {
      vertices.push_back(vertex_map.ptr<vec3>(0)[i]);
      vec3 n = normal_map.ptr<vec3>(0)[i];
      if (pinhole2opengl_normals) {
        n.y() = -n.y();  // revert y-axis
        n.z() = -n.z();  // revert z-axis
      }
      normals.push_back(n);
    }
  }

  if (!reclib::opengl::Mesh::valid(name)) {
    reclib::opengl::Mesh d = reclib::opengl::MeshImpl::from_geometry(
        name, false, vertices, std::vector<uint32_t>(), normals);
    d->primitive_type = GL_POINTS;
    return d;
  } else {
    reclib::opengl::Mesh d = reclib::opengl::Mesh::find(name);
    d->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        vertices[0].data(), vertices.size(), (uint32_t*)nullptr, 0,
        normals[0].data(), normals.size());
    d->geometry->update_meshes();
    return d;
  }
}

#if WITH_CUDA

// generates point cloud from depth image with normal and color attribute
reclib::opengl::Mesh reclib::opengl::cuda::pointcloud_norm_color(
    const GpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    const GpuMat& rgb, bool pinhole2opengl_normals, const std::string& name,
    const GpuMat& xyz, const ivec2& xy_offset, const float scale) {
  _RECLIB_ASSERT(depth.type() == CV_32FC1);
  _RECLIB_ASSERT(rgb.depth() == CV_32F);
  unsigned int channels = rgb.channels();
  _RECLIB_ASSERT(channels == 3 || channels == 4);

  GpuMat raw_vertex_map;
  if (xyz.cols == 0 && xyz.rows == 0) {
    raw_vertex_map = cv::cuda::createContinuous(depth.size(), CV_32FC3);
    reclib::cuda::ComputeVertexMap(depth, raw_vertex_map, intrinsics.Level(0),
                                   -1, xy_offset, scale);
  } else {
    raw_vertex_map = xyz;
  }

  GpuMat raw_normal_map = cv::cuda::createContinuous(depth.size(), CV_32FC3);
  reclib::cuda::ComputeNormalMap(raw_vertex_map, raw_normal_map);

  CpuMat vertex_map;
  CpuMat normal_map;
  CpuMat rgb_map;
  rgb_map.create(rgb.size(), rgb.type());

  vertex_map.create(raw_vertex_map.size(), raw_vertex_map.type());
  normal_map.create(raw_normal_map.size(), raw_normal_map.type());
  raw_vertex_map.download(vertex_map);
  raw_normal_map.download(normal_map);
  rgb.download(rgb_map);

  std::vector<vec3> vertices;
  std::vector<vec3> normals;
  std::vector<vec3> colors;
  for (int i = 0; i < vertex_map.rows * vertex_map.cols; i++) {
    if (depth.ptr<float>(0)[i] > 0) {
      vertices.push_back(vertex_map.ptr<vec3>(0)[i]);
      vec3 n = normal_map.ptr<vec3>(0)[i];
      if (pinhole2opengl_normals) {
        n.y() = -n.y();  // revert y-axis
        n.z() = -n.z();  // revert z-axis
      }
      normals.push_back(n);
      if (channels == 4) {
        colors.push_back(rgb.ptr<vec4>(0)[i].head<3>());
      } else {
        colors.push_back(rgb.ptr<vec3>(0)[i]);
      }
    }
  }

  if (!reclib::opengl::Mesh::valid(name)) {
    reclib::opengl::Mesh d = reclib::opengl::MeshImpl::from_geometry(
        name, false, vertices, std::vector<uint32_t>(), normals);
    d->geometry->add_attribute_vec3("colors", colors);
    d->geometry->update_meshes();
    d->primitive_type = GL_POINTS;
    return d;
  } else {
    reclib::opengl::Mesh d = reclib::opengl::Mesh::find(name);
    d->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        vertices[0].data(), vertices.size(), (uint32_t*)nullptr, 0,
        normals[0].data(), normals.size());
    d->geometry->add_attribute_vec3("colors", colors, false, true);
    d->geometry->update_meshes();
    return d;
  }
}
// generates point cloud from depth image with normal attribute
reclib::opengl::Mesh reclib::opengl::cuda::pointcloud_norm(
    const GpuMat& depth, const reclib::IntrinsicParameters& intrinsics,
    bool pinhole2opengl_normals, const std::string& name, const GpuMat& xyz,
    const ivec2& xy_offset, const float scale) {
  _RECLIB_ASSERT(depth.type() == CV_32FC1);

  GpuMat raw_vertex_map;
  if (xyz.cols == 0 && xyz.rows == 0) {
    raw_vertex_map = cv::cuda::createContinuous(depth.size(), CV_32FC3);
    reclib::cuda::ComputeVertexMap(depth, raw_vertex_map, intrinsics.Level(0),
                                   -1, xy_offset, scale);
  } else {
    raw_vertex_map = xyz;
  }

  GpuMat raw_normal_map = cv::cuda::createContinuous(depth.size(), CV_32FC3);
  reclib::cuda::ComputeNormalMap(raw_vertex_map, raw_normal_map);

  CpuMat vertex_map;
  CpuMat normal_map;

  vertex_map.create(raw_vertex_map.size(), raw_vertex_map.type());
  normal_map.create(raw_normal_map.size(), raw_normal_map.type());
  raw_vertex_map.download(vertex_map);
  raw_normal_map.download(normal_map);

  std::vector<vec3> vertices;
  std::vector<vec3> normals;
  for (int i = 0; i < vertex_map.rows * vertex_map.cols; i++) {
    if (depth.ptr<float>(0)[i] > 0) {
      vertices.push_back(vertex_map.ptr<vec3>(0)[i]);
      vec3 n = normal_map.ptr<vec3>(0)[i];
      if (pinhole2opengl_normals) {
        n.y() = -n.y();  // revert y-axis
        n.z() = -n.z();  // revert z-axis
      }
      normals.push_back(n);
    }
  }

  if (!reclib::opengl::Mesh::valid(name)) {
    reclib::opengl::Mesh d = reclib::opengl::MeshImpl::from_geometry(
        name, false, vertices, std::vector<uint32_t>(), normals);
    d->primitive_type = GL_POINTS;
    return d;
  } else {
    reclib::opengl::Mesh d = reclib::opengl::Mesh::find(name);
    d->geometry.cast<reclib::opengl::GeometryImpl>()->set(
        vertices[0].data(), vertices.size(), (uint32_t*)nullptr, 0,
        normals[0].data(), normals.size());
    d->geometry->update_meshes();
    return d;
  }
}

#endif  // WITH_CUDA

#endif  // HAS_OPENCV_MODULE