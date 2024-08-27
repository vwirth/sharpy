#version 330
in vec4 pos_world;
in vec3 norm_cam;
in vec3 norm_world;

layout(location = 0) out vec4 out_col;

void main() {
  // out_col = vec4(norm_world.x, norm_world.y, norm_world.z, 1);
  out_col = vec4(norm_world.x, norm_world.y, norm_world.z, 1);
}
