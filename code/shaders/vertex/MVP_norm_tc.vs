#version 330
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 2) in vec2 in_tc;

uniform mat4 model;
uniform mat4 view_normal;
uniform mat4 model_normal;
uniform mat4 view;
uniform mat4 proj;

out vec3 pos_world;
out vec3 norm_cam;
out vec3 norm_world;
out vec2 tc;

void main() {
  pos_world = (model * vec4(in_pos.x, in_pos.y, in_pos.z, 1)).xyz;
  norm_cam = normalize(mat3(view_normal) * mat3(model_normal) * in_norm);
  norm_world = normalize(mat3(model_normal) * in_norm);
  tc = in_tc;
  gl_Position = proj * view * vec4(in_pos.x, in_pos.y, in_pos.z, 1);
}
