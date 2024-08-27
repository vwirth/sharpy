#version 330
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 2) in vec4 in_color;

uniform mat4 model;
uniform mat4 model_normal;
uniform mat4 view_normal;
uniform mat4 view;
uniform mat4 proj;

out vec4 pv_color;
out vec3 norm_cam;
out vec3 norm_world;
out vec3 pos_world;

void main() {
  pv_color = in_color;

  norm_cam = normalize(mat3(view_normal) * mat3(model_normal) * in_norm);
  norm_world = normalize(mat3(model_normal) * in_norm);
  pos_world = (model * vec4(in_pos.x, in_pos.y, in_pos.z, 1)).xyz;

  gl_Position = proj * view * model * vec4(in_pos, 1.0);
}
