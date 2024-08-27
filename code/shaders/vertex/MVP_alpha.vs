#version 330
layout(location = 0) in vec3 in_pos;
layout(location = 1) in float in_alpha;

uniform mat4 model;
uniform mat4 model_normal;
uniform mat4 view_normal;
uniform mat4 view;
uniform mat4 proj;

out float pv_alpha;

void main() {
  pv_alpha = in_alpha;
  gl_Position = proj * view * model * vec4(in_pos, 1.0);
}
