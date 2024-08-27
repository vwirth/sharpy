#version 330
in vec3 pv_color;

layout(location = 0) out vec4 out_col;

void main() {
  out_col = vec4(pv_color.x, pv_color.y, pv_color.z, 1);
  // out_col = vec4(1, 0, 0, 1);
}
