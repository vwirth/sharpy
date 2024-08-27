#version 330
in float pv_alpha;
in vec3 pv_color;

layout(location = 0) out vec4 out_col;

void main() {
  out_col = vec4(pv_color.x, pv_color.y, pv_color.z, pv_alpha);
  // out_col = vec4(0,1,0,1);
}
