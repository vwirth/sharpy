#version 330
in float pv_alpha;

layout(location = 0) out vec4 out_col;

uniform vec3 color;

void main() {
  out_col = vec4(color.x, color.y, color.z, pv_alpha);
  // out_col = vec4(0,1,0,1);
}
