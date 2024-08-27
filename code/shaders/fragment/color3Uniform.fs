#version 330

layout(location = 0) out vec4 out_col;

uniform vec3 color;

void main() {
  out_col = vec4(color.x, color.y, color.z, 1);
  // out_col = vec4(0,1,0,1);
}
