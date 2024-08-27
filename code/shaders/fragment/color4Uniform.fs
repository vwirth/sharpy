#version 330


layout (location = 0) out vec4 out_col;

uniform vec4 color;

void main() {
    out_col = color;
    // out_col = vec4(0,1,0,1);
}
