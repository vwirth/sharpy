#version 330
layout (location = 0) in vec3 in_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec4 pos_world;

void main() {
    pos_world = model * vec4(in_pos, 1.0);
    gl_Position = proj * view  * model * vec4(in_pos, 1.0);
}
