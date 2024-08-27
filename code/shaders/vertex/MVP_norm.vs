#version 330
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_norm;

uniform mat4 model;
uniform mat4 model_normal;
uniform mat4 view_normal;
uniform mat4 view;
uniform mat4 proj;

out vec4 pos_world;
out vec3 norm_world;
out vec3 norm_cam;

void main() {
    pos_world = model * vec4(in_pos, 1.0);
    norm_cam = normalize(mat3(view_normal) * mat3(model_normal) * in_norm);
    norm_world = normalize(mat3(model_normal)*in_norm);
    gl_Position = proj * view  * model * vec4(in_pos, 1.0);
}
