#version 330
in vec4 pos_world;
in vec3 norm_world;

in float pv_alpha;

layout(location = 0) out vec4 out_col;

uniform mat4 model;
uniform mat4 view_normal;
uniform mat4 view;
uniform mat4 proj;

uniform vec3 ambient_color;
uniform vec3 diffuse_color;
uniform vec3 specular_color;
uniform vec3 light_dir;
uniform float opacity;
uniform float shinyness;

uniform bool use_alpha;

void main() {
  float k_ambient = 1;
  float k_diffuse = 1;
  float k_specular = 1;

  vec3 ambient = k_ambient * ambient_color;
  vec3 color = ambient;
  vec3 specular;

  if (norm_world.x != 0 || norm_world.y != 0 || norm_world.z != 0) {
    vec3 view_dir = vec3(-view[2][0], -view[2][1], -view[2][2]);
    view_dir = normalize(view_dir);
    vec3 norm = normalize(norm_world);
    float n_dot_l = clamp(dot(norm, light_dir), 0, 1);
    vec3 reflection = 2 * n_dot_l * norm - light_dir;

    vec3 diffuse = k_diffuse * n_dot_l * diffuse_color;
    specular = k_specular * pow(clamp(dot(reflection, view_dir), 0, 1), 0.1) *
               specular_color;

    color = color + diffuse + specular;
  }

  if (!use_alpha) {
    out_col = vec4(color.x, color.y, color.z, 1);
  } else {
    out_col = vec4(color.x, color.y, color.z, pv_alpha);
  }
  // out_col = vec4(specular.x, specular.y, specular.z, 1);

  // out_col = vec4(1, 0, 0, 1);
  // out_col = vec4(diffuse_color.x, diffuse_color.y, diffuse_color.z, 1);
  out_col = vec4(norm_world.x, norm_world.y, norm_world.z, 1);
}
