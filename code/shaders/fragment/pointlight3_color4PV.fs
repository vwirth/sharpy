#version 330

in vec3 norm_world;
in vec3 pos_world;
in vec4 pv_color;

layout(location = 0) out vec4 out_col;

uniform mat4 model;
uniform mat4 view_normal;
uniform mat4 view;
uniform mat4 proj;

uniform vec3 pl1_pos;
uniform vec3 pl1_col;
uniform float pl1_attenuation;

uniform vec3 pl2_pos;
uniform vec3 pl2_col;
uniform float pl2_attenuation;

uniform vec3 pl3_pos;
uniform vec3 pl3_col;
uniform float pl3_attenuation;

void main() {
  float k_ambient = 1;
  float k_diffuse = 0.9;
  float k_specular = 0.2;

  //   vec3 xTangent = dFdx(norm_world.xyz);
  //   vec3 yTangent = dFdy(norm_world.xyz);
  //   vec3 faceNormal = normalize(cross(xTangent, yTangent));

  vec3 ambient = k_ambient * pv_color.xyz;
  vec3 color = ambient;

  vec3 pos_camera = (view * vec4(pos_world.x, pos_world.y, pos_world.z, 1)).xyz;

  vec4 norm_view =
      view_normal * vec4(norm_world.x, norm_world.y, norm_world.z, 1);
  vec3 n = norm_view.xyz;

  if (n.x != 0 || n.y != 0 || n.z != 0) {
    vec3 dist_1 = pos_camera - pl1_pos * 0.001;
    vec3 dist_2 = pos_camera - pl2_pos * 0.001;
    vec3 dist_3 = pos_camera - pl3_pos * 0.001;

    vec3 light_dir1 = normalize(-dist_1);
    vec3 light_dir2 = normalize(-dist_2);
    vec3 light_dir3 = normalize(-dist_3);

    vec3 norm = normalize(n);

    float n_dot_l1 = clamp(dot(norm, light_dir1), 0, 1);
    float n_dot_l2 = clamp(dot(norm, light_dir2), 0, 1);
    float n_dot_l3 = clamp(dot(norm, light_dir3), 0, 1);

    float falloff1 =
        sqrt(dist_1.x * dist_1.x + dist_1.y * dist_1.y + dist_1.z * dist_1.z);
    float falloff2 =
        sqrt(dist_2.x * dist_2.x + dist_2.y * dist_2.y + dist_2.z * dist_2.z);
    float falloff3 =
        sqrt(dist_3.x * dist_3.x + dist_3.y * dist_3.y + dist_3.z * dist_3.z);

    // vec3 color1 = ambient * n_dot_l1 / (pl1_attenuation * falloff1 *
    // falloff1); vec3 color2 = ambient * n_dot_l2 / (pl2_attenuation * falloff2
    // * falloff2); vec3 color3 = ambient * n_dot_l3 / (pl3_attenuation *
    // falloff3 * falloff3);

    vec3 color1 = ambient * n_dot_l1;
    vec3 color2 = ambient * n_dot_l2;
    vec3 color3 = ambient * n_dot_l3;

    color = (color1 + color2 + color3) / 2.0;
    // color.x = n_dot_l1;
    // color.y = n_dot_l2;
    // color.z = n_dot_l3;

    // color.x = norm.x * 1;
    // color.y = norm.y * 1;
    // color.z = norm.z * 1;

    // color.x = abs(view[0][0]);
    // color.y = abs(view[1][1]);
    // color.z = abs(view[2][2]);
    // color = color1;
  }

  out_col = vec4(color.x, color.y, color.z, pv_color.w);
  // out_col = vec4(norm_world.x, norm_world.y, norm_world.z, pv_color.w);
  // out_col = vec4(color.x, 0, 0, 1);
}
