#version 450

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec3 a_normal;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_color;

layout(set = 0, binding = 0) uniform Locals {
    vec4 u_view_position;
    mat4 u_view_proj;
};

struct Data {
  mat4 s_models;
  vec3 s_color;
  float s_normal; // per instance normal adjustment
};

layout(set = 0, binding = 1) buffer Instances {
    Data[] data;
};

void main() {
    v_color = data[gl_InstanceIndex].s_color;

    mat3 normal_matrix = mat3(transpose(inverse(data[gl_InstanceIndex].s_models)));
    v_normal = normal_matrix * a_normal * data[gl_InstanceIndex].s_normal;

    v_position = a_position.xyz;

    vec4 model_space = data[gl_InstanceIndex].s_models * a_position;
    v_position = model_space.xyz;

    gl_Position = u_view_proj * model_space;
}
