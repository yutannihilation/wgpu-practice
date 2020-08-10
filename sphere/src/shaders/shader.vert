#version 450

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec3 a_normal;

layout(location = 2) in mat4 a_model;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Locals {
    vec3 u_view_position;
    mat4 u_view_proj;
};

void main() {
    mat3 normal_matrix = mat3(transpose(inverse(a_model)));
    v_normal = normal_matrix * a_normal;

    v_position = a_position.xyz;

    vec4 model_space = a_model * a_position;
    v_position = model_space.xyz;

    gl_Position = u_view_proj * model_space;
}
