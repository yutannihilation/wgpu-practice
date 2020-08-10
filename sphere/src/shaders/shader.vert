#version 450

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec3 a_normal;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_normal = a_normal;

    v_position = a_position.xyz;

    gl_Position = u_Transform * a_position;
}
