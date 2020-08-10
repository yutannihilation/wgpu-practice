#version 450

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec2 _a_tex_coords; //unused
layout(location = 2) in vec3 a_normal;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec2 _v_tex_coords;
layout(location = 2) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_normal = a_normal;

    _v_tex_coords = _a_tex_coords;

    v_position = a_position.xyz;

    gl_Position = u_Transform * a_position;
}
