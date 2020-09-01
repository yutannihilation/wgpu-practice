#version 450

layout(location = 0) in vec4 a_position;

layout(set = 0, binding = 0) uniform Locals {
    vec4 u_view_position;
    mat4 u_view_proj;
};

struct Data {
  mat4 s_models;
  vec4 s_color;
};

layout(set = 0, binding = 1) buffer Instances {
    Data[] data;
};

void main() {
    gl_Position = u_view_proj * data[gl_InstanceIndex].s_models * a_position;
}
