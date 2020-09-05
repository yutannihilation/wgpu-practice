#version 450

layout(location = 0) in vec4 a_position;

struct Data {
  mat4 s_models;
  vec4 s_color;
};

layout(set = 0, binding = 1) buffer Instances {
    Data[] data;
};

layout(set = 1, binding = 0) uniform Light {
    mat4 light_view_proj;
    vec4 light_position;
    vec3 light_color;
};

void main() {
    gl_Position = light_view_proj * data[gl_InstanceIndex].s_models * a_position;
}
