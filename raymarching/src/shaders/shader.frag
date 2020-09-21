#version 450

// layout(location = 0) in vec3 v_position;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 png_color;

layout(set = 0, binding = 0) uniform View {
    vec4 u_view_position;
    mat4 u_view_proj;
};

layout(set = 1, binding = 0) uniform Light {
    mat4 light_view_proj;
    vec4 light_position;
    vec3 light_color;
};

void main() {
    f_color = gl_FragCoord;
    png_color = f_color;
}
