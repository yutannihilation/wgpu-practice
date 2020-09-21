#version 450

// layout(location = 0) in vec4 a_position;

// layout(location = 0) out vec3 v_position;

layout(set = 0, binding = 0) uniform View {
    vec4 u_view_position;
    mat4 u_view_proj;
};

void main() {
//    gl_Position = u_view_proj * a_position;
    gl_Position = vec4(1.0);
}
