#version 450

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_staging;
layout(set = 0, binding = 1) uniform sampler s_staging;

float weight[5] = float[] (0.257317771, 0.208992692, 0.111973449, 0.039575066, 0.009226795);

layout(set = 1, binding = 0)
uniform Uniforms {
    uint horizontal;
};

void main() {
    vec2 tex_offset = 1.0 / textureSize(sampler2D(t_staging, s_staging), 0); // gets size of single texel
    f_color = texture(sampler2D(t_staging, s_staging), v_tex_coords) * weight[0]; // current fragment's contribution
    if (horizontal == 1) {
        for(int i = 1; i < 5; ++i) {
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords + vec2(tex_offset.x * i, 0.0)) * weight[i];
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords - vec2(tex_offset.x * i, 0.0)) * weight[i];
        }
    }
    if (horizontal == 0) {
        for(int i = 1; i < 5; ++i) {
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords + vec2(0.0, tex_offset.y * i)) * weight[i];
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords - vec2(0.0, tex_offset.y * i)) * weight[i];
        }
    }
}
