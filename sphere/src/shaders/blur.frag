#version 450

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_staging;
layout(set = 0, binding = 1) uniform sampler s_staging;

float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

layout(set = 1, binding = 0)
uniform Uniforms {
    bool horizontal;
};

void main() {
    vec2 tex_offset = 1.0 / textureSize(sampler2D(t_staging, s_staging), 0); // gets size of single texel
    f_color = texture(sampler2D(t_staging, s_staging), v_tex_coords) * weight[0]; // current fragment's contribution
    if (horizontal) {
        for(int i = 1; i < 5; ++i) {
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords + vec2(tex_offset.x * i, 0.0)) * weight[i];
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords - vec2(tex_offset.x * i, 0.0)) * weight[i];
        }
    } else {
        for(int i = 1; i < 5; ++i) {
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords + vec2(0.0, tex_offset.y * i)) * weight[i];
            f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords - vec2(0.0, tex_offset.y * i)) * weight[i];
        }
    }
}
