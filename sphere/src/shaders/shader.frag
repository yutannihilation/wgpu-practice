#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 0) out vec4 o_Target;

void main() {
    vec4 tex = vec4(0.7);
    float mag = length(v_TexCoord-vec2(0.5));
    o_Target = vec4(mix(tex.xyz, vec3(0.0), mag*mag), 1.0);
}
