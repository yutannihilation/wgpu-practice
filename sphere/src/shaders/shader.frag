#version 450

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec4 v_color;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 png_color;

layout(set = 0, binding = 0) uniform Locals {
    vec4 u_view_position;
    mat4 u_view_proj;
};

layout(set = 1, binding = 0) uniform Light {
    mat4 light_view_proj;
    vec4 light_position;
    vec3 light_color;
};
layout(set = 1, binding = 1) uniform texture2D t_Shadow;
layout(set = 1, binding = 2) uniform samplerShadow s_Shadow;

const int pcf_size = 4;

// original code is https://github.com/gfx-rs/wgpu-rs/blob/d6ff0b63505a883c847a08c99f0e2e009e15d2c4/examples/shadow/forward.frag#L31-L45
float fetch_shadow(vec4 homogeneous_coords, float bias) {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }

    // c.f. https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
    float z_local = homogeneous_coords.z / homogeneous_coords.w;
    if (z_local > 1.0) {
        return 0.0;
    }

    // To prevent shadow acne, add a small bias
    z_local -= bias;

    // compensate for the Y-flip difference between the NDC and texture coordinates
    const vec2 flip_correction = vec2(0.5, -0.5);

    float shadow = 0.0;
    
    vec2 texel_size = 1.0 / vec2(textureSize(sampler2DShadow(t_Shadow, s_Shadow), 0));

    for (int x = -pcf_size; x <= pcf_size; ++x) {
        for (int y = -pcf_size; y <= pcf_size; ++y) {
            vec2 offset = vec2(x, y) * texel_size;
            // compute texture coordinates for shadow lookup
            vec3 light_local = vec3(
                homogeneous_coords.xy * flip_correction / homogeneous_coords.w + 0.5 + offset,
                z_local
            );
            shadow += texture(sampler2DShadow(t_Shadow, s_Shadow), light_local);
        }
    }

    return shadow / pow(2 * pcf_size + 1, 2);
}


void main() {
    vec4 object_color = v_color;
    // We don't need (or want) much ambient light, so 0.1 is fine
    float ambient_strength = 0.1;
    vec3 ambient_color = light_color * ambient_strength;

    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(light_position.xyz - v_position);

    float bias = max(0.003 * (1.0 - dot(normal, light_dir)), 0.001);
    float shadow = fetch_shadow(light_view_proj * vec4(v_position, 1.0), bias);

    float diffuse_strength = max(dot(normal, light_dir), 0.0);
    vec3 diffuse_color = shadow * light_color * diffuse_strength;

    vec3 view_dir = normalize(u_view_position.xyz - v_position);
    vec3 reflect_dir = reflect(-light_dir, normal);

    float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec3 specular_color = specular_strength * light_color;

    vec3 result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

    // Since lights don't typically (afaik) cast transparency, so we use
    // the alpha here at the end.
    f_color = vec4(result, object_color.a);
    
    png_color = f_color;
}
