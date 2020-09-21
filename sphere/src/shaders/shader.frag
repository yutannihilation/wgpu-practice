#version 450

const int MAX_LIGHTS = 10;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec4 v_color;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 png_color;

layout(set = 0, binding = 0) uniform Globals {
    vec4 u_position;
    mat4 u_view_proj;
    int num_of_lights;
};

struct Light {
    mat4 view_proj;
    vec4 position;
    vec3 color;
};
layout(set = 1, binding = 0) uniform Lights {
    Light u_lights[MAX_LIGHTS];
};

layout(set = 1, binding = 1) uniform texture2DArray t_Shadow;
layout(set = 1, binding = 2) uniform samplerShadow s_Shadow;

const int pcf_size = 4;

// original code is https://github.com/gfx-rs/wgpu-rs/blob/d6ff0b63505a883c847a08c99f0e2e009e15d2c4/examples/shadow/forward.frag#L31-L45
float fetch_shadow(int light_id, vec4 homogeneous_coords, float bias) {
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
    
    vec2 texel_size = 1.0 / vec2(textureSize(sampler2DArrayShadow(t_Shadow, s_Shadow), 0));

    for (int x = -pcf_size; x <= pcf_size; ++x) {
        for (int y = -pcf_size; y <= pcf_size; ++y) {
            vec2 offset = vec2(x, y) * texel_size;
            // compute texture coordinates for shadow lookup
            vec4 light_local = vec4(
                homogeneous_coords.xy * flip_correction / homogeneous_coords.w + 0.5 + offset,
                light_id,
                z_local
            );
            shadow += texture(sampler2DArrayShadow(t_Shadow, s_Shadow), light_local);
        }
    }

    return shadow / pow(2 * pcf_size + 1, 2);
}


void main() {
    vec4 object_color = v_color;
    // We don't need (or want) much ambient light
    float ambient_strength = 0.005 / num_of_lights;

    vec3 normal = normalize(v_normal);

    vec3 color = vec3(0.0);

    for (int i=0; i<int(num_of_lights) && i<MAX_LIGHTS; ++i) {

        // ambient color --------------------------------

        vec3 ambient_color = u_lights[i].color * ambient_strength;

        // diffuse color --------------------------------

        vec3 light_dir = normalize(u_lights[i].position.xyz - v_position);

        float bias = max(0.003 * (1.0 - dot(normal, light_dir)), 0.001);
        float shadow = fetch_shadow(i, u_lights[i].view_proj * vec4(v_position, 1.0), bias);

        float diffuse_strength = max(dot(normal, light_dir), 0.0);
        vec3 diffuse_color = shadow * u_lights[i].color * diffuse_strength;

        // specular color --------------------------------
        
        vec3 view_dir = normalize(u_position.xyz - v_position);
        vec3 reflect_dir = reflect(-light_dir, normal);

        float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
        vec3 specular_color = specular_strength * u_lights[i].color;

        // blend --------------------------------

        color += (ambient_color + diffuse_color + specular_color);
    }

    f_color = vec4(color, 1.0) * object_color;
    
    png_color = f_color;
}
