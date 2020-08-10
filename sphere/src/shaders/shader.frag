#version 450

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec2 _v_tex_coords;
layout(location = 2) in vec3 v_normal;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 png_color;

layout(set = 1, binding = 0) uniform Light {
    vec3 light_position;
    vec3 light_color;
};

void main() {
    vec4 object_color = vec4(0.7, 0.2, 0.3, 1.0);
    // We don't need (or want) much ambient light, so 0.1 is fine
    float ambient_strength = 0.1;
    vec3 ambient_color = light_color * ambient_strength;

    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(light_position - v_position);

    float diffuse_strength = max(dot(normal, light_dir), 0.0);
    vec3 diffuse_color = light_color * diffuse_strength;

    vec3 result = (ambient_color + diffuse_color) * object_color.xyz;

    // Since lights don't typically (afaik) cast transparency, so we use
    // the alpha here at the end.
    f_color = vec4(result, object_color.a);
    
    png_color = f_color;
}
