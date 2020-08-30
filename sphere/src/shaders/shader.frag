#version 450

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec4 v_color;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 png_color;

layout(set = 0, binding = 0) uniform Locals {
    vec3 u_view_position;
    mat4 u_view_proj;
};

layout(set = 1, binding = 0) uniform Light {
    vec4 light_position;
    vec3 light_color;
};

void main() {
    vec4 object_color = v_color;
    // We don't need (or want) much ambient light, so 0.1 is fine
    float ambient_strength = 0.1;
    vec3 ambient_color = light_color * ambient_strength;

    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(light_position.xyz - v_position);

    float diffuse_strength = max(dot(normal, light_dir), 0.0);
    vec3 diffuse_color = light_color * diffuse_strength;

    vec3 view_dir = normalize(u_view_position - v_position);
    vec3 reflect_dir = reflect(-light_dir, normal);

    float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec3 specular_color = specular_strength * light_color;

    vec3 result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

    // Since lights don't typically (afaik) cast transparency, so we use
    // the alpha here at the end.
    f_color = vec4(result, object_color.a);
    
    png_color = f_color;
}
