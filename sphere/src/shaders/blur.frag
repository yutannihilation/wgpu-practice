#version 450

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_staging;
layout(set = 0, binding = 1) uniform sampler s_staging;
layout(set = 0, binding = 2) uniform texture2D t_depth;
layout(set = 0, binding = 3) uniform samplerShadow s_depth;

layout(set = 1, binding = 0)
uniform Uniforms {
    uint horizontal;
};

// Original code is http://blog.tuxedolabs.com/2018/05/04/bokeh-depth-of-field-in-single-pass.html
const float GOLDEN_ANGLE = 2.39996323; 
const float MAX_BLUR_SIZE = 30.0; 
const float RAD_SCALE = 0.5; // Smaller = nicer blur, larger = faster
const float FOCUS_SCALE = 2.0; // TODO: provide dynamically as uniform
// TOOD: the depth is around 0.9~0.95? Maybe I'm doing somthing wrong...
const float FOCUS_POINT = 0.92;
const float FAR = 1.0;

float getBlurSize(float depth) {
	// float coc = clamp((1.0 / FOCUS_POINT - 1.0 / depth) * FOCUS_SCALE, -1.0, 1.0);
	float coc = clamp((1.0 / FOCUS_POINT - 1.0 / depth) * FOCUS_SCALE, -1.0, 1.0);
	return abs(coc) * MAX_BLUR_SIZE;
}

void main() {
    vec2 texel_size = 1.0 / textureSize(sampler2D(t_staging, s_staging), 0);

    float center_depth = texture(sampler2D(t_depth, s_staging), v_tex_coords).r * FAR;

    // // debug
    // if (center_depth > 0.95) {
    //     f_color = vec4(0);
    //     return;
    // }

    float blur_size = getBlurSize(center_depth);

    vec3 color = texture(sampler2D(t_staging, s_staging), v_tex_coords).rgb;
    float cnt = 1.0;

    vec2 direction;
    if (horizontal == 1) {
        direction = vec2(1.0, 0.0);
    } else {
        direction = vec2(0.0, 1.0);
    }

	float radius = RAD_SCALE;
    for (float i = 0; i < blur_size; i++) {
		vec2 tc = v_tex_coords + direction * texel_size * i;
		vec3 sample_color = texture(sampler2D(t_staging, s_staging), tc).rgb;
        float sample_depth = texture(sampler2D(t_depth, s_staging), tc).r;
		// float sample_depth = texture(sampler2DShadow(t_depth, s_depth), vec3(tc, 1.0)) * FAR;
		float sample_blur_size = getBlurSize(sample_depth);
		if (sample_depth > center_depth)
			sample_blur_size = clamp(sample_blur_size, 0.0, blur_size * 2.0);
		float m = smoothstep(radius - 0.5, radius + 0.5, sample_blur_size);
		color += mix(color / cnt, sample_color, m);
		cnt += 1.0;
    }

	// float radius = RAD_SCALE;
	// for (float ang = 0.0; radius < MAX_BLUR_SIZE; ang += GOLDEN_ANGLE) {
	// 	vec2 tc = v_tex_coords + direction * vec2(cos(ang), sin(ang)) * texel_size * radius;
	// 	vec3 sample_color = texture(sampler2D(t_staging, s_staging), tc).rgb;
	// 	float sample_depth = texture(sampler2D(t_depth, s_staging), tc).r * FAR;
	// 	float sample_blur_size = getBlurSize(sample_depth);
	// 	if (sample_depth > center_depth)
	// 		sample_blur_size = clamp(sample_blur_size, 0.0, blur_size * 2.0);
	// 	float m = smoothstep(radius - 0.5, radius + 0.5, sample_blur_size);
	// 	color += mix(color / cnt, sample_color, m);
	// 	cnt += 1.0;
    //     radius += RAD_SCALE / radius;
	// }

    f_color = vec4(color / cnt, 1.0);
}
