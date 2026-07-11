VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec2 note_times;
attribute vec3 note_info;
attribute float note_depth;

uniform mat4 u_projection;
uniform float u_time;
uniform float u_scroll_speed;
uniform float u_width;
uniform float u_height;
uniform float u_note_width;
uniform float u_guide_line_y;
uniform float u_window_start;
uniform float u_window_end;
uniform float u_glow_strength;
uniform float u_overclock;
uniform vec3 u_colors[128];
uniform vec2 u_pitch_layout[128];
uniform int u_is_white_key[128];

varying vec3 v_fragColor;
varying vec2 v_pos;
varying float v_note_h;
varying float v_note_w;
varying float v_overclock_corrupt;

float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    float on_time = note_times.x;
    float off_time = note_times.y;
    
    if (off_time < u_window_start || on_time > u_window_end) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    
    int pitch = int(note_info.x + 0.5);
    int track = int(note_info.z + 0.5);
    
    float note_duration = max(off_time - on_time, 0.0001);
    float note_h = max(2.0, note_duration * u_scroll_speed);
    float note_y = u_guide_line_y + (u_time - on_time) * u_scroll_speed;
    vec2 key_layout = u_pitch_layout[pitch];
    vec2 instance_scale = vec2(key_layout.y, note_h);
    vec2 instance_offset = vec2(key_layout.x, note_y - note_h);

    vec2 warped_pos = pos;

    vec2 final_pos = warped_pos * instance_scale + instance_offset;

    if (u_overclock > 0.001) {
        float seed = float(pitch) * 1.618 + floor(on_time * 7.0) * 0.381;

        float t0 = floor(u_time * 4.0);
        float t1 = floor(u_time * 9.0);
        float t2 = floor(u_time * 17.0);

        float burst = step(0.55, hash(seed + t0 * 0.17)) * step(0.65, hash(seed * 3.1 + t1 * 0.31));
        float flicker = step(0.45, hash(seed * 5.7 + t2 * 0.53));

        float global_intensity = (burst * 0.7 + flicker * 0.3) * u_overclock;

        float vseed = seed + pos.x * 17.3 + pos.y * 31.7;
        float dx = (hash(vseed * 1.1 + t0) - 0.5) * 2.0;
        float dy = (hash(vseed * 2.3 + t1) - 0.5) * 2.0;

        float mag = global_intensity * mix(15.0, 120.0, hash(vseed * 3.7 + t2));
        final_pos += vec2(dx, dy) * mag;

        float slot = floor(u_time * 3.0);
        float note_id = hash(seed * 13.71);
        float target_id = hash(slot * 7.31);
        float is_corrupted = 1.0 - smoothstep(0.0, 0.003, abs(note_id - target_id));

        float slot_intensity = is_corrupted * u_overclock;

        float sdx = (hash(vseed * 4.1 + slot) - 0.5) * 2.0;
        float sdy = (hash(vseed * 5.3 + slot * 1.3) - 0.5) * 2.0;
        float slot_mag = slot_intensity * mix(50.0, 350.0, hash(vseed * 6.7 + slot * 0.7));
        final_pos += vec2(sdx, sdy) * slot_mag;

        float warp_pull = slot_intensity * pos.x * pos.y;
        final_pos = mix(final_pos, vec2(0.0, 0.0), warp_pull * mix(0.5, 1.0, hash(seed * 9.1 + slot)));

        v_overclock_corrupt = max(global_intensity * 0.4, slot_intensity);
    } else {
        v_overclock_corrupt = 0.0;
    }

    gl_Position = u_projection *
        vec4(final_pos, note_depth, 1.0);

    v_pos = pos;
    v_note_h = note_h;
    v_note_w = key_layout.y;
    
    int color_idx = int(mod(float(track), 128.0));
    v_fragColor = u_colors[color_idx] * 1.2;
}
"""

FRAG_SHADER = """#version 120
varying vec3 v_fragColor;
varying vec2 v_pos;
varying float v_note_h;
varying float v_note_w;
varying float v_overclock_corrupt;

uniform sampler2D u_note_texture;
uniform sampler2D u_note_edge_texture;
uniform float u_glow_strength;
uniform float u_time;

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 fw = fwidth(v_pos);
    float texture_border_y = 1.5 * fw.y;
    vec4 texture_color;
    bool is_edge = false;
    if (v_note_h < 4.0) {
        texture_color = texture2D(u_note_edge_texture, v_pos);
        is_edge = true;
    } else {
        if (v_pos.y < texture_border_y || v_pos.y > 1.0 - texture_border_y) {
            texture_color = texture2D(u_note_edge_texture, v_pos);
            is_edge = true;
        } else {
            texture_color = texture2D(u_note_texture, v_pos);
        }
    }

    vec4 final_color;
    if (is_edge) {
        float edge_bright = max(texture_color.r, max(texture_color.g, texture_color.b));
        final_color = vec4(v_fragColor * max(edge_bright, 0.20), texture_color.a);
    } else {
        final_color = texture_color * vec4(v_fragColor, 1.0);
    }

    if (v_overclock_corrupt > 0.001) {
        float shift = v_overclock_corrupt * 0.12;
        vec3 corrupted = final_color.rgb;
        corrupted.r = texture2D(u_note_texture, v_pos + vec2(shift, shift * 0.3)).r * v_fragColor.r;
        corrupted.b = texture2D(u_note_texture, v_pos - vec2(shift * 0.7, -shift * 0.2)).b * v_fragColor.b;

        float color_noise = hash21(v_pos * 100.0 + floor(u_time * 4.0));
        vec3 injection = vec3(
            step(0.75, color_noise) * 0.9,
            step(0.55, color_noise) * 0.4,
            step(0.35, color_noise) * 0.95
        );
        corrupted = mix(corrupted, injection, v_overclock_corrupt * 0.35);

        final_color.rgb = mix(final_color.rgb, corrupted, v_overclock_corrupt);
    }

    float body_gradient = mix(0.92, 1.08, pow(clamp(1.0 - v_pos.y, 0.0, 1.0), 1.3));
    float center_soft = 1.0 - abs(v_pos.x - 0.5) * 1.2;
    float center_gloss = clamp(center_soft, 0.0, 1.0) * 0.06;
    final_color.rgb *= body_gradient;
    final_color.rgb += v_fragColor * center_gloss;

    float top_band = 1.0 - smoothstep(0.0, max(0.012, 3.0 * fw.y), v_pos.y);
    float top_glint = pow(top_band, 1.8) * 0.22;
    final_color.rgb += vec3(1.0, 1.0, 1.0) * top_glint;

    float outline_px_uv = 1.0 / max(v_note_w, 1.0);
    float left_edge = step(v_pos.x, outline_px_uv);
    float right_edge = step(1.0 - outline_px_uv, v_pos.x);
    if (left_edge + right_edge > 0.0) {
        vec3 outline_color = v_fragColor * 0.35;
        final_color.rgb = outline_color;
    }

    if (u_glow_strength > 0.001) {
        vec2 centered = abs(v_pos - vec2(0.5, 0.5));
        float inner_glow = clamp(1.0 - max(centered.x * 1.4, centered.y * 1.8), 0.0, 1.0);
        float edge_glow = clamp(1.0 - min(min(v_pos.x, 1.0 - v_pos.x), min(v_pos.y, 1.0 - v_pos.y)) * 6.0, 0.0, 1.0);
        float glow = max(pow(inner_glow, 2.0), edge_glow * 0.35) * u_glow_strength;
        final_color.rgb += v_fragColor * glow * 0.55;
    }
    
    gl_FragColor = final_color;
}
"""

BLOOM_VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec2 note_times;
attribute vec3 note_info;

uniform mat4 u_projection;
uniform float u_time;
uniform float u_scroll_speed;
uniform float u_guide_line_y;
uniform float u_window_start;
uniform float u_window_end;
uniform float u_bloom_radius;
uniform vec3 u_colors[128];
uniform vec2 u_pitch_layout[128];

varying vec3 v_bloom_color;
varying vec2 v_bloom_uv;
varying float v_note_h;
varying vec2 v_quad_uv;

void main() {
    float on_time = note_times.x;
    float off_time = note_times.y;

    if (off_time < u_window_start || on_time > u_window_end) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    int pitch = int(note_info.x + 0.5);
    int track = int(note_info.z + 0.5);

    float note_duration = max(off_time - on_time, 0.0001);
    float note_h = max(2.0, note_duration * u_scroll_speed);
    float note_y = u_guide_line_y + (u_time - on_time) * u_scroll_speed;

    vec2 layout = u_pitch_layout[pitch];
    float pad_x = u_bloom_radius * 1.7;
    float pad_y = min(max(u_bloom_radius * 1.9, 12.0), note_h * 0.34 + u_bloom_radius * 0.62);
    vec2 expanded_scale = vec2(layout.y + 2.0 * pad_x, note_h + 2.0 * pad_y);
    vec2 expanded_offset = vec2(layout.x - pad_x, note_y - note_h - pad_y);
    vec2 final_pos = pos * expanded_scale + expanded_offset;

    gl_Position = u_projection * vec4(final_pos, 0.0, 1.0);

    v_bloom_uv = vec2(
        (pos.x * expanded_scale.x - pad_x) / max(layout.y, 1.0),
        (pos.y * expanded_scale.y - pad_y) / max(note_h, 1.0)
    );
    v_note_h = note_h;
    v_quad_uv = pos;

    int color_idx = int(mod(float(track), 128.0));
    v_bloom_color = u_colors[color_idx] * 1.18;
}
"""

BLOOM_FRAG_SHADER = """#version 120
varying vec3 v_bloom_color;
varying vec2 v_bloom_uv;
varying float v_note_h;
varying vec2 v_quad_uv;

uniform float u_bloom_strength;

void main() {
    vec2 outside = max(max(-v_bloom_uv, v_bloom_uv - vec2(1.0, 1.0)), vec2(0.0, 0.0));
    vec2 halo_scale = vec2(0.62, 1.18);
    float outside_dist = length(outside * halo_scale);
    float halo = pow(clamp(1.0 - outside_dist * 0.82, 0.0, 1.0), 1.32);

    float inside_mask = step(0.0, v_bloom_uv.x) * step(v_bloom_uv.x, 1.0) *
                        step(0.0, v_bloom_uv.y) * step(v_bloom_uv.y, 1.0);
    float center = 1.0 - max(abs(v_bloom_uv.x - 0.5) * 0.88, abs(v_bloom_uv.y - 0.5) * 1.02);
    float core = pow(clamp(center, 0.0, 1.0), 2.2) * inside_mask;

    float top_strip = clamp(1.0 - abs(v_bloom_uv.y) * 1.20, 0.0, 1.0) * inside_mask;
    float top_end = pow(clamp(1.0 - max(-(v_bloom_uv.y + 0.02), 0.0) * 2.45, 0.0, 1.0), 1.75);
    float bottom_end = pow(clamp(1.0 - max(v_bloom_uv.y - 1.10, 0.0) * 1.45, 0.0, 1.0), 1.30);
    float end_fade = top_end * bottom_end;
    float quad_edge = min(min(v_quad_uv.x, 1.0 - v_quad_uv.x), min(v_quad_uv.y, 1.0 - v_quad_uv.y));
    float quad_edge_fade = smoothstep(0.0, 0.16, quad_edge);
    float note_boost = clamp(v_note_h / 150.0, 0.0, 0.28);
    float alpha = (halo * 0.34 + core * 0.06 + top_strip * 0.05) * end_fade * quad_edge_fade * (0.84 + note_boost) * u_bloom_strength;

    if (alpha <= 0.001) {
        discard;
    }

    vec3 color = v_bloom_color * (0.58 + halo * 0.24 + core * 0.08);
    gl_FragColor = vec4(color, alpha);
}
"""

KEYBOARD_VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec4 instance_rect;
attribute float instance_is_pressed;
attribute vec3 instance_color;

uniform mat4 u_projection;
uniform float u_keyboard_y;
uniform float u_overclock;
uniform float u_time;

varying vec2 v_uv;
varying float v_is_pressed;
varying vec3 v_color;

float kb_hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    vec2 final_pos = pos * instance_rect.zw + instance_rect.xy;
    final_pos.y += u_keyboard_y;

    if (u_overclock > 0.001) {
        float seed = instance_rect.x * 0.13 + instance_rect.y * 0.37;

        float t0 = floor(u_time * 4.0);
        float t1 = floor(u_time * 9.0);

        float burst = step(0.55, kb_hash(seed + t0 * 0.17)) * step(0.65, kb_hash(seed * 3.1 + t1 * 0.31));
        float flicker = step(0.45, kb_hash(seed * 5.7 + t1 * 0.53));

        float global_intensity = (burst * 0.6 + flicker * 0.3) * u_overclock;

        float vseed = seed + pos.x * 17.3 + pos.y * 31.7;
        float dx = (kb_hash(vseed * 1.1 + t0) - 0.5) * 2.0;
        float dy = (kb_hash(vseed * 2.3 + t1) - 0.5) * 2.0;

        float mag = global_intensity * mix(8.0, 80.0, kb_hash(vseed * 3.7 + t0));
        final_pos += vec2(dx, dy) * mag;

        float slot = floor(u_time * 3.0);
        float key_id = kb_hash(seed * 13.71);
        float target_id = kb_hash(slot * 7.31 + 0.5);
        float is_corrupted = 1.0 - smoothstep(0.0, 0.005, abs(key_id - target_id));

        float slot_intensity = is_corrupted * u_overclock;

        float sdx = (kb_hash(vseed * 4.1 + slot) - 0.5) * 2.0;
        float sdy = (kb_hash(vseed * 5.3 + slot * 1.3) - 0.5) * 2.0;
        float slot_mag = slot_intensity * mix(15.0, 150.0, kb_hash(vseed * 6.7 + slot * 0.7));
        final_pos += vec2(sdx, sdy) * slot_mag;
    }

    gl_Position = u_projection * vec4(final_pos, 0.0, 1.0);
    v_uv = vec2(pos.x, 1.0 - pos.y);
    v_is_pressed = instance_is_pressed;
    v_color = instance_color;
}
"""

KEYBOARD_FRAG_SHADER = """#version 120
varying vec2 v_uv;
varying float v_is_pressed;
varying vec3 v_color;

uniform sampler2D u_texture_unpressed;
uniform sampler2D u_texture_pressed;
uniform bool u_is_white_key;
uniform float u_glow_strength;

void main() {
    vec4 texColor;
    float glow_mode = step(0.001, u_glow_strength);
    {
        vec4 pressed_tex = texture2D(u_texture_pressed, v_uv);
        vec3 pressed_lit = pressed_tex.rgb * (u_is_white_key ? 0.16 : 0.10);
        vec3 normal_pressed = pressed_tex.rgb * v_color;
        vec3 glow_pressed = normal_pressed * 0.82 + v_color * 0.28;
        vec3 pressed_result = mix(normal_pressed, glow_pressed, glow_mode);

        vec4 unpressed_tex = texture2D(u_texture_unpressed, v_uv);
        vec3 unpressed_lit = unpressed_tex.rgb * (u_is_white_key ? 0.08 : 0.04);
        vec3 unpressed_result = mix(unpressed_tex.rgb, unpressed_lit, glow_mode);

        float press_blend = clamp(v_is_pressed, 0.0, 1.0);
        texColor = mix(unpressed_tex, pressed_tex, press_blend);
        gl_FragColor = vec4(mix(unpressed_result, pressed_result, press_blend), texColor.a);
    }

    if (u_is_white_key) {
        vec2 fw_uv = fwidth(v_uv);
        float border_size = 0.5;
        float border_x = border_size * fw_uv.x;
        float border_y = border_size * fw_uv.y;

        if (v_uv.x < border_x || v_uv.x > 1.0 - border_x || v_uv.y < border_y || v_uv.y > 1.0 - border_y) {
            vec3 unpressed_border = mix(vec3(0.5, 0.5, 0.5), vec3(0.14, 0.14, 0.16), glow_mode);
            vec3 pressed_border = mix(vec3(0.2, 0.2, 0.2), vec3(0.18, 0.18, 0.20), glow_mode);
            vec3 border_color = mix(unpressed_border, pressed_border, clamp(v_is_pressed, 0.0, 1.0));
            gl_FragColor = vec4(border_color, texColor.a);
        }
    }

    float light_strength = max(max(v_color.r, v_color.g), v_color.b);
    if (glow_mode > 0.5 && light_strength > 0.001) {
        float luminance = dot(v_color, vec3(0.2126, 0.7152, 0.0722));
        vec2 centered = abs(v_uv - vec2(0.5, 0.48));
        float spread = clamp(1.0 - max(centered.x * 1.45, centered.y * 1.08), 0.0, 1.0);
        float spill = pow(spread, 2.0);
        float intensity = mix(0.62, 0.90, clamp(v_is_pressed, 0.0, 1.0)) * (0.82 + luminance * 0.55);
        gl_FragColor.rgb += v_color * spill * intensity;
    }

    if (v_is_pressed > 0.01 && glow_mode > 0.5) {
        vec2 centered = abs(v_uv - vec2(0.5, 0.5));
        float glow = clamp(1.0 - max(centered.x * 1.6, centered.y * 2.0), 0.0, 1.0);
        gl_FragColor.rgb += v_color * pow(glow, 2.0) * u_glow_strength * 0.82 * clamp(v_is_pressed, 0.0, 1.0);
    }
}
"""

KEYBOARD_BLOOM_VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec4 instance_rect;
attribute float instance_is_pressed;
attribute vec3 instance_color;

uniform mat4 u_projection;
uniform float u_keyboard_y;
uniform float u_overclock;
uniform float u_time;

varying vec2 v_uv;
varying float v_is_pressed;
varying vec3 v_color;

float kb_bloom_hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    vec2 final_pos = pos * instance_rect.zw + instance_rect.xy;
    final_pos.y += u_keyboard_y;

    if (u_overclock > 0.001) {
        float seed = instance_rect.x * 0.13 + instance_rect.y * 0.37;

        float t0 = floor(u_time * 4.0);
        float t1 = floor(u_time * 9.0);

        float burst = step(0.55, kb_bloom_hash(seed + t0 * 0.17)) * step(0.65, kb_bloom_hash(seed * 3.1 + t1 * 0.31));
        float flicker = step(0.45, kb_bloom_hash(seed * 5.7 + t1 * 0.53));

        float global_intensity = (burst * 0.6 + flicker * 0.3) * u_overclock;

        float vseed = seed + pos.x * 17.3 + pos.y * 31.7;
        float dx = (kb_bloom_hash(vseed * 1.1 + t0) - 0.5) * 2.0;
        float dy = (kb_bloom_hash(vseed * 2.3 + t1) - 0.5) * 2.0;

        float mag = global_intensity * mix(8.0, 80.0, kb_bloom_hash(vseed * 3.7 + t0));
        final_pos += vec2(dx, dy) * mag;

        float slot = floor(u_time * 3.0);
        float key_id = kb_bloom_hash(seed * 13.71);
        float target_id = kb_bloom_hash(slot * 7.31 + 0.5);
        float is_corrupted = 1.0 - smoothstep(0.0, 0.005, abs(key_id - target_id));

        float slot_intensity = is_corrupted * u_overclock;

        float sdx = (kb_bloom_hash(vseed * 4.1 + slot) - 0.5) * 2.0;
        float sdy = (kb_bloom_hash(vseed * 5.3 + slot * 1.3) - 0.5) * 2.0;
        float slot_mag = slot_intensity * mix(15.0, 150.0, kb_bloom_hash(vseed * 6.7 + slot * 0.7));
        final_pos += vec2(sdx, sdy) * slot_mag;
    }

    gl_Position = u_projection * vec4(final_pos, 0.0, 1.0);
    v_uv = vec2(pos.x, pos.y);
    v_is_pressed = instance_is_pressed;
    v_color = instance_color;
}
"""

KEYBOARD_BLOOM_FRAG_SHADER = """#version 120
varying vec2 v_uv;
varying float v_is_pressed;
varying vec3 v_color;

uniform bool u_is_white_key;
uniform float u_bloom_strength;

void main() {
    float light_strength = max(max(v_color.r, v_color.g), v_color.b);
    if (light_strength <= 0.001 || u_bloom_strength <= 0.001) {
        discard;
    }

    float width_shape = 1.0 - abs(v_uv.x - 0.5) * (u_is_white_key ? 1.05 : 1.35);
    float height_shape = 1.0 - abs(v_uv.y - 0.44) * (u_is_white_key ? 1.55 : 1.85);
    float core = pow(clamp(min(width_shape, height_shape), 0.0, 1.0), 2.0);

    vec2 centered = abs(v_uv - vec2(0.5, 0.42));
    vec2 halo_scale = u_is_white_key ? vec2(0.78, 1.18) : vec2(0.92, 1.35);
    float halo = pow(clamp(1.0 - length(centered * halo_scale) * 1.55, 0.0, 1.0), 1.55);

    float top_lift = pow(clamp(1.0 - max(v_uv.y - 0.15, 0.0) * 1.2, 0.0, 1.0), 1.25);
    float edge_fade = smoothstep(0.0, 0.14, min(min(v_uv.x, 1.0 - v_uv.x), min(v_uv.y, 1.0 - v_uv.y)));
    float pressed_boost = mix(0.68, 1.0, clamp(v_is_pressed, 0.0, 1.0));
    float alpha = (halo * 0.42 + core * 0.14) * top_lift * edge_fade * pressed_boost * u_bloom_strength;

    if (alpha <= 0.001) {
        discard;
    }

    vec3 color = v_color * (0.70 + halo * 0.24 + core * 0.10);
    gl_FragColor = vec4(color, alpha);
}
"""

SCREEN_BLOOM_VERT_SHADER = """#version 120
attribute vec2 pos;
varying vec2 v_uv;

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = pos * 0.5 + 0.5;
}
"""

BLOOM_EXTRACT_FRAG_SHADER = """#version 120
varying vec2 v_uv;

uniform sampler2D u_scene_texture;
uniform vec2 u_texel_size;

vec3 extract_bright(vec3 color) {
    float peak = max(max(color.r, color.g), color.b);
    float gate = smoothstep(0.48, 0.92, peak);
    return color * gate * gate;
}

void main() {
    vec2 t = u_texel_size;
    vec3 bright = vec3(0.0);

    bright += extract_bright(texture2D(u_scene_texture, v_uv).rgb) * 0.20;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2( t.x * 1.5, 0.0)).rgb) * 0.14;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2(-t.x * 1.5, 0.0)).rgb) * 0.14;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2(0.0,  t.y * 1.5)).rgb) * 0.14;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2(0.0, -t.y * 1.5)).rgb) * 0.14;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2( t.x * 2.5,  t.y * 2.5)).rgb) * 0.08;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2(-t.x * 2.5,  t.y * 2.5)).rgb) * 0.08;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2( t.x * 2.5, -t.y * 2.5)).rgb) * 0.08;
    bright += extract_bright(texture2D(u_scene_texture, v_uv + vec2(-t.x * 2.5, -t.y * 2.5)).rgb) * 0.08;

    gl_FragColor = vec4(bright, 1.0);
}
"""

SCREEN_BLOOM_BLUR_FRAG_SHADER = """#version 120
varying vec2 v_uv;

uniform sampler2D u_source_texture;
uniform vec2 u_texel_size;
uniform vec2 u_direction;

void main() {
    vec2 step_vec = u_texel_size * u_direction * 1.85;
    vec3 color = texture2D(u_source_texture, v_uv).rgb * 0.160000;

    color += texture2D(u_source_texture, v_uv + step_vec * 1.250000).rgb * 0.230000;
    color += texture2D(u_source_texture, v_uv - step_vec * 1.250000).rgb * 0.230000;

    color += texture2D(u_source_texture, v_uv + step_vec * 2.900000).rgb * 0.120000;
    color += texture2D(u_source_texture, v_uv - step_vec * 2.900000).rgb * 0.120000;

    color += texture2D(u_source_texture, v_uv + step_vec * 5.100000).rgb * 0.060000;
    color += texture2D(u_source_texture, v_uv - step_vec * 5.100000).rgb * 0.060000;

    color += texture2D(u_source_texture, v_uv + step_vec * 7.600000).rgb * 0.025000;
    color += texture2D(u_source_texture, v_uv - step_vec * 7.600000).rgb * 0.025000;

    gl_FragColor = vec4(color, 1.0);
}
"""

SCREEN_BLOOM_FRAG_SHADER = """#version 120
varying vec2 v_uv;

uniform sampler2D u_scene_texture;
uniform sampler2D u_bloom_texture;
uniform vec2 u_texel_size;
uniform float u_bloom_strength;

void main() {
    vec3 base = texture2D(u_scene_texture, v_uv).rgb;
    vec2 t = u_texel_size;

    vec3 bloom = texture2D(u_bloom_texture, v_uv).rgb * 0.34;
    bloom += texture2D(u_bloom_texture, v_uv + vec2( t.x * 0.9, 0.0)).rgb * 0.11;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(-t.x * 0.9, 0.0)).rgb * 0.11;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(0.0,  t.y * 0.9)).rgb * 0.11;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(0.0, -t.y * 0.9)).rgb * 0.11;

    bloom += texture2D(u_bloom_texture, v_uv + vec2( t.x * 2.1,  t.y * 2.1)).rgb * 0.09;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(-t.x * 2.1,  t.y * 2.1)).rgb * 0.09;
    bloom += texture2D(u_bloom_texture, v_uv + vec2( t.x * 2.1, -t.y * 2.1)).rgb * 0.09;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(-t.x * 2.1, -t.y * 2.1)).rgb * 0.09;

    bloom += texture2D(u_bloom_texture, v_uv + vec2( t.x * 4.9, 0.0)).rgb * 0.08;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(-t.x * 4.9, 0.0)).rgb * 0.08;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(0.0,  t.y * 4.9)).rgb * 0.08;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(0.0, -t.y * 4.9)).rgb * 0.08;

    bloom += texture2D(u_bloom_texture, v_uv + vec2( t.x * 7.8,  t.y * 7.8)).rgb * 0.05;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(-t.x * 7.8,  t.y * 7.8)).rgb * 0.05;
    bloom += texture2D(u_bloom_texture, v_uv + vec2( t.x * 7.8, -t.y * 7.8)).rgb * 0.05;
    bloom += texture2D(u_bloom_texture, v_uv + vec2(-t.x * 7.8, -t.y * 7.8)).rgb * 0.05;

    vec3 final_color = base + bloom * u_bloom_strength * 1.52;
    gl_FragColor = vec4(final_color, 1.0);
}
"""

GLOW_VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec4 instance_rect;
attribute vec4 instance_color;

uniform mat4 u_projection;

varying vec4 v_color;
varying vec2 v_uv;

void main() {
    vec2 final_pos = pos * instance_rect.zw + instance_rect.xy;
    gl_Position = u_projection * vec4(final_pos, 0.0, 1.0);
    v_color = instance_color;
    v_uv = pos;
}
"""

GLOW_FRAG_SHADER = """#version 120
varying vec4 v_color;
varying vec2 v_uv;

uniform float u_time;

void main() {
    vec2 uv = v_uv - vec2(0.5, 0.5);

    float dist_streak = length(uv * vec2(0.3, 6.0));
    float dist_core = length(uv * vec2(1.5, 2.0));

    float streak = exp(-dist_streak * 4.0) * 0.7;
    float core = exp(-dist_core * 5.0) * 0.9;

    float alpha = core + streak;

    vec3 final_color = mix(v_color.rgb, vec3(1.0), core * 0.2);

    float edge_fade_x = 1.0 - smoothstep(0.35, 0.5, abs(v_uv.x - 0.5));
    float edge_fade_y = 1.0 - smoothstep(0.35, 0.5, abs(v_uv.y - 0.5));
    float edge_fade = edge_fade_x * edge_fade_y;

    float pulse = 0.85 + 0.15 * sin(u_time * 3.0);

    gl_FragColor = vec4(final_color, alpha * v_color.a * edge_fade * pulse);
}
"""
