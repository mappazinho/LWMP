import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import random
import threading
import queue
import numpy as np
import ctypes
import xml.etree.ElementTree as ET
import os
import sys
from midi_parser import GPU_NOTE_DTYPE
from config import save_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_ROOT = getattr(sys, "_MEIPASS", _SCRIPT_DIR)


def _first_existing_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else ""


_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_SKIN_DIR = _first_existing_path(
    os.path.join(_PROJECT_DIR, "skin"),
    os.path.join(_SCRIPT_DIR, "skin"),
    os.path.join(_RUNTIME_ROOT, "skin"),
)
_COLORS_XML_PATH = _first_existing_path(
    os.path.join(_SCRIPT_DIR, "colors.xml"),
    os.path.join(_PROJECT_DIR, "MIDI", "colors.xml"),
    os.path.join(_RUNTIME_ROOT, "MIDI", "colors.xml"),
    os.path.join(_RUNTIME_ROOT, "colors.xml"),
)

VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec2 note_times;
attribute vec3 note_info;

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
uniform vec3 u_colors[128];
uniform vec2 u_pitch_layout[128];
uniform int u_is_white_key[128];

varying vec3 v_fragColor;
varying vec2 v_pos;
varying float v_note_h;
varying float v_note_w;

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
    vec2 instance_scale = vec2(layout.y, note_h);
    vec2 instance_offset = vec2(layout.x, note_y - note_h);
    vec2 final_pos = pos * instance_scale + instance_offset;
    
    float base_z = u_is_white_key[pitch] == 1 ? 0.0 : 0.5;
    float time_offset = mod(on_time, 1000.0) * 0.0004;

    float z_depth = base_z + time_offset;
    
    gl_Position = u_projection * vec4(final_pos, z_depth, 1.0);
    
    v_pos = pos;
    v_note_h = note_h;
    v_note_w = layout.y;
    
    int color_idx = int(mod(float(track), 128.0));
    v_fragColor = u_colors[color_idx] * 1.2;
}
"""

FRAG_SHADER = """#version 120
varying vec3 v_fragColor;
varying vec2 v_pos;
varying float v_note_h;
varying float v_note_w;

uniform sampler2D u_note_texture;
uniform sampler2D u_note_edge_texture;
uniform float u_glow_strength;

void main() {
    vec2 fw = fwidth(v_pos);
    float texture_border_y = 1.5 * fw.y;
    vec4 texture_color;
    if (v_note_h < 4.0) {
        texture_color = texture2D(u_note_edge_texture, v_pos);
    } else {
        if (v_pos.y < texture_border_y || v_pos.y > 1.0 - texture_border_y) {
            texture_color = texture2D(u_note_edge_texture, v_pos);
        } else {
            texture_color = texture2D(u_note_texture, v_pos);
        }
    }
    
    vec4 final_color = texture_color * vec4(v_fragColor, 1.0);

    float outline_px_uv = 1.0 / max(v_note_w, 1.0);
    float left_edge = step(v_pos.x, outline_px_uv);
    float right_edge = step(1.0 - outline_px_uv, v_pos.x);
    if (left_edge + right_edge > 0.0) {
        vec3 outline_color = vec3(0.08, 0.08, 0.10);
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

KEYBOARD_VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec4 instance_rect;
attribute float instance_is_pressed;
attribute vec3 instance_color;

uniform mat4 u_projection;
uniform float u_keyboard_y;

varying vec2 v_uv;
varying float v_is_pressed;
varying vec3 v_color;

void main() {
    vec2 final_pos = pos * instance_rect.zw + instance_rect.xy;
    final_pos.y += u_keyboard_y;
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
    if (v_is_pressed > 0.5) {
        texColor = texture2D(u_texture_pressed, v_uv);
        vec3 lit_base = texColor.rgb * (u_is_white_key ? 0.16 : 0.10);
        vec3 normal_pressed = texColor.rgb * v_color;
        vec3 glow_pressed = normal_pressed * 0.82 + v_color * 0.28;
        gl_FragColor = vec4(mix(normal_pressed, glow_pressed, glow_mode), texColor.a);
    } else {
        texColor = texture2D(u_texture_unpressed, v_uv);
        vec3 lit_base = texColor.rgb * (u_is_white_key ? 0.08 : 0.04);
        gl_FragColor = vec4(mix(texColor.rgb, lit_base, glow_mode), texColor.a);
    }

    if (u_is_white_key) {
        vec2 fw_uv = fwidth(v_uv);
        float border_size = 0.5;
        float border_x = border_size * fw_uv.x;
        float border_y = border_size * fw_uv.y;

        if (v_uv.x < border_x || v_uv.x > 1.0 - border_x || v_uv.y < border_y || v_uv.y > 1.0 - border_y) {
            vec3 border_color = mix(vec3(0.5, 0.5, 0.5), vec3(0.14, 0.14, 0.16), glow_mode);
            if (v_is_pressed > 0.5) {
                border_color = mix(vec3(0.2, 0.2, 0.2), vec3(0.18, 0.18, 0.20), glow_mode);
            }
            gl_FragColor = vec4(border_color, texColor.a);
        }
    }

    float light_strength = max(max(v_color.r, v_color.g), v_color.b);
    if (glow_mode > 0.5 && light_strength > 0.001) {
        float luminance = dot(v_color, vec3(0.2126, 0.7152, 0.0722));
        vec2 centered = abs(v_uv - vec2(0.5, 0.48));
        float spread = clamp(1.0 - max(centered.x * 1.45, centered.y * 1.08), 0.0, 1.0);
        float spill = pow(spread, 2.0);
        float intensity = (v_is_pressed > 0.5 ? 0.90 : 0.62) * (0.82 + luminance * 0.55);
        gl_FragColor.rgb += v_color * spill * intensity;
    }

    if (v_is_pressed > 0.5 && glow_mode > 0.5) {
        vec2 centered = abs(v_uv - vec2(0.5, 0.5));
        float glow = clamp(1.0 - max(centered.x * 1.6, centered.y * 2.0), 0.0, 1.0);
        gl_FragColor.rgb += v_color * pow(glow, 2.0) * u_glow_strength * 0.82;
    }
}
"""

class PianoRoll:
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.config = config
        self.screen = None
        
        self.shader = 0
        self.vao = 0
        self.vbo_vertices = 0
        self.vbo_stream_data = 0
        self.note_texture = 0
        self.note_edge_texture = 0
        self.u_note_texture_loc = -1
        self.u_note_edge_texture_loc = -1
        self.u_is_white_key_notes_loc = -1
        self.u_glow_strength_loc = -1
        
        self.all_notes_gpu = None
        self.data_queue = queue.Queue(maxsize=2)
        self.app_running = threading.Event()
        self.app_running.set()
        self.force_data_update = threading.Event()
        self.data_thread = None
        self.notes_to_draw = 0
        self.current_note_data = None
        self.get_current_time = None
        
        vis_cfg = self.config.get('visualizer', {})
        self.scroll_speed = float(vis_cfg.get('scroll_speed', 2500.0))
        self.scroll_slider_min = 200.0
        self.scroll_slider_max = 5000.0
        self.scroll_slider_dragging = False
        self.scroll_slider_rect = None
        self.scroll_slider_bar = None
        self.scroll_slider_handle_x = 0.0
        self.note_width = float(vis_cfg.get('note_width', 10.0))
        self.show_guide_line = bool(vis_cfg.get('show_guide_line', True))
        self.show_glow = bool(vis_cfg.get('show_glow', False))
        self.glow_strength = 1.0 if self.show_glow else 0.0
        self.show_key_press_glow = bool(vis_cfg.get('show_key_press_glow', True))
        self.show_key_light_fade = bool(vis_cfg.get('show_key_light_fade', False))
        self.glow_fade_duration = 0.1
        
        self.slider_min = 0.2
        self.slider_max = 5.0
        
        self.window_seconds = float(vis_cfg.get('seconds_before_cursor', 3.0))
        self.seconds_before_cursor = self.window_seconds
        self.seconds_after_cursor = self.window_seconds
        
        self.slider_dragging = False
        self.slider_rect = None
        self.slider_bar = None
        self.slider_handle_x = 0.0
        self.slider_area_height = 30
        
        self.streaming_vbo_capacity = int(vis_cfg.get('streaming_vbo_capacity', 2000000))
        self.guide_line_y_ratio = float(vis_cfg.get('guide_line_y_ratio', 0.8))
        self.use_gpu_cull = True
        self.data_update_interval = float(vis_cfg.get('data_update_interval', 0.01))

        self.show_keyboard = bool(vis_cfg.get('show_keyboard', True))
        self.keyboard_layout = None
        self.keyboard_textures = {}
        self.keyboard_texture_info = {}
        
        self.keyboard_shader = 0
        self.keyboard_vao_white = 0
        self.keyboard_vao_black = 0
        self.keyboard_vbo_quad = 0
        self.keyboard_vbo_white_keys = 0
        self.keyboard_vbo_black_keys = 0
        self.white_key_instance_data = None
        self.black_key_instance_data = None
        self.white_key_pitch_map = {}
        self.black_key_pitch_map = {}
        self.u_is_white_key_loc = -1
        
        self.active_pitches_last_frame = set()
        self.last_visible_notes = None
        self.glow_trails = {}
        self.last_glow_time = None
        self.render_notes_array = None
        self.render_on_times = None
        self.max_note_duration = 10.0
        
        self.pending_midi_data = None
        self.midi_data_lock = threading.Lock()
        
        self.color_button_rect = None
        self.glow_button_rect = None
        self.glow_options_button_rect = None
        self.glow_options_panel_rect = None
        self.glow_options_checkbox_rect = None
        self.key_light_fade_checkbox_rect = None
        self.glow_options_expanded = False
        self.color_button_size = 32
        self.controls_panel_expanded = False
        self.controls_toggle_rect = None
        self.controls_panel_rect = None
        self.controls_close_rect = None
        self.controls_panel_size = (300, 136)
        self.overlay_font = None
        self._text_texture_cache = {}
        self.hover_fade_states = {}
        self.hover_tooltip_text = None
        self.hover_mouse_pos = (0, 0)

    def _get_guide_line_y(self):
        if self.show_keyboard and 'white_key' in self.keyboard_texture_info:
            keyboard_height = self.keyboard_texture_info['white_key'].get('scaled_height')
            if keyboard_height is not None:
                return self.height - keyboard_height
        return self.height * self.guide_line_y_ratio

    def _combine_light_color(self, color_sum, weight_sum, color_peak):
        if weight_sum <= 0.0:
            return np.array(color_peak, dtype=np.float32)
        avg_color = color_sum / weight_sum
        combined = color_peak * 0.72 + avg_color * 0.28
        max_component = float(np.max(combined))
        if max_component > 1.0:
            combined = combined / max_component
        return np.clip(combined, 0.0, 1.0).astype(np.float32)

    def _color_luminance(self, color):
        return float(color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722)

    def _save_visualizer_config(self):
        vis_cfg = self.config.setdefault('visualizer', {})
        vis_cfg['show_glow'] = bool(self.show_glow)
        vis_cfg['show_key_press_glow'] = bool(self.show_key_press_glow)
        vis_cfg['show_key_light_fade'] = bool(self.show_key_light_fade)
        vis_cfg['seconds_before_cursor'] = float(self.window_seconds)
        vis_cfg['seconds_after_cursor'] = float(self.window_seconds)
        vis_cfg['scroll_speed'] = float(self.scroll_speed)
        vis_cfg['streaming_vbo_capacity'] = int(self.streaming_vbo_capacity)
        save_config(self.config)

    def _iter_hover_targets(self):
        if (not self.controls_panel_expanded) and self.controls_toggle_rect:
            yield ("controls_toggle", self.controls_toggle_rect, "Open control panel")
        if self.controls_panel_expanded and self.controls_close_rect:
            yield ("controls_close", self.controls_close_rect, "Close control panel")
        if self.glow_button_rect:
            yield ("glow_button", self.glow_button_rect, "Toggle glow mode")
        if self.glow_options_button_rect:
            yield ("glow_options", self.glow_options_button_rect, "Glow options")
        if self.color_button_rect:
            yield ("color_button", self.color_button_rect, "Randomize note colors")
        if self.glow_options_expanded and self.glow_options_checkbox_rect:
            yield ("glow_checkbox", self.glow_options_checkbox_rect, "Toggle glow on key press")
        if self.glow_options_expanded and self.key_light_fade_checkbox_rect:
            yield ("key_fade_checkbox", self.key_light_fade_checkbox_rect, "Toggle fade key lighting")

    def _update_hover_ui_state(self):
        if self.overlay_font is None:
            return
        mouse_pos = pygame.mouse.get_pos()
        self.hover_mouse_pos = mouse_pos

        hovered_id = None
        hovered_text = None
        for target_id, rect, tooltip in self._iter_hover_targets():
            if rect and rect.collidepoint(mouse_pos):
                hovered_id = target_id
                hovered_text = tooltip
                break

        self.hover_tooltip_text = hovered_text
        seen_ids = set()
        if hovered_id is not None:
            seen_ids.add(hovered_id)
        for target_id, _, _ in self._iter_hover_targets():
            seen_ids.add(target_id)

        for target_id in seen_ids:
            current = self.hover_fade_states.get(target_id, 0.0)
            target = 1.0 if target_id == hovered_id else 0.0
            current += (target - current) * 0.28
            if abs(current) < 0.01 and target == 0.0:
                self.hover_fade_states.pop(target_id, None)
            else:
                self.hover_fade_states[target_id] = current

    def _get_hover_alpha(self, target_id):
        return float(self.hover_fade_states.get(target_id, 0.0))

    def _draw_hover_highlight(self, rect, alpha):
        if not rect or alpha <= 0.001:
            return
        glColor4f(1.0, 1.0, 1.0, 0.14 * alpha)
        glBegin(GL_QUADS)
        glVertex2f(rect.x, rect.y); glVertex2f(rect.x + rect.width, rect.y)
        glVertex2f(rect.x + rect.width, rect.y + rect.height); glVertex2f(rect.x, rect.y + rect.height)
        glEnd()

    def _draw_hover_tooltip(self):
        if not self.hover_tooltip_text:
            return
        tex_info = self._get_text_texture(self.hover_tooltip_text, (235, 238, 244))
        if tex_info is None:
            return
        _, text_w, text_h = tex_info
        padding_x = 10
        padding_y = 8
        box_w = text_w + padding_x * 2
        box_h = text_h + padding_y * 2
        mouse_x, mouse_y = self.hover_mouse_pos
        box_x = min(mouse_x + 14, self.width - box_w - 8)
        box_y = min(mouse_y + 16, self.height - box_h - 8)

        glColor4f(0.06, 0.06, 0.09, 0.94)
        glBegin(GL_QUADS)
        glVertex2f(box_x, box_y); glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h); glVertex2f(box_x, box_y + box_h)
        glEnd()
        glColor4f(0.82, 0.84, 0.90, 0.9)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(box_x, box_y); glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h); glVertex2f(box_x, box_y + box_h)
        glEnd()
        self._draw_text_overlay(self.hover_tooltip_text, box_x + padding_x, box_y + padding_y)

    def _should_replace_spill_color(self, existing_entry, new_weight, new_distance, new_luminance):
        if existing_entry is None:
            return True
        if new_weight > existing_entry['weight'] + 1e-6:
            return True
        if abs(new_weight - existing_entry['weight']) <= 1e-6:
            if new_distance < existing_entry['distance']:
                return True
            if new_distance == existing_entry['distance'] and new_luminance > existing_entry['luminance']:
                return True
        return False

    def _load_colors_from_xml(self, filepath=None):
        if filepath is None:
            filepath = _COLORS_XML_PATH

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            colors = []
            for color_elem in root.findall('Color'):
                r = int(color_elem.get('R')) / 255.0
                g = int(color_elem.get('G')) / 255.0
                b = int(color_elem.get('B')) / 255.0
                colors.append([r, g, b])
            
            if not colors:
                raise ValueError("No colors in XML")
                
            num_colors = len(colors)
            full_color_list = []
            for i in range(128):
                full_color_list.append(colors[i % num_colors])
            
            return np.array(full_color_list, dtype=np.float32)

        except (ET.ParseError, FileNotFoundError, ValueError) as e:
            print(f"Error loading colors from {filepath}: {e}. Using default rainbow palette.")
            rainbow = [
                [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.5, 1.0],
                [0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.5],
                [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0], [0.8, 0.8, 0.8]
            ]
            full_color_list = []
            for i in range(128):
                full_color_list.append(rainbow[i % 16])
            return np.array(full_color_list, dtype=np.float32)

    def _load_texture(self, image_path):
        try:
            image = pygame.image.load(image_path).convert_alpha()
            width, height = image.get_size()
            image_data = pygame.image.tostring(image, "RGBA", True)

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

            return texture_id, width, height
        except (pygame.error, FileNotFoundError) as e:
            print(f"Error loading or converting texture from {image_path}: {e}")
            return None, 0, 0

    def _load_keyboard_assets(self):
        skin_dir = _SKIN_DIR
        try:
            key_type_to_filename = {
                'white_key': 'keyWhite.png',
                'white_pressed': 'keyWhitePressed.png',
                'black_key': 'keyBlack.png',
                'black_pressed': 'keyBlackPressed.png'
            }
            for key_type, filename in key_type_to_filename.items():
                tex_id, width, height = self._load_texture(os.path.join(skin_dir, filename))
                if tex_id is None:
                    raise pygame.error(f"Failed to load {filename}")
                self.keyboard_textures[key_type] = tex_id
                self.keyboard_texture_info[key_type] = {'width': width, 'height': height}
            print("Keyboard skin assets loaded as OpenGL textures.")
        except pygame.error as e:
            print(f"Could not load keyboard skin assets from '{skin_dir}': {e}")
            self.show_keyboard = False
            return

        self.keyboard_layout = {}
        white_key_indices = [0, 2, 4, 5, 7, 9, 11]
        key_is_white = [(i % 12) in white_key_indices for i in range(128)]
        self.is_white_key_data = np.array(key_is_white, dtype=np.int32)
        num_white_keys = sum(key_is_white)

        white_key_width = self.width / num_white_keys
        white_key_height = white_key_width * (self.keyboard_texture_info['white_key']['height'] / self.keyboard_texture_info['white_key']['width'])
        
        self.keyboard_texture_info['white_key']['scaled_height'] = white_key_height

        white_keys_geom = []
        black_keys_geom = []
        self.white_key_pitch_map.clear()
        self.black_key_pitch_map.clear()

        white_key_count = 0
        for pitch in range(128):
            is_white = key_is_white[pitch]
            x_pos = white_key_count * white_key_width
            
            if is_white:
                self.keyboard_layout[pitch] = {'type': 'white', 'x': x_pos, 'width': white_key_width, 'height': white_key_height}
                white_keys_geom.append([x_pos, 0, white_key_width, white_key_height])
                self.white_key_pitch_map[pitch] = len(white_keys_geom) - 1
                white_key_count += 1

        last_white_key_x = (white_key_count - 1) * white_key_width
        width_corr = self.width - (last_white_key_x + white_key_width)
        white_keys_geom[-1][2] += width_corr
        if self.keyboard_layout:
            max_pitch = max(self.keyboard_layout.keys())
            if max_pitch in self.keyboard_layout:
                 self.keyboard_layout[max_pitch]['width'] += width_corr

        black_key_width = white_key_width * 0.6
        black_key_height = white_key_height * 0.65

        for pitch in range(128):
            if not key_is_white[pitch]:
                anchor_pitch = pitch - 1
                while anchor_pitch > 0 and (anchor_pitch not in self.keyboard_layout or self.keyboard_layout[anchor_pitch]['type'] != 'white'):
                    anchor_pitch -= 1
                
                if anchor_pitch in self.keyboard_layout:
                    anchor_key = self.keyboard_layout[anchor_pitch]
                    x_pos = anchor_key['x'] + anchor_key['width'] - (black_key_width / 2)
                    
                    self.keyboard_layout[pitch] = {'type': 'black', 'x': x_pos, 'width': black_key_width, 'height': black_key_height}
                    black_keys_geom.append([x_pos, 0, black_key_width, black_key_height])
                    self.black_key_pitch_map[pitch] = len(black_keys_geom) - 1

        white_key_dtype = np.dtype([('rect', 'f4', 4), ('is_pressed', 'f4'), ('color', 'f4', 3)])
        self.white_key_instance_data = np.zeros(len(white_keys_geom), dtype=white_key_dtype)
        if white_keys_geom:
            self.white_key_instance_data['rect'] = np.array(white_keys_geom, dtype=np.float32)
            self.white_key_instance_data['is_pressed'] = 0.0
            self.white_key_instance_data['color'] = (0.0, 0.0, 0.0)

        black_key_dtype = np.dtype([('rect', 'f4', 4), ('is_pressed', 'f4'), ('color', 'f4', 3)])
        self.black_key_instance_data = np.zeros(len(black_keys_geom), dtype=black_key_dtype)
        if black_keys_geom:
            self.black_key_instance_data['rect'] = np.array(black_keys_geom, dtype=np.float32)
            self.black_key_instance_data['is_pressed'] = 0.0
            self.black_key_instance_data['color'] = (0.0, 0.0, 0.0)

        self.pitch_layout_data = np.zeros((128, 2), dtype=np.float32)
        for pitch, key_info in self.keyboard_layout.items():
            self.pitch_layout_data[pitch, 0] = key_info['x']
            self.pitch_layout_data[pitch, 1] = key_info['width']
            
    def init_pygame_and_gl(self):
        pygame.init()
        pygame.font.init()
        self.overlay_font = pygame.font.Font(None, 18)
        
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | pygame.HWSURFACE)
        pygame.display.set_caption("Piano Roll")
        self._init_slider_geometry()

        try:
            self.note_texture, _, _ = self._load_texture(os.path.join(_SKIN_DIR, "note.png"))
            self.note_edge_texture, _, _ = self._load_texture(os.path.join(_SKIN_DIR, "noteEdge.png"))
            if self.note_texture is None or self.note_edge_texture is None:
                raise pygame.error("Failed to load one or both note textures.")
            print("Note skin assets loaded.")
        except pygame.error as e:
            print(f"Could not load note textures, will fall back to solid colors. Error: {e}")
            self.note_texture = 0
            self.note_edge_texture = 0

        if self.show_keyboard:
            self._load_keyboard_assets()

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        try:
            self.shader = compileProgram(
                compileShader(VERT_SHADER, GL_VERTEX_SHADER),
                compileShader(FRAG_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as err:
            print("Note shader compilation failed:\n", err); raise
        
        if self.show_keyboard:
            try:
                self.keyboard_shader = compileProgram(
                    compileShader(KEYBOARD_VERT_SHADER, GL_VERTEX_SHADER),
                    compileShader(KEYBOARD_FRAG_SHADER, GL_FRAGMENT_SHADER)
                )
            except Exception as err:
                print("Keyboard shader compilation failed:\n", err); self.show_keyboard = False

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        pos_loc = glGetAttribLocation(self.shader, "pos")
        note_times_loc = glGetAttribLocation(self.shader, "note_times")
        note_info_loc = glGetAttribLocation(self.shader, "note_info")

        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        note_vertices = np.array([[0,0], [1,0], [1,1], [1,1], [0,1], [0,0]], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, note_vertices.nbytes, note_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

        self.vbo_stream_data = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
        self.note_size_bytes = GPU_NOTE_DTYPE.itemsize
        
        glBufferData(GL_ARRAY_BUFFER, self.streaming_vbo_capacity * self.note_size_bytes, None, GL_DYNAMIC_DRAW)
        
        glEnableVertexAttribArray(note_times_loc); glVertexAttribPointer(note_times_loc, 2, GL_FLOAT, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(0)); glVertexAttribDivisor(note_times_loc, 1)
        
        glEnableVertexAttribArray(note_info_loc)
        glVertexAttribPointer(note_info_loc, 3, GL_UNSIGNED_BYTE, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(8))
        glVertexAttribDivisor(note_info_loc, 1)

        if self.show_keyboard:
            glUseProgram(self.keyboard_shader)
            kb_pos_loc = glGetAttribLocation(self.keyboard_shader, "pos")
            kb_rect_loc = glGetAttribLocation(self.keyboard_shader, "instance_rect")
            kb_pressed_loc = glGetAttribLocation(self.keyboard_shader, "instance_is_pressed")
            kb_color_loc = glGetAttribLocation(self.keyboard_shader, "instance_color")
            
            self.keyboard_vbo_quad = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_quad)
            quad_vertices = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

            instance_stride = self.white_key_instance_data.strides[0]
            
            self.keyboard_vao_white = glGenVertexArrays(1)
            glBindVertexArray(self.keyboard_vao_white)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_quad)
            glEnableVertexAttribArray(kb_pos_loc); glVertexAttribPointer(kb_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
            
            self.keyboard_vbo_white_keys = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_white_keys)
            glBufferData(GL_ARRAY_BUFFER, self.white_key_instance_data.nbytes, self.white_key_instance_data, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(kb_rect_loc); glVertexAttribPointer(kb_rect_loc, 4, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(0)); glVertexAttribDivisor(kb_rect_loc, 1)
            glEnableVertexAttribArray(kb_pressed_loc); glVertexAttribPointer(kb_pressed_loc, 1, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(16)); glVertexAttribDivisor(kb_pressed_loc, 1)
            glEnableVertexAttribArray(kb_color_loc); glVertexAttribPointer(kb_color_loc, 3, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(20)); glVertexAttribDivisor(kb_color_loc, 1)
            
            self.keyboard_vao_black = glGenVertexArrays(1)
            glBindVertexArray(self.keyboard_vao_black)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_quad)
            glEnableVertexAttribArray(kb_pos_loc); glVertexAttribPointer(kb_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

            self.keyboard_vbo_black_keys = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_black_keys)
            glBufferData(GL_ARRAY_BUFFER, self.black_key_instance_data.nbytes, self.black_key_instance_data, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(kb_rect_loc); glVertexAttribPointer(kb_rect_loc, 4, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(0)); glVertexAttribDivisor(kb_rect_loc, 1)
            glEnableVertexAttribArray(kb_pressed_loc); glVertexAttribPointer(kb_pressed_loc, 1, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(16)); glVertexAttribDivisor(kb_pressed_loc, 1)
            glEnableVertexAttribArray(kb_color_loc); glVertexAttribPointer(kb_color_loc, 3, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(20)); glVertexAttribDivisor(kb_color_loc, 1)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            glUseProgram(0)

        glUseProgram(self.shader)
        self.projection_matrix = np.array([
            [2/self.width, 0, 0, -1],
            [0, -2/self.height, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
        self.u_time_loc = glGetUniformLocation(self.shader, "u_time")
        self.u_scroll_speed_loc = glGetUniformLocation(self.shader, "u_scroll_speed")
        self.u_colors_loc = glGetUniformLocation(self.shader, "u_colors")
        self.u_pitch_layout_loc = glGetUniformLocation(self.shader, "u_pitch_layout")
        self.u_is_white_key_notes_loc = glGetUniformLocation(self.shader, "u_is_white_key")
        self.u_window_start_loc = glGetUniformLocation(self.shader, "u_window_start")
        self.u_window_end_loc = glGetUniformLocation(self.shader, "u_window_end")
        
        glUniform1f(glGetUniformLocation(self.shader, "u_scroll_speed"), self.scroll_speed)
        glUniform1f(glGetUniformLocation(self.shader, "u_width"), float(self.width))
        glUniform1f(glGetUniformLocation(self.shader, "u_height"), float(self.height))
        glUniform1f(glGetUniformLocation(self.shader, "u_guide_line_y"), self._get_guide_line_y())
        glUniform2fv(self.u_pitch_layout_loc, 128, self.pitch_layout_data)
        
        if self.u_is_white_key_notes_loc != -1 and self.is_white_key_data is not None:
            glUniform1iv(self.u_is_white_key_notes_loc, 128, self.is_white_key_data)
        if self.u_window_start_loc != -1:
            glUniform1f(self.u_window_start_loc, 0.0)
        if self.u_window_end_loc != -1:
            glUniform1f(self.u_window_end_loc, 0.0)

        self.u_note_texture_loc = glGetUniformLocation(self.shader, "u_note_texture")
        self.u_note_edge_texture_loc = glGetUniformLocation(self.shader, "u_note_edge_texture")
        self.u_glow_strength_loc = glGetUniformLocation(self.shader, "u_glow_strength")
        glUniform1i(self.u_note_texture_loc, 0)
        glUniform1i(self.u_note_edge_texture_loc, 1)
        if self.u_glow_strength_loc != -1:
            glUniform1f(self.u_glow_strength_loc, self.glow_strength)

        self.channel_colors = self._load_colors_from_xml()
        glUniform3fv(self.u_colors_loc, 128, self.channel_colors)
        
        if self.show_keyboard:
            glUseProgram(self.keyboard_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.keyboard_shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
            glUniform1i(glGetUniformLocation(self.keyboard_shader, "u_texture_unpressed"), 0)
            glUniform1i(glGetUniformLocation(self.keyboard_shader, "u_texture_pressed"), 1)
            self.u_is_white_key_loc = glGetUniformLocation(self.keyboard_shader, "u_is_white_key")
            keyboard_glow_loc = glGetUniformLocation(self.keyboard_shader, "u_glow_strength")
            if keyboard_glow_loc != -1:
                glUniform1f(keyboard_glow_loc, self.glow_strength)

        glUseProgram(0)
        
        self.data_thread = threading.Thread(target=self._data_streamer_thread, daemon=True)
        self.data_thread.start()
        
        margin = 10
        self.color_button_rect = pygame.Rect(
            self.width - self.color_button_size - margin,
            margin,
            self.color_button_size,
            self.color_button_size
        )
        self.glow_button_rect = pygame.Rect(
            self.color_button_rect.x - self.color_button_size - 8,
            margin,
            self.color_button_size,
            self.color_button_size
        )
        self.glow_options_button_rect = pygame.Rect(
            self.glow_button_rect.x,
            self.glow_button_rect.y + self.color_button_size + 6,
            self.color_button_size,
            18
        )
        panel_width = 190
        panel_height = 82
        self.glow_options_panel_rect = pygame.Rect(
            self.glow_options_button_rect.x - (panel_width - self.glow_options_button_rect.width),
            self.glow_options_button_rect.y + self.glow_options_button_rect.height + 6,
            panel_width,
            panel_height
        )
        self.glow_options_checkbox_rect = pygame.Rect(
            self.glow_options_panel_rect.x + 10,
            self.glow_options_panel_rect.y + 28,
            16,
            16
        )
        self.key_light_fade_checkbox_rect = pygame.Rect(
            self.glow_options_panel_rect.x + 10,
            self.glow_options_panel_rect.y + 52,
            16,
            16
        )

    def load_midi(self, all_notes_gpu, get_current_time_func):
        """Prepare note data for the render thread."""
        self.get_current_time = get_current_time_func
        self.glow_trails.clear()
        self.last_glow_time = None
        render_sort_idx = np.argsort(all_notes_gpu['on_time'], kind='stable')
        render_notes_array = np.ascontiguousarray(all_notes_gpu[render_sort_idx])
        render_on_times = render_notes_array['on_time']
        durations = render_notes_array['off_time'] - render_notes_array['on_time']
        if len(durations) > 0:
            self.max_note_duration = float(np.max(durations))
        with self.midi_data_lock:
            self.pending_midi_data = {
                'all_notes_gpu': all_notes_gpu,
                'render_notes_array': render_notes_array,
                'render_on_times': render_on_times
            }
        
        print(f"MIDI data prepared: {len(all_notes_gpu)} notes. Max Duration: {self.max_note_duration:.2f}s")
    
    def _upload_pending_midi_data(self):
        """Swap in note data prepared by the worker thread."""
        with self.midi_data_lock:
            if self.pending_midi_data is None:
                return
            
            data = self.pending_midi_data
            self.pending_midi_data = None
        
        self.all_notes_gpu = data['all_notes_gpu']
        self.render_notes_array = data['render_notes_array']
        self.render_on_times = data['render_on_times']
        
        self.notes_to_draw = len(self.render_notes_array)
        
        self.force_data_update.set()
        print("MIDI data active on GPU thread.")
    
    def _data_streamer_thread(self):
        """Background thread for slicing visible notes."""
        last_update_time = -1.0
        
        while self.app_running.is_set():
            if self.get_current_time is None or self.render_notes_array is None:
                time.sleep(0.1)
                continue

            try:
                now = self.get_current_time()
            except:
                time.sleep(0.1)
                continue

            if abs(now - last_update_time) > self.data_update_interval or self.force_data_update.is_set():
                self.force_data_update.clear()
                last_update_time = now
                
                view_start = now - self.seconds_before_cursor
                view_end = now + self.seconds_after_cursor
                search_start = view_start - self.max_note_duration
                start_idx = np.searchsorted(self.render_on_times, search_start, side='left')
                end_idx = np.searchsorted(self.render_on_times, view_end, side='right')
                candidates = self.render_notes_array[start_idx:end_idx]
                
                if len(candidates) > 0:
                    mask = candidates['off_time'] > view_start
                    visible_slice = candidates[mask]
                    visible_count = len(visible_slice)
                else:
                    visible_slice = candidates
                    visible_count = 0
                
                if visible_count > self.streaming_vbo_capacity:
                    visible_slice = visible_slice[:self.streaming_vbo_capacity]
                    visible_count = self.streaming_vbo_capacity
                
                if visible_count > 0:
                    visible_slice_contiguous = np.ascontiguousarray(visible_slice)
                    
                    try:
                        self.data_queue.put((visible_slice_contiguous, visible_count), block=True, timeout=0.1)
                    except queue.Full:
                        print("Piano roll queue is full, frame skipped. This may indicate rendering lag.")
            
            time.sleep(0.005) # Yield

    def draw(self, current_time):
        """Draw the latest buffer provided by the data thread."""
        self._upload_pending_midi_data()
        if self.show_key_press_glow or self.show_key_light_fade:
            self._update_glow_trails(current_time)
        else:
            self.glow_trails.clear()

        glClearColor(0.05, 0.05, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        try:
            queue_data = self.data_queue.get_nowait()
            visible_notes, count = queue_data
            self.last_visible_notes = visible_notes
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            glBufferData(GL_ARRAY_BUFFER, self.streaming_vbo_capacity * self.note_size_bytes, None, GL_DYNAMIC_DRAW)
            glBufferSubData(GL_ARRAY_BUFFER, 0, visible_notes.nbytes, visible_notes)
            
            self.notes_to_draw = count
            
        except queue.Empty:
            pass

        if self.notes_to_draw > 0 and self.render_notes_array is not None:
            window_start = current_time - self.seconds_before_cursor
            window_end = current_time + self.seconds_after_cursor
            
            glEnable(GL_DEPTH_TEST)
            glUseProgram(self.shader)
            
            glUniform1f(self.u_time_loc, current_time)
            glUniform1f(self.u_scroll_speed_loc, self.scroll_speed)
            glUniform1f(glGetUniformLocation(self.shader, "u_guide_line_y"), self._get_guide_line_y())
            glUniform1f(self.u_window_start_loc, window_start)
            glUniform1f(self.u_window_end_loc, window_end)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.note_texture)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.note_edge_texture)

            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self.notes_to_draw)
            
            glBindVertexArray(0)
            glUseProgram(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        if self.show_glow and self.show_key_press_glow:
            self._draw_active_note_glow_overlay(current_time)

        if self.show_keyboard and self.keyboard_layout:
            self._draw_keyboard_opengl(current_time)

        if self.show_guide_line:
            glDisable(GL_DEPTH_TEST)
            glUseProgram(0)
            glMatrixMode(GL_PROJECTION);glPushMatrix();glLoadIdentity();glOrtho(0,self.width,self.height,0,-1,1)
            glMatrixMode(GL_MODELVIEW);glPushMatrix();glLoadIdentity()
            glLineWidth(2.0);glColor3f(0.8,0.0,0.0)
            guide_y = self._get_guide_line_y()
            glBegin(GL_LINES);glVertex2f(0, guide_y);glVertex2f(self.width, guide_y);glEnd()
            glPopMatrix();glMatrixMode(GL_PROJECTION);glPopMatrix();glMatrixMode(GL_MODELVIEW)

        self._draw_slider_overlay()

        pygame.display.flip()

    def _update_glow_trails(self, current_time):
        if self.last_glow_time is not None and current_time + 0.001 < self.last_glow_time:
            self.glow_trails.clear()
        self.last_glow_time = current_time

        active_pitch_data = {}
        if self.last_visible_notes is not None and len(self.last_visible_notes) > 0:
            active_mask = (self.last_visible_notes['on_time'] <= current_time) & (self.last_visible_notes['off_time'] > current_time)
            active_notes = self.last_visible_notes[active_mask]
            for note in active_notes:
                pitch = int(note['pitch'])
                track = int(note['track'])
                color = self.channel_colors[track % 128]
                if pitch not in active_pitch_data:
                    active_pitch_data[pitch] = {
                        'pitch': pitch,
                        'color_sum': np.array(color, dtype=np.float32),
                        'count': 1,
                    }
                else:
                    active_pitch_data[pitch]['color_sum'] += color
                    active_pitch_data[pitch]['count'] += 1

            for pitch, pitch_data in active_pitch_data.items():
                avg_color = pitch_data['color_sum'] / max(1, pitch_data['count'])
                self.glow_trails[pitch] = {
                    'pitch': pitch,
                    'color': avg_color,
                    'live_count': pitch_data['count'],
                    'last_active_time': current_time,
                }

        expired_keys = []
        for key, trail in list(self.glow_trails.items()):
            elapsed = max(0.0, current_time - trail['last_active_time'])
            fade = 1.0 - (elapsed / self.glow_fade_duration)
            if fade <= 0.0:
                expired_keys.append(key)

        for key in expired_keys:
            self.glow_trails.pop(key, None)

    def _draw_active_note_glow_overlay(self, current_time):
        if not self.glow_trails:
            return

        guide_y = self._get_guide_line_y()
        active_pitch_keys = set()
        if self.last_visible_notes is not None and len(self.last_visible_notes) > 0:
            active_mask = (self.last_visible_notes['on_time'] <= current_time) & (self.last_visible_notes['off_time'] > current_time)
            active_notes = self.last_visible_notes[active_mask]
            for note in active_notes:
                active_pitch_keys.add(int(note['pitch']))

        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        for trail in self.glow_trails.values():
            elapsed = max(0.0, current_time - trail['last_active_time'])
            fade = 1.0 - (elapsed / self.glow_fade_duration)
            if fade <= 0.0:
                continue

            pitch = trail['pitch']
            key_info = self.keyboard_layout.get(pitch)
            if not key_info:
                continue

            color = trail['color']
            center_x = key_info['x'] + key_info['width'] * 0.5
            center_y = guide_y - 2.0
            radius_x = max(key_info['width'] * 1.85, 14.0)
            radius_y = 16.0 if key_info['type'] == 'black' else 22.0
            live_boost = 1.18 if pitch in active_pitch_keys else 1.0
            repeat_boost = min(1.35, 1.0 + 0.14 * max(0, int(trail.get('live_count', 1)) - 1))
            alpha_boost = live_boost * repeat_boost

            glBegin(GL_TRIANGLE_FAN)
            glColor4f(color[0], color[1], color[2], 0.52 * fade * alpha_boost)
            glVertex2f(center_x, center_y)
            glColor4f(color[0], color[1], color[2], 0.0)
            for i in range(33):
                angle = (2.0 * np.pi * i) / 32.0
                glVertex2f(center_x + np.cos(angle) * radius_x, center_y + np.sin(angle) * radius_y)
            glEnd()

            glBegin(GL_TRIANGLE_FAN)
            glColor4f(color[0], color[1], color[2], 0.30 * fade * alpha_boost)
            glVertex2f(center_x, center_y - 8.0)
            glColor4f(color[0], color[1], color[2], 0.0)
            for i in range(33):
                angle = (2.0 * np.pi * i) / 32.0
                glVertex2f(center_x + np.cos(angle) * radius_x * 1.25, center_y - 8.0 + np.sin(angle) * radius_y * 1.65)
            glEnd()

        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _get_trail_fade(self, current_time, pitch):
        trail = self.glow_trails.get(pitch)
        if not trail:
            return None, 0.0
        elapsed = max(0.0, current_time - trail['last_active_time'])
        fade = 1.0 - (elapsed / self.glow_fade_duration)
        if fade <= 0.0:
            return None, 0.0
        return trail, fade

    def _draw_keyboard_opengl(self, current_time):
        """Draw the keyboard overlay using the latest visible-note slice."""
        if self.keyboard_shader == 0:
            return

        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.keyboard_shader)

        keyboard_height = self.keyboard_texture_info['white_key']['scaled_height']
        keyboard_y = self.height - keyboard_height
        glUniform1f(glGetUniformLocation(self.keyboard_shader, "u_keyboard_y"), keyboard_y)

        if len(self.white_key_instance_data) > 0:
            self.white_key_instance_data['is_pressed'].fill(0.0)
            self.white_key_instance_data['color'][:] = 0.0
        if len(self.black_key_instance_data) > 0:
            self.black_key_instance_data['is_pressed'].fill(0.0)
            self.black_key_instance_data['color'][:] = 0.0

        current_active_pitches = set()
        active_pitch_colors = {}
        if self.last_visible_notes is not None and len(self.last_visible_notes) > 0:
            active_mask = (self.last_visible_notes['on_time'] <= current_time) & (self.last_visible_notes['off_time'] > current_time)
            active_notes = self.last_visible_notes[active_mask]
            for note in active_notes:
                pitch = int(note['pitch'])
                current_active_pitches.add(pitch)
                track = int(note['track'])
                color = np.array(self.channel_colors[track % 128], dtype=np.float32)
                on_time = float(note['on_time'])
                existing = active_pitch_colors.get(pitch)
                if existing is None or on_time >= existing['on_time']:
                    active_pitch_colors[pitch] = {
                        'color': color,
                        'on_time': on_time,
                    }

        if self.show_key_light_fade:
            for pitch, trail in self.glow_trails.items():
                if pitch in active_pitch_colors:
                    continue
                trail_obj, fade = self._get_trail_fade(current_time, pitch)
                if trail_obj is None or fade <= 0.0:
                    continue
                active_pitch_colors[pitch] = {
                    'color': np.array(trail_obj['color'], dtype=np.float32) * fade,
                    'on_time': -1.0,
                }

        key_light_colors = {}
        light_falloff = (1.0, 0.88, 0.68, 0.50, 0.35, 0.23, 0.14, 0.08)
        for pitch, pitch_color_data in active_pitch_colors.items():
            base_color = pitch_color_data['color']
            luminance_boost = 0.88 + self._color_luminance(base_color) * 0.45
            for distance, weight in enumerate(light_falloff[1:], start=1):
                neighbors = (pitch - distance, pitch + distance)
                for neighbor in neighbors:
                    if neighbor < 0 or neighbor > 127 or neighbor not in self.keyboard_layout:
                        continue
                    effective_weight = min(1.0, weight * luminance_boost)
                    weighted_color = np.array(base_color * effective_weight, dtype=np.float32)
                    existing = key_light_colors.get(neighbor)
                    if self._should_replace_spill_color(
                        existing,
                        effective_weight,
                        distance,
                        self._color_luminance(base_color),
                    ):
                        key_light_colors[neighbor] = {
                            'color': weighted_color,
                            'weight': effective_weight,
                            'distance': distance,
                            'luminance': self._color_luminance(base_color),
                        }

        for pitch, color_data in key_light_colors.items():
            clamped_color = np.clip(color_data['color'], 0.0, 1.0).astype(np.float32)
            if pitch in self.white_key_pitch_map:
                idx = self.white_key_pitch_map[pitch]
                self.white_key_instance_data[idx]['color'] = clamped_color
            elif pitch in self.black_key_pitch_map:
                idx = self.black_key_pitch_map[pitch]
                self.black_key_instance_data[idx]['color'] = clamped_color

        for pitch, pitch_color_data in active_pitch_colors.items():
            active_color = np.clip(pitch_color_data['color'], 0.0, 1.0).astype(np.float32)
            if pitch in self.white_key_pitch_map:
                idx = self.white_key_pitch_map[pitch]
                self.white_key_instance_data[idx]['color'] = active_color
                self.white_key_instance_data[idx]['is_pressed'] = 1.0
            elif pitch in self.black_key_pitch_map:
                idx = self.black_key_pitch_map[pitch]
                self.black_key_instance_data[idx]['color'] = active_color
                self.black_key_instance_data[idx]['is_pressed'] = 1.0
        
        self.active_pitches_last_frame = current_active_pitches

        glUniform1i(self.u_is_white_key_loc, 1)
        
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.keyboard_textures['white_key'])
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, self.keyboard_textures['white_pressed'])
        glBindVertexArray(self.keyboard_vao_white)
        glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_white_keys)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.white_key_instance_data.nbytes, self.white_key_instance_data)
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, len(self.white_key_instance_data))

        if len(self.black_key_instance_data) > 0:
            glUniform1i(self.u_is_white_key_loc, 0)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.keyboard_textures['black_key'])
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, self.keyboard_textures['black_pressed'])
            glBindVertexArray(self.keyboard_vao_black)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_black_keys)
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.black_key_instance_data.nbytes, self.black_key_instance_data)
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, len(self.black_key_instance_data))
        
        glBindVertexArray(0)
        glUseProgram(0)

    def _init_slider_geometry(self):
        padding = 16
        panel_width, panel_height = self.controls_panel_size
        panel_x = padding - 4
        panel_y = padding - 2
        self.controls_toggle_rect = pygame.Rect(panel_x, panel_y, 34, 34)
        self.controls_panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        self.controls_close_rect = pygame.Rect(panel_x + panel_width - 34 - 8, panel_y + 8, 34, 34)

        bar_width = panel_width - 32
        bar_height = 6
        x = panel_x + 16
        y = panel_y + 42
        self.slider_rect = pygame.Rect(x, y, bar_width, self.slider_area_height)
        self.slider_bar = (x, y + (self.slider_area_height - bar_height) // 2, bar_width, bar_height)
        self._recalc_slider_handle()
        scroll_y = y + self.slider_area_height + 18
        self.scroll_slider_rect = pygame.Rect(x, scroll_y, bar_width, self.slider_area_height)
        self.scroll_slider_bar = (x, scroll_y + (self.slider_area_height - bar_height) // 2, bar_width, bar_height)
        self._recalc_scroll_slider_handle()

    def _recalc_slider_handle(self):
        if not self.slider_rect: return
        t = (self.window_seconds - self.slider_min) / (self.slider_max - self.slider_min)
        t = max(0.0, min(1.0, t))
        self.slider_handle_x = self.slider_rect.x + t * self.slider_rect.width

    def _recalc_scroll_slider_handle(self):
        if not self.scroll_slider_rect: return
        t = (self.scroll_speed - self.scroll_slider_min) / (self.scroll_slider_max - self.scroll_slider_min)
        t = max(0.0, min(1.0, t))
        self.scroll_slider_handle_x = self.scroll_slider_rect.x + t * self.scroll_slider_rect.width

    def handle_slider_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if (not self.controls_panel_expanded) and self.controls_toggle_rect and self.controls_toggle_rect.collidepoint(event.pos):
                self.controls_panel_expanded = not self.controls_panel_expanded
                self.slider_dragging = False
                self.scroll_slider_dragging = False
                return
            if self.controls_panel_expanded and self.controls_close_rect and self.controls_close_rect.collidepoint(event.pos):
                self.controls_panel_expanded = False
                self.slider_dragging = False
                self.scroll_slider_dragging = False
                return
            if self.glow_options_button_rect and self.glow_options_button_rect.collidepoint(event.pos):
                self.glow_options_expanded = not self.glow_options_expanded
                return
            if self.glow_options_expanded and self.glow_options_checkbox_rect and self.glow_options_checkbox_rect.collidepoint(event.pos):
                self.show_key_press_glow = not self.show_key_press_glow
                if not self.show_key_press_glow and not self.show_key_light_fade:
                    self.glow_trails.clear()
                self._save_visualizer_config()
                return
            if self.glow_options_expanded and self.key_light_fade_checkbox_rect and self.key_light_fade_checkbox_rect.collidepoint(event.pos):
                self.show_key_light_fade = not self.show_key_light_fade
                if not self.show_key_press_glow and not self.show_key_light_fade:
                    self.glow_trails.clear()
                self._save_visualizer_config()
                return
            if self.glow_button_rect and self.glow_button_rect.collidepoint(event.pos):
                self.toggle_glow()
                return
            if self.color_button_rect and self.color_button_rect.collidepoint(event.pos):
                self.randomize_colors()
                return
            if self.controls_panel_expanded and self.slider_rect and self.slider_rect.collidepoint(event.pos):
                self.slider_dragging = True
                self._update_slider_from_pos(event.pos[0])
            if self.controls_panel_expanded and self.scroll_slider_rect and self.scroll_slider_rect.collidepoint(event.pos):
                self.scroll_slider_dragging = True
                self._update_scroll_slider_from_pos(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.slider_dragging = False
            self.scroll_slider_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.slider_dragging:
            self._update_slider_from_pos(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and self.scroll_slider_dragging:
            self._update_scroll_slider_from_pos(event.pos[0])
    
    def randomize_colors(self):
        """Shuffle the track colors randomly and update the shader."""
        indices = list(range(128))
        random.shuffle(indices)
        self.channel_colors = self.channel_colors[indices]
        glUseProgram(self.shader)
        glUniform3fv(self.u_colors_loc, 128, self.channel_colors)
        glUseProgram(0)
        print("Colors randomized!")

    def toggle_glow(self):
        self.show_glow = not self.show_glow
        self.glow_strength = 1.0 if self.show_glow else 0.0

        glUseProgram(self.shader)
        if self.u_glow_strength_loc != -1:
            glUniform1f(self.u_glow_strength_loc, self.glow_strength)
        glUseProgram(0)

        if self.show_keyboard and self.keyboard_shader:
            glUseProgram(self.keyboard_shader)
            keyboard_glow_loc = glGetUniformLocation(self.keyboard_shader, "u_glow_strength")
            if keyboard_glow_loc != -1:
                glUniform1f(keyboard_glow_loc, self.glow_strength)
            glUseProgram(0)

        self._save_visualizer_config()
        print(f"Piano roll glow {'enabled' if self.show_glow else 'disabled'}.")

    def _update_slider_from_pos(self, x_pos):
        t = (x_pos - self.slider_rect.x) / float(self.slider_rect.width)
        t = max(0.0, min(1.0, t))
        new_val = self.slider_min + t * (self.slider_max - self.slider_min)
        self.window_seconds = new_val
        self.seconds_before_cursor = new_val
        self.seconds_after_cursor = new_val
        self._recalc_slider_handle()
        self._save_visualizer_config()
        self.force_data_update.set() # Force update when slider moves

    def _update_scroll_slider_from_pos(self, x_pos):
        t = (x_pos - self.scroll_slider_rect.x) / float(self.scroll_slider_rect.width)
        t = max(0.0, min(1.0, t))
        new_val = self.scroll_slider_min + t * (self.scroll_slider_max - self.scroll_slider_min)
        self.scroll_speed = new_val
        self._recalc_scroll_slider_handle()
        self._save_visualizer_config()

    def _get_text_texture(self, text, color=(225, 228, 235)):
        if self.overlay_font is None:
            return None
        key = (text, color)
        if key in self._text_texture_cache:
            return self._text_texture_cache[key]

        surface = self.overlay_font.render(text, True, color)
        width, height = surface.get_size()
        image_data = pygame.image.tostring(surface, "RGBA", True)
        self._text_texture_cache[key] = (image_data, width, height)
        return self._text_texture_cache[key]

    def _draw_text_overlay(self, text, x, y, color=(225, 228, 235)):
        tex_info = self._get_text_texture(text, color)
        if tex_info is None:
            return
        pixel_data, width, height = tex_info
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glWindowPos2i(int(x), int(self.height - y - height))
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data)

    def _draw_slider_overlay(self):
        if not self.slider_rect:
            return
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        self._update_hover_ui_state()
        handle_half = 6
        if self.controls_panel_expanded and self.controls_panel_rect:
            px, py, pw, ph = self.controls_panel_rect
            glColor4f(0.08, 0.08, 0.11, 0.88)
            glBegin(GL_QUADS)
            glVertex2f(px, py); glVertex2f(px + pw, py)
            glVertex2f(px + pw, py + ph); glVertex2f(px, py + ph)
            glEnd()
            glColor4f(0.62, 0.64, 0.70, 0.95)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(px, py); glVertex2f(px + pw, py)
            glVertex2f(px + pw, py + ph); glVertex2f(px, py + ph)
            glEnd()

        if (not self.controls_panel_expanded) and self.controls_toggle_rect:
            bx, by = self.controls_toggle_rect.x, self.controls_toggle_rect.y
            bs = self.controls_toggle_rect.width
            glColor4f(0.10, 0.10, 0.14, 0.82)
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            glColor4f(0.78, 0.80, 0.86, 0.95)
            glLineWidth(2.0)
            for i in range(3):
                ly = by + 9 + i * 7
                glBegin(GL_LINES)
                glVertex2f(bx + 8, ly)
                glVertex2f(bx + bs - 8, ly)
                glEnd()
            glColor4f(0.35, 0.38, 0.44, 0.9)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            self._draw_hover_highlight(self.controls_toggle_rect, self._get_hover_alpha("controls_toggle"))

        if self.controls_panel_expanded and self.controls_panel_rect:
            if self.controls_close_rect:
                bx, by = self.controls_close_rect.x, self.controls_close_rect.y
                bs = self.controls_close_rect.width
                glColor4f(0.10, 0.10, 0.14, 0.82)
                glBegin(GL_QUADS)
                glVertex2f(bx, by); glVertex2f(bx + bs, by)
                glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
                glEnd()
                glColor4f(0.86, 0.86, 0.90, 0.98)
                glLineWidth(2.0)
                glBegin(GL_LINES)
                glVertex2f(bx + 9, by + 9); glVertex2f(bx + bs - 9, by + bs - 9)
                glVertex2f(bx + bs - 9, by + 9); glVertex2f(bx + 9, by + bs - 9)
                glEnd()
                glColor4f(0.35, 0.38, 0.44, 0.9)
                glLineWidth(1.0)
                glBegin(GL_LINE_LOOP)
                glVertex2f(bx, by); glVertex2f(bx + bs, by)
                glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
                glEnd()
                self._draw_hover_highlight(self.controls_close_rect, self._get_hover_alpha("controls_close"))

            self._draw_text_overlay("Note Length", self.slider_rect.x, self.slider_rect.y - 12)
            self._draw_text_overlay("Scroll Speed", self.scroll_slider_rect.x, self.scroll_slider_rect.y - 12)

            x, y, w, h = self.slider_bar
            glColor3f(0.2, 0.2, 0.25)
            glBegin(GL_QUADS); glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h); glEnd()
            filled_w = (self.slider_handle_x - x)
            glColor3f(0.4, 0.7, 0.9)
            glBegin(GL_QUADS); glVertex2f(x, y); glVertex2f(x + filled_w, y); glVertex2f(x + filled_w, y + h); glVertex2f(x, y + h); glEnd()
            hx = self.slider_handle_x
            hy = y + h / 2
            glColor3f(0.9, 0.9, 0.95)
            glBegin(GL_QUADS); glVertex2f(hx - handle_half, hy - handle_half); glVertex2f(hx + handle_half, hy - handle_half); glVertex2f(hx + handle_half, hy + handle_half); glVertex2f(hx - handle_half, hy + handle_half); glEnd()

            if self.scroll_slider_bar:
                x2, y2, w2, h2 = self.scroll_slider_bar
                glColor3f(0.22, 0.22, 0.28)
                glBegin(GL_QUADS); glVertex2f(x2, y2); glVertex2f(x2 + w2, y2); glVertex2f(x2 + w2, y2 + h2); glVertex2f(x2, y2 + h2); glEnd()
                filled_w2 = (self.scroll_slider_handle_x - x2)
                glColor3f(0.5, 0.8, 0.5)
                glBegin(GL_QUADS); glVertex2f(x2, y2); glVertex2f(x2 + filled_w2, y2); glVertex2f(x2 + filled_w2, y2 + h2); glVertex2f(x2, y2 + h2); glEnd()
                hx2 = self.scroll_slider_handle_x
                hy2 = y2 + h2 / 2
                glColor3f(0.95, 0.95, 0.95)
                glBegin(GL_QUADS); glVertex2f(hx2 - handle_half, hy2 - handle_half); glVertex2f(hx2 + handle_half, hy2 - handle_half); glVertex2f(hx2 + handle_half, hy2 + handle_half); glVertex2f(hx2 - handle_half, hy2 + handle_half); glEnd()

        if self.color_button_rect:
            bx, by = self.color_button_rect.x, self.color_button_rect.y
            bs = self.color_button_size
            glColor4f(0.15, 0.15, 0.2, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            
            bar_margin = 6
            bar_height = (bs - 4 * bar_margin) / 3
            bar_width = bs - 2 * bar_margin
            glColor4f(1.0, 0.3, 0.3, 0.9)
            ry = by + bar_margin
            glBegin(GL_QUADS)
            glVertex2f(bx + bar_margin, ry)
            glVertex2f(bx + bar_margin + bar_width, ry)
            glVertex2f(bx + bar_margin + bar_width, ry + bar_height)
            glVertex2f(bx + bar_margin, ry + bar_height)
            glEnd()
            
            glColor4f(0.3, 1.0, 0.3, 0.9)
            gy = by + bar_margin * 2 + bar_height
            glBegin(GL_QUADS)
            glVertex2f(bx + bar_margin, gy)
            glVertex2f(bx + bar_margin + bar_width, gy)
            glVertex2f(bx + bar_margin + bar_width, gy + bar_height)
            glVertex2f(bx + bar_margin, gy + bar_height)
            glEnd()
            
            glColor4f(0.3, 0.3, 1.0, 0.9)
            bby = by + bar_margin * 3 + bar_height * 2
            glBegin(GL_QUADS)
            glVertex2f(bx + bar_margin, bby)
            glVertex2f(bx + bar_margin + bar_width, bby)
            glVertex2f(bx + bar_margin + bar_width, bby + bar_height)
            glVertex2f(bx + bar_margin, bby + bar_height)
            glEnd()
            
            glColor4f(0.5, 0.5, 0.55, 0.8)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            self._draw_hover_highlight(self.color_button_rect, self._get_hover_alpha("color_button"))

        if self.glow_button_rect:
            bx, by = self.glow_button_rect.x, self.glow_button_rect.y
            bs = self.color_button_size
            if self.show_glow:
                glColor4f(0.95, 0.78, 0.24, 0.92)
            else:
                glColor4f(0.15, 0.15, 0.2, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()

            cx = bx + bs * 0.5
            cy = by + bs * 0.5
            radius = bs * 0.16
            glColor4f(1.0, 0.98, 0.85, 1.0)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(17):
                angle = (2.0 * np.pi * i) / 16.0
                glVertex2f(cx + np.cos(angle) * radius, cy + np.sin(angle) * radius)
            glEnd()

            glLineWidth(2.0)
            glBegin(GL_LINES)
            for i in range(8):
                angle = (2.0 * np.pi * i) / 8.0
                inner = radius + 3.0
                outer = radius + 8.0
                glVertex2f(cx + np.cos(angle) * inner, cy + np.sin(angle) * inner)
                glVertex2f(cx + np.cos(angle) * outer, cy + np.sin(angle) * outer)
            glEnd()

            glColor4f(0.5, 0.5, 0.55, 0.8)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            self._draw_hover_highlight(self.glow_button_rect, self._get_hover_alpha("glow_button"))

        if self.glow_options_button_rect:
            bx, by = self.glow_options_button_rect.x, self.glow_options_button_rect.y
            bw, bh = self.glow_options_button_rect.width, self.glow_options_button_rect.height
            glColor4f(0.12, 0.12, 0.16, 0.82)
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bw, by)
            glVertex2f(bx + bw, by + bh); glVertex2f(bx, by + bh)
            glEnd()
            glColor4f(0.78, 0.80, 0.86, 0.95)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex2f(bx + 8, by + 6); glVertex2f(bx + bw * 0.5, by + bh - 5)
            glVertex2f(bx + bw - 8, by + 6); glVertex2f(bx + bw * 0.5, by + bh - 5)
            glEnd()
            glColor4f(0.5, 0.5, 0.55, 0.8)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, by); glVertex2f(bx + bw, by)
            glVertex2f(bx + bw, by + bh); glVertex2f(bx, by + bh)
            glEnd()
            self._draw_hover_highlight(self.glow_options_button_rect, self._get_hover_alpha("glow_options"))

        if self.glow_options_expanded and self.glow_options_panel_rect:
            px, py, pw, ph = self.glow_options_panel_rect
            glColor4f(0.08, 0.08, 0.11, 0.92)
            glBegin(GL_QUADS)
            glVertex2f(px, py); glVertex2f(px + pw, py)
            glVertex2f(px + pw, py + ph); glVertex2f(px, py + ph)
            glEnd()
            glColor4f(0.62, 0.64, 0.70, 0.95)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(px, py); glVertex2f(px + pw, py)
            glVertex2f(px + pw, py + ph); glVertex2f(px, py + ph)
            glEnd()

            self._draw_text_overlay("Glow on key press", px + 32, py + 30)
            self._draw_text_overlay("Fade key lighting", px + 32, py + 54)

            if self.glow_options_checkbox_rect:
                cbx, cby, cbw, cbh = (
                    self.glow_options_checkbox_rect.x,
                    self.glow_options_checkbox_rect.y,
                    self.glow_options_checkbox_rect.width,
                    self.glow_options_checkbox_rect.height,
                )
                glColor4f(0.12, 0.12, 0.16, 0.92)
                glBegin(GL_QUADS)
                glVertex2f(cbx, cby); glVertex2f(cbx + cbw, cby)
                glVertex2f(cbx + cbw, cby + cbh); glVertex2f(cbx, cby + cbh)
                glEnd()
                glColor4f(0.72, 0.74, 0.80, 0.95)
                glLineWidth(1.0)
                glBegin(GL_LINE_LOOP)
                glVertex2f(cbx, cby); glVertex2f(cbx + cbw, cby)
                glVertex2f(cbx + cbw, cby + cbh); glVertex2f(cbx, cby + cbh)
                glEnd()
                if self.show_key_press_glow:
                    glColor4f(0.95, 0.78, 0.24, 0.95)
                    glLineWidth(2.0)
                    glBegin(GL_LINES)
                    glVertex2f(cbx + 3, cby + 9); glVertex2f(cbx + 7, cby + 13)
                    glVertex2f(cbx + 7, cby + 13); glVertex2f(cbx + 13, cby + 4)
                    glEnd()
                self._draw_hover_highlight(self.glow_options_checkbox_rect, self._get_hover_alpha("glow_checkbox"))

            if self.key_light_fade_checkbox_rect:
                cbx, cby, cbw, cbh = (
                    self.key_light_fade_checkbox_rect.x,
                    self.key_light_fade_checkbox_rect.y,
                    self.key_light_fade_checkbox_rect.width,
                    self.key_light_fade_checkbox_rect.height,
                )
                glColor4f(0.12, 0.12, 0.16, 0.92)
                glBegin(GL_QUADS)
                glVertex2f(cbx, cby); glVertex2f(cbx + cbw, cby)
                glVertex2f(cbx + cbw, cby + cbh); glVertex2f(cbx, cby + cbh)
                glEnd()
                glColor4f(0.72, 0.74, 0.80, 0.95)
                glLineWidth(1.0)
                glBegin(GL_LINE_LOOP)
                glVertex2f(cbx, cby); glVertex2f(cbx + cbw, cby)
                glVertex2f(cbx + cbw, cby + cbh); glVertex2f(cbx, cby + cbh)
                glEnd()
                if self.show_key_light_fade:
                    glColor4f(0.95, 0.78, 0.24, 0.95)
                    glLineWidth(2.0)
                    glBegin(GL_LINES)
                    glVertex2f(cbx + 3, cby + 9); glVertex2f(cbx + 7, cby + 13)
                    glVertex2f(cbx + 7, cby + 13); glVertex2f(cbx + 13, cby + 4)
                    glEnd()
                self._draw_hover_highlight(self.key_light_fade_checkbox_rect, self._get_hover_alpha("key_fade_checkbox"))

        self._draw_hover_tooltip()
        
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

    def cleanup(self):
        self.app_running.clear()
        if self.data_thread:
            self.data_thread.join(timeout=0.5)

        self._text_texture_cache.clear()

        glDeleteProgram(self.shader)
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(2, [self.vbo_vertices, self.vbo_stream_data])
        
        if self.show_keyboard:
            glDeleteProgram(self.keyboard_shader)
            glDeleteVertexArrays(2, [self.keyboard_vao_white, self.keyboard_vao_black])
            glDeleteBuffers(3, [self.keyboard_vbo_quad, self.keyboard_vbo_white_keys, self.keyboard_vbo_black_keys])
            glDeleteTextures(list(self.keyboard_textures.values()))

        if self.note_texture:
            glDeleteTextures(2, [self.note_texture, self.note_edge_texture])

        pygame.quit()
