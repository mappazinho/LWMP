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
    
    int color_idx = int(mod(float(track), 128.0));
    v_fragColor = u_colors[color_idx] * 1.2;
}
"""

FRAG_SHADER = """#version 120
varying vec3 v_fragColor;
varying vec2 v_pos;
varying float v_note_h;

uniform sampler2D u_note_texture;
uniform sampler2D u_note_edge_texture;
uniform float u_glow_strength;

void main() {
    vec2 fw = fwidth(v_pos);
    float texture_border_y = 1.5 * fw.y;
    float side_border_width = 0.5 * fw.x;
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

    if (v_pos.x < side_border_width || v_pos.x > 1.0 - side_border_width) {
        vec3 outline_color = v_fragColor * 0.3;
        final_color = vec4(outline_color, 1.0);
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
        gl_FragColor = vec4(mix(normal_pressed, lit_base, glow_mode), texColor.a);
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
        vec2 centered = abs(v_uv - vec2(0.5, 0.48));
        float spread = clamp(1.0 - max(centered.x * 1.7, centered.y * 1.2), 0.0, 1.0);
        float spill = pow(spread, 2.0);
        float intensity = v_is_pressed > 0.5 ? 0.86 : 0.54;
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
        self.color_button_size = 32
        self.controls_panel_expanded = False
        self.controls_toggle_rect = None
        self.controls_panel_rect = None
        self.controls_close_rect = None
        self.controls_panel_size = (300, 136)
        self.overlay_font = None
        self._text_texture_cache = {}

    def _get_guide_line_y(self):
        if self.show_keyboard and 'white_key' in self.keyboard_texture_info:
            keyboard_height = self.keyboard_texture_info['white_key'].get('scaled_height')
            if keyboard_height is not None:
                return self.height - keyboard_height
        return self.height * self.guide_line_y_ratio

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
        self._update_glow_trails(current_time)
        
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

        if self.show_glow:
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
                color = self.channel_colors[track % 128]
                if pitch not in active_pitch_colors:
                    active_pitch_colors[pitch] = np.array(color, dtype=np.float32)
                else:
                    active_pitch_colors[pitch] += color

        key_light_colors = {}
        light_falloff = (1.0, 0.55, 0.28, 0.14, 0.07)
        for pitch, color_sum in active_pitch_colors.items():
            base_color = np.clip(color_sum, 0.0, 1.0)
            for distance, weight in enumerate(light_falloff):
                neighbors = (pitch,) if distance == 0 else (pitch - distance, pitch + distance)
                for neighbor in neighbors:
                    if neighbor < 0 or neighbor > 127 or neighbor not in self.keyboard_layout:
                        continue
                    if neighbor not in key_light_colors:
                        key_light_colors[neighbor] = np.array(base_color * weight, dtype=np.float32)
                    else:
                        key_light_colors[neighbor] += base_color * weight

        for pitch, color in key_light_colors.items():
            clamped_color = np.clip(color, 0.0, 1.0)
            if pitch in self.white_key_pitch_map:
                idx = self.white_key_pitch_map[pitch]
                self.white_key_instance_data[idx]['color'] = clamped_color
                if pitch in current_active_pitches:
                    self.white_key_instance_data[idx]['is_pressed'] = 1.0
            elif pitch in self.black_key_pitch_map:
                idx = self.black_key_pitch_map[pitch]
                self.black_key_instance_data[idx]['color'] = clamped_color
                if pitch in current_active_pitches:
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

        print(f"Piano roll glow {'enabled' if self.show_glow else 'disabled'}.")

    def _update_slider_from_pos(self, x_pos):
        t = (x_pos - self.slider_rect.x) / float(self.slider_rect.width)
        t = max(0.0, min(1.0, t))
        new_val = self.slider_min + t * (self.slider_max - self.slider_min)
        self.window_seconds = new_val
        self.seconds_before_cursor = new_val
        self.seconds_after_cursor = new_val
        self._recalc_slider_handle()
        self.force_data_update.set() # Force update when slider moves

    def _update_scroll_slider_from_pos(self, x_pos):
        t = (x_pos - self.scroll_slider_rect.x) / float(self.scroll_slider_rect.width)
        t = max(0.0, min(1.0, t))
        new_val = self.scroll_slider_min + t * (self.scroll_slider_max - self.scroll_slider_min)
        self.scroll_speed = new_val
        self._recalc_scroll_slider_handle()

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
