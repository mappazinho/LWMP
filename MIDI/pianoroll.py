# pianoroll.py - OPTIMIZED VERSION

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
from midi_parser import GPU_NOTE_DTYPE

# --- Constants ---
# Downgraded to GLSL 1.20 for maximum compatibility
# Removed layout qualifiers, switched in/out to attribute/varying
VERT_SHADER = """#version 120
attribute vec2 pos;
attribute vec2 note_times;
attribute vec3 note_info; // Passed as float vec3, will be cast to int

uniform mat4 u_projection;
uniform float u_time;
uniform float u_scroll_speed;
uniform float u_width;
uniform float u_height;
uniform float u_note_width;
uniform float u_guide_line_y;
uniform float u_window_start;
uniform float u_window_end;
uniform vec3 u_colors[128];
uniform vec2 u_pitch_layout[128];
uniform int u_is_white_key[128];

varying vec3 v_fragColor;
varying vec2 v_pos;
varying float v_note_h;

void main() {
    float on_time = note_times.x;
    float off_time = note_times.y;
    
    // Early exit: cull notes outside time window
    if (off_time < u_window_start || on_time > u_window_end) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    
    // Cast float attributes to int with a small bias to prevent precision errors
    // e.g., 1.0 might become 0.9999, casting to int makes it 0. Adding 0.5 makes it 1.49 -> 1.
    int pitch = int(note_info.x + 0.5);
    int track = int(note_info.z + 0.5);
    
    float note_duration = max(off_time - on_time, 0.0001);
    float note_h = max(2.0, note_duration * u_scroll_speed);
    float note_y = u_guide_line_y + (u_time - on_time) * u_scroll_speed;
    
    // Use pitch layout directly
    vec2 layout = u_pitch_layout[pitch];
    vec2 instance_scale = vec2(layout.y, note_h);
    vec2 instance_offset = vec2(layout.x, note_y - note_h);
    vec2 final_pos = pos * instance_scale + instance_offset;
    
    // Depth for layering:
    // --- Z-Depth Layering Logic ---
    // The final Z position of a vertex is `gl_Position.z = -z_depth` because
    // of the projection matrix. With `glDepthFunc(GL_LESS)`, this means that
    // the vertex with the LARGER `z_depth` value will be rendered ON TOP.
    // This section calculates `z_depth` to control note layering.

    // The layering priority is designed to mimic the keyboard's highlight logic,
    // which follows a "Last Note On" (LNO) rule.
    // Priority Order:
    // 1. Key Color: Black keys are always on top of white keys.
    // 2. Start Time: The most recently started note on a key is on top.

    // 1. Base layer by key color.
    // White keys occupy the z_depth range ~[0.0, 0.4].
    // Black keys occupy the z_depth range ~[0.5, 0.9].
    // This large gap ensures a black key is always drawn over any white key.
    float base_z = u_is_white_key[pitch] == 1 ? 0.0 : 0.5;

    // 2. Add offset based on the note's start time (`on_time`).
    // A larger `on_time` (a note that starts later) results in a larger `time_offset`.
    // This gives it a larger `z_depth`, causing it to be rendered on top.
    // `mod` prevents the value from growing infinitely during a long song, resetting every 1000 seconds.
    // The scaling factor maps this to the [0.0, 0.4] range, fitting within our `base_z` layers.
    float time_offset = mod(on_time, 1000.0) * 0.0004;

    float z_depth = base_z + time_offset;
    
    gl_Position = u_projection * vec4(final_pos, z_depth, 1.0);
    
    // Pass through
    v_pos = pos;
    v_note_h = note_h;
    
    // Manual modulo for older GLSL versions if % operator is not supported for ints
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

void main() {
    // fwidth is supported in GLSL 1.20 (GL 2.1+)
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

void main() {
    vec4 texColor;
    if (v_is_pressed > 0.5) {
        texColor = texture2D(u_texture_pressed, v_uv);
        gl_FragColor = texColor * vec4(v_color, 1.0);
    } else {
        texColor = texture2D(u_texture_unpressed, v_uv);
        gl_FragColor = texColor;
    }

    if (u_is_white_key) {
        vec2 fw_uv = fwidth(v_uv);
        float border_size = 0.5;
        float border_x = border_size * fw_uv.x;
        float border_y = border_size * fw_uv.y;

        if (v_uv.x < border_x || v_uv.x > 1.0 - border_x || v_uv.y < border_y || v_uv.y > 1.0 - border_y) {
            vec3 border_color = vec3(0.5, 0.5, 0.5);
            if (v_is_pressed > 0.5) {
                border_color = vec3(0.2, 0.2, 0.2);
            }
            gl_FragColor = vec4(border_color, texColor.a);
        }
    }
}
"""

class PianoRoll:
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.config = config
        self.screen = None
        
        # Note rendering resources
        self.shader = 0
        self.vao = 0
        self.vbo_vertices = 0
        self.vbo_stream_data = 0
        self.note_texture = 0
        self.note_edge_texture = 0
        # Initialize locations to -1
        self.u_note_texture_loc = -1
        self.u_note_edge_texture_loc = -1
        self.u_is_white_key_notes_loc = -1
        
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
        
        self.slider_min = 0.2
        self.slider_max = 5.0
        
        # Unify window size logic. The slider controls a single `window_seconds` value.
        # We'll use seconds_before_cursor as the authoritative value from config,
        # and apply it to all related variables for consistent behavior.
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

        # Keyboard Visualization
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
        self.render_notes_array = None
        self.render_on_times = None
        self.max_note_duration = 10.0
        
        self.pending_midi_data = None
        self.midi_data_lock = threading.Lock()
        
        # Randomize Colors Button (top-right corner)
        self.color_button_rect = None  # Will be set in init_pygame_and_gl
        self.color_button_size = 32  # Button size in pixels

    def _load_colors_from_xml(self, filepath=None):
        if filepath is None:
            # Build absolute path to colors.xml, assuming it's in the same dir as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, "colors.xml")

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
            # Fallback rainbow palette (16 distinctive colors)
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
        except pygame.error as e:
            print(f"Error loading or converting texture from {image_path}: {e}")
            return None, 0, 0

    def _load_keyboard_assets(self):
        skin_dir = "skin"
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
            self.white_key_instance_data['color'] = (1.0, 1.0, 1.0)

        black_key_dtype = np.dtype([('rect', 'f4', 4), ('is_pressed', 'f4'), ('color', 'f4', 3)])
        self.black_key_instance_data = np.zeros(len(black_keys_geom), dtype=black_key_dtype)
        if black_keys_geom:
            self.black_key_instance_data['rect'] = np.array(black_keys_geom, dtype=np.float32)
            self.black_key_instance_data['is_pressed'] = 0.0
            self.black_key_instance_data['color'] = (1.0, 1.0, 1.0)

        self.pitch_layout_data = np.zeros((128, 2), dtype=np.float32)
        for pitch, key_info in self.keyboard_layout.items():
            self.pitch_layout_data[pitch, 0] = key_info['x']
            self.pitch_layout_data[pitch, 1] = key_info['width']
            
    def init_pygame_and_gl(self):
        pygame.init()
        pygame.font.init()
        
        # REMOVED: gl_set_attribute calls that cause "Unknown OpenGL attribute" errors on older systems
        # Default context is usually compatibility profile, which is what we want for legacy GLSL support.
        
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | pygame.HWSURFACE)
        pygame.display.set_caption("Piano Roll")
        self._init_slider_geometry()

        try:
            self.note_texture, _, _ = self._load_texture(os.path.join("skin", "note.png"))
            self.note_edge_texture, _, _ = self._load_texture(os.path.join("skin", "noteEdge.png"))
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

        # Set up VAO/VBOs for Notes
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
        
        # Downgrade: Use float pointer for integer data since legacy GLSL 1.20 doesn't support integer attributes well
        # We manually add +0.5 in shader to fix precision
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
        glUniform1f(glGetUniformLocation(self.shader, "u_guide_line_y"), self.height * self.guide_line_y_ratio)
        glUniform2fv(self.u_pitch_layout_loc, 128, self.pitch_layout_data)
        
        if self.u_is_white_key_notes_loc != -1 and self.is_white_key_data is not None:
            glUniform1iv(self.u_is_white_key_notes_loc, 128, self.is_white_key_data)
        if self.u_window_start_loc != -1:
            glUniform1f(self.u_window_start_loc, 0.0)
        if self.u_window_end_loc != -1:
            glUniform1f(self.u_window_end_loc, 0.0)

        self.u_note_texture_loc = glGetUniformLocation(self.shader, "u_note_texture")
        self.u_note_edge_texture_loc = glGetUniformLocation(self.shader, "u_note_edge_texture")
        glUniform1i(self.u_note_texture_loc, 0)
        glUniform1i(self.u_note_edge_texture_loc, 1)

        self.channel_colors = self._load_colors_from_xml()
        glUniform3fv(self.u_colors_loc, 128, self.channel_colors)
        
        if self.show_keyboard:
            glUseProgram(self.keyboard_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.keyboard_shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
            glUniform1i(glGetUniformLocation(self.keyboard_shader, "u_texture_unpressed"), 0)
            glUniform1i(glGetUniformLocation(self.keyboard_shader, "u_texture_pressed"), 1)
            self.u_is_white_key_loc = glGetUniformLocation(self.keyboard_shader, "u_is_white_key")

        glUseProgram(0)
        
        self.data_thread = threading.Thread(target=self._data_streamer_thread, daemon=True)
        self.data_thread.start()
        
        # Initialize color button rect (top-right corner with margin)
        margin = 10
        self.color_button_rect = pygame.Rect(
            self.width - self.color_button_size - margin,
            margin,
            self.color_button_size,
            self.color_button_size
        )

    def load_midi(self, all_notes_gpu, get_current_time_func):
        """Thread-safe: Prepares data for GPU upload and optimizes structures."""
        self.get_current_time = get_current_time_func
        
        # --- 2. RENDER OPTIMIZATION DATA (Logic from yapper) ---
        # For fast slicing, we need the main rendering array to be strictly sorted by on_time.
        # This allows O(1) slicing instead of complex intersections.
        render_sort_idx = np.argsort(all_notes_gpu['on_time'], kind='stable')
        # Create a contiguous copy for fast upload
        render_notes_array = np.ascontiguousarray(all_notes_gpu[render_sort_idx])
        render_on_times = render_notes_array['on_time']
        
        # Calculate max note duration for smarter culling
        durations = render_notes_array['off_time'] - render_notes_array['on_time']
        if len(durations) > 0:
            self.max_note_duration = float(np.max(durations))
        
        # Store prepared data
        with self.midi_data_lock:
            self.pending_midi_data = {
                'all_notes_gpu': all_notes_gpu,
                'render_notes_array': render_notes_array,
                'render_on_times': render_on_times
            }
        
        print(f"MIDI data prepared: {len(all_notes_gpu)} notes. Max Duration: {self.max_note_duration:.2f}s")
    
    def _upload_pending_midi_data(self):
        """Called from render thread to flip pointers to new data."""
        with self.midi_data_lock:
            if self.pending_midi_data is None:
                return
            
            data = self.pending_midi_data
            self.pending_midi_data = None
        
        # Update Main Structures
        self.all_notes_gpu = data['all_notes_gpu']
        self.render_notes_array = data['render_notes_array']
        self.render_on_times = data['render_on_times']
        
        self.notes_to_draw = len(self.render_notes_array)
        
        # Trigger immediate data update
        self.force_data_update.set()
        print("MIDI data active on GPU thread.")
    
    def _data_streamer_thread(self):
        """OPTIMIZED: Background thread for slicing visible notes (logic from yapper)."""
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
                
                # OPTIMIZATION: Compute visible time window
                # The visual window where notes should actually be rendered
                view_start = now - self.seconds_before_cursor
                view_end = now + self.seconds_after_cursor
                
                # To catch long notes that started way before the window but are still playing,
                # we must search back by max_note_duration.
                search_start = view_start - self.max_note_duration
                
                # Binary search (fast O(log N)) on sorted start times
                start_idx = np.searchsorted(self.render_on_times, search_start, side='left')
                end_idx = np.searchsorted(self.render_on_times, view_end, side='right')
                
                # Candidate slice (contains all potential notes)
                candidates = self.render_notes_array[start_idx:end_idx]
                
                if len(candidates) > 0:
                    # FILTER: Discard notes that finished before the view started.
                    # This dramatically reduces the vertex count by culling notes that 
                    # were included only because of the wide search window.
                    # 'off_time' is the end time of the note.
                    mask = candidates['off_time'] > view_start
                    visible_slice = candidates[mask]
                    visible_count = len(visible_slice)
                else:
                    visible_slice = candidates
                    visible_count = 0
                
                # Safety Cap
                if visible_count > self.streaming_vbo_capacity:
                    visible_slice = visible_slice[:self.streaming_vbo_capacity]
                    visible_count = self.streaming_vbo_capacity
                
                if visible_count > 0:
                    # Make contiguous just in case, though it should be already
                    visible_slice_contiguous = np.ascontiguousarray(visible_slice)
                    
                    try:
                        # Use a timeout to avoid deadlocking if the main thread is stuck
                        self.data_queue.put((visible_slice_contiguous, visible_count), block=True, timeout=0.1)
                    except queue.Full:
                        print("Piano roll queue is full, frame skipped. This may indicate rendering lag.")
            
            time.sleep(0.005) # Yield

    def draw(self, current_time):
        """OPTIMIZED: Draws the buffer provided by the data thread."""
        self._upload_pending_midi_data()
        
        glClearColor(0.05, 0.05, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # --- GPU BUFFER UPDATE (Logic from yapper) ---
        try:
            # Poll queue for new note data
            queue_data = self.data_queue.get_nowait()
            visible_notes, count = queue_data
            
            # Save for keyboard lookup (Zero-cost optimization)
            self.last_visible_notes = visible_notes
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            
            # CRITICAL OPTIMIZATION: Buffer Orphaning
            # Calling glBufferData with NULL (None) tells the driver to allocate new memory,
            # allowing the GPU to continue processing the old buffer while we fill the new one.
            # This prevents CPU-GPU synchronization stalls.
            glBufferData(GL_ARRAY_BUFFER, self.streaming_vbo_capacity * self.note_size_bytes, None, GL_DYNAMIC_DRAW)
            
            # Upload new data
            glBufferSubData(GL_ARRAY_BUFFER, 0, visible_notes.nbytes, visible_notes)
            
            self.notes_to_draw = count
            
        except queue.Empty:
            # No new data this frame, redraw previous frame (or do nothing if nothing uploaded yet)
            pass

        # --- RENDER NOTES ---
        if self.notes_to_draw > 0 and self.render_notes_array is not None:
            window_start = current_time - self.seconds_before_cursor
            window_end = current_time + self.seconds_after_cursor
            
            glEnable(GL_DEPTH_TEST)
            glUseProgram(self.shader)
            
            glUniform1f(self.u_time_loc, current_time)
            glUniform1f(self.u_scroll_speed_loc, self.scroll_speed)
            glUniform1f(self.u_window_start_loc, window_start)
            glUniform1f(self.u_window_end_loc, window_end)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.note_texture)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.note_edge_texture)

            glBindVertexArray(self.vao)
            # Bind the buffer again just to be safe
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            
            # Draw only the count we uploaded
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self.notes_to_draw)
            
            glBindVertexArray(0)
            glUseProgram(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Draw keyboard overlay
        if self.show_keyboard and self.keyboard_layout:
            self._draw_keyboard_opengl(current_time)

        # Draw guide line
        if self.show_guide_line:
            glDisable(GL_DEPTH_TEST)
            glUseProgram(0)
            glMatrixMode(GL_PROJECTION);glPushMatrix();glLoadIdentity();glOrtho(0,self.width,self.height,0,-1,1)
            glMatrixMode(GL_MODELVIEW);glPushMatrix();glLoadIdentity()
            glLineWidth(2.0);glColor3f(0.8,0.0,0.0)
            guide_y = self.height * self.guide_line_y_ratio
            glBegin(GL_LINES);glVertex2f(0, guide_y);glVertex2f(self.width, guide_y);glEnd()
            glPopMatrix();glMatrixMode(GL_PROJECTION);glPopMatrix();glMatrixMode(GL_MODELVIEW)

        self._draw_slider_overlay()

        pygame.display.flip()

    def _draw_keyboard_opengl(self, current_time):
        """OPTIMIZED: Use the already-calculated visible notes for 0-cost lookup."""
        # Safe guard: If keyboard shader isn't ready or location not found (-1), don't crash
        if self.keyboard_shader == 0:
            return

        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.keyboard_shader)

        keyboard_height = self.keyboard_texture_info['white_key']['scaled_height']
        keyboard_y = self.height - keyboard_height
        glUniform1f(glGetUniformLocation(self.keyboard_shader, "u_keyboard_y"), keyboard_y)

        # OPTIMIZATION: Filter active notes from the visible batch
        # This replaces the complex and buggy incremental search with a simple filter
        # on the small subset of notes already resident in CPU memory.
        current_active_pitches = set()
        
        if self.last_visible_notes is not None and len(self.last_visible_notes) > 0:
            # We iterate the visible batch. This is fast (typically < 500 items).
            # Note: We can iterate directly or use numpy masking. Iteration is fine for this size.
            # We filter for notes that started before now and haven't ended yet.
            active_mask = (self.last_visible_notes['on_time'] <= current_time) & (self.last_visible_notes['off_time'] > current_time)
            active_notes = self.last_visible_notes[active_mask]
            
            # Since active_notes is sorted by on_time (inherited from render_notes_array),
            # iterating through it ensures that later notes overwrite earlier notes.
            for note in active_notes:
                pitch = note['pitch']
                current_active_pitches.add(pitch)
                
                # FIXED: Always update the key state and color for active notes.
                # Previously, we skipped this if the pitch was already active, which prevented
                # the color from updating when one note took over another on the same key.
                track = note['track']
                color = self.channel_colors[track % 128]
                
                if pitch in self.white_key_pitch_map:
                    idx = self.white_key_pitch_map[pitch]
                    self.white_key_instance_data[idx]['is_pressed'] = 1.0
                    self.white_key_instance_data[idx]['color'] = color
                elif pitch in self.black_key_pitch_map:
                    idx = self.black_key_pitch_map[pitch]
                    self.black_key_instance_data[idx]['is_pressed'] = 1.0
                    self.black_key_instance_data[idx]['color'] = color
        
        # OPTIMIZATION: Only reset keys that were just released
        released_keys = self.active_pitches_last_frame - current_active_pitches
        for pitch in released_keys:
            if pitch in self.white_key_pitch_map:
                idx = self.white_key_pitch_map[pitch]
                self.white_key_instance_data[idx]['is_pressed'] = 0.0
                self.white_key_instance_data[idx]['color'] = (1.0, 1.0, 1.0)
            elif pitch in self.black_key_pitch_map:
                idx = self.black_key_pitch_map[pitch]
                self.black_key_instance_data[idx]['is_pressed'] = 0.0
                self.black_key_instance_data[idx]['color'] = (1.0, 1.0, 1.0)
        
        self.active_pitches_last_frame = current_active_pitches

        # Draw white keys
        # self.u_is_white_key_loc being -1 is handled by GL (ignored), safe if not initialized
        glUniform1i(self.u_is_white_key_loc, 1)
        
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.keyboard_textures['white_key'])
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, self.keyboard_textures['white_pressed'])
        glBindVertexArray(self.keyboard_vao_white)
        glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_white_keys)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.white_key_instance_data.nbytes, self.white_key_instance_data)
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, len(self.white_key_instance_data))

        # Draw black keys
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

    # --- UI Helpers ---
    def _init_slider_geometry(self):
        padding = 16
        bar_width = max(140, self.width // 5)
        bar_height = 6
        x = padding
        y = padding
        # Window slider
        self.slider_rect = pygame.Rect(x, y, bar_width, self.slider_area_height)
        self.slider_bar = (x, y + (self.slider_area_height - bar_height) // 2, bar_width, bar_height)
        self._recalc_slider_handle()
        # Scroll speed slider
        scroll_y = y + self.slider_area_height + 8
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
            # Check color button first
            if self.color_button_rect and self.color_button_rect.collidepoint(event.pos):
                self.randomize_colors()
                return  # Don't check other controls
            if self.slider_rect and self.slider_rect.collidepoint(event.pos):
                self.slider_dragging = True
                self._update_slider_from_pos(event.pos[0])
            if self.scroll_slider_rect and self.scroll_slider_rect.collidepoint(event.pos):
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
        # Generate a random permutation
        indices = list(range(128))
        random.shuffle(indices)
        
        # Reorder colors based on shuffled indices
        self.channel_colors = self.channel_colors[indices]
        
        # Upload new colors to shader
        glUseProgram(self.shader)
        glUniform3fv(self.u_colors_loc, 128, self.channel_colors)
        glUseProgram(0)
        print("Colors randomized!")

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

    def _draw_slider_overlay(self):
        if not self.slider_rect: return
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        handle_half = 6
        # Window slider
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

        # Scroll slider
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
        
        # Draw Color Randomize Button (top-right with RGB bars icon)
        if self.color_button_rect:
            bx, by = self.color_button_rect.x, self.color_button_rect.y
            bs = self.color_button_size
            
            # Button background (semi-transparent dark)
            glColor4f(0.15, 0.15, 0.2, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            
            # RGB bars icon (3 horizontal bars: Red, Green, Blue)
            bar_margin = 6
            bar_height = (bs - 4 * bar_margin) / 3
            bar_width = bs - 2 * bar_margin
            
            # Red bar (top)
            glColor4f(1.0, 0.3, 0.3, 0.9)
            ry = by + bar_margin
            glBegin(GL_QUADS)
            glVertex2f(bx + bar_margin, ry)
            glVertex2f(bx + bar_margin + bar_width, ry)
            glVertex2f(bx + bar_margin + bar_width, ry + bar_height)
            glVertex2f(bx + bar_margin, ry + bar_height)
            glEnd()
            
            # Green bar (middle)
            glColor4f(0.3, 1.0, 0.3, 0.9)
            gy = by + bar_margin * 2 + bar_height
            glBegin(GL_QUADS)
            glVertex2f(bx + bar_margin, gy)
            glVertex2f(bx + bar_margin + bar_width, gy)
            glVertex2f(bx + bar_margin + bar_width, gy + bar_height)
            glVertex2f(bx + bar_margin, gy + bar_height)
            glEnd()
            
            # Blue bar (bottom)
            glColor4f(0.3, 0.3, 1.0, 0.9)
            bby = by + bar_margin * 3 + bar_height * 2
            glBegin(GL_QUADS)
            glVertex2f(bx + bar_margin, bby)
            glVertex2f(bx + bar_margin + bar_width, bby)
            glVertex2f(bx + bar_margin + bar_width, bby + bar_height)
            glVertex2f(bx + bar_margin, bby + bar_height)
            glEnd()
            
            # Button border
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