import math
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import threading
import numpy as np
import ctypes
import os
import sys
import bisect

from config import save_config
from piano.shaders import (
    VERT_SHADER, FRAG_SHADER,
    BLOOM_VERT_SHADER, BLOOM_FRAG_SHADER,
    KEYBOARD_VERT_SHADER, KEYBOARD_FRAG_SHADER,
    KEYBOARD_BLOOM_VERT_SHADER, KEYBOARD_BLOOM_FRAG_SHADER,
    SCREEN_BLOOM_VERT_SHADER, BLOOM_EXTRACT_FRAG_SHADER,
    SCREEN_BLOOM_BLUR_FRAG_SHADER, SCREEN_BLOOM_FRAG_SHADER,
    GLOW_VERT_SHADER, GLOW_FRAG_SHADER,
)
from piano.skin_utils import (
    _SKIN_ROOT, _SKIN_DIR, _COLORS_XML_PATH,
    _resolve_skin_dir, _load_colors_from_xml, _load_skin_config,
)
from piano.note_utils import (
    RENDER_NOTE_DTYPE, _build_base_render_data, _build_render_data_for_mode,
)
from piano.bloom import BloomMixin
from piano.glow import GlowMixin
from piano.keyboard import KeyboardMixin
from piano.overlays import OverlayMixin

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else _SCRIPT_DIR


class PianoRoll(BloomMixin, GlowMixin, KeyboardMixin, OverlayMixin):
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.config = config
        self.export_mode = False
        self.screen = None
        
        self.shader = 0
        self.bloom_shader = 0
        self.vao = 0
        self.bloom_vao = 0
        self.vbo_vertices = 0
        self.vbo_stream_data = 0
        self.vbo_stream_capacity_bytes = 0
        self.pbo_ids = []
        self._pbo_index = 0
        self.note_texture = 0
        self.note_edge_texture = 0
        self.u_note_texture_loc = -1
        self.u_note_edge_texture_loc = -1
        self.u_is_white_key_notes_loc = -1
        self.u_glow_strength_loc = -1
        self.u_overclock_loc = -1
        self.u_bloom_time_loc = -1
        self.u_bloom_timespan_loc = -1
        self.u_bloom_pitch_layout_loc = -1
        self.u_bloom_colors_loc = -1
        self.u_bloom_radius_loc = -1
        self.u_bloom_strength_loc = -1
        self.screen_bloom_shader = 0
        self.bloom_extract_shader = 0
        self.bloom_blur_shader = 0
        self.screen_bloom_vao = 0
        self.screen_bloom_vbo = 0
        self.scene_fbo = 0
        self.scene_color_texture = 0
        self.scene_depth_rbo = 0
        self.bloom_fbo = 0
        self.bloom_blur_fbo = 0
        self.bloom_texture = 0
        self.bloom_blur_texture = 0
        self.bloom_width = 0
        self.bloom_height = 0
        self.glow_shader = 0
        self.glow_vao = 0
        self.glow_vbo_instance = 0
        self.glow_instance_data = np.zeros(128, dtype=[
            ('rect', 'f4', 4),
            ('color', 'f4', 4)
        ])
        self.u_screen_bloom_texel_size_loc = -1
        self.u_screen_bloom_strength_loc = -1
        self.u_bloom_extract_texel_size_loc = -1
        self.u_bloom_blur_texel_size_loc = -1
        self.u_bloom_blur_direction_loc = -1
        
        self.all_notes_gpu = None
        self.app_running = threading.Event()
        self.app_running.set()
        self.notes_to_draw = 0
        self.current_note_data = None
        self.get_current_time = None
        self.last_frame_time = 0.0
        self.export_total_duration = 0.0
        self._time_index = None
        self.capacity_warning_active = False
        self.capacity_warning_visible_count = 0
        self.vbo_debug = False
        
        vis_cfg = self.config.get('visualizer', {})
        gui_cfg = self.config.get('gui', {})
        self.scroll_speed = float(vis_cfg.get('scroll_speed', 2500.0))
        self.scroll_slider_min = 400.0
        self.scroll_slider_max = 5000.0
        self.scroll_slider_dragging = False
        self.scroll_slider_rect = None
        self.scroll_slider_bar = None
        self.scroll_slider_handle_x = 0.0
        self.fps_cap = int(vis_cfg.get('fps_cap', 120))
        self.fps_slider_min = 30.0
        self.fps_slider_max = 5000.0
        self.fps_slider_dragging = False
        self.fps_slider_rect = None
        self.fps_slider_bar = None
        self.fps_slider_handle_x = 0.0
        self.note_width = float(vis_cfg.get('note_width', 10.0))
        self.show_guide_line = bool(vis_cfg.get('show_guide_line', True))
        self.show_glow = bool(vis_cfg.get('show_glow', False))
        self.show_bloom = bool(vis_cfg.get('show_bloom', False))
        self.show_spike_bloom = bool(vis_cfg.get('show_spike_bloom', True))
        self.glow_strength = 1.0 if self.show_glow else 0.0
        self.bloom_base_strength = 0.25
        self.bloom_strength = self.bloom_base_strength if self.show_bloom else 0.0
        self.bloom_radius = 26.0
        self.bloom_min_strength = 0.15
        self.bloom_response_reference = 900.0
        self.bloom_response_curve = 1.18
        self.bloom_response_scale = 1.95
        self.bloom_emergency_strength = 0.04
        self.bloom_emergency_threshold = 2400.0
        self.bloom_emergency_full_threshold = 4000.0
        self.last_bloom_update_time = time.perf_counter()
        self.show_key_press_glow = bool(vis_cfg.get('show_key_press_glow', True))
        self.show_key_light_fade = bool(vis_cfg.get('show_key_light_fade', False))
        self.overclock_mode = bool(vis_cfg.get('overclock_mode', False))
        self.hide_buttons = bool(vis_cfg.get('hide_buttons', False))
        self.renderer_mode = str(vis_cfg.get('renderer_mode', 'default'))
        self.renderer_modes = ['default', 'channel_split', 'horizontal']
        self.active_channels = []
        self.num_active_lanes = 0
        self.glow_cull_threshold = int(vis_cfg.get('glow_cull_threshold', 128))
        self.glow_fade_duration = 0.1
        self.nps_spikes = []
        self._spike_rank_map = {}
        self._split_fade_trails = {}
        self.spike_bloom_rise_duration = 1.75
        self.spike_bloom_fall_duration = 1.75
        self.spike_bloom_min_boost = 0.14
        self.spike_bloom_max_boost = 0.48
        
        self.slider_area_height = 30
        
        self.streaming_vbo_capacity = int(vis_cfg.get('streaming_vbo_capacity', 5000000))
        self.guide_line_y_ratio = float(vis_cfg.get('guide_line_y_ratio', 0.8))
        self.use_gpu_cull = True


        self.show_keyboard = bool(vis_cfg.get('show_keyboard', True))
        self.preferred_color_mode = "track"
        self.note_color_mode = "track"
        self.background_color = tuple(
            max(0.0, min(1.0, float(c) / 255.0))
            for c in gui_cfg.get('pianoroll_bg', [13, 13, 20])[:3]
        )
        self.keyboard_layout = None
        self.keyboard_textures = {}
        self.keyboard_texture_info = {}
        
        self.keyboard_shader = 0
        self.keyboard_bloom_shader = 0
        self.keyboard_vao_white = 0
        self.keyboard_vao_black = 0
        self.keyboard_bloom_vao_white = 0
        self.keyboard_bloom_vao_black = 0
        self.keyboard_vbo_quad = 0
        self.keyboard_vbo_white_keys = 0
        self.keyboard_vbo_black_keys = 0
        self.white_key_instance_data = None
        self.black_key_instance_data = None
        self.white_key_pitch_map = {}
        self.black_key_pitch_map = {}
        self.u_is_white_key_loc = -1
        self.u_keyboard_bloom_is_white_key_loc = -1
        self.u_keyboard_bloom_strength_loc = -1
        self.u_keyboard_overclock_loc = -1
        self.u_keyboard_bloom_overclock_loc = -1
        
        self.active_pitches_last_frame = set()
        self.last_visible_notes = None
        self.glow_trails = {}
        self.glow_cull_factor = 1.0
        self.last_glow_time = None
        self.base_render_notes = None
        self.base_render_on_times = None
        self.render_notes_array = None
        self.render_on_times = None
        self.render_notes_by_mode = {}
        self.render_on_times_by_mode = {}
        self.max_note_duration = 10.0

        self._vis_active_pos = None
        self._vis_active_indices = None
        self._vis_active_count = 0
        self._vis_spawn_idx = 0
        self._vis_expire_ptr = 0
        self._vis_expire_order = None
        self._vis_expire_times = None
        self._vis_last_spawn_limit = None
        self._vis_last_renderer_mode = None
        self._vis_force_reset = True
        self.overlap_cull_duration_similarity = 0.82
        self.overlap_cull_coverage_threshold = 0.88
        self.overlap_cull_recent_candidates = 4
        
        self.pending_midi_data = None
        self.midi_data_lock = threading.Lock()
        
        self.color_button_rect = None
        self.color_mode_button_rect = None
        self.fun_button_rect = None
        self.glow_button_rect = None
        self.glow_options_button_rect = None
        self.glow_options_panel_rect = None
        self.renderer_mode_button_rect = None
        self.glow_options_checkbox_rect = None
        self.key_light_fade_checkbox_rect = None
        self.bloom_checkbox_rect = None
        self.spike_bloom_checkbox_rect = None
        self.glow_options_expanded = False
        self.fun_options_expanded = False
        self.fun_options_panel_rect = None
        self.overclock_mode = False
        self.overclock_intensity = 0.0
        self._fps_history = []
        self._last_fps_time = time.perf_counter()
        self.color_button_size = 32
        self.controls_panel_expanded = False
        self.controls_toggle_rect = None
        self.controls_panel_rect = None
        self.controls_close_rect = None
        self.controls_panel_size = (300, 184)
        self.overlay_font = None
        self._text_texture_cache = {}
        self.hover_fade_states = {}
        self.hover_tooltip_text = None
        self.hover_mouse_pos = (0, 0)
        gui_cfg = self.config.get('gui', {})
        render_cfg = self.config.get('render', {})
        self.live_show_stats_overlay = bool(gui_cfg.get('show_pianoroll_stats_overlay', False))
        self.export_show_stats_overlay = bool(render_cfg.get('show_stats_overlay', False))
        self.live_note_count_passed = 0
        self.live_nps_value = 0
        self.live_bpm_value = 0.0
        self.live_polyphony_value = 0
        self.stats_on_times = None
        self.stats_off_times_sorted = None
        self.stats_tempo_events = [(0.0, 120.0)]
        self.stats_tempo_times = np.array([0.0], dtype=np.float64)
        self.stats_tempo_bpms = np.array([120.0], dtype=np.float32)
        self.stats_total_duration = 0.0
        self.stats_total_notes = 0
        self.stats_multiplier = 1.0
        self.stats_modification_enabled = False
        self.stats_selected_spike = None
        self.stats_spike_intensity = 1.0

    def _get_timespan(self):
        return max(0.001, self._get_guide_line_y() / max(1.0, self.scroll_speed))

    def _get_quantized_time(self, current_time):
        guide_y = self._get_guide_line_y()
        timespan = self._get_timespan()
        sec_per_px = timespan / max(1.0, guide_y)
        return math.floor(current_time / max(1e-10, sec_per_px)) * sec_per_px

    def _get_guide_line_y(self):
        if self.show_keyboard and 'white_key' in self.keyboard_texture_info:
            keyboard_height = self.keyboard_texture_info['white_key'].get('scaled_height')
            if keyboard_height is not None:
                return self.height - keyboard_height
        return self.height * self.guide_line_y_ratio

    def _save_visualizer_config(self):
        vis_cfg = self.config.setdefault('visualizer', {})
        vis_cfg['show_glow'] = bool(self.show_glow)
        vis_cfg['show_bloom'] = bool(self.show_bloom)
        vis_cfg['show_spike_bloom'] = bool(self.show_spike_bloom)
        vis_cfg['show_key_press_glow'] = bool(self.show_key_press_glow)
        vis_cfg['show_key_light_fade'] = bool(self.show_key_light_fade)
        vis_cfg['overclock_mode'] = bool(self.overclock_mode)
        vis_cfg['hide_buttons'] = bool(self.hide_buttons)
        vis_cfg['renderer_mode'] = str(self.renderer_mode)
        vis_cfg['scroll_speed'] = float(self.scroll_speed)
        vis_cfg['fps_cap'] = int(self.fps_cap)
        vis_cfg['streaming_vbo_capacity'] = int(self.streaming_vbo_capacity)
        save_config(self.config)

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

    def set_export_mode(self, enabled=True):
        self.export_mode = bool(enabled)
        if self.export_mode:
            self.controls_panel_expanded = False
            self.glow_options_expanded = False
            self.hover_tooltip_text = None

    def _rebuild_color_mode_render_data(self):
        if self.all_notes_gpu is None:
            return
        active_mode = "channel" if self.note_color_mode == "channel" else "track"
        if self.base_render_notes is None:
            self.base_render_notes, self.base_render_on_times = _build_base_render_data(self.all_notes_gpu)
        if active_mode not in self.render_notes_by_mode:
            render_notes_array, render_on_times = _build_render_data_for_mode(self.base_render_notes, active_mode)
            self.render_notes_by_mode[active_mode] = render_notes_array
            self.render_on_times_by_mode[active_mode] = render_on_times
        else:
            render_notes_array = self.render_notes_by_mode[active_mode]
            render_on_times = self.render_on_times_by_mode[active_mode]
        if self.export_mode:
            self.render_notes_array = render_notes_array
            self.render_on_times = render_on_times
            self._build_time_index()
            self._rebuild_visible_acceleration()
            self.notes_to_draw = 0
            self.last_visible_notes = np.empty(0, dtype=RENDER_NOTE_DTYPE)
        else:
            with self.midi_data_lock:
                self.pending_midi_data = {
                    'all_notes_gpu': self.all_notes_gpu,
                    'base_render_notes': self.base_render_notes,
                    'base_render_on_times': self.base_render_on_times,
                    'render_notes_by_mode': self.render_notes_by_mode,
                    'render_on_times_by_mode': self.render_on_times_by_mode,
                    'render_notes_array': render_notes_array,
                    'render_on_times': render_on_times
                }
    def set_preferred_color_mode(self, mode):
        mode = "channel" if mode == "channel" else "track"
        self.preferred_color_mode = mode
        self.note_color_mode = mode
        self._rebuild_color_mode_render_data()

    def toggle_note_color_mode(self):
        self.note_color_mode = "channel" if self.note_color_mode == "track" else "track"
        self._rebuild_color_mode_render_data()
        print(f"Piano roll note colors switched to {self.note_color_mode}.")

    def _smoothstep(self, edge0, edge1, x):
        if edge0 == edge1:
            return 1.0 if x >= edge1 else 0.0
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def set_nps_spikes(self, spikes):
        self.nps_spikes = list(spikes or [])
        self._rebuild_spike_rank_map()

    def _rebuild_spike_rank_map(self):
        self._spike_rank_map = {}
        if not self.nps_spikes:
            return
        ranked = sorted(self.nps_spikes, key=lambda item: (item[1], item[0]))
        count = len(ranked)
        for idx, (spike_time, spike_value) in enumerate(ranked):
            norm = 1.0 if count <= 1 else idx / float(count - 1)
            weighted_norm = pow(norm, 1.35)
            amplitude = self.spike_bloom_min_boost + (self.spike_bloom_max_boost - self.spike_bloom_min_boost) * weighted_norm
            self._spike_rank_map[(float(spike_time), int(spike_value))] = amplitude

    def load_midi(self, all_notes_gpu, get_current_time_func):
        """Prepare note data for the render thread."""
        self.get_current_time = get_current_time_func
        self.glow_trails.clear()
        self._split_fade_trails.clear()
        self.last_glow_time = None
        self.base_render_notes, self.base_render_on_times = _build_base_render_data(all_notes_gpu)
        self.active_channels = sorted(set(int(c) for c in np.unique(self.base_render_notes['padding'])))
        self.num_active_lanes = len(self.active_channels)
        self.render_notes_by_mode = {}
        self.render_on_times_by_mode = {}
        active_mode = "channel" if self.note_color_mode == "channel" else "track"
        render_notes_array, render_on_times = _build_render_data_for_mode(self.base_render_notes, active_mode)
        self.render_notes_by_mode[active_mode] = render_notes_array
        self.render_on_times_by_mode[active_mode] = render_on_times
        durations = render_notes_array['off_time'] - render_notes_array['on_time']
        if len(durations) > 0:
            self.max_note_duration = float(np.max(durations))
            self.export_total_duration = float(np.max(render_notes_array['off_time']))
        else:
            self.export_total_duration = 0.0
        if self.export_mode:
            self.all_notes_gpu = all_notes_gpu
            self.render_notes_array = render_notes_array
            self.render_on_times = render_on_times
            self._rebuild_visible_acceleration()
            self.notes_to_draw = 0
            self.last_visible_notes = np.empty(0, dtype=RENDER_NOTE_DTYPE)
        else:
            with self.midi_data_lock:
                self.pending_midi_data = {
                    'all_notes_gpu': all_notes_gpu,
                    'base_render_notes': self.base_render_notes,
                    'base_render_on_times': self.base_render_on_times,
                    'render_notes_by_mode': self.render_notes_by_mode,
                    'render_on_times_by_mode': self.render_on_times_by_mode,
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
        self.base_render_notes = data.get('base_render_notes')
        self.base_render_on_times = data.get('base_render_on_times')
        self.render_notes_by_mode = data.get('render_notes_by_mode', {})
        self.render_on_times_by_mode = data.get('render_on_times_by_mode', {})
        self.render_notes_array = data['render_notes_array']
        self.render_on_times = data['render_on_times']
        self._build_time_index()
        self._rebuild_visible_acceleration()
        self.notes_to_draw = 0
        self.last_visible_notes = np.empty(0, dtype=RENDER_NOTE_DTYPE)
        print("MIDI data active on GPU thread.")

    def _build_time_index(self):
        if self.render_on_times is None or len(self.render_on_times) < 2:
            self._time_index = None
            return
        tmin = float(self.render_on_times[0])
        tmax = float(self.render_on_times[-1])
        step = 0.5
        buckets = np.arange(tmin, tmax + step, step, dtype=np.float64)
        indices = np.searchsorted(self.render_on_times, buckets, side='left')
        self._time_index = (indices, tmin, step)

    def _search_by_on_time(self, view_start, view_end):
        if self.render_notes_array is None or self.render_on_times is None:
            return 0, 0
        if view_end <= view_start:
            return 0, 0
        ti = self._time_index
        if ti is not None and len(self.render_on_times) > 100000:
            indices, tmin, step = ti
            n = len(indices)
            b_start = int((view_start - tmin) / step)
            b_start = max(0, min(b_start, n - 2))
            approx_start = int(indices[b_start])
            b_end = int((view_end - tmin) / step) + 1
            b_end = max(0, min(b_end, n - 1))
            approx_end = int(indices[b_end])
            if approx_end - approx_start <= 0:
                approx_end = min(approx_start + 2000, len(self.render_on_times))
            if approx_end - approx_start > 0:
                sub = self.render_on_times[approx_start:approx_end]
                start_idx = approx_start + int(np.searchsorted(sub, view_start, side='left'))
                end_idx = approx_start + int(np.searchsorted(sub, view_end, side='right'))
                return start_idx, end_idx - start_idx
        start_idx = np.searchsorted(self.render_on_times, view_start, side='left')
        end_idx = np.searchsorted(self.render_on_times, view_end, side='right')
        return start_idx, end_idx - start_idx

    def _slice_visible_notes_for_time(self, current_time, sustained_lookback=None):
        if self.render_notes_array is None or self.render_on_times is None:
            return 0, 0

        self.capacity_warning_active = False
        self.capacity_warning_visible_count = 0
        timespan = self._get_timespan()
        guide_y = self._get_guide_line_y()
        future_visible = max(0.1, (self.height - guide_y) * timespan / max(1.0, guide_y))
        if sustained_lookback is None:
            max_dur = getattr(self, 'max_note_duration', 10.0)
            sustained_lookback = min(max_dur, 15.0, timespan)
        view_start = current_time - timespan - sustained_lookback
        view_end = current_time + future_visible + 2.0
        return self._search_by_on_time(view_start, view_end)

    def _rebuild_visible_acceleration(self):
        n = 0 if self.render_notes_array is None else len(self.render_notes_array)
        if n == 0:
            self._vis_active_pos = np.empty(0, dtype=np.int32)
            self._vis_active_indices = np.empty(0, dtype=np.int32)
            self._vis_active_count = 0
            self._vis_spawn_idx = 0
            self._vis_expire_ptr = 0
            self._vis_expire_order = np.empty(0, dtype=np.int32)
            self._vis_expire_times = np.empty(0, dtype=np.float32)
            self._vis_last_spawn_limit = None
            self._vis_last_renderer_mode = self.renderer_mode
            self._vis_force_reset = True
            return

        idx_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
        off = self.render_notes_array['off_time']
        self._vis_expire_order = np.argsort(off, kind='stable').astype(idx_dtype, copy=False)
        self._vis_expire_times = off[self._vis_expire_order].copy()
        self._vis_active_pos = np.full(n, -1, dtype=idx_dtype)
        self._vis_active_indices = np.empty(n, dtype=idx_dtype)
        self._vis_active_count = 0
        self._vis_spawn_idx = 0
        self._vis_expire_ptr = 0
        self._vis_last_spawn_limit = None
        self._vis_last_renderer_mode = self.renderer_mode
        self._vis_force_reset = True

    def _reset_visible_active_at(self, spawn_limit, expire_before):
        if self._vis_active_pos is None or len(self._vis_active_pos) == 0:
            self._vis_active_count = 0
            self._vis_spawn_idx = 0
            self._vis_expire_ptr = 0
            self._vis_last_spawn_limit = spawn_limit
            return

        prev_count = int(self._vis_active_count)
        active_pos = self._vis_active_pos
        active_indices = self._vis_active_indices
        for i in range(prev_count):
            idx = int(active_indices[i])
            active_pos[idx] = -1

        active_count = 0
        self._vis_expire_ptr = int(bisect.bisect_left(self._vis_expire_times, expire_before))
        self._vis_spawn_idx = int(bisect.bisect_right(self.render_on_times, spawn_limit))

        max_dur = float(getattr(self, 'max_note_duration', 0.0))
        cand_start = float(expire_before) - max_dur - 0.25
        sidx, cnt = self._search_by_on_time(cand_start, spawn_limit)
        if cnt > 0:
            off = self.render_notes_array['off_time'][sidx:sidx + cnt]
            mask = off >= expire_before
            local = np.flatnonzero(mask)
            if local.size:
                active_idx = (local + sidx).astype(active_indices.dtype, copy=False)
                active_count = int(active_idx.size)
                active_indices[:active_count] = active_idx
                active_pos[active_idx] = np.arange(active_count, dtype=active_pos.dtype)

        self._vis_active_count = active_count
        self._vis_last_spawn_limit = spawn_limit

    def _update_visible_notes(self, current_time):
        self.capacity_warning_active = False
        self.capacity_warning_visible_count = 0

        if self._vis_active_pos is None:
            self._rebuild_visible_acceleration()

        if (self.render_notes_array is None or len(self.render_notes_array) == 0 or self._vis_active_pos is None):
            self.last_visible_notes = np.empty(0, dtype=RENDER_NOTE_DTYPE)
            self.notes_to_draw = 0
            return

        timespan = self._get_timespan()
        guide_y = max(1.0, self._get_guide_line_y())

        if self.renderer_mode == 'horizontal':
            future_visible = timespan
            spawn_ahead = 0.1
        elif self.renderer_mode == 'channel_split':
            future_visible = 0.0
            spawn_ahead = 0.1
        else:
            future_visible = max(0.0, (self.height - guide_y) * timespan / guide_y)
            spawn_ahead = timespan + 0.1

        expire_before = float(current_time) - float(future_visible) - 0.05
        spawn_limit = float(current_time) + float(spawn_ahead)

        need_reset = (self._vis_force_reset or self._vis_last_spawn_limit is None or self.renderer_mode != self._vis_last_renderer_mode or spawn_limit < float(self._vis_last_spawn_limit) - 1e-6)

        if need_reset:
            self._reset_visible_active_at(spawn_limit, expire_before)
            self._vis_force_reset = False
            self._vis_last_renderer_mode = self.renderer_mode

        active_pos = self._vis_active_pos
        active_indices = self._vis_active_indices
        active_count = int(self._vis_active_count)
        expire_ptr = int(self._vis_expire_ptr)
        spawn_idx = int(self._vis_spawn_idx)
        expire_order = self._vis_expire_order
        expire_times = self._vis_expire_times
        on_times = self.render_on_times
        off_times = self.render_notes_array['off_time']
        n = len(on_times)

        new_expire_ptr = int(bisect.bisect_left(expire_times, expire_before))
        if new_expire_ptr > expire_ptr:
            expired = expire_order[expire_ptr:new_expire_ptr]
            for idx in expired:
                p = int(active_pos[idx])
                if p != -1:
                    active_count -= 1
                    last = int(active_indices[active_count])
                    active_indices[p] = last
                    active_pos[last] = p
                    active_pos[idx] = -1
            expire_ptr = new_expire_ptr

        new_spawn_idx = int(bisect.bisect_right(on_times, spawn_limit))
        if new_spawn_idx > spawn_idx:
            spawn_off = off_times[spawn_idx:new_spawn_idx]
            keep = np.flatnonzero(spawn_off >= expire_before)
            if keep.size > 0:
                new_indices = keep.astype(active_indices.dtype, copy=False)
                new_indices += spawn_idx
                n_new = len(new_indices)
                active_indices[active_count:active_count + n_new] = new_indices
                active_pos[new_indices] = np.arange(active_count, active_count + n_new, dtype=active_pos.dtype)
                active_count += n_new
            spawn_idx = new_spawn_idx

        self._vis_active_count = active_count
        self._vis_expire_ptr = expire_ptr
        self._vis_spawn_idx = spawn_idx
        self._vis_last_spawn_limit = spawn_limit

        if active_count <= 0:
            self.last_visible_notes = np.empty(0, dtype=RENDER_NOTE_DTYPE)
            self.notes_to_draw = 0
            return

        idx = active_indices[:active_count].copy()
        idx.sort(kind='stable')
        self.last_visible_notes = self.render_notes_array[idx]
        self.notes_to_draw = active_count

    def init_pygame_and_gl(self, hidden=False, disable_vsync=False):
        import piano.skin_utils as _su
        _su._SKIN_DIR = _resolve_skin_dir(
            _su._load_skin_config().get("visualizer", {}).get("skin_name", "default"),
            _su._SKIN_ROOT,
        ) or _su._SKIN_DIR

        pygame.init()
        pygame.font.init()
        self.overlay_font = pygame.font.Font(None, 18)

        flags = DOUBLEBUF | OPENGL | pygame.HWSURFACE
        if hidden and hasattr(pygame, "HIDDEN"):
            flags |= pygame.HIDDEN
        if disable_vsync:
            pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("Piano Roll")
        self._init_slider_geometry()

        try:
            self.note_texture, _, _ = self._load_texture(os.path.join(_su._SKIN_DIR, "note.png"))
            self.note_edge_texture, _, _ = self._load_texture(os.path.join(_su._SKIN_DIR, "noteEdge.png"))
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
        glDisable(GL_SCISSOR_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        try:
            self.shader = compileProgram(
                compileShader(VERT_SHADER, GL_VERTEX_SHADER),
                compileShader(FRAG_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as err:
            print("Note shader compilation failed:\n", err); raise

        try:
            self.bloom_shader = compileProgram(
                compileShader(BLOOM_VERT_SHADER, GL_VERTEX_SHADER),
                compileShader(BLOOM_FRAG_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as err:
            print("Bloom shader compilation failed:\n", err)
            self.bloom_shader = 0

        try:
            self.screen_bloom_shader = compileProgram(
                compileShader(SCREEN_BLOOM_VERT_SHADER, GL_VERTEX_SHADER),
                compileShader(SCREEN_BLOOM_FRAG_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as err:
            print("Screen bloom shader compilation failed:\n", err)
            self.screen_bloom_shader = 0

        try:
            self.bloom_extract_shader = compileProgram(
                compileShader(SCREEN_BLOOM_VERT_SHADER, GL_VERTEX_SHADER),
                compileShader(BLOOM_EXTRACT_FRAG_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as err:
            print("Bloom extract shader compilation failed:\n", err)
            self.bloom_extract_shader = 0

        try:
            self.bloom_blur_shader = compileProgram(
                compileShader(SCREEN_BLOOM_VERT_SHADER, GL_VERTEX_SHADER),
                compileShader(SCREEN_BLOOM_BLUR_FRAG_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as err:
            print("Bloom blur shader compilation failed:\n", err)
            self.bloom_blur_shader = 0
        
        if self.show_keyboard:
            try:
                self.keyboard_shader = compileProgram(
                    compileShader(KEYBOARD_VERT_SHADER, GL_VERTEX_SHADER),
                    compileShader(KEYBOARD_FRAG_SHADER, GL_FRAGMENT_SHADER)
                )
            except Exception as err:
                print("Keyboard shader compilation failed:\n", err); self.show_keyboard = False
            if self.show_keyboard:
                try:
                    self.keyboard_bloom_shader = compileProgram(
                        compileShader(KEYBOARD_BLOOM_VERT_SHADER, GL_VERTEX_SHADER),
                        compileShader(KEYBOARD_BLOOM_FRAG_SHADER, GL_FRAGMENT_SHADER)
                    )
                except Exception as err:
                    print("Keyboard bloom shader compilation failed:\n", err)
                    self.keyboard_bloom_shader = 0

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
        self.note_size_bytes = RENDER_NOTE_DTYPE.itemsize
        self.vbo_stream_capacity_bytes = self.streaming_vbo_capacity * self.note_size_bytes
        glBufferData(GL_ARRAY_BUFFER, self.vbo_stream_capacity_bytes, None, GL_STREAM_DRAW)
        
        glEnableVertexAttribArray(note_times_loc); glVertexAttribPointer(note_times_loc, 2, GL_FLOAT, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(0)); glVertexAttribDivisor(note_times_loc, 1)
        glEnableVertexAttribArray(note_info_loc)
        glVertexAttribPointer(note_info_loc, 4, GL_UNSIGNED_BYTE, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(8))
        glVertexAttribDivisor(note_info_loc, 1)

        if self.bloom_shader:
            bloom_pos_loc = glGetAttribLocation(self.bloom_shader, "pos")
            bloom_note_times_loc = glGetAttribLocation(self.bloom_shader, "note_times")
            bloom_note_info_loc = glGetAttribLocation(self.bloom_shader, "note_info")
            self.bloom_vao = glGenVertexArrays(1)
            glBindVertexArray(self.bloom_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
            glEnableVertexAttribArray(bloom_pos_loc)
            glVertexAttribPointer(bloom_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            glEnableVertexAttribArray(bloom_note_times_loc)
            glVertexAttribPointer(bloom_note_times_loc, 2, GL_FLOAT, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(0))
            glVertexAttribDivisor(bloom_note_times_loc, 1)
            glEnableVertexAttribArray(bloom_note_info_loc)
            glVertexAttribPointer(bloom_note_info_loc, 4, GL_UNSIGNED_BYTE, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(8))
            glVertexAttribDivisor(bloom_note_info_loc, 1)

        if self.screen_bloom_shader:
            screen_pos_loc = glGetAttribLocation(self.screen_bloom_shader, "pos")
            self.screen_bloom_vao = glGenVertexArrays(1)
            glBindVertexArray(self.screen_bloom_vao)
            self.screen_bloom_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.screen_bloom_vbo)
            screen_quad = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, screen_quad.nbytes, screen_quad, GL_STATIC_DRAW)
            glEnableVertexAttribArray(screen_pos_loc)
            glVertexAttribPointer(screen_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

        try:
            self.glow_shader = compileProgram(
                compileShader(GLOW_VERT_SHADER, GL_VERTEX_SHADER),
                compileShader(GLOW_FRAG_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as err:
            print("Procedural glow shader compilation failed:\n", err)
            self.glow_shader = 0

        if self.glow_shader:
            self.glow_vao = glGenVertexArrays(1)
            glBindVertexArray(self.glow_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
            pos_loc = glGetAttribLocation(self.glow_shader, "pos")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
            self.glow_vbo_instance = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.glow_vbo_instance)
            glBufferData(GL_ARRAY_BUFFER, self.glow_instance_data.nbytes, self.glow_instance_data, GL_DYNAMIC_DRAW)
            rect_loc = glGetAttribLocation(self.glow_shader, "instance_rect")
            color_loc = glGetAttribLocation(self.glow_shader, "instance_color")
            stride = self.glow_instance_data.strides[0]
            glEnableVertexAttribArray(rect_loc)
            glVertexAttribPointer(rect_loc, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glVertexAttribDivisor(rect_loc, 1)
            glEnableVertexAttribArray(color_loc)
            glVertexAttribPointer(color_loc, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
            glVertexAttribDivisor(color_loc, 1)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)

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

            if self.keyboard_bloom_shader:
                bloom_pos_loc = glGetAttribLocation(self.keyboard_bloom_shader, "pos")
                bloom_rect_loc = glGetAttribLocation(self.keyboard_bloom_shader, "instance_rect")
                bloom_pressed_loc = glGetAttribLocation(self.keyboard_bloom_shader, "instance_is_pressed")
                bloom_color_loc = glGetAttribLocation(self.keyboard_bloom_shader, "instance_color")
                self.keyboard_bloom_vao_white = glGenVertexArrays(1)
                glBindVertexArray(self.keyboard_bloom_vao_white)
                glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_quad)
                glEnableVertexAttribArray(bloom_pos_loc); glVertexAttribPointer(bloom_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
                glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_white_keys)
                glEnableVertexAttribArray(bloom_rect_loc); glVertexAttribPointer(bloom_rect_loc, 4, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(0)); glVertexAttribDivisor(bloom_rect_loc, 1)
                glEnableVertexAttribArray(bloom_pressed_loc); glVertexAttribPointer(bloom_pressed_loc, 1, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(16)); glVertexAttribDivisor(bloom_pressed_loc, 1)
                glEnableVertexAttribArray(bloom_color_loc); glVertexAttribPointer(bloom_color_loc, 3, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(20)); glVertexAttribDivisor(bloom_color_loc, 1)
                self.keyboard_bloom_vao_black = glGenVertexArrays(1)
                glBindVertexArray(self.keyboard_bloom_vao_black)
                glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_quad)
                glEnableVertexAttribArray(bloom_pos_loc); glVertexAttribPointer(bloom_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
                glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_black_keys)
                glEnableVertexAttribArray(bloom_rect_loc); glVertexAttribPointer(bloom_rect_loc, 4, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(0)); glVertexAttribDivisor(bloom_rect_loc, 1)
                glEnableVertexAttribArray(bloom_pressed_loc); glVertexAttribPointer(bloom_pressed_loc, 1, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(16)); glVertexAttribDivisor(bloom_pressed_loc, 1)
                glEnableVertexAttribArray(bloom_color_loc); glVertexAttribPointer(bloom_color_loc, 3, GL_FLOAT, GL_FALSE, instance_stride, ctypes.c_void_p(20)); glVertexAttribDivisor(bloom_color_loc, 1)

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
        self.u_timespan_loc = glGetUniformLocation(self.shader, "u_timespan")
        self.u_colors_loc = glGetUniformLocation(self.shader, "u_colors")
        self.u_pitch_layout_loc = glGetUniformLocation(self.shader, "u_pitch_layout")
        self.u_is_white_key_notes_loc = glGetUniformLocation(self.shader, "u_is_white_key")
        
        glUniform1f(glGetUniformLocation(self.shader, "u_timespan"), self._get_timespan())
        glUniform1f(glGetUniformLocation(self.shader, "u_width"), float(self.width))
        glUniform1f(glGetUniformLocation(self.shader, "u_height"), float(self.height))
        glUniform1f(glGetUniformLocation(self.shader, "u_guide_line_y"), self._get_guide_line_y())
        glUniform2fv(self.u_pitch_layout_loc, 128, self.pitch_layout_data)
        
        if self.u_is_white_key_notes_loc != -1 and self.is_white_key_data is not None:
            glUniform1iv(self.u_is_white_key_notes_loc, 128, self.is_white_key_data)

        self.u_note_texture_loc = glGetUniformLocation(self.shader, "u_note_texture")
        self.u_note_edge_texture_loc = glGetUniformLocation(self.shader, "u_note_edge_texture")
        self.u_glow_strength_loc = glGetUniformLocation(self.shader, "u_glow_strength")
        self.u_overclock_loc = glGetUniformLocation(self.shader, "u_overclock")
        glUniform1i(self.u_note_texture_loc, 0)
        glUniform1i(self.u_note_edge_texture_loc, 1)
        if self.u_glow_strength_loc != -1:
            glUniform1f(self.u_glow_strength_loc, self.glow_strength)
        if self.u_overclock_loc != -1:
            glUniform1f(self.u_overclock_loc, 0.0)

        self.u_renderer_mode_loc = glGetUniformLocation(self.shader, "u_renderer_mode")
        self.u_render_channel_loc = glGetUniformLocation(self.shader, "u_render_channel")
        self.u_lane_height_loc = glGetUniformLocation(self.shader, "u_lane_height")
        self.u_lane_guide_ratio_loc = glGetUniformLocation(self.shader, "u_lane_guide_ratio")
        self.u_channel_to_lane_loc = glGetUniformLocation(self.shader, "u_channel_to_lane")

        if self.bloom_shader:
            glUseProgram(self.bloom_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.bloom_shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
            self.u_bloom_time_loc = glGetUniformLocation(self.bloom_shader, "u_time")
            self.u_bloom_timespan_loc = glGetUniformLocation(self.bloom_shader, "u_timespan")
            self.u_bloom_pitch_layout_loc = glGetUniformLocation(self.bloom_shader, "u_pitch_layout")
            self.u_bloom_colors_loc = glGetUniformLocation(self.bloom_shader, "u_colors")
            self.u_bloom_radius_loc = glGetUniformLocation(self.bloom_shader, "u_bloom_radius")
            self.u_bloom_strength_loc = glGetUniformLocation(self.bloom_shader, "u_bloom_strength")
            glUniform1f(self.u_bloom_timespan_loc, self._get_timespan())
            glUniform1f(glGetUniformLocation(self.bloom_shader, "u_guide_line_y"), self._get_guide_line_y())
            glUniform2fv(self.u_bloom_pitch_layout_loc, 128, self.pitch_layout_data)
            if self.u_bloom_radius_loc != -1:
                glUniform1f(self.u_bloom_radius_loc, self.bloom_radius)
            if self.u_bloom_strength_loc != -1:
                glUniform1f(self.u_bloom_strength_loc, self.bloom_strength)

            self.u_bloom_renderer_mode_loc = glGetUniformLocation(self.bloom_shader, "u_renderer_mode")
            self.u_bloom_render_channel_loc = glGetUniformLocation(self.bloom_shader, "u_render_channel")
            self.u_bloom_lane_height_loc = glGetUniformLocation(self.bloom_shader, "u_lane_height")
            self.u_bloom_lane_guide_ratio_loc = glGetUniformLocation(self.bloom_shader, "u_lane_guide_ratio")
            self.u_bloom_channel_to_lane_loc = glGetUniformLocation(self.bloom_shader, "u_channel_to_lane")
            glUniform1f(glGetUniformLocation(self.bloom_shader, "u_height"), float(self.height))

        if self.screen_bloom_shader:
            glUseProgram(self.screen_bloom_shader)
            glUniform1i(glGetUniformLocation(self.screen_bloom_shader, "u_scene_texture"), 0)
            glUniform1i(glGetUniformLocation(self.screen_bloom_shader, "u_bloom_texture"), 1)
            self.u_screen_bloom_texel_size_loc = glGetUniformLocation(self.screen_bloom_shader, "u_texel_size")
            self.u_screen_bloom_strength_loc = glGetUniformLocation(self.screen_bloom_shader, "u_bloom_strength")
            if self.u_screen_bloom_texel_size_loc != -1:
                glUniform2f(self.u_screen_bloom_texel_size_loc, 1.0 / float(max(1, self.width // 2)), 1.0 / float(max(1, self.height // 2)))
            if self.u_screen_bloom_strength_loc != -1:
                glUniform1f(self.u_screen_bloom_strength_loc, self.bloom_strength)

        if self.bloom_extract_shader:
            glUseProgram(self.bloom_extract_shader)
            glUniform1i(glGetUniformLocation(self.bloom_extract_shader, "u_scene_texture"), 0)
            self.u_bloom_extract_texel_size_loc = glGetUniformLocation(self.bloom_extract_shader, "u_texel_size")
            if self.u_bloom_extract_texel_size_loc != -1:
                glUniform2f(self.u_bloom_extract_texel_size_loc, 1.0 / float(self.width), 1.0 / float(self.height))

        if self.bloom_blur_shader:
            glUseProgram(self.bloom_blur_shader)
            glUniform1i(glGetUniformLocation(self.bloom_blur_shader, "u_source_texture"), 0)
            self.u_bloom_blur_texel_size_loc = glGetUniformLocation(self.bloom_blur_shader, "u_texel_size")
            self.u_bloom_blur_direction_loc = glGetUniformLocation(self.bloom_blur_shader, "u_direction")

        self._init_scene_bloom_resources()

        if self.screen_bloom_shader and self.u_screen_bloom_texel_size_loc != -1:
            glUseProgram(self.screen_bloom_shader)
            glUniform2f(self.u_screen_bloom_texel_size_loc, 1.0 / float(max(1, self.bloom_width)), 1.0 / float(max(1, self.bloom_height)))
        if self.bloom_blur_shader and self.u_bloom_blur_texel_size_loc != -1:
            glUseProgram(self.bloom_blur_shader)
            glUniform2f(self.u_bloom_blur_texel_size_loc, 1.0 / float(max(1, self.bloom_width)), 1.0 / float(max(1, self.bloom_height)))

        self.channel_colors = _load_colors_from_xml(_COLORS_XML_PATH)
        glUseProgram(self.shader)
        glUniform3fv(self.u_colors_loc, 128, self.channel_colors)
        if self.bloom_shader and self.u_bloom_colors_loc != -1:
            glUseProgram(self.bloom_shader)
            glUniform3fv(self.u_bloom_colors_loc, 128, self.channel_colors)
        
        if self.show_keyboard:
            glUseProgram(self.keyboard_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.keyboard_shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
            glUniform1i(glGetUniformLocation(self.keyboard_shader, "u_texture_unpressed"), 0)
            glUniform1i(glGetUniformLocation(self.keyboard_shader, "u_texture_pressed"), 1)
            self.u_is_white_key_loc = glGetUniformLocation(self.keyboard_shader, "u_is_white_key")
            keyboard_glow_loc = glGetUniformLocation(self.keyboard_shader, "u_glow_strength")
            if keyboard_glow_loc != -1:
                glUniform1f(keyboard_glow_loc, self.glow_strength)
            self.u_keyboard_overclock_loc = glGetUniformLocation(self.keyboard_shader, "u_overclock")
            if self.u_keyboard_overclock_loc != -1:
                glUniform1f(self.u_keyboard_overclock_loc, 0.0)
            keyboard_time_loc = glGetUniformLocation(self.keyboard_shader, "u_time")
            if keyboard_time_loc != -1:
                glUniform1f(keyboard_time_loc, 0.0)
            if self.keyboard_bloom_shader:
                glUseProgram(self.keyboard_bloom_shader)
                glUniformMatrix4fv(glGetUniformLocation(self.keyboard_bloom_shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
                glUniform1f(glGetUniformLocation(self.keyboard_bloom_shader, "u_keyboard_y"), self.height - self.keyboard_texture_info['white_key']['scaled_height'])
                self.u_keyboard_bloom_is_white_key_loc = glGetUniformLocation(self.keyboard_bloom_shader, "u_is_white_key")
                self.u_keyboard_bloom_strength_loc = glGetUniformLocation(self.keyboard_bloom_shader, "u_bloom_strength")
                if self.u_keyboard_bloom_strength_loc != -1:
                    glUniform1f(self.u_keyboard_bloom_strength_loc, (self.bloom_base_strength if self.show_bloom else 0.0) * 0.82)
                self.u_keyboard_bloom_overclock_loc = glGetUniformLocation(self.keyboard_bloom_shader, "u_overclock")
                if self.u_keyboard_bloom_overclock_loc != -1:
                    glUniform1f(self.u_keyboard_bloom_overclock_loc, 0.0)
                keyboard_bloom_time_loc = glGetUniformLocation(self.keyboard_bloom_shader, "u_time")
                if keyboard_bloom_time_loc != -1:
                    glUniform1f(keyboard_bloom_time_loc, 0.0)

        glUseProgram(0)
        
        
        margin = 10
        self.color_button_rect = pygame.Rect(self.width - self.color_button_size - margin, margin, self.color_button_size, self.color_button_size)
        self.fun_button_rect = pygame.Rect(self.color_button_rect.x, self.color_button_rect.y + self.color_button_size + 6, self.color_button_size, self.color_button_size)
        self.color_mode_button_rect = pygame.Rect(self.color_button_rect.x - self.color_button_size - 8, margin, self.color_button_size, self.color_button_size)
        self.glow_button_rect = pygame.Rect(self.color_mode_button_rect.x - self.color_button_size - 8, margin, self.color_button_size, self.color_button_size)
        self.glow_options_button_rect = pygame.Rect(self.glow_button_rect.x, self.glow_button_rect.y + self.color_button_size + 6, self.color_button_size, 18)
        panel_width = 190
        panel_height = 124
        self.glow_options_panel_rect = pygame.Rect(
            self.glow_options_button_rect.x - (panel_width - self.glow_options_button_rect.width),
            self.glow_options_button_rect.y + self.glow_options_button_rect.height + 6,
            panel_width, panel_height
        )
        self.glow_options_checkbox_rect = pygame.Rect(self.glow_options_panel_rect.x + 10, self.glow_options_panel_rect.y + 28, 16, 16)
        self.key_light_fade_checkbox_rect = pygame.Rect(self.glow_options_panel_rect.x + 10, self.glow_options_panel_rect.y + 52, 16, 16)
        self.bloom_checkbox_rect = pygame.Rect(self.glow_options_panel_rect.x + 10, self.glow_options_panel_rect.y + 76, 16, 16)
        self.spike_bloom_checkbox_rect = pygame.Rect(self.glow_options_panel_rect.x + 10, self.glow_options_panel_rect.y + 100, 16, 16)

        fun_panel_width = 190
        fun_panel_height = 76
        self.fun_options_panel_rect = pygame.Rect(
            self.fun_button_rect.x - (fun_panel_width - self.fun_button_rect.width),
            self.fun_button_rect.y + self.fun_button_rect.height + 6,
            fun_panel_width, fun_panel_height
        )
        self.overclock_checkbox_rect = pygame.Rect(self.fun_options_panel_rect.x + 10, self.fun_options_panel_rect.y + 28, 16, 16)

        rm_labels = {"default": "Default", "channel_split": "Channel Split", "horizontal": "Horizontal View"}
        rm_label = rm_labels.get(self.renderer_mode, "Default")
        rm_w = max(100, self.overlay_font.size(rm_label)[0] + 24)
        self.renderer_mode_button_rect = pygame.Rect(self.width // 2 - rm_w // 2, 6, rm_w, 22)

    def draw(self, current_time, present=True):
        """Draw the latest buffer provided by the data thread."""
        self.last_frame_time = current_time
        t0 = time.perf_counter()
        now = t0
        dt = now - self._last_fps_time
        self._last_fps_time = now
        if dt > 0:
            self._fps_history.append(dt)
            if len(self._fps_history) > 60:
                self._fps_history.pop(0)
        if self.overclock_mode:
            if len(self._fps_history) >= 2:
                avg_dt = sum(self._fps_history) / len(self._fps_history)
                fps = 1.0 / avg_dt if avg_dt > 0 else 60.0
                fps_intensity = max(0.0, min(1.0, (60.0 - fps) / 59.0))
            else:
                fps_intensity = 0.0
            nps = self.live_nps_value
            if nps < 10000:
                nps_intensity = 0.0
            elif nps < 50000:
                nps_intensity = (nps - 10000) / 40000.0 * 0.1
            elif nps < 200000:
                nps_intensity = 0.1 + (nps - 50000) / 150000.0 * 0.2
            else:
                nps_intensity = 0.3
            self.overclock_intensity = min(1.0, 0.05 + fps_intensity + nps_intensity)
        elif not self.overclock_mode:
            self.overclock_intensity = 0.0
        self._init_slider_geometry()
        self._upload_pending_midi_data()
        if self.show_glow and (self.show_key_press_glow or self.show_key_light_fade):
            self._update_glow_trails(current_time)
            self._update_glow_cull()
        else:
            self.glow_trails.clear()
            self._split_fade_trails.clear()
        t1 = time.perf_counter()
        self._update_visible_notes(current_time)
        if self.notes_to_draw > 0:
            pitch_mod = self.last_visible_notes['pitch'].astype(np.int32) % 12
            is_sharp = np.isin(pitch_mod, [1, 3, 6, 8, 10])
            n_sharp = np.count_nonzero(is_sharp)
            if 0 < n_sharp < self.notes_to_draw:
                order = np.concatenate([np.flatnonzero(~is_sharp), np.flatnonzero(is_sharp)])
                self.last_visible_notes = self.last_visible_notes[order]
        t_vbo = time.perf_counter()
        if self.notes_to_draw > 0:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            data_size = self.last_visible_notes.nbytes
            actual_len = len(self.last_visible_notes)
            if actual_len != self.notes_to_draw:
                print(f"[VBO_DEBUG] MISMATCH: notes_to_draw={self.notes_to_draw} != len(notes)={actual_len}, clamping!")
                self.notes_to_draw = min(self.notes_to_draw, actual_len)
                data_size = self.notes_to_draw * self.note_size_bytes
            do_log = (data_size > self.vbo_stream_capacity_bytes) or (self.notes_to_draw > 500000) or self.vbo_debug
            if do_log:
                err = glGetError()
                print(f"[VBO_DEBUG] upload: n={self.notes_to_draw}  size={data_size}  cap={self.vbo_stream_capacity_bytes}  resize={data_size > self.vbo_stream_capacity_bytes}  glerr={err}")
            if data_size > self.vbo_stream_capacity_bytes:
                self.vbo_stream_capacity_bytes = data_size + 4096
                glBufferData(GL_ARRAY_BUFFER, self.vbo_stream_capacity_bytes, None, GL_STREAM_DRAW)
                err = glGetError()
                if err or do_log:
                    print(f"[VBO_DEBUG] resize: new_cap={self.vbo_stream_capacity_bytes}  glerr={err}")
            glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, self.last_visible_notes[:self.notes_to_draw])
            err = glGetError()
            if err:
                print(f"[VBO_DEBUG] GL error after glBufferSubData: {err}")
        t2 = time.perf_counter()

        is_channel_split = self.renderer_mode == 'channel_split' and len(self.active_channels) > 0
        is_horizontal = self.renderer_mode == 'horizontal'

        if self.show_bloom and self.screen_bloom_shader and self.scene_fbo:
            glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)
            glViewport(0, 0, self.width, self.height)
            self._render_scene_content(current_time)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self._render_bloom_extract()
            self._render_bloom_blur()
            glViewport(0, 0, self.width, self.height)
            self._draw_scene_bloom_composite()
        else:
            self._render_scene_content(current_time)
        t3 = time.perf_counter()
        _vbo_debug_suffix = ""
        if (self.notes_to_draw > 500000 or self.vbo_debug) and self.notes_to_draw > 0:
            err = glGetError()
            if err:
                print(f"[VBO_DEBUG] GL error after draw: {err}")
            first_on = self.last_visible_notes[0]['on_time'] if len(self.last_visible_notes) else -1
            last_on = self.last_visible_notes[min(self.notes_to_draw - 1, len(self.last_visible_notes) - 1)]['on_time'] if self.notes_to_draw > 0 else -1
            arr_len = len(self.last_visible_notes)
            _vbo_debug_suffix = f"  vbo_draw={self.notes_to_draw}  vbo_arr={arr_len}  on_rng=[{first_on:.2f},{last_on:.2f}]"

        if not self.export_mode:
            self._draw_slider_overlay()
            self._draw_capacity_warning_overlay()
            if self.live_show_stats_overlay:
                self._draw_stats_overlay(current_time)
        else:
            self._draw_stats_overlay(current_time)
        t4 = time.perf_counter()

        if hasattr(self, '_profile_count'):
            self._profile_count += 1
        else:
            self._profile_count = 0
        if self._profile_count % 60 == 0:
            t_prep = (t1 - t0) * 1000
            t_notes = (t_vbo - t1) * 1000
            t_vbo_ms = (t2 - t_vbo) * 1000
            t_render = (t3 - t2) * 1000
            t_overlay = (t4 - t3) * 1000
            t_total = (t4 - t0) * 1000
            print(f"[PROFILE] prep={t_prep:.1f}ms  notes={t_notes:.1f}ms  vbo={t_vbo_ms:.1f}ms  render={t_render:.1f}ms  overlay={t_overlay:.1f}ms  total={t_total:.1f}ms  visible_notes={self.notes_to_draw}{_vbo_debug_suffix}")

        if present:
            pygame.display.flip()

    def _init_pbo(self):
        try:
            ids = glGenBuffers(2)
            for buf_id in ids:
                glBindBuffer(GL_PIXEL_PACK_BUFFER, buf_id)
                glBufferData(GL_PIXEL_PACK_BUFFER, self.width * self.height * 3, None, GL_STREAM_READ)
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
            self.pbo_ids = list(ids)
            self._pbo_index = 0
            self._pbo_filled = False
        except Exception:
            self.pbo_ids = []

    def capture_frame_rgb(self):
        if not self.pbo_ids:
            self._init_pbo()
        if not self.pbo_ids:
            pixel_bytes = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            frame = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))
            frame = np.flipud(frame)
            return frame.tobytes()

        w, h = self.width, self.height
        size = w * h * 3
        cur = self._pbo_index
        nxt = 1 - cur

        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[cur])
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        if not self._pbo_filled:
            self._pbo_filled = True
            self._pbo_index = nxt
            pixel_bytes = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
            frame = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((h, w, 3))
            frame = np.flipud(frame)
            return frame.tobytes()

        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[nxt])
        ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        if ptr is not None:
            raw = bytes(ctypes.string_at(ptr, size))
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        else:
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
            glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
            raw = None
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        self._pbo_index = nxt

        if raw is None:
            pixel_bytes = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
            frame = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((h, w, 3))
        else:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        frame = np.flipud(frame)
        return frame.tobytes()

    def _render_scene_content(self, current_time):
        glClearColor(self.background_color[0], self.background_color[1], self.background_color[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        is_channel_split = self.renderer_mode == 'channel_split' and len(self.active_channels) > 0
        is_horizontal = self.renderer_mode == 'horizontal'

        if self.notes_to_draw > 0 and self.render_notes_array is not None:

            glDisable(GL_DEPTH_TEST)
            glUseProgram(self.shader)

            rnd_time = self._get_quantized_time(current_time)
            glUniform1f(self.u_time_loc, rnd_time)
            glUniform1f(self.u_timespan_loc, self._get_timespan())
            if self.u_overclock_loc != -1:
                glUniform1f(self.u_overclock_loc, self.overclock_intensity if not is_channel_split and not is_horizontal else 0.0)

            active_ch = np.zeros(16, dtype=np.int32)
            for ch in self.active_channels:
                if 0 <= ch < 16:
                    active_ch[ch] = 1

            channel_to_lane = np.full(16, -1, dtype=np.int32)
            for lane_idx, ch in enumerate(self.active_channels):
                if 0 <= ch < 16:
                    channel_to_lane[ch] = lane_idx

            if is_channel_split:
                visible = self.last_visible_notes[:self.notes_to_draw] if self.last_visible_notes is not None and self.notes_to_draw > 0 else None

                if visible is not None and len(visible) > 0:
                    on_t = visible['on_time']
                    off_t = visible['off_time']
                    active_mask = (on_t <= current_time) & (off_t > current_time)
                    active_notes = visible[active_mask]
                    active_count = len(active_notes)
                else:
                    active_notes = None
                    active_count = 0

                num_lanes = len(self.active_channels)
                lane_height = self.height / max(1, num_lanes)

                active_pairs = set()
                if active_count > 0:
                    for i in range(active_count):
                        ch = int(active_notes[i]['padding'])
                        p = int(active_notes[i]['pitch'])
                        active_pairs.add((ch, p))

                if self.show_key_light_fade:
                    active_color_map = {}
                    if active_count > 0:
                        for i in range(active_count):
                            note = active_notes[i]
                            ch = int(note['padding'])
                            p = int(note['pitch'])
                            color = np.array(self.channel_colors[int(note['track']) % 128], dtype=np.float32)
                            key = (ch, p)
                            existing = active_color_map.get(key)
                            if existing is None or float(note['on_time']) >= existing[1]:
                                active_color_map[key] = (color, float(note['on_time']))

                    for key, (color, _) in active_color_map.items():
                        self._split_fade_trails[key] = {
                            'color': color,
                            'fade_start': current_time,
                        }

                    expired = []
                    for key, trail in self._split_fade_trails.items():
                        if key in active_pairs:
                            continue
                        elapsed = current_time - trail['fade_start']
                        if elapsed > self.glow_fade_duration:
                            expired.append(key)
                    for key in expired:
                        del self._split_fade_trails[key]

                glUniform1i(self.u_renderer_mode_loc, 2)
                glUniform1f(self.u_lane_height_loc, lane_height)
                glUniform1f(self.u_lane_guide_ratio_loc, 0.82)
                if self.u_channel_to_lane_loc != -1:
                    glUniform1iv(self.u_channel_to_lane_loc, 16, channel_to_lane)

                glDisable(GL_BLEND)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.note_texture)
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, self.note_edge_texture)

                if active_count > 0:
                    glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
                    data_size = active_notes.nbytes
                    if data_size > self.vbo_stream_capacity_bytes:
                        self.vbo_stream_capacity_bytes = data_size
                    glBufferData(GL_ARRAY_BUFFER, self.vbo_stream_capacity_bytes, None, GL_DYNAMIC_DRAW)
                    glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, active_notes)
                    glBindVertexArray(self.vao)
                    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, active_count)
                    glBindVertexArray(0)

                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glUseProgram(0)
                glDisable(GL_DEPTH_TEST)

                if self.show_key_light_fade and self._split_fade_trails:
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
                    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
                    cell_w = self.width / 128.0
                    for (ch, p), trail in self._split_fade_trails.items():
                        if (ch, p) in active_pairs:
                            continue
                        elapsed = current_time - trail['fade_start']
                        fade = 1.0 - (elapsed / self.glow_fade_duration)
                        if fade <= 0.0:
                            continue
                        if ch not in self.active_channels:
                            continue
                        lane_idx = self.active_channels.index(ch)
                        r, g, b = trail['color']
                        glColor4f(r, g, b, fade * 0.7)
                        x0 = p * cell_w
                        y0 = lane_idx * lane_height
                        glBegin(GL_QUADS)
                        glVertex2f(x0, y0)
                        glVertex2f(x0 + cell_w, y0)
                        glVertex2f(x0 + cell_w, y0 + lane_height)
                        glVertex2f(x0, y0 + lane_height)
                        glEnd()
                    glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
                    glDisable(GL_BLEND)

                glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
                glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
                glLineWidth(0.5)
                glColor4f(0.35, 0.35, 0.45, 0.15)
                for row in range(17):
                    y = int(row * lane_height)
                    glBegin(GL_LINES)
                    glVertex2f(0, y); glVertex2f(self.width, y)
                    glEnd()
                glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

                for lane_idx, ch in enumerate(self.active_channels):
                    y = int(lane_idx * lane_height)
                    self._draw_text_overlay(f"CH {ch}", 6, y + 3, color=(160, 170, 190), alpha=0.6)

            else:
                if is_horizontal:
                    glUniform1i(self.u_renderer_mode_loc, 3)
                else:
                    glUniform1i(self.u_renderer_mode_loc, 0)
                    glUniform1f(glGetUniformLocation(self.shader, "u_guide_line_y"), self._get_guide_line_y())
                if self.u_channel_to_lane_loc != -1:
                    glUniform1iv(self.u_channel_to_lane_loc, 16, channel_to_lane)

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.note_texture)
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, self.note_edge_texture)

        glDisable(GL_DEPTH_TEST)
        if not is_channel_split:
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self.notes_to_draw)
            if self.notes_to_draw > 500000 or self.vbo_debug:
                err = glGetError()
                if err:
                    print(f"[VBO_DEBUG] GL error at draw call (n={self.notes_to_draw}): {err}")

        glBindVertexArray(0)
        glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        if is_channel_split or is_horizontal:
            pass
        else:
            if self.show_glow and self.show_key_press_glow:
                self._draw_active_note_glow_overlay(current_time)
            if self.show_keyboard and self.keyboard_layout:
                self._draw_keyboard_opengl(current_time)

        if not is_channel_split and not is_horizontal and self.show_guide_line:
            glDisable(GL_DEPTH_TEST)
            glUseProgram(0)
            glMatrixMode(GL_PROJECTION);glPushMatrix();glLoadIdentity();glOrtho(0,self.width,self.height,0,-1,1)
            glMatrixMode(GL_MODELVIEW);glPushMatrix();glLoadIdentity()
            guide_y = self._get_guide_line_y()
            kb_area_h = self.height - guide_y
            transition_h = max(3.0, kb_area_h * 0.02)
            ribbon_h = kb_area_h * 0.05
            spacer_h = 2.0
            total_h = transition_h + ribbon_h + spacer_h
            top_y = guide_y - total_h
            bg = self.background_color
            glBegin(GL_QUADS)
            glColor4f(bg[0], bg[1], bg[2], 1.0); glVertex2f(0, top_y)
            glColor4f(bg[0], bg[1], bg[2], 1.0); glVertex2f(self.width, top_y)
            glColor4f(0.03, 0.03, 0.05, 1.0); glVertex2f(self.width, top_y + transition_h)
            glColor4f(0.03, 0.03, 0.05, 1.0); glVertex2f(0, top_y + transition_h)
            glEnd()
            glBegin(GL_QUADS)
            ry = top_y + transition_h
            glColor3f(0.55, 0.08, 0.06); glVertex2f(0, ry)
            glColor3f(0.55, 0.08, 0.06); glVertex2f(self.width, ry)
            glColor3f(0.35, 0.04, 0.03); glVertex2f(self.width, ry + ribbon_h)
            glColor3f(0.35, 0.04, 0.03); glVertex2f(0, ry + ribbon_h)
            glEnd()
            sy = ry + ribbon_h
            glColor3f(0.08, 0.08, 0.12)
            glBegin(GL_QUADS); glVertex2f(0, sy); glVertex2f(self.width, sy); glVertex2f(self.width, sy + spacer_h); glVertex2f(0, sy + spacer_h); glEnd()
            glPopMatrix();glMatrixMode(GL_PROJECTION);glPopMatrix();glMatrixMode(GL_MODELVIEW)

    def _draw_keyboard_channel_lane(self, current_time, channel, lane_idx, lane_height):
        if self.keyboard_shader == 0 or not self.keyboard_layout:
            return
        lane_top = lane_idx * lane_height
        lane_bottom = lane_top + lane_height
        kb_ratio = 0.18
        kb_height = lane_height * kb_ratio
        kb_y = lane_bottom - kb_height

        orig_kb_h = self.keyboard_texture_info['white_key']['scaled_height']
        if orig_kb_h <= 0:
            return
        scale = kb_height / orig_kb_h

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_SCISSOR_TEST)
        glScissor(0, int(lane_top), self.width, int(lane_height))

        glUseProgram(self.keyboard_shader)
        glUniform1f(glGetUniformLocation(self.keyboard_shader, "u_keyboard_y"), kb_y)
        if self.u_keyboard_overclock_loc != -1:
            glUniform1f(self.u_keyboard_overclock_loc, 0.0)
        kb_time_loc = glGetUniformLocation(self.keyboard_shader, "u_time")
        if kb_time_loc != -1:
            glUniform1f(kb_time_loc, self.last_frame_time)

        scaled_white = self.white_key_instance_data.copy()
        scaled_black = self.black_key_instance_data.copy()
        scaled_white['rect'][:, 1] = 0.0
        scaled_white['rect'][:, 3] *= scale
        scaled_black['rect'][:, 1] = 0.0
        scaled_black['rect'][:, 3] *= scale

        if len(scaled_white) > 0:
            scaled_white['is_pressed'].fill(0.0)
            scaled_white['color'][:] = 0.0
        if len(scaled_black) > 0:
            scaled_black['is_pressed'].fill(0.0)
            scaled_black['color'][:] = 0.0

        if self.last_visible_notes is not None and len(self.last_visible_notes) > 0:
            ch_mask = self.last_visible_notes['padding'] == channel
            ch_notes = self.last_visible_notes[ch_mask]
            if len(ch_notes) > 0:
                active_mask = (ch_notes['on_time'] <= current_time) & (ch_notes['off_time'] > current_time)
                active_notes = ch_notes[active_mask]
                for note in active_notes:
                    pitch = int(note['pitch'])
                    ch = int(note['padding'])
                    color = np.array(self.channel_colors[ch % 128], dtype=np.float32)
                    if pitch in self.white_key_pitch_map:
                        idx = self.white_key_pitch_map[pitch]
                        scaled_white[idx]['color'] = np.clip(color * 0.7, 0.0, 1.0)
                        scaled_white[idx]['is_pressed'] = 1.0
                    elif pitch in self.black_key_pitch_map:
                        idx = self.black_key_pitch_map[pitch]
                        scaled_black[idx]['color'] = np.clip(color * 0.7, 0.0, 1.0)
                        scaled_black[idx]['is_pressed'] = 1.0

        if len(scaled_white) > 0:
            glBindVertexArray(self.keyboard_vao_white)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_white_keys)
            glBufferSubData(GL_ARRAY_BUFFER, 0, scaled_white.nbytes, scaled_white)
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, len(scaled_white))

        if len(scaled_black) > 0:
            glBindVertexArray(self.keyboard_vao_black)
            glBindBuffer(GL_ARRAY_BUFFER, self.keyboard_vbo_black_keys)
            glBufferSubData(GL_ARRAY_BUFFER, 0, scaled_black.nbytes, scaled_black)
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, len(scaled_black))

        glBindVertexArray(0)
        glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_SCISSOR_TEST)

    def cleanup(self):
        self.app_running.clear()
        self._text_texture_cache.clear()

        glDeleteProgram(self.shader)
        if self.bloom_shader:
            glDeleteProgram(self.bloom_shader)
        if self.screen_bloom_shader:
            glDeleteProgram(self.screen_bloom_shader)
        if self.bloom_extract_shader:
            glDeleteProgram(self.bloom_extract_shader)
        if self.bloom_blur_shader:
            glDeleteProgram(self.bloom_blur_shader)
        glDeleteVertexArrays(1, [self.vao])
        if self.bloom_vao:
            glDeleteVertexArrays(1, [self.bloom_vao])
        if self.screen_bloom_vao:
            glDeleteVertexArrays(1, [self.screen_bloom_vao])
        glDeleteBuffers(2, [self.vbo_vertices, self.vbo_stream_data])
        if self.pbo_ids:
            glDeleteBuffers(2, self.pbo_ids)
            self.pbo_ids = []
        if self.screen_bloom_vbo:
            glDeleteBuffers(1, [self.screen_bloom_vbo])
        if self.scene_depth_rbo:
            glDeleteRenderbuffers(1, [self.scene_depth_rbo])
        if self.scene_color_texture:
            glDeleteTextures(1, [self.scene_color_texture])
        if self.bloom_texture:
            glDeleteTextures(1, [self.bloom_texture])
        if self.bloom_blur_texture:
            glDeleteTextures(1, [self.bloom_blur_texture])
        if self.scene_fbo:
            glDeleteFramebuffers(1, [self.scene_fbo])
        if self.bloom_fbo:
            glDeleteFramebuffers(1, [self.bloom_fbo])
        if self.bloom_blur_fbo:
            glDeleteFramebuffers(1, [self.bloom_blur_fbo])
        
        if self.show_keyboard:
            glDeleteProgram(self.keyboard_shader)
            if self.keyboard_bloom_shader:
                glDeleteProgram(self.keyboard_bloom_shader)
            glDeleteVertexArrays(2, [self.keyboard_vao_white, self.keyboard_vao_black])
            if self.keyboard_bloom_vao_white or self.keyboard_bloom_vao_black:
                glDeleteVertexArrays(2, [self.keyboard_bloom_vao_white, self.keyboard_bloom_vao_black])
            glDeleteBuffers(3, [self.keyboard_vbo_quad, self.keyboard_vbo_white_keys, self.keyboard_vbo_black_keys])
            glDeleteTextures(list(self.keyboard_textures.values()))

        if self.note_texture:
            glDeleteTextures(2, [self.note_texture, self.note_edge_texture])
        if self.glow_shader:
            glDeleteProgram(self.glow_shader)
        if self.glow_vao:
            glDeleteVertexArrays(1, [self.glow_vao])
        if self.glow_vbo_instance:
            glDeleteBuffers(1, [self.glow_vbo_instance])

        pygame.quit()
