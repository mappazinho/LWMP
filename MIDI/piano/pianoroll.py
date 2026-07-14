import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import threading
import queue
import numpy as np
import ctypes
import os
import sys

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
    _order_visible_notes_for_draw, _assign_note_depths,
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
        self.u_bloom_scroll_speed_loc = -1
        self.u_bloom_pitch_layout_loc = -1
        self.u_bloom_colors_loc = -1
        self.u_bloom_window_start_loc = -1
        self.u_bloom_window_end_loc = -1
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
        self.data_queue = queue.Queue(maxsize=2)
        self.app_running = threading.Event()
        self.app_running.set()
        self.force_data_update = threading.Event()
        self.data_thread = None
        self.last_stream_signature = None
        self.notes_to_draw = 0
        self.current_note_data = None
        self.get_current_time = None
        self.last_frame_time = 0.0
        self.export_total_duration = 0.0
        self.capacity_warning_active = False
        self.capacity_warning_visible_count = 0
        
        vis_cfg = self.config.get('visualizer', {})
        gui_cfg = self.config.get('gui', {})
        self.scroll_speed = float(vis_cfg.get('scroll_speed', 2500.0))
        self.scroll_slider_min = 200.0
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
        self.anesthesia_mode = bool(vis_cfg.get('anesthesia_mode', False))
        self.hide_buttons = bool(vis_cfg.get('hide_buttons', False))
        self.glow_cull_threshold = int(vis_cfg.get('glow_cull_threshold', 128))
        self.glow_fade_duration = 0.1
        self.nps_spikes = []
        self._spike_rank_map = {}
        self.spike_bloom_rise_duration = 1.75
        self.spike_bloom_fall_duration = 1.75
        self.spike_bloom_min_boost = 0.14
        self.spike_bloom_max_boost = 0.48
        
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
        self.glow_options_checkbox_rect = None
        self.key_light_fade_checkbox_rect = None
        self.bloom_checkbox_rect = None
        self.spike_bloom_checkbox_rect = None
        self.glow_options_expanded = False
        self.fun_options_expanded = False
        self.fun_options_panel_rect = None
        self.overclock_mode = False
        self.overclock_intensity = 0.0
        self.anesthesia_mode = False
        self.anesthesia_shrink = 0.0
        self.anesthesia_remove = 0.0
        self.anesthesia_checkbox_rect = None
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
        vis_cfg['anesthesia_mode'] = bool(self.anesthesia_mode)
        vis_cfg['hide_buttons'] = bool(self.hide_buttons)
        vis_cfg['seconds_before_cursor'] = float(self.window_seconds)
        vis_cfg['seconds_after_cursor'] = float(self.window_seconds)
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
            self.force_data_update.set()

    def set_preferred_color_mode(self, mode):
        mode = "channel" if mode == "channel" else "track"
        self.preferred_color_mode = mode
        self.note_color_mode = mode
        self._rebuild_color_mode_render_data()

    def toggle_note_color_mode(self):
        self.note_color_mode = "channel" if self.note_color_mode == "track" else "track"
        self._rebuild_color_mode_render_data()
        print(f"Piano roll note colors switched to {self.note_color_mode}.")

    def _visible_slice_signature(self, visible_slice, visible_count):
        if visible_count <= 0 or visible_slice is None or len(visible_slice) == 0:
            return (0,)
        first = visible_slice[0]
        last = visible_slice[-1]
        mid = visible_slice[visible_count // 2]
        return (
            int(visible_count),
            float(first['on_time']), float(first['off_time']), int(first['pitch']), int(first['track']),
            float(mid['on_time']), float(mid['off_time']), int(mid['pitch']), int(mid['track']),
            float(last['on_time']), float(last['off_time']), int(last['pitch']), int(last['track']),
        )

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
        self.last_glow_time = None
        self.base_render_notes, self.base_render_on_times = _build_base_render_data(all_notes_gpu)
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
        self.last_stream_signature = None
        self.notes_to_draw = 0
        self.last_visible_notes = np.empty(0, dtype=RENDER_NOTE_DTYPE)
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
                self.capacity_warning_active = False
                self.capacity_warning_visible_count = 0
                
                view_start = now - self.seconds_before_cursor
                view_end = now + self.seconds_after_cursor

                if self.anesthesia_mode and self.anesthesia_shrink > 0.001:
                    guide_y = self._get_guide_line_y()
                    full_after = max(0.1, (self.height - guide_y) / max(1.0, self.scroll_speed))
                    full_before = max(0.1, (guide_y - 20.0) / max(1.0, self.scroll_speed))
                    window_scale = max(0.03, 1.0 - self.anesthesia_shrink)
                    view_start = now - full_before * window_scale
                    view_end = now + full_after * window_scale

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
                    self.capacity_warning_active = True
                    self.capacity_warning_visible_count = int(visible_count)
                    active_or_distance = np.where(
                        visible_slice['on_time'] <= now,
                        np.where(visible_slice['off_time'] > now, 0.0, now - visible_slice['off_time']),
                        visible_slice['on_time'] - now,
                    )
                    keep_idx = np.argpartition(active_or_distance, self.streaming_vbo_capacity - 1)[:self.streaming_vbo_capacity]
                    visible_slice = visible_slice[np.sort(keep_idx)]
                    visible_count = self.streaming_vbo_capacity
                visible_slice = _order_visible_notes_for_draw(visible_slice, self.is_white_key_data)

                visible_slice_contiguous = np.ascontiguousarray(visible_slice)
                _assign_note_depths(visible_slice_contiguous)
                visible_signature = self._visible_slice_signature(visible_slice_contiguous, visible_count)
                if visible_signature == self.last_stream_signature:
                    time.sleep(0.005)
                    continue
                self.last_stream_signature = visible_signature

                try:
                    self.data_queue.put((visible_slice_contiguous, visible_count), block=True, timeout=0.1)
                except queue.Full:
                    print("Piano roll queue is full, frame skipped.")
            
            time.sleep(0.005)

    def _slice_visible_notes_for_time(self, current_time):
        if self.render_notes_array is None or self.render_on_times is None:
            return np.empty(0, dtype=RENDER_NOTE_DTYPE), 0

        self.capacity_warning_active = False
        self.capacity_warning_visible_count = 0
        view_start = current_time - self.seconds_before_cursor
        view_end = current_time + self.seconds_after_cursor

        if self.anesthesia_mode and self.anesthesia_shrink > 0.001:
            guide_y = self._get_guide_line_y()
            full_after = max(0.1, (self.height - guide_y) / max(1.0, self.scroll_speed))
            full_before = max(0.1, (guide_y - 20.0) / max(1.0, self.scroll_speed))
            window_scale = max(0.005, 1.0 - self.anesthesia_shrink)
            view_start = current_time - full_before * window_scale
            view_end = current_time + full_after * window_scale

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
            self.capacity_warning_active = True
            self.capacity_warning_visible_count = int(visible_count)
            active_or_distance = np.where(
                visible_slice['on_time'] <= current_time,
                np.where(visible_slice['off_time'] > current_time, 0.0, current_time - visible_slice['off_time']),
                visible_slice['on_time'] - current_time,
            )
            keep_idx = np.argpartition(active_or_distance, self.streaming_vbo_capacity - 1)[:self.streaming_vbo_capacity]
            visible_slice = visible_slice[np.sort(keep_idx)]
            visible_count = self.streaming_vbo_capacity

        visible_slice = _order_visible_notes_for_draw(visible_slice, self.is_white_key_data)
        visible_count = len(visible_slice)
        visible_slice_contiguous = np.ascontiguousarray(visible_slice)
        _assign_note_depths(visible_slice_contiguous)
        return visible_slice_contiguous, visible_count

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
        note_depth_loc = glGetAttribLocation(self.shader, "note_depth")

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
        glBufferData(GL_ARRAY_BUFFER, self.vbo_stream_capacity_bytes, None, GL_DYNAMIC_DRAW)
        
        glEnableVertexAttribArray(note_times_loc); glVertexAttribPointer(note_times_loc, 2, GL_FLOAT, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(0)); glVertexAttribDivisor(note_times_loc, 1)
        glEnableVertexAttribArray(note_info_loc)
        glVertexAttribPointer(note_info_loc, 3, GL_UNSIGNED_BYTE, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(8))
        glVertexAttribDivisor(note_info_loc, 1)
        glEnableVertexAttribArray(note_depth_loc)
        glVertexAttribPointer(note_depth_loc, 1, GL_FLOAT, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(12))
        glVertexAttribDivisor(note_depth_loc, 1)

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
            glVertexAttribPointer(bloom_note_info_loc, 3, GL_UNSIGNED_BYTE, GL_FALSE, self.note_size_bytes, ctypes.c_void_p(8))
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
        self.u_overclock_loc = glGetUniformLocation(self.shader, "u_overclock")
        glUniform1i(self.u_note_texture_loc, 0)
        glUniform1i(self.u_note_edge_texture_loc, 1)
        if self.u_glow_strength_loc != -1:
            glUniform1f(self.u_glow_strength_loc, self.glow_strength)
        if self.u_overclock_loc != -1:
            glUniform1f(self.u_overclock_loc, 0.0)

        if self.bloom_shader:
            glUseProgram(self.bloom_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.bloom_shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
            self.u_bloom_time_loc = glGetUniformLocation(self.bloom_shader, "u_time")
            self.u_bloom_scroll_speed_loc = glGetUniformLocation(self.bloom_shader, "u_scroll_speed")
            self.u_bloom_pitch_layout_loc = glGetUniformLocation(self.bloom_shader, "u_pitch_layout")
            self.u_bloom_colors_loc = glGetUniformLocation(self.bloom_shader, "u_colors")
            self.u_bloom_window_start_loc = glGetUniformLocation(self.bloom_shader, "u_window_start")
            self.u_bloom_window_end_loc = glGetUniformLocation(self.bloom_shader, "u_window_end")
            self.u_bloom_radius_loc = glGetUniformLocation(self.bloom_shader, "u_bloom_radius")
            self.u_bloom_strength_loc = glGetUniformLocation(self.bloom_shader, "u_bloom_strength")
            glUniform1f(self.u_bloom_scroll_speed_loc, self.scroll_speed)
            glUniform1f(glGetUniformLocation(self.bloom_shader, "u_guide_line_y"), self._get_guide_line_y())
            glUniform2fv(self.u_bloom_pitch_layout_loc, 128, self.pitch_layout_data)
            if self.u_bloom_window_start_loc != -1:
                glUniform1f(self.u_bloom_window_start_loc, 0.0)
            if self.u_bloom_window_end_loc != -1:
                glUniform1f(self.u_bloom_window_end_loc, 0.0)
            if self.u_bloom_radius_loc != -1:
                glUniform1f(self.u_bloom_radius_loc, self.bloom_radius)
            if self.u_bloom_strength_loc != -1:
                glUniform1f(self.u_bloom_strength_loc, self.bloom_strength)

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
        
        if not self.export_mode:
            self.data_thread = threading.Thread(target=self._data_streamer_thread, daemon=True)
            self.data_thread.start()
        
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
        self.anesthesia_checkbox_rect = pygame.Rect(self.fun_options_panel_rect.x + 10, self.fun_options_panel_rect.y + 52, 16, 16)

    def draw(self, current_time, present=True):
        """Draw the latest buffer provided by the data thread."""
        self.last_frame_time = current_time
        now = time.perf_counter()
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
        if self.anesthesia_mode:
            nps = self.live_nps_value
            if nps < 50000:
                self.anesthesia_shrink = 0.0
                self.anesthesia_remove = 0.0
            elif nps < 150000:
                raw = (nps - 50000) / 100000.0
                self.anesthesia_shrink = raw * raw * 0.7
                self.anesthesia_remove = 0.0
            elif nps < 200000:
                self.anesthesia_shrink = 0.7 + (nps - 150000) / 50000.0 * 0.15
                self.anesthesia_remove = min(1.0, (nps - 150000) / 50000.0)
            elif nps < 1000000:
                raw = (nps - 200000) / 800000.0
                self.anesthesia_shrink = 0.85 + raw * 0.15
                self.anesthesia_remove = 1.0
            else:
                self.anesthesia_shrink = 1.0
                self.anesthesia_remove = 1.0
        else:
            self.anesthesia_shrink = 0.0
            self.anesthesia_remove = 0.0
        self._init_slider_geometry()
        self._upload_pending_midi_data()
        if self.show_glow and (self.show_key_press_glow or self.show_key_light_fade):
            self._update_glow_trails(current_time)
            self._update_glow_cull()
        else:
            self.glow_trails.clear()

        if self.export_mode:
            visible_notes, count = self._slice_visible_notes_for_time(current_time)
            self.last_visible_notes = visible_notes
            self.notes_to_draw = count
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            if count > 0:
                glBufferSubData(GL_ARRAY_BUFFER, 0, visible_notes.nbytes, visible_notes)
        else:
            try:
                queue_data = self.data_queue.get_nowait()
                visible_notes, count = queue_data
                self.last_visible_notes = visible_notes
                self.notes_to_draw = count
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
                if count > 0:
                    glBufferSubData(GL_ARRAY_BUFFER, 0, visible_notes.nbytes, visible_notes)
            except queue.Empty:
                pass

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

        if not self.export_mode:
            self._draw_slider_overlay()
            self._draw_capacity_warning_overlay()
            if self.live_show_stats_overlay:
                self._draw_stats_overlay(current_time)
        else:
            self._draw_stats_overlay(current_time)

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

        if self.notes_to_draw > 0 and self.render_notes_array is not None:
            window_start = current_time - self.seconds_before_cursor
            window_end = current_time + self.seconds_after_cursor

            if self.anesthesia_mode and self.anesthesia_shrink > 0.001:
                guide_y = self._get_guide_line_y()
                full_after = max(0.1, (self.height - guide_y) / max(1.0, self.scroll_speed))
                full_before = max(0.1, (guide_y - 20.0) / max(1.0, self.scroll_speed))
                window_scale = max(0.03, 1.0 - self.anesthesia_shrink)
                window_start = current_time - full_before * window_scale
                window_end = current_time + full_after * window_scale

            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)
            glUseProgram(self.shader)

            glUniform1f(self.u_time_loc, current_time)
            glUniform1f(self.u_scroll_speed_loc, self.scroll_speed)
            glUniform1f(glGetUniformLocation(self.shader, "u_guide_line_y"), self._get_guide_line_y())
            glUniform1f(self.u_window_start_loc, window_start)
            glUniform1f(self.u_window_end_loc, window_end)
            if self.u_overclock_loc != -1:
                glUniform1f(self.u_overclock_loc, self.overclock_intensity)

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
            glDisable(GL_DEPTH_TEST)

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

    def cleanup(self):
        self.app_running.clear()
        if self.data_thread:
            self.data_thread.join(timeout=0.5)

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
