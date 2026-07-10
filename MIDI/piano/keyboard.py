import os
import numpy as np
import ctypes
from OpenGL.GL import *

from piano.skin_utils import _SKIN_DIR


class KeyboardMixin:
    """Keyboard layout, asset loading, and rendering methods for PianoRoll."""

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
        except Exception as e:
            print(f"Could not load keyboard skin assets from '{skin_dir}': {e}")
            self.show_keyboard = False
            return

        self.keyboard_layout = {}
        sharp_ratio = 0.65
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

        start_pitch = 0
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

        def white_count_before(pitch):
            return sum(1 for i in range(start_pitch, pitch) if key_is_white[i])

        def black_key_nudge(note_val):
            if note_val in (1, 6):
                return -(sharp_ratio / 5.0)
            if note_val in (3, 10):
                return sharp_ratio / 5.0
            return 0.0

        black_key_width = white_key_width * sharp_ratio
        black_key_height = white_key_height * 0.65

        for pitch in range(128):
            if not key_is_white[pitch]:
                note_val = pitch % 12
                start_offset = -sharp_ratio / 2.0
                x_pos = white_key_width * (white_count_before(pitch) + start_offset + black_key_nudge(note_val))
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

        if self.show_glow and self.show_key_light_fade:
            for pitch, trail in self.glow_trails.items():
                if pitch in active_pitch_colors:
                    continue
                trail_obj, fade = self._get_trail_fade(current_time, pitch)
                if trail_obj is None or fade <= 0.0:
                    continue
                active_pitch_colors[pitch] = {
                    'color': np.array(trail_obj['color'], dtype=np.float32),
                    'on_time': -1.0,
                    'press_fade': fade,
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
                        existing, effective_weight, distance, self._color_luminance(base_color),
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
            press_val = float(pitch_color_data.get('press_fade', 1.0))
            if pitch in self.white_key_pitch_map:
                idx = self.white_key_pitch_map[pitch]
                self.white_key_instance_data[idx]['color'] = active_color
                self.white_key_instance_data[idx]['is_pressed'] = press_val
            elif pitch in self.black_key_pitch_map:
                idx = self.black_key_pitch_map[pitch]
                self.black_key_instance_data[idx]['color'] = active_color
                self.black_key_instance_data[idx]['is_pressed'] = press_val
        
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

    def _draw_keyboard_bloom(self):
        if self.keyboard_bloom_shader == 0:
            return

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glUseProgram(self.keyboard_bloom_shader)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        keyboard_height = self.keyboard_texture_info['white_key']['scaled_height']
        keyboard_y = self.height - keyboard_height
        glUniform1f(glGetUniformLocation(self.keyboard_bloom_shader, "u_keyboard_y"), keyboard_y)
        if self.u_keyboard_bloom_strength_loc != -1:
            glUniform1f(
                self.u_keyboard_bloom_strength_loc,
                (self.bloom_base_strength if self.show_bloom else 0.0) * 0.82
            )

        glUniform1i(self.u_keyboard_bloom_is_white_key_loc, 1)
        glBindVertexArray(self.keyboard_bloom_vao_white)
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, len(self.white_key_instance_data))

        if len(self.black_key_instance_data) > 0:
            glUniform1i(self.u_keyboard_bloom_is_white_key_loc, 0)
            glBindVertexArray(self.keyboard_bloom_vao_black)
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, len(self.black_key_instance_data))

        glBindVertexArray(0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(0)
        glDepthMask(GL_TRUE)
