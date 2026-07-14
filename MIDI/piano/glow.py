import numpy as np
from OpenGL.GL import *


class GlowMixin:
    """Glow trail and overlay methods for PianoRoll."""

    def _update_glow_cull(self):
        """Adaptive glow culling - reduces glow when many notes are active."""
        total_notes = sum(t.get('live_count', 1) for t in self.glow_trails.values())
        threshold = max(1, self.glow_cull_threshold)
        if total_notes <= threshold:
            self.glow_cull_factor = 1.0
        else:
            over = (total_notes - threshold) / threshold
            self.glow_cull_factor = max(0.0, 1.0 - over)

    def _update_glow_trails(self, current_time):
        if self.last_glow_time is not None and current_time + 0.001 < self.last_glow_time:
            self.glow_trails.clear()
            self._split_fade_trails.clear()
        self.last_glow_time = current_time

        active_pitch_data = {}
        if self.last_visible_notes is not None and len(self.last_visible_notes) > 0:
            active_mask = (self.last_visible_notes['on_time'] <= current_time) & (self.last_visible_notes['off_time'] > current_time)
            active_notes = self.last_visible_notes[active_mask]
            for note in active_notes:
                pitch = int(note['pitch'])
                track = int(note['track'])
                color = self.channel_colors[track % 128]
                on_time = float(note['on_time'])
                if pitch not in active_pitch_data:
                    active_pitch_data[pitch] = {
                        'pitch': pitch,
                        'color_sum': np.array(color, dtype=np.float32),
                        'count': 1,
                        'on_time': on_time,
                    }
                else:
                    active_pitch_data[pitch]['color_sum'] += color
                    active_pitch_data[pitch]['count'] += 1
                    if on_time > active_pitch_data[pitch]['on_time']:
                        active_pitch_data[pitch]['on_time'] = on_time

            for pitch, pitch_data in active_pitch_data.items():
                avg_color = pitch_data['color_sum'] / max(1, pitch_data['count'])
                existing = self.glow_trails.get(pitch)
                note_on = pitch_data['on_time']
                if existing and existing.get('on_time', 0.0) >= note_on:
                    existing['live_count'] = 1
                    existing['color'] = avg_color
                    existing['fade_start_time'] = current_time
                else:
                    self.glow_trails[pitch] = {
                        'pitch': pitch,
                        'color': avg_color,
                        'live_count': 1,
                        'on_time': note_on,
                        'fade_start_time': current_time,
                    }

        expired_keys = []
        for key, trail in list(self.glow_trails.items()):
            elapsed = max(0.0, current_time - trail.get('fade_start_time', current_time))
            fade = 1.0 - (elapsed / self.glow_fade_duration)
            if fade <= 0.0:
                expired_keys.append(key)

        for key in expired_keys:
            self.glow_trails.pop(key, None)

    def _draw_active_note_glow_overlay(self, current_time):
        if not self.glow_trails or self.glow_cull_factor <= 0.01 or not self.glow_shader:
            return

        guide_y = self._get_guide_line_y()
        active_pitch_keys = set()
        if self.last_visible_notes is not None and len(self.last_visible_notes) > 0:
            active_mask = (self.last_visible_notes['on_time'] <= current_time) & (self.last_visible_notes['off_time'] > current_time)
            active_notes = self.last_visible_notes[active_mask]
            for note in active_notes:
                active_pitch_keys.add(int(note['pitch']))

        count = 0
        for pitch, trail in self.glow_trails.items():
            elapsed = max(0.0, current_time - trail.get('fade_start_time', current_time))
            fade = 1.0 - (elapsed / self.glow_fade_duration)
            if fade <= 0.0:
                continue

            key_info = self.keyboard_layout.get(pitch)
            if not key_info:
                continue

            color = trail['color']
            is_active = pitch in active_pitch_keys
            alpha_boost = (1.0 if is_active else 0.9) * self.glow_cull_factor
            alpha = fade * alpha_boost

            quad_w = max(key_info['width'] * 8.0, 120.0)
            quad_h = 160.0 if key_info['type'] == 'black' else 240.0

            center_x = key_info['x'] + key_info['width'] * 0.5
            center_y = guide_y - 2.0

            x0 = center_x - quad_w * 0.5
            y0 = center_y - quad_h * 0.5

            if x0 + quad_w < 0 or x0 > self.width or y0 + quad_h < 0 or y0 > self.height:
                continue

            self.glow_instance_data[count]['rect'] = (x0, y0, quad_w, quad_h)
            self.glow_instance_data[count]['color'] = (color[0], color[1], color[2], alpha)
            count += 1

        if count == 0:
            return

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        glUseProgram(self.glow_shader)
        glUniformMatrix4fv(glGetUniformLocation(self.glow_shader, "u_projection"), 1, GL_FALSE, self.projection_matrix.T)
        glUniform1f(glGetUniformLocation(self.glow_shader, "u_time"), current_time)

        glBindVertexArray(self.glow_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.glow_vbo_instance)
        glBufferSubData(GL_ARRAY_BUFFER, 0, count * self.glow_instance_data.itemsize, self.glow_instance_data[:count])
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count)

        glBindVertexArray(0)
        glUseProgram(0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _get_trail_fade(self, current_time, pitch):
        trail = self.glow_trails.get(pitch)
        if not trail:
            return None, 0.0
        elapsed = max(0.0, current_time - trail.get('fade_start_time', current_time))
        fade = 1.0 - (elapsed / self.glow_fade_duration)
        if fade <= 0.0:
            return None, 0.0
        return trail, fade
