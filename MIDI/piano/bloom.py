import time
import numpy as np
from OpenGL.GL import *


class BloomMixin:
    """Bloom rendering methods for PianoRoll."""

    def _init_scene_bloom_resources(self):
        if not self.screen_bloom_shader:
            return

        self.scene_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)

        self.scene_color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.scene_color_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.scene_color_texture, 0)

        self.scene_depth_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.scene_depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.scene_depth_rbo)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Scene bloom framebuffer incomplete: {status}")
            if self.scene_depth_rbo:
                glDeleteRenderbuffers(1, [self.scene_depth_rbo])
                self.scene_depth_rbo = 0
            if self.scene_color_texture:
                glDeleteTextures(1, [self.scene_color_texture])
                self.scene_color_texture = 0
            if self.scene_fbo:
                glDeleteFramebuffers(1, [self.scene_fbo])
                self.scene_fbo = 0

        self.bloom_width = max(1, (self.width * 3) // 4)
        self.bloom_height = max(1, (self.height * 3) // 4)
        self.bloom_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_fbo)

        self.bloom_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bloom_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.bloom_width, self.bloom_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.bloom_texture, 0)

        bloom_status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if bloom_status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Bloom extract framebuffer incomplete: {bloom_status}")
            if self.bloom_texture:
                glDeleteTextures(1, [self.bloom_texture])
                self.bloom_texture = 0
            if self.bloom_fbo:
                glDeleteFramebuffers(1, [self.bloom_fbo])
                self.bloom_fbo = 0

        self.bloom_blur_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_blur_fbo)

        self.bloom_blur_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bloom_blur_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.bloom_width, self.bloom_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.bloom_blur_texture, 0)

        blur_status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if blur_status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Bloom blur framebuffer incomplete: {blur_status}")
            if self.bloom_blur_texture:
                glDeleteTextures(1, [self.bloom_blur_texture])
                self.bloom_blur_texture = 0
            if self.bloom_blur_fbo:
                glDeleteFramebuffers(1, [self.bloom_blur_fbo])
                self.bloom_blur_fbo = 0

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

    def _draw_scene_bloom_composite(self):
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_BLEND)
        glUseProgram(self.screen_bloom_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_color_texture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.bloom_texture)
        if self.u_screen_bloom_strength_loc != -1:
            current_time = self.last_frame_time
            if self.get_current_time and not self.export_mode:
                try:
                    current_time = self.get_current_time()
                except Exception:
                    current_time = self.last_frame_time
            bloom_strength = self.bloom_strength + self._compute_spike_bloom_boost(current_time)
            glUniform1f(self.u_screen_bloom_strength_loc, bloom_strength)
        glBindVertexArray(self.screen_bloom_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glEnable(GL_BLEND)
        glDepthMask(GL_TRUE)

    def _render_bloom_extract(self):
        if not self.bloom_extract_shader or not self.bloom_fbo:
            return
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_fbo)
        glViewport(0, 0, self.bloom_width, self.bloom_height)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_BLEND)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.bloom_extract_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_color_texture)
        glBindVertexArray(self.screen_bloom_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glEnable(GL_BLEND)
        glDepthMask(GL_TRUE)

    def _render_bloom_blur(self):
        if (
            not self.bloom_blur_shader
            or not self.bloom_fbo
            or not self.bloom_blur_fbo
            or not self.bloom_texture
            or not self.bloom_blur_texture
        ):
            return

        glViewport(0, 0, self.bloom_width, self.bloom_height)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_BLEND)
        glUseProgram(self.bloom_blur_shader)
        glBindVertexArray(self.screen_bloom_vao)

        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_blur_fbo)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bloom_texture)
        if self.u_bloom_blur_direction_loc != -1:
            glUniform2f(self.u_bloom_blur_direction_loc, 1.0, 0.0)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_fbo)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, self.bloom_blur_texture)
        if self.u_bloom_blur_direction_loc != -1:
            glUniform2f(self.u_bloom_blur_direction_loc, 0.0, 1.0)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindVertexArray(0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glEnable(GL_BLEND)
        glDepthMask(GL_TRUE)

    def _draw_note_bloom(self, current_time, window_start, window_end):
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glUseProgram(self.bloom_shader)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        glUniform1f(self.u_bloom_time_loc, current_time)
        glUniform1f(self.u_bloom_scroll_speed_loc, self.scroll_speed)
        glUniform1f(glGetUniformLocation(self.bloom_shader, "u_guide_line_y"), self._get_guide_line_y())
        glUniform1f(self.u_bloom_window_start_loc, window_start)
        glUniform1f(self.u_bloom_window_end_loc, window_end)
        if self.u_bloom_radius_loc != -1:
            glUniform1f(self.u_bloom_radius_loc, self.bloom_radius)
        if self.u_bloom_strength_loc != -1:
            glUniform1f(self.u_bloom_strength_loc, self.bloom_strength)

        glBindVertexArray(self.bloom_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self.notes_to_draw)
        glBindVertexArray(0)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(0)
        glDepthMask(GL_TRUE)

    def _update_bloom_compensation(self):
        now = time.perf_counter()
        dt = max(0.0, min(0.25, now - self.last_bloom_update_time))
        self.last_bloom_update_time = now

        if not self.show_bloom:
            self.bloom_strength = 0.0
            return

        visible_count = max(0.0, float(self.notes_to_draw))
        normalized_load = visible_count / max(1.0, self.bloom_response_reference)
        response = np.exp(-self.bloom_response_scale * pow(normalized_load, self.bloom_response_curve))
        target_strength = self.bloom_base_strength * (
            self.bloom_min_strength + (0.96 - self.bloom_min_strength) * response
        )

        if visible_count > self.bloom_emergency_threshold:
            emergency_t = (visible_count - self.bloom_emergency_threshold) / max(
                1.0, self.bloom_emergency_full_threshold - self.bloom_emergency_threshold
            )
            emergency_t = np.clip(emergency_t, 0.0, 1.0)
            emergency_t = 1.0 - np.exp(-4.8 * emergency_t * emergency_t)
            emergency_target = self.bloom_base_strength * self.bloom_emergency_strength
            target_strength = target_strength + (emergency_target - target_strength) * emergency_t

        smooth = 1.0 - np.exp(-dt * 5.4)
        self.bloom_strength += (target_strength - self.bloom_strength) * smooth

    def _compute_spike_bloom_boost(self, current_time):
        if not self.show_bloom or not self.show_spike_bloom or not self.nps_spikes:
            return 0.0

        strongest = 0.0
        for spike_time, spike_value in self.nps_spikes:
            spike_time = float(spike_time)
            amplitude = self._spike_rank_map.get(
                (float(spike_time), int(spike_value)),
                self.spike_bloom_min_boost,
            )

            if current_time < spike_time - self.spike_bloom_rise_duration:
                continue
            if current_time <= spike_time:
                rise_t = (current_time - (spike_time - self.spike_bloom_rise_duration)) / max(0.001, self.spike_bloom_rise_duration)
                weight = self._ease_in_spike(rise_t)
            elif current_time <= spike_time + self.spike_bloom_fall_duration:
                fall_t = (current_time - spike_time) / max(0.001, self.spike_bloom_fall_duration)
                weight = self._ease_out_spike(fall_t)
            else:
                continue

            strongest = max(strongest, amplitude * weight)

        return strongest

    def _ease_in_spike(self, t):
        t = float(np.clip(t, 0.0, 1.0))
        return pow(t, 2.35)

    def _ease_out_spike(self, t):
        t = float(np.clip(t, 0.0, 1.0))
        return pow(1.0 - t, 1.85)
