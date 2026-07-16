import math
import time
import random
import numpy as np
import pygame
from OpenGL.GL import *


class OverlayMixin:
    """UI overlay drawing methods for PianoRoll: stats, sliders, buttons, tooltips, hover."""

    def _iter_hover_targets(self):
        if self.hide_buttons:
            return
        if (not self.controls_panel_expanded) and self.controls_toggle_rect:
            yield ("controls_toggle", self.controls_toggle_rect, "Open control panel")
        if self.controls_panel_expanded and self.controls_close_rect:
            yield ("controls_close", self.controls_close_rect, "Close control panel")
        if self.color_mode_button_rect:
            next_mode = "channel" if self.note_color_mode == "track" else "track"
            yield ("color_mode_button", self.color_mode_button_rect, f"Switch to {next_mode} colors")
        if self.glow_button_rect:
            yield ("glow_button", self.glow_button_rect, "Toggle lighting mode")
        if self.glow_options_button_rect:
            yield ("glow_options", self.glow_options_button_rect, "Lighting options")
        if self.color_button_rect:
            yield ("color_button", self.color_button_rect, "Randomize note colors")
        if self.fun_button_rect:
            yield ("fun_button", self.fun_button_rect, "Fun")
        if self.fun_options_expanded and self.overclock_checkbox_rect:
            yield ("overclock_checkbox", self.overclock_checkbox_rect, "GPU artifact corruption effect")
        if self.glow_options_expanded and self.glow_options_checkbox_rect:
            yield ("glow_checkbox", self.glow_options_checkbox_rect, "Toggle glow on key press")
        if self.glow_options_expanded and self.key_light_fade_checkbox_rect:
            yield ("key_fade_checkbox", self.key_light_fade_checkbox_rect, "Toggle light fadingg")
        if self.glow_options_expanded and self.bloom_checkbox_rect:
            yield ("bloom_checkbox", self.bloom_checkbox_rect, "Toggle scene bloom")
        if self.glow_options_expanded and self.spike_bloom_checkbox_rect:
            yield ("spike_bloom_checkbox", self.spike_bloom_checkbox_rect, "Dynamically adjust bloom according to NPS")
        if self.renderer_mode_button_rect:
            yield ("renderer_mode_button", self.renderer_mode_button_rect, "Switch renderer mode")

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

    def _draw_text_overlay(self, text, x, y, color=(225, 228, 235), alpha=1.0):
        tex_info = self._get_text_texture(text, color)
        if tex_info is None:
            return
        pixel_data, width, height = tex_info
        glPushAttrib(GL_ENABLE_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, max(0.0, min(1.0, float(alpha))))
        glWindowPos2i(int(x), int(self.height - y - height))
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data)
        glPopAttrib()

    def _format_time_overlay(self, seconds):
        total_seconds = max(0, int(seconds))
        minutes, sec = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _init_slider_geometry(self):
        padding = 16
        panel_width, panel_height = self.controls_panel_size
        panel_x = padding - 4
        panel_y = padding - 2
        stats_rect = self._get_stats_overlay_box_rect(self.last_frame_time)
        if stats_rect is not None:
            panel_y = max(panel_y, stats_rect.bottom + 10)
        self.controls_toggle_rect = pygame.Rect(panel_x, panel_y, 34, 34)
        self.controls_panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        self.controls_close_rect = pygame.Rect(panel_x + panel_width - 34 - 8, panel_y + 8, 34, 34)

        bar_width = panel_width - 32
        bar_height = 6
        x = panel_x + 16
        y = panel_y + 42
        scroll_y = y
        self.scroll_slider_rect = pygame.Rect(x, scroll_y, bar_width, self.slider_area_height)
        self.scroll_slider_bar = (x, scroll_y + (self.slider_area_height - bar_height) // 2, bar_width, bar_height)
        self._recalc_scroll_slider_handle()
        fps_y = scroll_y + self.slider_area_height + 18
        self.fps_slider_rect = pygame.Rect(x, fps_y, bar_width, self.slider_area_height)
        self.fps_slider_bar = (x, fps_y + (self.slider_area_height - bar_height) // 2, bar_width, bar_height)
        self._recalc_fps_slider_handle()

    def _recalc_scroll_slider_handle(self):
        if not self.scroll_slider_rect: return
        t = (self.scroll_speed - self.scroll_slider_min) / (self.scroll_slider_max - self.scroll_slider_min)
        t = max(0.0, min(1.0, t))
        self.scroll_slider_handle_x = self.scroll_slider_rect.x + t * self.scroll_slider_rect.width

    def _recalc_fps_slider_handle(self):
        if not self.fps_slider_rect:
            return
        t = (float(self.fps_cap) - self.fps_slider_min) / (self.fps_slider_max - self.fps_slider_min)
        t = max(0.0, min(1.0, t))
        self.fps_slider_handle_x = self.fps_slider_rect.x + t * self.fps_slider_rect.width

    def handle_slider_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.hide_buttons:
                if (not self.controls_panel_expanded) and self.controls_toggle_rect and self.controls_toggle_rect.collidepoint(event.pos):
                    self.controls_panel_expanded = not self.controls_panel_expanded
                    self.scroll_slider_dragging = False
                    self.fps_slider_dragging = False
                    return
                if self.controls_panel_expanded and self.controls_close_rect and self.controls_close_rect.collidepoint(event.pos):
                    self.controls_panel_expanded = False
                    self.scroll_slider_dragging = False
                    self.fps_slider_dragging = False
                    return
                if self.glow_options_button_rect and self.glow_options_button_rect.collidepoint(event.pos):
                    self.glow_options_expanded = not self.glow_options_expanded
                    return
                if self.glow_options_expanded and self.glow_options_checkbox_rect and self.glow_options_checkbox_rect.collidepoint(event.pos):
                    self.show_key_press_glow = not self.show_key_press_glow
                    if not self.show_key_press_glow and not self.show_key_light_fade:
                        self.glow_trails.clear()
                        self._split_fade_trails.clear()
                    self._save_visualizer_config()
                    return
                if self.glow_options_expanded and self.key_light_fade_checkbox_rect and self.key_light_fade_checkbox_rect.collidepoint(event.pos):
                    self.show_key_light_fade = not self.show_key_light_fade
                    if not self.show_key_press_glow and not self.show_key_light_fade:
                        self.glow_trails.clear()
                        self._split_fade_trails.clear()
                    self._save_visualizer_config()
                    return
                if self.glow_options_expanded and self.bloom_checkbox_rect and self.bloom_checkbox_rect.collidepoint(event.pos):
                    self.show_bloom = not self.show_bloom
                    self.bloom_strength = self.bloom_base_strength if self.show_bloom else 0.0
                    self.last_bloom_update_time = time.perf_counter()
                    if self.bloom_shader and self.u_bloom_strength_loc != -1:
                        glUseProgram(self.bloom_shader)
                        glUniform1f(self.u_bloom_strength_loc, self.bloom_strength)
                        glUseProgram(0)
                    self._save_visualizer_config()
                    return
                if self.glow_options_expanded and self.spike_bloom_checkbox_rect and self.spike_bloom_checkbox_rect.collidepoint(event.pos):
                    self.show_spike_bloom = not self.show_spike_bloom
                    self._save_visualizer_config()
                    return
                if self.color_mode_button_rect and self.color_mode_button_rect.collidepoint(event.pos):
                    self.toggle_note_color_mode()
                    return
                if self.glow_button_rect and self.glow_button_rect.collidepoint(event.pos):
                    self.toggle_glow()
                    return
                if self.color_button_rect and self.color_button_rect.collidepoint(event.pos):
                    self.randomize_colors()
                    return
                if self.fun_button_rect and self.fun_button_rect.collidepoint(event.pos):
                    self.fun_options_expanded = not self.fun_options_expanded
                    return
                if self.fun_options_expanded and self.overclock_checkbox_rect and self.overclock_checkbox_rect.collidepoint(event.pos):
                    self.overclock_mode = not self.overclock_mode
                    if not self.overclock_mode:
                        self.overclock_intensity = 0.0
                    self._save_visualizer_config()
                    return
                if self.renderer_mode_button_rect and self.renderer_mode_button_rect.collidepoint(event.pos):
                    idx = self.renderer_modes.index(self.renderer_mode) if self.renderer_mode in self.renderer_modes else 0
                    self.renderer_mode = self.renderer_modes[(idx + 1) % len(self.renderer_modes)]
                    self._save_visualizer_config()
                    return
            if self.controls_panel_expanded and self.scroll_slider_rect and self.scroll_slider_rect.collidepoint(event.pos):
                self.scroll_slider_dragging = True
                self._update_scroll_slider_from_pos(event.pos[0])
            if self.controls_panel_expanded and self.fps_slider_rect and self.fps_slider_rect.collidepoint(event.pos):
                self.fps_slider_dragging = True
                self._update_fps_slider_from_pos(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.scroll_slider_dragging = False
            self.fps_slider_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.scroll_slider_dragging:
            self._update_scroll_slider_from_pos(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and self.fps_slider_dragging:
            self._update_fps_slider_from_pos(event.pos[0])

    def randomize_colors(self):
        """Shuffle the track colors randomly and update the shader."""
        indices = list(range(128))
        random.shuffle(indices)
        self.channel_colors = self.channel_colors[indices]
        glUseProgram(self.shader)
        glUniform3fv(self.u_colors_loc, 128, self.channel_colors)
        glUseProgram(0)
        if self.bloom_shader and self.u_bloom_colors_loc != -1:
            glUseProgram(self.bloom_shader)
            glUniform3fv(self.u_bloom_colors_loc, 128, self.channel_colors)
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

    def _update_scroll_slider_from_pos(self, x_pos):
        t = (x_pos - self.scroll_slider_rect.x) / float(self.scroll_slider_rect.width)
        t = max(0.0, min(1.0, t))
        new_val = self.scroll_slider_min + t * (self.scroll_slider_max - self.scroll_slider_min)
        self.scroll_speed = new_val
        self._recalc_scroll_slider_handle()
        self._save_visualizer_config()

    def _update_fps_slider_from_pos(self, x_pos):
        t = (x_pos - self.fps_slider_rect.x) / float(self.fps_slider_rect.width)
        t = max(0.0, min(1.0, t))
        new_val = self.fps_slider_min + t * (self.fps_slider_max - self.fps_slider_min)
        self.fps_cap = max(1, int(round(new_val)))
        self._recalc_fps_slider_handle()
        self._save_visualizer_config()

    def _get_stats_overlay_box_rect(self, current_time):
        if self.export_mode or not self.live_show_stats_overlay:
            return None
        if self.render_on_times is None or self.overlay_font is None:
            return None

        total_notes = int(self.stats_total_notes or len(self.render_on_times))
        total_duration = float(self.stats_total_duration or self.export_total_duration)
        on_screen_count = len(self.last_visible_notes) if self.last_visible_notes is not None else 0
        lines = [
            f"Notes: {self.live_note_count_passed:,} / {total_notes:,}",
            f"On screen: {on_screen_count:,}",
            f"NPS: {self.live_nps_value:,}",
            f"BPM: {self.live_bpm_value:.2f}".rstrip('0').rstrip('.'),
            f"Polyphony: {self.live_polyphony_value:,}",
            f"Time: {self._format_time_overlay(current_time)} / {self._format_time_overlay(total_duration)}",
        ]
        tex_infos = [self._get_text_texture(line, (235, 238, 244)) for line in lines]
        if any(info is None for info in tex_infos):
            return None

        line_height = max(info[2] for info in tex_infos)
        content_width = max(info[1] for info in tex_infos)
        padding_x = 12
        padding_y = 10
        spacing = 4
        box_x = 14
        box_y = 14
        box_w = content_width + (padding_x * 2)
        box_h = (line_height * len(lines)) + (spacing * (len(lines) - 1)) + (padding_y * 2)
        return pygame.Rect(int(box_x), int(box_y), int(box_w), int(box_h))

    def _draw_stats_overlay(self, current_time):
        if self.render_on_times is None:
            return
        if self.export_mode:
            if not self.export_show_stats_overlay:
                return
            stats_on_times = self.stats_on_times if self.stats_on_times is not None else self.render_on_times
            note_count_passed = int(np.searchsorted(stats_on_times, current_time, side='right'))
            left_idx = int(np.searchsorted(stats_on_times, max(0.0, current_time - 1.0), side='left'))
            nps_value = max(0, note_count_passed - left_idx)
            if self.stats_off_times_sorted is not None and len(self.stats_off_times_sorted) > 0:
                ended_idx = int(np.searchsorted(self.stats_off_times_sorted, current_time, side='right'))
                polyphony_value = max(0, note_count_passed - ended_idx)
            else:
                polyphony_value = 0
            if self.stats_tempo_times is not None and len(self.stats_tempo_times) > 0:
                tempo_idx = int(np.searchsorted(self.stats_tempo_times, current_time, side='right')) - 1
                if tempo_idx < 0:
                    tempo_idx = 0
                bpm_value = float(self.stats_tempo_bpms[tempo_idx])
            else:
                bpm_value = 120.0
        elif not self.live_show_stats_overlay:
            return
        else:
            note_count_passed = self.live_note_count_passed
            nps_value = self.live_nps_value
            bpm_value = self.live_bpm_value
            polyphony_value = self.live_polyphony_value
        total_notes = int(self.stats_total_notes or len(self.render_on_times))
        total_duration = float(self.stats_total_duration or self.export_total_duration)
        stats_multiplier = self.stats_multiplier if self.export_mode and self.stats_modification_enabled else 1.0
        spike_nps_multiplier = self._compute_export_spike_nps_multiplier(current_time)
        spike_note_delta = self._compute_export_spike_note_delta(current_time)
        spike_total_note_delta = self._get_export_spike_total_note_delta()
        scaled_note_count = int(round(note_count_passed * stats_multiplier)) + spike_note_delta
        scaled_total_notes = int(round(total_notes * stats_multiplier)) + spike_total_note_delta
        scaled_nps_value = int(round(nps_value * stats_multiplier * spike_nps_multiplier))
        scaled_polyphony_value = int(round(polyphony_value * stats_multiplier))
        on_screen_count = len(self.last_visible_notes) if self.last_visible_notes is not None else 0

        lines = [
            f"Notes: {scaled_note_count:,} / {scaled_total_notes:,}",
            f"On screen: {on_screen_count:,}",
            f"NPS: {scaled_nps_value:,}",
            f"BPM: {bpm_value:.2f}".rstrip('0').rstrip('.'),
            f"Polyphony: {scaled_polyphony_value:,}",
            f"Time: {self._format_time_overlay(current_time)} / {self._format_time_overlay(total_duration)}",
        ]

        tex_infos = [self._get_text_texture(line, (235, 238, 244)) for line in lines]
        if any(info is None for info in tex_infos):
            return

        line_height = max(info[2] for info in tex_infos)
        content_width = max(info[1] for info in tex_infos)
        padding_x = 12
        padding_y = 10
        spacing = 4
        box_x = 14
        box_y = 14
        box_w = content_width + (padding_x * 2)
        box_h = (line_height * len(lines)) + (spacing * (len(lines) - 1)) + (padding_y * 2)

        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self.stats_modification_enabled:
            glColor4f(0.12, 0.03, 0.05, 0.62)
        else:
            glColor4f(0.03, 0.03, 0.05, 0.62)
        glBegin(GL_QUADS)
        glVertex2f(box_x, box_y); glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h); glVertex2f(box_x, box_y + box_h)
        glEnd()
        glColor4f(0.72, 0.74, 0.80, 0.55)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(box_x, box_y); glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h); glVertex2f(box_x, box_y + box_h)
        glEnd()

        text_y = box_y + padding_y
        for line in lines:
            self._draw_text_overlay(line, box_x + padding_x, text_y, color=(235, 238, 244), alpha=0.94)
            text_y += line_height + spacing

        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

    def _draw_export_watermark(self, current_time):
        if not self.export_mode or self.export_total_duration <= 0.0:
            return

        total_lifetime = 5.0
        start_time = max(0.0, self.export_total_duration - total_lifetime)
        if current_time < start_time:
            return

        local_t = min(total_lifetime, max(0.0, current_time - start_time))
        fade_in = 1.0
        hold = 3.0
        fade_out = 1.0

        if local_t < fade_in:
            alpha_factor = local_t / fade_in
        elif local_t < fade_in + hold:
            alpha_factor = 1.0
        else:
            fade_t = (local_t - fade_in - hold) / fade_out
            alpha_factor = 1.0 - fade_t

        alpha_factor = max(0.0, min(1.0, alpha_factor))
        if alpha_factor <= 0.0:
            return

        text = "Rendered with LWMP"
        tex_info = self._get_text_texture(text, (235, 238, 244))
        if tex_info is None:
            return
        _, width, height = tex_info
        margin_x = 18
        margin_y = 16
        self._draw_text_overlay(
            text,
            self.width - width - margin_x,
            self.height - height - margin_y,
            color=(235, 238, 244),
            alpha=0.25 * alpha_factor,
        )

    def _draw_capacity_warning_overlay(self):
        if self.export_mode or not self.capacity_warning_active:
            return

        message = "VBO capacity drained, reducing visible notes..."
        detail = f"Visible notes: {self.capacity_warning_visible_count:,}   Cap: {self.streaming_vbo_capacity:,}"
        msg_info = self._get_text_texture(message, (242, 244, 248))
        detail_info = self._get_text_texture(detail, (198, 204, 214))
        if msg_info is None or detail_info is None:
            return

        msg_w, msg_h = msg_info[1], msg_info[2]
        detail_w, detail_h = detail_info[1], detail_info[2]
        spacing = 6
        box_w = max(msg_w, detail_w) + 40
        box_h = msg_h + detail_h + spacing + 28
        box_x = (self.width - box_w) * 0.5
        box_y = (self.height - box_h) * 0.46

        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(0.04, 0.045, 0.06, 0.80)
        glBegin(GL_QUADS)
        glVertex2f(box_x, box_y); glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h); glVertex2f(box_x, box_y + box_h)
        glEnd()

        glColor4f(0.85, 0.62, 0.24, 0.60)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(box_x, box_y); glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h); glVertex2f(box_x, box_y + box_h)
        glEnd()

        msg_x = box_x + (box_w - msg_w) * 0.5
        msg_y = box_y + 10
        detail_x = box_x + (box_w - detail_w) * 0.5
        detail_y = msg_y + msg_h + spacing
        self._draw_text_overlay(message, msg_x, msg_y, color=(242, 244, 248), alpha=0.98)
        self._draw_text_overlay(detail, detail_x, detail_y, color=(198, 204, 214), alpha=0.90)

        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

    def _draw_slider_overlay(self):
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        if self.renderer_mode_button_rect and not self.hide_buttons:
            bx, by = self.renderer_mode_button_rect.x, self.renderer_mode_button_rect.y
            bw, bh = self.renderer_mode_button_rect.width, self.renderer_mode_button_rect.height
            rm_hov = self.renderer_mode_button_rect.collidepoint(pygame.mouse.get_pos())
            rm_bg = (0.26, 0.30, 0.40) if rm_hov else (0.16, 0.18, 0.24)
            glColor4f(*rm_bg, 0.92)
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bw, by)
            glVertex2f(bx + bw, by + bh); glVertex2f(bx, by + bh)
            glEnd()
            rm_label = "Channel Split" if self.renderer_mode == 'channel_split' else "Default"
            rm_tc = (110, 210, 255) if self.renderer_mode == 'channel_split' else (160, 170, 190)
            self._draw_text_overlay(rm_label, bx + 10, by + 4, color=rm_tc, alpha=0.95)
            glColor4f(0.35, 0.42, 0.58, 0.55)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, by); glVertex2f(bx + bw, by)
            glVertex2f(bx + bw, by + bh); glVertex2f(bx, by + bh)
            glEnd()
        self._update_hover_ui_state()
        handle_half = 6
        if self.controls_panel_expanded and self.controls_panel_rect and not self.hide_buttons:
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

        if (not self.controls_panel_expanded) and self.controls_toggle_rect and not self.hide_buttons:
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

        if self.controls_panel_expanded and self.controls_panel_rect and not self.hide_buttons:
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

            self._draw_text_overlay("Scroll Speed", self.scroll_slider_rect.x, self.scroll_slider_rect.y - 12)
            self._draw_text_overlay("FPS Cap", self.fps_slider_rect.x, self.fps_slider_rect.y - 12)

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

            if self.fps_slider_bar:
                x3, y3, w3, h3 = self.fps_slider_bar
                glColor3f(0.22, 0.22, 0.28)
                glBegin(GL_QUADS); glVertex2f(x3, y3); glVertex2f(x3 + w3, y3); glVertex2f(x3 + w3, y3 + h3); glVertex2f(x3, y3 + h3); glEnd()
                filled_w3 = (self.fps_slider_handle_x - x3)
                glColor3f(0.86, 0.72, 0.34)
                glBegin(GL_QUADS); glVertex2f(x3, y3); glVertex2f(x3 + filled_w3, y3); glVertex2f(x3 + filled_w3, y3 + h3); glVertex2f(x3, y3 + h3); glEnd()
                hx3 = self.fps_slider_handle_x
                hy3 = y3 + h3 / 2
                glColor3f(0.95, 0.95, 0.95)
                glBegin(GL_QUADS); glVertex2f(hx3 - handle_half, hy3 - handle_half); glVertex2f(hx3 + handle_half, hy3 - handle_half); glVertex2f(hx3 + handle_half, hy3 + handle_half); glVertex2f(hx3 - handle_half, hy3 + handle_half); glEnd()

        if self.color_button_rect and not self.hide_buttons:
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

        if self.fun_button_rect and not self.hide_buttons:
            bx, by = self.fun_button_rect.x, self.fun_button_rect.y
            bs = self.color_button_size
            glColor4f(0.15, 0.15, 0.2, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            cx, cy = bx + bs * 0.5, by + bs * 0.5
            r = bs * 0.32
            glColor4f(0.9, 0.35, 0.45, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(cx - r, cy - r * 0.3)
            glVertex2f(cx + r, cy - r * 0.3)
            glVertex2f(cx + r, cy + r * 0.3)
            glVertex2f(cx - r, cy + r * 0.3)
            glEnd()
            ribbon_w = r * 0.2
            glColor4f(1.0, 0.85, 0.2, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(cx - ribbon_w, by + bs * 0.2)
            glVertex2f(cx + ribbon_w, by + bs * 0.2)
            glVertex2f(cx + ribbon_w, by + bs * 0.8)
            glVertex2f(cx - ribbon_w, by + bs * 0.8)
            glEnd()
            glBegin(GL_QUADS)
            glVertex2f(bx + bs * 0.2, cy - ribbon_w)
            glVertex2f(bx + bs * 0.8, cy - ribbon_w)
            glVertex2f(bx + bs * 0.8, cy + ribbon_w)
            glVertex2f(bx + bs * 0.2, cy + ribbon_w)
            glEnd()
            bow_r = r * 0.28
            glColor4f(1.0, 0.85, 0.2, 0.9)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(17):
                angle = (i / 16.0) * 6.2831853
                glVertex2f(cx + math.cos(angle) * bow_r, cy + math.sin(angle) * bow_r)
            glEnd()
            glColor4f(0.5, 0.5, 0.55, 0.8)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            self._draw_hover_highlight(self.fun_button_rect, self._get_hover_alpha("fun_button"))

        if self.color_mode_button_rect and not self.hide_buttons:
            bx, by = self.color_mode_button_rect.x, self.color_mode_button_rect.y
            bs = self.color_button_size
            if self.note_color_mode == "channel":
                glColor4f(0.20, 0.42, 0.78, 0.82)
                label = "C"
            else:
                glColor4f(0.20, 0.62, 0.34, 0.82)
                label = "T"
            glBegin(GL_QUADS)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            glColor4f(0.5, 0.5, 0.55, 0.8)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, by); glVertex2f(bx + bs, by)
            glVertex2f(bx + bs, by + bs); glVertex2f(bx, by + bs)
            glEnd()
            self._draw_text_overlay(label, bx + 11, by + 6, color=(240, 243, 248), alpha=0.98)
            self._draw_hover_highlight(self.color_mode_button_rect, self._get_hover_alpha("color_mode_button"))

        if self.glow_button_rect and not self.hide_buttons:
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

        if self.glow_options_button_rect and not self.hide_buttons:
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

        if self.glow_options_expanded and self.glow_options_panel_rect and not self.hide_buttons:
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
            self._draw_text_overlay("Bloom scene", px + 32, py + 78)
            self._draw_text_overlay("Spike bloom", px + 32, py + 102)

            checkbox_rects = [
                (self.glow_options_checkbox_rect, self.show_key_press_glow, "glow_checkbox"),
                (self.key_light_fade_checkbox_rect, self.show_key_light_fade, "key_fade_checkbox"),
                (self.bloom_checkbox_rect, self.show_bloom, "bloom_checkbox"),
                (self.spike_bloom_checkbox_rect, self.show_spike_bloom, "spike_bloom_checkbox"),
            ]
            for cb_rect, is_checked, hover_id in checkbox_rects:
                if not cb_rect:
                    continue
                cbx, cby, cbw, cbh = cb_rect.x, cb_rect.y, cb_rect.width, cb_rect.height
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
                if is_checked:
                    glColor4f(0.95, 0.78, 0.24, 0.95)
                    glLineWidth(2.0)
                    glBegin(GL_LINES)
                    glVertex2f(cbx + 3, cby + 9); glVertex2f(cbx + 7, cby + 13)
                    glVertex2f(cbx + 7, cby + 13); glVertex2f(cbx + 13, cby + 4)
                    glEnd()
                self._draw_hover_highlight(cb_rect, self._get_hover_alpha(hover_id))

        if self.fun_options_expanded and self.fun_options_panel_rect and not self.hide_buttons:
            px, py, pw, ph = self.fun_options_panel_rect
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

            self._draw_text_overlay("Overclock Mode", px + 32, py + 30)

            cb_rect = self.overclock_checkbox_rect
            if cb_rect:
                cbx, cby, cbw, cbh = cb_rect.x, cb_rect.y, cb_rect.width, cb_rect.height
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
                if self.overclock_mode:
                    glColor4f(0.95, 0.78, 0.24, 0.95)
                    glLineWidth(2.0)
                    glBegin(GL_LINES)
                    glVertex2f(cbx + 3, cby + 9); glVertex2f(cbx + 7, cby + 13)
                    glVertex2f(cbx + 7, cby + 13); glVertex2f(cbx + 13, cby + 4)
                    glEnd()
                self._draw_hover_highlight(cb_rect, self._get_hover_alpha("overclock_checkbox"))

        self._draw_hover_tooltip()
        
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

    def set_live_stats(self, note_count_passed, nps_value, bpm_value=0.0, polyphony_value=0):
        self.live_note_count_passed = max(0, int(note_count_passed))
        self.live_nps_value = max(0, int(nps_value))
        self.live_bpm_value = max(0.0, float(bpm_value))
        self.live_polyphony_value = max(0, int(polyphony_value))

    def set_stats_context(
        self,
        on_times,
        off_times_sorted,
        tempo_events,
        total_duration,
        total_notes,
        stats_multiplier=1.0,
        stats_modification_enabled=False,
        selected_spike=None,
        spike_intensity=1.0,
    ):
        self.stats_on_times = on_times
        self.stats_off_times_sorted = off_times_sorted
        tempo_events = tempo_events or [(0.0, 120.0)]
        self.stats_tempo_events = tempo_events
        self.stats_tempo_times = np.array([t for t, _ in tempo_events], dtype=np.float64)
        self.stats_tempo_bpms = np.array([bpm for _, bpm in tempo_events], dtype=np.float32)
        self.stats_total_duration = max(0.0, float(total_duration))
        self.stats_total_notes = max(0, int(total_notes))
        self.stats_multiplier = max(0.1, float(stats_multiplier))
        self.stats_modification_enabled = bool(stats_modification_enabled)
        self.stats_selected_spike = selected_spike if selected_spike else None
        self.stats_spike_intensity = max(0.0, float(spike_intensity))

    def _compute_export_spike_nps_multiplier(self, current_time):
        if not self.export_mode or not self.stats_modification_enabled or not self.stats_selected_spike:
            return 1.0
        if self.stats_spike_intensity == 1.0:
            return 1.0
        spike_time = float(self.stats_selected_spike[0])
        spike_window = 1.0
        distance = abs(float(current_time) - spike_time)
        if distance >= spike_window:
            return 1.0
        t = 1.0 - (distance / spike_window)
        eased = t * t * (3.0 - 2.0 * t)
        return 1.0 + ((self.stats_spike_intensity - 1.0) * eased)

    def _compute_export_spike_note_delta(self, current_time):
        if not self.export_mode or not self.stats_modification_enabled or not self.stats_selected_spike:
            return 0
        spike_intensity = self.stats_spike_intensity
        if spike_intensity == 1.0:
            return 0
        spike_time = float(self.stats_selected_spike[0])
        spike_value = max(0.0, float(self.stats_selected_spike[1]))
        spike_window = 1.0
        extra_notes_total = spike_value * (spike_intensity - 1.0)
        if extra_notes_total == 0.0:
            return 0
        window_start = spike_time - spike_window
        window_end = spike_time + spike_window
        current_time = float(current_time)
        if current_time <= window_start:
            return 0
        if current_time >= window_end:
            return int(round(extra_notes_total))
        progress = (current_time - window_start) / max(0.001, (window_end - window_start))
        eased_progress = progress * progress * (3.0 - 2.0 * progress)
        return int(round(extra_notes_total * eased_progress))

    def _get_export_spike_total_note_delta(self):
        if not self.export_mode or not self.stats_modification_enabled or not self.stats_selected_spike:
            return 0
        spike_intensity = self.stats_spike_intensity
        if spike_intensity == 1.0:
            return 0
        spike_value = max(0.0, float(self.stats_selected_spike[1]))
        return int(round(spike_value * (spike_intensity - 1.0)))
