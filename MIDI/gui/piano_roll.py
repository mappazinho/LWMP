"""Auto-extracted mixin for DpgMidiPlayerApp."""
import os
import threading
import time
import traceback
import math
import subprocess
import tempfile
import wave
from collections import deque

import numpy as np
import dearpygui.dearpygui as dpg

class PianoRollMixin:
    """Methods for piano_roll."""

    def on_pianoroll_stats_overlay_toggle(self, sender, app_data):
        enabled = bool(app_data)
        self._gui_cfg()["show_pianoroll_stats_overlay"] = enabled
        self._save_config(self._CONFIG)
        self._apply_performance_overlay_layout()
        if self.piano_roll is not None:
            self.piano_roll.live_show_stats_overlay = enabled

    def _on_hide_buttons_toggle(self, sender, app_data):
        enabled = bool(app_data)
        self._gui_cfg()["hide_buttons"] = enabled
        self._save_config(self._CONFIG)
        if self.piano_roll is not None:
            self.piano_roll.hide_buttons = enabled
            self.piano_roll._save_visualizer_config()


    def show_piano_roll_dialog(self):
        if self.piano_roll and self.piano_roll.app_running.is_set():
            self._message_info("Piano Roll", "Piano Roll is already running.")
            return
        self._center_modal("piano_roll_window", 320, 220)
        dpg.configure_item("piano_roll_window", show=True)


    def launch_selected_piano_roll(self):
        selected = dpg.get_value("piano_roll_resolution")
        width_str, height_str = selected.split(" x ")
        dpg.configure_item("piano_roll_window", show=False)
        self.launch_piano_roll(int(width_str), int(height_str))


    def launch_piano_roll(self, width, height):
        if not self._PianoRoll:
            return
        self.last_piano_roll_res = (width, height)
        self.piano_roll = self._PianoRoll(width, height, self._CONFIG)
        self.piano_roll.live_show_stats_overlay = bool(self._gui_cfg().get("show_pianoroll_stats_overlay", False))
        self.piano_roll.hide_buttons = bool(self._gui_cfg().get("hide_buttons", False))
        self.piano_roll.set_live_stats(0, 0, 0.0, 0)
        if self.controller.parsed_midi is not None:
            self.piano_roll.set_stats_context(
                self.controller.parsed_midi.note_events_for_playback["on_time"],
                getattr(self.controller.parsed_midi, "sorted_off_times", None),
                getattr(self.controller.parsed_midi, "tempo_events", None),
                float(self.controller.total_song_duration),
                int(self.controller.total_song_notes),
            )
        self.piano_roll.set_nps_spikes(getattr(self.controller, "max_nps_spikes", []))
        self.piano_roll.set_preferred_color_mode(getattr(self.controller.parsed_midi, "preferred_color_mode", "track"))
        self.piano_roll_thread = threading.Thread(target=self.run_piano_roll, daemon=True)
        self.piano_roll_thread.start()
        dpg.disable_item("skin_button")

        if self.controller.parsed_midi:
            notes_for_gpu = self.controller.parsed_midi.note_data_for_gpu.copy()
            t = threading.Timer(
                0.5,
                lambda pr=self.piano_roll, f=self.get_current_playback_time_thread_safe: (
                    pr.load_midi(notes_for_gpu, f) if pr is not None else None
                ),
            )
            t.daemon = True
            t.start()
            self._load_midi_timer = t


    def run_piano_roll(self):
        piano_roll_instance = self.piano_roll
        try:
            piano_roll_instance.init_pygame_and_gl()
            clock = self._pygame.time.Clock()
            self._pygame.time.set_timer(self._pygame.USEREVENT, 250)
            last_caption_update_time = 0

            while piano_roll_instance.app_running.is_set():
                for event in self._pygame.event.get():
                    if event.type == self._pygame.QUIT:
                        piano_roll_instance.app_running.clear()
                    piano_roll_instance.handle_slider_event(event)

                if not piano_roll_instance.app_running.is_set():
                    break

                current_time = self.get_current_playback_time()
                piano_roll_instance.draw(current_time)

                now = time.monotonic()
                if now - last_caption_update_time > 0.2:
                    fps = clock.get_fps()
                    self._pygame.display.set_caption(
                        f"Piano Roll - {fps:.1f} FPS - scroll {piano_roll_instance.scroll_speed:.0f}"
                    )
                    last_caption_update_time = now

                target_fps = max(1, int(getattr(piano_roll_instance, "fps_cap", 120)))
                clock.tick(target_fps)
        except Exception as e:
            traceback.print_exc()
            self._queue_ui(dpg.set_value, "status_text", f"Piano Roll Error: {e}")
            self._queue_ui(self._message_error, "Piano Roll Error", str(e))
        finally:
            try:
                piano_roll_instance.cleanup()
            except Exception:
                traceback.print_exc()
            if dpg.does_item_exist("skin_button"):
                dpg.enable_item("skin_button")


