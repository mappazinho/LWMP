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

class TransportMixin:
    """Methods for transport."""

    def _refresh_transport_button_state(self):
        has_midi = self.controller.parsed_midi is not None
        if has_midi:
            dpg.show_item("play_button")
            dpg.show_item("stop_button")
            dpg.show_item("piano_roll_button")
            dpg.show_item("render_button")
            dpg.enable_item("play_button")
            dpg.enable_item("stop_button")
            if self._PianoRoll is not None:
                dpg.enable_item("piano_roll_button")
            else:
                dpg.disable_item("piano_roll_button")
            if self._BassMidiEngine is not None:
                dpg.enable_item("render_button")
            else:
                dpg.disable_item("render_button")
        else:
            dpg.hide_item("play_button")
            dpg.hide_item("stop_button")
            dpg.hide_item("piano_roll_button")
            dpg.hide_item("render_button")
            dpg.disable_item("play_button")
            dpg.disable_item("stop_button")
            dpg.disable_item("piano_roll_button")
            dpg.disable_item("render_button")


    def initialize_audio_backend(self):
        current_mode = self._normalize_backend_mode(self._CONFIG["audio"].get("omnimidi_load_preference", "path"))
        self._refresh_soundfont_visibility()
        if current_mode == "bassmidi":
            self.set_status("Select a SoundFont to initialize BASSMIDI.")
            self._ensure_bassmidi_soundfont(self._complete_initialize_audio_backend)
            return

        self._complete_initialize_audio_backend()


    def _complete_initialize_audio_backend(self):
        self.set_status("Initializing audio backend...")
        self.controller.init_midi_backends(
            volume=dpg.get_value("volume_slider"),
            voices=dpg.get_value("voices_slider"),
            set_status=self.set_status,
            prompt_info=self._message_info,
            prompt_warning=self._message_warning,
            prompt_error=self._message_error,
            pick_soundfont=None,
            launch_sweep=self._launch_audio_sweep,
        )
        self.controller.set_playback_speed(dpg.get_value("speed_slider"))
        dpg.set_value("backend_hint_text", self._build_audio_hint_text())
        self._refresh_soundfont_text()
        self._refresh_soundfont_visibility()
        self._refresh_transport_button_state()


    def apply_audio_mode(self):
        selected_label = dpg.get_value("backend_combo")
        selected_mode = self._normalize_backend_mode(self.backend_values.get(selected_label, "path"))
        self._stop_playback_for_backend_reinit()
        if self.controller.active_midi_backend:
            try:
                self.controller.shutdown()
            except Exception as e:
                print(f"Failed to shutdown current backend: {e}")
        self._CONFIG["audio"]["omnimidi_load_preference"] = selected_mode
        self._save_config(self._CONFIG)
        self.controller.config["audio"]["omnimidi_load_preference"] = selected_mode
        dpg.set_value("backend_combo", self._get_combo_label_for_mode(selected_mode))
        self.controller.active_midi_backend = None
        self._refresh_soundfont_visibility()
        self.set_status("Reinitializing audio backend...")
        self.initialize_audio_backend()


    def on_volume_change(self, sender, app_data):
        self._CONFIG["audio"]["volume"] = float(app_data)
        self._save_config(self._CONFIG)
        if self.controller.active_midi_backend and hasattr(self.controller.active_midi_backend, "set_volume"):
            try:
                self.controller.active_midi_backend.set_volume(float(app_data))
            except Exception as e:
                print(f"Failed to set volume: {e}")


    def on_voice_limit_change(self, sender, app_data):
        self._CONFIG["audio"]["voices"] = int(app_data)
        self._save_config(self._CONFIG)
        if self.controller.active_midi_backend and hasattr(self.controller.active_midi_backend, "set_voices"):
            try:
                self.controller.active_midi_backend.set_voices(int(app_data))
            except Exception as e:
                print(f"Failed to set voices: {e}")


    def on_speed_change(self, sender, app_data):
        self._CONFIG["audio"]["speed"] = float(app_data)
        self._save_config(self._CONFIG)
        with self.playback_lock:
            self.controller.set_playback_speed(float(app_data))


