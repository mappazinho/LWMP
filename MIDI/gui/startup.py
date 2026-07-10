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

class StartupMixin:
    """Methods for startup."""

    def _build_startup_summary(self):
        recommended_mode_label = self.all_backend_labels[self.recommended_mode]
        res_w, res_h = self.recommended_piano_roll_res
        note_limit = self.recommended_note_limit / 1_000_000
        return (
            f"Recommended Usage: {recommended_mode_label}\n"
            f"Recommended Notes: {note_limit:.1f} Million (based on free RAM)\n"
            f"Recommended Res:   {res_w} x {res_h} (for Piano Roll)"
        )


    def _build_startup_warning(self):
        warnings = []
        if not self.has_bundled_omnimidi:
            warnings.append("Bundled synth DLL not found. System PATH mode is recommended.")
        if not self.controller.bass_engine_cls:
            warnings.append("BASSMIDI engine not available. Buffered prerender mode is disabled.")
        return "\n".join(warnings)


    def _prepare_startup_screen(self):
        self._prepare_recommendation_info()
        dpg.set_value("startup_summary", self._build_startup_summary())
        warning_text = self._build_startup_warning()
        dpg.set_value("startup_warning", warning_text if warning_text else " ")
        dpg.configure_item("startup_warning", color=(225, 132, 104) if warning_text else (196, 198, 204))
        dpg.configure_item("startup_bass_button", enabled=bool(self.controller.bass_engine_cls))
        dpg.set_value("backend_hint_text", self._build_audio_hint_text())
        viewport_w = dpg.get_viewport_client_width() or 900
        viewport_h = dpg.get_viewport_client_height() or 700
        startup_w = 560
        startup_h = 360
        dpg.set_item_pos(
            "startup_window",
            [
                max(20, int((viewport_w - startup_w) * 0.5)),
                max(20, int((viewport_h - startup_h) * 0.5)),
            ],
        )
        dpg.configure_item("startup_window", show=True)


    def _finalize_startup_choice(self, mode, resolution):
        selected_mode = self._normalize_backend_mode(mode)
        self.recommended_piano_roll_res = resolution
        self._CONFIG["audio"]["omnimidi_load_preference"] = selected_mode
        self._CONFIG["gui"]["startup_completed"] = True
        self._save_config(self._CONFIG)
        self.controller.config["audio"]["omnimidi_load_preference"] = selected_mode
        self.controller.config["gui"]["startup_completed"] = True
        self._prepare_recommendation_info()
        dpg.set_value("backend_combo", self._get_combo_label_for_mode(selected_mode))
        dpg.configure_item("startup_window", show=False)
        self.startup_ready = True
        self.initialize_audio_backend()


    def use_bassmidi_startup(self):
        self._finalize_startup_choice("bassmidi", self.recommended_piano_roll_res)


    def use_recommended_startup(self):
        self._finalize_startup_choice(self.recommended_mode, self.recommended_piano_roll_res)


    def use_default_startup(self):
        self._finalize_startup_choice("path", (1280, 720))


