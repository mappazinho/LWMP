#!/usr/bin/env python3

import atexit
import bisect
import ctypes
import heapq
import math
import multiprocessing
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox

import dearpygui.dearpygui as dpg
import numpy as np
import psutil

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, "PARSER"))

from config import load_config, save_config, setup_omnimidi_preference
from midi_parser import GPU_NOTE_DTYPE, MidiParser
from player_controller import PlayerController

try:
    from pianoroll import PianoRoll
    import pygame
except ImportError:
    print("pianoroll.py not found. Piano roll will be disabled.")
    PianoRoll = None

DEBUG = False

try:
    if os.name == "nt":
        from midi_engine import OmniMidiEngine
    else:
        OmniMidiEngine = None
except ImportError:
    OmniMidiEngine = None
    print("midi_engine.py or OmniMIDI.dll not found.")
except Exception as e:
    OmniMidiEngine = None
    print(f"Error importing OmniMidiEngine: {e}")

try:
    from midi_engine_cython import BassMidiEngine
    print("Loaded Cython BASSMIDI engine.")
except ImportError:
    try:
        from bassmidi_engine import BassMidiEngine
        print("Loaded Python BASSMIDI engine.")
    except ImportError:
        BassMidiEngine = None
        print("No BASSMIDI engine found.")
except Exception as e:
    try:
        from bassmidi_engine import BassMidiEngine
        print(f"Cython BASSMIDI engine failed, using Python fallback: {e}")
    except Exception as python_error:
        BassMidiEngine = None
        print(f"Error importing BassMidiEngine: {python_error}")

CONFIG = setup_omnimidi_preference(load_config())


def run_parser_process(filepath, result_queue):
    try:
        parser = MidiParser(filepath)
        total_events = parser.count_total_events()
        result_queue.put(("total_events", total_events))
        parser.parse(result_queue, total_events=total_events)
        result_data = {
            "filename": parser.filename,
            "ticks_per_beat": parser.ticks_per_beat,
            "total_duration_sec": parser.total_duration_sec,
            "note_data_for_gpu": parser.note_data_for_gpu,
            "note_events_for_playback": parser.note_events_for_playback,
            "pitch_bend_events": parser.pitch_bend_events,
            "control_change_events": getattr(parser, "control_change_events", []),
        }
        result_queue.put(("success", result_data))
    except Exception as e:
        traceback.print_exc()
        result_queue.put(("error", str(e)))


class DpgMidiPlayerApp:
    def __init__(self):
        self.controller = PlayerController(
            CONFIG,
            script_dir,
            BassMidiEngine,
            OmniMidiEngine,
            save_config,
            debug=DEBUG,
        )

        self.playback_thread = None
        self.audio_sweep_thread = None
        self.playback_lock = threading.Lock()
        self.process = None
        self.cpu_history = deque([0.0] * 100, maxlen=100)
        self.nps_history = deque([0] * 100, maxlen=100)
        self.ui_actions = queue.Queue()
        self.loading_visible = False
        self.was_piano_roll_open_before_unload = False
        self.last_piano_roll_res = None
        self.piano_roll = None
        self.piano_roll_thread = None
        self._seek_was_active = False
        self._cleaned_up = False
        self.startup_ready = False
        self.pending_midi_name = None
        self.has_bundled_omnimidi = os.path.exists(os.path.join(script_dir, "OmniMIDI.dll"))
        self.recommended_mode = "local" if self.has_bundled_omnimidi else "path"
        self.recommended_note_limit = 20_000_000
        self.last_cpu_sample_time = 0.0
        self.last_cpu_percent = 0.0

        self.all_backend_labels = {
            "bassmidi": "BASSMIDI (Buffered)",
            "path": "OmniMIDI (System PATH)",
            "local": "OmniMIDI (Bundled DLL)",
        }
        self.backend_labels = self._build_backend_labels()
        self.backend_values = {label: value for value, label in self.backend_labels.items()}

        self.screen_width, self.screen_height = self._get_screen_size()
        self.recommended_piano_roll_res = (1366, 768) if self.screen_height > 768 else (640, 360)
        self.available_resolutions = self._build_resolution_list()

        self._build_ui()
        self._bind_theme()
        self._initialize_process_monitoring()
        self._prepare_startup_screen()
        atexit.register(self.cleanup)

    def _gui_cfg(self):
        return CONFIG["gui"]

    def _color_tuple(self, key):
        color = self._gui_cfg()[key]
        return int(color[0]), int(color[1]), int(color[2])

    def _mix_color(self, base, target, amount):
        amount = max(0.0, min(float(amount), 1.0))
        return [
            int(round((base[i] * (1.0 - amount)) + (target[i] * amount)))
            for i in range(3)
        ]

    def _derive_palette_from_seed(self, seed):
        seed = [int(max(0, min(255, c))) for c in seed[:3]]
        dark = [16, 18, 22]
        panel = [26, 28, 34]
        frame = [34, 38, 46]
        light = [232, 235, 240]
        muted = [150, 156, 166]

        return {
            "theme_seed": seed,
            "window_bg": self._mix_color(dark, seed, 0.08),
            "child_bg": self._mix_color(panel, seed, 0.12),
            "frame_bg": self._mix_color(frame, seed, 0.15),
            "frame_bg_hovered": self._mix_color(frame, seed, 0.28),
            "frame_bg_active": self._mix_color(frame, seed, 0.42),
            "button": self._mix_color(seed, light, 0.08),
            "button_hovered": self._mix_color(seed, light, 0.2),
            "button_active": self._mix_color(seed, dark, 0.15),
            "accent_text": self._mix_color(seed, light, 0.42),
            "muted_text": self._mix_color(muted, seed, 0.16),
            "body_text": self._mix_color(light, seed, 0.08),
        }

    def _set_item_visibility(self, item_tag, visible):
        if dpg.does_item_exist(item_tag):
            dpg.configure_item(item_tag, show=bool(visible))

    def _nps_series_theme_tag(self, value):
        if value > 1_000_000:
            return "plot_theme_dark_red"
        if value > 500_000:
            return "plot_theme_red"
        if value > 200_000:
            return "plot_theme_orange"
        if value > 100_000:
            return "plot_theme_yellow"
        return "plot_theme_normal"

    def _cpu_series_theme_tag(self, value):
        if value > 75.0:
            return "plot_theme_red"
        if value > 50.0:
            return "plot_theme_orange"
        if value > 30.0:
            return "plot_theme_yellow"
        return "plot_theme_normal"

    def _apply_graph_series_colors(self, nps_value, cpu_value):
        if dpg.does_item_exist("nps_series"):
            dpg.bind_item_theme("nps_series", self._nps_series_theme_tag(nps_value))
        if dpg.does_item_exist("cpu_series"):
            dpg.bind_item_theme("cpu_series", self._cpu_series_theme_tag(cpu_value))

    def _update_now_playing_header(self, midi_name=None):
        if midi_name is not None:
            self.pending_midi_name = midi_name

        active_name = None
        if self.controller.parsed_midi is not None and getattr(self.controller.parsed_midi, "filename", None):
            active_name = os.path.basename(self.controller.parsed_midi.filename)
        elif self.pending_midi_name:
            active_name = self.pending_midi_name

        header_text = "Now Playing"
        if active_name:
            header_text = f"Now Playing: {active_name}"

        if dpg.does_item_exist("now_playing_text"):
            dpg.set_value("now_playing_text", header_text)

    def _get_screen_size(self):
        try:
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            root = tk.Tk()
            root.withdraw()
            try:
                return root.winfo_screenwidth(), root.winfo_screenheight()
            finally:
                root.destroy()

    def _run_dialog(self, callback):
        root = tk.Tk()
        root.withdraw()
        try:
            return callback(root)
        finally:
            root.destroy()

    def _build_resolution_list(self):
        common = [
            (640, 360),
            (854, 480),
            (1024, 576),
            (1280, 720),
            (1366, 768),
            (1600, 900),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
        ]
        available = [res for res in common if res[0] <= self.screen_width and res[1] <= self.screen_height]
        native = (self.screen_width, self.screen_height)
        if native not in available:
            available.append(native)
        available.sort(key=lambda item: item[0])
        return available

    def _build_backend_labels(self):
        labels = {
            "bassmidi": self.all_backend_labels["bassmidi"],
            "path": self.all_backend_labels["path"],
        }
        if self.has_bundled_omnimidi:
            labels["local"] = self.all_backend_labels["local"]
        return labels

    def _get_combo_label_for_mode(self, mode):
        return self.backend_labels.get(mode, self.backend_labels["path"])

    def _normalize_backend_mode(self, mode):
        if mode == "local" and not self.has_bundled_omnimidi:
            return "path"
        if mode not in self.backend_labels:
            return "path"
        return mode

    def _detect_recommended_note_limit(self):
        try:
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            est_limit = int(available_ram_gb * 19_000_000)
            return max(est_limit, 1_000_000)
        except Exception:
            return 20_000_000

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
            warnings.append("Bundled OmniMIDI DLL not found. System PATH mode is recommended.")
        if not self.controller.bass_engine_cls:
            warnings.append("BASSMIDI engine not available. Buffered prerender mode is disabled.")
        return "\n".join(warnings)

    def _prepare_startup_screen(self):
        self.recommended_note_limit = self._detect_recommended_note_limit()
        dpg.set_value("startup_summary", self._build_startup_summary())
        warning_text = self._build_startup_warning()
        dpg.set_value("startup_warning", warning_text if warning_text else " ")
        dpg.configure_item("startup_warning", color=(225, 132, 104) if warning_text else (196, 198, 204))
        dpg.configure_item("startup_bass_button", enabled=bool(self.controller.bass_engine_cls))
        dpg.set_value("backend_hint_text", self._build_audio_hint_text())
        dpg.configure_item("startup_window", show=True)

    def _finalize_startup_choice(self, mode, resolution):
        selected_mode = self._normalize_backend_mode(mode)
        self.recommended_piano_roll_res = resolution
        CONFIG["audio"]["omnimidi_load_preference"] = selected_mode
        save_config(CONFIG)
        self.controller.config["audio"]["omnimidi_load_preference"] = selected_mode
        dpg.set_value("backend_combo", self._get_combo_label_for_mode(selected_mode))
        dpg.configure_item("startup_window", show=False)
        self.initialize_audio_backend()
        self.startup_ready = True

    def use_bassmidi_startup(self):
        self._finalize_startup_choice("bassmidi", self.recommended_piano_roll_res)

    def use_recommended_startup(self):
        self._finalize_startup_choice(self.recommended_mode, self.recommended_piano_roll_res)

    def use_default_startup(self):
        self._finalize_startup_choice("path", (1280, 720))

    def _build_audio_hint_text(self):
        lines = []
        lines.append(
            "Bundled OmniMIDI DLL: detected" if self.has_bundled_omnimidi else "Bundled OmniMIDI DLL: missing"
        )
        lines.append(
            "BASSMIDI: available" if self.controller.bass_engine_cls else "BASSMIDI: unavailable"
        )
        if self.startup_ready:
            current_mode = self.controller.config["audio"].get("omnimidi_load_preference", "path")
            lines.append(f"Current Mode: {self.all_backend_labels.get(current_mode, self.all_backend_labels['path'])}")
        else:
            lines.append("Choose a startup mode to initialize audio.")
        return "\n".join(lines)

    def _apply_gui_customization(self):
        gui_cfg = self._gui_cfg()
        self._set_item_visibility("subtitle_text", gui_cfg["show_subtitle"])
        self._set_item_visibility("audio_panel", gui_cfg["show_audio_panel"])
        self._set_item_visibility("status_text", gui_cfg["show_status_line"])
        self._set_item_visibility("backend_hint_text", gui_cfg["show_backend_hint"])
        self._set_item_visibility("performance_section", gui_cfg["show_performance_panel"])
        self._set_item_visibility("nps_graph_cell", gui_cfg["show_nps_graph"])
        self._set_item_visibility("cpu_graph_cell", gui_cfg["show_cpu_graph"])
        self._bind_theme()

    def show_customize_window(self):
        gui_cfg = self._gui_cfg()
        dpg.set_value("custom_show_subtitle", gui_cfg["show_subtitle"])
        dpg.set_value("custom_show_audio_panel", gui_cfg["show_audio_panel"])
        dpg.set_value("custom_show_performance_panel", gui_cfg["show_performance_panel"])
        dpg.set_value("custom_show_status_line", gui_cfg["show_status_line"])
        dpg.set_value("custom_show_backend_hint", gui_cfg["show_backend_hint"])
        dpg.set_value("custom_show_nps_graph", gui_cfg["show_nps_graph"])
        dpg.set_value("custom_show_cpu_graph", gui_cfg["show_cpu_graph"])
        dpg.set_value("custom_theme_seed", gui_cfg["theme_seed"])

        for key in (
            "window_bg",
            "child_bg",
            "frame_bg",
            "frame_bg_hovered",
            "frame_bg_active",
            "button",
            "button_hovered",
            "button_active",
            "accent_text",
            "muted_text",
            "body_text",
        ):
            dpg.set_value(f"custom_{key}", gui_cfg[key])

        dpg.configure_item("customize_window", show=True)

    def apply_theme_seed(self):
        gui_cfg = self._gui_cfg()
        palette = self._derive_palette_from_seed(dpg.get_value("custom_theme_seed"))
        for key, value in palette.items():
            gui_cfg[key] = value
            if dpg.does_item_exist(f"custom_{key}"):
                dpg.set_value(f"custom_{key}", value)
        self._bind_theme()

    def save_customize_settings(self):
        gui_cfg = self._gui_cfg()
        gui_cfg["show_subtitle"] = bool(dpg.get_value("custom_show_subtitle"))
        gui_cfg["show_audio_panel"] = bool(dpg.get_value("custom_show_audio_panel"))
        gui_cfg["show_performance_panel"] = bool(dpg.get_value("custom_show_performance_panel"))
        gui_cfg["show_status_line"] = bool(dpg.get_value("custom_show_status_line"))
        gui_cfg["show_backend_hint"] = bool(dpg.get_value("custom_show_backend_hint"))
        gui_cfg["show_nps_graph"] = bool(dpg.get_value("custom_show_nps_graph"))
        gui_cfg["show_cpu_graph"] = bool(dpg.get_value("custom_show_cpu_graph"))
        gui_cfg["theme_seed"] = [int(v) for v in dpg.get_value("custom_theme_seed")[:3]]

        for key in (
            "window_bg",
            "child_bg",
            "frame_bg",
            "frame_bg_hovered",
            "frame_bg_active",
            "button",
            "button_hovered",
            "button_active",
            "accent_text",
            "muted_text",
            "body_text",
        ):
            gui_cfg[key] = [int(v) for v in dpg.get_value(f"custom_{key}")[:3]]

        save_config(CONFIG)
        self._apply_gui_customization()
        dpg.configure_item("customize_window", show=False)

    def _build_ui(self):
        dpg.create_context()
        dpg.create_viewport(title="Lightweight MIDI Player (DearPyGUI)", width=1040, height=720)

        default_mode = self._normalize_backend_mode(CONFIG["audio"].get("omnimidi_load_preference", "path"))
        default_combo_label = self._get_combo_label_for_mode(default_mode)

        with dpg.window(tag="main_window", label="LWMP", no_title_bar=True, no_scrollbar=True, no_scroll_with_mouse=True):
            with dpg.group(tag="app_shell"):
                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=False):
                    dpg.add_table_column(init_width_or_weight=1.15)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    with dpg.table_row():
                        with dpg.table_cell():
                            dpg.add_text("Lightweight MIDI Player", tag="title_text")
                            dpg.add_text(
                                "DearPyGUI shell for playback, parsing, and piano roll control.",
                                tag="subtitle_text",
                            )
                        with dpg.table_cell():
                            with dpg.group(horizontal=True):
                                dpg.add_button(tag="load_button", label="Load MIDI", callback=self.on_load_unload, width=118, height=30)
                                dpg.add_button(tag="play_button", label="Play", callback=self.toggle_play_pause, enabled=False, show=False, width=90, height=30)
                                dpg.add_button(tag="stop_button", label="Stop", callback=self.stop_playback, enabled=False, show=False, width=90, height=30)
                                dpg.add_button(
                                    tag="piano_roll_button",
                                    label="Piano Roll",
                                    callback=self.show_piano_roll_dialog,
                                    enabled=False,
                                    show=False,
                                    width=114,
                                    height=30,
                                )

                dpg.add_separator()

                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True):
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(init_width_or_weight=1.0)

                    with dpg.table_row():
                        with dpg.table_cell():
                            dpg.add_text("Now Playing", tag="now_playing_text", color=(223, 177, 103), wrap=470)
                            dpg.add_text(
                                "Choose a startup mode to initialize audio.",
                                tag="status_text",
                                wrap=470,
                                color=(196, 198, 204),
                            )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Position", color=(160, 166, 178))
                                dpg.add_text("00:00 / 00:00", tag="time_text")
                                dpg.add_spacer(width=14)
                                dpg.add_text("Notes", color=(160, 166, 178))
                                dpg.add_text("0 / 0", tag="note_count_value")
                            dpg.add_text("Seek", color=(160, 166, 178))
                            dpg.add_slider_float(
                                tag="seek_slider",
                                min_value=0.0,
                                max_value=100.0,
                                default_value=0.0,
                                enabled=False,
                                width=-1,
                                format="%.3f",
                            )

                            with dpg.item_handler_registry(tag="seek_handler"):
                                dpg.add_item_activated_handler(callback=self.on_seek_start)
                                dpg.add_item_deactivated_after_edit_handler(callback=self.on_seek_end)
                            dpg.bind_item_handler_registry("seek_slider", "seek_handler")

                        with dpg.table_cell(tag="audio_panel"):
                            dpg.add_text("Audio", color=(223, 177, 103))
                            dpg.add_text("", tag="backend_hint_text", wrap=470, color=(160, 166, 178))
                            dpg.add_text("Playback Mode", color=(160, 166, 178))
                            dpg.add_combo(
                                items=list(self.backend_values.keys()),
                                default_value=default_combo_label,
                                tag="backend_combo",
                                width=-1,
                            )
                            dpg.add_button(label="Apply Audio Mode", callback=self.apply_audio_mode, width=-1, height=30)
                            dpg.add_text("Current SoundFont", color=(160, 166, 178))
                            dpg.add_text("No SoundFont selected", tag="soundfont_text", wrap=470)
                            dpg.add_button(label="Change SoundFont", callback=self.change_soundfont, width=-1, height=28)
                            dpg.add_text("Synth Controls", color=(160, 166, 178))
                            dpg.add_slider_float(
                                label="Volume",
                                tag="volume_slider",
                                min_value=0.0,
                                max_value=1.0,
                                default_value=0.5,
                                width=-1,
                                callback=self.on_volume_change,
                            )
                            dpg.add_slider_int(
                                label="Max Voices",
                                tag="voices_slider",
                                min_value=1,
                                max_value=2000,
                                default_value=512,
                                width=-1,
                                callback=self.on_voice_limit_change,
                            )

                dpg.add_separator()
                with dpg.group(tag="performance_section"):
                    dpg.add_text("Performance", color=(223, 177, 103))
                    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True):
                        dpg.add_table_column(init_width_or_weight=1.0)
                        dpg.add_table_column(init_width_or_weight=1.0)
                        dpg.add_table_column(init_width_or_weight=0.9)
                        dpg.add_table_column(init_width_or_weight=1.2)
                        with dpg.table_row():
                            with dpg.table_cell():
                                with dpg.group(horizontal=True):
                                    dpg.add_text("NPS", color=(160, 166, 178))
                                    dpg.add_text("0", tag="nps_text")
                            with dpg.table_cell():
                                dpg.add_text("Max: 0", tag="nps_max_text", color=(196, 198, 204))
                            with dpg.table_cell():
                                with dpg.group(horizontal=True):
                                    dpg.add_text("CPU", color=(160, 166, 178))
                                    dpg.add_text("0.0%", tag="cpu_text")
                            with dpg.table_cell():
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Runtime", color=(160, 166, 178))
                                    dpg.add_text("Slowdown: 0.0%", tag="slowdown_text")

                    plot_x = list(range(100))
                    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True):
                        dpg.add_table_column(init_width_or_weight=1.0)
                        dpg.add_table_column(init_width_or_weight=1.0)
                        with dpg.table_row():
                            with dpg.table_cell(tag="nps_graph_cell"):
                                with dpg.plot(
                                    label="NPS Graph",
                                    height=150,
                                    width=-1,
                                    anti_aliased=True,
                                    no_menus=True,
                                    no_box_select=True,
                                    no_mouse_pos=True,
                                ):
                                    dpg.add_plot_axis(
                                        dpg.mvXAxis,
                                        tag="nps_x_axis",
                                        no_tick_labels=True,
                                        no_tick_marks=True,
                                        no_initial_fit=True,
                                        no_menus=True,
                                        lock_min=True,
                                        lock_max=True,
                                    )
                                    with dpg.plot_axis(
                                        dpg.mvYAxis,
                                        tag="nps_y_axis",
                                        no_initial_fit=True,
                                        no_menus=True,
                                        lock_min=True,
                                        lock_max=True,
                                        tick_format="%.0f",
                                    ):
                                        dpg.add_line_series(plot_x, list(self.nps_history), tag="nps_series")

                            with dpg.table_cell(tag="cpu_graph_cell"):
                                with dpg.plot(
                                    label="CPU Graph",
                                    height=150,
                                    width=-1,
                                    anti_aliased=True,
                                    no_menus=True,
                                    no_box_select=True,
                                    no_mouse_pos=True,
                                ):
                                    dpg.add_plot_axis(
                                        dpg.mvXAxis,
                                        tag="cpu_x_axis",
                                        no_tick_labels=True,
                                        no_tick_marks=True,
                                        no_initial_fit=True,
                                        no_menus=True,
                                        lock_min=True,
                                        lock_max=True,
                                    )
                                    with dpg.plot_axis(
                                        dpg.mvYAxis,
                                        tag="cpu_y_axis",
                                        no_initial_fit=True,
                                        no_menus=True,
                                        lock_min=True,
                                        lock_max=True,
                                        tick_format="%.0f",
                                    ):
                                        dpg.add_line_series(plot_x, list(self.cpu_history), tag="cpu_series")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        tag="customize_button",
                        label="Customize",
                        callback=self.show_customize_window,
                        width=104,
                        height=30,
                    )

            with dpg.window(
                tag="loading_window",
                label="Loading MIDI",
                modal=True,
                show=False,
                no_resize=True,
                no_collapse=True,
                width=420,
                height=120,
            ):
                dpg.add_text("Initializing...", tag="loading_label")
                dpg.add_progress_bar(tag="loading_progress", default_value=0.0, width=-1)

            with dpg.window(
                tag="piano_roll_window",
                label="Piano Roll Settings",
                modal=True,
                show=False,
                no_resize=True,
                no_collapse=True,
                width=320,
                height=220,
            ):
                dpg.add_text("Select Resolution")
                dpg.add_combo(
                    items=[f"{w} x {h}" for w, h in self.available_resolutions],
                    default_value=f"{self.recommended_piano_roll_res[0]} x {self.recommended_piano_roll_res[1]}",
                    tag="piano_roll_resolution",
                    width=220,
                )
                dpg.add_button(label="Launch Piano Roll", callback=self.launch_selected_piano_roll)
                dpg.add_button(label="Close", callback=lambda: dpg.configure_item("piano_roll_window", show=False))

            with dpg.window(
                tag="customize_window",
                label="Customize UI",
                modal=True,
                show=False,
                no_resize=True,
                no_collapse=True,
                width=640,
                height=620,
            ):
                dpg.add_text("Visibility", color=(223, 177, 103))
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Subtitle", tag="custom_show_subtitle")
                    dpg.add_checkbox(label="Audio Panel", tag="custom_show_audio_panel")
                    dpg.add_checkbox(label="Performance", tag="custom_show_performance_panel")
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Status Line", tag="custom_show_status_line")
                    dpg.add_checkbox(label="Backend Hint", tag="custom_show_backend_hint")
                    dpg.add_checkbox(label="NPS Graph", tag="custom_show_nps_graph")
                    dpg.add_checkbox(label="CPU Graph", tag="custom_show_cpu_graph")

                dpg.add_separator()
                dpg.add_text("Master Theme Color", color=(223, 177, 103))
                dpg.add_color_edit(label="Theme Seed", tag="custom_theme_seed", no_alpha=True, width=-1)
                dpg.add_button(label="Generate Palette From Theme Color", callback=self.apply_theme_seed, width=-1, height=32)

                dpg.add_separator()
                dpg.add_text("Colors", color=(223, 177, 103))
                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    with dpg.table_row():
                        with dpg.table_cell():
                            dpg.add_color_edit(label="Window BG", tag="custom_window_bg", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Child BG", tag="custom_child_bg", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Frame BG", tag="custom_frame_bg", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Frame Hover", tag="custom_frame_bg_hovered", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Frame Active", tag="custom_frame_bg_active", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Button", tag="custom_button", no_alpha=True, width=-1)
                        with dpg.table_cell():
                            dpg.add_color_edit(label="Button Hover", tag="custom_button_hovered", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Button Active", tag="custom_button_active", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Accent Text", tag="custom_accent_text", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Muted Text", tag="custom_muted_text", no_alpha=True, width=-1)
                            dpg.add_color_edit(label="Body Text", tag="custom_body_text", no_alpha=True, width=-1)

                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save", callback=self.save_customize_settings, width=140, height=32)
                    dpg.add_button(label="Close", callback=lambda: dpg.configure_item("customize_window", show=False), width=140, height=32)

            with dpg.window(
                tag="startup_window",
                label="Startup Setup",
                modal=True,
                show=False,
                no_resize=True,
                no_collapse=True,
                no_close=True,
                width=560,
                height=360,
            ):
                dpg.add_text("Choose Audio Mode", color=(223, 177, 103))
                dpg.add_spacer(height=4)
                dpg.add_text(
                    "Select the backend before loading a MIDI. You can still change it later from the Audio panel.",
                    wrap=520,
                    color=(196, 198, 204),
                )
                dpg.add_spacer(height=10)
                dpg.add_text("", tag="startup_summary", wrap=520, color=(160, 196, 255))
                dpg.add_spacer(height=10)
                dpg.add_text("", tag="startup_warning", wrap=520)
                dpg.add_spacer(height=14)
                dpg.add_button(
                    tag="startup_bass_button",
                    label="Use BASSMIDI (Buffered)",
                    callback=self.use_bassmidi_startup,
                    width=-1,
                    height=34,
                )
                dpg.add_spacer(height=6)
                dpg.add_button(
                    label="Use Recommended Settings",
                    callback=self.use_recommended_startup,
                    width=-1,
                    height=34,
                )
                dpg.add_spacer(height=6)
                dpg.add_button(
                    label="Use Defaults (System PATH, 720p)",
                    callback=self.use_default_startup,
                    width=-1,
                    height=34,
                )

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        dpg.set_axis_limits("nps_x_axis", 0, 99)
        dpg.set_axis_limits("cpu_x_axis", 0, 99)
        dpg.set_axis_limits("nps_y_axis", 0, 100)
        dpg.set_axis_limits("cpu_y_axis", 0, 100)
        self._apply_gui_customization()

    def _bind_theme(self):
        gui_cfg = self._gui_cfg()
        if dpg.does_item_exist("global_theme"):
            dpg.delete_item("global_theme")
        with dpg.theme(tag="global_theme") as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, self._color_tuple("window_bg"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, self._color_tuple("child_bg"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, self._color_tuple("frame_bg"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, self._color_tuple("frame_bg_hovered"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, self._color_tuple("frame_bg_active"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, self._color_tuple("button"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self._color_tuple("button_hovered"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self._color_tuple("button_active"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, self._color_tuple("button"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, self._color_tuple("accent_text"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, self._color_tuple("frame_bg"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, self._color_tuple("frame_bg_hovered"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, self._color_tuple("frame_bg_active"), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 6, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 4, 4, category=dpg.mvThemeCat_Core)
        dpg.bind_theme(global_theme)

        for theme_tag, color in (
            ("plot_theme_normal", (96, 152, 255)),
            ("plot_theme_yellow", (232, 212, 92)),
            ("plot_theme_orange", (232, 145, 58)),
            ("plot_theme_red", (220, 70, 62)),
            ("plot_theme_dark_red", (128, 24, 24)),
        ):
            if not dpg.does_item_exist(theme_tag):
                with dpg.theme(tag=theme_tag):
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)

        if dpg.does_item_exist("title_text"):
            dpg.configure_item("title_text", color=self._color_tuple("accent_text"))
        if dpg.does_item_exist("subtitle_text"):
            dpg.configure_item("subtitle_text", color=self._color_tuple("muted_text"))
        if dpg.does_item_exist("now_playing_text"):
            dpg.configure_item("now_playing_text", color=self._color_tuple("accent_text"))
        if dpg.does_item_exist("status_text"):
            dpg.configure_item("status_text", color=self._color_tuple("body_text"))
        self._apply_graph_series_colors(
            self.nps_history[-1] if self.nps_history else 0,
            self.last_cpu_percent,
        )

    def _initialize_process_monitoring(self):
        try:
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)
        except Exception as e:
            print(f"Failed to initialize psutil: {e}")
            self.process = None

    def _message_info(self, title, text):
        self._run_dialog(lambda root: messagebox.showinfo(title, text, parent=root))

    def _message_warning(self, title, text):
        self._run_dialog(lambda root: messagebox.showwarning(title, text, parent=root))

    def _message_error(self, title, text):
        self._run_dialog(lambda root: messagebox.showerror(title, text, parent=root))

    def _pick_soundfont(self):
        return self._run_dialog(
            lambda root: filedialog.askopenfilename(
                parent=root,
                filetypes=(
                    ("SoundFont", "*.sf2 *.sfz"),
                    ("SF2 SoundFont", "*.sf2"),
                    ("SFZ SoundFont", "*.sfz"),
                    ("All files", "*.*"),
                ),
            )
        )

    def _current_soundfont_label(self):
        sf_path = CONFIG["audio"].get("soundfont_path")
        if sf_path:
            return os.path.basename(sf_path)
        return "No SoundFont selected"

    def _refresh_soundfont_text(self):
        if dpg.does_item_exist("soundfont_text"):
            dpg.set_value("soundfont_text", self._current_soundfont_label())

    def _refresh_transport_button_state(self):
        has_midi = self.controller.parsed_midi is not None
        if has_midi:
            dpg.show_item("play_button")
            dpg.show_item("stop_button")
            dpg.show_item("piano_roll_button")
            dpg.enable_item("play_button")
            dpg.enable_item("stop_button")
            if PianoRoll is not None:
                dpg.enable_item("piano_roll_button")
            else:
                dpg.disable_item("piano_roll_button")
        else:
            dpg.hide_item("play_button")
            dpg.hide_item("stop_button")
            dpg.hide_item("piano_roll_button")
            dpg.disable_item("play_button")
            dpg.disable_item("stop_button")
            dpg.disable_item("piano_roll_button")

    def _wait_for_audio_sweep(self, timeout=2.0):
        if self.audio_sweep_thread and self.audio_sweep_thread.is_alive():
            self.audio_sweep_thread.join(timeout)
        self.audio_sweep_thread = None

    def _launch_audio_sweep(self, backend):
        self._wait_for_audio_sweep(timeout=2.0)

        def _run_sweep():
            try:
                backend.test_piano_sweep()
            except Exception as e:
                print(f"Audio sweep failed: {e}")

        self.audio_sweep_thread = threading.Thread(target=_run_sweep, daemon=True)
        self.audio_sweep_thread.start()

    def _stop_playback_for_backend_reinit(self):
        self._wait_for_audio_sweep(timeout=2.0)
        self.controller.playing = False
        self.controller.paused = False
        self.controller.paused_for_seeking = False
        with self.playback_lock:
            self.controller.seek_request_time = None

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(1.0)

        self.playback_thread = None
        dpg.configure_item("play_button", label="Play")
        dpg.configure_item("voices_slider", enabled=True)

        if self.controller.active_midi_backend:
            try:
                self.controller.reset_playback_state()
            except Exception as e:
                print(f"Failed to reset backend before reinit: {e}")

        self.reset_graph_history()
        dpg.set_value("seek_slider", 0.0)
        if self.controller.parsed_midi:
            dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
            dpg.set_value("status_text", "Ready to play.")
        else:
            dpg.set_value("time_text", "00:00 / 00:00")
            dpg.set_value("status_text", "No file loaded.")
        self._refresh_transport_button_state()

    def _queue_ui(self, callback, *args, **kwargs):
        self.ui_actions.put((callback, args, kwargs))

    def _process_ui_queue(self):
        while True:
            try:
                callback, args, kwargs = self.ui_actions.get_nowait()
            except queue.Empty:
                break
            callback(*args, **kwargs)

    def format_time(self, seconds):
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def format_nps(self, n):
        n = int(n)
        if n < 1_000_000:
            return str(n)
        return f"{n/1_000_000:.1f}M".replace(".0M", "M")

    def set_status(self, text):
        dpg.set_value("status_text", text)
        dpg.set_value("backend_hint_text", self._build_audio_hint_text())

    def initialize_audio_backend(self):
        self.set_status("Initializing audio backend...")
        self.controller.init_midi_backends(
            volume=dpg.get_value("volume_slider"),
            voices=dpg.get_value("voices_slider"),
            set_status=self.set_status,
            prompt_info=self._message_info,
            prompt_warning=self._message_warning,
            prompt_error=self._message_error,
            pick_soundfont=self._pick_soundfont,
            launch_sweep=self._launch_audio_sweep,
        )
        dpg.set_value("backend_hint_text", self._build_audio_hint_text())
        self._refresh_soundfont_text()
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
        CONFIG["audio"]["omnimidi_load_preference"] = selected_mode
        save_config(CONFIG)
        self.controller.config["audio"]["omnimidi_load_preference"] = selected_mode
        dpg.set_value("backend_combo", self._get_combo_label_for_mode(selected_mode))
        self.controller.active_midi_backend = None
        self.set_status("Reinitializing audio backend...")
        self.initialize_audio_backend()

    def change_soundfont(self):
        sf_path = self._pick_soundfont()
        if not sf_path:
            return

        CONFIG["audio"]["soundfont_path"] = sf_path
        save_config(CONFIG)
        self.controller.config["audio"]["soundfont_path"] = sf_path
        self._refresh_soundfont_text()

        current_mode = self._normalize_backend_mode(CONFIG["audio"].get("omnimidi_load_preference", "path"))
        if current_mode == "bassmidi":
            self._stop_playback_for_backend_reinit()
            if self.controller.active_midi_backend:
                try:
                    self.controller.shutdown()
                except Exception as e:
                    print(f"Failed to shutdown current backend before soundfont change: {e}")
            self.controller.active_midi_backend = None
            self.set_status("Reinitializing BASSMIDI with new SoundFont...")
            self.initialize_audio_backend()
        else:
            self.set_status(f"SoundFont set to {os.path.basename(sf_path)}. It will be used when BASSMIDI is selected.")

    def on_volume_change(self, sender, app_data):
        if self.controller.active_midi_backend and hasattr(self.controller.active_midi_backend, "set_volume"):
            try:
                self.controller.active_midi_backend.set_volume(float(app_data))
            except Exception as e:
                print(f"Failed to set volume: {e}")

    def on_voice_limit_change(self, sender, app_data):
        if self.controller.active_midi_backend and hasattr(self.controller.active_midi_backend, "set_voices"):
            try:
                self.controller.active_midi_backend.set_voices(int(app_data))
            except Exception as e:
                print(f"Failed to set voices: {e}")

    def on_load_unload(self):
        if self.controller.parsed_midi is None:
            self.load_file()
        else:
            self.unload_file()

    def load_file(self):
        if not self.startup_ready:
            self._message_warning("Startup Required", "Choose a startup audio mode before loading a MIDI.")
            return

        if self.controller.playing:
            self.controller.playing = False
            self.controller.paused = False
            dpg.configure_item("play_button", label="Play")

        self.reset_playback_state()
        self.controller.parsed_midi = None
        self.controller.total_song_notes = 0
        self.controller.total_song_duration = 0.0
        self.pending_midi_name = None
        self._update_now_playing_header()
        dpg.set_value("time_text", "00:00 / 00:00")
        dpg.set_value("note_count_value", "0 / 0")
        dpg.set_value("seek_slider", 0.0)
        dpg.configure_item("seek_slider", enabled=False, max_value=100.0)
        self._refresh_transport_button_state()

        if self.controller.parser_process and self.controller.parser_process.is_alive():
            self._message_warning("Busy", "Already parsing a file. Please wait.")
            return

        filepath = self._run_dialog(
            lambda root: filedialog.askopenfilename(
                parent=root,
                filetypes=(("MIDI files", "*.mid"), ("All files", "*.*")),
            )
        )
        if not filepath:
            return

        self._update_now_playing_header(os.path.basename(filepath))
        dpg.set_value("status_text", f"Parsing {os.path.basename(filepath)}...")
        dpg.configure_item("load_button", enabled=False)
        dpg.set_value("loading_label", "Initializing...")
        dpg.set_value("loading_progress", 0.0)
        dpg.configure_item("loading_window", show=True)
        self.loading_visible = True

        try:
            self.controller.start_parse_job(
                filepath,
                multiprocessing.Queue,
                multiprocessing.Process,
                run_parser_process,
            )
        except Exception as e:
            self.controller.clear_parse_job()
            self.loading_visible = False
            dpg.configure_item("loading_window", show=False)
            self._message_error("Error", f"Failed to start parser: {e}")
            self.pending_midi_name = None
            self._update_now_playing_header()
            dpg.set_value("status_text", "Failed to start parser.")
            dpg.configure_item("load_button", enabled=True)

    def _poll_parser(self):
        try:
            for status, payload in self.controller.poll_parser_messages():
                event = self.controller.handle_parser_message(status, payload, start_padding=3.0, end_padding=3.0)
                if event["kind"] == "total_events":
                    if self.loading_visible:
                        dpg.set_value("loading_label", f"Found {event['total_events']:,} events. Parsing...")
                elif event["kind"] == "progress":
                    if self.loading_visible:
                        progress_payload = event["payload"]
                        if isinstance(progress_payload, dict):
                            current = progress_payload.get("current", 0)
                            total = progress_payload.get("total", 1)
                            eta = progress_payload.get("eta", 0)
                            fraction = (current / total) if total > 0 else 0.0
                            dpg.set_value("loading_progress", fraction)
                            eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "Calculating..."
                            dpg.set_value("loading_label", f"Parsing... {current:,} / {total:,} events ({eta_str})")
                        else:
                            dpg.set_value("loading_label", str(progress_payload))
                elif event["kind"] == "success":
                    self.loading_visible = False
                    dpg.configure_item("loading_window", show=False)
                    self._update_now_playing_header()
                    dpg.set_value("status_text", "Ready to play.")
                    dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
                    dpg.set_value("note_count_value", f"0 / {self.controller.total_song_notes:,}")
                    dpg.set_value("seek_slider", 0.0)
                    dpg.configure_item("seek_slider", enabled=True, max_value=float(self.controller.total_song_duration))
                    dpg.configure_item("play_button", label="Play")
                    self._refresh_transport_button_state()
                    dpg.configure_item("load_button", enabled=True, label="Unload MIDI")
                    self.reset_graph_history()

                    if self.was_piano_roll_open_before_unload and self.last_piano_roll_res and PianoRoll:
                        self.launch_piano_roll(*self.last_piano_roll_res)
                    self.was_piano_roll_open_before_unload = False
                    return
                else:
                    self.loading_visible = False
                    dpg.configure_item("loading_window", show=False)
                    self._message_error("Parse Error", f"Could not load MIDI file: {event['payload']}")
                    self.pending_midi_name = None
                    self._update_now_playing_header()
                    dpg.set_value("status_text", "Failed to load file.")
                    dpg.configure_item("load_button", enabled=True)
                    return
        except Exception as e:
            self.loading_visible = False
            dpg.configure_item("loading_window", show=False)
            self.controller.clear_parse_job()
            self._message_error("Error", f"Error checking parser status: {e}")
            self.pending_midi_name = None
            self._update_now_playing_header()
            dpg.set_value("status_text", "Error during parsing.")
            dpg.configure_item("load_button", enabled=True)

    def on_seek_start(self, sender, app_data):
        if not self.controller.parsed_midi:
            return
        if not self._seek_was_active:
            paused_for_seek = self.controller.begin_seek()
            if paused_for_seek:
                dpg.set_value("status_text", "Seeking...")
            self._seek_was_active = True

    def on_seek_end(self, sender, app_data):
        if not self.controller.parsed_midi:
            self._seek_was_active = False
            return
        seek_time = dpg.get_value("seek_slider")
        self.panic_all_notes_off()
        with self.playback_lock:
            resumed = self.controller.complete_seek(seek_time)
        if resumed:
            dpg.configure_item("play_button", label="Pause")
            dpg.set_value("status_text", "Playing...")
        if not self.controller.playing or self.controller.paused:
            dpg.set_value(
                "time_text",
                f"{self.format_time(seek_time)} / {self.format_time(self.controller.total_song_duration)}",
            )
        self._seek_was_active = False

    def toggle_play_pause(self):
        if self.controller.playing:
            if self.controller.paused:
                self.controller.resume_playback()
                dpg.configure_item("play_button", label="Pause")
                dpg.set_value("status_text", "Playing...")
            else:
                self.controller.pause_playback()
                dpg.configure_item("play_button", label="Resume")
                dpg.set_value("status_text", "Paused")
        else:
            if self.controller.parsed_midi is None:
                return
            current_time = self.get_current_playback_time()
            self.controller.start_playback(current_time)
            dpg.configure_item("play_button", label="Pause")
            dpg.set_value("status_text", "Playing...")
            dpg.configure_item("voices_slider", enabled=False)
            self.playback_thread = threading.Thread(target=self.play_music_thread, daemon=True)
            self.playback_thread.start()

    def stop_playback(self):
        self.controller.stop_playback()
        self.reset_playback_state()
        dpg.configure_item("play_button", label="Play")
        if self.controller.parsed_midi:
            dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
            dpg.set_value("status_text", "Ready to play.")
        dpg.configure_item("voices_slider", enabled=True)

    def unload_file(self):
        if self.controller.playing:
            self.controller.playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(0.1)

        self.controller.unload_midi()
        self.reset_playback_state()
        self.pending_midi_name = None
        self._update_now_playing_header()
        dpg.set_value("status_text", "No file loaded.")
        dpg.set_value("time_text", "00:00 / 00:00")
        dpg.set_value("note_count_value", "0 / 0")
        dpg.set_value("seek_slider", 0.0)
        dpg.configure_item("seek_slider", enabled=False, max_value=100.0)
        dpg.configure_item("play_button", label="Play")
        self._refresh_transport_button_state()

        if self.piano_roll and self.piano_roll.app_running.is_set():
            self.was_piano_roll_open_before_unload = True
            self.piano_roll.app_running.clear()
            if self.piano_roll_thread and self.piano_roll_thread.is_alive():
                self.piano_roll_thread.join(0.2)
            self.piano_roll = None
        else:
            self.was_piano_roll_open_before_unload = False

        dpg.configure_item("load_button", label="Load MIDI", enabled=True)
        dpg.configure_item("voices_slider", enabled=True)

    def reset_graph_history(self):
        self.nps_history.clear()
        self.cpu_history.clear()
        for _ in range(100):
            self.nps_history.append(0)
            self.cpu_history.append(0.0)
        self._update_plot_series()
        self._apply_graph_series_colors(0, 0.0)
        dpg.set_value("nps_max_text", "Max: 0")
        dpg.set_value("nps_text", "0")
        dpg.set_value("cpu_text", "0.0%")
        dpg.set_value("slowdown_text", "Slowdown: 0.0%")

    def reset_playback_state(self):
        self.controller.reset_playback_state()
        dpg.set_value("seek_slider", 0.0)
        self.reset_graph_history()
        self.panic_all_notes_off()

    def playback_finished(self):
        self.controller.finish_playback()
        dpg.configure_item("play_button", label="Play")
        if self.controller.parsed_midi:
            dpg.set_value("status_text", "Finished.")
            finished = self.format_time(self.controller.total_song_duration)
            dpg.set_value("time_text", f"{finished} / {finished}")
            dpg.set_value("seek_slider", self.controller.total_song_duration)
        dpg.configure_item("voices_slider", enabled=True)
        self._refresh_transport_button_state()
        self.panic_all_notes_off()

    def panic_all_notes_off(self):
        if self.controller.active_midi_backend is None:
            return
        try:
            self.controller.panic_all_notes_off()
        except Exception as e:
            print(f"Error during MIDI backend panic: {e}")

    def set_pitch_bend_range(self, semitones=12):
        self.controller.set_pitch_bend_range(semitones=semitones)

    def play_music_thread(self):
        if self.controller.active_midi_backend and getattr(self.controller.active_midi_backend, "buffering_enabled", False):
            self.play_music_thread_buffered()
        else:
            self.play_music_thread_realtime()

    def play_music_thread_buffered(self):
        try:
            if not self.controller.parsed_midi:
                self._queue_ui(self.playback_finished)
                return

            self._queue_ui(dpg.set_value, "status_text", "Pre-rendering events...")

            if self.controller.active_midi_backend:
                self.set_pitch_bend_range(semitones=12)
                self.controller.active_midi_backend.stop()

            note_events = self.controller.parsed_midi.note_events_for_playback
            pitch_bend_events = self.controller.parsed_midi.pitch_bend_events
            control_change_events = getattr(self.controller.parsed_midi, "control_change_events", [])

            count_notes = len(note_events)
            count_bends = len(pitch_bend_events)
            count_ccs = len(control_change_events)
            total_ops = (count_notes * 2) + count_bends + count_ccs

            times = np.empty(total_ops, dtype=np.float64)
            statuses = np.empty(total_ops, dtype=np.uint32)
            params = np.empty(total_ops, dtype=np.uint32)

            times[:count_notes] = note_events["on_time"]
            statuses[:count_notes] = 0x90 + note_events["channel"]
            params[:count_notes] = (note_events["velocity"].astype(np.uint32) << 8) | note_events["pitch"].astype(np.uint32)

            times[count_notes : count_notes * 2] = note_events["off_time"]
            statuses[count_notes : count_notes * 2] = 0x80 + note_events["channel"]
            params[count_notes : count_notes * 2] = note_events["pitch"].astype(np.uint32)

            if count_bends > 0:
                pb_arr = np.array(pitch_bend_events, dtype=[("time", "f8"), ("chan", "u4"), ("val", "u4")])
                start_idx = count_notes * 2
                end_idx = start_idx + count_bends
                times[start_idx:end_idx] = pb_arr["time"]
                statuses[start_idx:end_idx] = 0xE0 + pb_arr["chan"]
                bend_lsb = pb_arr["val"] & 0x7F
                bend_msb = (pb_arr["val"] >> 7) & 0x7F
                params[start_idx:end_idx] = (bend_msb << 8) | bend_lsb

            if count_ccs > 0:
                cc_arr = np.array(control_change_events, dtype=[("time", "f8"), ("chan", "u4"), ("cc", "u4"), ("val", "u4")])
                start_idx = (count_notes * 2) + count_bends
                end_idx = start_idx + count_ccs
                times[start_idx:end_idx] = cc_arr["time"]
                statuses[start_idx:end_idx] = 0xB0 + cc_arr["chan"]
                params[start_idx:end_idx] = (cc_arr["val"] << 8) | cc_arr["cc"]

            sort_indices = np.argsort(times)
            times = times[sort_indices]
            statuses = statuses[sort_indices]
            params = params[sort_indices]

            self.controller.active_midi_backend.upload_events(times, statuses, params)

            start_time = self.get_current_playback_time()
            self.controller.active_midi_backend.set_current_time(start_time)
            has_started_playback = False

            while self.controller.playing:
                if not self.controller.playing:
                    break

                requested_time = None
                with self.playback_lock:
                    if self.controller.seek_request_time is not None:
                        requested_time = self.controller.seek_request_time
                        self.controller.seek_request_time = None

                if requested_time is not None:
                    self.controller.active_midi_backend.stop()
                    self.controller.active_midi_backend.set_current_time(requested_time)
                    self.controller.buffered_playback_start_offset = requested_time
                    has_started_playback = False

                buffer_lvl = self.controller.active_midi_backend.fill_buffer(60.0)
                is_active = self.controller.active_midi_backend.is_active()

                if self.controller.paused:
                    self.controller.active_midi_backend.pause()
                    if buffer_lvl >= 60.0:
                        time.sleep(0.1)
                    else:
                        time.sleep(0.005)
                    continue

                if not has_started_playback:
                    if buffer_lvl > 4.0:
                        self._queue_ui(dpg.set_value, "status_text", "Playing...")
                        self.controller.active_midi_backend.play()
                        has_started_playback = True
                    else:
                        self._queue_ui(dpg.set_value, "status_text", f"Prerendering... {buffer_lvl:.1f}s / 4.0s")
                else:
                    if buffer_lvl < 0.2:
                        self.controller.active_midi_backend.pause()
                        self._queue_ui(dpg.set_value, "status_text", "Buffering...")
                        has_started_playback = False
                    elif not self.controller.paused and not is_active and buffer_lvl > 2.0:
                        self.controller.active_midi_backend.play()
                        self._queue_ui(dpg.set_value, "status_text", "Playing...")

                if buffer_lvl >= 60.0:
                    time.sleep(0.1)
                else:
                    time.sleep(0.005)

        except Exception as e:
            print(f"Buffered playback error: {e}")
            traceback.print_exc()
        finally:
            self._queue_ui(self.playback_finished)

    def play_music_thread_realtime(self):
        try:
            if not self.controller.parsed_midi:
                self._queue_ui(self.playback_finished)
                return

            self._queue_ui(dpg.set_value, "status_text", "Starting playback...")

            if self.controller.active_midi_backend:
                self.set_pitch_bend_range(semitones=12)

            note_events = self.controller.parsed_midi.note_events_for_playback
            pitch_bend_events = self.controller.parsed_midi.pitch_bend_events
            control_change_events = getattr(self.controller.parsed_midi, "control_change_events", [])

            num_note_events = len(note_events)
            num_pitch_bend_events = len(pitch_bend_events)
            num_control_change_events = len(control_change_events)

            if self.controller.active_midi_backend:
                for channel in range(16):
                    status = 0xE0 + channel
                    param = (0x40 << 8) | 0x00
                    self.controller.active_midi_backend.send_raw_event(status, param)

            start_time = self.get_current_playback_time()
            note_event_index = bisect.bisect_left(note_events["on_time"], start_time)
            pitch_bend_index = bisect.bisect_left(pitch_bend_events, (start_time, -float("inf"), -float("inf")))
            control_change_index = bisect.bisect_left(control_change_events, (start_time, -float("inf"), -float("inf"), -float("inf")))

            with self.playback_lock:
                self.controller.last_processed_event_time = start_time

            note_off_heap = []
            if note_event_index > 0:
                notes_before_now = note_events[:note_event_index]
                active_notes = notes_before_now[notes_before_now["off_time"] > start_time]
                for note in active_notes:
                    heapq.heappush(note_off_heap, (note["off_time"], note["pitch"], note["channel"]))

            while self.controller.playing and (
                note_event_index < num_note_events
                or pitch_bend_index < num_pitch_bend_events
                or control_change_index < num_control_change_events
                or len(note_off_heap) > 0
            ):
                while self.controller.paused:
                    if not self.controller.playing:
                        break
                    time.sleep(0.01)
                if not self.controller.playing:
                    break

                requested_time = None
                with self.playback_lock:
                    if self.controller.seek_request_time is not None:
                        requested_time = self.controller.seek_request_time
                        self.controller.seek_request_time = None

                if requested_time is not None:
                    with self.playback_lock:
                        self.controller.last_processed_event_time = requested_time
                    note_event_index = bisect.bisect_left(note_events["on_time"], requested_time)
                    pitch_bend_index = bisect.bisect_left(pitch_bend_events, (requested_time, -float("inf"), -float("inf")))
                    control_change_index = bisect.bisect_left(control_change_events, (requested_time, -float("inf"), -float("inf"), -float("inf")))
                    self.controller.playback_start_time = time.monotonic() - requested_time
                    self.controller.total_paused_duration = 0.0
                    self.controller.paused_at_time = 0.0
                    self.controller.notes_played_count = note_event_index
                    self.controller.nps_event_timestamps.clear()
                    note_off_heap.clear()
                    if note_event_index > 0:
                        notes_before_now = note_events[:note_event_index]
                        active_notes = notes_before_now[notes_before_now["off_time"] > requested_time]
                        for note in active_notes:
                            heapq.heappush(note_off_heap, (note["off_time"], note["pitch"], note["channel"]))

                next_note_on_time = note_events[note_event_index]["on_time"] if note_event_index < num_note_events else float("inf")
                next_note_off_time = note_off_heap[0][0] if note_off_heap else float("inf")
                next_pitch_bend_time = pitch_bend_events[pitch_bend_index][0] if pitch_bend_index < num_pitch_bend_events else float("inf")
                next_control_change_time = control_change_events[control_change_index][0] if control_change_index < num_control_change_events else float("inf")
                event_time_sec = min(next_note_on_time, next_note_off_time, next_pitch_bend_time, next_control_change_time)

                if event_time_sec == float("inf"):
                    break

                with self.playback_lock:
                    self.controller.last_processed_event_time = event_time_sec

                target_wall_time = self.controller.playback_start_time + event_time_sec + self.controller.total_paused_duration
                sleep_duration = target_wall_time - time.monotonic()
                self.controller.current_lag = max(0, -sleep_duration)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                if not self.controller.playing:
                    break
                if self.controller.paused:
                    continue
                with self.playback_lock:
                    if self.controller.seek_request_time is not None:
                        continue

                try:
                    while note_off_heap and note_off_heap[0][0] <= event_time_sec:
                        off_time, pitch, channel = heapq.heappop(note_off_heap)
                        status = 0x80 + channel
                        if self.controller.active_midi_backend:
                            self.controller.active_midi_backend.send_raw_event(status, pitch)

                    while note_event_index < num_note_events and note_events[note_event_index]["on_time"] <= event_time_sec:
                        note = note_events[note_event_index]
                        note_event_index += 1
                        pitch = int(note["pitch"])
                        vel = int(note["velocity"])
                        channel = int(note["channel"])

                        self.controller.notes_played_count += 1
                        self.controller.nps_event_timestamps.append(note["on_time"])

                        if vel >= 20:
                            status_on = 0x90 + channel
                            if self.controller.active_midi_backend:
                                param = (vel << 8) | pitch
                                self.controller.active_midi_backend.send_raw_event(status_on, param)
                            heapq.heappush(note_off_heap, (note["off_time"], pitch, channel))

                    while pitch_bend_index < num_pitch_bend_events and pitch_bend_events[pitch_bend_index][0] <= event_time_sec:
                        _time, channel, pitch_value = pitch_bend_events[pitch_bend_index]
                        status = 0xE0 + channel
                        data1 = pitch_value & 0x7F
                        data2 = (pitch_value >> 7) & 0x7F
                        if self.controller.active_midi_backend:
                            param = (data2 << 8) | data1
                            self.controller.active_midi_backend.send_raw_event(status, param)
                        pitch_bend_index += 1

                    while control_change_index < num_control_change_events and control_change_events[control_change_index][0] <= event_time_sec:
                        _time, channel, controller, value = control_change_events[control_change_index]
                        status = 0xB0 + channel
                        if self.controller.active_midi_backend:
                            param = (value << 8) | controller
                            self.controller.active_midi_backend.send_raw_event(status, param)
                        control_change_index += 1
                except Exception as e:
                    print(f"MIDI backend send error: {e}")
                    self._queue_ui(dpg.set_value, "status_text", f"Playback Error: {e}")
                    break
        except Exception as e:
            print(f"Playback thread error: {e}")
            traceback.print_exc()
        finally:
            self._queue_ui(self.playback_finished)

    def get_current_playback_time(self):
        current_time = self.controller.get_current_playback_time()
        if not self.controller.playing:
            return dpg.get_value("seek_slider")
        return current_time

    def show_piano_roll_dialog(self):
        if self.piano_roll and self.piano_roll.app_running.is_set():
            self._message_info("Piano Roll", "Piano Roll is already running.")
            return
        dpg.configure_item("piano_roll_window", show=True)

    def launch_selected_piano_roll(self):
        selected = dpg.get_value("piano_roll_resolution")
        width_str, height_str = selected.split(" x ")
        dpg.configure_item("piano_roll_window", show=False)
        self.launch_piano_roll(int(width_str), int(height_str))

    def launch_piano_roll(self, width, height):
        if not PianoRoll:
            return
        self.last_piano_roll_res = (width, height)
        self.piano_roll = PianoRoll(width, height, CONFIG)
        self.piano_roll_thread = threading.Thread(target=self.run_piano_roll, daemon=True)
        self.piano_roll_thread.start()

        if self.controller.parsed_midi:
            notes_for_gpu = np.ascontiguousarray(self.controller.parsed_midi.note_data_for_gpu)
            threading.Timer(
                0.5,
                lambda: self.piano_roll.load_midi(notes_for_gpu, self.get_current_playback_time_thread_safe),
            ).start()

    def get_current_playback_time_thread_safe(self):
        return self.controller.current_playback_time_for_threads

    def run_piano_roll(self):
        piano_roll_instance = self.piano_roll
        try:
            piano_roll_instance.init_pygame_and_gl()
            clock = pygame.time.Clock()
            pygame.time.set_timer(pygame.USEREVENT, 250)
            last_caption_update_time = 0

            while piano_roll_instance.app_running.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        piano_roll_instance.app_running.clear()
                    piano_roll_instance.handle_slider_event(event)

                if not piano_roll_instance.app_running.is_set():
                    break

                current_time = self.get_current_playback_time()
                piano_roll_instance.draw(current_time)

                now = time.monotonic()
                if now - last_caption_update_time > 0.2:
                    fps = clock.get_fps()
                    pygame.display.set_caption(
                        f"Piano Roll - {fps:.1f} FPS - window {piano_roll_instance.window_seconds:.2f}s - scroll {piano_roll_instance.scroll_speed:.0f}"
                    )
                    last_caption_update_time = now

                clock.tick(0)
        except Exception as e:
            traceback.print_exc()
            self._queue_ui(dpg.set_value, "status_text", f"Piano Roll Error: {e}")
            self._queue_ui(self._message_error, "Piano Roll Error", str(e))
        finally:
            try:
                piano_roll_instance.cleanup()
            except Exception:
                traceback.print_exc()

    def _update_plot_series(self):
        x_values = list(range(100))
        dpg.set_value("nps_series", [x_values, list(self.nps_history)])
        dpg.set_value("cpu_series", [x_values, list(self.cpu_history)])
        current_max_nps = max(self.nps_history) if self.nps_history else 0
        if current_max_nps <= 100:
            nps_axis_top = 100
        else:
            nps_axis_top = math.ceil(current_max_nps * 1.15 / 50) * 50
        dpg.set_axis_limits("nps_y_axis", 0, nps_axis_top)
        dpg.set_axis_limits("cpu_y_axis", 0, 100)
        self._apply_graph_series_colors(
            self.nps_history[-1] if self.nps_history else 0,
            self.last_cpu_percent,
        )

    def update_cpu_graph(self):
        now = time.monotonic()
        if now - self.last_cpu_sample_time < 0.25:
            return

        self.last_cpu_sample_time = now
        cpu_percent = 0.0
        if self.process:
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
                for child in self.process.children(recursive=True):
                    try:
                        cpu_percent += child.cpu_percent(interval=None)
                    except psutil.NoSuchProcess:
                        continue
            except Exception:
                cpu_percent = 0.0
        cpu_percent = max(0.0, min(cpu_percent, 100.0))
        self.last_cpu_percent = cpu_percent
        self.cpu_history.append(cpu_percent)
        dpg.set_value("cpu_text", f"{cpu_percent:.1f}%")

    def update_gui_counters(self):
        now = time.monotonic()
        current_time = self.get_current_playback_time()
        self.controller.current_playback_time_for_threads = current_time

        if self.controller.playing and not self.controller.paused:
            if current_time < 0:
                current_time = 0.0
            if self.controller.parsed_midi and current_time > self.controller.total_song_duration:
                current_time = self.controller.total_song_duration
            if self.controller.parsed_midi:
                dpg.set_value(
                    "time_text",
                    f"{self.format_time(current_time)} / {self.format_time(self.controller.total_song_duration)}",
                )
            if not self.controller.is_seeking and not dpg.is_item_active("seek_slider"):
                dpg.set_value("seek_slider", current_time)

            if self.controller.parsed_midi:
                on_times = self.controller.parsed_midi.note_events_for_playback["on_time"]
                played_idx = bisect.bisect_left(on_times, current_time)
                dpg.set_value("note_count_value", f"{played_idx:,} / {self.controller.total_song_notes:,}")
                start_nps_time = max(0, current_time - 1.0)
                start_idx = bisect.bisect_left(on_times, start_nps_time)
                nps = played_idx - start_idx
                self.nps_history.append(nps)
                dpg.set_value("nps_text", self.format_nps(nps))
        elif self.controller.parsed_midi and not self.controller.playing:
            self.nps_history.append(0)
            dpg.set_value("nps_text", "0")

        current_max_nps = max(self.nps_history) if self.nps_history else 0
        graph_top_value = max(100, (math.ceil(current_max_nps * 1.15 / 50) * 50))
        dpg.set_value("nps_max_text", f"Max: {self.format_nps(graph_top_value)}")
        self._update_plot_series()

        if now - self.controller.last_lag_update_time > 0.5:
            delta_time = now - self.controller.last_lag_update_time
            self.controller.last_lag_update_time = now

            is_buffered = False
            if self.controller.active_midi_backend and hasattr(self.controller.active_midi_backend, "buffering_enabled"):
                is_buffered = self.controller.active_midi_backend.buffering_enabled

            if is_buffered:
                try:
                    buf_lvl = self.controller.active_midi_backend.get_buffer_level()
                    dpg.set_value("slowdown_text", f"Buffer: {buf_lvl:.1f}s")
                except Exception:
                    dpg.set_value("slowdown_text", "Buffer: N/A")
            else:
                if self.controller.playing and not self.controller.paused:
                    delta_lag = self.controller.current_lag - self.controller.last_lag_value
                    if delta_time > 0:
                        slowdown = delta_lag / delta_time
                        self.controller.slowdown_percentage = max(0, slowdown * 100)
                else:
                    self.controller.slowdown_percentage = 0.0

                self.controller.last_lag_value = self.controller.current_lag
                dpg.set_value("slowdown_text", f"Slowdown: {self.controller.slowdown_percentage:.1f}%")

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("Cleaning up resources...")
        if self.controller.playing:
            self.controller.playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(0.1)
        self._wait_for_audio_sweep(timeout=2.0)

        if self.piano_roll:
            self.piano_roll.app_running.clear()
            if self.piano_roll_thread and self.piano_roll_thread.is_alive():
                self.piano_roll_thread.join(0.2)

        self.controller.shutdown()

    def run(self):
        try:
            while dpg.is_dearpygui_running():
                self._process_ui_queue()
                self._poll_parser()
                self.update_cpu_graph()
                self.update_gui_counters()
                dpg.render_dearpygui_frame()
        finally:
            self.cleanup()
            dpg.destroy_context()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = DpgMidiPlayerApp()
    app.run()
