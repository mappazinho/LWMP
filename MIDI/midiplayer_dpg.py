#!/usr/bin/env python3

import atexit
import bisect
import ctypes
import heapq
import math
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from collections import deque

import dearpygui.dearpygui as dpg
import numpy as np
import psutil

script_dir = os.path.dirname(os.path.abspath(__file__))
runtime_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else script_dir
sys.path.append(script_dir)
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, "PARSER"))

from config import load_config, save_config, setup_omnimidi_preference
from midi_parser import GPU_NOTE_DTYPE, MidiParser
from player_controller import PlayerController

try:
    from piano import PianoRoll, SkinBrowser, _SKIN_ROOT, _list_available_skins, _resolve_skin_dir
    import pygame
except ImportError:
    print("piano package not found. Piano roll will be disabled.")
    PianoRoll = None
    _SKIN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "skin")
    def _list_available_skins(root):
        return []
    def _resolve_skin_dir(name, root):
        return None

DEBUG = False
AUDIO_MIN_NOTE_VELOCITY = 10

OmniMidiEngine = None

try:
    from midi_engine import OmniMidiEngine
    print("Loaded OmniMIDI engine.")
except ImportError:
    OmniMidiEngine = None
    print("No OmniMIDI engine found.")
except Exception as e:
    OmniMidiEngine = None
    print(f"Error importing OmniMidiEngine: {e}")

try:
    from midi_engine_cython import BassMidiEngine
    print("Loaded Cython BASSMIDI engine.")
except ImportError:
    BassMidiEngine = None
    print("No BASSMIDI engine found.")
except Exception as e:
    BassMidiEngine = None
    print(f"Error importing BassMidiEngine: {e}")

CONFIG = setup_omnimidi_preference(load_config())

_BUNDLED_FFMPEG_CANDIDATES = [
    os.path.join(script_dir, "ffmpeg.exe"),
    os.path.join(script_dir, "ffmpeg"),
    os.path.join(parent_dir, "ffmpeg.exe"),
    os.path.join(parent_dir, "ffmpeg"),
]


def _build_parser_result_payload(parser, use_disk_backing=False, result_queue=None):
    result_data = {
        "filename": parser.filename,
        "ticks_per_beat": parser.ticks_per_beat,
        "total_duration_sec": parser.total_duration_sec,
        "tempo_events": getattr(parser, "tempo_events", [(0.0, 120.0)]),
        "program_change_events": getattr(parser, "program_change_events", []),
        "pitch_bend_events": parser.pitch_bend_events,
        "control_change_events": getattr(parser, "control_change_events", []),
        "preferred_color_mode": getattr(parser, "preferred_color_mode", "track"),
    }
    if use_disk_backing:
        temp_dir = tempfile.mkdtemp(prefix="lwmp_parse_")
        gpu_path = os.path.join(temp_dir, "note_data_for_gpu.npy")
        playback_path = os.path.join(temp_dir, "note_events_for_playback.npy")
        if result_queue is not None:
            result_queue.put((
                "progress",
                {
                    "fraction": 0.995,
                    "overlay": "99.5%",
                    "detail": "Preparing low-memory result transfer...",
                },
            ))
        np.save(gpu_path, parser.note_data_for_gpu, allow_pickle=False)
        np.save(playback_path, parser.note_events_for_playback, allow_pickle=False)
        result_data["disk_backed_arrays"] = {
            "note_data_for_gpu": gpu_path,
            "note_events_for_playback": playback_path,
        }
        result_data["backing_temp_dir"] = temp_dir
    else:
        result_data["note_data_for_gpu"] = parser.note_data_for_gpu
        result_data["note_events_for_playback"] = parser.note_events_for_playback
    return result_data


def run_parser_process(filepath, result_queue, fallback_event_threshold=0):
    try:
        parser = MidiParser(filepath)
        total_events = parser.count_total_events()
        result_queue.put(("total_events", total_events))
        parser.parse(
            result_queue,
            total_events=total_events,
            fallback_event_threshold=int(fallback_event_threshold or 0),
        )
        use_disk_backing = bool(fallback_event_threshold and total_events > int(fallback_event_threshold))
        result_data = _build_parser_result_payload(
            parser,
            use_disk_backing=use_disk_backing,
            result_queue=result_queue,
        )
        result_queue.put(("success", result_data))
    except Exception as e:
        traceback.print_exc()
        result_queue.put(("error", str(e)))


from gui import (
    RenderMixin, PlaybackMixin, SoundfontMixin, LibraryMixin,
    SkinMixin, StartupMixin, GuiConfigMixin, TransportMixin,
    PianoRollMixin, NpsSpikesMixin,
)


class DpgMidiPlayerApp(
    RenderMixin,
    PlaybackMixin,
    SoundfontMixin,
    LibraryMixin,
    SkinMixin,
    StartupMixin,
    GuiConfigMixin,
    TransportMixin,
    PianoRollMixin,
    NpsSpikesMixin,
):
    # Module-level references for mixin access
    _save_config = staticmethod(save_config)
    _load_config = staticmethod(load_config)
    _script_dir = script_dir
    _runtime_dir = runtime_dir
    _DEBUG = DEBUG
    _AUDIO_MIN_NOTE_VELOCITY = AUDIO_MIN_NOTE_VELOCITY
    _SKIN_ROOT = _SKIN_ROOT
    _PianoRoll = PianoRoll
    _BassMidiEngine = BassMidiEngine
    _OmniMidiEngine = OmniMidiEngine
    _PlayerController = PlayerController
    _SkinBrowser = SkinBrowser
    _list_available_skins = staticmethod(_list_available_skins)
    _resolve_skin_dir = staticmethod(_resolve_skin_dir)
    _pygame = pygame

    def __init__(self):
        self._CONFIG = CONFIG
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
        self.render_thread = None
        self.render_start_time_monotonic = 0.0
        self.render_stage_timing = {}
        self._render_cancelled = threading.Event()
        self._ffmpeg_processes = []
        self._render_current_time = 0.0
        self._pending_confirm_yes = None
        self._pending_confirm_no = None
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
        self._manual_stop_requested = False
        self._cleaned_up = False
        self.startup_ready = False
        self.pending_midi_name = None
        self.has_bundled_omnimidi = os.path.exists(os.path.join(runtime_dir, "SYNTH.dll"))
        self.recommended_mode = "local" if self.has_bundled_omnimidi else "path"
        self.recommended_note_limit = 20_000_000
        self.last_cpu_sample_time = 0.0
        self.last_cpu_percent = 0.0
        self.pending_soundfont_callback = None
        self.loading_total_events = 0
        self.loading_started_at = 0.0
        self.pending_directory_callback = None
        self.library_file_labels = []
        self.library_file_map = {}
        self.last_library_click_label = None
        self.last_library_click_time = 0.0
        self.soundfont_file_labels = []
        self.soundfont_file_map = {}
        self.last_soundfont_click_label = None
        self.last_soundfont_click_time = 0.0
        self.render_spike_label_map = {}
        self._parse_normalize_thread = None

        self.all_backend_labels = {
            "bassmidi": "BASSMIDI (Buffered)",
            "path": "OmniMIDI",
            "local": "Custom Synth (Bundled SYNTH.dll)",
        }
        self.backend_labels = self._build_backend_labels()
        self.backend_values = {label: value for value, label in self.backend_labels.items()}

        self.screen_width, self.screen_height = self._get_screen_size()
        self.recommended_piano_roll_res = (1366, 768) if self.screen_height > 768 else (640, 360)
        self.available_resolutions = self._build_resolution_list()

        self._build_ui()
        self._refresh_soundfont_visibility()
        self._refresh_library_directory_ui()
        self.refresh_library_files()
        self._bind_theme()
        self._initialize_process_monitoring()
        self._prepare_recommendation_info()
        if CONFIG["gui"].get("startup_completed", False):
            self.startup_ready = True
            self.initialize_audio_backend()
        else:
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
            "pianoroll_bg": self._mix_color([10, 10, 16], seed, 0.06),
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
        if dpg.does_item_exist("nps_area_series"):
            dpg.bind_item_theme("nps_area_series", self._nps_series_theme_tag(nps_value))
        if dpg.does_item_exist("nps_series"):
            dpg.bind_item_theme("nps_series", self._nps_series_theme_tag(nps_value))
        if dpg.does_item_exist("cpu_area_series"):
            dpg.bind_item_theme("cpu_area_series", self._cpu_series_theme_tag(cpu_value))
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
            return 1366, 768

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














    def _open_directory_dialog(self, callback):
        self.pending_directory_callback = callback
        dpg.show_item("library_directory_dialog")

    def _on_directory_dialog_selected(self, sender, app_data):
        directory = None
        if isinstance(app_data, dict):
            directory = app_data.get("file_path_name")
            if not directory:
                current_path = app_data.get("current_path")
                current_filter = app_data.get("current_filter", "")
                if current_path:
                    directory = os.path.join(current_path, current_filter) if current_filter and current_filter != ".*" else current_path
        callback = self.pending_directory_callback
        self.pending_directory_callback = None
        dpg.configure_item("library_directory_dialog", show=False)
        if callback and directory:
            self._queue_ui(callback, directory)

    def _on_directory_dialog_cancel(self):
        self.pending_directory_callback = None
















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


    def _build_status_recommendation_text(self):
        recommended_mode_label = self.all_backend_labels[self.recommended_mode]
        res_w, res_h = self.recommended_piano_roll_res
        note_limit = self.recommended_note_limit / 1_000_000
        try:
            vm = psutil.virtual_memory()
            free_ram_gb = vm.available / (1024 ** 3)
        except Exception:
            free_ram_gb = 0.0
        cpu_threads = os.cpu_count() or 0
        return (
            f"Recommended Backend: {recommended_mode_label}\n"
            f"Recommended Note Capacity: {note_limit:.1f} Million notes (based on free RAM)\n"
            f"Recommended Piano Roll Resolution: {res_w} x {res_h}\n"
            f"System CPU Threads: {cpu_threads} | Free RAM: {free_ram_gb:.1f} GB"
        )

    def _prepare_recommendation_info(self):
        self.recommended_note_limit = self._detect_recommended_note_limit()
        if dpg.does_item_exist("status_info_text"):
            dpg.set_value("status_info_text", self._build_status_recommendation_text())







    def _build_audio_hint_text(self):
        lines = []
        lines.append(
            "Bundled custom synth DLL: detected" if self.has_bundled_omnimidi else "Bundled custom synth DLL: missing"
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












    def _build_ui(self):
        dpg.create_context()
        dpg.create_viewport(title="LWMP - v1.3.0", width=1040, height=800)

        default_mode = self._normalize_backend_mode(CONFIG["audio"].get("omnimidi_load_preference", "path"))
        default_combo_label = self._get_combo_label_for_mode(default_mode)

        with dpg.window(tag="main_window", label="LWMP", no_title_bar=True, no_scrollbar=True, no_scroll_with_mouse=True):
            with dpg.group(tag="app_shell"):
                with dpg.group(horizontal=True):
                    dpg.add_text("Lightweight MIDI Player", tag="title_text")
                    dpg.add_spacer(width=18)
                    dpg.add_button(
                        tag="library_button",
                        label="MIDI Library",
                        callback=self.show_library_window,
                        width=112,
                        height=30,
                    )
                    dpg.add_button(tag="load_button", label="Load MIDI", callback=self.on_load_unload, width=118, height=30)
                    dpg.add_button(
                        tag="play_button", label="Play", callback=self.toggle_play_pause, enabled=False, show=False, width=90, height=30
                    )
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
                    dpg.add_button(
                        tag="render_button",
                        label="Render Video",
                        callback=self.show_render_window,
                        enabled=False,
                        show=False,
                        width=118,
                        height=30,
                    )
                    dpg.add_button(
                        tag="skin_button",
                        label="Skins",
                        callback=self.show_skin_window,
                        width=80,
                        height=30,
                    )

                dpg.add_separator()

                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True):
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(init_width_or_weight=1.0)

                    with dpg.table_row():
                        with dpg.table_cell():
                            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column(init_width_or_weight=1.0)
                                dpg.add_table_column(init_width_or_weight=0.0)
                                with dpg.table_row():
                                    with dpg.table_cell():
                                        dpg.add_text("Now Playing", tag="now_playing_text", color=(223, 177, 103), wrap=470)
                                    with dpg.table_cell():
                                        dpg.add_button(
                                            tag="nps_spikes_button",
                                            label="NPS Spikes",
                                            callback=self.show_nps_spikes_window,
                                            width=100,
                                            height=28,
                                            show=False,
                                        )
                            with dpg.group(tag="status_line_group"):
                                dpg.add_text(
                                    "Choose a startup mode to initialize audio.",
                                    tag="status_text",
                                    wrap=470,
                                    color=(196, 198, 204),
                                )
                                with dpg.group(tag="parse_progress_group", show=False):
                                    dpg.add_text("Parsing progress", tag="parse_progress_title", color=(160, 166, 178))
                                    dpg.add_progress_bar(
                                        tag="parse_progress_bar",
                                        default_value=0.0,
                                        width=-1,
                                        overlay="0%",
                                    )
                                    dpg.add_text("", tag="parse_progress_text", wrap=470, color=(160, 166, 178))
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Position", color=(160, 166, 178))
                                    dpg.add_text("00:00 / 00:00", tag="time_text")
                                    dpg.add_spacer(width=14)
                                    dpg.add_text("Notes", color=(160, 166, 178))
                                    dpg.add_text("0 / 0", tag="note_count_value")
                                with dpg.group(horizontal=True):
                                    dpg.add_text("BPM", color=(160, 166, 178))
                                    dpg.add_text("0", tag="bpm_value")
                                    dpg.add_spacer(width=14)
                                    dpg.add_text("Poly", color=(160, 166, 178))
                                    dpg.add_text("0", tag="polyphony_value")
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
                                dpg.add_spacer(height=6)
                                dpg.add_text("", tag="status_info_text", wrap=470, color=(160, 166, 178))

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
                            with dpg.group(tag="soundfont_group"):
                                dpg.add_text("Current SoundFont", color=(160, 166, 178))
                                dpg.add_text("No SoundFont selected", tag="soundfont_text", wrap=470)
                                dpg.add_button(label="Change SoundFont", callback=self.show_soundfont_library_window, width=-1, height=28)
                            dpg.add_text("Synth Controls", color=(160, 166, 178))
                            dpg.add_slider_float(
                                label="",
                                tag="volume_slider",
                                min_value=0.0,
                                max_value=1.0,
                                default_value=float(CONFIG["audio"].get("volume", 0.5)),
                                width=-1,
                                format="Volume: %.2f",
                                callback=self.on_volume_change,
                            )
                            dpg.add_slider_int(
                                label="",
                                tag="voices_slider",
                                min_value=1,
                                max_value=2000,
                                default_value=int(CONFIG["audio"].get("voices", 512)),
                                width=-1,
                                format="Voices: %d",
                                callback=self.on_voice_limit_change,
                            )
                            dpg.add_slider_float(
                                label="",
                                tag="speed_slider",
                                min_value=0.25,
                                max_value=2.0,
                                default_value=float(CONFIG["audio"].get("speed", 1.0)),
                                width=-1,
                                format="Speed: %.2fx",
                                callback=self.on_speed_change,
                            )

                dpg.add_separator()
                with dpg.group(tag="performance_section"):
                    dpg.add_text("Performance", color=(223, 177, 103))
                    dpg.add_checkbox(
                        tag="pianoroll_stats_overlay_checkbox",
                        label="Show piano roll stats overlay",
                        default_value=bool(self._gui_cfg().get("show_pianoroll_stats_overlay", False)),
                        callback=self.on_pianoroll_stats_overlay_toggle,
                    )
                    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True):
                        dpg.add_table_column(init_width_or_weight=1.0)
                        dpg.add_table_column(init_width_or_weight=1.0)
                        dpg.add_table_column(init_width_or_weight=0.9)
                        dpg.add_table_column(init_width_or_weight=1.2)
                        with dpg.table_row():
                            with dpg.table_cell():
                                with dpg.group(horizontal=True, tag="nps_primary_group"):
                                    dpg.add_text("NPS", color=(160, 166, 178))
                                    dpg.add_text("0", tag="nps_text")
                                with dpg.group(horizontal=True, tag="cpu_overlay_group", show=False):
                                    dpg.add_text("CPU", color=(160, 166, 178))
                                    dpg.add_text("0.0%", tag="cpu_text_overlay")
                            with dpg.table_cell():
                                with dpg.group(tag="nps_max_primary_group"):
                                    dpg.add_text("Max: 0", tag="nps_max_text", color=(196, 198, 204))
                                with dpg.group(tag="runtime_overlay_group", show=False):
                                    dpg.add_text("Runtime", color=(160, 166, 178))
                                    dpg.add_text("Slowdown: 0.0%", tag="slowdown_text_overlay")
                                    dpg.add_progress_bar(
                                        tag="buffer_progress_overlay",
                                        default_value=0.0,
                                        width=-1,
                                        overlay="Buffer: 0.0s / 60.0s",
                                    )
                                    dpg.add_progress_bar(
                                        tag="recovery_buffer_progress_overlay",
                                        default_value=0.0,
                                        width=-1,
                                        overlay="Recovery: 0.0s / 4.0s",
                                    )
                            with dpg.table_cell():
                                with dpg.group(horizontal=True, tag="cpu_primary_group"):
                                    dpg.add_text("CPU", color=(160, 166, 178))
                                    dpg.add_text("0.0%", tag="cpu_text")
                            with dpg.table_cell():
                                with dpg.group(tag="runtime_primary_group"):
                                    dpg.add_text("Runtime", color=(160, 166, 178))
                                    dpg.add_text("Slowdown: 0.0%", tag="slowdown_text")
                                    dpg.add_progress_bar(
                                        tag="buffer_progress",
                                        default_value=0.0,
                                        width=-1,
                                        overlay="Buffer: 0.0s / 60.0s",
                                    )
                                    dpg.add_progress_bar(
                                        tag="recovery_buffer_progress",
                                        default_value=0.0,
                                        width=-1,
                                        overlay="Recovery: 0.0s / 4.0s",
                                    )

                    plot_x = list(range(100))
                    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True):
                        dpg.add_table_column(init_width_or_weight=1.0)
                        dpg.add_table_column(init_width_or_weight=1.0)
                        with dpg.table_row():
                            with dpg.table_cell(tag="nps_graph_cell"):
                                with dpg.group(tag="nps_graph_group"):
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
                                            dpg.add_area_series(plot_x, list(self.nps_history), tag="nps_area_series")
                                            dpg.add_line_series(plot_x, list(self.nps_history), tag="nps_series")

                            with dpg.table_cell(tag="cpu_graph_cell"):
                                with dpg.group(tag="cpu_graph_group"):
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
                                            dpg.add_area_series(plot_x, list(self.cpu_history), tag="cpu_area_series")
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
            dpg.add_progress_bar(tag="loading_progress", default_value=0.0, width=-1, overlay="0%")

        with dpg.file_dialog(
            tag="midi_file_dialog",
            show=False,
            modal=True,
            directory_selector=False,
            width=760,
            height=460,
            callback=self._on_midi_dialog_selected,
        ):
            dpg.add_file_extension(".mid", color=(120, 200, 255, 255))
            dpg.add_file_extension(".*")

        with dpg.file_dialog(
            tag="soundfont_file_dialog",
            show=False,
            modal=True,
            directory_selector=False,
            width=760,
            height=460,
            callback=self._on_soundfont_dialog_selected,
            cancel_callback=lambda: self._on_soundfont_dialog_cancel(),
        ):
            dpg.add_file_extension(".sf2", color=(120, 200, 255, 255))
            dpg.add_file_extension(".sfz", color=(120, 200, 255, 255))
            dpg.add_file_extension(".*")

        with dpg.file_dialog(
            tag="library_directory_dialog",
            show=False,
            modal=True,
            directory_selector=True,
            width=760,
            height=460,
            callback=self._on_directory_dialog_selected,
            cancel_callback=lambda: self._on_directory_dialog_cancel(),
        ):
            dpg.add_file_extension(".*")

        with dpg.window(
            tag="message_window",
            label="Message",
            modal=True,
            show=False,
            no_resize=True,
            no_collapse=True,
            width=460,
            height=180,
        ):
            dpg.add_text("", tag="message_title", color=(223, 177, 103))
            dpg.add_spacer(height=8)
            dpg.add_text("", tag="message_body", wrap=420)
            dpg.add_spacer(height=12)
            dpg.add_button(
                label="OK",
                width=100,
                height=30,
                callback=lambda: dpg.configure_item("message_window", show=False),
            )

        with dpg.window(
            tag="confirm_window",
            label="Confirm",
            modal=True,
            show=False,
            no_resize=True,
            no_collapse=True,
            width=460,
            height=180,
        ):
            dpg.add_text("", tag="confirm_title", color=(223, 177, 103))
            dpg.add_spacer(height=8)
            dpg.add_text("", tag="confirm_body", wrap=420)
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Yes",
                    tag="confirm_yes_button",
                    width=100,
                    height=30,
                    callback=self._on_confirm_yes,
                )
                dpg.add_button(
                    label="No",
                    tag="confirm_no_button",
                    width=100,
                    height=30,
                    callback=self._on_confirm_no,
                )

        with dpg.window(
            tag="library_window",
            label="MIDI Library",
            modal=False,
            show=False,
            no_collapse=True,
            width=720,
            height=520,
        ):
            dpg.add_text("Library Folders", color=(223, 177, 103))
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="library_path_input",
                    hint="Add a MIDI folder path",
                    width=-70,
                    on_enter=True,
                    callback=self.add_library_directory_from_input,
                )
                dpg.add_button(label="Add", width=56, height=28, callback=self.add_library_directory_from_input)
            dpg.add_combo(
                items=["No folders configured"],
                default_value="No folders configured",
                tag="library_directory_combo",
                width=-1,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Remove Folder", width=120, height=28, callback=self.remove_selected_library_directory)
                dpg.add_button(label="Refresh Files", width=110, height=28, callback=self.refresh_library_files)
                dpg.add_text("Files: 0", tag="library_count_text", color=(160, 166, 178))
            dpg.add_separator()
            dpg.add_text("MIDI Files", color=(223, 177, 103))
            dpg.add_input_text(
                tag="library_search_input",
                hint="Search MIDI library",
                width=-1,
                callback=self.refresh_library_files,
            )
            dpg.add_listbox(
                tag="library_file_list",
                items=["No MIDI files found"],
                width=-1,
                num_items=14,
                callback=self.on_library_file_selected,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Selected", width=140, height=30, callback=self.load_selected_library_file)

        with dpg.window(
            tag="soundfont_library_window",
            label="SoundFont Library",
            modal=False,
            show=False,
            no_collapse=True,
            width=720,
            height=520,
        ):
            dpg.add_text("Library Folders", color=(223, 177, 103))
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="soundfont_path_input",
                    hint="Add a SoundFont folder path",
                    width=-70,
                    on_enter=True,
                    callback=self.add_soundfont_directory_from_input,
                )
                dpg.add_button(label="Add", width=56, height=28, callback=self.add_soundfont_directory_from_input)
            dpg.add_combo(
                items=["No folders configured"],
                default_value="No folders configured",
                tag="soundfont_directory_combo",
                width=-1,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Remove Folder", width=120, height=28, callback=self.remove_selected_soundfont_directory)
                dpg.add_button(label="Refresh Files", width=110, height=28, callback=self.refresh_soundfont_files)
                dpg.add_text("Files: 0", tag="soundfont_count_text", color=(160, 166, 178))
            dpg.add_separator()
            dpg.add_text("SoundFonts", color=(223, 177, 103))
            dpg.add_input_text(
                tag="soundfont_search_input",
                hint="Search SoundFont library",
                width=-1,
                callback=self.refresh_soundfont_files,
            )
            dpg.add_listbox(
                tag="soundfont_file_list",
                items=["No SoundFonts found"],
                width=-1,
                num_items=14,
                callback=self.on_soundfont_file_selected,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Selected", width=140, height=30, callback=self.load_selected_soundfont_file)

        with dpg.window(
            tag="nps_spikes_window",
            label="NPS Spikes",
            modal=False,
            show=False,
            no_collapse=True,
            width=460,
            height=320,
        ):
            dpg.add_text("Top Parsed NPS Spikes", color=(223, 177, 103))
            dpg.add_spacer(height=6)
            dpg.add_text("", tag="nps_spikes_summary", color=(196, 198, 204), wrap=420)
            dpg.add_spacer(height=8)
            dpg.add_text("", tag="nps_spikes_text", color=(160, 166, 178), wrap=420)

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
            tag="render_window",
            label="Render Video",
            modal=True,
            show=False,
            no_resize=True,
            no_collapse=True,
            width=520,
            height=520,
        ):
            dpg.add_text("Video Output", color=(223, 177, 103))
            dpg.add_text("FFmpeg path (PATH/TO/ffmpeg.exe)", color=(160, 166, 178))
            dpg.add_input_text(
                tag="render_ffmpeg_path",
                default_value=str(self._render_cfg().get("ffmpeg_path", "ffmpeg")),
                width=-1,
            )
            dpg.add_text("Output file", color=(160, 166, 178))
            dpg.add_input_text(
                tag="render_output_path",
                default_value=str(self._render_cfg().get("output_path", "")),
                width=-1,
            )
            dpg.add_spacer(height=4)
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Resolution", color=(160, 166, 178))
                        dpg.add_combo(
                            tag="render_resolution",
                            items=[f"{w} x {h}" for w, h in self.available_resolutions],
                            default_value=str(self._render_cfg().get("resolution", f"{self.recommended_piano_roll_res[0]} x {self.recommended_piano_roll_res[1]}")),
                            width=-1,
                        )
                    with dpg.table_cell():
                        dpg.add_text("Codec", color=(160, 166, 178))
                        dpg.add_combo(
                            tag="render_codec",
                            items=["H.264", "H.265", "MPEG-4"],
                            default_value=str(self._render_cfg().get("codec", "H.264")),
                            width=-1,
                        )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Framerate", color=(160, 166, 178))
                        dpg.add_input_int(
                            tag="render_framerate",
                            default_value=int(self._render_cfg().get("framerate", 60)),
                            step=1,
                            width=-1,
                        )
                    with dpg.table_cell():
                        dpg.add_text("Bitrate", color=(160, 166, 178))
                        dpg.add_input_text(
                            tag="render_bitrate",
                            default_value=str(self._render_cfg().get("bitrate", "20M")),
                            width=-1,
                        )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Audio Bitrate", color=(160, 166, 178))
                        dpg.add_combo(
                            tag="render_audio_bitrate",
                            items=["128k", "192k", "256k", "320k"],
                            default_value=str(self._render_cfg().get("audio_bitrate", "320k")),
                            width=-1,
                        )
                    with dpg.table_cell():
                        dpg.add_spacer(height=1)
            dpg.add_spacer(height=6)
            dpg.add_checkbox(
                tag="render_audio_checkbox",
                label="Render audio and mux into final video",
                default_value=bool(self._render_cfg().get("render_audio", True)),
            )
            dpg.add_checkbox(
                tag="render_audio_limiter_checkbox",
                label="Apply audio limiter (prevent clipping)",
                default_value=bool(self._render_cfg().get("audio_limiter", False)),
            )
            dpg.add_checkbox(
                tag="render_stats_overlay_checkbox",
                label="Draw notes, NPS, BPM, polyphony, and time in video",
                default_value=bool(self._render_cfg().get("show_stats_overlay", False)),
                callback=self._toggle_render_stats_controls,
            )
            with dpg.group(tag="render_stats_mod_group", show=bool(self._render_cfg().get("show_stats_overlay", False))):
                dpg.add_checkbox(
                    tag="render_stats_mod_checkbox",
                    label="Stats Modification",
                    default_value=bool(self._render_cfg().get("enable_stats_modification", False)),
                    callback=self._toggle_render_stats_mod_controls,
                )
                dpg.add_text("Render-only stat edits", color=(160, 166, 178))
                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    with dpg.table_row():
                        with dpg.table_cell():
                            dpg.add_text("Stats Multiplier", color=(160, 166, 178))
                            dpg.add_input_text(
                                tag="render_stats_multiplier",
                                default_value=str(self._render_cfg().get("stats_multiplier", 1.0)),
                                width=-1,
                            )
                        with dpg.table_cell():
                            dpg.add_text("NPS Spike", color=(160, 166, 178))
                            dpg.add_combo(
                                tag="render_spike_select",
                                items=["None"],
                                default_value=str(self._render_cfg().get("spike_selection", "None") or "None"),
                                width=-1,
                            )
                    with dpg.table_row():
                        with dpg.table_cell():
                            dpg.add_text("Spike Intensity", color=(160, 166, 178))
                            dpg.add_input_text(
                                tag="render_spike_intensity",
                                default_value=str(self._render_cfg().get("spike_intensity", 1.0)),
                                width=-1,
                            )
                        with dpg.table_cell():
                            dpg.add_spacer(height=1)
            dpg.add_checkbox(
                tag="render_watermark_checkbox",
                label="Draw 'Rendered with LWMP' watermark",
                default_value=bool(self._render_cfg().get("show_watermark", True)),
            )
            dpg.add_spacer(height=8)
            dpg.add_spacer(height=8)
            dpg.add_progress_bar(tag="render_progress_bar", default_value=0.0, width=-1, overlay="Idle")
            dpg.add_text("", tag="render_status_text", wrap=480, color=(196, 198, 204))
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(tag="start_render_button", label="Start Render", callback=self.start_render_video, width=150, height=32)
                dpg.add_checkbox(tag='render_live_preview_checkbox', label='Live Preview', default_value=False)

        with dpg.window(
            tag="skin_window",
            label="Skin Browser",
            modal=False,
            show=False,
            no_collapse=True,
            width=700,
            height=500,
        ):
            with dpg.group(horizontal=True):
                # Left panel: skin list
                with dpg.child_window(width=200, border=True):
                    dpg.add_text("Available Skins", color=(223, 177, 103))
                    dpg.add_listbox(
                        tag="skin_list",
                        items=[],
                        width=-1,
                        num_items=18,
                        callback=self._on_skin_selected,
                    )
                # Right panel: preview + apply
                with dpg.child_window(width=-1, border=True):
                    dpg.add_text("Preview", color=(223, 177, 103))
                    dpg.add_drawlist(
                        tag="skin_preview_drawlist",
                        width=460,
                        height=340,
                    )
                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            tag="apply_skin_button",
                            label="Apply Skin",
                            callback=self._apply_selected_skin,
                            width=120,
                            height=30,
                        )
                        dpg.add_text(
                            tag="skin_status_text",
                            default_value="",
                            color=(160, 166, 178),
                        )

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
            dpg.add_text("Theme Seed", color=(160, 166, 178))
            dpg.add_color_edit(label="", tag="custom_theme_seed", no_alpha=True, width=-1)
            dpg.add_button(label="Generate Palette From Theme Color", callback=self.apply_theme_seed, width=-1, height=32)

            dpg.add_separator()
            dpg.add_text("Colors", color=(223, 177, 103))
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Window BG", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_window_bg", no_alpha=True, width=-1)
                        dpg.add_text("Piano Roll BG", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_pianoroll_bg", no_alpha=True, width=-1)
                        dpg.add_text("Child BG", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_child_bg", no_alpha=True, width=-1)
                        dpg.add_text("Frame BG", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_frame_bg", no_alpha=True, width=-1)
                        dpg.add_text("Frame Hover", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_frame_bg_hovered", no_alpha=True, width=-1)
                        dpg.add_text("Frame Active", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_frame_bg_active", no_alpha=True, width=-1)
                        dpg.add_text("Button", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_button", no_alpha=True, width=-1)
                    with dpg.table_cell():
                        dpg.add_text("Button Hover", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_button_hovered", no_alpha=True, width=-1)
                        dpg.add_text("Button Active", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_button_active", no_alpha=True, width=-1)
                        dpg.add_text("Accent Text", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_accent_text", no_alpha=True, width=-1)
                        dpg.add_text("Muted Text", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_muted_text", no_alpha=True, width=-1)
                        dpg.add_text("Body Text", color=(160, 166, 178))
                        dpg.add_color_edit(label="", tag="custom_body_text", no_alpha=True, width=-1)

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.save_customize_settings, width=140, height=32)

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
                label="Use Defaults (OmniMIDI PATH, 720p)",
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


    def _initialize_process_monitoring(self):
        try:
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)
        except Exception as e:
            print(f"Failed to initialize psutil: {e}")
            self.process = None

    def _show_message_window(self, title, text):
        if dpg.does_item_exist("message_title"):
            dpg.set_value("message_title", title)
        if dpg.does_item_exist("message_body"):
            dpg.set_value("message_body", text)
        if dpg.does_item_exist("message_window"):
            dpg.configure_item("message_window", label=title, show=True)

    def _show_confirm_dialog(self, title, text, on_yes, on_no=None):
        self._pending_confirm_yes = on_yes
        self._pending_confirm_no = on_no
        if dpg.does_item_exist("confirm_title"):
            dpg.set_value("confirm_title", title)
        if dpg.does_item_exist("confirm_body"):
            dpg.set_value("confirm_body", text)
        if dpg.does_item_exist("confirm_window"):
            dpg.configure_item("confirm_window", label=title, show=True)
            self._center_modal("confirm_window", 460, 180)
            dpg.focus_item("confirm_window")

    def _on_confirm_yes(self, sender=None, app_data=None):
        dpg.configure_item("confirm_window", show=False)
        cb = self._pending_confirm_yes
        self._pending_confirm_yes = None
        self._pending_confirm_no = None
        if cb:
            cb()

    def _on_confirm_no(self, sender=None, app_data=None):
        dpg.configure_item("confirm_window", show=False)
        cb = self._pending_confirm_no
        self._pending_confirm_yes = None
        self._pending_confirm_no = None
        if cb:
            cb()

    def _set_parse_progress_visible(self, visible):
        if dpg.does_item_exist("parse_progress_group"):
            dpg.configure_item("parse_progress_group", show=bool(visible))

    def _update_parse_progress(self, fraction, overlay, detail, title="Parsing progress"):
        if dpg.does_item_exist("parse_progress_title"):
            dpg.set_value("parse_progress_title", title)
        if dpg.does_item_exist("parse_progress_bar"):
            dpg.set_value("parse_progress_bar", max(0.0, min(float(fraction), 1.0)))
            dpg.configure_item("parse_progress_bar", overlay=str(overlay))
        if dpg.does_item_exist("parse_progress_text"):
            dpg.set_value("parse_progress_text", detail)

    def _reset_parse_progress(self):
        self.loading_visible = False
        self.loading_total_events = 0
        self._update_parse_progress(0.0, "0%", "", title="Parsing progress")
        self._set_parse_progress_visible(False)

    def _message_info(self, title, text):
        self._show_message_window(title, text)

    def _message_warning(self, title, text):
        self._show_message_window(title, text)

    def _message_error(self, title, text):
        self._show_message_window(title, text)

    def _open_midi_file_dialog(self):
        dpg.show_item("midi_file_dialog")



    def _extract_dialog_path(self, app_data):
        if isinstance(app_data, dict):
            return app_data.get("file_path_name")
        return None

    def _on_midi_dialog_selected(self, sender, app_data):
        filepath = self._extract_dialog_path(app_data)
        if filepath:
            dpg.configure_item("midi_file_dialog", show=False)
            self._queue_ui(self._begin_load_file, filepath)







    def _center_modal(self, tag, width, height):
        viewport_w = dpg.get_viewport_client_width() or 900
        viewport_h = dpg.get_viewport_client_height() or 700
        dpg.set_item_pos(
            tag,
            [
                max(20, int((viewport_w - width) * 0.5)),
                max(20, int((viewport_h - height) * 0.5)),
            ],
        )


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

    def format_bpm(self, bpm):
        bpm = max(0.0, float(bpm))
        text = f"{bpm:.2f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def _build_midi_info_text(self):
        return f"Max NPS: {self.format_nps(self.controller.max_nps)}"

    def set_status(self, text):
        dpg.set_value("status_text", text)
        dpg.set_value("backend_hint_text", self._build_audio_hint_text())









    def on_load_unload(self):
        if self.controller.parsed_midi is None:
            self.load_file()
        else:
            self.unload_file()

    def _prepare_for_new_midi_load(self):
        if self.piano_roll and self.piano_roll.app_running.is_set():
            self.was_piano_roll_open_before_unload = True
            self.piano_roll.app_running.clear()
            if self.piano_roll_thread and self.piano_roll_thread.is_alive():
                self.piano_roll_thread.join(0.2)
            self.piano_roll = None
            self.piano_roll_thread = None
        else:
            self.was_piano_roll_open_before_unload = False

    def load_file(self, filepath=None):
        if not self.startup_ready:
            self._message_warning("Startup Required", "Choose a startup audio mode before loading a MIDI.")
            return

        if self.controller.playing:
            self.controller.playing = False
            self.controller.paused = False
            dpg.configure_item("play_button", label="Play")

        self._prepare_for_new_midi_load()
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

        if self._parse_normalize_thread and self._parse_normalize_thread.is_alive():
            self._message_warning("Busy", "Still normalizing previous parse. Please wait.")
            return

        if filepath:
            self._begin_load_file(filepath)
        else:
            self._open_midi_file_dialog()

    def _begin_load_file(self, filepath):
        self._update_now_playing_header(os.path.basename(filepath))
        dpg.set_value("status_text", f"Parsing {os.path.basename(filepath)}...")
        dpg.configure_item("load_button", enabled=False)
        self.loading_total_events = 0
        self.loading_started_at = time.monotonic()
        self._set_parse_progress_visible(True)
        self._update_parse_progress(0.0, "Counting...", "Counting events...")
        self.loading_visible = True
        self._queue_ui(self._start_parse_job, filepath)

    def _start_parse_job(self, filepath):
        if not self.loading_visible:
            return

        try:
            self.controller.start_parse_job(
                filepath,
                multiprocessing.Queue,
                multiprocessing.Process,
                run_parser_process,
                fallback_event_threshold=self.recommended_note_limit * 2,
            )
        except Exception as e:
            self.controller.clear_parse_job()
            self._reset_parse_progress()
            self._message_error("Error", f"Failed to start parser: {e}")
            self.pending_midi_name = None
            self._update_now_playing_header()
            dpg.set_value("status_text", "Failed to start parser.")
            dpg.configure_item("load_button", enabled=True)

    def _poll_parser(self):
        if self._parse_normalize_thread and not self._parse_normalize_thread.is_alive():
            self._parse_normalize_thread = None
        if self.loading_visible and self.loading_total_events <= 0:
            pulse = (time.monotonic() - self.loading_started_at) % 1.0
            pulse_value = 0.15 + (0.7 * pulse)
            self._update_parse_progress(pulse_value, "Counting...", "Counting events...")
        try:
            for status, payload in self.controller.poll_parser_messages():
                if status == "total_events":
                    self.loading_total_events = int(payload or 0)
                    if self.loading_visible:
                        self._update_parse_progress(0.0, "0.0%", f"Found {int(payload or 0):,} events. Parsing...")
                elif status == "progress":
                    if self.loading_visible:
                        progress_payload = payload
                        if isinstance(progress_payload, dict):
                            if "fraction" in progress_payload:
                                self._update_parse_progress(
                                    progress_payload.get("fraction", 0.0),
                                    progress_payload.get("overlay", "Working..."),
                                    progress_payload.get("detail", "Working..."),
                                )
                            else:
                                current = progress_payload.get("current", 0)
                                total = progress_payload.get("total", 1)
                                eta = progress_payload.get("eta", 0)
                                fraction = (current / total) if total > 0 else 0.0
                                eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "Calculating..."
                                self._update_parse_progress(
                                    fraction,
                                    f"{fraction * 100:.1f}%",
                                    f"Parsing... {current:,} / {total:,} events ({eta_str})",
                                )
                        else:
                            self._update_parse_progress(0.0, "Working...", str(progress_payload))
                elif status == "success":
                    self._update_parse_progress(0.95, "Normalizing...", "Normalizing parsed data...")
                    self._parse_normalize_thread = threading.Thread(
                        target=self._normalize_parse_in_thread,
                        args=(payload,),
                        daemon=True,
                    )
                    self._parse_normalize_thread.start()
                    return
                else:
                    self._reset_parse_progress()
                    self.controller.clear_parse_job()
                    self._message_error("Parse Error", f"Could not load MIDI file: {payload}")
                    self.pending_midi_name = None
                    self._update_now_playing_header()
                    dpg.set_value("status_text", "Failed to load file.")
                    dpg.configure_item("load_button", enabled=True)
                    return
        except Exception as e:
            self._reset_parse_progress()
            self.controller.clear_parse_job()
            self._message_error("Error", f"Error checking parser status: {e}")
            self.pending_midi_name = None
            self._update_now_playing_header()
            dpg.set_value("status_text", "Error during parsing.")
            dpg.configure_item("load_button", enabled=True)

    def _normalize_parse_in_thread(self, payload):
        try:
            self.controller.normalize_parsed_payload(
                payload, start_padding=3.0, end_padding=3.0
            )
            self.controller.clear_parse_job()
            self._queue_ui(self._on_parse_normalized)
        except Exception as e:
            self._queue_ui(self._on_parse_normalize_error, str(e))

    def _on_parse_normalized(self):
        self._reset_parse_progress()
        self._update_now_playing_header()
        dpg.set_value("status_text", "Ready to play.")
        dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
        dpg.set_value("note_count_value", f"0 / {self.controller.total_song_notes:,}")
        initial_bpm = 120.0
        if getattr(self.controller.parsed_midi, "tempo_bpms", None) is not None and len(self.controller.parsed_midi.tempo_bpms) > 0:
            initial_bpm = float(self.controller.parsed_midi.tempo_bpms[0])
        dpg.set_value("bpm_value", self.format_bpm(initial_bpm))
        dpg.set_value("polyphony_value", "0")
        self._set_parse_progress_visible(True)
        self._update_parse_progress(
            1.0,
            "Parsed",
            self._build_midi_info_text(),
            title="MIDI Info",
        )
        self._refresh_nps_spikes_window()
        dpg.set_value("seek_slider", 0.0)
        dpg.configure_item("seek_slider", enabled=True, max_value=float(self.controller.total_song_duration))
        dpg.configure_item("play_button", label="Play")
        self._refresh_transport_button_state()
        dpg.configure_item("load_button", enabled=True, label="Unload MIDI")
        self.reset_graph_history()

        if self.was_piano_roll_open_before_unload and self.last_piano_roll_res and PianoRoll:
            self.launch_piano_roll(*self.last_piano_roll_res)
        self.was_piano_roll_open_before_unload = False

    def _on_parse_normalize_error(self, error_str):
        self._reset_parse_progress()
        self.controller.clear_parse_job()
        self._message_error("Parse Error", f"Could not load MIDI file: {error_str}")
        self.pending_midi_name = None
        self._update_now_playing_header()
        dpg.set_value("status_text", "Failed to load file.")
        dpg.configure_item("load_button", enabled=True)



    def unload_file(self):
        if self.controller.playing:
            self.controller.playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(0.1)

        self.controller.unload_midi()
        self._reset_parse_progress()
        self.reset_playback_state()
        self.pending_midi_name = None
        self._update_now_playing_header()
        dpg.set_value("status_text", "No file loaded.")
        dpg.set_value("time_text", "00:00 / 00:00")
        dpg.set_value("note_count_value", "0 / 0")
        dpg.set_value("bpm_value", "0")
        dpg.set_value("polyphony_value", "0")
        self._refresh_nps_spikes_window()
        if dpg.does_item_exist("nps_spikes_window"):
            dpg.configure_item("nps_spikes_window", show=False)
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
        dpg.set_value("bpm_value", "0")
        dpg.set_value("polyphony_value", "0")
        dpg.set_value("cpu_text", "0.0%")
        dpg.set_value("cpu_text_overlay", "0.0%")
        dpg.set_value("slowdown_text", "Slowdown: 0.0%")
        dpg.set_value("slowdown_text_overlay", "Slowdown: 0.0%")
        dpg.set_value("buffer_progress", 0.0)
        dpg.configure_item("buffer_progress", overlay="Buffer: 0.0s / 60.0s")
        dpg.set_value("buffer_progress_overlay", 0.0)
        dpg.configure_item("buffer_progress_overlay", overlay="Buffer: 0.0s / 60.0s")
        dpg.set_value("recovery_buffer_progress", 0.0)
        dpg.configure_item("recovery_buffer_progress", overlay="Recovery: 0.0s / 4.0s")
        dpg.set_value("recovery_buffer_progress_overlay", 0.0)
        dpg.configure_item("recovery_buffer_progress_overlay", overlay="Recovery: 0.0s / 4.0s")



    def panic_all_notes_off(self):
        if self.controller.active_midi_backend is None:
            return
        try:
            self.controller.panic_all_notes_off()
        except Exception as e:
            print(f"Error during MIDI backend panic: {e}")

    def set_pitch_bend_range(self, semitones=12):
        self.controller.set_pitch_bend_range(semitones=semitones)

















    def _preferred_drawtext_font(self):
        font_candidates = [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            r"/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        for candidate in font_candidates:
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _escape_drawtext_value(self, value):
        escaped = str(value).replace("\\", "/")
        escaped = escaped.replace(":", r"\:")
        escaped = escaped.replace("'", r"\'")
        escaped = escaped.replace(",", r"\,")
        return escaped












    def _update_plot_series(self):
        x_values = list(range(100))
        dpg.set_value("nps_area_series", [x_values, list(self.nps_history)])
        dpg.set_value("nps_series", [x_values, list(self.nps_history)])
        dpg.set_value("cpu_area_series", [x_values, list(self.cpu_history)])
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
        dpg.set_value("cpu_text_overlay", f"{cpu_percent:.1f}%")

    def update_gui_counters(self):
        now = time.monotonic()
        current_time = self.get_current_playback_time()
        if self.controller.total_song_duration > 0:
            current_time = min(current_time, self.controller.total_song_duration)
        self.controller.current_playback_time_for_threads = current_time
        parsed_midi = self.controller.parsed_midi
        played_idx = 0
        nps = 0
        bpm_value = 0.0
        polyphony = 0

        if parsed_midi:
            on_times = parsed_midi.note_events_for_playback["on_time"]
            played_idx = bisect.bisect_left(on_times, max(0.0, current_time))
            off_times = getattr(parsed_midi, "sorted_off_times", None)
            if off_times is not None and len(off_times) > 0:
                ended_idx = bisect.bisect_right(off_times, max(0.0, current_time))
                polyphony = max(0, played_idx - ended_idx)
            tempo_times = getattr(parsed_midi, "tempo_times", None)
            tempo_bpms = getattr(parsed_midi, "tempo_bpms", None)
            if tempo_times is not None and tempo_bpms is not None and len(tempo_times) > 0:
                tempo_idx = bisect.bisect_right(tempo_times, max(0.0, current_time)) - 1
                if tempo_idx < 0:
                    tempo_idx = 0
                bpm_value = float(tempo_bpms[tempo_idx])
            else:
                bpm_value = 120.0

        if self.controller.playing and not self.controller.paused:
            if current_time < 0:
                current_time = 0.0
            if parsed_midi and current_time > self.controller.total_song_duration:
                current_time = self.controller.total_song_duration
            if parsed_midi:
                dpg.set_value(
                    "time_text",
                    f"{self.format_time(current_time)} / {self.format_time(self.controller.total_song_duration)}",
                )
            if not self.controller.is_seeking and not dpg.is_item_active("seek_slider"):
                dpg.set_value("seek_slider", current_time)

            if parsed_midi:
                on_times = parsed_midi.note_events_for_playback["on_time"]
                played_idx = bisect.bisect_left(on_times, current_time)
                dpg.set_value("note_count_value", f"{played_idx:,} / {self.controller.total_song_notes:,}")
                start_nps_time = max(0, current_time - 1.0)
                start_idx = bisect.bisect_left(on_times, start_nps_time)
                nps = played_idx - start_idx
                self.nps_history.append(nps)
                dpg.set_value("nps_text", self.format_nps(nps))
        elif parsed_midi and not self.controller.playing:
            dpg.set_value("note_count_value", f"{played_idx:,} / {self.controller.total_song_notes:,}")
            dpg.set_value(
                "time_text",
                f"{self.format_time(current_time)} / {self.format_time(self.controller.total_song_duration)}",
            )
            if (
                not self._manual_stop_requested
                and self.controller.total_song_duration > 0
                and current_time >= self.controller.total_song_duration - 0.001
                and dpg.get_value("status_text") in {"Playing...", "Paused", "Seeking..."}
            ):
                dpg.set_value("status_text", "Finished.")
                dpg.set_value("seek_slider", self.controller.total_song_duration)
            self.nps_history.append(0)
            dpg.set_value("nps_text", "0")

        dpg.set_value("bpm_value", self.format_bpm(bpm_value))
        dpg.set_value("polyphony_value", f"{polyphony:,}")

        if (
            parsed_midi
            and not self._manual_stop_requested
            and self.controller.total_song_duration > 0
            and current_time >= self.controller.total_song_duration - 0.001
            and dpg.get_value("status_text") == "Playing..."
        ):
            dpg.set_value("status_text", "Finished.")
            dpg.configure_item("play_button", label="Play")
            dpg.set_value(
                "time_text",
                f"{self.format_time(self.controller.total_song_duration)} / {self.format_time(self.controller.total_song_duration)}",
            )
            dpg.set_value("seek_slider", self.controller.total_song_duration)

        if self.piano_roll and self.piano_roll.app_running.is_set():
            self.piano_roll.set_live_stats(played_idx, nps, bpm_value, polyphony)

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
                    clamped_buf = max(0.0, min(float(buf_lvl), 60.0))
                    dpg.set_value("slowdown_text", "Buffered playback")
                    dpg.set_value("slowdown_text_overlay", "Buffered playback")
                    dpg.set_value("buffer_progress", clamped_buf / 60.0)
                    dpg.configure_item("buffer_progress", overlay=f"Buffer: {buf_lvl:.1f}s / 60.0s")
                    dpg.set_value("buffer_progress_overlay", clamped_buf / 60.0)
                    dpg.configure_item("buffer_progress_overlay", overlay=f"Buffer: {buf_lvl:.1f}s / 60.0s")
                    recovery_target = max(0.1, float(getattr(self.controller, "recovery_buffer_target", 4.0)))
                    recovery_lvl = max(
                        0.0,
                        min(float(getattr(self.controller, "recovery_buffer_level", 0.0)), recovery_target),
                    )
                    dpg.set_value("recovery_buffer_progress", recovery_lvl / recovery_target)
                    dpg.set_value("recovery_buffer_progress_overlay", recovery_lvl / recovery_target)
                    if getattr(self.controller, "recovery_active", False):
                        dpg.configure_item(
                            "recovery_buffer_progress",
                            overlay=f"Recovery: {recovery_lvl:.1f}s / {recovery_target:.1f}s",
                        )
                        dpg.configure_item(
                            "recovery_buffer_progress_overlay",
                            overlay=f"Recovery: {recovery_lvl:.1f}s / {recovery_target:.1f}s",
                        )
                    elif buf_lvl >= recovery_target:
                        dpg.configure_item("recovery_buffer_progress", overlay="Recovery: Ready")
                        dpg.configure_item("recovery_buffer_progress_overlay", overlay="Recovery: Ready")
                    else:
                        dpg.configure_item(
                            "recovery_buffer_progress",
                            overlay=f"Recovery: {min(float(buf_lvl), recovery_target):.1f}s / {recovery_target:.1f}s",
                        )
                        dpg.configure_item(
                            "recovery_buffer_progress_overlay",
                            overlay=f"Recovery: {min(float(buf_lvl), recovery_target):.1f}s / {recovery_target:.1f}s",
                        )
                except Exception:
                    dpg.set_value("slowdown_text", "Buffered playback")
                    dpg.set_value("slowdown_text_overlay", "Buffered playback")
                    dpg.set_value("buffer_progress", 0.0)
                    dpg.configure_item("buffer_progress", overlay="Buffer: N/A")
                    dpg.set_value("buffer_progress_overlay", 0.0)
                    dpg.configure_item("buffer_progress_overlay", overlay="Buffer: N/A")
                    dpg.set_value("recovery_buffer_progress", 0.0)
                    dpg.configure_item("recovery_buffer_progress", overlay="Recovery: N/A")
                    dpg.set_value("recovery_buffer_progress_overlay", 0.0)
                    dpg.configure_item("recovery_buffer_progress_overlay", overlay="Recovery: N/A")
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
                dpg.set_value("slowdown_text_overlay", f"Slowdown: {self.controller.slowdown_percentage:.1f}%")
                dpg.set_value("buffer_progress", 0.0)
                dpg.configure_item("buffer_progress", overlay="Buffer: N/A")
                dpg.set_value("buffer_progress_overlay", 0.0)
                dpg.configure_item("buffer_progress_overlay", overlay="Buffer: N/A")
                dpg.set_value("recovery_buffer_progress", 0.0)
                dpg.configure_item("recovery_buffer_progress", overlay="Recovery: N/A")
                dpg.set_value("recovery_buffer_progress_overlay", 0.0)
                dpg.configure_item("recovery_buffer_progress_overlay", overlay="Recovery: N/A")

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("Cleaning up resources...")
        self.controller.playing = False
        self.controller.paused = False
        self.controller.paused_for_seeking = False
        with self.playback_lock:
            self.controller.seek_request_time = None

        try:
            self.controller.stop_backend()
        except Exception:
            traceback.print_exc()

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(0.5)
            if self.playback_thread.is_alive():
                print("Cleanup: playback thread still alive, continuing shutdown.")
        self.playback_thread = None

        if self.audio_sweep_thread and self.audio_sweep_thread.is_alive():
            self.audio_sweep_thread.join(0.05)
        self.audio_sweep_thread = None

        if self.piano_roll:
            self.piano_roll.app_running.clear()
            if self.piano_roll_thread and self.piano_roll_thread.is_alive():
                self.piano_roll_thread.join(0.5)
                if self.piano_roll_thread.is_alive():
                    print("Cleanup: piano roll thread still alive, continuing shutdown.")
        self.piano_roll_thread = None

        if self.render_thread and self.render_thread.is_alive():
            print("Cleanup: render thread still active, continuing shutdown without waiting.")
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
    if os.name == "nt" and any(arg.lower() == "-log" for arg in sys.argv[1:]) and not any(arg.startswith("--multiprocessing-fork") for arg in sys.argv[1:]):
        try:
            kernel32 = ctypes.windll.kernel32
            attached = bool(kernel32.AttachConsole(ctypes.c_uint(-1).value))
            if not attached:
                kernel32.AllocConsole()
            try:
                kernel32.SetConsoleTitleW("LWMP Log Console")
            except Exception:
                pass
            try:
                sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
                sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            except OSError:
                pass
            try:
                sys.stdin = open("CONIN$", "r", encoding="utf-8", buffering=1)
            except OSError:
                pass
            print("[LWMP] Log console enabled via -log")
        except Exception:
            pass
        sys.argv = [sys.argv[0], *[arg for arg in sys.argv[1:] if arg.lower() != "-log"]]
    app = DpgMidiPlayerApp()
    app.run()
