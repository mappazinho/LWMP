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
AUDIO_MIN_NOTE_VELOCITY = 10

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
        self.render_thread = None
        self.render_start_time_monotonic = 0.0
        self.render_stage_timing = {}
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
        self.has_bundled_omnimidi = os.path.exists(os.path.join(script_dir, "OmniMIDI.dll"))
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

    def _library_cfg(self):
        return CONFIG["library"]

    def _render_cfg(self):
        return CONFIG["render"]

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

    def _get_library_directories(self):
        dirs = []
        seen = set()
        for directory in self._library_cfg().get("midi_directories", []):
            if not directory:
                continue
            norm = os.path.normcase(os.path.abspath(directory))
            if norm in seen or not os.path.isdir(directory):
                continue
            seen.add(norm)
            dirs.append(os.path.abspath(directory))
        return dirs

    def _get_soundfont_directories(self):
        dirs = []
        seen = set()
        for directory in self._library_cfg().get("soundfont_directories", []):
            if not directory:
                continue
            norm = os.path.normcase(os.path.abspath(directory))
            if norm in seen or not os.path.isdir(directory):
                continue
            seen.add(norm)
            dirs.append(os.path.abspath(directory))
        return dirs

    def _save_library_directories(self, directories):
        self._library_cfg()["midi_directories"] = directories
        save_config(CONFIG)

    def _save_soundfont_directories(self, directories):
        self._library_cfg()["soundfont_directories"] = directories
        save_config(CONFIG)

    def _format_library_file_label(self, root_dir, full_path):
        root_name = os.path.basename(root_dir.rstrip("\\/")) or root_dir
        rel_path = os.path.relpath(full_path, root_dir)
        return f"{root_name} | {rel_path}"

    def _refresh_library_directory_ui(self):
        directories = self._get_library_directories()
        if dpg.does_item_exist("library_directory_combo"):
            combo_items = directories if directories else ["No folders configured"]
            current_value = combo_items[0]
            dpg.configure_item("library_directory_combo", items=combo_items)
            dpg.set_value("library_directory_combo", current_value)

    def _refresh_soundfont_directory_ui(self):
        directories = self._get_soundfont_directories()
        if dpg.does_item_exist("soundfont_directory_combo"):
            combo_items = directories if directories else ["No folders configured"]
            current_value = combo_items[0]
            dpg.configure_item("soundfont_directory_combo", items=combo_items)
            dpg.set_value("soundfont_directory_combo", current_value)

    def refresh_library_files(self, sender=None, app_data=None):
        directories = self._get_library_directories()
        search_text = ""
        previous_selection = None
        if dpg.does_item_exist("library_search_input"):
            search_text = dpg.get_value("library_search_input").strip().lower()
        if dpg.does_item_exist("library_file_list"):
            previous_selection = dpg.get_value("library_file_list")

        entries = []
        for root_dir in directories:
            try:
                for current_root, _, filenames in os.walk(root_dir):
                    for filename in filenames:
                        lower_name = filename.lower()
                        if not (lower_name.endswith(".mid") or lower_name.endswith(".midi")):
                            continue
                        full_path = os.path.join(current_root, filename)
                        label = self._format_library_file_label(root_dir, full_path)
                        if search_text and search_text not in label.lower():
                            continue
                        entries.append((label, full_path))
            except Exception as e:
                print(f"Failed to scan MIDI library directory '{root_dir}': {e}")

        entries.sort(key=lambda item: item[0].lower())
        self.library_file_labels = [label for label, _ in entries]
        self.library_file_map = {label: path for label, path in entries}

        if dpg.does_item_exist("library_file_list"):
            items = self.library_file_labels if self.library_file_labels else ["No MIDI files found"]
            dpg.configure_item("library_file_list", items=items)
            if previous_selection in self.library_file_map:
                dpg.set_value("library_file_list", previous_selection)
            elif not self.library_file_labels:
                dpg.set_value("library_file_list", items[0])
                self.last_library_click_label = None
                self.last_library_click_time = 0.0

        if dpg.does_item_exist("library_count_text"):
            dpg.set_value("library_count_text", f"Files: {len(self.library_file_labels):,}")

    def refresh_soundfont_files(self, sender=None, app_data=None):
        directories = self._get_soundfont_directories()
        search_text = ""
        previous_selection = None
        if dpg.does_item_exist("soundfont_search_input"):
            search_text = dpg.get_value("soundfont_search_input").strip().lower()
        if dpg.does_item_exist("soundfont_file_list"):
            previous_selection = dpg.get_value("soundfont_file_list")

        entries = []
        for root_dir in directories:
            try:
                for current_root, _, filenames in os.walk(root_dir):
                    for filename in filenames:
                        lower_name = filename.lower()
                        if not (lower_name.endswith(".sf2") or lower_name.endswith(".sfz")):
                            continue
                        full_path = os.path.join(current_root, filename)
                        label = self._format_library_file_label(root_dir, full_path)
                        if search_text and search_text not in label.lower():
                            continue
                        entries.append((label, full_path))
            except Exception as e:
                print(f"Failed to scan soundfont library directory '{root_dir}': {e}")

        entries.sort(key=lambda item: item[0].lower())
        self.soundfont_file_labels = [label for label, _ in entries]
        self.soundfont_file_map = {label: path for label, path in entries}

        if dpg.does_item_exist("soundfont_file_list"):
            items = self.soundfont_file_labels if self.soundfont_file_labels else ["No SoundFonts found"]
            dpg.configure_item("soundfont_file_list", items=items)
            if previous_selection in self.soundfont_file_map:
                dpg.set_value("soundfont_file_list", previous_selection)
            elif not self.soundfont_file_labels:
                dpg.set_value("soundfont_file_list", items[0])
                self.last_soundfont_click_label = None
                self.last_soundfont_click_time = 0.0

        if dpg.does_item_exist("soundfont_count_text"):
            dpg.set_value("soundfont_count_text", f"Files: {len(self.soundfont_file_labels):,}")

    def _add_library_directory(self, directory):
        if not directory:
            return
        directory = os.path.abspath(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            self._message_error("Invalid Folder", f"Folder not found:\n{directory}")
            return
        directories = self._get_library_directories()
        norm = os.path.normcase(directory)
        if any(os.path.normcase(existing) == norm for existing in directories):
            self._message_warning("Already Added", "That MIDI folder is already in the library.")
            return
        directories.append(directory)
        directories.sort(key=lambda item: item.lower())
        self._save_library_directories(directories)
        if dpg.does_item_exist("library_path_input"):
            dpg.set_value("library_path_input", "")
        self._refresh_library_directory_ui()
        self.refresh_library_files()

    def _add_soundfont_directory(self, directory):
        if not directory:
            return
        directory = os.path.abspath(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            self._message_error("Invalid Folder", f"Folder not found:\n{directory}")
            return
        directories = self._get_soundfont_directories()
        norm = os.path.normcase(directory)
        if any(os.path.normcase(existing) == norm for existing in directories):
            self._message_warning("Already Added", "That SoundFont folder is already in the library.")
            return
        directories.append(directory)
        directories.sort(key=lambda item: item.lower())
        self._save_soundfont_directories(directories)
        if dpg.does_item_exist("soundfont_path_input"):
            dpg.set_value("soundfont_path_input", "")
        self._refresh_soundfont_directory_ui()
        self.refresh_soundfont_files()

    def add_library_directory_from_input(self, sender=None, app_data=None):
        if not dpg.does_item_exist("library_path_input"):
            return
        self._add_library_directory(dpg.get_value("library_path_input"))

    def add_soundfont_directory_from_input(self, sender=None, app_data=None):
        if not dpg.does_item_exist("soundfont_path_input"):
            return
        self._add_soundfont_directory(dpg.get_value("soundfont_path_input"))

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

    def show_library_window(self, sender=None, app_data=None):
        self._refresh_library_directory_ui()
        self.refresh_library_files()
        dpg.configure_item("library_window", show=True)

    def show_soundfont_library_window(self, sender=None, app_data=None):
        self._refresh_soundfont_directory_ui()
        self.refresh_soundfont_files()
        dpg.configure_item("soundfont_library_window", show=True)

    def _build_nps_spikes_text(self):
        spikes = getattr(self.controller, "max_nps_spikes", [])
        if not spikes:
            return "No spikes detected."
        return "\n".join(
            f"{idx + 1}. {self.format_time(spike_time)} - {self.format_nps(spike_value)}"
            for idx, (spike_time, spike_value) in enumerate(spikes)
        )

    def _refresh_nps_spikes_window(self):
        if dpg.does_item_exist("nps_spikes_summary"):
            dpg.set_value("nps_spikes_summary", f"Max NPS: {self.format_nps(self.controller.max_nps)}")
        if dpg.does_item_exist("nps_spikes_text"):
            dpg.set_value("nps_spikes_text", self._build_nps_spikes_text())

    def show_nps_spikes_window(self, sender=None, app_data=None):
        self._refresh_nps_spikes_window()
        dpg.configure_item("nps_spikes_window", show=True)

    def browse_library_directory(self, sender=None, app_data=None):
        self._open_directory_dialog(self._add_library_directory)

    def remove_selected_library_directory(self, sender=None, app_data=None):
        if not dpg.does_item_exist("library_directory_combo"):
            return
        selected = dpg.get_value("library_directory_combo")
        directories = self._get_library_directories()
        if selected not in directories:
            return
        directories = [directory for directory in directories if directory != selected]
        self._save_library_directories(directories)
        self._refresh_library_directory_ui()
        self.refresh_library_files()

    def remove_selected_soundfont_directory(self, sender=None, app_data=None):
        if not dpg.does_item_exist("soundfont_directory_combo"):
            return
        selected = dpg.get_value("soundfont_directory_combo")
        directories = self._get_soundfont_directories()
        if selected not in directories:
            return
        directories = [directory for directory in directories if directory != selected]
        self._save_soundfont_directories(directories)
        self._refresh_soundfont_directory_ui()
        self.refresh_soundfont_files()

    def load_selected_library_file(self, sender=None, app_data=None):
        if not dpg.does_item_exist("library_file_list"):
            return
        selected_label = dpg.get_value("library_file_list")
        filepath = self.library_file_map.get(selected_label)
        if not filepath:
            return
        dpg.configure_item("library_window", show=False)
        self.load_file(filepath)

    def load_selected_soundfont_file(self, sender=None, app_data=None):
        if not dpg.does_item_exist("soundfont_file_list"):
            return
        selected_label = dpg.get_value("soundfont_file_list")
        sf_path = self.soundfont_file_map.get(selected_label)
        if not sf_path:
            return
        dpg.configure_item("soundfont_library_window", show=False)
        callback = self.pending_soundfont_callback
        self.pending_soundfont_callback = None
        if callback:
            callback(sf_path)
        else:
            self._apply_soundfont_selection(sf_path)

    def on_library_file_selected(self, sender, app_data):
        selected_label = app_data
        if selected_label not in self.library_file_map:
            self.last_library_click_label = None
            self.last_library_click_time = 0.0
            return

        now = time.monotonic()
        if (
            selected_label == self.last_library_click_label
            and (now - self.last_library_click_time) <= 0.35
        ):
            self.last_library_click_label = None
            self.last_library_click_time = 0.0
            self.load_selected_library_file()
            return

        self.last_library_click_label = selected_label
        self.last_library_click_time = now

    def on_soundfont_file_selected(self, sender, app_data):
        selected_label = app_data
        if selected_label not in self.soundfont_file_map:
            self.last_soundfont_click_label = None
            self.last_soundfont_click_time = 0.0
            return

        now = time.monotonic()
        if (
            selected_label == self.last_soundfont_click_label
            and (now - self.last_soundfont_click_time) <= 0.35
        ):
            self.last_soundfont_click_label = None
            self.last_soundfont_click_time = 0.0
            self.load_selected_soundfont_file()
            return

        self.last_soundfont_click_label = selected_label
        self.last_soundfont_click_time = now

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

    def _build_startup_warning(self):
        warnings = []
        if not self.has_bundled_omnimidi:
            warnings.append("Bundled OmniMIDI DLL not found. System PATH mode is recommended.")
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
        CONFIG["audio"]["omnimidi_load_preference"] = selected_mode
        CONFIG["gui"]["startup_completed"] = True
        save_config(CONFIG)
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
        self._set_item_visibility("title_text", gui_cfg["show_subtitle"])
        self._set_item_visibility("audio_panel", gui_cfg["show_audio_panel"])
        self._set_item_visibility("status_line_group", gui_cfg["show_status_line"])
        self._set_item_visibility("backend_hint_text", gui_cfg["show_backend_hint"])
        self._set_item_visibility("performance_section", gui_cfg["show_performance_panel"])
        self._set_item_visibility("nps_graph_group", gui_cfg["show_nps_graph"])
        self._set_item_visibility("cpu_graph_group", gui_cfg["show_cpu_graph"])
        self._apply_performance_overlay_layout()
        self._bind_theme()

    def _apply_performance_overlay_layout(self):
        show_overlay_stats = bool(self._gui_cfg().get("show_pianoroll_stats_overlay", False))
        self._set_item_visibility("nps_primary_group", not show_overlay_stats)
        self._set_item_visibility("nps_max_primary_group", not show_overlay_stats)
        self._set_item_visibility("cpu_primary_group", not show_overlay_stats)
        self._set_item_visibility("runtime_primary_group", not show_overlay_stats)
        self._set_item_visibility("cpu_overlay_group", show_overlay_stats)
        self._set_item_visibility("runtime_overlay_group", show_overlay_stats)

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
            "pianoroll_bg",
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
            "pianoroll_bg",
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
            height=420,
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
                tag="render_stats_overlay_checkbox",
                label="Draw notes passed, NPS, and time in video",
                default_value=bool(self._render_cfg().get("show_stats_overlay", False)),
            )
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
                    with dpg.theme_component(dpg.mvAreaSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(
                            dpg.mvPlotCol_Fill,
                            (color[0], color[1], color[2], 96),
                            category=dpg.mvThemeCat_Plots,
                        )

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

    def _show_message_window(self, title, text):
        if dpg.does_item_exist("message_title"):
            dpg.set_value("message_title", title)
        if dpg.does_item_exist("message_body"):
            dpg.set_value("message_body", text)
        if dpg.does_item_exist("message_window"):
            dpg.configure_item("message_window", label=title, show=True)

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

    def _open_soundfont_dialog(self, callback):
        self.pending_soundfont_callback = callback
        self.show_soundfont_library_window()

    def _apply_soundfont_selection(self, sf_path):
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

    def _extract_dialog_path(self, app_data):
        if isinstance(app_data, dict):
            return app_data.get("file_path_name")
        return None

    def _on_midi_dialog_selected(self, sender, app_data):
        filepath = self._extract_dialog_path(app_data)
        if filepath:
            dpg.configure_item("midi_file_dialog", show=False)
            self._queue_ui(self._begin_load_file, filepath)

    def _on_soundfont_dialog_selected(self, sender, app_data):
        sf_path = self._extract_dialog_path(app_data)
        callback = self.pending_soundfont_callback
        self.pending_soundfont_callback = None
        if callback and sf_path:
            dpg.configure_item("soundfont_file_dialog", show=False)
            self._queue_ui(callback, sf_path)

    def _on_soundfont_dialog_cancel(self):
        callback = self.pending_soundfont_callback
        self.pending_soundfont_callback = None
        if callback:
            callback(None)

    def _ensure_bassmidi_soundfont(self, on_ready):
        sf_path = self.controller.config["audio"].get("soundfont_path")
        if sf_path and os.path.exists(sf_path):
            on_ready()
            return

        if sf_path and not os.path.exists(sf_path):
            self.controller.config["audio"]["soundfont_path"] = None
            CONFIG["audio"]["soundfont_path"] = None
            save_config(CONFIG)

        def _after_pick(selected_path):
            if selected_path:
                self._apply_soundfont_selection(selected_path)
                on_ready()
                return

            self._refresh_soundfont_text()
            self.set_status("No SoundFont selected. Playback will be silent.")

        self._open_soundfont_dialog(_after_pick)

    def _current_soundfont_label(self):
        sf_path = CONFIG["audio"].get("soundfont_path")
        if sf_path:
            return os.path.basename(sf_path)
        return "No SoundFont selected"

    def _refresh_soundfont_text(self):
        if dpg.does_item_exist("soundfont_text"):
            dpg.set_value("soundfont_text", self._current_soundfont_label())

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

    def _refresh_transport_button_state(self):
        has_midi = self.controller.parsed_midi is not None
        if has_midi:
            dpg.show_item("nps_spikes_button")
            dpg.show_item("play_button")
            dpg.show_item("stop_button")
            dpg.show_item("piano_roll_button")
            dpg.show_item("render_button")
            dpg.enable_item("nps_spikes_button")
            dpg.enable_item("play_button")
            dpg.enable_item("stop_button")
            if PianoRoll is not None:
                dpg.enable_item("piano_roll_button")
            else:
                dpg.disable_item("piano_roll_button")
            if BassMidiEngine is not None:
                dpg.enable_item("render_button")
            else:
                dpg.disable_item("render_button")
        else:
            dpg.hide_item("nps_spikes_button")
            dpg.hide_item("play_button")
            dpg.hide_item("stop_button")
            dpg.hide_item("piano_roll_button")
            dpg.hide_item("render_button")
            dpg.disable_item("nps_spikes_button")
            dpg.disable_item("play_button")
            dpg.disable_item("stop_button")
            dpg.disable_item("piano_roll_button")
            dpg.disable_item("render_button")

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

    def initialize_audio_backend(self):
        current_mode = self._normalize_backend_mode(CONFIG["audio"].get("omnimidi_load_preference", "path"))
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
        self.pending_soundfont_callback = None
        self.show_soundfont_library_window()

    def on_volume_change(self, sender, app_data):
        CONFIG["audio"]["volume"] = float(app_data)
        save_config(CONFIG)
        if self.controller.active_midi_backend and hasattr(self.controller.active_midi_backend, "set_volume"):
            try:
                self.controller.active_midi_backend.set_volume(float(app_data))
            except Exception as e:
                print(f"Failed to set volume: {e}")

    def on_voice_limit_change(self, sender, app_data):
        CONFIG["audio"]["voices"] = int(app_data)
        save_config(CONFIG)
        if self.controller.active_midi_backend and hasattr(self.controller.active_midi_backend, "set_voices"):
            try:
                self.controller.active_midi_backend.set_voices(int(app_data))
            except Exception as e:
                print(f"Failed to set voices: {e}")

    def on_speed_change(self, sender, app_data):
        CONFIG["audio"]["speed"] = float(app_data)
        save_config(CONFIG)
        with self.playback_lock:
            self.controller.set_playback_speed(float(app_data))

    def on_pianoroll_stats_overlay_toggle(self, sender, app_data):
        enabled = bool(app_data)
        self._gui_cfg()["show_pianoroll_stats_overlay"] = enabled
        save_config(CONFIG)
        self._apply_performance_overlay_layout()
        if self.piano_roll is not None:
            self.piano_roll.live_show_stats_overlay = enabled

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
        if self.loading_visible and self.loading_total_events <= 0:
            pulse = (time.monotonic() - self.loading_started_at) % 1.0
            pulse_value = 0.15 + (0.7 * pulse)
            self._update_parse_progress(pulse_value, "Counting...", "Counting events...")
        try:
            for status, payload in self.controller.poll_parser_messages():
                event = self.controller.handle_parser_message(status, payload, start_padding=3.0, end_padding=3.0)
                if event["kind"] == "total_events":
                    self.loading_total_events = int(event["total_events"] or 0)
                    if self.loading_visible:
                        self._update_parse_progress(0.0, "0.0%", f"Found {event['total_events']:,} events. Parsing...")
                elif event["kind"] == "progress":
                    if self.loading_visible:
                        progress_payload = event["payload"]
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
                elif event["kind"] == "success":
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
                    return
                else:
                    self._reset_parse_progress()
                    self._message_error("Parse Error", f"Could not load MIDI file: {event['payload']}")
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
                self._manual_stop_requested = False
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
            self._manual_stop_requested = False
            current_time = self.get_current_playback_time()
            self.controller.start_playback(current_time)
            dpg.configure_item("play_button", label="Pause")
            dpg.set_value("status_text", "Playing...")
            dpg.configure_item("voices_slider", enabled=False)
            self.playback_thread = threading.Thread(target=self.play_music_thread, daemon=True)
            self.playback_thread.start()

    def stop_playback(self):
        self._manual_stop_requested = True
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

    def reset_playback_state(self):
        self.controller.reset_playback_state()
        dpg.set_value("seek_slider", 0.0)
        self.reset_graph_history()
        self.panic_all_notes_off()

    def playback_finished(self):
        self.controller.finish_playback()
        dpg.configure_item("play_button", label="Play")
        if self.controller.parsed_midi:
            if self._manual_stop_requested:
                dpg.set_value("status_text", "Ready to play.")
                dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
                dpg.set_value("seek_slider", 0.0)
            else:
                dpg.set_value("status_text", "Finished.")
                finished = self.format_time(self.controller.total_song_duration)
                dpg.set_value("time_text", f"{finished} / {finished}")
                dpg.set_value("seek_slider", self.controller.total_song_duration)
        self._manual_stop_requested = False
        dpg.configure_item("voices_slider", enabled=True)
        dpg.set_value("recovery_buffer_progress", 0.0)
        dpg.configure_item("recovery_buffer_progress", overlay="Recovery: 0.0s / 4.0s")
        dpg.set_value("recovery_buffer_progress_overlay", 0.0)
        dpg.configure_item("recovery_buffer_progress_overlay", overlay="Recovery: 0.0s / 4.0s")
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

            times, statuses, params = self._build_buffered_event_arrays(self.controller.parsed_midi)

            self.controller.active_midi_backend.upload_events(times, statuses, params)

            start_time = self.get_current_playback_time()
            self.controller.active_midi_backend.set_current_time(start_time)
            has_started_playback = False
            emergency_recovery = False
            startup_prerender_target = 3.0 if start_time <= 0.001 else 4.0
            recovery_target = 4.0
            self.controller.recovery_active = False
            self.controller.recovery_buffer_level = 0.0
            self.controller.recovery_buffer_target = recovery_target

            while self.controller.playing:
                if not self.controller.playing:
                    break

                requested_time = None
                with self.playback_lock:
                    if self.controller.seek_request_time is not None:
                        requested_time = self.controller.seek_request_time
                        self.controller.seek_request_time = None

                if requested_time is not None:
                    if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                        self.controller.active_midi_backend.set_emergency_recovery(False)
                        emergency_recovery = False
                    self.controller.recovery_active = False
                    self.controller.recovery_buffer_level = 0.0
                    self.controller.active_midi_backend.stop()
                    self.controller.active_midi_backend.set_current_time(requested_time)
                    self.controller.buffered_playback_start_offset = requested_time
                    has_started_playback = False
                    startup_prerender_target = 4.0

                buffer_lvl = self.controller.active_midi_backend.fill_buffer(60.0)
                is_active = self.controller.active_midi_backend.is_active()
                self.controller.recovery_buffer_target = recovery_target

                if self.controller.paused:
                    if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                        self.controller.active_midi_backend.set_emergency_recovery(False)
                        emergency_recovery = False
                    self.controller.recovery_active = False
                    self.controller.recovery_buffer_level = 0.0
                    self.controller.active_midi_backend.pause()
                    if buffer_lvl >= 60.0:
                        time.sleep(0.1)
                    else:
                        time.sleep(0.005)
                    continue

                if not has_started_playback:
                    if buffer_lvl >= startup_prerender_target:
                        if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                            self.controller.active_midi_backend.set_emergency_recovery(False)
                            emergency_recovery = False
                        self.controller.recovery_active = False
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        self._queue_ui(dpg.set_value, "status_text", "Playing...")
                        self.controller.active_midi_backend.play()
                        has_started_playback = True
                    else:
                        self.controller.recovery_active = False
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        self._queue_ui(
                            dpg.set_value,
                            "status_text",
                            f"Prerendering... {buffer_lvl:.1f}s / {startup_prerender_target:.1f}s",
                        )
                else:
                    if emergency_recovery or buffer_lvl < 0.2:
                        progress = max(0.0, min(buffer_lvl / recovery_target, 1.0))
                        normal_voice_limit = max(
                            1,
                            int(
                                getattr(
                                    self.controller.active_midi_backend,
                                    "normal_voice_limit",
                                    CONFIG["audio"].get("voices", 512),
                                )
                            ),
                        )
                        min_voice_limit = min(normal_voice_limit, 32)
                        eased_progress = progress ** 1.65
                        skip_velocity_below = max(0, min(127, int(round(127.0 * (1.0 - eased_progress)))))
                        recovery_voice_limit = max(
                            min_voice_limit,
                            min(
                                normal_voice_limit,
                                int(
                                    round(
                                        min_voice_limit
                                        + ((normal_voice_limit - min_voice_limit) * eased_progress)
                                    )
                                ),
                            ),
                        )
                        skip_note_ons = progress <= 0.04 and recovery_voice_limit <= min_voice_limit
                        if hasattr(self.controller.active_midi_backend, "configure_emergency_recovery"):
                            self.controller.active_midi_backend.configure_emergency_recovery(
                                skip_velocity_below,
                                recovery_voice_limit,
                                skip_note_ons,
                            )
                        elif not emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                            self.controller.active_midi_backend.set_emergency_recovery(True)
                        emergency_recovery = True
                        self.controller.recovery_active = buffer_lvl < recovery_target
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        if not self.controller.paused and not is_active and buffer_lvl > 0.02:
                            self.controller.active_midi_backend.play()
                        if buffer_lvl >= recovery_target:
                            if hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                                self.controller.active_midi_backend.set_emergency_recovery(False)
                            emergency_recovery = False
                            self.controller.recovery_active = False
                            self._queue_ui(dpg.set_value, "status_text", "Playing...")
                        else:
                            self._queue_ui(
                                dpg.set_value,
                                "status_text",
                                f"Recovering buffer... {buffer_lvl:.1f}s / {recovery_target:.1f}s",
                            )
                    else:
                        self.controller.recovery_active = False
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                            self.controller.active_midi_backend.set_emergency_recovery(False)
                            emergency_recovery = False
                            self._queue_ui(dpg.set_value, "status_text", "Playing...")
                        if not self.controller.paused and not is_active and buffer_lvl > 2.0:
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
            self.controller.recovery_active = False
            self.controller.recovery_buffer_level = 0.0
            if (
                self.controller.active_midi_backend
                and hasattr(self.controller.active_midi_backend, "set_emergency_recovery")
            ):
                try:
                    self.controller.active_midi_backend.set_emergency_recovery(False)
                except Exception:
                    pass
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
            program_change_events = getattr(self.controller.parsed_midi, "program_change_events", [])
            pitch_bend_events = self.controller.parsed_midi.pitch_bend_events
            control_change_events = getattr(self.controller.parsed_midi, "control_change_events", [])

            num_note_events = len(note_events)
            num_program_change_events = len(program_change_events)
            num_pitch_bend_events = len(pitch_bend_events)
            num_control_change_events = len(control_change_events)

            if self.controller.active_midi_backend:
                for channel in range(16):
                    status = 0xE0 + channel
                    param = (0x40 << 8) | 0x00
                    self.controller.active_midi_backend.send_raw_event(status, param)

            start_time = self.get_current_playback_time()
            note_event_index = bisect.bisect_left(note_events["on_time"], start_time)
            program_change_index = bisect.bisect_left(program_change_events, (start_time, -float("inf"), -float("inf")))
            pitch_bend_index = bisect.bisect_left(pitch_bend_events, (start_time, -float("inf"), -float("inf")))
            control_change_index = bisect.bisect_left(control_change_events, (start_time, -float("inf"), -float("inf"), -float("inf")))

            def _apply_program_state(target_time):
                if not self.controller.active_midi_backend:
                    return
                latest_programs = {}
                for change_time, channel, program in program_change_events:
                    if change_time > target_time:
                        break
                    latest_programs[int(channel)] = int(program)
                for channel, program in latest_programs.items():
                    self.controller.active_midi_backend.send_raw_event(0xC0 + channel, program)

            _apply_program_state(start_time)

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
                or program_change_index < num_program_change_events
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
                    program_change_index = bisect.bisect_left(program_change_events, (requested_time, -float("inf"), -float("inf")))
                    pitch_bend_index = bisect.bisect_left(pitch_bend_events, (requested_time, -float("inf"), -float("inf")))
                    control_change_index = bisect.bisect_left(control_change_events, (requested_time, -float("inf"), -float("inf"), -float("inf")))
                    self.controller.playback_start_time = time.monotonic() - (requested_time / max(self.controller.playback_speed, 0.01))
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
                    _apply_program_state(requested_time)

                next_note_on_time = note_events[note_event_index]["on_time"] if note_event_index < num_note_events else float("inf")
                next_note_off_time = note_off_heap[0][0] if note_off_heap else float("inf")
                next_program_change_time = program_change_events[program_change_index][0] if program_change_index < num_program_change_events else float("inf")
                next_pitch_bend_time = pitch_bend_events[pitch_bend_index][0] if pitch_bend_index < num_pitch_bend_events else float("inf")
                next_control_change_time = control_change_events[control_change_index][0] if control_change_index < num_control_change_events else float("inf")
                event_time_sec = min(next_note_on_time, next_note_off_time, next_program_change_time, next_pitch_bend_time, next_control_change_time)

                if event_time_sec == float("inf"):
                    break

                with self.playback_lock:
                    self.controller.last_processed_event_time = event_time_sec

                target_wall_time = self.controller.playback_start_time + (event_time_sec / max(self.controller.playback_speed, 0.01)) + self.controller.total_paused_duration
                sleep_duration = target_wall_time - time.monotonic()
                self.controller.current_lag = max(0, -sleep_duration) * self.controller.playback_speed
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

                        if vel >= AUDIO_MIN_NOTE_VELOCITY:
                            status_on = 0x90 + channel
                            if self.controller.active_midi_backend:
                                param = (vel << 8) | pitch
                            self.controller.active_midi_backend.send_raw_event(status_on, param)
                            heapq.heappush(note_off_heap, (note["off_time"], pitch, channel))

                    while program_change_index < num_program_change_events and program_change_events[program_change_index][0] <= event_time_sec:
                        _time, channel, program = program_change_events[program_change_index]
                        if self.controller.active_midi_backend:
                            self.controller.active_midi_backend.send_raw_event(0xC0 + int(channel), int(program))
                        program_change_index += 1

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
        self._center_modal("piano_roll_window", 320, 220)
        dpg.configure_item("piano_roll_window", show=True)

    def show_render_window(self, sender=None, app_data=None):
        if self.controller.parsed_midi is None:
            self._message_warning("No MIDI", "Load a MIDI before starting a render.")
            return
        if self.render_thread and self.render_thread.is_alive():
            self._message_info("Render In Progress", "A video render is already running.")
            return
        current_ffmpeg = dpg.get_value("render_ffmpeg_path").strip()
        if not current_ffmpeg or current_ffmpeg.lower() == "ffmpeg":
            bundled_ffmpeg = self._bundled_ffmpeg_path()
            if bundled_ffmpeg:
                dpg.set_value("render_ffmpeg_path", bundled_ffmpeg)
        if not dpg.get_value("render_output_path") and self.controller.parsed_midi:
            source_name = os.path.splitext(os.path.basename(self.controller.parsed_midi.filename))[0]
            default_path = os.path.join(os.path.dirname(self.controller.parsed_midi.filename), f"{source_name}_render.mp4")
            dpg.set_value("render_output_path", default_path)
        self._center_modal("render_window", 520, 420)
        dpg.configure_item("render_window", show=True)

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
        self.piano_roll.live_show_stats_overlay = bool(self._gui_cfg().get("show_pianoroll_stats_overlay", False))
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

        if self.controller.parsed_midi:
            notes_for_gpu = np.ascontiguousarray(self.controller.parsed_midi.note_data_for_gpu)
            threading.Timer(
                0.5,
                lambda: self.piano_roll.load_midi(notes_for_gpu, self.get_current_playback_time_thread_safe),
            ).start()

    def get_current_playback_time_thread_safe(self):
        return self.controller.current_playback_time_for_threads

    def _set_render_progress(self, fraction, overlay, detail):
        if dpg.does_item_exist("render_progress_bar"):
            dpg.set_value("render_progress_bar", max(0.0, min(float(fraction), 1.0)))
            dpg.configure_item("render_progress_bar", overlay=str(overlay))
        if dpg.does_item_exist("render_status_text"):
            dpg.set_value("render_status_text", detail)

    def _format_render_eta(self, seconds_remaining):
        seconds_remaining = max(0, int(round(float(seconds_remaining))))
        hours, rem = divmod(seconds_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _render_progress_with_timing(self, stage_name, stage_start_time, stage_fraction, overall_fraction, overlay, detail):
        stage_fraction = max(0.0, min(float(stage_fraction), 1.0))
        overall_fraction = max(0.0, min(float(overall_fraction), 1.0))
        detail_text = str(detail)
        now = time.monotonic()
        overall_start_time = getattr(self, "render_start_time_monotonic", 0.0) or stage_start_time
        elapsed = max(0.0, now - overall_start_time)
        detail_text = f"{detail_text}\nTime elapsed: {self._format_render_eta(elapsed)}"
        stage_elapsed = max(0.0, now - stage_start_time)

        state = self.render_stage_timing.get(stage_name)
        if state is None or abs(state.get("start_time", 0.0) - stage_start_time) > 1e-6:
            state = {
                "start_time": stage_start_time,
                "last_time": now,
                "last_fraction": stage_fraction,
                "ema_rate": 0.0,
            }
            self.render_stage_timing[stage_name] = state
        else:
            dt = max(0.0, now - state["last_time"])
            df = max(0.0, stage_fraction - state["last_fraction"])
            if dt > 0.05 and df > 0.0:
                instant_rate = df / dt
                if state["ema_rate"] <= 0.0:
                    state["ema_rate"] = instant_rate
                else:
                    state["ema_rate"] = (state["ema_rate"] * 0.65) + (instant_rate * 0.35)
            state["last_time"] = now
            state["last_fraction"] = stage_fraction

        eta_seconds = None
        if stage_fraction > 0.001 and stage_fraction < 1.0 and stage_elapsed > 0.25:
            if state["ema_rate"] > 1e-6:
                eta_seconds = max(0.0, (1.0 - stage_fraction) / state["ema_rate"])
            else:
                estimated_total = stage_elapsed / stage_fraction
                eta_seconds = max(0.0, estimated_total - stage_elapsed)
        if eta_seconds is not None:
            detail_text = f"{detail_text}\n{stage_name} time left: {self._format_render_eta(eta_seconds)}"
        self._set_render_progress(overall_fraction, overlay, detail_text)

    def _bundled_ffmpeg_path(self):
        for candidate in _BUNDLED_FFMPEG_CANDIDATES:
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _resolve_ffmpeg_path(self, requested_path):
        requested_path = (requested_path or "").strip()
        if not requested_path:
            requested_path = "ffmpeg"
        if os.path.isfile(requested_path):
            return requested_path
        if requested_path.lower() == "ffmpeg":
            bundled = self._bundled_ffmpeg_path()
            if bundled:
                return bundled
        resolved = shutil.which(requested_path)
        return resolved

    def _normalize_render_output_path(self, output_path, codec_label):
        output_path = (output_path or "").strip()
        if not output_path:
            return output_path
        _, default_ext = self._codec_settings(codec_label)
        root, ext = os.path.splitext(output_path)
        if not ext:
            return output_path + default_ext
        return output_path

    def _format_ffmpeg_error(self, ffmpeg_cmd, returncode, stderr_output):
        stderr_output = (stderr_output or "").strip()
        if stderr_output:
            stderr_lines = [line.rstrip() for line in stderr_output.splitlines() if line.strip()]
            if len(stderr_lines) > 18:
                stderr_lines = stderr_lines[-18:]
            stderr_text = "\n".join(stderr_lines)
        else:
            stderr_text = f"ffmpeg exited with code {returncode}."
        return (
            f"FFmpeg failed while encoding the video.\n\n"
            f"Command:\n{' '.join(ffmpeg_cmd)}\n\n"
            f"Details:\n{stderr_text}"
        )

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

    def _build_watermark_filter(self, total_duration):
        start_time = max(0.0, float(total_duration) - 5.0)
        fade_in_end = start_time + 1.0
        hold_end = start_time + 4.0
        end_time = start_time + 5.0
        alpha_expr = (
            f"0.25*if(lt(t,{start_time:.3f}),0,"
            f"if(lt(t,{fade_in_end:.3f}),(t-{start_time:.3f})/1.0,"
            f"if(lt(t,{hold_end:.3f}),1,"
            f"if(lt(t,{end_time:.3f}),({end_time:.3f}-t)/1.0,0))))"
        )
        filter_parts = [
            "drawtext=text='Rendered with LWMP'",
            "fontcolor=white",
            "fontsize=28",
            "x=w-tw-18",
            "y=h-th-16",
            f"alpha='{alpha_expr}'",
        ]
        fontfile = self._preferred_drawtext_font()
        if fontfile:
            filter_parts.insert(1, f"fontfile='{self._escape_drawtext_value(fontfile)}'")
        return ":".join(filter_parts)

    def _build_buffered_event_arrays(self, parsed_midi):
        note_events = parsed_midi.note_events_for_playback
        audible_note_events = note_events[note_events["velocity"] >= AUDIO_MIN_NOTE_VELOCITY]
        program_change_events = getattr(parsed_midi, "program_change_events", [])
        pitch_bend_events = parsed_midi.pitch_bend_events
        control_change_events = getattr(parsed_midi, "control_change_events", [])

        count_notes = len(audible_note_events)
        count_programs = len(program_change_events)
        count_bends = len(pitch_bend_events)
        count_ccs = len(control_change_events)
        total_ops = (count_notes * 2) + count_programs + count_bends + count_ccs

        times = np.empty(total_ops, dtype=np.float64)
        statuses = np.empty(total_ops, dtype=np.uint32)
        params = np.empty(total_ops, dtype=np.uint32)
        priorities = np.empty(total_ops, dtype=np.uint8)

        times[:count_notes] = audible_note_events["on_time"]
        statuses[:count_notes] = 0x90 + audible_note_events["channel"]
        params[:count_notes] = (audible_note_events["velocity"].astype(np.uint32) << 8) | audible_note_events["pitch"].astype(np.uint32)
        priorities[:count_notes] = 2

        times[count_notes : count_notes * 2] = audible_note_events["off_time"]
        statuses[count_notes : count_notes * 2] = 0x80 + audible_note_events["channel"]
        params[count_notes : count_notes * 2] = audible_note_events["pitch"].astype(np.uint32)
        priorities[count_notes : count_notes * 2] = 0

        if count_programs > 0:
            pc_arr = np.array(program_change_events, dtype=[("time", "f8"), ("chan", "u4"), ("program", "u4")])
            start_idx = count_notes * 2
            end_idx = start_idx + count_programs
            times[start_idx:end_idx] = pc_arr["time"]
            statuses[start_idx:end_idx] = 0xC0 + pc_arr["chan"]
            params[start_idx:end_idx] = pc_arr["program"]
            priorities[start_idx:end_idx] = 1

        if count_bends > 0:
            pb_arr = np.array(pitch_bend_events, dtype=[("time", "f8"), ("chan", "u4"), ("val", "u4")])
            start_idx = (count_notes * 2) + count_programs
            end_idx = start_idx + count_bends
            times[start_idx:end_idx] = pb_arr["time"]
            statuses[start_idx:end_idx] = 0xE0 + pb_arr["chan"]
            bend_lsb = pb_arr["val"] & 0x7F
            bend_msb = (pb_arr["val"] >> 7) & 0x7F
            params[start_idx:end_idx] = (bend_msb << 8) | bend_lsb
            priorities[start_idx:end_idx] = 1

        if count_ccs > 0:
            cc_arr = np.array(control_change_events, dtype=[("time", "f8"), ("chan", "u4"), ("cc", "u4"), ("val", "u4")])
            start_idx = (count_notes * 2) + count_programs + count_bends
            end_idx = start_idx + count_ccs
            times[start_idx:end_idx] = cc_arr["time"]
            statuses[start_idx:end_idx] = 0xB0 + cc_arr["chan"]
            params[start_idx:end_idx] = (cc_arr["val"] << 8) | cc_arr["cc"]
            priorities[start_idx:end_idx] = 1

        sort_indices = np.lexsort((np.arange(total_ops, dtype=np.int64), priorities, times))
        return times[sort_indices], statuses[sort_indices], params[sort_indices]

    def _render_audio_to_wav(self, wav_path, parsed_midi):
        stage_start_time = time.monotonic()
        bass_cls = self.controller.bass_engine_cls
        if bass_cls is None:
            raise RuntimeError("BASSMIDI engine not available for export.")

        latest_config = load_config()
        audio_cfg = latest_config.get("audio", {})
        soundfont_path = audio_cfg.get("soundfont_path")
        if not soundfont_path or not os.path.exists(soundfont_path):
            raise RuntimeError("A valid SoundFont is required for video rendering.")

        engine = bass_cls({}, soundfont_path=soundfont_path, buffering=True, debug=DEBUG)
        try:
            render_volume = float(audio_cfg.get("volume", 0.5))
            render_voices = int(audio_cfg.get("voices", 512))
            engine.set_volume(render_volume)
            if hasattr(engine, "set_voices"):
                engine.set_voices(render_voices)
            if hasattr(engine, "set_emergency_recovery"):
                engine.set_emergency_recovery(False)
                if hasattr(engine, "set_voices"):
                    engine.set_voices(render_voices)
            if hasattr(engine, "set_pitch_bend_range"):
                engine.set_pitch_bend_range(12)

            times, statuses, params = self._build_buffered_event_arrays(parsed_midi)
            engine.upload_events(times, statuses, params)
            engine.set_current_time(0.0)

            total_duration = float(parsed_midi.total_duration_sec)
            chunk_seconds = 1.0 / 60.0
            rendered_audio_time = 0.0
            sample_rate = 44100
            channels = 2

            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                while rendered_audio_time < total_duration:
                    remaining = total_duration - rendered_audio_time
                    requested_seconds = min(chunk_seconds, remaining)
                    target_frames = max(1, int(round(requested_seconds * sample_rate)))
                    target_samples = target_frames * channels
                    pcm_chunk = engine.render_pcm_chunk(requested_seconds)
                    if not pcm_chunk:
                        pcm_int16 = np.zeros(target_samples, dtype=np.int16)
                    else:
                        pcm_float = np.frombuffer(pcm_chunk, dtype=np.float32)
                        pcm_int16 = np.clip(pcm_float, -1.0, 1.0)
                        pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)
                        sample_delta = target_samples - pcm_int16.size
                        if sample_delta > 0:
                            pcm_int16 = np.pad(pcm_int16, (0, sample_delta), mode="constant")
                        elif sample_delta < 0:
                            pcm_int16 = pcm_int16[:target_samples]
                    wav_file.writeframes(pcm_int16.tobytes())
                    rendered_audio_time = min(total_duration, rendered_audio_time + requested_seconds)
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Audio",
                        stage_start_time,
                        rendered_audio_time / max(total_duration, 0.001),
                        0.45 * (rendered_audio_time / max(total_duration, 0.001)),
                        f"Audio {rendered_audio_time:.1f}s / {total_duration:.1f}s",
                        "Rendering audio...",
                    )
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass

    def _codec_settings(self, codec_label):
        if codec_label == "H.265":
            return ["-c:v", "libx265", "-pix_fmt", "yuv420p"], ".mp4"
        if codec_label == "MPEG-4":
            return ["-c:v", "mpeg4"], ".mp4"
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p"], ".mp4"

    def _render_video_stream_only(self, ffmpeg_bin, parsed_midi, settings, video_output_path):
        stage_start_time = time.monotonic()
        codec_args, _ = self._codec_settings(settings["codec"])
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{settings['width']}x{settings['height']}",
            "-r", str(settings["framerate"]),
            "-i", "-",
            *codec_args,
            "-b:v", settings["bitrate"],
            "-an",
            video_output_path,
        ]

        self._queue_ui(
            self._render_progress_with_timing,
            "Video",
            stage_start_time,
            0.0,
            0.48,
            "Video",
            "Rendering piano roll frames...",
        )
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        piano_roll = PianoRoll(settings["width"], settings["height"], CONFIG)
        piano_roll.set_export_mode(True)
        piano_roll.set_nps_spikes(getattr(self.controller, "max_nps_spikes", []))
        piano_roll.set_stats_context(
            parsed_midi.note_events_for_playback["on_time"],
            getattr(parsed_midi, "sorted_off_times", None),
            getattr(parsed_midi, "tempo_events", None),
            float(parsed_midi.total_duration_sec),
            int(len(parsed_midi.note_events_for_playback)),
        )
        piano_roll.set_preferred_color_mode(getattr(parsed_midi, "preferred_color_mode", "track"))
        ffmpeg_stderr_chunks = []

        def _drain_ffmpeg_stderr():
            try:
                while process.stderr:
                    chunk = process.stderr.read(4096)
                    if not chunk:
                        break
                    ffmpeg_stderr_chunks.append(chunk)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_ffmpeg_stderr, daemon=True)
        stderr_thread.start()
        try:
            try:
                piano_roll.init_pygame_and_gl(hidden=True)
            except Exception:
                piano_roll.init_pygame_and_gl(hidden=False)
            notes_for_gpu = np.ascontiguousarray(parsed_midi.note_data_for_gpu)
            piano_roll.load_midi(notes_for_gpu, lambda: 0.0)
            total_duration = float(parsed_midi.total_duration_sec)
            total_frames = max(1, int(math.ceil(total_duration * settings["framerate"])))

            for frame_idx in range(total_frames):
                current_time = min(total_duration, frame_idx / float(settings["framerate"]))
                piano_roll.draw(current_time, present=False)
                try:
                    process.stdin.write(piano_roll.capture_frame_rgb())
                except BrokenPipeError:
                    break
                if process.poll() is not None:
                    break
                if frame_idx % max(1, settings["framerate"] // 2) == 0:
                    fraction_start = 0.5 if settings["render_audio"] else 0.02
                    fraction_span = 0.28 if settings["render_audio"] else 0.94
                    stage_fraction = (frame_idx + 1) / float(total_frames)
                    fraction = fraction_start + (fraction_span * ((frame_idx + 1) / float(total_frames)))
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Video",
                        stage_start_time,
                        stage_fraction,
                        fraction,
                        f"Frame {frame_idx + 1:,} / {total_frames:,}",
                        "Rendering piano roll frames...",
                    )
        finally:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                piano_roll.cleanup()
            except Exception:
                pass

        process.wait()
        stderr_thread.join(timeout=2.0)
        ffmpeg_stderr = b"".join(ffmpeg_stderr_chunks).decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(self._format_ffmpeg_error(ffmpeg_cmd, process.returncode, ffmpeg_stderr))

    def _finalize_render_output(self, ffmpeg_bin, video_path, output_path, settings, total_duration, audio_path=None):
        stage_start_time = time.monotonic()
        codec_args, _ = self._codec_settings(settings["codec"])
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-i", video_path,
        ]
        if audio_path:
            ffmpeg_cmd.extend(["-i", audio_path])
        if settings.get("show_watermark", True):
            ffmpeg_cmd.extend(["-vf", self._build_watermark_filter(total_duration)])
        ffmpeg_cmd.extend([
            *codec_args,
            "-b:v", settings["bitrate"],
        ])
        if audio_path:
            ffmpeg_cmd.extend([
            "-c:a", "aac",
            "-b:a", settings["audio_bitrate"],
            "-shortest",
            ])
        else:
            ffmpeg_cmd.append("-an")
        ffmpeg_cmd.append(output_path)
        self._queue_ui(
            self._render_progress_with_timing,
            "Finalizing",
            stage_start_time,
            0.0,
            0.82,
            "Finalizing",
            "Finalizing and writing video...",
        )
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        stderr_output = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(self._format_ffmpeg_error(ffmpeg_cmd, process.returncode, stderr_output))

    def start_render_video(self, sender=None, app_data=None):
        if self.render_thread and self.render_thread.is_alive():
            self._message_warning("Render Busy", "A render is already in progress.")
            return
        if self.controller.parsed_midi is None:
            self._message_warning("No MIDI", "Load a MIDI before rendering.")
            return
        if self.piano_roll and self.piano_roll.app_running.is_set():
            self._message_warning("Close Piano Roll", "Close the live piano roll before starting a video render.")
            return
        if PianoRoll is None:
            self._message_warning("Piano Roll Unavailable", "Piano roll rendering is not available in this build.")
            return

        ffmpeg_path = dpg.get_value("render_ffmpeg_path").strip()
        output_path = self._normalize_render_output_path(dpg.get_value("render_output_path").strip(), dpg.get_value("render_codec"))
        if not output_path:
            self._message_warning("Output Required", "Enter an output video path before rendering.")
            return

        codec_label = dpg.get_value("render_codec")
        resolution = dpg.get_value("render_resolution")
        framerate = max(1, int(dpg.get_value("render_framerate")))
        bitrate = dpg.get_value("render_bitrate").strip() or "20M"
        audio_bitrate = dpg.get_value("render_audio_bitrate").strip() or "320k"
        render_audio = bool(dpg.get_value("render_audio_checkbox"))
        show_stats_overlay = bool(dpg.get_value("render_stats_overlay_checkbox"))
        show_watermark = bool(dpg.get_value("render_watermark_checkbox"))

        render_cfg = self._render_cfg()
        render_cfg["ffmpeg_path"] = ffmpeg_path
        render_cfg["output_path"] = output_path
        render_cfg["codec"] = codec_label
        render_cfg["resolution"] = resolution
        render_cfg["framerate"] = framerate
        render_cfg["bitrate"] = bitrate
        render_cfg["audio_bitrate"] = audio_bitrate
        render_cfg["render_audio"] = render_audio
        render_cfg["show_stats_overlay"] = show_stats_overlay
        render_cfg["show_watermark"] = show_watermark
        save_config(CONFIG)

        width_str, height_str = resolution.split(" x ")
        settings = {
            "ffmpeg_path": ffmpeg_path,
            "output_path": output_path,
            "codec": codec_label,
            "width": int(width_str),
            "height": int(height_str),
            "framerate": framerate,
            "bitrate": bitrate,
            "audio_bitrate": audio_bitrate,
            "render_audio": render_audio,
            "show_stats_overlay": show_stats_overlay,
            "show_watermark": show_watermark,
        }

        dpg.set_value("render_output_path", output_path)
        dpg.disable_item("start_render_button")
        self._set_render_progress(0.0, "Preparing", "Preparing video render...")
        self.render_start_time_monotonic = time.monotonic()
        self.render_stage_timing = {}
        self.render_thread = threading.Thread(target=self._render_video_job, args=(settings,), daemon=True)
        self.render_thread.start()

    def _render_video_job(self, settings):
        ffmpeg_bin = self._resolve_ffmpeg_path(settings["ffmpeg_path"])
        if not ffmpeg_bin:
            self._queue_ui(self._set_render_progress, 0.0, "FFmpeg missing", "FFmpeg executable not found. Set a valid ffmpeg path.")
            self._queue_ui(dpg.enable_item, "start_render_button")
            return

        parsed_midi = self.controller.parsed_midi
        if parsed_midi is None:
            self._queue_ui(self._set_render_progress, 0.0, "No MIDI", "No parsed MIDI is available for rendering.")
            self._queue_ui(dpg.enable_item, "start_render_button")
            return

        output_path = settings["output_path"]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            with tempfile.TemporaryDirectory(prefix="lwmp_render_") as temp_dir:
                wav_path = os.path.join(temp_dir, "audio.wav")
                video_only_path = os.path.join(temp_dir, "video_only.mp4")

                if settings["render_audio"]:
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Audio",
                        time.monotonic(),
                        0.0,
                        0.02,
                        "Audio",
                        "Rendering full audio track...",
                    )
                    self._render_audio_to_wav(wav_path, parsed_midi)
                else:
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Video",
                        time.monotonic(),
                        0.0,
                        0.02,
                        "Video",
                        "Skipping audio render.",
                    )

                self._render_video_stream_only(ffmpeg_bin, parsed_midi, settings, video_only_path)

                if settings["render_audio"]:
                    self._finalize_render_output(
                        ffmpeg_bin,
                        video_only_path,
                        output_path,
                        settings,
                        parsed_midi.total_duration_sec,
                        audio_path=wav_path,
                    )
                else:
                    self._finalize_render_output(
                        ffmpeg_bin,
                        video_only_path,
                        output_path,
                        settings,
                        parsed_midi.total_duration_sec,
                    )

            total_elapsed = self._format_render_eta(time.monotonic() - self.render_start_time_monotonic)
            self._queue_ui(
                self._set_render_progress,
                1.0,
                "Done",
                f"Render complete: {output_path}\nTime elapsed: {total_elapsed}",
            )
            self._queue_ui(self.set_status, f"Render complete: {os.path.basename(output_path)}")
            self._queue_ui(self._message_info, "Render Complete", f"Video saved to:\n{output_path}")
        except Exception as e:
            traceback.print_exc()
            self._queue_ui(self._set_render_progress, 0.0, "Render Failed", str(e))
            self._queue_ui(self._message_error, "Render Failed", str(e))
        finally:
            self._queue_ui(dpg.enable_item, "start_render_button")

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
    app = DpgMidiPlayerApp()
    app.run()
