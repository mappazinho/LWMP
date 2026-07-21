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

class SoundfontMixin:
    """Methods for soundfont."""

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


    def _save_soundfont_directories(self, directories):
        self._library_cfg()["soundfont_directories"] = directories
        self._save_config(self._CONFIG)


    def _refresh_soundfont_directory_ui(self):
        directories = self._get_soundfont_directories()
        if dpg.does_item_exist("soundfont_directory_combo"):
            combo_items = directories if directories else ["No folders configured"]
            current_value = combo_items[0]
            dpg.configure_item("soundfont_directory_combo", items=combo_items)
            dpg.set_value("soundfont_directory_combo", current_value)


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


    def add_soundfont_directory_from_input(self, sender=None, app_data=None):
        if not dpg.does_item_exist("soundfont_path_input"):
            return
        self._add_soundfont_directory(dpg.get_value("soundfont_path_input"))


    def show_soundfont_library_window(self, sender=None, app_data=None):
        self._refresh_soundfont_directory_ui()
        self.refresh_soundfont_files()
        dpg.configure_item("soundfont_library_window", show=True)


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


    def _open_soundfont_dialog(self, callback):
        self.pending_soundfont_callback = callback
        self.show_soundfont_library_window()


    def _apply_soundfont_selection(self, sf_path):
        if not sf_path:
            return

        self._CONFIG["audio"]["soundfont_path"] = sf_path
        self._save_config(self._CONFIG)
        self.controller.config["audio"]["soundfont_path"] = sf_path
        self._refresh_soundfont_text()

        current_mode = self._normalize_backend_mode(self._CONFIG["audio"].get("omnimidi_load_preference", "path"))
        if current_mode == "bassmidi":
            if self.controller.parsed_midi is not None:
                if self.controller.reload_soundfont(sf_path):
                    self.set_status(f"SoundFont changed to {os.path.basename(sf_path)}.")
                else:
                    self.set_status(f"SoundFont will change on next MIDI load.")
            else:
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
            self._CONFIG["audio"]["soundfont_path"] = None
            self._save_config(self._CONFIG)

        def _after_pick(selected_path):
            if selected_path:
                self._apply_soundfont_selection(selected_path)
                on_ready()
                return

            self._refresh_soundfont_text()
            self.set_status("No SoundFont selected. Playback will be silent.")

        self._open_soundfont_dialog(_after_pick)


    def _current_soundfont_label(self):
        sf_path = self._CONFIG["audio"].get("soundfont_path")
        if sf_path:
            return os.path.basename(sf_path)
        return "No SoundFont selected"


    def _refresh_soundfont_text(self):
        if dpg.does_item_exist("soundfont_text"):
            dpg.set_value("soundfont_text", self._current_soundfont_label())


    def _refresh_soundfont_visibility(self):
        if dpg.does_item_exist("soundfont_group"):
            selected_mode = None
            if dpg.does_item_exist("backend_combo"):
                selected_label = dpg.get_value("backend_combo")
                selected_mode = self._normalize_backend_mode(
                    self.backend_values.get(selected_label, self._CONFIG["audio"].get("omnimidi_load_preference", "path"))
                )
            if selected_mode is None:
                selected_mode = self.controller.active_backend_mode or self._normalize_backend_mode(
                    self._CONFIG["audio"].get("omnimidi_load_preference", "path")
                )
            dpg.configure_item("soundfont_group", show=(selected_mode == "bassmidi"))


    def change_soundfont(self):
        self.pending_soundfont_callback = None
        self.show_soundfont_library_window()


