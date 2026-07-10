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

class LibraryMixin:
    """Methods for library."""

    def _library_cfg(self):
        return self._CONFIG["library"]


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


    def _save_library_directories(self, directories):
        self._library_cfg()["midi_directories"] = directories
        self._save_config(self._CONFIG)


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


    def add_library_directory_from_input(self, sender=None, app_data=None):
        if not dpg.does_item_exist("library_path_input"):
            return
        self._add_library_directory(dpg.get_value("library_path_input"))


    def show_library_window(self, sender=None, app_data=None):
        self._refresh_library_directory_ui()
        self.refresh_library_files()
        dpg.configure_item("library_window", show=True)


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


    def load_selected_library_file(self, sender=None, app_data=None):
        if not dpg.does_item_exist("library_file_list"):
            return
        selected_label = dpg.get_value("library_file_list")
        filepath = self.library_file_map.get(selected_label)
        if not filepath:
            return
        dpg.configure_item("library_window", show=False)
        self.load_file(filepath)


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


