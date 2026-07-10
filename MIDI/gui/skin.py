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

class SkinMixin:
    """Methods for skin."""

    def show_skin_window(self, sender=None, app_data=None):
        if self.piano_roll and self.piano_roll.app_running.is_set():
            self._message_warning("Skin Browser", "Close the Piano Roll before opening the Skin Browser.")
            return
        if not self._SkinBrowser:
            self._message_warning("Skin Browser", "Piano roll module not available.")
            return
        current = self._CONFIG.get("visualizer", {}).get("skin_name", "default")
        browser = self._SkinBrowser(self._CONFIG, self._SKIN_ROOT, current)
        result = browser.run_browser()
        if result:
            self._CONFIG.setdefault("visualizer", {})["skin_name"] = result
            self._save_config(self._CONFIG)
            self._message_info("Skin Applied", f"Skin '{result}' applied. Restart piano roll to see changes.")


    def _refresh_skin_list(self):
        skins = self._list_available_skins(self._SKIN_ROOT)
        if not skins:
            skins = ["(no skins found)"]
        dpg.configure_item("skin_list", items=skins)
        current = self._CONFIG.get("visualizer", {}).get("skin_name", "default")
        if current in skins:
            dpg.set_value("skin_list", current)
        elif skins:
            dpg.set_value("skin_list", skins[0])
        self._update_skin_preview()


    def _on_skin_selected(self, sender=None, app_data=None):
        self._update_skin_preview()


    def _update_skin_preview(self):
        tag = "skin_preview_drawlist"
        if not dpg.does_item_exist(tag):
            return
        dpg.delete_item(tag, children_only=True)

        selected = dpg.get_value("skin_list")
        if not selected or selected.startswith("("):
            return

        skin_dir = self._resolve_skin_dir(selected, self._SKIN_ROOT)
        if skin_dir is None:
            dpg.draw_text(parent=tag, pos=[10, 10], text="Skin not found.",
                          color=(200, 80, 80))
            return

        # Load textures from the skin folder
        textures = {}
        tex_files = {
            "keyWhite": "keyWhite.png",
            "keyWhitePressed": "keyWhitePressed.png",
            "keyBlack": "keyBlack.png",
            "keyBlackPressed": "keyBlackPressed.png",
            "note": "note.png",
            "noteEdge": "noteEdge.png",
        }
        for key, filename in tex_files.items():
            filepath = os.path.join(skin_dir, filename)
            tex_tag = f"skin_tex_{key}"
            if dpg.does_item_exist(tex_tag):
                dpg.delete_item(tex_tag)
            if os.path.isfile(filepath):
                try:
                    w, h, ch, data = dpg.load_image(filepath)
                    textures[key] = dpg.add_static_texture(w, h, data, tag=tex_tag)
                except Exception:
                    pass

        # Draw the preview
        W, H = 460, 340
        gui_cfg = self._CONFIG.get("gui", {})
        bg = tuple(gui_cfg.get("pianoroll_bg", [13, 13, 20]))
        cursor_col = (60, 60, 80, 180)
        note_colors = [
            (120, 180, 255), (255, 140, 100), (100, 255, 160),
            (255, 220, 100), (200, 130, 255), (255, 100, 180),
        ]

        # Background
        dpg.draw_rectangle(parent=tag, pmin=[0, 0], pmax=[W, H],
                           color=bg, fill=bg)

        # Keyboard area at the bottom
        kb_y = H - 80
        white_w = 18
        num_whites = W // white_w
        white_indices = [0, 2, 4, 5, 7, 9, 11]
        is_white = [(i % 12) in white_indices for i in range(128)]
        # Draw white keys
        if "keyWhite" in textures:
            for i in range(num_whites):
                x0 = i * white_w
                dpg.draw_image(parent=tag, texture_id=textures["keyWhite"],
                               pmin=[x0, kb_y], pmax=[x0 + white_w, H],
                               uv_min=[0, 0], uv_max=[1, 1])
        else:
            for i in range(num_whites):
                x0 = i * white_w
                dpg.draw_rectangle(parent=tag, pmin=[x0, kb_y],
                                   pmax=[x0 + white_w, H],
                                   color=(200, 200, 200),
                                   fill=(240, 240, 240))

        # Draw black keys
        black_w = int(white_w * 0.6)
        black_h = 50
        wi = 0
        for note in range(128):
            if is_white[note]:
                wi += 1
                continue
            if wi == 0:
                continue
            x_center = (wi - 1) * white_w + white_w
            x0 = x_center - black_w // 2
            if x0 + black_w > W:
                break
            if "keyBlack" in textures:
                dpg.draw_image(parent=tag, texture_id=textures["keyBlack"],
                               pmin=[x0, kb_y], pmax=[x0 + black_w, kb_y + black_h],
                               uv_min=[0, 0], uv_max=[1, 1])
            else:
                dpg.draw_rectangle(parent=tag, pmin=[x0, kb_y],
                                   pmax=[x0 + black_w, kb_y + black_h],
                                   color=(30, 30, 30), fill=(40, 40, 40))

        # Pressed keys
        pressed_whites = [3, 7, 10, 15]
        pressed_blacks = [5, 12]
        for pi in pressed_whites:
            if pi < num_whites:
                x0 = pi * white_w
                if "keyWhitePressed" in textures:
                    dpg.draw_image(parent=tag, texture_id=textures["keyWhitePressed"],
                                   pmin=[x0, kb_y], pmax=[x0 + white_w, H],
                                   uv_min=[0, 0], uv_max=[1, 1])
                else:
                    dpg.draw_rectangle(parent=tag, pmin=[x0, kb_y],
                                       pmax=[x0 + white_w, H],
                                       color=(180, 180, 180),
                                       fill=(210, 210, 220))
        wi = 0
        pressed_count = 0
        for note in range(128):
            if is_white[note]:
                wi += 1
                continue
            if wi == 0:
                continue
            if pressed_count in pressed_blacks:
                x_center = (wi - 1) * white_w + white_w
                x0 = x_center - black_w // 2
                if x0 + black_w <= W:
                    if "keyBlackPressed" in textures:
                        dpg.draw_image(parent=tag, texture_id=textures["keyBlackPressed"],
                                       pmin=[x0, kb_y],
                                       pmax=[x0 + black_w, kb_y + black_h],
                                       uv_min=[0, 0], uv_max=[1, 1])
                    else:
                        dpg.draw_rectangle(parent=tag, pmin=[x0, kb_y],
                                           pmax=[x0 + black_w, kb_y + black_h],
                                           color=(20, 20, 20),
                                           fill=(60, 60, 70))
            pressed_count += 1

        # Cursor line
        cursor_y = kb_y - 4
        dpg.draw_line(parent=tag, p1=[0, cursor_y], p2=[W, cursor_y],
                      color=cursor_col, thickness=2)

        # Sample notes above the cursor
        note_data = [
            (60, 20, 80), (64, 50, 120), (67, 90, 60),
            (72, 130, 150), (55, 170, 100), (59, 220, 70),
            (62, 260, 130), (65, 300, 90), (69, 340, 110),
        ]
        for idx, (pitch, nx, nw) in enumerate(note_data):
            col = note_colors[idx % len(note_colors)]
            ny = 20 + (90 - pitch) * 2.5
            nh = 10
            if "note" in textures:
                dpg.draw_image(parent=tag, texture_id=textures["note"],
                               pmin=[nx, ny], pmax=[nx + nw, ny + nh],
                               uv_min=[0, 0], uv_max=[1, 1],
                               color=col)
                if "noteEdge" in textures:
                    edge_w = min(8, nw // 3)
                    dpg.draw_image(parent=tag, texture_id=textures["noteEdge"],
                                   pmin=[nx, ny], pmax=[nx + edge_w, ny + nh],
                                   uv_min=[0, 0], uv_max=[1, 1],
                                   color=col)
                    dpg.draw_image(parent=tag, texture_id=textures["noteEdge"],
                                   pmin=[nx + nw - edge_w, ny],
                                   pmax=[nx + nw, ny + nh],
                                   uv_min=[1, 0], uv_max=[0, 1],
                                   color=col)
            else:
                dpg.draw_rectangle(parent=tag, pmin=[nx, ny],
                                   pmax=[nx + nw, ny + nh],
                                   color=col, fill=col)


    def _apply_selected_skin(self, sender=None, app_data=None):
        selected = dpg.get_value("skin_list")
        if not selected or selected.startswith("("):
            return
        skin_dir = self._resolve_skin_dir(selected, self._SKIN_ROOT)
        if skin_dir is None:
            dpg.set_value("skin_status_text", f"Skin '{selected}' not found!")
            return
        self._CONFIG.setdefault("visualizer", {})["skin_name"] = selected
        self._save_config(self._CONFIG)
        dpg.set_value("skin_status_text", f"Applied: {selected}. Restart piano roll to see changes.")


