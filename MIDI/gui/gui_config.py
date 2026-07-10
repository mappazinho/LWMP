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

class GuiConfigMixin:
    """Methods for gui_config."""

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

    # --- Skin Browser ---


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

        self._save_config(self._CONFIG)
        self._apply_gui_customization()
        dpg.configure_item("customize_window", show=False)


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


