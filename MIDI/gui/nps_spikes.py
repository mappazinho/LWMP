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

class NpsSpikesMixin:
    """Methods for nps_spikes."""

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


