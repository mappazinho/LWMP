import json
import os
import threading
from pathlib import Path

import numpy as np

try:
    from pedalboard import Pedalboard, load_plugin
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False


DEFAULT_VST_DIRS = [
    os.path.expandvars(r"%PROGRAMFILES%\VSTPlugins"),
    os.path.expandvars(r"%PROGRAMFILES%\Common Files\VST3"),
    os.path.expandvars(r"%PROGRAMFILES(x86)%\VSTPlugins"),
    os.path.expandvars(r"%PROGRAMFILES(x86)%\Common Files\VST3"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Common\VST3"),
    os.path.expandvars(r"%COMMONPROGRAMFILES%\VST3"),
]


def scan_vst_plugins(directories=None, extensions=(".dll", ".vst3")):
    plugins = []
    dirs = directories or DEFAULT_VST_DIRS
    for d in dirs:
        d = os.path.expandvars(d)
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(extensions):
                    full = os.path.join(root, f)
                    plugins.append({"name": Path(f).stem, "path": full})
    return plugins


class VSTEffect:
    def __init__(self, plugin_path, name=None):
        self.plugin_path = plugin_path
        self.name = name or Path(plugin_path).stem
        self.plugin = None
        self.enabled = True
        self._load()

    def _load(self):
        if not PEDALBOARD_AVAILABLE:
            raise RuntimeError("pedalboard is not installed")
        self.plugin = load_plugin(self.plugin_path)

    def get_parameters(self):
        if self.plugin is None:
            return []
        params = []
        try:
            if hasattr(self.plugin, 'parameters') and isinstance(self.plugin.parameters, dict):
                for pname, param in self.plugin.parameters.items():
                    try:
                        val = getattr(self.plugin, pname, None)
                        if val is None:
                            continue
                        units = getattr(param, 'units', '')
                        display = pname.replace('_', ' ').title()
                        if units:
                            display = f"{display} ({units})"
                        params.append({
                            "name": pname,
                            "value": float(val) * 100.0,
                            "min": 0.0,
                            "max": 100.0,
                            "label": display,
                        })
                    except Exception:
                        pass
            else:
                for name in dir(self.plugin):
                    if name.startswith("_"):
                        continue
                    try:
                        val = getattr(self.plugin, name)
                        if isinstance(val, (int, float)):
                            display = name.replace('_', ' ').title()
                            params.append({"name": name, "value": val * 100.0, "min": 0.0, "max": 100.0, "label": display})
                    except Exception:
                        pass
        except Exception:
            pass
        return params

    def set_parameter(self, name, value):
        if self.plugin is not None:
            setattr(self.plugin, name, value)

    def get_parameter_value(self, name):
        if self.plugin is not None:
            return getattr(self.plugin, name, None)
        return None

    def process(self, audio, sample_rate=44100, num_channels=2):
        if self.plugin is None or not self.enabled:
            return audio
        try:
            num_frames = len(audio) // num_channels
            interleaved = audio[:num_frames * num_channels]
            deinterleaved = interleaved.reshape(num_frames, num_channels).T
            result = self.plugin(deinterleaved, sample_rate)
            if result.ndim == 2:
                return result.T.reshape(-1).astype(np.float32)
            return audio
        except Exception:
            return audio

    def to_dict(self):
        d = {
            "plugin_path": self.plugin_path,
            "name": self.name,
            "enabled": self.enabled,
            "parameters": {},
        }
        for p in self.get_parameters():
            d["parameters"][p["name"]] = p["value"]
        return d


class VSTEffectChain:
    def __init__(self, buffer_size=512, sample_rate=44100):
        self.effects = []
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._process_callback = None

    def add_effect(self, plugin_path, name=None):
        with self._lock:
            effect = VSTEffect(plugin_path, name)
            self.effects.append(effect)
            return effect

    def remove_effect(self, index):
        with self._lock:
            if 0 <= index < len(self.effects):
                return self.effects.pop(index)
        return None

    def move_effect(self, from_idx, to_idx):
        with self._lock:
            if 0 <= from_idx < len(self.effects) and 0 <= to_idx < len(self.effects):
                eff = self.effects.pop(from_idx)
                self.effects.insert(to_idx, eff)

    def toggle_effect(self, index):
        with self._lock:
            if 0 <= index < len(self.effects):
                self.effects[index].enabled = not self.effects[index].enabled

    def clear(self):
        with self._lock:
            self.effects.clear()

    def process_audio(self, audio):
        with self._lock:
            for effect in self.effects:
                if effect.enabled:
                    audio = effect.process(audio, self.sample_rate)
        return audio

    def make_processor_callback(self):
        def callback(audio):
            return self.process_audio(audio)
        self._process_callback = callback
        return callback

    def get_state(self):
        return {
            "buffer_size": self.buffer_size,
            "sample_rate": self.sample_rate,
            "effects": [e.to_dict() for e in self.effects],
        }

    def load_state(self, state):
        self.clear()
        self.buffer_size = state.get("buffer_size", self.buffer_size)
        self.sample_rate = state.get("sample_rate", self.sample_rate)
        for eff_state in state.get("effects", []):
            path = eff_state.get("plugin_path", "")
            if os.path.exists(path):
                try:
                    eff = self.add_effect(path, eff_state.get("name"))
                    eff.enabled = eff_state.get("enabled", True)
                    for pname, pval in eff_state.get("parameters", {}).items():
                        eff.set_parameter(pname, pval)
                except Exception:
                    pass


def save_chain_config(chain, filepath):
    state = chain.get_state()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_chain_config(chain, filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        state = json.load(f)
    chain.load_state(state)
