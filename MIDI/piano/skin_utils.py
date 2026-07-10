import os
import sys
import zipfile
import numpy as np

from config import load_config as _load_skin_config

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNTIME_ROOT = getattr(sys, "_MEIPASS", _SCRIPT_DIR)
_EXE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else _SCRIPT_DIR
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)


def _first_existing_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else ""


def _ensure_external_skin_dir():
    import shutil
    if not getattr(sys, "frozen", False):
        return
    bundled_skin_dir = os.path.join(_RUNTIME_ROOT, "skin")
    external_skin_dir = os.path.join(_EXE_DIR, "skin")
    if not os.path.isdir(bundled_skin_dir):
        return
    try:
        os.makedirs(external_skin_dir, exist_ok=True)
        for entry in os.listdir(bundled_skin_dir):
            if entry.startswith("."):
                continue
            src = os.path.join(bundled_skin_dir, entry)
            dst = os.path.join(external_skin_dir, entry)
            if os.path.exists(dst):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    except Exception as e:
        print(f"Failed to prepare external skin directory '{external_skin_dir}': {e}")


def _resolve_skin_dir(skin_name, skin_root):
    """Resolve a skin name to a directory path.
    Supports folder skins (skin_root/<name>/) and zip skins (skin_root/<name>.zip).
    Returns the resolved directory path, or None if not found."""
    folder_path = os.path.join(skin_root, skin_name)
    if os.path.isdir(folder_path):
        return folder_path
    zip_path = os.path.join(skin_root, skin_name + ".zip")
    if os.path.isfile(zip_path):
        cache_dir = os.path.join(skin_root, ".cache", skin_name)
        marker = os.path.join(cache_dir, ".extracted")
        if not os.path.exists(marker):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(cache_dir)
                with open(marker, "w") as mf:
                    mf.write(str(os.path.getmtime(zip_path)))
                print(f"Extracted skin '{skin_name}' from zip to cache.")
            except Exception as e:
                print(f"Failed to extract skin zip '{zip_path}': {e}")
                return None
        return cache_dir
    return None


def _list_available_skins(skin_root):
    """List all available skin names in the skin root."""
    skins = []
    if not os.path.isdir(skin_root):
        return skins
    for entry in os.listdir(skin_root):
        if entry.startswith("."):
            continue
        full = os.path.join(skin_root, entry)
        if os.path.isdir(full):
            skins.append(entry)
        elif entry.lower().endswith(".zip"):
            skins.append(entry[:-4])
    return sorted(skins)


def _load_colors_from_xml(filepath):
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        colors = []
        for color_elem in root.findall('Color'):
            r = int(color_elem.get('R')) / 255.0
            g = int(color_elem.get('G')) / 255.0
            b = int(color_elem.get('B')) / 255.0
            colors.append([r, g, b])
        if not colors:
            raise ValueError("No colors in XML")
        num_colors = len(colors)
        full_color_list = []
        for i in range(128):
            full_color_list.append(colors[i % num_colors])
        return np.array(full_color_list, dtype=np.float32)
    except (ET.ParseError, FileNotFoundError, ValueError) as e:
        print(f"Error loading colors from {filepath}: {e}. Using default rainbow palette.")
        rainbow = [
            [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.5, 1.0],
            [0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.5],
            [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0], [0.8, 0.8, 0.8]
        ]
        full_color_list = []
        for i in range(128):
            full_color_list.append(rainbow[i % 16])
        return np.array(full_color_list, dtype=np.float32)


_ensure_external_skin_dir()
_SKIN_ROOT = _first_existing_path(
    os.path.join(_EXE_DIR, "skin"),
    os.path.join(_PROJECT_DIR, "skin"),
    os.path.join(_SCRIPT_DIR, "skin"),
    os.path.join(_RUNTIME_ROOT, "skin"),
)

_skin_cfg = _load_skin_config()
_skin_name = _skin_cfg.get("visualizer", {}).get("skin_name", "default")
_SKIN_DIR = _resolve_skin_dir(_skin_name, _SKIN_ROOT)
if _SKIN_DIR is None:
    _SKIN_DIR = _resolve_skin_dir("default", _SKIN_ROOT)
if _SKIN_DIR is None:
    _SKIN_DIR = _SKIN_ROOT
    print(f"Warning: Skin '{_skin_name}' not found, using skin root.")
else:
    print(f"Loaded skin: {_skin_name} from {_SKIN_DIR}")

_COLORS_XML_PATH = _first_existing_path(
    os.path.join(_SCRIPT_DIR, "MIDI", "colors.xml"),
    os.path.join(_PROJECT_DIR, "MIDI", "colors.xml"),
    os.path.join(_RUNTIME_ROOT, "MIDI", "colors.xml"),
    os.path.join(_RUNTIME_ROOT, "colors.xml"),
)
