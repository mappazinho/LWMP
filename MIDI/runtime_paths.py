import os
import sys


def _unique_dirs(candidates):
    seen = set()
    result = []
    for candidate in candidates:
        if not candidate:
            continue
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        result.append(os.path.abspath(candidate))
    return result


def get_runtime_search_dirs(module_file):
    module_dir = os.path.dirname(os.path.abspath(module_file))
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else module_dir
    meipass_dir = getattr(sys, "_MEIPASS", None)

    return _unique_dirs(
        [
            os.path.join(module_dir, "bassmidi"),
            module_dir,
            os.path.join(exe_dir, "MIDI", "bassmidi"),
            os.path.join(exe_dir, "MIDI"),
            os.path.join(exe_dir, "bassmidi"),
            exe_dir,
            os.path.join(meipass_dir, "MIDI", "bassmidi") if meipass_dir else None,
            os.path.join(meipass_dir, "MIDI") if meipass_dir else None,
            os.path.join(meipass_dir, "bassmidi") if meipass_dir else None,
            meipass_dir,
        ]
    )


def _get_bass_library_name_sets():
    if sys.platform.startswith("win"):
        return [("bass.dll", "bassmidi.dll")]
    if sys.platform == "darwin":
        return [
            ("libbass.dylib", "libbassmidi.dylib"),
            ("libbass.so", "libbassmidi.so"),
        ]
    return [
        ("libbass.so", "libbassmidi.so"),
        ("bass.so", "bassmidi.so"),
    ]


def resolve_bass_library_paths(module_file):
    for base_dir in get_runtime_search_dirs(module_file):
        for bass_name, bassmidi_name in _get_bass_library_name_sets():
            bass_path = os.path.join(base_dir, bass_name)
            bassmidi_path = os.path.join(base_dir, bassmidi_name)
            if os.path.exists(bass_path) and os.path.exists(bassmidi_path):
                return bass_path, bassmidi_path, base_dir
    return None, None, None


def add_dll_search_dir(directory):
    if not directory or not os.path.isdir(directory):
        return None
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return None
    try:
        return add_dir(directory)
    except OSError:
        return None
