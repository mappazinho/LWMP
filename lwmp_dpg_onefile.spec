# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_submodules


PROJECT_ROOT = os.path.abspath(os.getcwd())
MIDI_DIR = os.path.join(PROJECT_ROOT, "MIDI")
PARSER_DIR = os.path.join(PROJECT_ROOT, "PARSER")
SKIN_DIR = os.path.join(PROJECT_ROOT, "skin")

datas = [
    (SKIN_DIR, "skin"),
    (os.path.join(MIDI_DIR, "colors.xml"), "MIDI"),
]

binaries = [
    (os.path.join(MIDI_DIR, "bassmidi", "bass.dll"), os.path.join("MIDI", "bassmidi")),
    (os.path.join(MIDI_DIR, "bassmidi", "bassmidi.dll"), os.path.join("MIDI", "bassmidi")),
    (os.path.join(MIDI_DIR, "bassmidi", "bassasio.dll"), os.path.join("MIDI", "bassmidi")),
]

hiddenimports = [
    "midi_parser_cython",
    "midi_engine_cython",
    "bassmidi_engine",
    "player_controller",
    "pianoroll",
    "config",
    "runtime_paths",
]
hiddenimports += collect_submodules("OpenGL")


a = Analysis(
    [os.path.join(MIDI_DIR, "midiplayer.py")],
    pathex=[PROJECT_ROOT, MIDI_DIR, PARSER_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="LWMP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
)
