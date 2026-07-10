"""Backward-compatible re-export wrapper.

The actual code now lives in the piano/ package.
This file exists so that any code doing ``from pianoroll import ...``
continues to work without changes.
"""

from piano.pianoroll import PianoRoll
from piano.skin_browser import SkinBrowser
from piano.skin_utils import (
    _SKIN_ROOT,
    _SKIN_DIR,
    _COLORS_XML_PATH,
    _resolve_skin_dir,
    _list_available_skins,
)
from piano.note_utils import RENDER_NOTE_DTYPE
