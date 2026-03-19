# filename: setup.py
# [CHANGED] This setup file builds ALL THREE Cython modules
# and enables C++ support for the parser.

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sys

# --- Define the 'midi_parser_cython' module ---
ext_midi_parser = Extension(
    "PARSER.midi_parser_cython", # Output name is PARSER.midi_parser_cython
    ["PARSER/midi_parser_cython.pyx"], # Source is in PARSER/
    include_dirs=[numpy.get_include(), "PARSER"],
    language="c++"  # Enable C++ mode
)

# --- Define the 'midi_engine_cython' module ---
ext_midi_engine = Extension(
    "MIDI.midi_engine_cython", # Output name is MIDI.midi_engine_cython
    ["MIDI/midi_engine_cython.pyx"], # Source is in MIDI/
    include_dirs=[numpy.get_include(), "MIDI"],
    libraries=["dl"] if sys.platform.startswith("linux") else [],
    # No language="c++" needed, it uses C
    # Linux needs libdl for dlopen/dlsym; Windows/macOS do not.
)

ext_modules = [ext_midi_parser, ext_midi_engine]

setup(
    name="Fast MIDI Tools",
    packages=["PARSER", "MIDI"],
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[numpy.get_include()]
)
