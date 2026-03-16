# LWMP
Lightweight MIDI Player for heavy MIDI files/Black MIDI, made with Python and Cython.

### Features:
• Fast MIDI loading, whilst optimized for RAM usage  
• Piano Roll with custom skin support  
• Pre-render function using BASSMIDI  
• Simple GUI for noobies who just want to load and enjoy  
• Graphs for CPU usage and NPS (Notes Per Second)  
• Realtime note counter

# Build instructions (for Windows)
Just launch build.bat, and all dependencies and compilation happens automatically.  

### For Linux (not tested as of now):
1. Download and install all dependencies:  
`pip install --no-warn-script-location numpy cython setuptools pyopengl pyopengl_accelerate pygame dearpygui`

2. Compile required Cython files:  
`python setup.py build_ext --inplace`

3. Launch LWMP:  
`python MIDI/midiplayer.py`

Requires Python 3.13+ for building!
