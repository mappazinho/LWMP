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

### ⚠️ Still in beta stage, lots of bugs still needs fixing and features are somewhat incomplete.

## Credits
- [BASSMIDI](https://www.un4seen.com/bass.html) libraries and [drivers](ttps://github.com/kode54/BASSMIDI-Driver) by Un4seen and kode54 for MIDI synthesis and audio streaming.
- [KDMAPI](https://pypi.org/project/kdmapi/) for realtime audio playback
- [NumPy](https://numpy.org/) for array processing and playback/parser data handling.
- [Cython](https://cython.org/) for the compiled parser and audio engine modules.
- [pygame](https://www.pygame.org/news) and [PyOpenGL](https://pyopengl.sourceforge.net/) for the piano roll renderer.
- [DearPyGUI](https://github.com/hoffstadt/DearPyGui) for the modern GUI frontend.
- [psutil](https://github.com/giampaolo/psutil) for CPU and memory statistics.
