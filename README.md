# LWMP
Lightweight MIDI Player for heavy MIDI files/Black MIDI, made with Python and Cython.

### Features:
• Fast MIDI loading, whilst optimized for RAM usage  
• Piano Roll with custom skin support  
• Pre-render function using BASSMIDI  
• Simple GUI for noobies who just want to load and enjoy  
• Graphs for CPU usage and NPS (Notes Per Second)  
• Realtime note counter  
• Support for custom bundled OmniMIDI dlls

# Build instructions
### For Windows
Just launch build.bat, and all dependencies and compilation happens automatically.  

### For Linux (tested, cython compilation broken as of now!):
1. Download and install all dependencies:  
`pip install --no-warn-script-location numpy cython setuptools pyopengl pyopengl_accelerate pygame dearpygui`  
*If you get an error for externally-managed-environment, include* `--break-system-packages`

2. Clone git of repository:  
`git clone https://github.com/mappazinho/LWMP.git`

4. Compile required Cython files:  
`python setup.py build_ext --inplace`

5. Launch LWMP:  
`python MIDI/midiplayer_dpg.py`

Requires Python 3.11+ for building!

### ⚠️ Still in beta stage, lots of bugs still needs fixing and features are somewhat incomplete.

## Credits
- [BASSMIDI](https://www.un4seen.com/bass.html) libraries by Un4seen for MIDI synthesis and audio streaming.
- [BASSMIDI drivers](https://github.com/kode54/BASSMIDI-Driver) by kode54
- [KDMAPI](https://pypi.org/project/kdmapi/) for realtime audio playback
- [NumPy](https://numpy.org/) for array processing and playback/parser data handling.
- [Cython](https://cython.org/) for the compiled parser and audio engine modules.
- [pygame](https://www.pygame.org/news) and [PyOpenGL](https://pyopengl.sourceforge.net/) for the piano roll renderer.
- [DearPyGUI](https://github.com/hoffstadt/DearPyGui) for the modern GUI frontend.
- [psutil](https://github.com/giampaolo/psutil) for CPU and memory statistics.
