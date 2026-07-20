# LWMP
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/858fae1a-a357-4075-96f6-2cd0e23db111" />

Lightweight MIDI Player for heavy MIDI files/Black MIDI, made with (mostly) Python and Cython.

### Features:
• Fast MIDI loading, whilst optimized for RAM usage  
• Piano Roll with custom skin support and light effects  
• Audio pre-render mode using BASSMIDI  
• Simple GUI for noobies who just want to load and enjoy  
• Graphs for CPU usage and NPS (Notes Per Second)  
• Realtime note counter  
• Support for custom bundled synth (SYNTH.dll) dlls  
• And more!

## Using custom synths:
As of v1.0.5, compiled .EXE builds now support custom synthesizer DLLs as a MIDI backend. To load these DLLs:  

#### NOTE: DLL MUST BE NAMED AS SYNTH.dll, OTHERWISE NOTHING WILL BE SHOWN ON LWMP!  

1. Place synthesizer DLL next to where `LWMP_v1-0-5.exe` is located  
2. Open LWMP, and set audio mode to Custom Synth (Bundled DLL)  
3. Apply audio mode and enjoy!  

# Build instructions

### Prerequisites
Since proprietary audio binaries are not hosted in this repository, you must manually download the required Un4seen BASS binaries for prerender support.

1. Go to the [Un4seen BASS Website](https://www.un4seen.com/).
2. Download the main **BASS** package and the **BASSMIDI** add-on package for your operating system.
3. Extract the archives and place the following files inside `MIDI/bassmidi' (make a new folder called bassmidi):
   * **Windows:** `bass.dll` and `bassmidi.dll`
   * **Linux:** `libbass.so` and `libbassmidi.so`.

### For Windows
**To compile cython modules, you require MSVC build tools which can be downloaded with the Visual Studio Installer**

Just launch build.bat, and all dependencies and compilation happens automatically.  

### For Linux (tested, audio broken as of now!):
1. Download and install all dependencies:  
`pip install --no-warn-script-location numpy cython setuptools pyopengl pyopengl_accelerate pygame dearpygui tqdm`  
*If you get an error for externally-managed-environment, include* `--break-system-packages`

2. Clone git of repository:  
`git clone https://github.com/mappazinho/LWMP.git`

4. Compile required Cython files:  
`python setup.py build_ext --inplace`

5. Launch LWMP:  
`python MIDI/midiplayer.py`

Requires Python 3.11+ for building!

## Credits
- [BASSMIDI](https://www.un4seen.com/bass.html) libraries by Un4seen for MIDI synthesis and audio streaming.
- [BASSMIDI drivers](https://github.com/kode54/BASSMIDI-Driver) by kode54
- [KDMAPI](https://pypi.org/project/kdmapi/) for realtime audio playback
- [NumPy](https://numpy.org/) for array processing and playback/parser data handling.
- [Cython](https://cython.org/) for the compiled parser and audio engine modules.
- [pygame](https://www.pygame.org/news) and [PyOpenGL](https://pyopengl.sourceforge.net/) for the piano roll renderer.
- [DearPyGUI](https://github.com/hoffstadt/DearPyGui) for the modern GUI frontend.
- [psutil](https://github.com/giampaolo/psutil) for CPU and memory statistics.
- [Piano From Above](https://github.com/brian-pantano/PianoFromAbove) for piano roll skin and depth references
- [MPGL](https://github.com/SonoSooS/_MPGL) for VBO/VAO and depth references

  ## Note: This repository **fully** contains AI-generated slop code. I cannot deem myself the true owner of this code, considering the ethical implications with **fully** AI-generated code. This project was purely made for fun, so have at it 👍
