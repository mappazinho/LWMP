import ctypes
import os
import sys
import threading
import time

BASS_OK = 0
BASS_ERROR_ENDED = 27
BASS_CONFIG_BUFFER = 0
BASS_CONFIG_UPDATEPERIOD = 1
BASS_CONFIG_FLOATDSP = 6
BASS_DEVICE_DEFAULT = 0
BASS_DEVICE_FREQ = 1024 

BASS_STREAM_AUTOFREE = 0x40000
BASS_STREAM_DECODE = 0x200000
BASS_SAMPLE_FLOAT = 0x100
BASS_ATTRIB_VOL = 2
BASS_ATTRIB_MIDI_VOICES = 0x12003
BASS_MIDI_DECAYEND = 0x1000
BASS_MIDI_NOFX = 0x2000
BASS_MIDI_DECAYSEEK = 0x4000
BASS_MIDI_SINCINTER = 0x800000

BASS_POS_BYTE = 0
STREAMPROC_PUSH = -1
BASS_FILEPOS_BUFFER = 5

BASS_MIDI_EVENT_NOTE = 1
BASS_MIDI_EVENT_PITCH = 4
BASS_MIDI_EVENT_PITCHRANGE = 5
BASS_MIDI_EVENT_NOTESOFF = 18
BASS_MIDI_FONT_FORCELOAD = 0x4000000
BASS_MIDI_SF_DEFAULT = 0
BASS_UNICODE = 0x80000000

_script_dir = os.path.dirname(os.path.abspath(__file__))
_bass_dir = os.path.join(_script_dir, "bassmidi")

try:
    if os.path.exists(os.path.join(_bass_dir, "bass.dll")):
        bass_path = os.path.join(_bass_dir, "bass.dll")
        bassmidi_path = os.path.join(_bass_dir, "bassmidi.dll")
    else:
        bass_path = os.path.join(_script_dir, "bass.dll")
        bassmidi_path = os.path.join(_script_dir, "bassmidi.dll")

    print(f"Loading BASS from: {bass_path}")
    bass = ctypes.cdll.LoadLibrary(bass_path)
    bassmidi = ctypes.cdll.LoadLibrary(bassmidi_path)
except Exception as e:
    print(f"Error loading BASS libraries: {e}")
    bass = None
    bassmidi = None

HSTREAM = ctypes.c_uint
DWORD = ctypes.c_uint
BOOL = ctypes.c_int
QWORD = ctypes.c_ulonglong

class BASS_MIDI_FONT(ctypes.Structure):
    _fields_ = [
        ("font", HSTREAM),
        ("preset", ctypes.c_int),
        ("bank", ctypes.c_int)
    ]

if bass:
    bass.BASS_Init.argtypes = [ctypes.c_int, DWORD, DWORD, ctypes.c_void_p, ctypes.c_void_p]
    bass.BASS_Init.restype = BOOL
    bass.BASS_Free.restype = BOOL
    bass.BASS_SetConfig.argtypes = [DWORD, DWORD]
    bass.BASS_SetConfig.restype = BOOL
    bass.BASS_ErrorGetCode.restype = ctypes.c_int
    
    bass.BASS_StreamCreate.argtypes = [DWORD, DWORD, DWORD, ctypes.c_void_p, ctypes.c_void_p]
    bass.BASS_StreamCreate.restype = HSTREAM
    bass.BASS_StreamFree.argtypes = [HSTREAM]
    bass.BASS_StreamFree.restype = BOOL
    
    bass.BASS_StreamPutData.argtypes = [HSTREAM, ctypes.c_void_p, DWORD]
    bass.BASS_StreamPutData.restype = DWORD
    bass.BASS_StreamGetFilePosition.argtypes = [HSTREAM, DWORD]
    bass.BASS_StreamGetFilePosition.restype = QWORD
    
    bass.BASS_ChannelPlay.argtypes = [HSTREAM, BOOL]
    bass.BASS_ChannelPlay.restype = BOOL
    bass.BASS_ChannelPause.argtypes = [HSTREAM]
    bass.BASS_ChannelPause.restype = BOOL
    bass.BASS_ChannelStop.argtypes = [HSTREAM]
    bass.BASS_ChannelStop.restype = BOOL
    bass.BASS_ChannelIsActive.argtypes = [HSTREAM]
    bass.BASS_ChannelIsActive.restype = DWORD
    
    bass.BASS_ChannelSetPosition.argtypes = [HSTREAM, QWORD, DWORD]
    bass.BASS_ChannelSetPosition.restype = BOOL
    bass.BASS_ChannelGetPosition.argtypes = [HSTREAM, DWORD]
    bass.BASS_ChannelGetPosition.restype = QWORD
    
    bass.BASS_ChannelGetData.argtypes = [HSTREAM, ctypes.c_void_p, DWORD]
    bass.BASS_ChannelGetData.restype = DWORD
    
    bass.BASS_ChannelSeconds2Bytes.argtypes = [HSTREAM, ctypes.c_double]
    bass.BASS_ChannelSeconds2Bytes.restype = QWORD
    bass.BASS_ChannelBytes2Seconds.argtypes = [HSTREAM, QWORD]
    bass.BASS_ChannelBytes2Seconds.restype = ctypes.c_double
    bass.BASS_ChannelSetAttribute.argtypes = [HSTREAM, DWORD, ctypes.c_float]
    bass.BASS_ChannelSetAttribute.restype = BOOL

if bassmidi:
    bassmidi.BASS_MIDI_StreamCreate.argtypes = [DWORD, DWORD, DWORD]
    bassmidi.BASS_MIDI_StreamCreate.restype = HSTREAM
    bassmidi.BASS_MIDI_StreamEvent.argtypes = [HSTREAM, DWORD, DWORD, DWORD]
    bassmidi.BASS_MIDI_StreamEvent.restype = BOOL
    
    bassmidi.BASS_MIDI_FontInit.argtypes = [ctypes.c_void_p, DWORD]
    bassmidi.BASS_MIDI_FontInit.restype = HSTREAM
    USE_FONTLOAD = False

    bassmidi.BASS_MIDI_FontFree.argtypes = [HSTREAM]
    bassmidi.BASS_MIDI_FontFree.restype = BOOL
    
    bassmidi.BASS_MIDI_StreamSetFonts.argtypes = [HSTREAM, ctypes.POINTER(BASS_MIDI_FONT), DWORD]
    bassmidi.BASS_MIDI_StreamSetFonts.restype = BOOL


class BassMidiEngine:
    def __init__(self, audio_cfg, soundfont_path=None, buffering=False, debug=False):
        self.is_initialized = False
        self.midi_stream = 0
        self.decode_stream = 0
        self.playback_stream = 0
        self.soundfont = 0
        self.buffering_enabled = buffering
        self.debug_mode = bool(debug)
        self.total_bytes_pushed = 0
        self.volume_level = 1.0
        
        if not bass or not bassmidi:
            raise Exception("BASS libraries not loaded.")

        bass.BASS_SetConfig(BASS_CONFIG_FLOATDSP, 1)
        bass.BASS_SetConfig(BASS_CONFIG_UPDATEPERIOD, 1)
        bass.BASS_SetConfig(BASS_CONFIG_BUFFER, 300)
        
        if not bass.BASS_Init(-1, 44100, 0, None, None):
            err = bass.BASS_ErrorGetCode()
            if err != 14:
                print(f"BASS_Init failed: {err}")
                return
        
        self.is_initialized = True
        
        if self.buffering_enabled:
            flags = BASS_STREAM_DECODE | BASS_SAMPLE_FLOAT | BASS_MIDI_SINCINTER
            self.decode_stream = bassmidi.BASS_MIDI_StreamCreate(16, flags, 44100)
            if not self.decode_stream:
                print(f"Decode Stream Create Failed: {bass.BASS_ErrorGetCode()}")
                return

            self.playback_stream = bass.BASS_StreamCreate(44100, 2, BASS_SAMPLE_FLOAT, ctypes.c_void_p(STREAMPROC_PUSH), None)
            if not self.playback_stream:
                print(f"Playback Stream Create Failed: {bass.BASS_ErrorGetCode()}")
                return
            bass.BASS_ChannelSetAttribute(self.playback_stream, BASS_ATTRIB_VOL, ctypes.c_float(self.volume_level))
            
            self.midi_stream = self.playback_stream
        else:
            flags = BASS_STREAM_AUTOFREE | BASS_SAMPLE_FLOAT | BASS_MIDI_SINCINTER
            self.midi_stream = bassmidi.BASS_MIDI_StreamCreate(16, flags, 44100)
            if not self.midi_stream:
                print(f"MIDI Stream Create Failed: {bass.BASS_ErrorGetCode()}")
                return
            bass.BASS_ChannelSetAttribute(self.midi_stream, BASS_ATTRIB_VOL, ctypes.c_float(self.volume_level))

        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        self.load_soundfont(target, soundfont_path)
        
        print("BassMidiEngine initialized.")

    def load_soundfont(self, stream, path):
        if not path or not os.path.exists(path): 
            print("No SoundFont found/provided.")
            return

        try:
            c_path = path.encode('mbcs')
        except:
            c_path = path.encode('utf-8')
            
        print(f"[DEBUG] Loading SoundFont: {c_path}")
        
        pre_err = bass.BASS_ErrorGetCode()
        if pre_err: print(f"[DEBUG] Pre-existing BASS error: {pre_err}")

        if USE_FONTLOAD:
            print("[DEBUG] Using BASS_MIDI_FontLoad")
            self.soundfont = bassmidi.BASS_MIDI_FontLoad(c_path, 0)
        else:
            print("[DEBUG] Using BASS_MIDI_FontInit")
            self.soundfont = bassmidi.BASS_MIDI_FontInit(c_path, 0)
            
        if not self.soundfont:
            err = bass.BASS_ErrorGetCode()
            print(f"Font load failed: {err}")
            return

        font_struct = BASS_MIDI_FONT()
        font_struct.font = self.soundfont
        font_struct.preset = -1
        font_struct.bank = 0
        
        if not bassmidi.BASS_MIDI_StreamSetFonts(stream, ctypes.byref(font_struct), 1):
            print(f"SetFonts failed: {bass.BASS_ErrorGetCode()}")

    def send_raw_event(self, event, param):
        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if not target: return
        
        status = event & 0xFF
        chan = status & 0x0F
        cmd = status & 0xF0
        
        if cmd == 0x90 or cmd == 0x80:
            bassmidi.BASS_MIDI_StreamEvent(target, chan, BASS_MIDI_EVENT_NOTE, param)
        elif cmd == 0xE0:
            d1 = param & 0xFF
            d2 = (param >> 8) & 0xFF
            val = d1 | (d2 << 7)
            bassmidi.BASS_MIDI_StreamEvent(target, chan, BASS_MIDI_EVENT_PITCH, val)
        elif cmd == 0xB0:
            controller = param & 0xFF
            value = (param >> 8) & 0xFF
            bassmidi.BASS_MIDI_StreamEvent(target, chan, controller, value)

    def send_all_notes_off(self):
        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if not target: return
        for c in range(16):
            bassmidi.BASS_MIDI_StreamEvent(target, c, BASS_MIDI_EVENT_NOTESOFF, 0)

    def set_pitch_bend_range(self, semitones):
        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if not target:
            return
        value = max(0, min(int(semitones), 127))
        for channel in range(16):
            bassmidi.BASS_MIDI_StreamEvent(target, channel, BASS_MIDI_EVENT_PITCHRANGE, value)

    def render_forward(self, seconds):
        if not self.buffering_enabled or not self.decode_stream: return 0.0
        
        bytes_needed = bass.BASS_ChannelSeconds2Bytes(self.decode_stream, seconds)
        if bytes_needed <= 0: return 0.0
        
        chunk_size = 65536
        buf = ctypes.create_string_buffer(chunk_size)
        total_written = 0
        remaining = bytes_needed
        
        while remaining > 0:
            to_read = min(remaining, chunk_size)
            read = bass.BASS_ChannelGetData(self.decode_stream, buf, to_read)
            
            if read == 0xFFFFFFFF:
                break
            if read == 0:
                break
                
            queued = bass.BASS_StreamPutData(self.playback_stream, buf, read)
            if queued == 0xFFFFFFFF:
                 err = bass.BASS_ErrorGetCode()
                 print(f"[DEBUG] BASS_StreamPutData failed: {err} (Buffer Full?)")
                 break
            
            # BASS_StreamPutData returns the queue level on success, not the
            # number of bytes written from this call.
            total_written += read
            remaining -= read
            
        self.total_bytes_pushed += total_written
        return bass.BASS_ChannelBytes2Seconds(self.playback_stream, total_written)

    def get_buffer_level(self):
        """Returns seconds of audio currently buffered in the playback stream."""
        if not self.buffering_enabled or not self.playback_stream: return 0.0
        
        current_pos_bytes = bass.BASS_ChannelGetPosition(self.playback_stream, BASS_POS_BYTE)
        if current_pos_bytes == 0xFFFFFFFFFFFFFFFF: return 0.0
        
        bytes_buffered = self.total_bytes_pushed - current_pos_bytes
        if bytes_buffered < 0: bytes_buffered = 0 
        
        return bass.BASS_ChannelBytes2Seconds(self.playback_stream, bytes_buffered)

    def get_position_seconds(self):
        if not self.playback_stream: return 0.0
        b = bass.BASS_ChannelGetPosition(self.playback_stream, BASS_POS_BYTE)
        if b == 0xFFFFFFFFFFFFFFFF: return 0.0
        return bass.BASS_ChannelBytes2Seconds(self.playback_stream, b)

    def is_active(self):
        if not self.midi_stream: return False
        return bass.BASS_ChannelIsActive(self.midi_stream) == 1 # BASS_ACTIVE_PLAYING

    def play(self):
        if self.midi_stream: bass.BASS_ChannelPlay(self.midi_stream, False)

    def pause(self):
        if self.midi_stream: bass.BASS_ChannelPause(self.midi_stream)

    def stop(self):
        if self.midi_stream: 
            bass.BASS_ChannelStop(self.midi_stream)
            bass.BASS_ChannelSetPosition(self.midi_stream, 0, BASS_POS_BYTE)
            self.total_bytes_pushed = 0
        if self.buffering_enabled and self.decode_stream:
            bass.BASS_ChannelSetPosition(self.decode_stream, 0, BASS_POS_BYTE)

    def shutdown(self):
        if self.midi_stream: bass.BASS_StreamFree(self.midi_stream)
        if self.decode_stream: bass.BASS_StreamFree(self.decode_stream)
        if self.soundfont: bassmidi.BASS_MIDI_FontFree(self.soundfont)
        if self.is_initialized: bass.BASS_Free()
    
    def test_piano_sweep(self):
        print("Python fallback test_piano_sweep not implemented")

    def set_volume(self, volume):
        self.volume_level = float(volume)
        target = self.playback_stream if self.buffering_enabled else self.midi_stream
        if target:
            bass.BASS_ChannelSetAttribute(target, BASS_ATTRIB_VOL, ctypes.c_float(self.volume_level))

    def set_voices(self, voices):
        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if target:
            bass.BASS_ChannelSetAttribute(target, BASS_ATTRIB_MIDI_VOICES, ctypes.c_float(float(voices)))
