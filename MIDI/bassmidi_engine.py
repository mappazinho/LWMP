import ctypes
import os
import sys
import threading
import time
from runtime_paths import add_dll_search_dir, resolve_bass_library_paths

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
BASS_ATTRIB_FREQ = 1
BASS_ATTRIB_VOL = 2
BASS_ATTRIB_MIDI_VOICES = 0x12003
BASS_MIDI_DECAYEND = 0x1000
BASS_MIDI_NOFX = 0x2000
BASS_MIDI_DECAYSEEK = 0x4000
BASS_MIDI_SINCINTER = 0x800000

BASS_POS_BYTE = 0
STREAMPROC_PUSH = -1
BASS_FILEPOS_BUFFER = 5
BASS_MIDI_EVENTS_RAW = 0x10000

BASS_MIDI_EVENT_NOTE = 1
BASS_MIDI_EVENT_PITCH = 4
BASS_MIDI_EVENT_PITCHRANGE = 5
BASS_MIDI_EVENT_NOTESOFF = 18
BASS_MIDI_FONT_FORCELOAD = 0x4000000
BASS_MIDI_SF_DEFAULT = 0
BASS_UNICODE = 0x80000000

_script_dir = os.path.dirname(os.path.abspath(__file__))
_dll_dir_handle = None

try:
    bass_path, bassmidi_path, bass_dir = resolve_bass_library_paths(__file__)
    if not bass_path or not bassmidi_path:
        raise FileNotFoundError("bass.dll or bassmidi.dll not found in runtime search paths")

    print(f"Loading BASS from: {bass_path}")
    _dll_dir_handle = add_dll_search_dir(bass_dir)
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
    bassmidi.BASS_MIDI_StreamEvents.argtypes = [HSTREAM, DWORD, ctypes.c_void_p, DWORD]
    bassmidi.BASS_MIDI_StreamEvents.restype = DWORD
    bassmidi.BASS_MIDI_StreamEvent.argtypes = [HSTREAM, DWORD, DWORD, DWORD]
    bassmidi.BASS_MIDI_StreamEvent.restype = BOOL
    
    bassmidi.BASS_MIDI_FontInit.argtypes = [ctypes.c_void_p, DWORD]
    bassmidi.BASS_MIDI_FontInit.restype = HSTREAM
    USE_FONTLOAD = False

    bassmidi.BASS_MIDI_FontLoad.argtypes = [HSTREAM, ctypes.c_int, ctypes.c_int]
    bassmidi.BASS_MIDI_FontLoad.restype = BOOL

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
        self.playback_speed = 1.0
        self.normal_voice_limit = 512
        self.emergency_voice_limit = 96
        self.emergency_velocity = 100
        self.emergency_recovery_enabled = False
        self.soundfont_preset = -1
        self.soundfont_bank = 0
        
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
            bass.BASS_ChannelSetAttribute(self.decode_stream, BASS_ATTRIB_MIDI_VOICES, ctypes.c_float(float(self.normal_voice_limit)))
            bass.BASS_ChannelSetAttribute(self.playback_stream, BASS_ATTRIB_VOL, ctypes.c_float(self.volume_level))
            
            self.midi_stream = self.playback_stream
        else:
            flags = BASS_STREAM_AUTOFREE | BASS_SAMPLE_FLOAT | BASS_MIDI_SINCINTER
            self.midi_stream = bassmidi.BASS_MIDI_StreamCreate(16, flags, 44100)
            if not self.midi_stream:
                print(f"MIDI Stream Create Failed: {bass.BASS_ErrorGetCode()}")
                return
            bass.BASS_ChannelSetAttribute(self.midi_stream, BASS_ATTRIB_MIDI_VOICES, ctypes.c_float(float(self.normal_voice_limit)))
            bass.BASS_ChannelSetAttribute(self.midi_stream, BASS_ATTRIB_VOL, ctypes.c_float(self.volume_level))

        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        self.load_soundfont(target, soundfont_path)
        
        print("BassMidiEngine initialized.")

        self.event_buffer = []
        self.event_count = 0
        self.current_event_idx = 0
        self.simulated_time = 0.0

    def load_soundfont(self, stream, path):
        if not path or not os.path.exists(path): 
            print("No SoundFont found/provided.")
            return

        ext = os.path.splitext(path)[1].lower()
        c_path = ctypes.create_unicode_buffer(path)
        flags = BASS_UNICODE
        print(f"[DEBUG] Loading SoundFont: {path} (ext={ext})")
        
        pre_err = bass.BASS_ErrorGetCode()
        if pre_err: print(f"[DEBUG] Pre-existing BASS error: {pre_err}")

        if USE_FONTLOAD:
            print("[DEBUG] Using BASS_MIDI_FontLoad")
            self.soundfont = bassmidi.BASS_MIDI_FontLoad(c_path, flags)
        else:
            print("[DEBUG] Using BASS_MIDI_FontInit")
            self.soundfont = bassmidi.BASS_MIDI_FontInit(c_path, flags)
            
        if not self.soundfont:
            err = bass.BASS_ErrorGetCode()
            print(f"Font load failed: {err}")
            return

        if ext == ".sfz":
            self.soundfont_preset = 0
            self.soundfont_bank = 0
            bassmidi.BASS_MIDI_FontLoad(self.soundfont, 0, 0)
        else:
            self.soundfont_preset = -1
            self.soundfont_bank = 0

        font_struct = BASS_MIDI_FONT()
        font_struct.font = self.soundfont
        font_struct.preset = self.soundfont_preset
        font_struct.bank = self.soundfont_bank
        
        if not bassmidi.BASS_MIDI_StreamSetFonts(stream, ctypes.byref(font_struct), 1):
            print(f"SetFonts failed: {bass.BASS_ErrorGetCode()}")

    def upload_events(self, times, statuses, params):
        if not self.buffering_enabled:
            return
        self.event_buffer = list(zip(times.tolist(), statuses.tolist(), params.tolist()))
        self.event_count = len(self.event_buffer)
        self.current_event_idx = 0
        self.simulated_time = 0.0
        self.total_bytes_pushed = 0
        if self.playback_stream:
            bass.BASS_ChannelSetPosition(self.playback_stream, 0, BASS_POS_BYTE)
        if self.decode_stream:
            bass.BASS_ChannelSetPosition(self.decode_stream, 0, BASS_POS_BYTE)

    def set_current_time(self, seconds):
        self.simulated_time = float(seconds)
        idx = 0
        while idx < self.event_count and self.event_buffer[idx][0] < self.simulated_time:
            idx += 1
        self.current_event_idx = idx

    def send_raw_event(self, event, param):
        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if not target: return
        
        status = event & 0xFF
        chan = status & 0x0F
        cmd = status & 0xF0
        
        if cmd == 0x90 or cmd == 0x80:
            if cmd == 0x90 and self.buffering_enabled and self.emergency_recovery_enabled:
                pitch = param & 0xFF
                velocity = (param >> 8) & 0xFF
                if velocity > 0:
                    param = pitch | (int(self.emergency_velocity) << 8)
            bassmidi.BASS_MIDI_StreamEvent(target, chan, BASS_MIDI_EVENT_NOTE, param)
        elif cmd == 0xE0:
            d1 = param & 0xFF
            d2 = (param >> 8) & 0xFF
            val = d1 | (d2 << 7)
            bassmidi.BASS_MIDI_StreamEvent(target, chan, BASS_MIDI_EVENT_PITCH, val)
        elif cmd == 0xB0:
            controller = param & 0xFF
            value = (param >> 8) & 0xFF
            raw_event = (ctypes.c_ubyte * 3)(status, controller, value)
            bassmidi.BASS_MIDI_StreamEvents(target, BASS_MIDI_EVENTS_RAW, raw_event, 3)
        elif cmd == 0xC0:
            program = param & 0x7F
            raw_event = (ctypes.c_ubyte * 2)(status, program)
            bassmidi.BASS_MIDI_StreamEvents(target, BASS_MIDI_EVENTS_RAW, raw_event, 2)

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

    def render_pcm_chunk(self, seconds):
        if not self.buffering_enabled or not self.decode_stream or not hasattr(self, "event_buffer"):
            return b""
        if seconds <= 0.0:
            return b""

        target_end = self.simulated_time + float(seconds)
        chunks = []
        event_epsilon = 1.0 / 44100.0

        while self.simulated_time < target_end:
            while self.current_event_idx < self.event_count:
                event_time, status, param = self.event_buffer[self.current_event_idx]
                if event_time <= self.simulated_time + event_epsilon:
                    self.send_raw_event(int(status), int(param))
                    self.current_event_idx += 1
                else:
                    break

            next_event_time = target_end
            if self.current_event_idx < self.event_count:
                next_event_time = min(next_event_time, float(self.event_buffer[self.current_event_idx][0]))

            segment_seconds = next_event_time - self.simulated_time
            if segment_seconds <= 0.0:
                if target_end - self.simulated_time <= event_epsilon:
                    break
                segment_seconds = min(event_epsilon, target_end - self.simulated_time)
            elif segment_seconds < event_epsilon:
                segment_seconds = min(event_epsilon, target_end - self.simulated_time)

            bytes_needed = bass.BASS_ChannelSeconds2Bytes(self.decode_stream, segment_seconds)
            if bytes_needed <= 0:
                break

            chunk_len = int(bytes_needed)
            buf = ctypes.create_string_buffer(chunk_len)
            read = bass.BASS_ChannelGetData(self.decode_stream, buf, chunk_len)
            if read == 0xFFFFFFFF:
                break
            if read == 0:
                break
            chunks.append(buf.raw[:read])
            read_seconds = bass.BASS_ChannelBytes2Seconds(self.decode_stream, read)
            if read_seconds <= 0.0:
                break
            self.simulated_time += read_seconds

        return b"".join(chunks)

    def set_volume(self, volume):
        self.volume_level = float(volume)
        target = self.playback_stream if self.buffering_enabled else self.midi_stream
        if target:
            bass.BASS_ChannelSetAttribute(target, BASS_ATTRIB_VOL, ctypes.c_float(self.volume_level))

    def set_speed(self, speed):
        self.playback_speed = max(0.1, min(float(speed), 4.0))
        target = self.playback_stream if self.buffering_enabled else self.midi_stream
        if target:
            bass.BASS_ChannelSetAttribute(
                target,
                BASS_ATTRIB_FREQ,
                ctypes.c_float(44100.0 * self.playback_speed),
            )

    def set_voices(self, voices):
        self.normal_voice_limit = int(voices)
        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if target:
            effective = self.emergency_voice_limit if self.emergency_recovery_enabled else self.normal_voice_limit
            bass.BASS_ChannelSetAttribute(target, BASS_ATTRIB_MIDI_VOICES, ctypes.c_float(float(effective)))

    def set_emergency_recovery(self, enabled):
        self.emergency_recovery_enabled = bool(enabled)
        target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if target:
            effective = self.emergency_voice_limit if self.emergency_recovery_enabled else self.normal_voice_limit
            bass.BASS_ChannelSetAttribute(target, BASS_ATTRIB_MIDI_VOICES, ctypes.c_float(float(effective)))
