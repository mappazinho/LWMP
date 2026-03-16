# midi_engine.pyx
# cython: language_level=3, cdivision=True

from libc.stdint cimport uint8_t, uint32_t
from cpython cimport bool

import cython
import ctypes
import os
import time as pytime

cimport ctypes
cimport numpy as cnp
import numpy as pynp


# --- BASS/OMNIMIDI API SETUP ---

cpdef void init_numpy():
    """Initializes the NumPy C-API. Must be called once per process."""
    cnp.import_array()
    return

class BassError(Exception):
    """Custom exception for BASS library errors."""
    pass

cdef object bass, bassmidi, bassasio
_script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    bass_path = os.path.join(_script_dir, "bass.dll")
    bassmidi_path = os.path.join(_script_dir, "bassmidi.dll")
    bassasio_path = os.path.join(_script_dir, "bassasio.dll")

    bass = ctypes.cdll.LoadLibrary(bass_path)
    bassmidi = ctypes.cdll.LoadLibrary(bassmidi_path)
    try:
        bassasio = ctypes.cdll.LoadLibrary(bassasio_path)
    except (OSError, FileNotFoundError):
        bassasio = None
except (OSError, FileNotFoundError):
    # This check is primarily for the BASS engine. OmniMIDI will be handled in its own class.
    print("Warning: BASS libraries (bass.dll, bassmidi.dll) not found. Built-in engine will not be available.")
    bass = None
    bassmidi = None
    bassasio = None

# --- BASS Constants & Structures ---
if bass:
    BASS_DEVICE_ASIO = 0x800
    BASS_CONFIG_UPDATEPERIOD = 1
    BASS_CONFIG_ASIO_NOTIFY = 18
    BASS_STREAM_AUTOFREE = 0x40000
    BASS_ATTRIB_MIDI_VOICES = 0x12003
    BASS_ATTRIB_MIDI_VOICES_ACTIVE = 0x12004
    BASS_DEVICE_DEFAULT = 0x10
    BASS_MIDI_EVENT_NOTE = 1
    BASS_DATA_MIDI_ACTIVE = 0x10000000
    BASS_SAMPLE_FLOAT = 0x100
    BASS_CONFIG_MIDI_ASIO = 0x10400
    BASS_STREAM_DECODE = 0x200000
    BASS_CONFIG_FLOATDSP = 6
    BASS_CONFIG_BUFFER = 0
    BASS_MIDI_ASYNC = 0x400000
    BASS_MIDI_EVENTS_ASYNC = 0x40000000

    class BASS_ASIO_DEVICEINFO(ctypes.Structure):
        _fields_ = [
            ('name', ctypes.c_char_p),
            ('driver', ctypes.c_char_p),
        ]

    class BASS_ASIO_CHANNELINFO(ctypes.Structure):
        _fields_ = [
            ('group', ctypes.c_uint),
            ('format', ctypes.c_uint),
            ('name', ctypes.c_char_p),
        ]

# --- Player's Internal Event Structure for BASS ---
cdef struct PlayerEvent:
    double timestamp
    uint32_t event
    uint32_t param

PlayerEvent_dtype = pynp.dtype([
    ('timestamp', pynp.double),
    ('event', pynp.uint32),
    ('param', pynp.uint32)
], align=True)


# --- BUILT-IN BASS ENGINE CLASS ---
cdef class MidiEngine:
    cdef bool is_initialized
    cdef bool use_asio
    cdef unsigned int stream_handle
    cdef public bint is_playing
    cdef int velocity_threshold

    def __cinit__(self, bool use_asio_driver=False, int update_period=5, int device=-1, int buffer_ms=100, int velocity_threshold=0):
        if not bass:
            raise BassError("BASS libraries not found. Cannot initialize built-in engine.")
        
        self.is_initialized = False
        self.use_asio = use_asio_driver and (bassasio is not None)
        self.stream_handle = 0
        self.is_playing = False
        self.velocity_threshold = velocity_threshold

        print(f"[Engine] Initializing BASS. ASIO: {self.use_asio}, Device: {device}, Vel Threshold: {self.velocity_threshold}")
        
        # Configure BASS functions prototypes if they haven't been already
        # This is a safety measure in case the top-level load fails but somehow the class is instantiated.
        self._configure_bass_prototypes()

        cdef bint ok
        cdef int asio_error_code
        bass.BASS_SetConfig(BASS_CONFIG_FLOATDSP, 1)

        if self.use_asio:
            info = BASS_ASIO_DEVICEINFO()
            i = 0
            found_asio = False
            while bassasio.BASS_ASIO_GetDeviceInfo(i, ctypes.byref(info)):
                found_asio = True
                i += 1
            
            ok = bass.BASS_Init(0, 44100, 0, None, None)
            if not ok: self._raise_error("BASS_Init (for ASIO) failed")
            
            ok = bassasio.BASS_ASIO_Init(device, 0)
            if not ok:
                asio_error_code = bassasio.BASS_ASIO_ErrorGetCode()
                self._raise_error(f"BASS_ASIO_Init failed for device #{device} (ASIO Error Code: {asio_error_code}).")
            
            ok = bassasio.BASS_ASIO_SetRate(44100.0)
            if not ok:
                asio_error_code = bassasio.BASS_ASIO_ErrorGetCode()
                self._raise_error(f"BASS_ASIO_SetRate failed (ASIO Error Code: {asio_error_code})")

            bass.BASS_SetConfig(BASS_CONFIG_ASIO_NOTIFY, 1)
        else:
            bass.BASS_SetConfig(BASS_CONFIG_BUFFER, buffer_ms)
            if not bass.BASS_Init(device, 44100, 0, None, None): self._raise_error("BASS_Init failed")

        bass.BASS_SetConfig(BASS_CONFIG_UPDATEPERIOD, update_period)
        self.is_initialized = True
        print("[Engine] BASS initialized successfully.")

    

    cdef _configure_bass_prototypes(self):
        bass.BASS_Init.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        bass.BASS_Init.restype = ctypes.c_bool
        bass.BASS_ErrorGetCode.restype = ctypes.c_int
        bass.BASS_GetCPU.restype = ctypes.c_float
        bass.BASS_ChannelGetData.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        bass.BASS_ChannelGetData.restype = ctypes.c_uint
        bass.BASS_SetConfig.argtypes = [ctypes.c_uint, ctypes.c_uint]
        bass.BASS_SetConfig.restype = ctypes.c_bool
        bass.BASS_ChannelPlay.argtypes = [ctypes.c_uint, ctypes.c_bool]
        bass.BASS_ChannelPlay.restype = ctypes.c_bool
        bass.BASS_ChannelSetAttribute.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_float]
        bass.BASS_ChannelSetAttribute.restype = ctypes.c_bool
        bass.BASS_ChannelGetAttribute.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_float)]
        bass.BASS_ChannelGetAttribute.restype = ctypes.c_bool
        bass.BASS_Free.restype = ctypes.c_bool
        bass.BASS_SetVolume.argtypes = [ctypes.c_float]
        bass.BASS_SetVolume.restype = ctypes.c_bool

        bassmidi.BASS_MIDI_FontInit.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        bassmidi.BASS_MIDI_FontInit.restype = ctypes.c_uint
        bassmidi.BASS_MIDI_StreamCreate.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
        bassmidi.BASS_MIDI_StreamCreate.restype = ctypes.c_uint
        bassmidi.BASS_MIDI_StreamSetFonts.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        bassmidi.BASS_MIDI_StreamSetFonts.restype = ctypes.c_bool
        bassmidi.BASS_MIDI_StreamEvent.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
        bassmidi.BASS_MIDI_StreamEvent.restype = ctypes.c_bool

        if bassasio:
            bassasio.BASS_ASIO_Init.argtypes = [ctypes.c_int, ctypes.c_uint]
            bassasio.BASS_ASIO_Init.restype = ctypes.c_bool
            bassasio.BASS_ASIO_GetDeviceInfo.argtypes = [ctypes.c_uint, ctypes.POINTER(BASS_ASIO_DEVICEINFO)]
            bassasio.BASS_ASIO_GetDeviceInfo.restype = ctypes.c_bool
            bassasio.BASS_ASIO_ChannelEnableBASS.argtypes = [ctypes.c_bool, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool]
            bassasio.BASS_ASIO_ChannelEnableBASS.restype = ctypes.c_bool
            bassasio.BASS_ASIO_Start.argtypes = [ctypes.c_uint, ctypes.c_uint]
            bassasio.BASS_ASIO_Start.restype = ctypes.c_bool
            bassasio.BASS_ASIO_Free.restype = ctypes.c_bool
            bassasio.BASS_ASIO_ErrorGetCode.restype = ctypes.c_int
            bassasio.BASS_ASIO_ChannelGetInfo.argtypes = [ctypes.c_bool, ctypes.c_uint, ctypes.POINTER(BASS_ASIO_CHANNELINFO)]
            bassasio.BASS_ASIO_ChannelGetInfo.restype = ctypes.c_bool
            bassasio.BASS_ASIO_SetRate.argtypes = [ctypes.c_double]
            bassasio.BASS_ASIO_SetRate.restype = ctypes.c_bool
            
    cdef _raise_error(self, str message):
        error_code = bass.BASS_ErrorGetCode()
        raise BassError(f"{message} (BASS Error Code: {error_code})")

    cpdef set_volume(self, float volume):
        """Sets the master volume for the BASS engine."""
        if self.is_initialized:
            # BASS_SetVolume takes a float from 0.0 to 1.0
            bass.BASS_SetVolume(volume)
            

    cpdef float get_cpu_usage(self):
        return bass.BASS_GetCPU()

    cpdef int get_active_voices(self, uint32_t stream_handle):
        if stream_handle == 0:
            return 0
        voices = ctypes.c_float(0.0)
        result = bass.BASS_ChannelGetAttribute(stream_handle, BASS_ATTRIB_MIDI_VOICES_ACTIVE, ctypes.byref(voices))
        if not result:
            return 0
        return int(voices.value)

    cpdef uint32_t get_stream_handle(self):
        return self.stream_handle

    cpdef set_voices(self, int voices):
        if not self.stream_handle: return
        cdef float f_voices = voices
        bass.BASS_ChannelSetAttribute(self.stream_handle, BASS_ATTRIB_MIDI_VOICES, f_voices)

    cpdef send_event_batch(self, list events):
        """Sends a batch of raw MIDI events to the stream."""
        if not self.stream_handle: return
        cdef uint32_t event, param, chan, event_type
        for e in events:
            event, param = e
            chan = event & 0x0F
            event_type = event & 0xF0
            # We only need to handle note events in the batch for now
            if event_type == 0x90 or event_type == 0x80:
                bassmidi.BASS_MIDI_StreamEvent(self.stream_handle, chan, BASS_MIDI_EVENT_NOTE, param)

    cpdef load_soundfont(self, str sf2_path):
        cdef unsigned int font_handle
        cdef uint32_t stream_flags
        cdef uint32_t freq
        c_path = sf2_path.encode('utf-8')

        font_handle = bassmidi.BASS_MIDI_FontInit(c_path, 0)
        if not font_handle: self._raise_error(f"Failed to load SoundFont: {sf2_path}")

        c_font_obj = ctypes.c_uint(font_handle)
        
        if not self.stream_handle:
            stream_flags = BASS_SAMPLE_FLOAT | BASS_MIDI_ASYNC | BASS_MIDI_EVENTS_ASYNC
            freq = 44100
            
            if self.use_asio:
                stream_flags |= BASS_STREAM_DECODE
            else:
                stream_flags |= BASS_STREAM_AUTOFREE
            
            self.stream_handle = bassmidi.BASS_MIDI_StreamCreate(16, stream_flags, freq)
            if not self.stream_handle: self._raise_error("BASS_MIDI_StreamCreate failed")

        if not bassmidi.BASS_MIDI_StreamSetFonts(self.stream_handle, ctypes.byref(c_font_obj), 1):
            self._raise_error("BASS_MIDI_StreamSetFonts failed")
        print("[Engine] SoundFont loaded successfully.")

    cpdef bint start_stream(self):
        if not self.stream_handle: return False
        if self.use_asio:
            if not bassasio.BASS_ASIO_ChannelEnableBASS(False, 0, self.stream_handle, True):
                self._raise_error(f"BASS_ASIO_ChannelEnableBASS failed")
            if not bassasio.BASS_ASIO_Start(0, 0):
                self._raise_error(f"BASS_ASIO_Start failed")
            return True
        else:
            return bass.BASS_ChannelPlay(self.stream_handle, False)

    cpdef send_raw_event(self, uint32_t event, uint32_t param):
        if not self.stream_handle: return
        cdef uint32_t chan = event & 0x0F
        cdef uint32_t event_type = event & 0xF0
        if event_type == 0x90 or event_type == 0x80:
             bassmidi.BASS_MIDI_StreamEvent(self.stream_handle, chan, BASS_MIDI_EVENT_NOTE, param)
        elif event_type == 0xB0:
             bassmidi.BASS_MIDI_StreamEvent(self.stream_handle, chan, event_type, param)
             
    cpdef send_all_notes_off(self):
        for ch in range(16):
            self.send_raw_event(0xB0 | ch, (123 << 8) | 0)

    cpdef shutdown(self):
        """Explicitly shuts down the BASS engine and frees resources."""
        if self.is_initialized:
            print("[Engine] Shutting down BASS resources.")
            # Move the contents of __dealloc__ here
            if self.use_asio: bassasio.BASS_ASIO_Free()
            bass.BASS_Free()
            self.is_initialized = False # Prevent double-freeing

    def __dealloc__(self):
        # __dealloc__ will now act as a fallback
        self.shutdown()


# --- OMNIMIDI EXTERNAL ENGINE CLASS ---

# This matches the DebugInfo struct in OmniMIDI.h
class DebugInfo(ctypes.Structure):
    _fields_ = [
        ('RenderingTime', ctypes.c_float),
        ('ActiveVoices', ctypes.c_uint * 16),
        ('ASIOInputLatency', ctypes.c_double),
        ('ASIOOutputLatency', ctypes.c_double),
        ('HealthThreadTime', ctypes.c_double),
        ('ATThreadTime', ctypes.c_double),
        ('EPThreadTime', ctypes.c_double),
        ('CookedThreadTime', ctypes.c_double),
        ('CurrentSFList', ctypes.c_uint),
        ('AudioLatency', ctypes.c_double),
        ('AudioBufferSize', ctypes.c_uint),
    ]

cdef class OmniMidiEngine:
    """
    An engine that sends MIDI events using the definitive OmniMIDI.h API.
    """
    cdef object lib
    cdef public bint is_initialized
    cdef object debug_info_ptr

    def __cinit__(self, dict audio_cfg):
        self.is_initialized = False
        self.lib = None
        self.debug_info_ptr = None

        try:
            dll_path = audio_cfg.get("omnimidi_dll_path", "OmniMIDI.dll")
            self.lib = ctypes.cdll.LoadLibrary(dll_path)
            print(f"[Engine] Attempting to load DLL from: {dll_path}")
        except (OSError, FileNotFoundError):
            raise Exception("OmniMIDI.dll not found. Please ensure OmniMIDI is installed.")
        # --- Prototype functions from OmniMIDI.h ---
        self.lib.IsKDMAPIAvailable.restype = ctypes.c_bool
        self.lib.InitializeKDMAPIStream.restype = ctypes.c_bool
        self.lib.SendDirectData.argtypes = [ctypes.c_uint]
        self.lib.ResetKDMAPIStream.argtypes = []
        self.lib.TerminateKDMAPIStream.restype = ctypes.c_bool
        # Note: Prototyping for GetDriverDebugInfo is moved below

        if not self.lib.IsKDMAPIAvailable():
            raise Exception("OmniMIDI reported that the KDMAPI is not available.")
            
        if not self.lib.InitializeKDMAPIStream():
            raise Exception("Failed to initialize the OmniMIDI KDMAPI Stream.")

        # Try to get the pointer to the debug info struct, but don't crash if it fails
        try:
            self.lib.GetDriverDebugInfo.restype = ctypes.POINTER(DebugInfo)
            self.debug_info_ptr = self.lib.GetDriverDebugInfo()
        except AttributeError:
            print("[Engine] Warning: GetDriverDebugInfo not found in this DLL version. Performance stats will be unavailable.")
            self.debug_info_ptr = None # Ensure it's None if the function isn't found
        
        print("[Engine] OmniMIDI API Stream initialized successfully.")
        self.is_initialized = True

    cpdef float get_cpu_usage(self):
        if self.debug_info_ptr:
            return self.debug_info_ptr[0].RenderingTime
        return 0.0

    cpdef int get_active_voices(self):
        if self.debug_info_ptr:
            return sum(self.debug_info_ptr[0].ActiveVoices)
        return 0

    cpdef send_raw_event(self, uint32_t event, uint32_t param):
        if not self.is_initialized: return
        cdef uint32_t pitch = param & 0xFF
        cdef uint32_t velocity = (param >> 8) & 0xFF
        cdef uint32_t status = event & 0xFF
        cdef uint32_t message = (velocity << 16) | (pitch << 8) | status
        self.lib.SendDirectData(message)

    cpdef send_all_notes_off(self):
        if not self.is_initialized: return
        self.lib.ResetKDMAPIStream()

    cpdef send_event_batch(self, list events):
        """Sends a batch of raw MIDI events using SendDirectData."""
        if not self.is_initialized: return
        cdef uint32_t event, param, pitch, velocity, status, message
        for e in events:
            event, param = e
            pitch = param & 0xFF
            velocity = (param >> 8) & 0xFF
            status = event & 0xFF
            message = (velocity << 16) | (pitch << 8) | status
            self.lib.SendDirectData(message)

    cpdef load_soundfont(self, str sf2_path):
        print("[Engine] SoundFont must be configured in OmniMIDI.")
    cpdef set_voices(self, int voices):
        print("[Engine] Max voices should be set via OmniMIDI's settings.")
    cpdef bint start_stream(self): return True
    cpdef uint32_t get_stream_handle(self): return 0

    # Add this method inside the OmniMidiEngine class
    cpdef shutdown(self):
        """Explicitly shuts down the OmniMIDI stream."""
        if self.is_initialized:
            print("[Engine] Terminating OmniMIDI API Stream.")
            self.lib.TerminateKDMAPIStream()
            self.is_initialized = False # Prevent double-freeing

    def __dealloc__(self):
        # __dealloc__ will now act as a fallback
        self.shutdown()