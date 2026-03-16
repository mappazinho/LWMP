
import ctypes
import os

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

class OmniMidiEngine:
    """
    An engine that sends MIDI events using the definitive OmniMIDI.h API.
    """
    def __init__(self, audio_cfg, load_from_path=False):
        self.is_initialized = False
        self.lib = None
        self.debug_info_ptr = None

        # Explicitly load winmm.dll from the system directory first.
        # This can sometimes help OmniMIDI hook into the correct multimedia functions.
        try:
            ctypes.WinDLL('winmm.dll')
            print("[Engine] Pre-loaded winmm.dll")
        except OSError as e:
            print(f"[Engine] Warning: Could not pre-load winmm.dll: {e}")

        if load_from_path:
            print("[Engine] Attempting to load OmniMIDI.dll from system PATH.")
            print(f"[Engine] PATH environment variable: {os.environ.get('PATH', 'PATH not set')}")
            found_in_path = False
            dll_location = "system PATH" # Default for PATH search
            for p_dir in os.environ.get('PATH', '').split(os.pathsep):
                potential_dll_path = os.path.join(p_dir, "OmniMIDI.dll")
                if os.path.exists(potential_dll_path):
                    print(f"[Engine] OmniMIDI.dll found in PATH at: {potential_dll_path}")
                    dll_location = potential_dll_path # Use the found path if explicit
                    found_in_path = True
                    break
            
            if not found_in_path:
                print("[Engine] Warning: OmniMIDI.dll was not found in any directory listed in PATH.")

            try:
                self.lib = ctypes.cdll.LoadLibrary("OmniMIDI.dll")
            except (OSError, FileNotFoundError):
                raise Exception("OmniMIDI.dll not found in system PATH. Please ensure it's installed or available on PATH.")
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_dir, "OmniMIDI.dll")
            dll_location = dll_path # Set dll_location for local load
            try: # [FIX] Added missing try block
                self.lib = ctypes.cdll.LoadLibrary(dll_path)
                print(f"[Engine] Attempting to load OmniMIDI.dll from: {dll_path}")
            except (OSError, FileNotFoundError):
                raise Exception(f"OmniMIDI.dll not found at '{dll_path}'. Please ensure it's in the same directory as the script.")
        
        self.lib.IsKDMAPIAvailable.restype = ctypes.c_bool
        self.lib.InitializeKDMAPIStream.restype = ctypes.c_bool
        self.lib.SendDirectData.argtypes = [ctypes.c_uint]
        self.lib.ResetKDMAPIStream.argtypes = []
        self.lib.TerminateKDMAPIStream.restype = ctypes.c_bool

        if not self.lib.IsKDMAPIAvailable():
            raise Exception(f"OmniMIDI ({dll_location}) reported that the KDMAPI is not available. Please check your OmniMIDI installation.")
            
        if not self.lib.InitializeKDMAPIStream(): # [FIX] Moved the try-except block for GetDriverDebugInfo here
            raise Exception(f"Failed to initialize the OmniMIDI KDMAPI Stream from {dll_location}. Please check your OmniMIDI installation.") 
        try:
            self.lib.GetDriverDebugInfo.restype = ctypes.POINTER(DebugInfo)
            self.debug_info_ptr = self.lib.GetDriverDebugInfo()
        except AttributeError:
            print("[Engine] Warning: GetDriverDebugInfo not found in this DLL version. Performance stats will be unavailable.")
            self.debug_info_ptr = None
        
        print("[Engine] OmniMIDI API Stream initialized successfully.")
        self.is_initialized = True

    def get_cpu_usage(self):
        if self.debug_info_ptr:
            return self.debug_info_ptr[0].RenderingTime
        return 0.0

    def get_active_voices(self):
        if self.debug_info_ptr:
            return sum(self.debug_info_ptr[0].ActiveVoices)
        return 0

    def send_raw_event(self, event, param):
        if not self.is_initialized: return

        status = event & 0xFF
        velocity = (param >> 8) & 0xFF

        # If it's a note-on event, check velocity
        if (status & 0xF0) == 0x90 and velocity < 20:
            return # Ignore low velocity notes

        pitch = param & 0xFF
        message = (velocity << 16) | (pitch << 8) | status
        self.lib.SendDirectData(message)

    def send_all_notes_off(self):
        if not self.is_initialized: return
        self.lib.ResetKDMAPIStream()

    def send_event_batch(self, events):
        """Sends a batch of raw MIDI events using SendDirectData."""
        if not self.is_initialized: return
        for e in events:
            event, param = e
            
            status = event & 0xFF
            velocity = (param >> 8) & 0xFF

            if (status & 0xF0) == 0x90 and velocity < 20:
                continue

            pitch = param & 0xFF
            message = (velocity << 16) | (pitch << 8) | status
            self.lib.SendDirectData(message)

    def load_soundfont(self, sf2_path):
        print("[Engine] SoundFont must be configured in OmniMIDI.")
        
    def set_voices(self, voices):
        print("[Engine] Max voices should be set via OmniMIDI's settings.")
        
    def start_stream(self):
        return True
        
    def get_stream_handle(self):
        return 0

    def shutdown(self):
        """Explicitly shuts down the OmniMIDI stream."""
        if self.is_initialized:
            print("[Engine] Terminating OmniMIDI API Stream.")
            self.lib.TerminateKDMAPIStream()
            self.is_initialized = False

    def __del__(self):
        self.shutdown()
