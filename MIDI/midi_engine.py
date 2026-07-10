
import ctypes
import os
import sys

_BATCH_SEND_MAX = 512

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
        self._dll_dir_handle = None
        self._init_working_dir = None
        self.load_from_path = bool(load_from_path)
        self.backend_display_name = "OmniMIDI" if self.load_from_path else "Custom Synth"
        self._batch_lib = None
        self._send_direct_data_addr = None

        # Explicitly load winmm.dll from the system directory first.
        # This can sometimes help the synth hook into the correct multimedia functions.
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
                raise Exception("OmniMIDI.dll not found in system PATH. Please ensure OmniMIDI is installed or available on PATH.")
        else:
            synth_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(synth_dir, "SYNTH.dll")
            dll_location = dll_path # Set dll_location for local load
            try: # [FIX] Added missing try block
                if hasattr(os, "add_dll_directory") and os.path.isdir(synth_dir):
                    try:
                        self._dll_dir_handle = os.add_dll_directory(synth_dir)
                        print(f"[Engine] Added DLL search directory: {synth_dir}")
                    except OSError as e:
                        print(f"[Engine] Warning: Could not add DLL search directory '{synth_dir}': {e}")
                self._init_working_dir = os.getcwd()
                if os.path.isdir(synth_dir):
                    os.chdir(synth_dir)
                    print(f"[Engine] Switched working directory to: {synth_dir}")
                print(f"[Engine] Attempting to load SYNTH.dll from: {dll_path}")
                self.lib = ctypes.cdll.LoadLibrary(dll_path)
            except (OSError, FileNotFoundError) as e:
                location_hint = "next to the EXE" if getattr(sys, "frozen", False) else "in the MIDI folder"
                raise Exception(f"Failed to load SYNTH.dll from '{dll_path}'. Ensure it is {location_hint}. Loader error: {e}")
        
        self.lib.IsKDMAPIAvailable.restype = ctypes.c_bool
        self.lib.InitializeKDMAPIStream.restype = ctypes.c_bool
        self.lib.SendDirectData.argtypes = [ctypes.c_uint]
        self.lib.ResetKDMAPIStream.argtypes = []
        self.lib.TerminateKDMAPIStream.restype = ctypes.c_bool

        if not self.lib.IsKDMAPIAvailable():
            if self.load_from_path:
                raise Exception(f"{self.backend_display_name} ({dll_location}) reported that the KDMAPI is not available. Please check your OmniMIDI installation.")
            raise Exception(f"{self.backend_display_name} ({dll_location}) reported that the KDMAPI is not available. Please check the bundled SYNTH.dll.")
            
        if not self.lib.InitializeKDMAPIStream(): # [FIX] Moved the try-except block for GetDriverDebugInfo here
            if self.load_from_path:
                raise Exception(f"Failed to initialize the {self.backend_display_name} KDMAPI stream from {dll_location}. Please check your OmniMIDI installation.")
            raise Exception(f"Failed to initialize the {self.backend_display_name} KDMAPI stream from {dll_location}. Please check the bundled SYNTH.dll.")
        try:
            self.lib.GetDriverDebugInfo.restype = ctypes.POINTER(DebugInfo)
            self.debug_info_ptr = self.lib.GetDriverDebugInfo()
        except AttributeError:
            print("[Engine] Warning: GetDriverDebugInfo not found in this DLL version. Performance stats will be unavailable.")
            self.debug_info_ptr = None
        finally:
            if self._init_working_dir is not None:
                try:
                    os.chdir(self._init_working_dir)
                    print(f"[Engine] Restored working directory to: {self._init_working_dir}")
                except OSError as e:
                    print(f"[Engine] Warning: Could not restore working directory: {e}")
                self._init_working_dir = None
        
        print(f"[Engine] {self.backend_display_name} API Stream initialized successfully.")
        self.is_initialized = True

        self._load_batch_helper()

    def _load_batch_helper(self):
        synth_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
        batch_dll = os.path.join(synth_dir, "midi_batch_send.dll")
        if not os.path.exists(batch_dll):
            print("[Engine] midi_batch_send.dll not found; using Python batch fallback.")
            return
        try:
            self._batch_lib = ctypes.cdll.LoadLibrary(batch_dll)
            self._batch_lib.BatchSendDirectData.restype = ctypes.c_int
            self._batch_lib.BatchSendDirectData.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.c_int,
            ]
            self._send_direct_data_addr = ctypes.cast(
                self.lib.SendDirectData, ctypes.c_void_p
            ).value
            print("[Engine] midi_batch_send.dll loaded; native batch send active.")
        except Exception as e:
            print(f"[Engine] Warning: Could not load batch helper: {e}")
            self._batch_lib = None
            self._send_direct_data_addr = None

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
        """Sends a batch of raw MIDI events using SendDirectData.

        When the native batch helper is available, packs messages into a C
        array and dispatches them in a single ctypes call (chunked to avoid
        large alloca overhead).  Falls back to per-event Python loop otherwise.
        """
        if not self.is_initialized or not events:
            return

        send_fn = self._send_direct_data_addr
        batch_lib = self._batch_lib

        if batch_lib is not None and send_fn is not None:
            packed = []
            for event, param in events:
                status = event & 0xFF
                velocity = (param >> 8) & 0xFF
                if (status & 0xF0) == 0x90 and velocity < 20:
                    continue
                pitch = param & 0xFF
                packed.append((velocity << 16) | (pitch << 8) | status)

            if not packed:
                return

            arr_type = ctypes.c_uint * len(packed)
            c_arr = arr_type(*packed)
            batch_lib.BatchSendDirectData(send_fn, c_arr, len(packed))
            return

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
        print(f"[Engine] SoundFont must be configured in {self.backend_display_name}.")
        
    def set_voices(self, voices):
        print(f"[Engine] Max voices should be set via {self.backend_display_name}'s settings.")
        
    def start_stream(self):
        return True
        
    def get_stream_handle(self):
        return 0

    def shutdown(self):
        """Explicitly shuts down the synth stream."""
        if self.is_initialized:
            print(f"[Engine] Terminating {self.backend_display_name} API Stream.")
            self.lib.TerminateKDMAPIStream()
            self.is_initialized = False

    def __del__(self):
        self.shutdown()
