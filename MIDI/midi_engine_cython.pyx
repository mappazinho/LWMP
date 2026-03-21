# cython: language_level=3
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t
from libc.stddef cimport wchar_t
from cpython.unicode cimport PyUnicode_AsWideCharString
from cpython.mem cimport PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
import ctypes
import os
import sys
from runtime_paths import add_dll_search_dir, resolve_bass_library_paths
from time import sleep 
from libc.math cimport sin, M_PI, sqrt

cdef extern from *:
    """
    #ifdef _WIN32
    #include <windows.h>
    typedef HMODULE lwmp_lib_handle;
    typedef FARPROC lwmp_func_ptr;
    static lwmp_lib_handle lwmp_dlopen(const char* path) { return LoadLibraryA(path); }
    static lwmp_func_ptr lwmp_dlsym(lwmp_lib_handle h, const char* name) { return GetProcAddress(h, name); }
    static int lwmp_dlclose(lwmp_lib_handle h) { return FreeLibrary(h); }
    #else
    #include <dlfcn.h>
    typedef void* lwmp_lib_handle;
    typedef void* lwmp_func_ptr;
    static lwmp_lib_handle lwmp_dlopen(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
    static lwmp_func_ptr lwmp_dlsym(lwmp_lib_handle h, const char* name) { return dlsym(h, name); }
    static int lwmp_dlclose(lwmp_lib_handle h) { return dlclose(h); }
    #endif
    """
    ctypedef void* lwmp_lib_handle
    ctypedef void* lwmp_func_ptr
    lwmp_lib_handle lwmp_dlopen(const char* path)
    lwmp_func_ptr lwmp_dlsym(lwmp_lib_handle h, const char* name)
    int lwmp_dlclose(lwmp_lib_handle h)

cdef enum:
    BASS_OK = 0
    BASS_ERROR_ENDED = 27
    BASS_CONFIG_BUFFER = 0
    BASS_CONFIG_UPDATEPERIOD = 1
    BASS_CONFIG_FLOATDSP = 6
    
    BASS_STREAM_AUTOFREE = 0x40000
    BASS_STREAM_DECODE = 0x200000
    BASS_SAMPLE_FLOAT = 0x100
    BASS_MIDI_DECAYEND = 0x1000
    BASS_MIDI_NOFX = 0x2000
    BASS_MIDI_DECAYSEEK = 0x4000
    BASS_MIDI_SINCINTER = 0x800000
    
    BASS_POS_BYTE = 0
    STREAMPROC_PUSH = -1
    
    BASS_MIDI_EVENTS_RAW = 0x10000
    BASS_MIDI_EVENT_NOTE = 1
    BASS_MIDI_EVENT_PITCH = 4
    BASS_MIDI_EVENT_PITCHRANGE = 5
    BASS_MIDI_EVENT_NOTESOFF = 18
    BASS_MIDI_EVENTS_ASYNC = 0x40000000
    
    BASS_UNICODE = 0x80000000
    BASS_ATTRIB_FREQ = 1
    BASS_ATTRIB_VOL = 2
    BASS_ATTRIB_MIDI_VOICES = 0x12003

ctypedef uint32_t DWORD
ctypedef int BOOL
ctypedef uint64_t QWORD
ctypedef uint32_t HSTREAM
ctypedef uint32_t HSOUNDFONT
ctypedef void* HMUSIC 
ctypedef void* HSAMPLE 

ctypedef BOOL (*t_BASS_Init)(int device, DWORD freq, DWORD flags, void* win, void* dsguid)
ctypedef BOOL (*t_BASS_Free)()
ctypedef BOOL (*t_BASS_SetConfig)(DWORD option, DWORD value)
ctypedef DWORD (*t_BASS_GetConfig)(DWORD option)
ctypedef int (*t_BASS_ErrorGetCode)() nogil
ctypedef HSTREAM (*t_BASS_StreamCreate)(DWORD freq, DWORD chans, DWORD flags, void* proc, void* user)
ctypedef BOOL (*t_BASS_StreamFree)(HSTREAM handle)
ctypedef DWORD (*t_BASS_StreamPutData)(HSTREAM handle, const void* buffer, DWORD length) nogil
ctypedef QWORD (*t_BASS_StreamGetFilePosition)(HSTREAM handle, DWORD mode)
ctypedef BOOL (*t_BASS_ChannelPlay)(HSTREAM handle, BOOL restart)
ctypedef BOOL (*t_BASS_ChannelPause)(HSTREAM handle)
ctypedef BOOL (*t_BASS_ChannelStop)(HSTREAM handle)
ctypedef DWORD (*t_BASS_ChannelIsActive)(HSTREAM handle)
ctypedef BOOL (*t_BASS_ChannelSetPosition)(HSTREAM handle, QWORD pos, DWORD mode)
ctypedef QWORD (*t_BASS_ChannelGetPosition)(HSTREAM handle, DWORD mode)
ctypedef DWORD (*t_BASS_ChannelGetData)(HSTREAM handle, void* buffer, DWORD length) nogil
ctypedef QWORD (*t_BASS_ChannelSeconds2Bytes)(HSTREAM handle, double pos) nogil
ctypedef double (*t_BASS_ChannelBytes2Seconds)(HSTREAM handle, QWORD pos) nogil
ctypedef BOOL (*t_BASS_SetVolume)(float volume)
ctypedef BOOL (*t_BASS_ChannelSetAttribute)(HSTREAM handle, DWORD attrib, float value)

ctypedef HSTREAM (*t_BASS_MIDI_StreamCreate)(DWORD channels, DWORD flags, DWORD freq)
ctypedef DWORD (*t_BASS_MIDI_StreamEvents)(HSTREAM handle, DWORD mode, const void* events, DWORD length) nogil
ctypedef BOOL (*t_BASS_MIDI_StreamEvent)(HSTREAM handle, DWORD chan, DWORD event, DWORD param) nogil
ctypedef HSOUNDFONT (*t_BASS_MIDI_FontInit)(const void* file, DWORD flags)
ctypedef BOOL (*t_BASS_MIDI_FontLoad)(HSOUNDFONT handle, int preset, int bank)
ctypedef BOOL (*t_BASS_MIDI_FontFree)(HSOUNDFONT handle)
ctypedef BOOL (*t_BASS_MIDI_StreamSetFonts)(HSTREAM handle, void* fonts, DWORD count)

cdef struct BassFuncs:
    t_BASS_Init Init
    t_BASS_Free Free
    t_BASS_SetConfig SetConfig
    t_BASS_GetConfig GetConfig
    t_BASS_ErrorGetCode ErrorGetCode
    t_BASS_StreamCreate StreamCreate
    t_BASS_StreamFree StreamFree
    t_BASS_StreamPutData StreamPutData
    t_BASS_ChannelPlay ChannelPlay
    t_BASS_ChannelPause ChannelPause
    t_BASS_ChannelStop ChannelStop
    t_BASS_ChannelIsActive ChannelIsActive
    t_BASS_ChannelSetPosition ChannelSetPosition
    t_BASS_ChannelGetPosition ChannelGetPosition
    t_BASS_ChannelGetData ChannelGetData
    t_BASS_ChannelSeconds2Bytes ChannelSeconds2Bytes
    t_BASS_ChannelBytes2Seconds ChannelBytes2Seconds
    t_BASS_SetVolume SetVolume
    t_BASS_ChannelSetAttribute ChannelSetAttribute
    
    t_BASS_MIDI_StreamCreate MIDI_StreamCreate
    t_BASS_MIDI_StreamEvents MIDI_StreamEvents
    t_BASS_MIDI_StreamEvent MIDI_StreamEvent
    t_BASS_MIDI_FontInit MIDI_FontInit
    t_BASS_MIDI_FontLoad MIDI_FontLoad
    t_BASS_MIDI_FontFree MIDI_FontFree
    t_BASS_MIDI_StreamSetFonts MIDI_StreamSetFonts

cdef struct BASS_MIDI_FONT:
    HSOUNDFONT font
    int preset
    int bank

cdef struct MiniEvent:
    double time
    uint32_t status
    uint32_t param

cdef void* get_func(lwmp_lib_handle h, const char* name):
    return <void*>lwmp_dlsym(h, name)

cdef class BassMidiEngine:
    cdef lwmp_lib_handle hBass
    cdef lwmp_lib_handle hBassMidi
    cdef BassFuncs f
    cdef object dll_dir_handle
    
    cdef public bint is_initialized
    cdef public bint buffering_enabled
    cdef public bint debug_mode
    cdef public float volume_level
    cdef public float playback_speed
    cdef public int normal_voice_limit
    cdef public int emergency_voice_limit
    cdef public int emergency_velocity
    cdef public bint emergency_recovery_enabled
    
    cdef public HSTREAM midi_stream
    cdef public HSTREAM decode_stream
    cdef public HSTREAM playback_stream
    cdef HSOUNDFONT soundfont
    cdef int soundfont_preset
    cdef int soundfont_bank
    
    cdef uint64_t total_bytes_pushed
    
    cdef MiniEvent* event_buffer
    cdef size_t event_count
    cdef size_t current_event_idx
    cdef double simulated_time
    
    def __cinit__(self, dict audio_cfg, str soundfont_path=None, bint buffering=False, bint debug=False):
        self.is_initialized = False
        self.buffering_enabled = buffering
        self.debug_mode = debug
        self.volume_level = 1.0
        self.playback_speed = 1.0
        self.normal_voice_limit = 512
        self.emergency_voice_limit = 96
        self.emergency_velocity = 100
        self.emergency_recovery_enabled = False
        self.midi_stream = 0
        self.decode_stream = 0
        self.playback_stream = 0
        self.soundfont = 0
        self.soundfont_preset = -1
        self.soundfont_bank = 0
        self.total_bytes_pushed = 0
        self.event_buffer = NULL
        self.event_count = 0
        self.current_event_idx = 0
        self.simulated_time = 0.0
        self.hBass = NULL
        self.hBassMidi = NULL
        self.dll_dir_handle = None

        cdef object bass_info = resolve_bass_library_paths(__file__)
        cdef object bass_path_str = bass_info[0]
        cdef object bassmidi_path_str = bass_info[1]
        cdef object bass_dir_str = bass_info[2]
        if not bass_path_str or not bassmidi_path_str:
            raise ImportError("Could not locate BASS or BASSMIDI runtime libraries")
        self.dll_dir_handle = add_dll_search_dir(bass_dir_str)
        cdef bytes bass_path
        cdef bytes bassmidi_path
        if os.name == "nt":
            bass_path = (<str>bass_path_str).encode('mbcs')
            bassmidi_path = (<str>bassmidi_path_str).encode('mbcs')
        else:
            bass_path = os.fsencode(<str>bass_path_str)
            bassmidi_path = os.fsencode(<str>bassmidi_path_str)

        if self.debug_mode:
            print(f"Loading BASS from: {bass_path.decode(errors='replace')}")
        
        self.hBass = lwmp_dlopen(bass_path)
        if not self.hBass:
            raise ImportError("Could not load BASS runtime library")
            
        self.hBassMidi = lwmp_dlopen(bassmidi_path)
        if not self.hBassMidi:
            lwmp_dlclose(self.hBass)
            raise ImportError("Could not load BASSMIDI runtime library")

        self.f.Init = <t_BASS_Init>get_func(self.hBass, "BASS_Init")
        self.f.Free = <t_BASS_Free>get_func(self.hBass, "BASS_Free")
        self.f.SetConfig = <t_BASS_SetConfig>get_func(self.hBass, "BASS_SetConfig")
        self.f.GetConfig = <t_BASS_GetConfig>get_func(self.hBass, "BASS_GetConfig")
        self.f.ErrorGetCode = <t_BASS_ErrorGetCode>get_func(self.hBass, "BASS_ErrorGetCode")
        self.f.StreamCreate = <t_BASS_StreamCreate>get_func(self.hBass, "BASS_StreamCreate")
        self.f.StreamFree = <t_BASS_StreamFree>get_func(self.hBass, "BASS_StreamFree")
        self.f.StreamPutData = <t_BASS_StreamPutData>get_func(self.hBass, "BASS_StreamPutData")
        self.f.ChannelPlay = <t_BASS_ChannelPlay>get_func(self.hBass, "BASS_ChannelPlay")
        self.f.ChannelPause = <t_BASS_ChannelPause>get_func(self.hBass, "BASS_ChannelPause")
        self.f.ChannelStop = <t_BASS_ChannelStop>get_func(self.hBass, "BASS_ChannelStop")
        self.f.ChannelIsActive = <t_BASS_ChannelIsActive>get_func(self.hBass, "BASS_ChannelIsActive")
        self.f.ChannelSetPosition = <t_BASS_ChannelSetPosition>get_func(self.hBass, "BASS_ChannelSetPosition")
        self.f.ChannelGetPosition = <t_BASS_ChannelGetPosition>get_func(self.hBass, "BASS_ChannelGetPosition")
        self.f.ChannelGetData = <t_BASS_ChannelGetData>get_func(self.hBass, "BASS_ChannelGetData")
        self.f.ChannelSeconds2Bytes = <t_BASS_ChannelSeconds2Bytes>get_func(self.hBass, "BASS_ChannelSeconds2Bytes")
        self.f.ChannelBytes2Seconds = <t_BASS_ChannelBytes2Seconds>get_func(self.hBass, "BASS_ChannelBytes2Seconds")
        self.f.SetVolume = <t_BASS_SetVolume>get_func(self.hBass, "BASS_SetVolume")
        self.f.ChannelSetAttribute = <t_BASS_ChannelSetAttribute>get_func(self.hBass, "BASS_ChannelSetAttribute")

        self.f.MIDI_StreamCreate = <t_BASS_MIDI_StreamCreate>get_func(self.hBassMidi, "BASS_MIDI_StreamCreate")
        self.f.MIDI_StreamEvents = <t_BASS_MIDI_StreamEvents>get_func(self.hBassMidi, "BASS_MIDI_StreamEvents")
        self.f.MIDI_StreamEvent = <t_BASS_MIDI_StreamEvent>get_func(self.hBassMidi, "BASS_MIDI_StreamEvent")
        self.f.MIDI_FontInit = <t_BASS_MIDI_FontInit>get_func(self.hBassMidi, "BASS_MIDI_FontInit")
        self.f.MIDI_FontLoad = <t_BASS_MIDI_FontLoad>get_func(self.hBassMidi, "BASS_MIDI_FontLoad")
        self.f.MIDI_FontFree = <t_BASS_MIDI_FontFree>get_func(self.hBassMidi, "BASS_MIDI_FontFree")
        self.f.MIDI_StreamSetFonts = <t_BASS_MIDI_StreamSetFonts>get_func(self.hBassMidi, "BASS_MIDI_StreamSetFonts")

        self.f.SetConfig(BASS_CONFIG_FLOATDSP, 1)
        self.f.SetConfig(BASS_CONFIG_UPDATEPERIOD, 1)
        self.f.SetConfig(BASS_CONFIG_BUFFER, 60000)
        
        if not self.f.Init(-1, 44100, 0, NULL, NULL):
            err = self.f.ErrorGetCode()
            if err != 14: # BASS_ERROR_ALREADY
                raise RuntimeError(f"BASS_Init failed: {err}")
        
        self.is_initialized = True
        
        cdef DWORD flags
        if self.buffering_enabled:
            flags = BASS_STREAM_DECODE | BASS_SAMPLE_FLOAT | BASS_MIDI_SINCINTER
            self.decode_stream = self.f.MIDI_StreamCreate(16, flags, 44100)
            if not self.decode_stream: raise RuntimeError(f"Decode Stream Create Failed: {self.f.ErrorGetCode()}")
            self.f.ChannelSetAttribute(self.decode_stream, BASS_ATTRIB_MIDI_VOICES, <float>self.normal_voice_limit)
            
            self.playback_stream = self.f.StreamCreate(44100, 2, BASS_SAMPLE_FLOAT, <void*>STREAMPROC_PUSH, NULL)
            if not self.playback_stream: raise RuntimeError(f"Playback Stream Create Failed: {self.f.ErrorGetCode()}")
            self.f.ChannelSetAttribute(self.playback_stream, BASS_ATTRIB_VOL, self.volume_level)
            self.f.ChannelSetAttribute(self.playback_stream, BASS_ATTRIB_FREQ, 44100.0 * self.playback_speed)
            self.midi_stream = self.playback_stream 
        else:
            flags = BASS_STREAM_AUTOFREE | BASS_SAMPLE_FLOAT | BASS_MIDI_SINCINTER | BASS_MIDI_EVENTS_ASYNC
            self.midi_stream = self.f.MIDI_StreamCreate(16, flags, 44100)
            if not self.midi_stream: raise RuntimeError(f"MIDI Stream Create Failed: {self.f.ErrorGetCode()}")
            self.f.ChannelSetAttribute(self.midi_stream, BASS_ATTRIB_MIDI_VOICES, <float>self.normal_voice_limit)
            self.f.ChannelSetAttribute(self.midi_stream, BASS_ATTRIB_VOL, self.volume_level)
            self.f.ChannelSetAttribute(self.midi_stream, BASS_ATTRIB_FREQ, 44100.0 * self.playback_speed)

        if soundfont_path:
            self.load_soundfont(self.decode_stream if self.buffering_enabled else self.midi_stream, soundfont_path)

    cpdef upload_events(self, double[:] times, uint32_t[:] statuses, uint32_t[:] params):
        """Loads sorted event arrays into internal C memory."""
        if self.event_buffer != NULL:
            free(self.event_buffer)
            self.event_buffer = NULL
            self.event_count = 0
        
        cdef size_t count = times.shape[0]
        if count == 0: return

        if self.debug_mode: print(f"[Cython] Uploading {count} events...")
        
        self.event_buffer = <MiniEvent*>malloc(count * sizeof(MiniEvent))
        if self.event_buffer == NULL:
            raise MemoryError("Failed to allocate event buffer")
        
        cdef size_t i
        for i in range(count):
            self.event_buffer[i].time = times[i]
            self.event_buffer[i].status = statuses[i]
            self.event_buffer[i].param = params[i]
            
        self.event_count = count
        self.current_event_idx = 0
        self.simulated_time = 0.0
        self.total_bytes_pushed = 0
        if self.playback_stream:
            self.f.ChannelSetPosition(self.playback_stream, 0, BASS_POS_BYTE)
        if self.decode_stream:
            self.f.ChannelSetPosition(self.decode_stream, 0, BASS_POS_BYTE)
        if self.debug_mode: print(f"[Cython] Events uploaded successfully.")

    cpdef fill_buffer(self, double limit_seconds):
        """
        Main tight loop. Checks buffer level and renders if needed.
        Returns True if more rendering is needed (buffer not full), False if buffer full or done.
        """
        if not self.buffering_enabled or self.event_buffer == NULL:
            return 0.0

        cdef double buffer_level
        cdef double chunk_dur = 0.0005
        cdef double rendered_dur
        cdef HSTREAM decode = self.decode_stream
        cdef HSTREAM playback = self.playback_stream
        cdef MiniEvent* ev
        
        cdef QWORD pos_bytes = self.f.ChannelGetPosition(playback, BASS_POS_BYTE)
        if pos_bytes == <QWORD>-1:
            pos_bytes = 0
        cdef int64_t diff = <int64_t>self.total_bytes_pushed - <int64_t>pos_bytes
        if diff < 0:
            diff = 0
        buffer_level = self.f.ChannelBytes2Seconds(playback, diff)
        
        if buffer_level >= limit_seconds:
            return buffer_level # Buffer healthy

        cdef int loops = 0
        cdef int max_loops = 400

        while loops < max_loops:
            while self.current_event_idx < self.event_count:
                ev = &self.event_buffer[self.current_event_idx]
                if ev.time <= self.simulated_time:
                    self.send_raw_event(ev.status, ev.param)
                    self.current_event_idx += 1
                else:
                    break

            rendered_dur = self.render_forward(chunk_dur)
            if rendered_dur <= 0.0:
                if self.current_event_idx < self.event_count:
                    ev = &self.event_buffer[self.current_event_idx]
                    if ev.time > self.simulated_time:
                        rendered_dur = self.queue_silence(chunk_dur)
                if rendered_dur <= 0.0:
                    break
            self.simulated_time += rendered_dur
            loops += 1
            
            pos_bytes = self.f.ChannelGetPosition(playback, BASS_POS_BYTE)
            if pos_bytes == <QWORD>-1:
                pos_bytes = 0
            diff = <int64_t>self.total_bytes_pushed - <int64_t>pos_bytes
            if diff < 0:
                diff = 0
            if self.f.ChannelBytes2Seconds(playback, diff) >= limit_seconds:
                break

        return self.get_buffer_level()

    cpdef double queue_silence(self, double seconds):
        if not self.buffering_enabled or not self.playback_stream or seconds <= 0.0:
            return 0.0

        cdef HSTREAM playback = self.playback_stream
        cdef QWORD bytes_needed = self.f.ChannelSeconds2Bytes(playback, seconds)
        if bytes_needed == 0 or bytes_needed == <QWORD>-1:
            return 0.0

        cdef DWORD chunk_size = 65536
        cdef char* buf = <char*>malloc(chunk_size)
        if not buf:
            return 0.0

        cdef QWORD remaining = bytes_needed
        cdef DWORD queued_bytes
        cdef DWORD total_written = 0
        cdef DWORD to_write
        cdef DWORD err_val = 0xFFFFFFFF

        try:
            memset(buf, 0, chunk_size)
            with nogil:
                while remaining > 0:
                    to_write = chunk_size
                    if remaining < chunk_size:
                        to_write = <DWORD>remaining

                    queued_bytes = self.f.StreamPutData(playback, buf, to_write)
                    if queued_bytes == err_val:
                        break

                    total_written += to_write
                    remaining -= to_write

            self.total_bytes_pushed += total_written
            return self.f.ChannelBytes2Seconds(playback, total_written)
        finally:
            free(buf)

    cpdef set_current_time(self, double seconds):
        self.simulated_time = seconds
        cdef size_t L = 0
        cdef size_t R = self.event_count
        cdef size_t M
        
        while L < R:
            M = (L + R) // 2
            if self.event_buffer[M].time < seconds:
                L = M + 1
            else:
                R = M
        self.current_event_idx = L
        if self.debug_mode: print(f"[Cython] Seek to {seconds}s -> Index {L}")

    cpdef set_volume(self, float volume):
        self.volume_level = volume
        if self.buffering_enabled:
            if self.playback_stream:
                self.f.ChannelSetAttribute(self.playback_stream, BASS_ATTRIB_VOL, volume)
        elif self.midi_stream:
            self.f.ChannelSetAttribute(self.midi_stream, BASS_ATTRIB_VOL, volume)

    cpdef pause(self):
        if self.midi_stream: self.f.ChannelPause(self.midi_stream)

    cpdef play(self):
        if self.midi_stream: self.f.ChannelPlay(self.midi_stream, False)

    cpdef set_speed(self, float speed):
        if speed < 0.1:
            speed = 0.1
        elif speed > 4.0:
            speed = 4.0
        self.playback_speed = speed
        cdef HSTREAM target = self.playback_stream if self.buffering_enabled else self.midi_stream
        if target:
            self.f.ChannelSetAttribute(target, BASS_ATTRIB_FREQ, 44100.0 * speed)

    cpdef stop(self):
        if self.midi_stream:
            self.f.ChannelStop(self.midi_stream)
            self.f.ChannelSetPosition(self.midi_stream, 0, BASS_POS_BYTE)
        self.total_bytes_pushed = 0
        if self.buffering_enabled and self.decode_stream:
            self.f.ChannelSetPosition(self.decode_stream, 0, BASS_POS_BYTE)
        self.current_event_idx = 0
        self.simulated_time = 0.0
        self.send_all_notes_off()

    cpdef set_voices(self, int voices):
        self.normal_voice_limit = voices
        cdef HSTREAM target = self.decode_stream if self.buffering_enabled else self.midi_stream
        cdef int effective_voices = self.emergency_voice_limit if self.emergency_recovery_enabled else self.normal_voice_limit
        if target:
             self.f.ChannelSetAttribute(target, BASS_ATTRIB_MIDI_VOICES, <float>effective_voices)

    cpdef set_emergency_recovery(self, bint enabled):
        self.emergency_recovery_enabled = enabled
        cdef HSTREAM target = self.decode_stream if self.buffering_enabled else self.midi_stream
        cdef int effective_voices = self.emergency_voice_limit if enabled else self.normal_voice_limit
        if target:
            self.f.ChannelSetAttribute(target, BASS_ATTRIB_MIDI_VOICES, <float>effective_voices)

    cpdef load_soundfont(self, HSTREAM stream, str path):
        if self.debug_mode: print(f"[Cython] load_soundfont called for path: {path}")
        if not path or not os.path.exists(path): return

        cdef str ext = os.path.splitext(path)[1].lower()
        cdef wchar_t* wide_path = NULL
        cdef bytes fs_path = b""
        cdef const void* c_path = NULL
        cdef DWORD font_flags = 0
        cdef BASS_MIDI_FONT font_struct
        cdef BOOL ok = 0

        try:
            if self.soundfont:
                self.f.MIDI_FontFree(self.soundfont)
                self.soundfont = 0

            if os.name == "nt":
                wide_path = PyUnicode_AsWideCharString(path, NULL)
                c_path = <const void*>wide_path
                font_flags = BASS_UNICODE
            else:
                fs_path = os.fsencode(path)
                c_path = <const void*>fs_path
                font_flags = 0

            self.soundfont = self.f.MIDI_FontInit(c_path, font_flags)
            if self.debug_mode:
                print(f"[Cython] MIDI_FontInit handle={self.soundfont} err={self.f.ErrorGetCode()}")
            if not self.soundfont:
                return

            if ext == ".sfz":
                self.soundfont_preset = 0
                self.soundfont_bank = 0
                self.f.MIDI_FontLoad(self.soundfont, 0, 0)
            else:
                self.soundfont_preset = -1
                self.soundfont_bank = 0

            font_struct.font = self.soundfont
            font_struct.preset = self.soundfont_preset
            font_struct.bank = self.soundfont_bank
            ok = self.f.MIDI_StreamSetFonts(stream, &font_struct, 1)
            if self.debug_mode:
                print(
                    f"[Cython] MIDI_StreamSetFonts ok={ok} err={self.f.ErrorGetCode()} "
                    f"stream={stream} preset={self.soundfont_preset} bank={self.soundfont_bank}"
                )
        finally:
            if wide_path != NULL:
                PyMem_Free(wide_path)

    cpdef send_raw_event(self, uint32_t event, uint32_t param):
        cdef HSTREAM target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if not target: return
        cdef uint32_t status = event & 0xFF
        cdef uint32_t chan = status & 0x0F
        cdef uint32_t cmd = status & 0xF0
        cdef uint32_t d1
        cdef uint32_t d2
        cdef BOOL ok = 0
        cdef uint8_t raw_cc[3]
        
        if cmd == 0x90 or cmd == 0x80:
            if cmd == 0x90 and self.buffering_enabled and self.emergency_recovery_enabled:
                d1 = param & 0xFF
                d2 = (param >> 8) & 0xFF
                if d2 > 0:
                    param = d1 | (<uint32_t>self.emergency_velocity << 8)
            ok = self.f.MIDI_StreamEvent(target, chan, BASS_MIDI_EVENT_NOTE, param)
        elif cmd == 0xE0:
            d1 = param & 0xFF
            d2 = (param >> 8) & 0xFF
            ok = self.f.MIDI_StreamEvent(target, chan, BASS_MIDI_EVENT_PITCH, d1 | (d2 << 7))
        elif cmd == 0xB0:
            raw_cc[0] = <uint8_t>status
            raw_cc[1] = <uint8_t>(param & 0xFF)
            raw_cc[2] = <uint8_t>((param >> 8) & 0xFF)
            self.f.MIDI_StreamEvents(target, BASS_MIDI_EVENTS_RAW, raw_cc, 3)
            ok = 1
        if self.debug_mode and not ok:
            print(f"[Cython] MIDI_StreamEvent failed: status=0x{status:02X} chan={chan} param={param} err={self.f.ErrorGetCode()}")

    cpdef send_all_notes_off(self):
        cdef HSTREAM target = self.decode_stream if self.buffering_enabled else self.midi_stream
        if not target: return
        cdef int c
        for c in range(16):
            self.f.MIDI_StreamEvent(target, c, BASS_MIDI_EVENT_NOTESOFF, 0)

    cpdef set_pitch_bend_range(self, int semitones):
        cdef HSTREAM target = self.decode_stream if self.buffering_enabled else self.midi_stream
        cdef int c
        cdef uint32_t value
        if not target:
            return
        if semitones < 0:
            semitones = 0
        elif semitones > 127:
            semitones = 127
        value = <uint32_t>semitones
        for c in range(16):
            self.f.MIDI_StreamEvent(target, c, BASS_MIDI_EVENT_PITCHRANGE, value)
    
    cpdef test_piano_sweep(self):
        if not self.soundfont: return
        cdef HSTREAM sweep_stream = self.f.MIDI_StreamCreate(16, BASS_SAMPLE_FLOAT | BASS_STREAM_AUTOFREE, 44100)
        if not sweep_stream: return
        cdef BASS_MIDI_FONT font_struct
        font_struct.font = self.soundfont
        font_struct.preset = self.soundfont_preset
        font_struct.bank = self.soundfont_bank
        self.f.MIDI_StreamSetFonts(sweep_stream, &font_struct, 1)
        self.f.ChannelSetAttribute(sweep_stream, BASS_ATTRIB_VOL, self.volume_level)
        self.f.ChannelPlay(sweep_stream, False)
        cdef int note
        for note in range(60, 73):
            self.f.MIDI_StreamEvent(sweep_stream, 0, BASS_MIDI_EVENT_NOTE, note | (100 << 8))
            sleep(0.1)
            self.f.MIDI_StreamEvent(sweep_stream, 0, BASS_MIDI_EVENT_NOTE, note)
        self.f.StreamFree(sweep_stream)

    cpdef bytes render_pcm_chunk(self, double seconds):
        if not self.buffering_enabled or not self.decode_stream or self.event_buffer == NULL or seconds <= 0.0:
            return b""

        cdef HSTREAM decode = self.decode_stream
        cdef double target_end = self.simulated_time + seconds
        cdef double segment_seconds
        cdef double read_seconds
        cdef double next_event_time
        cdef double event_epsilon = 1.0 / 44100.0
        cdef QWORD bytes_needed
        cdef DWORD read_bytes
        cdef DWORD err_val = 0xFFFFFFFF
        cdef char* buf = NULL
        cdef bytes out_bytes = b""
        cdef MiniEvent* ev

        chunks = []
        while self.simulated_time < target_end:
            while self.current_event_idx < self.event_count:
                ev = &self.event_buffer[self.current_event_idx]
                if ev.time <= self.simulated_time + event_epsilon:
                    self.send_raw_event(ev.status, ev.param)
                    self.current_event_idx += 1
                else:
                    break

            next_event_time = target_end
            if self.current_event_idx < self.event_count:
                ev = &self.event_buffer[self.current_event_idx]
                if ev.time < next_event_time:
                    next_event_time = ev.time

            segment_seconds = next_event_time - self.simulated_time
            if segment_seconds <= 0.0:
                if target_end - self.simulated_time <= event_epsilon:
                    break
                segment_seconds = min(event_epsilon, target_end - self.simulated_time)
            elif segment_seconds < event_epsilon:
                segment_seconds = min(event_epsilon, target_end - self.simulated_time)

            bytes_needed = self.f.ChannelSeconds2Bytes(decode, segment_seconds)
            if bytes_needed == 0 or bytes_needed == <QWORD>-1:
                break

            buf = <char*>malloc(<size_t>bytes_needed)
            if buf == NULL:
                raise MemoryError("Failed to allocate PCM render buffer")
            try:
                read_bytes = self.f.ChannelGetData(decode, buf, <DWORD>bytes_needed)
                if read_bytes == err_val:
                    break
                if read_bytes == 0:
                    break
                chunks.append(PyBytes_FromStringAndSize(buf, read_bytes))
                read_seconds = self.f.ChannelBytes2Seconds(decode, read_bytes)
                if read_seconds <= 0.0:
                    break
                self.simulated_time += read_seconds
            finally:
                free(buf)
                buf = NULL

        if chunks:
            out_bytes = b"".join(chunks)
        return out_bytes

    cpdef double render_forward(self, double seconds):
        if not self.buffering_enabled: return 0.0
        
        cdef HSTREAM decode = self.decode_stream
        cdef HSTREAM playback = self.playback_stream
        
        if not decode or not playback: return 0.0
        
        cdef QWORD bytes_needed = self.f.ChannelSeconds2Bytes(decode, seconds)
        if bytes_needed == 0 or bytes_needed == <QWORD>-1: return 0.0
        
        cdef DWORD chunk_size = 65536 
        cdef char* buf = <char*>malloc(chunk_size)
        if not buf: return 0.0
        
        cdef DWORD total_written = 0
        cdef QWORD remaining = bytes_needed
        cdef DWORD read_bytes
        cdef DWORD queued_bytes
        cdef DWORD err_val = 0xFFFFFFFF 
        cdef DWORD to_read
        
        try:
            with nogil:
                while remaining > 0:
                    to_read = chunk_size
                    if remaining < chunk_size:
                        to_read = <DWORD>remaining
                    
                    read_bytes = self.f.ChannelGetData(decode, buf, to_read)
                    if read_bytes == err_val or read_bytes == 0:
                        break
                    
                    queued_bytes = self.f.StreamPutData(playback, buf, read_bytes)
                    if queued_bytes == err_val:
                        break

                    # BASS_StreamPutData accepts the full block and returns the
                    # queue level, not the number of bytes consumed from this call.
                    total_written += read_bytes
                    remaining -= read_bytes
            
            self.total_bytes_pushed += total_written
            return self.f.ChannelBytes2Seconds(playback, total_written)
        finally:
            free(buf)

    cpdef double get_buffer_level(self):
        if not self.buffering_enabled or not self.playback_stream: return 0.0
        cdef QWORD current_pos_bytes = self.f.ChannelGetPosition(self.playback_stream, BASS_POS_BYTE)
        if current_pos_bytes == <QWORD>-1: return 0.0
        cdef int64_t diff = <int64_t>self.total_bytes_pushed - <int64_t>current_pos_bytes
        if diff < 0: diff = 0
        return self.f.ChannelBytes2Seconds(self.playback_stream, diff)

    cpdef double get_position_seconds(self):
        if not self.playback_stream: return 0.0
        cdef QWORD b = self.f.ChannelGetPosition(self.playback_stream, BASS_POS_BYTE)
        if b == <QWORD>-1: return 0.0
        return self.f.ChannelBytes2Seconds(self.playback_stream, b)

    cpdef bint is_active(self):
        if not self.midi_stream: return False
        return self.f.ChannelIsActive(self.midi_stream) == 1

    cpdef shutdown(self):
        if self.event_buffer != NULL:
            free(self.event_buffer)
            self.event_buffer = NULL
        if self.midi_stream:
            self.f.StreamFree(self.midi_stream)
            self.midi_stream = 0
        if self.decode_stream:
            self.f.StreamFree(self.decode_stream)
            self.decode_stream = 0
        self.playback_stream = 0
        if self.soundfont:
            self.f.MIDI_FontFree(self.soundfont)
            self.soundfont = 0
        self.soundfont_preset = -1
        self.soundfont_bank = 0
        if self.is_initialized:
            self.f.Free()
            self.is_initialized = False
        if self.hBassMidi:
            lwmp_dlclose(self.hBassMidi)
            self.hBassMidi = NULL
        if self.hBass:
            lwmp_dlclose(self.hBass)
            self.hBass = NULL
        self.dll_dir_handle = None

    def __dealloc__(self):
        self.shutdown()
