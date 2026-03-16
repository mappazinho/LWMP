# filename: midi_parser_cython.pxd

cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t

# --- C-level Struct for GPU Note Data ---
cdef struct GpuNote:
    np.float32_t on_time
    np.float32_t off_time
    np.uint8_t   pitch
    np.uint8_t   velocity
    np.uint8_t   track
    np.uint8_t   padding

# --- C-level Struct for Playback Event Data ---
cdef struct PlaybackEvent:
    double on_time
    double off_time
    uint8_t pitch
    uint8_t velocity
    uint8_t channel
    uint16_t track_num

# --- Expose the public attributes of our Cython class ---
cdef class MidiParser:
    # Public attributes (read-only)
    cdef public np.ndarray note_data_for_gpu
    cdef public np.ndarray note_events_for_playback
    cdef public list program_change_events
    cdef public list pitch_bend_events
    cdef public double total_duration_sec
    cdef public int ticks_per_beat
    cdef public str filename
    
    # Internal C helper functions
    cdef _get_var_len(self, unsigned char* data, int i, int max_len)

    # --- [FIX] Removed the '_stream_track_events' declaration ---
    # It's a 'def' (Python) method, so it doesn't
    # need to be in the 'cdef class' header.