# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import struct
import numpy as np
cimport numpy as np
import time
import cython

from libc.string cimport memcpy
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libcpp.algorithm cimport sort as std_sort

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.queue cimport queue
from libcpp.pair cimport pair
from libcpp.vector cimport vector

np.import_array()

GPU_NOTE_DTYPE = np.dtype([
    ('on_time', 'f4'),
    ('off_time', 'f4'),
    ('pitch', 'u1'),
    ('velocity', 'u1'),
    ('track', 'u1'),
    ('padding', 'u1')
], align=True)

PLAYBACK_EVENT_DTYPE = np.dtype([
    ('on_time', 'f8'),
    ('off_time', 'f8'),
    ('pitch', 'u1'),
    ('velocity', 'u1'),
    ('channel', 'u1'),
    ('track_num', 'u2')
], align=True)

cdef struct NoteOnInfo:
    double on_time_sec
    uint8_t velocity
    uint16_t track_num

cdef struct RawEvent:
    uint32_t abs_tick
    uint8_t event_type
    uint8_t channel
    uint8_t data1
    uint8_t data2
    uint8_t meta_type
    uint32_t meta_val

cdef struct TrackState:
    const uint8_t* ptr
    const uint8_t* end
    uint32_t current_tick
    int last_status
    int track_index

cdef bint compareGpuNote(const GpuNote& a, const GpuNote& b) nogil:
    return a.on_time < b.on_time

cdef bint comparePlaybackEvent(const PlaybackEvent& a, const PlaybackEvent& b) nogil:
    return a.on_time < b.on_time


cdef inline uint32_t read_varlen(const uint8_t* p, const uint8_t* end, const uint8_t** out_next) nogil:
    cdef uint32_t value = 0
    cdef uint8_t byte
    while p < end:
        byte = p[0]
        value = (value << 7) | (byte & 0x7F)
        p += 1
        if (byte & 0x80) == 0:
            break
    out_next[0] = p
    return value


cdef inline bint next_raw_event(TrackState* ts, RawEvent* ev) nogil:
    cdef const uint8_t* p = ts.ptr
    cdef const uint8_t* end = ts.end
    cdef uint32_t delta
    cdef uint8_t status, byte
    cdef const uint8_t* next_p
    cdef uint32_t varlen
    if p >= end:
        return False

    varlen = read_varlen(p, end, &next_p)
    p = next_p
    ts.current_tick += varlen
    if p >= end:
        ts.ptr = p
        return False

    status = p[0]
    if status < 0x80:
        if ts.last_status == -1:
            ts.ptr = p + 1
            return next_raw_event(ts, ev)
        status = <uint8_t>ts.last_status
        p -= 1
    else:
        ts.last_status = status

    ev.abs_tick = ts.current_tick
    ev.event_type = status & 0xF0
    ev.channel = status & 0x0F
    ev.meta_type = 0
    ev.meta_val = 0

    p += 1
    if p >= end:
        ts.ptr = p
        return False

    if ev.event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
        if p + 1 >= end:
            ts.ptr = end
            return False
        ev.data1 = p[0]
        ev.data2 = p[1]
        ts.ptr = p + 2
        return True
    elif ev.event_type in (0xC0, 0xD0):
        ev.data1 = p[0]
        ev.data2 = 0
        ts.ptr = p + 1
        return True
    elif status == 0xFF:
        if p >= end:
            ts.ptr = end
            return False
        ev.event_type = status
        ev.meta_type = p[0]
        p += 1
        if p >= end:
            ts.ptr = end
            return False
        varlen = read_varlen(p, end, &next_p)
        p = next_p
        if ev.meta_type == 0x51 and varlen == 3 and p + 2 < end:
            ev.data1 = 0
            ev.data2 = 0
            ev.meta_val = (p[0] << 16) | (p[1] << 8) | p[2]
        ts.ptr = p + varlen
        return True
    elif status == 0xF0 or status == 0xF7:
        varlen = read_varlen(p, end, &next_p)
        ts.ptr = next_p + varlen
        return next_raw_event(ts, ev)
    else:
        ts.last_status = -1
        ts.ptr = p + 1
        return next_raw_event(ts, ev)


cdef inline void heap_push(vector[uint64_t]& h, uint64_t v) noexcept nogil:
    cdef size_t idx
    cdef size_t parent
    h.push_back(v)
    idx = h.size() - 1
    while idx > 0:
        parent = (idx - 1) >> 1
        if h[parent] <= h[idx]:
            break
        h[parent], h[idx] = h[idx], h[parent]
        idx = parent


cdef inline uint64_t heap_pop(vector[uint64_t]& h) noexcept nogil:
    cdef uint64_t ret = h[0]
    cdef size_t last = h.size() - 1
    cdef size_t idx = 0
    cdef size_t left, right, smallest
    h[0] = h[last]
    h.pop_back()
    while True:
        left = (idx << 1) + 1
        right = left + 1
        smallest = idx
        if left < h.size() and h[left] < h[smallest]:
            smallest = left
        if right < h.size() and h[right] < h[smallest]:
            smallest = right
        if smallest == idx:
            break
        h[smallest], h[idx] = h[idx], h[smallest]
        idx = smallest
    return ret


cdef class MidiParser:
    def __init__(self, str filename):
        self.filename = filename
        self.ticks_per_beat = 480
        
        self.note_data_for_gpu = np.empty((0,), dtype=GPU_NOTE_DTYPE)
        self.note_events_for_playback = np.empty((0,), dtype=PLAYBACK_EVENT_DTYPE)
        self.program_change_events = []
        self.pitch_bend_events = []
        self.control_change_events = []
        self.total_duration_sec = 0.0
        self.preferred_color_mode = "track"
        self.preferred_color_mode = "track"

    def count_total_events(self):
        """
        Quickly scans the MIDI file to count the total number of events
        without the overhead of full parsing. This is used for progress bars.
        """
        cdef TrackState ts
        cdef RawEvent raw_ev
        cdef bytes data_bytes
        cdef uint64_t total_event_count = 0

        with open(self.filename, 'rb') as f:
            if f.read(4) != b'MThd':
                return 0
            header_length = struct.unpack('>I', f.read(4))[0]
            header_data = f.read(header_length)
            file_format, num_tracks, _ = struct.unpack('>HHH', header_data)

            for _ in range(num_tracks):
                track_chunk_id = f.read(4)
                if not track_chunk_id:
                    break
                track_length = struct.unpack('>I', f.read(4))[0]
                
                if track_chunk_id == b'MTrk':
                    track_data = f.read(track_length)
                    data_bytes = track_data
                    
                    ts.ptr = <const uint8_t*> data_bytes
                    ts.end = ts.ptr + len(data_bytes)
                    ts.current_tick = 0
                    ts.last_status = -1
                    ts.track_index = 0

                    while next_raw_event(&ts, &raw_ev):
                        total_event_count += 1
                else:
                    f.seek(track_length, 1)
        
        return int(total_event_count)

    def parse(self, progress_queue=None, total_events=0, fallback_event_threshold=0):
        def _log(message):
            if progress_queue:
                if isinstance(message, dict):
                    progress_queue.put(('progress', message))
                else:
                    progress_queue.put(('progress', str(message)))
            else:
                print(message)

        self.note_data_for_gpu = np.empty((0,), dtype=GPU_NOTE_DTYPE)
        self.note_events_for_playback = np.empty((0,), dtype=PLAYBACK_EVENT_DTYPE)
        self.program_change_events = []
        self.pitch_bend_events = []
        self.control_change_events = []
        self.total_duration_sec = 0.0
        
        _log(f"Opening {self.filename}...")
        cdef double start_parse_time = time.monotonic()
        
        cdef list track_buffers = []
        cdef vector[TrackState] track_states
        cdef vector[RawEvent] current_events
        cdef int num_tracks, file_format
        
        with open(self.filename, 'rb') as f:
            if f.read(4) != b'MThd':
                raise ValueError("Not a valid MIDI file: 'MThd' chunk not found.")
            header_length = struct.unpack('>I', f.read(4))[0]
            header_data = f.read(header_length)
            
            file_format, num_tracks, self.ticks_per_beat = struct.unpack('>HHH', header_data)
            
            _log(f"Format: {file_format}, Tracks: {num_tracks}, Ticks/Beat: {self.ticks_per_beat}")
            
            if self.ticks_per_beat & 0x8000:
                frames_per_second = -((self.ticks_per_beat >> 8) - 256)
                ticks_per_frame = self.ticks_per_beat & 0xFF
                self.ticks_per_beat = frames_per_second * ticks_per_frame
                _log(f"SMPTE timecode found. Ticks/sec calculated as: {self.ticks_per_beat}")

            for track_idx in range(num_tracks):
                if progress_queue and num_tracks > 0:
                    load_fraction = 0.02 + (0.08 * (track_idx / max(1, num_tracks)))
                    _log({
                        "fraction": load_fraction,
                        "overlay": f"{load_fraction * 100:.1f}%",
                        "detail": f"Loading track {track_idx + 1:,} / {num_tracks:,}...",
                    })
                track_chunk_id = f.read(4)
                if not track_chunk_id:
                    _log(f"Warning: Reached end of file. Expected {num_tracks} tracks, found {track_idx}.")
                    break
                track_length = struct.unpack('>I', f.read(4))[0]
                track_data = f.read(track_length)
                
                if track_chunk_id == b'MTrk':
                    track_buffers.append(track_data)
                else:
                    _log(f"Skipping unknown chunk: {track_chunk_id.decode('ascii', 'ignore')}")

        if fallback_event_threshold and total_events > fallback_event_threshold:
            _log(
                f"Large MIDI detected ({total_events:,} events > {int(fallback_event_threshold):,} threshold). "
                f"Switching to low-memory parser..."
            )
            self._parse_low_memory_from_buffers(track_buffers, progress_queue=progress_queue, total_events=total_events)
            return

        cdef vector[GpuNote] temp_gpu_notes_vec
        cdef vector[PlaybackEvent] temp_playback_events_vec
        cdef vector[PlaybackEvent] temp_pitch_bend_vec
        cdef vector[PlaybackEvent] temp_control_change_vec
        cdef vector[NoteOnInfo] note_on_stack[128][16]
        cdef vector[uint64_t] event_heap
        event_heap.reserve(len(track_buffers))
        track_states.reserve(len(track_buffers))
        current_events.reserve(len(track_buffers))
        cdef TrackState ts
        cdef bytes data_bytes
        cdef RawEvent raw_ev
        cdef uint64_t heap_val
        cdef int i
        for i in range(len(track_buffers)):
            data_bytes = track_buffers[i]
            ts.ptr = <const uint8_t*> data_bytes
            ts.end = ts.ptr + len(data_bytes)
            ts.current_tick = 0
            ts.last_status = -1
            ts.track_index = i
            if next_raw_event(&ts, &raw_ev):
                track_states.push_back(ts)
                current_events.push_back(raw_ev)
                heap_val = (((<uint64_t>raw_ev.abs_tick) << 32) | <uint64_t>i)
                heap_push(event_heap, heap_val)
            else:
                track_states.push_back(ts)
                current_events.push_back(raw_ev)

        cdef double current_time_sec = 0.0
        cdef uint32_t current_tempo_usec = 500000
        cdef uint32_t last_event_tick = 0
        cdef double max_off_time = 0.0 
        cdef long total_event_count = 0
        cdef long last_log_event_count = 0
        cdef double loop_start_time = time.monotonic()
        cdef double last_log_time = loop_start_time
        cdef bint do_log = progress_queue is not None
        cdef uint32_t ticks_per_beat_local = self.ticks_per_beat
        cdef uint64_t mask_track = 0xFFFFFFFF

        _log("Parsing all track events (Cython)...")
        
        cdef uint32_t abs_tick, tick_delta
        cdef int track_num, event_type, channel, data1, data2, pitch_bend_value
        cdef double seconds_per_tick, on_time, off_time
        cdef uint8_t vel
        cdef uint16_t on_track_num
        cdef int first_note_track = -1
        cdef int second_note_track = -1
        cdef bint many_note_tracks = False
        cdef uint32_t channel_mask = 0
        cdef pair[uint8_t, uint8_t] key
        cdef NoteOnInfo on_info
        cdef GpuNote gpu_note
        cdef PlaybackEvent playback_event

        with nogil:
            while event_heap.size() > 0:
                heap_val = heap_pop(event_heap)
                abs_tick = <uint32_t>(heap_val >> 32)
                track_num = <int>(heap_val & mask_track)
                raw_ev = current_events[track_num]
                
                total_event_count += 1
                if do_log and (total_event_count - last_log_event_count) >= 5000:
                    with gil:
                        now = time.monotonic()
                        if (now - last_log_time) >= 0.1:
                            last_log_event_count = total_event_count
                            last_log_time = now
                            
                            elapsed_time = now - loop_start_time
                            evts_per_sec = total_event_count / elapsed_time if elapsed_time > 0 else 0
                            
                            eta_seconds = 0
                            if evts_per_sec > 0 and total_events > 0:
                                remaining_events = total_events - total_event_count
                                eta_seconds = remaining_events / evts_per_sec

                            parse_fraction = (total_event_count / total_events) if total_events > 0 else 0.0
                            overall_fraction = 0.10 + (0.75 * parse_fraction)
                            _log({
                                "fraction": overall_fraction,
                                "overlay": f"{overall_fraction * 100:.1f}%",
                                "detail": f"Parsing... {total_event_count:,} / {total_events:,} events (ETA: {eta_seconds:.1f}s)" if total_events > 0 else f"Parsing... {total_event_count:,} events",
                                "current": total_event_count,
                                "total": total_events,
                                "eta": eta_seconds
                            })

                tick_delta = abs_tick - last_event_tick
                if tick_delta > 0:
                    seconds_per_tick = current_tempo_usec / (ticks_per_beat_local * 1_000_000.0)
                    current_time_sec += tick_delta * seconds_per_tick

                last_event_tick = abs_tick
                event_type = raw_ev.event_type
                channel = raw_ev.channel
                data1 = raw_ev.data1
                data2 = raw_ev.data2

                if event_type == 0x90 and data2 > 0:
                    on_info.on_time_sec = current_time_sec
                    on_info.velocity = data2
                    on_info.track_num = track_num
                    note_on_stack[data1][channel].push_back(on_info)

                elif event_type == 0x80 or (event_type == 0x90 and data2 == 0):
                    if note_on_stack[data1][channel].size() > 0:
                        on_info = note_on_stack[data1][channel].back()
                        note_on_stack[data1][channel].pop_back()
                        
                        on_time = on_info.on_time_sec
                        vel = on_info.velocity
                        on_track_num = on_info.track_num
                        off_time = current_time_sec
                        
                        if off_time > max_off_time:
                            max_off_time = off_time

                        if first_note_track == -1:
                            first_note_track = on_track_num
                        elif on_track_num != first_note_track:
                            if second_note_track == -1:
                                second_note_track = on_track_num
                            elif on_track_num != second_note_track:
                                many_note_tracks = True
                        channel_mask |= (1 << channel)
                        
                        gpu_note.on_time = <np.float32_t>on_time
                        gpu_note.off_time = <np.float32_t>off_time
                        gpu_note.pitch = data1
                        gpu_note.velocity = vel
                        gpu_note.track = <uint8_t>(on_track_num & 0xFF)
                        gpu_note.padding = <uint8_t>channel
                        temp_gpu_notes_vec.push_back(gpu_note)
                        
                        playback_event.on_time = on_time
                        playback_event.off_time = off_time
                        playback_event.pitch = data1
                        playback_event.velocity = vel
                        playback_event.channel = channel
                        playback_event.track_num = on_track_num
                        temp_playback_events_vec.push_back(playback_event)

                elif event_type == 0xFF and raw_ev.meta_type == 0x51:
                    current_tempo_usec = raw_ev.meta_val
                
                elif event_type == 0xE0:
                    pitch_bend_value = (data2 << 7) | data1
                    playback_event.on_time = current_time_sec
                    playback_event.off_time = current_time_sec
                    playback_event.pitch = 0
                    playback_event.velocity = 0
                    playback_event.channel = channel
                    playback_event.track_num = pitch_bend_value
                    temp_pitch_bend_vec.push_back(playback_event)

                elif event_type == 0xB0:
                    playback_event.on_time = current_time_sec
                    playback_event.off_time = current_time_sec
                    playback_event.pitch = data1
                    playback_event.velocity = data2
                    playback_event.channel = channel
                    playback_event.track_num = 0
                    temp_control_change_vec.push_back(playback_event)

                ts = track_states[track_num]
                if next_raw_event(&ts, &raw_ev):
                    track_states[track_num] = ts
                    current_events[track_num] = raw_ev
                    heap_val = (((<uint64_t>raw_ev.abs_tick) << 32) | <uint64_t>track_num)
                    heap_push(event_heap, heap_val)

        _log("Note matching complete. Finalizing...")
        if progress_queue:
            _log({
                "fraction": 0.88,
                "overlay": "88.0%",
                "detail": "Finalizing parsed note data...",
            })
        
        cdef size_t n_gpu_notes = temp_gpu_notes_vec.size()
        cdef size_t n_playback_events = temp_playback_events_vec.size()
        
        if n_gpu_notes > 0:
            _log(f"Sorting {n_gpu_notes:,} notes...")
            std_sort(temp_gpu_notes_vec.begin(), temp_gpu_notes_vec.end(), compareGpuNote)
            
            self.note_data_for_gpu = np.empty((n_gpu_notes,), dtype=GPU_NOTE_DTYPE)
            memcpy(np.PyArray_DATA(self.note_data_for_gpu),
                   &temp_gpu_notes_vec[0],
                   n_gpu_notes * sizeof(GpuNote))
            self.preferred_color_mode = "channel" if ((not many_note_tracks) and channel_mask and (channel_mask & (channel_mask - 1))) else "track"
            
        if n_playback_events > 0:
            if progress_queue:
                _log({
                    "fraction": 0.94,
                    "overlay": "94.0%",
                    "detail": "Sorting playback events...",
                })
            _log(f"Sorting {n_playback_events:,} playback events...")
            std_sort(temp_playback_events_vec.begin(), temp_playback_events_vec.end(), comparePlaybackEvent)
            
            self.note_events_for_playback = np.empty((n_playback_events,), dtype=PLAYBACK_EVENT_DTYPE)
            memcpy(np.PyArray_DATA(self.note_events_for_playback),
                   &temp_playback_events_vec[0],
                   n_playback_events * sizeof(PlaybackEvent))
            
        if temp_pitch_bend_vec.size() > 0:
            self.pitch_bend_events = [(temp_pitch_bend_vec[i].on_time,
                                       temp_pitch_bend_vec[i].channel,
                                       temp_pitch_bend_vec[i].track_num)
                                      for i in range(temp_pitch_bend_vec.size())]
        if temp_control_change_vec.size() > 0:
            self.control_change_events = [(temp_control_change_vec[i].on_time,
                                           temp_control_change_vec[i].channel,
                                           temp_control_change_vec[i].pitch,
                                           temp_control_change_vec[i].velocity)
                                          for i in range(temp_control_change_vec.size())]
        self.total_duration_sec = max_off_time
        if progress_queue:
            _log({
                "fraction": 0.98,
                "overlay": "98.0%",
                "detail": "Building final MIDI data...",
            })

        track_buffers = []
        event_heap = vector[uint64_t]()
        track_states = vector[TrackState]()
        current_events = vector[RawEvent]()
        temp_gpu_notes_vec = vector[GpuNote]()
        temp_playback_events_vec = vector[PlaybackEvent]()
        temp_pitch_bend_vec = vector[PlaybackEvent]()
        temp_control_change_vec = vector[PlaybackEvent]()

        cdef double end_parse_time = time.monotonic()
        cdef double total_parse_time = end_parse_time - start_parse_time
        cdef double events_per_sec = total_event_count / total_parse_time if total_parse_time > 0 else 0
        
        _log(f"Finalized in {total_parse_time:.2f} seconds.")
        final_summary = (
            f"Notes: {n_gpu_notes:,} | "
            f"Duration: {self.total_duration_sec:.2f}s"
        )
        _log(final_summary)

    def _parse_low_memory_from_buffers(self, list track_buffers, progress_queue=None, total_events=0):
        def _log(message):
            if progress_queue:
                if isinstance(message, dict):
                    progress_queue.put(('progress', message))
                else:
                    progress_queue.put(('progress', str(message)))
            else:
                print(message)

        cdef double start_parse_time = time.monotonic()
        cdef vector[TrackState] track_states
        cdef vector[RawEvent] current_events
        cdef vector[uint64_t] event_heap
        cdef vector[GpuNote] temp_gpu_notes_vec
        cdef vector[PlaybackEvent] temp_playback_events_vec
        cdef vector[PlaybackEvent] temp_pitch_bend_vec
        cdef vector[PlaybackEvent] temp_control_change_vec
        cdef vector[NoteOnInfo] note_on_stack[128][16]
        cdef TrackState ts
        cdef bytes data_bytes
        cdef RawEvent raw_ev
        cdef uint64_t heap_val
        cdef int i
        cdef uint64_t mask_track = 0xFFFFFFFF

        cdef double current_time_sec = 0.0
        cdef uint32_t current_tempo_usec = 500000
        cdef uint32_t last_event_tick = 0
        cdef uint32_t ticks_per_beat_local = self.ticks_per_beat
        cdef double max_off_time = 0.0
        cdef long total_event_count = 0
        cdef long last_log_event_count = 0
        cdef double loop_start_time = time.monotonic()
        cdef double last_log_time = loop_start_time
        cdef bint do_log = progress_queue is not None

        cdef uint32_t abs_tick, tick_delta
        cdef int track_num, event_type, channel, data1, data2, pitch_bend_value
        cdef double seconds_per_tick, on_time, off_time
        cdef uint8_t vel
        cdef uint16_t on_track_num
        cdef int first_note_track = -1
        cdef int second_note_track = -1
        cdef bint many_note_tracks = False
        cdef uint32_t channel_mask = 0
        cdef NoteOnInfo on_info
        cdef GpuNote gpu_note
        cdef PlaybackEvent playback_event

        _log("Parsing all track events (Cython low-memory pass 1/2: notes)...")

        event_heap.reserve(len(track_buffers))
        track_states.reserve(len(track_buffers))
        current_events.reserve(len(track_buffers))
        for i in range(len(track_buffers)):
            data_bytes = track_buffers[i]
            ts.ptr = <const uint8_t*> data_bytes
            ts.end = ts.ptr + len(data_bytes)
            ts.current_tick = 0
            ts.last_status = -1
            ts.track_index = i
            if next_raw_event(&ts, &raw_ev):
                track_states.push_back(ts)
                current_events.push_back(raw_ev)
                heap_val = (((<uint64_t>raw_ev.abs_tick) << 32) | <uint64_t>i)
                heap_push(event_heap, heap_val)
            else:
                track_states.push_back(ts)
                current_events.push_back(raw_ev)

        with nogil:
            while event_heap.size() > 0:
                heap_val = heap_pop(event_heap)
                abs_tick = <uint32_t>(heap_val >> 32)
                track_num = <int>(heap_val & mask_track)
                raw_ev = current_events[track_num]

                total_event_count += 1
                if do_log and (total_event_count - last_log_event_count) >= 5000:
                    with gil:
                        now = time.monotonic()
                        if (now - last_log_time) >= 0.1:
                            last_log_event_count = total_event_count
                            last_log_time = now
                            elapsed_time = now - loop_start_time
                            evts_per_sec = total_event_count / elapsed_time if elapsed_time > 0 else 0
                            eta_seconds = 0
                            if evts_per_sec > 0 and total_events > 0:
                                remaining_events = total_events - total_event_count
                                eta_seconds = remaining_events / evts_per_sec
                            parse_fraction = (total_event_count / total_events) if total_events > 0 else 0.0
                            overall_fraction = 0.10 + (0.40 * parse_fraction)
                            _log({
                                "fraction": overall_fraction,
                                "overlay": f"{overall_fraction * 100:.1f}%",
                                "detail": f"Low-memory pass 1/2... {total_event_count:,} / {total_events:,} events (ETA: {eta_seconds:.1f}s)" if total_events > 0 else f"Low-memory pass 1/2... {total_event_count:,} events",
                                "current": total_event_count,
                                "total": total_events,
                                "eta": eta_seconds
                            })

                tick_delta = abs_tick - last_event_tick
                if tick_delta > 0:
                    seconds_per_tick = current_tempo_usec / (ticks_per_beat_local * 1_000_000.0)
                    current_time_sec += tick_delta * seconds_per_tick

                last_event_tick = abs_tick
                event_type = raw_ev.event_type
                channel = raw_ev.channel
                data1 = raw_ev.data1
                data2 = raw_ev.data2

                if event_type == 0x90 and data2 > 0:
                    on_info.on_time_sec = current_time_sec
                    on_info.velocity = data2
                    on_info.track_num = track_num
                    note_on_stack[data1][channel].push_back(on_info)

                elif event_type == 0x80 or (event_type == 0x90 and data2 == 0):
                    if note_on_stack[data1][channel].size() > 0:
                        on_info = note_on_stack[data1][channel].back()
                        note_on_stack[data1][channel].pop_back()

                        on_time = on_info.on_time_sec
                        vel = on_info.velocity
                        on_track_num = on_info.track_num
                        off_time = current_time_sec

                        if off_time > max_off_time:
                            max_off_time = off_time

                        if first_note_track == -1:
                            first_note_track = on_track_num
                        elif on_track_num != first_note_track:
                            if second_note_track == -1:
                                second_note_track = on_track_num
                            elif on_track_num != second_note_track:
                                many_note_tracks = True
                        channel_mask |= (1 << channel)

                        gpu_note.on_time = <np.float32_t>on_time
                        gpu_note.off_time = <np.float32_t>off_time
                        gpu_note.pitch = data1
                        gpu_note.velocity = vel
                        gpu_note.track = <uint8_t>(on_track_num & 0xFF)
                        gpu_note.padding = <uint8_t>channel
                        temp_gpu_notes_vec.push_back(gpu_note)

                elif event_type == 0xFF and raw_ev.meta_type == 0x51:
                    current_tempo_usec = raw_ev.meta_val

                ts = track_states[track_num]
                if next_raw_event(&ts, &raw_ev):
                    track_states[track_num] = ts
                    current_events[track_num] = raw_ev
                    heap_val = (((<uint64_t>raw_ev.abs_tick) << 32) | <uint64_t>track_num)
                    heap_push(event_heap, heap_val)

        _log("Low-memory pass 1/2 complete. Finalizing note data...")
        if progress_queue:
            _log({
                "fraction": 0.52,
                "overlay": "52.0%",
                "detail": "Sorting note data...",
            })

        cdef size_t n_gpu_notes = temp_gpu_notes_vec.size()
        cdef size_t n_playback_events = 0
        if n_gpu_notes > 0:
            std_sort(temp_gpu_notes_vec.begin(), temp_gpu_notes_vec.end(), compareGpuNote)
            self.note_data_for_gpu = np.empty((n_gpu_notes,), dtype=GPU_NOTE_DTYPE)
            memcpy(np.PyArray_DATA(self.note_data_for_gpu),
                   &temp_gpu_notes_vec[0],
                   n_gpu_notes * sizeof(GpuNote))
            self.preferred_color_mode = "channel" if ((not many_note_tracks) and channel_mask and (channel_mask & (channel_mask - 1))) else "track"
        else:
            self.note_data_for_gpu = np.empty((0,), dtype=GPU_NOTE_DTYPE)
            self.preferred_color_mode = "track"

        temp_gpu_notes_vec = vector[GpuNote]()
        event_heap = vector[uint64_t]()
        track_states = vector[TrackState]()
        current_events = vector[RawEvent]()

        current_time_sec = 0.0
        current_tempo_usec = 500000
        last_event_tick = 0
        total_event_count = 0
        last_log_event_count = 0
        loop_start_time = time.monotonic()
        last_log_time = loop_start_time

        _log("Parsing all track events (Cython low-memory pass 2/2: playback)...")

        event_heap.reserve(len(track_buffers))
        track_states.reserve(len(track_buffers))
        current_events.reserve(len(track_buffers))
        for data1 in range(128):
            for channel in range(16):
                note_on_stack[data1][channel].clear()
        for i in range(len(track_buffers)):
            data_bytes = track_buffers[i]
            ts.ptr = <const uint8_t*> data_bytes
            ts.end = ts.ptr + len(data_bytes)
            ts.current_tick = 0
            ts.last_status = -1
            ts.track_index = i
            if next_raw_event(&ts, &raw_ev):
                track_states.push_back(ts)
                current_events.push_back(raw_ev)
                heap_val = (((<uint64_t>raw_ev.abs_tick) << 32) | <uint64_t>i)
                heap_push(event_heap, heap_val)
            else:
                track_states.push_back(ts)
                current_events.push_back(raw_ev)

        with nogil:
            while event_heap.size() > 0:
                heap_val = heap_pop(event_heap)
                abs_tick = <uint32_t>(heap_val >> 32)
                track_num = <int>(heap_val & mask_track)
                raw_ev = current_events[track_num]

                total_event_count += 1
                if do_log and (total_event_count - last_log_event_count) >= 5000:
                    with gil:
                        now = time.monotonic()
                        if (now - last_log_time) >= 0.1:
                            last_log_event_count = total_event_count
                            last_log_time = now
                            elapsed_time = now - loop_start_time
                            evts_per_sec = total_event_count / elapsed_time if elapsed_time > 0 else 0
                            eta_seconds = 0
                            if evts_per_sec > 0 and total_events > 0:
                                remaining_events = total_events - total_event_count
                                eta_seconds = remaining_events / evts_per_sec
                            parse_fraction = (total_event_count / total_events) if total_events > 0 else 0.0
                            overall_fraction = 0.54 + (0.34 * parse_fraction)
                            _log({
                                "fraction": overall_fraction,
                                "overlay": f"{overall_fraction * 100:.1f}%",
                                "detail": f"Low-memory pass 2/2... {total_event_count:,} / {total_events:,} events (ETA: {eta_seconds:.1f}s)" if total_events > 0 else f"Low-memory pass 2/2... {total_event_count:,} events",
                                "current": total_event_count,
                                "total": total_events,
                                "eta": eta_seconds
                            })

                tick_delta = abs_tick - last_event_tick
                if tick_delta > 0:
                    seconds_per_tick = current_tempo_usec / (ticks_per_beat_local * 1_000_000.0)
                    current_time_sec += tick_delta * seconds_per_tick

                last_event_tick = abs_tick
                event_type = raw_ev.event_type
                channel = raw_ev.channel
                data1 = raw_ev.data1
                data2 = raw_ev.data2

                if event_type == 0x90 and data2 > 0:
                    on_info.on_time_sec = current_time_sec
                    on_info.velocity = data2
                    on_info.track_num = track_num
                    note_on_stack[data1][channel].push_back(on_info)

                elif event_type == 0x80 or (event_type == 0x90 and data2 == 0):
                    if note_on_stack[data1][channel].size() > 0:
                        on_info = note_on_stack[data1][channel].back()
                        note_on_stack[data1][channel].pop_back()

                        playback_event.on_time = on_info.on_time_sec
                        playback_event.off_time = current_time_sec
                        playback_event.pitch = data1
                        playback_event.velocity = on_info.velocity
                        playback_event.channel = channel
                        playback_event.track_num = on_info.track_num
                        temp_playback_events_vec.push_back(playback_event)

                elif event_type == 0xFF and raw_ev.meta_type == 0x51:
                    current_tempo_usec = raw_ev.meta_val

                elif event_type == 0xE0:
                    pitch_bend_value = (data2 << 7) | data1
                    playback_event.on_time = current_time_sec
                    playback_event.off_time = current_time_sec
                    playback_event.pitch = 0
                    playback_event.velocity = 0
                    playback_event.channel = channel
                    playback_event.track_num = pitch_bend_value
                    temp_pitch_bend_vec.push_back(playback_event)

                elif event_type == 0xB0:
                    playback_event.on_time = current_time_sec
                    playback_event.off_time = current_time_sec
                    playback_event.pitch = data1
                    playback_event.velocity = data2
                    playback_event.channel = channel
                    playback_event.track_num = 0
                    temp_control_change_vec.push_back(playback_event)

                ts = track_states[track_num]
                if next_raw_event(&ts, &raw_ev):
                    track_states[track_num] = ts
                    current_events[track_num] = raw_ev
                    heap_val = (((<uint64_t>raw_ev.abs_tick) << 32) | <uint64_t>track_num)
                    heap_push(event_heap, heap_val)

        _log("Low-memory pass 2/2 complete. Finalizing playback data...")
        if progress_queue:
            _log({
                "fraction": 0.92,
                "overlay": "92.0%",
                "detail": "Sorting playback events...",
            })

        n_playback_events = temp_playback_events_vec.size()
        if n_playback_events > 0:
            std_sort(temp_playback_events_vec.begin(), temp_playback_events_vec.end(), comparePlaybackEvent)
            self.note_events_for_playback = np.empty((n_playback_events,), dtype=PLAYBACK_EVENT_DTYPE)
            memcpy(np.PyArray_DATA(self.note_events_for_playback),
                   &temp_playback_events_vec[0],
                   n_playback_events * sizeof(PlaybackEvent))
        else:
            self.note_events_for_playback = np.empty((0,), dtype=PLAYBACK_EVENT_DTYPE)

        if temp_pitch_bend_vec.size() > 0:
            self.pitch_bend_events = [(temp_pitch_bend_vec[i].on_time,
                                       temp_pitch_bend_vec[i].channel,
                                       temp_pitch_bend_vec[i].track_num)
                                      for i in range(temp_pitch_bend_vec.size())]
        else:
            self.pitch_bend_events = []

        if temp_control_change_vec.size() > 0:
            self.control_change_events = [(temp_control_change_vec[i].on_time,
                                           temp_control_change_vec[i].channel,
                                           temp_control_change_vec[i].pitch,
                                           temp_control_change_vec[i].velocity)
                                          for i in range(temp_control_change_vec.size())]
        else:
            self.control_change_events = []

        self.total_duration_sec = max_off_time
        if progress_queue:
            _log({
                "fraction": 0.98,
                "overlay": "98.0%",
                "detail": "Building final MIDI data...",
            })

        cdef double end_parse_time = time.monotonic()
        cdef double total_parse_time = end_parse_time - start_parse_time
        _log(f"Low-memory parser finalized in {total_parse_time:.2f} seconds.")
        _log(
            f"Notes: {n_gpu_notes:,} | Playback Events: {n_playback_events:,} | "
            f"Duration: {self.total_duration_sec:.2f}s"
        )

    cdef _get_var_len(self, unsigned char* data, int i, int max_len):
        """Reads a MIDI variable-length quantity from a raw char* buffer."""
        cdef uint32_t value = 0
        cdef int start_i = i
        cdef uint8_t byte
        
        while True:
            if i >= max_len:
                break
            byte = data[i]
            value = (value << 7) | (byte & 0x7F)
            i += 1
            if not (byte & 0x80): 
                break
        return value, (i - start_i)

    def _stream_track_events(self, bytes data_bytes):
        """ 
        A Cython-optimized generator that yields raw events from a track.
        (absolute_tick, (event_type, channel, data1, data2)) 
        """
        cdef int i = 0
        cdef int length = len(data_bytes)
        cdef int current_tick = 0
        cdef int last_status = -1 # Use -1 to indicate no status
        
        cdef uint32_t delta_ticks, var_len_val, tempo
        cdef int bytes_read, status, event_type, channel, meta_type, start_byte
        cdef int data1, data2
        cdef uint8_t byte
        
        while i < length:
            value, b_read = 0, 0
            while True:
                if i + b_read >= length:
                    break
                byte = data_bytes[i + b_read]
                value = (value << 7) | (byte & 0x7F)
                b_read += 1
                if not (byte & 0x80): 
                    break
            delta_ticks, bytes_read = value, b_read
            
            i += bytes_read
            current_tick += delta_ticks
            
            if i >= length: break
            status = data_bytes[i]
            
            if status < 0x80: # Running Status
                if last_status == -1:
                    i += 1
                    continue
                status = last_status
                i -= 1 # Rewind
            else:
                last_status = status
            
            event_type = status & 0xF0
            channel = status & 0x0F
            i += 1

            if i >= length: break
            
            if event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0): # 2 Data Byte
                if i + 1 >= length: break
                data1 = data_bytes[i]
                data2 = data_bytes[i+1]
                yield (current_tick, (event_type, channel, data1, data2))
                i += 2
            elif event_type in (0xC0, 0xD0): # 1 Data Byte
                data1 = data_bytes[i]
                yield (current_tick, (event_type, channel, data1, 0))
                i += 1
            elif status == 0xFF: # Meta Event
                if i >= length: break
                meta_type = data_bytes[i]
                i += 1
                
                value, b_read = 0, 0
                while True:
                    if i + b_read >= length:
                        break
                    byte = data_bytes[i + b_read]
                    value = (value << 7) | (byte & 0x7F)
                    b_read += 1
                    if not (byte & 0x80): 
                        break
                var_len_val, bytes_read = value, b_read
                
                i += bytes_read
                start_byte = i
                
                if meta_type == 0x51 and var_len_val == 3: # Set Tempo
                    tempo = (data_bytes[start_byte] << 16) | (data_bytes[start_byte+1] << 8) | data_bytes[start_byte+2]
                    yield (current_tick, (status, 0, meta_type, tempo))
                
                i += var_len_val # Skip meta event data
                
            elif status == 0xF0 or status == 0xF7: # System Exclusive
                value, b_read = 0, 0
                while True:
                    if i + b_read >= length:
                        break
                    byte = data_bytes[i + b_read]
                    value = (value << 7) | (byte & 0x7F)
                    b_read += 1
                    if not (byte & 0x80): 
                        break
                var_len_val, bytes_read = value, b_read
                
                i += bytes_read
                i += var_len_val
            else:
                if last_status != -1:
                    last_status = -1
                continue
