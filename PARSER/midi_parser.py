# filename: midi_parser.py
#
# This file is now a 'shim' that imports the optimized Cython version.
# Your main application (midiplayer.py) can keep importing 'MidiParser'
# from this file without any changes.

try:
    # --- [CHANGED] ---
    # Attempt to import the fast Cython module
    from midi_parser_cython import MidiParser, GPU_NOTE_DTYPE, PLAYBACK_EVENT_DTYPE
    print("Loaded optimized Cython MIDI parser.")
    # --- [END CHANGED] ---

except ImportError:
    print("="*50)
    print("WARNING: Could not import 'midi_parser_cython'.")
    print("Falling back to slow, pure-Python MIDI parser.")
    print("Please run 'build.bat' to compile the Cython version.")
    print("="*50)
    
    # --- This is your original Python code as a fallback ---
    import struct
    import heapq
    import numpy as np
    import time # For parse timer
    from tqdm import tqdm

    GPU_NOTE_DTYPE = np.dtype([
        ('on_time', 'f4'),      # vec2 note_times
        ('off_time', 'f4'),     #
        ('pitch', 'u1'),        # uvec3 note_info (byte 0) -> this is data1
        ('velocity', 'u1'),     # uvec3 note_info (byte 1) -> this is data2
        ('track', 'u1'),        # uvec3 note_info (byte 2) -> this is the channel
        ('padding', 'u1')       # Padding to align to 4 bytes (total 12 bytes)
    ], align=True) # <-- [CHANGED] align=True for safety
    
    # [NEW] Add the playback DType for compatibility
    PLAYBACK_EVENT_DTYPE = np.dtype([
        ('on_time', 'f8'), ('off_time', 'f8'), ('pitch', 'u1'),
        ('velocity', 'u1'), ('channel', 'u1'), ('track_num', 'u2')
    ], align=True) # <-- [FIX] align=True is critical
    

    class MidiParser:
        """
        A highly optimized MIDI file parser designed for large "Black MIDI" files.
        It minimizes memory usage by directly creating the data structures required
        by the visualizer, bypassing large intermediate Python lists.
        """
        def __init__(self, filename):
            self.filename = filename
            self.ticks_per_beat = 480
            
            # Public attributes after parsing
            self.note_data_for_gpu = None
            self.note_events_for_playback_list = [] # <-- Changed name
            self.note_events_for_playback = None # <-- Final np array
            self.program_change_events = []
            self.pitch_bend_events = []
            self.total_duration_sec = 0.0 # Will be set at end of parse()

        def count_total_events(self):
            # Pure python fallback doesn't need a progress bar,
            # as it's not meant for huge files.
            return 0

        def parse(self, progress_queue=None, total_events=0):
            def _log(message):
                if progress_queue:
                    # Fallback parser doesn't send dicts, just strings
                    progress_queue.put(('progress', str(message)))
                else:
                    print(message)

            # Reset all data attributes
            self.note_data_for_gpu = None
            self.note_events_for_playback_list = []
            self.note_events_for_playback = None
            self.program_change_events = []
            self.pitch_bend_events = []
            self.total_duration_sec = 0.0
            
            _log(f"Opening {self.filename}...")
            start_parse_time = time.monotonic()
            
            with open(self.filename, 'rb') as f:
                header_chunk_id = f.read(4)
                if header_chunk_id != b'MThd':
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


                track_event_streams = []
                for track_num in range(num_tracks):
                    track_chunk_id = f.read(4)
                    
                    if not track_chunk_id:
                        _log(f"Warning: Reached end of file. Expected {num_tracks} tracks, found {track_num}.")
                        break
                        
                    track_length = struct.unpack('>I', f.read(4))[0]
                    track_data = f.read(track_length)
                    
                    if track_chunk_id == b'MTrk':
                        track_event_streams.append(self._stream_track_events(track_data))
                    else:
                        _log(f"Skipping unknown chunk: {track_chunk_id.decode('ascii', 'ignore')}")

            # Main processing loop
            event_heap = []
            for i, stream in enumerate(track_event_streams):
                try:
                    first_tick, first_event = next(stream)
                    heapq.heappush(event_heap, (first_tick, i, first_event))
                except StopIteration:
                    continue # Ignore empty tracks

            current_time_sec = 0.0
            current_tempo_usec = 500000 # MIDI default (120 BPM)
            last_event_tick = 0
            note_on_dict = {}  # Key: (note, channel), Value: list of (on_time_sec, velocity, track_num)
            
            max_off_time = 0.0 

            temp_gpu_notes = []
            total_event_count = 0
            
            _log("Parsing all track events (Python Fallback)...")
            loop_start_time = time.monotonic()
            last_log_event_count = 0

            while event_heap:
                abs_tick, track_num, event = heapq.heappop(event_heap)
                
                total_event_count += 1
                if progress_queue and (total_event_count - last_log_event_count) >= 25000:
                    last_log_event_count = total_event_count
                    now = time.monotonic()
                    elapsed = now - loop_start_time
                    if elapsed > 0.5:
                        evts_per_sec = total_event_count / elapsed
                        _log(f"Parsing... {total_event_count:,} events ({evts_per_sec:,.0f} evt/s)")


                tick_delta = abs_tick - last_event_tick
                if tick_delta > 0:
                    seconds_per_tick = current_tempo_usec / (self.ticks_per_beat * 1_000_000)
                    current_time_sec += tick_delta * seconds_per_tick
                
                last_event_tick = abs_tick

                event_type, channel, data1, data2 = event

                if event_type == 0x90 and data2 > 0:  # Note On
                    key = (data1, channel)
                    note_on_dict.setdefault(key, []).append((current_time_sec, data2, track_num))

                elif event_type == 0x80 or (event_type == 0x90 and data2 == 0):  # Note Off
                    key = (data1, channel)
                    if key in note_on_dict and note_on_dict[key]:
                        on_time, vel, on_track_num = note_on_dict[key].pop(0)
                        off_time = current_time_sec
                        
                        if off_time > max_off_time:
                            max_off_time = off_time
                        
                        temp_gpu_notes.append((on_time, off_time, data1, vel, channel, 0))
                        self.note_events_for_playback_list.append((on_time, off_time, data1, vel, channel, on_track_num))

                        if not note_on_dict[key]:
                            del note_on_dict[key]

                elif event_type == 0xFF and data1 == 0x51:  # Set Tempo
                    current_tempo_usec = data2
                
                elif event_type == 0xE0: # Pitch Bend
                    pitch_bend_value = (data2 << 7) | data1
                    self.pitch_bend_events.append((current_time_sec, channel, pitch_bend_value))

                try:
                    next_tick, next_event = next(track_event_streams[track_num])
                    heapq.heappush(event_heap, (next_tick, track_num, next_event))
                except StopIteration:
                    continue # This track is finished

            # Finalization
            _log("Note matching complete. Finalizing...")
            
            if temp_gpu_notes:
                _log(f"Sorting {len(temp_gpu_notes)} notes...")
                self.note_data_for_gpu = np.array(temp_gpu_notes, dtype=GPU_NOTE_DTYPE)
                self.note_data_for_gpu.sort(order='on_time') # Sort after creation
            else:
                self.note_data_for_gpu = np.empty((0,), dtype=GPU_NOTE_DTYPE)
                
            if self.note_events_for_playback_list:
                _log(f"Sorting {len(self.note_events_for_playback_list)} playback events...")
                self.note_events_for_playback = np.array(
                    self.note_events_for_playback_list,
                    dtype=PLAYBACK_EVENT_DTYPE
                )
                self.note_events_for_playback.sort(order='on_time')
            else:
                self.note_events_for_playback = np.empty((0,), dtype=PLAYBACK_EVENT_DTYPE)
            
            self.note_events_for_playback_list.clear() 
            
            self.total_duration_sec = max_off_time

            end_parse_time = time.monotonic()
            total_parse_time = end_parse_time - start_parse_time
            events_per_sec = total_event_count / total_parse_time if total_parse_time > 0 else 0
            
            _log(f"Finalized in {total_parse_time:.2f} seconds.")
            final_summary = (
                f"Notes: {len(self.note_data_for_gpu):,} | "
                f"Events: {total_event_count:,} ({events_per_sec:,.0f} evt/s) | "
                f"Duration: {self.total_duration_sec:.2f}s"
            )
            _log(final_summary)


        def _stream_track_events(self, data):
            """ 
            A generator that yields raw events from a track as tuples of:
            (absolute_tick, (event_type, channel, data1, data2)) 
            """
            i = 0
            current_tick = 0
            last_status = None # For running status
            
            while i < len(data):
                delta_ticks, bytes_read = self._get_var_len(data, i)
                i += bytes_read
                current_tick += delta_ticks

                event_tuple = None
                
                if i >= len(data): break
                status = data[i]
                
                if status < 0x80:
                    if last_status is None:
                        i += 1
                        continue
                    status = last_status
                    i -= 1 # Rewind
                else:
                    last_status = status
                
                event_type = status & 0xF0
                channel = status & 0x0F
                i += 1

                if i >= len(data): break
                
                if event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    if i + 1 >= len(data): break
                    data1 = data[i]
                    data2 = data[i+1]
                    event_tuple = (event_type, channel, data1, data2)
                    i += 2
                elif event_type in (0xC0, 0xD0):
                    data1 = data[i]
                    event_tuple = (event_type, channel, data1, 0)
                    i += 1
                elif status == 0xFF:
                    if i >= len(data): break
                    meta__type = data[i]
                    i += 1
                    
                    length, bytes_read_len = self._get_var_len(data, i)
                    i += bytes_read_len
                    start_byte = i
                    
                    if meta_type == 0x51 and length == 3: # Set Tempo
                        tempo = int.from_bytes(data[start_byte : start_byte + length], 'big')
                        event_tuple = (status, 0, meta_type, tempo)
                    
                    i += length
                    
                elif status == 0xF0 or status == 0xF7:
                    length, bytes_read_len = self._get_var_len(data, i)
                    i += bytes_read_len
                    i += length
                else:
                    if last_status is not None:
                        last_status = None
                    continue
                
                if event_tuple:
                    yield (current_tick, event_tuple)

        def _get_var_len(self, data, start):
            """Reads a MIDI variable-length quantity."""
            value, i = 0, 0
            while True:
                if start + i >= len(data):
                    return value, i
                byte = data[start + i]
                value = (value << 7) | (byte & 0x7F)
                i += 1
                if not (byte & 0x80): 
                    break
            return value, i