try:
    from midi_parser_cython import MidiParser, GPU_NOTE_DTYPE, PLAYBACK_EVENT_DTYPE
    print("Loaded Cython MIDI parser.")

except ImportError:
    print("="*50)
    print("WARNING: Could not import 'midi_parser_cython'.")
    print("Falling back to slow, pure-Python MIDI parser.")
    print("Please run 'build.bat' to compile the Cython version.")
    print("="*50)
    
    import struct
    import heapq
    import numpy as np
    import time
    from tqdm import tqdm

    GPU_NOTE_DTYPE = np.dtype([
        ('on_time', 'f4'),
        ('off_time', 'f4'),
        ('pitch', 'u1'),
        ('velocity', 'u1'),
        ('track', 'u1'),
        ('padding', 'u1')
    ], align=True)
    
    PLAYBACK_EVENT_DTYPE = np.dtype([
        ('on_time', 'f8'), ('off_time', 'f8'), ('pitch', 'u1'),
        ('velocity', 'u1'), ('channel', 'u1'), ('track_num', 'u2')
    ], align=True)
    

    class MidiParser:
        """
        A highly optimized MIDI file parser designed for large "Black MIDI" files.
        It minimizes memory usage by directly creating the data structures required
        by the visualizer, bypassing large intermediate Python lists.
        """
        def __init__(self, filename):
            self.filename = filename
            self.ticks_per_beat = 480
            
            self.note_data_for_gpu = None
            self.note_events_for_playback_list = []
            self.note_events_for_playback = None
            self.program_change_events = []
            self.pitch_bend_events = []
            self.control_change_events = []
            self.total_duration_sec = 0.0
            self.preferred_color_mode = "track"

        def count_total_events(self):
            return 0

        def parse(self, progress_queue=None, total_events=0, fallback_event_threshold=0):
            def _log(message):
                if progress_queue:
                    progress_queue.put(('progress', str(message)))
                else:
                    print(message)

            self.note_data_for_gpu = None
            self.note_events_for_playback_list = []
            self.note_events_for_playback = None
            self.program_change_events = []
            self.pitch_bend_events = []
            self.control_change_events = []
            self.total_duration_sec = 0.0
            self.preferred_color_mode = "track"
            
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
                    if progress_queue and num_tracks > 0:
                        load_fraction = 0.02 + (0.08 * (track_num / max(1, num_tracks)))
                        _log({
                            "fraction": load_fraction,
                            "overlay": f"{load_fraction * 100:.1f}%",
                            "detail": f"Loading track {track_num + 1:,} / {num_tracks:,}...",
                        })
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

            event_heap = []
            for i, stream in enumerate(track_event_streams):
                try:
                    first_tick, first_event = next(stream)
                    heapq.heappush(event_heap, (first_tick, i, first_event))
                except StopIteration:
                    continue

            current_time_sec = 0.0
            current_tempo_usec = 500000
            last_event_tick = 0
            note_on_dict = {}
            
            max_off_time = 0.0 

            temp_gpu_notes = []
            total_event_count = 0
            first_note_track = None
            second_note_track = None
            many_note_tracks = False
            channel_mask = 0
            
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
                        eta_seconds = 0
                        if evts_per_sec > 0 and total_events > 0:
                            remaining_events = max(0, total_events - total_event_count)
                            eta_seconds = remaining_events / evts_per_sec
                        parse_fraction = (total_event_count / total_events) if total_events > 0 else 0.0
                        overall_fraction = 0.10 + (0.75 * parse_fraction)
                        _log({
                            "fraction": overall_fraction,
                            "overlay": f"{overall_fraction * 100:.1f}%",
                            "detail": f"Parsing... {total_event_count:,} / {total_events:,} events (ETA: {eta_seconds:.1f}s)" if total_events > 0 else f"Parsing... {total_event_count:,} events",
                            "current": total_event_count,
                            "total": total_events,
                            "eta": eta_seconds,
                        })


                tick_delta = abs_tick - last_event_tick
                if tick_delta > 0:
                    seconds_per_tick = current_tempo_usec / (self.ticks_per_beat * 1_000_000)
                    current_time_sec += tick_delta * seconds_per_tick
                
                last_event_tick = abs_tick

                event_type, channel, data1, data2 = event

                if event_type == 0x90 and data2 > 0:
                    key = (data1, channel)
                    note_on_dict.setdefault(key, []).append((current_time_sec, data2, track_num))

                elif event_type == 0x80 or (event_type == 0x90 and data2 == 0):
                    key = (data1, channel)
                    if key in note_on_dict and note_on_dict[key]:
                        on_time, vel, on_track_num = note_on_dict[key].pop(0)
                        off_time = current_time_sec
                        
                        if off_time > max_off_time:
                            max_off_time = off_time

                        if first_note_track is None:
                            first_note_track = on_track_num
                        elif on_track_num != first_note_track:
                            if second_note_track is None:
                                second_note_track = on_track_num
                            elif on_track_num != second_note_track:
                                many_note_tracks = True
                        channel_mask |= (1 << channel)
                        
                        temp_gpu_notes.append((on_time, off_time, data1, vel, on_track_num % 256, channel))
                        self.note_events_for_playback_list.append((on_time, off_time, data1, vel, channel, on_track_num))

                        if not note_on_dict[key]:
                            del note_on_dict[key]

                elif event_type == 0xFF and data1 == 0x51:
                    current_tempo_usec = data2
                
                elif event_type == 0xC0:
                    self.program_change_events.append((current_time_sec, channel, data1))

                elif event_type == 0xE0:
                    pitch_bend_value = (data2 << 7) | data1
                    self.pitch_bend_events.append((current_time_sec, channel, pitch_bend_value))
                elif event_type == 0xB0:
                    self.control_change_events.append((current_time_sec, channel, data1, data2))

                try:
                    next_tick, next_event = next(track_event_streams[track_num])
                    heapq.heappush(event_heap, (next_tick, track_num, next_event))
                except StopIteration:
                    continue

            _log("Note matching complete. Finalizing...")
            if progress_queue:
                _log({
                    "fraction": 0.88,
                    "overlay": "88.0%",
                    "detail": "Finalizing parsed note data...",
                })
            
            if temp_gpu_notes:
                _log(f"Sorting {len(temp_gpu_notes)} notes...")
                self.note_data_for_gpu = np.array(temp_gpu_notes, dtype=GPU_NOTE_DTYPE)
                self.note_data_for_gpu.sort(order='on_time')
                use_channel_colors = (not many_note_tracks) and channel_mask and (channel_mask & (channel_mask - 1))
                self.preferred_color_mode = "channel" if use_channel_colors else "track"
            else:
                self.note_data_for_gpu = np.empty((0,), dtype=GPU_NOTE_DTYPE)
                
            if self.note_events_for_playback_list:
                if progress_queue:
                    _log({
                        "fraction": 0.94,
                        "overlay": "94.0%",
                        "detail": "Sorting playback events...",
                    })
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
            if progress_queue:
                _log({
                    "fraction": 0.98,
                    "overlay": "98.0%",
                    "detail": "Building final MIDI data...",
                })

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
