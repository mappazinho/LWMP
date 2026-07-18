import os
import gc
import sys
import shutil
import threading
import time
import types
from collections import deque
from queue import Empty

import numpy as np


class PlayerController:
    def __init__(
        self,
        config,
        script_dir,
        bass_engine_cls,
        omni_engine_cls,
        save_config_fn,
        debug=False,
        platform_name=os.name,
    ):
        self.config = config
        self.script_dir = script_dir
        self.bass_engine_cls = bass_engine_cls
        self.omni_engine_cls = omni_engine_cls
        self.save_config = save_config_fn
        self.debug = debug
        self.platform_name = platform_name

        self.parsed_midi = None
        self.total_song_notes = 0
        self.total_song_duration = 0.0
        self.max_nps = 0
        self.max_nps_spikes = []
        self.active_midi_backend = None
        self.active_backend_mode = None

        self.playing = False
        self.paused = False
        self.playback_start_time = 0.0
        self.paused_at_time = 0.0
        self.total_paused_duration = 0.0
        self.notes_played_count = 0
        self.last_processed_event_time = 0.0
        self.current_lag = 0.0
        self.buffered_playback_start_offset = 0.0
        self.current_playback_time_for_threads = 0.0
        self.playback_speed = 1.0
        self.recovery_active = False
        self.recovery_buffer_level = 0.0
        self.recovery_buffer_target = 4.0

        self.nps_event_timestamps = deque()
        self.last_nps_graph_update_time = 0.0
        self.last_lag_update_time = 0.0
        self.last_lag_value = 0.0
        self.slowdown_percentage = 0.0

        self.is_seeking = False
        self.paused_for_seeking = False
        self.seek_request_time = None
        self.parser_process = None
        self.parser_queue = None

    def _release_parsed_midi_storage(self):
        temp_dir = getattr(self.parsed_midi, "_backing_temp_dir", None) if self.parsed_midi is not None else None
        if self.parsed_midi is not None:
            for attr in ('note_data_for_gpu', 'note_events_for_playback', 'sorted_off_times', 'tempo_times', 'tempo_bpms'):
                try:
                    arr = getattr(self.parsed_midi, attr, None)
                    if arr is not None:
                        try:
                            arr._mmap.close()
                        except Exception:
                            pass
                    setattr(self.parsed_midi, attr, None)
                except Exception:
                    pass
        self.parsed_midi = None
        gc.collect()
        gc.collect()
        if temp_dir and os.path.isdir(temp_dir):
            for f in os.listdir(temp_dir):
                fp = os.path.join(temp_dir, f)
                try:
                    os.remove(fp)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    def init_midi_backends(
        self,
        volume,
        voices,
        set_status,
        refresh_status=None,
        prompt_info=None,
        prompt_warning=None,
        prompt_error=None,
        pick_soundfont=None,
        launch_sweep=None,
    ):
        load_from_path_pref = self.config["audio"].get("omnimidi_load_preference", "local")
        bundled_synth_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else self.script_dir
        bundled_omnimidi_exists = os.path.exists(os.path.join(bundled_synth_dir, "SYNTH.dll"))
        if load_from_path_pref == "local" and not bundled_omnimidi_exists:
            load_from_path_pref = "path"
            self.config["audio"]["omnimidi_load_preference"] = "path"

        if load_from_path_pref == "bassmidi":
            if self.bass_engine_cls:
                try:
                    print("Initializing BASSMIDI Engine (Buffered)...")
                    sf_path = self.config["audio"].get("soundfont_path")
                    if sf_path and not os.path.exists(sf_path):
                        missing_name = os.path.basename(sf_path)
                        if prompt_info:
                            prompt_info(
                                "SoundFont Missing",
                                f"Configured SoundFont ({missing_name}) not found.\nPlease select a SoundFont (.sf2 or .sfz) to use.",
                            )
                        sf_path = None
                        self.config["audio"]["soundfont_path"] = None
                        self.save_config(self.config)

                    if not sf_path:
                        if prompt_info:
                            prompt_info(
                                "SoundFont Missing",
                                "No SoundFont is configured.\nPlease select a SoundFont (.sf2 or .sfz) to use.",
                            )
                        sf_path = pick_soundfont() if pick_soundfont else None
                        if sf_path:
                            self.config["audio"]["soundfont_path"] = sf_path
                            self.save_config(self.config)
                        else:
                            if prompt_warning:
                                prompt_warning("No SoundFont", "No SoundFont selected. Playback will be silent.")
                            self.config["audio"]["soundfont_path"] = None
                            self.save_config(self.config)
                            sf_path = None

                    sf_ext = os.path.splitext(sf_path)[1].lower() if sf_path else ""
                    is_sfz = sf_ext == ".sfz"
                    self.active_midi_backend = self.bass_engine_cls(
                        {}, soundfont_path=sf_path, buffering=True, debug=self.debug
                    )
                    self.active_midi_backend.set_volume(volume)
                    self.active_midi_backend.set_voices(voices)

                    if self.active_midi_backend.midi_stream:
                        self.active_backend_mode = "bassmidi"
                        set_status("BASSMIDI Engine Initialized (Buffered).")
                        if launch_sweep:
                            launch_sweep(self.active_midi_backend)
                        return self.active_midi_backend
                except Exception as e:
                    print(f"BASSMIDI Init Failed: {e}")
                    self.active_midi_backend = None
                    self.active_backend_mode = None
            elif prompt_error:
                prompt_error("Error", "BASSMIDI libraries not found or failed to load.")

        if self.platform_name == "nt" and self.omni_engine_cls is not None:
            should_load_from_path = load_from_path_pref == "path"
            status_msg = (
                "Initializing OmniMIDI (System PATH)..."
                if should_load_from_path
                else "Initializing Custom Synth (Bundled SYNTH.dll)..."
            )
            set_status(status_msg)
            if refresh_status:
                refresh_status()

            try:
                self.active_midi_backend = self.omni_engine_cls({}, load_from_path=should_load_from_path)
                self.active_backend_mode = "path" if should_load_from_path else "local"
                set_status(
                    f"{'OmniMIDI' if should_load_from_path else 'Custom Synth'} Initialized ({'System PATH' if should_load_from_path else 'Bundled SYNTH.dll'})."
                )
                return self.active_midi_backend
            except Exception as e:
                set_status(f"{'OmniMIDI' if should_load_from_path else 'Custom Synth'} Init Error: {e}. Playback disabled.")
                print(f"MIDI Backend Init Error: {e}")
                self.active_midi_backend = None
                self.active_backend_mode = None

        if self.active_midi_backend is None:
            self.active_backend_mode = None
            set_status("No MIDI backend found. Playback disabled.")
        return self.active_midi_backend

    def stop_backend(self):
        if self.active_midi_backend and hasattr(self.active_midi_backend, "stop"):
            self.active_midi_backend.stop()

    def start_parse_job(self, filepath, queue_factory, process_factory, process_target, fallback_event_threshold=0, disk_backing_threshold=0):
        self.parser_queue = queue_factory()
        self.parser_process = process_factory(
            target=process_target,
            args=(filepath, self.parser_queue, int(fallback_event_threshold or 0), int(disk_backing_threshold or 0)),
            daemon=True,
        )
        self.parser_process.start()
        return self.parser_process

    def clear_parse_job(self):
        self.parser_process = None
        self.parser_queue = None

    def poll_parser_messages(self):
        if self.parser_queue is None:
            return []
        messages = []
        while True:
            try:
                messages.append(self.parser_queue.get_nowait())
            except Empty:
                break
        return messages

    def normalize_parsed_payload(self, payload, start_padding=3.0, end_padding=3.0, progress_callback=None):
        self._release_parsed_midi_storage()
        payload = dict(payload)
        disk_backed_arrays = payload.pop("disk_backed_arrays", None)
        backing_temp_dir = payload.pop("backing_temp_dir", None)
        self.parsed_midi = types.SimpleNamespace(**payload)

        if progress_callback:
            progress_callback(0.95, "Normalizing...", "Loading data arrays...")

        if disk_backed_arrays:
            try:
                try:
                    self.parsed_midi.note_data_for_gpu = np.load(
                        disk_backed_arrays["note_data_for_gpu"],
                        mmap_mode="r+", allow_pickle=True,
                    )
                except Exception as e:
                    print(f"Failed to mmap GPU note data, loading without mmap: {e}")
                    self.parsed_midi.note_data_for_gpu = np.load(
                        disk_backed_arrays["note_data_for_gpu"],
                        allow_pickle=True,
                    )
                try:
                    self.parsed_midi.note_events_for_playback = np.load(
                        disk_backed_arrays["note_events_for_playback"],
                        mmap_mode="r+", allow_pickle=True,
                    )
                except Exception as e:
                    print(f"Failed to mmap playback events, loading without mmap: {e}")
                    self.parsed_midi.note_events_for_playback = np.load(
                        disk_backed_arrays["note_events_for_playback"],
                        allow_pickle=True,
                    )
                self.parsed_midi._backing_temp_dir = backing_temp_dir
            except Exception:
                if backing_temp_dir:
                    shutil.rmtree(backing_temp_dir, ignore_errors=True)
                raise

        if progress_callback:
            progress_callback(0.96, "Normalizing...", "Applying time offsets...")

        if hasattr(self.parsed_midi, "note_data_for_gpu") and self.parsed_midi.note_data_for_gpu.size > 0:
            self.parsed_midi.note_data_for_gpu["on_time"] += start_padding
            self.parsed_midi.note_data_for_gpu["off_time"] += start_padding

        if self.parsed_midi.note_events_for_playback.size > 0:
            self.parsed_midi.note_events_for_playback["on_time"] += start_padding
            self.parsed_midi.note_events_for_playback["off_time"] += start_padding

        if progress_callback:
            progress_callback(0.97, "Normalizing...", "Processing events...")

        if hasattr(self.parsed_midi, "pitch_bend_events") and self.parsed_midi.pitch_bend_events:
            self.parsed_midi.pitch_bend_events = [
                (t + start_padding, c, p) for t, c, p in self.parsed_midi.pitch_bend_events
            ]

        if hasattr(self.parsed_midi, "program_change_events") and self.parsed_midi.program_change_events:
            self.parsed_midi.program_change_events = [
                (t + start_padding, c, program)
                for t, c, program in self.parsed_midi.program_change_events
            ]

        if hasattr(self.parsed_midi, "control_change_events") and self.parsed_midi.control_change_events:
            self.parsed_midi.control_change_events = [
                (t + start_padding, c, cc, value)
                for t, c, cc, value in self.parsed_midi.control_change_events
            ]

        if progress_callback:
            progress_callback(0.975, "Normalizing...", "Building tempo map...")

        tempo_events = getattr(self.parsed_midi, "tempo_events", None) or [(0.0, 120.0)]
        self.parsed_midi.tempo_events = [
            (float(t) + start_padding, float(bpm)) for t, bpm in tempo_events
        ]
        if not self.parsed_midi.tempo_events or self.parsed_midi.tempo_events[0][0] > 0.0:
            self.parsed_midi.tempo_events.insert(0, (0.0, 120.0))
        self.parsed_midi.tempo_times = np.array(
            [t for t, _ in self.parsed_midi.tempo_events], dtype=np.float64
        )
        self.parsed_midi.tempo_bpms = np.array(
            [bpm for _, bpm in self.parsed_midi.tempo_events], dtype=np.float32
        )

        if hasattr(self.parsed_midi, "total_duration_sec"):
            self.parsed_midi.total_duration_sec += start_padding + end_padding

        if progress_callback:
            progress_callback(0.98, "Normalizing...", "Sorting note data...")

        if self.parsed_midi.note_events_for_playback.size > 0:
            self.parsed_midi.sorted_off_times = np.sort(
                self.parsed_midi.note_events_for_playback["off_time"].astype(np.float64, copy=True)
            )
        else:
            self.parsed_midi.sorted_off_times = np.empty((0,), dtype=np.float64)

        if progress_callback:
            progress_callback(0.99, "Normalizing...", "Computing NPS statistics...")

        self.total_song_notes = len(self.parsed_midi.note_events_for_playback)
        self.total_song_duration = self.parsed_midi.total_duration_sec
        self.max_nps, self.max_nps_spikes = self._compute_nps_stats(self.parsed_midi.note_events_for_playback)
        return self.parsed_midi

    def _compute_nps_stats(self, note_events, top_k=5, min_separation=0.75):
        if note_events is None or len(note_events) == 0:
            return 0, []

        on_times = note_events["on_time"]
        n = len(on_times)

        left_indices = np.searchsorted(on_times, on_times - 1.0, side='left')
        counts = np.arange(n, dtype=np.int64) - left_indices + 1
        max_count = int(counts.max())

        diff_left = np.empty(n, dtype=counts.dtype)
        diff_left[0] = -1
        diff_left[1:] = counts[:-1]
        diff_right = np.empty(n, dtype=counts.dtype)
        diff_right[:-1] = counts[1:]
        diff_right[-1] = -1

        is_peak = ((counts > diff_left) & (counts >= diff_right)) | (
            (counts >= diff_left) & (counts > diff_right)
        )
        peak_indices = np.where(is_peak & (counts > 0))[0]

        if len(peak_indices) == 0:
            best_idx = int(counts.argmax())
            candidates = [(float(on_times[best_idx]), int(counts[best_idx]))]
        else:
            candidates = [(float(on_times[i]), int(counts[i])) for i in peak_indices]

        selected = []
        for spike_time, spike_value in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if all(abs(spike_time - existing_time) >= min_separation for existing_time, _ in selected):
                selected.append((spike_time, spike_value))
                if len(selected) >= top_k:
                    break

        selected.sort(key=lambda item: item[0])
        return max_count, selected

    def handle_parser_message(self, status, payload, start_padding=3.0, end_padding=3.0):
        if status == "total_events":
            return {"kind": "total_events", "total_events": payload}

        if status == "progress":
            return {"kind": "progress", "payload": payload}

        if status == "success":
            parsed = self.normalize_parsed_payload(
                payload, start_padding=start_padding, end_padding=end_padding
            )
            self.clear_parse_job()
            return {
                "kind": "success",
                "parsed_midi": parsed,
                "total_song_notes": self.total_song_notes,
                "total_song_duration": self.total_song_duration,
            }

        self.clear_parse_job()
        return {"kind": "error", "payload": payload}

    def reset_playback_state(self):
        self.playback_start_time = 0.0
        self.paused_at_time = 0.0
        self.total_paused_duration = 0.0
        self.buffered_playback_start_offset = 0.0
        self.notes_played_count = 0
        self.last_processed_event_time = 0.0
        self.current_lag = 0.0
        self.recovery_active = False
        self.recovery_buffer_level = 0.0
        self.recovery_buffer_target = 4.0
        self.nps_event_timestamps.clear()
        self.stop_backend()

    def start_playback(self, current_time):
        if current_time == 0.0:
            self.reset_playback_state()
        self.playing = True
        self.paused = False
        self.playback_start_time = time.monotonic() - (current_time / max(self.playback_speed, 0.01))
        self.total_paused_duration = 0.0
        self.paused_at_time = 0.0

    def pause_playback(self):
        self.paused = True
        self.paused_at_time = time.monotonic()
        if self.active_midi_backend and hasattr(self.active_midi_backend, "pause"):
            self.active_midi_backend.pause()

    def resume_playback(self):
        self.paused = False
        if self.active_midi_backend and hasattr(self.active_midi_backend, "play"):
            self.active_midi_backend.play()
        if self.paused_at_time > 0.0:
            pause_duration = time.monotonic() - self.paused_at_time
            self.total_paused_duration += pause_duration
            self.paused_at_time = 0.0

    def stop_playback(self):
        self.playing = False
        self.reset_playback_state()

    def finish_playback(self):
        self.playing = False
        self.paused = False
        self.recovery_active = False
        self.recovery_buffer_level = 0.0
        final_time = max(0.0, float(self.total_song_duration))
        self.current_playback_time_for_threads = final_time
        self.last_processed_event_time = final_time
        self.buffered_playback_start_offset = final_time

    def unload_midi(self):
        self.playing = False
        self.reset_playback_state()
        self._release_parsed_midi_storage()
        self.total_song_notes = 0
        self.total_song_duration = 0.0
        self.max_nps = 0
        self.max_nps_spikes = []

    def begin_seek(self):
        self.is_seeking = True
        if self.playing and not self.paused:
            self.paused_for_seeking = True
            self.paused = True
            self.paused_at_time = time.monotonic()
            if self.active_midi_backend and hasattr(self.active_midi_backend, "pause"):
                self.active_midi_backend.pause()
            return True
        self.paused_for_seeking = False
        return False

    def complete_seek(self, seek_time):
        self.is_seeking = False
        self.seek_request_time = seek_time
        self.last_processed_event_time = seek_time

        resumed = False
        if self.paused_for_seeking:
            self.paused = False
            if self.paused_at_time > 0.0:
                pause_duration = time.monotonic() - self.paused_at_time
                self.total_paused_duration += pause_duration
                self.paused_at_time = 0.0
            resumed = True

        self.paused_for_seeking = False
        return resumed

    def panic_all_notes_off(self):
        if self.active_midi_backend is None:
            return
        self.active_midi_backend.send_all_notes_off()

    def set_pitch_bend_range(self, semitones=12):
        if not self.active_midi_backend:
            return
        print(f"Setting pitch bend range to +/- {semitones} semitones on all channels.")
        if hasattr(self.active_midi_backend, "set_pitch_bend_range"):
            self.active_midi_backend.set_pitch_bend_range(semitones)
            return
        for channel in range(16):
            status = 0xB0 | channel
            self.active_midi_backend.send_raw_event(status, (0 << 8) | 101)
            self.active_midi_backend.send_raw_event(status, (0 << 8) | 100)
            self.active_midi_backend.send_raw_event(status, (semitones << 8) | 6)
            self.active_midi_backend.send_raw_event(status, (0 << 8) | 38)
            self.active_midi_backend.send_raw_event(status, (127 << 8) | 101)
            self.active_midi_backend.send_raw_event(status, (127 << 8) | 100)

    def get_current_playback_time(self):
        if threading.current_thread() is not threading.main_thread() and not self.playing:
            return min(self.current_playback_time_for_threads, self.total_song_duration) if self.total_song_duration > 0 else self.current_playback_time_for_threads

        if self.active_midi_backend and getattr(self.active_midi_backend, "buffering_enabled", False):
            if self.playing:
                current_time = self.buffered_playback_start_offset + self.active_midi_backend.get_position_seconds()
            else:
                current_time = self.buffered_playback_start_offset
            return min(current_time, self.total_song_duration) if self.total_song_duration > 0 else current_time

        if self.playing and not self.paused:
            ideal_time = (time.monotonic() - self.playback_start_time - self.total_paused_duration) * self.playback_speed
            current_time = ideal_time - self.current_lag
            return min(current_time, self.total_song_duration) if self.total_song_duration > 0 else current_time
        if self.playing and self.paused:
            if self.paused_at_time > 0:
                current_time = (self.paused_at_time - self.playback_start_time - self.total_paused_duration) * self.playback_speed
            else:
                current_time = self.last_processed_event_time
            return min(current_time, self.total_song_duration) if self.total_song_duration > 0 else current_time
        return min(self.current_playback_time_for_threads, self.total_song_duration) if self.total_song_duration > 0 else self.current_playback_time_for_threads

    def set_playback_speed(self, speed):
        new_speed = max(0.1, min(float(speed), 4.0))
        current_time = self.get_current_playback_time()
        self.playback_speed = new_speed

        if self.active_midi_backend and hasattr(self.active_midi_backend, "set_speed"):
            self.active_midi_backend.set_speed(new_speed)

        if self.active_midi_backend and getattr(self.active_midi_backend, "buffering_enabled", False):
            return current_time

        now = time.monotonic()
        if self.playing:
            self.playback_start_time = now - (current_time / max(self.playback_speed, 0.01)) - self.total_paused_duration
            if self.paused:
                self.paused_at_time = now
        return current_time

    def shutdown(self):
        self.playing = False
        self.stop_backend()
        if self.active_midi_backend:
            self.active_midi_backend.shutdown()
            self.active_midi_backend = None
        self.active_backend_mode = None
        if self.parser_process and self.parser_process.is_alive():
            self.parser_process.terminate()
            self.parser_process.join(0.1)
        self.clear_parse_job()
        self._release_parsed_midi_storage()
