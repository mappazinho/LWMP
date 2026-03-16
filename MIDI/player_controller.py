import os
import threading
import time
import types
from collections import deque
from queue import Empty


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
        self.active_midi_backend = None

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
        bundled_omnimidi_exists = os.path.exists(os.path.join(self.script_dir, "OmniMIDI.dll"))
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
                                f"Configured SoundFont ({missing_name}) not found.\nPlease select a SoundFont (.sf2) to use.",
                            )
                        sf_path = None
                        self.config["audio"]["soundfont_path"] = None
                        self.save_config(self.config)

                    if not sf_path:
                        if prompt_info:
                            prompt_info(
                                "SoundFont Missing",
                                "No SoundFont is configured.\nPlease select a SoundFont (.sf2) to use.",
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

                    self.active_midi_backend = self.bass_engine_cls(
                        {}, soundfont_path=sf_path, buffering=True, debug=self.debug
                    )
                    self.active_midi_backend.set_volume(volume)
                    self.active_midi_backend.set_voices(voices)

                    if self.active_midi_backend.midi_stream:
                        set_status("BASSMIDI Engine Initialized (Buffered).")
                        if launch_sweep:
                            launch_sweep(self.active_midi_backend)
                        return self.active_midi_backend
                except Exception as e:
                    print(f"BASSMIDI Init Failed: {e}")
                    self.active_midi_backend = None
            elif prompt_error:
                prompt_error("Error", "BASSMIDI libraries not found or failed to load.")

        if self.platform_name == "nt" and self.omni_engine_cls is not None:
            should_load_from_path = load_from_path_pref == "path"
            status_msg = (
                "Initializing OmniMIDI (System Path)..."
                if should_load_from_path
                else "Initializing OmniMIDI (Local DLL)..."
            )
            set_status(status_msg)
            if refresh_status:
                refresh_status()

            try:
                self.active_midi_backend = self.omni_engine_cls({}, load_from_path=should_load_from_path)
                set_status(f"OmniMIDI Engine Initialized ({'Path' if should_load_from_path else 'Local'}).")
                return self.active_midi_backend
            except Exception as e:
                set_status(f"OmniMIDI Engine Init Error: {e}. Playback disabled.")
                print(f"MIDI Backend Init Error: {e}")
                self.active_midi_backend = None

        if self.active_midi_backend is None:
            set_status("No MIDI backend found. Playback disabled.")
        return self.active_midi_backend

    def stop_backend(self):
        if self.active_midi_backend and hasattr(self.active_midi_backend, "stop"):
            self.active_midi_backend.stop()

    def start_parse_job(self, filepath, queue_factory, process_factory, process_target):
        self.parser_queue = queue_factory()
        self.parser_process = process_factory(target=process_target, args=(filepath, self.parser_queue), daemon=True)
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

    def normalize_parsed_payload(self, payload, start_padding=3.0, end_padding=3.0):
        self.parsed_midi = types.SimpleNamespace(**payload)

        if self.parsed_midi.note_events_for_playback.size > 0:
            self.parsed_midi.note_events_for_playback["on_time"] += start_padding
            self.parsed_midi.note_events_for_playback["off_time"] += start_padding

        if hasattr(self.parsed_midi, "pitch_bend_events") and self.parsed_midi.pitch_bend_events:
            self.parsed_midi.pitch_bend_events = [
                (t + start_padding, c, p) for t, c, p in self.parsed_midi.pitch_bend_events
            ]

        if hasattr(self.parsed_midi, "total_duration_sec"):
            self.parsed_midi.total_duration_sec += start_padding + end_padding

        self.total_song_notes = len(self.parsed_midi.note_events_for_playback)
        self.total_song_duration = self.parsed_midi.total_duration_sec
        return self.parsed_midi

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
        self.nps_event_timestamps.clear()
        self.stop_backend()

    def start_playback(self, current_time):
        if current_time == 0.0:
            self.reset_playback_state()
        self.playing = True
        self.paused = False
        self.playback_start_time = time.monotonic() - current_time
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

    def unload_midi(self):
        self.playing = False
        self.reset_playback_state()
        self.parsed_midi = None
        self.total_song_notes = 0
        self.total_song_duration = 0.0

    def begin_seek(self):
        self.is_seeking = True
        if self.playing and not self.paused:
            self.paused_for_seeking = True
            self.paused = True
            self.paused_at_time = time.monotonic()
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
            return self.current_playback_time_for_threads

        if self.active_midi_backend and getattr(self.active_midi_backend, "buffering_enabled", False):
            if self.playing:
                return self.buffered_playback_start_offset + self.active_midi_backend.get_position_seconds()
            return self.buffered_playback_start_offset

        if self.playing and not self.paused:
            ideal_time = time.monotonic() - self.playback_start_time - self.total_paused_duration
            return ideal_time - self.current_lag
        if self.playing and self.paused:
            if self.paused_at_time > 0:
                return self.paused_at_time - self.playback_start_time - self.total_paused_duration
            return self.last_processed_event_time
        return self.current_playback_time_for_threads

    def shutdown(self):
        self.playing = False
        self.stop_backend()
        if self.active_midi_backend:
            self.active_midi_backend.shutdown()
            self.active_midi_backend = None
        if self.parser_process and self.parser_process.is_alive():
            self.parser_process.terminate()
            self.parser_process.join(0.1)
        self.clear_parse_job()
