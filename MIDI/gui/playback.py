"""Auto-extracted mixin for DpgMidiPlayerApp."""
import bisect
import heapq
import os
import threading
import time
import traceback
import math
import subprocess
import tempfile
import wave
from collections import deque

import numpy as np
import dearpygui.dearpygui as dpg

class PlaybackMixin:
    """Methods for playback."""

    def _stop_playback_for_backend_reinit(self):
        self._wait_for_audio_sweep(timeout=2.0)
        self.controller.playing = False
        self.controller.paused = False
        self.controller.paused_for_seeking = False
        with self.playback_lock:
            self.controller.seek_request_time = None

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(1.0)

        self.playback_thread = None
        dpg.configure_item("play_button", label="Play")
        dpg.configure_item("voices_slider", enabled=True)

        if self.controller.active_midi_backend:
            try:
                self.controller.reset_playback_state()
            except Exception as e:
                print(f"Failed to reset backend before reinit: {e}")

        self.reset_graph_history()
        dpg.set_value("seek_slider", 0.0)
        if self.controller.parsed_midi:
            dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
            dpg.set_value("status_text", "Ready to play.")
        else:
            dpg.set_value("time_text", "00:00 / 00:00")
            dpg.set_value("status_text", "No file loaded.")
        self._refresh_transport_button_state()


    def on_seek_start(self, sender, app_data):
        if not self.controller.parsed_midi:
            return
        if not self._seek_was_active:
            paused_for_seek = self.controller.begin_seek()
            if paused_for_seek:
                dpg.set_value("status_text", "Seeking...")
            self._seek_was_active = True


    def on_seek_end(self, sender, app_data):
        if not self.controller.parsed_midi:
            self._seek_was_active = False
            return
        seek_time = dpg.get_value("seek_slider")
        self.panic_all_notes_off()
        with self.playback_lock:
            resumed = self.controller.complete_seek(seek_time)
        if resumed:
            dpg.configure_item("play_button", label="Pause")
            dpg.set_value("status_text", "Playing...")
        if not self.controller.playing or self.controller.paused:
            dpg.set_value(
                "time_text",
                f"{self.format_time(seek_time)} / {self.format_time(self.controller.total_song_duration)}",
            )
        self._seek_was_active = False


    def toggle_play_pause(self):
        if self.controller.playing:
            if self.controller.paused:
                self._manual_stop_requested = False
                self.controller.resume_playback()
                dpg.configure_item("play_button", label="Pause")
                dpg.set_value("status_text", "Playing...")
            else:
                self.controller.pause_playback()
                dpg.configure_item("play_button", label="Resume")
                dpg.set_value("status_text", "Paused")
        else:
            if self.controller.parsed_midi is None:
                return
            if self.controller.active_midi_backend is None:
                self._message_warning("No Audio Backend", "No MIDI backend is loaded. Check your SoundFont or audio settings.")
                return
            self._manual_stop_requested = False
            current_time = self.get_current_playback_time()
            self.controller.start_playback(current_time)
            dpg.configure_item("play_button", label="Pause")
            dpg.set_value("status_text", "Playing...")
            dpg.configure_item("voices_slider", enabled=False)
            self.playback_thread = threading.Thread(target=self.play_music_thread, daemon=True)
            self.playback_thread.start()


    def stop_playback(self):
        self._manual_stop_requested = True
        self.controller.stop_playback()
        self.reset_playback_state()
        dpg.configure_item("play_button", label="Play")
        if self.controller.parsed_midi:
            dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
            dpg.set_value("status_text", "Ready to play.")
        dpg.configure_item("voices_slider", enabled=True)


    def reset_playback_state(self):
        self.controller.reset_playback_state()
        dpg.set_value("seek_slider", 0.0)
        self.reset_graph_history()
        self.panic_all_notes_off()


    def playback_finished(self):
        self.controller.finish_playback()
        dpg.configure_item("play_button", label="Play")
        if self.controller.parsed_midi:
            if self._manual_stop_requested:
                dpg.set_value("status_text", "Ready to play.")
                dpg.set_value("time_text", f"00:00 / {self.format_time(self.controller.total_song_duration)}")
                dpg.set_value("seek_slider", 0.0)
            else:
                dpg.set_value("status_text", "Finished.")
                finished = self.format_time(self.controller.total_song_duration)
                dpg.set_value("time_text", f"{finished} / {finished}")
                dpg.set_value("seek_slider", self.controller.total_song_duration)
        self._manual_stop_requested = False
        dpg.configure_item("voices_slider", enabled=True)
        dpg.set_value("recovery_buffer_progress", 0.0)
        dpg.configure_item("recovery_buffer_progress", overlay="Recovery: 0.0s / 4.0s")
        dpg.set_value("recovery_buffer_progress_overlay", 0.0)
        dpg.configure_item("recovery_buffer_progress_overlay", overlay="Recovery: 0.0s / 4.0s")
        self._refresh_transport_button_state()
        self.panic_all_notes_off()


    def play_music_thread(self):
        if self.controller.active_midi_backend and getattr(self.controller.active_midi_backend, "buffering_enabled", False):
            self.play_music_thread_buffered()
        else:
            self.play_music_thread_realtime()


    def play_music_thread_buffered(self):
        try:
            if not self.controller.parsed_midi:
                self._queue_ui(self.playback_finished)
                return

            self._queue_ui(dpg.set_value, "status_text", "Pre-rendering events...")

            if self.controller.active_midi_backend:
                self.set_pitch_bend_range(semitones=12)
                self.controller.active_midi_backend.stop()

            times, statuses, params = self._build_buffered_event_arrays(self.controller.parsed_midi)

            self.controller.active_midi_backend.upload_events(times, statuses, params)

            # Warmup: render a short burst to force BASS MIDI to load SFZ
            # samples into memory. Without this, the first playback with
            # SFZ soundfonts blocks inside ChannelGetData while loading
            # samples from disk, causing prerendering to appear stuck.
            self._queue_ui(dpg.set_value, "status_text", "Loading soundfont...")
            self.controller.active_midi_backend.set_current_time(0.0)
            self.controller.active_midi_backend.fill_buffer(1.0)
            self.controller.active_midi_backend.reset_for_prerender()

            start_time = self.get_current_playback_time()
            self.controller.active_midi_backend.set_current_time(start_time)
            has_started_playback = False
            emergency_recovery = False
            startup_prerender_target = 3.0 if start_time <= 0.001 else 4.0
            recovery_target = 4.0
            self.controller.recovery_active = False
            self.controller.recovery_buffer_level = 0.0
            self.controller.recovery_buffer_target = recovery_target

            while self.controller.playing:
                if not self.controller.playing:
                    break

                requested_time = None
                with self.playback_lock:
                    if self.controller.seek_request_time is not None:
                        requested_time = self.controller.seek_request_time
                        self.controller.seek_request_time = None

                if requested_time is not None:
                    if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                        self.controller.active_midi_backend.set_emergency_recovery(False)
                        emergency_recovery = False
                    self.controller.recovery_active = False
                    self.controller.recovery_buffer_level = 0.0
                    self.controller.active_midi_backend.stop()
                    self.controller.active_midi_backend.set_current_time(requested_time)
                    self.controller.buffered_playback_start_offset = requested_time
                    has_started_playback = False
                    startup_prerender_target = 4.0

                buffer_lvl = self.controller.active_midi_backend.fill_buffer(60.0)
                is_active = self.controller.active_midi_backend.is_active()
                self.controller.recovery_buffer_target = recovery_target

                if self.controller.paused:
                    if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                        self.controller.active_midi_backend.set_emergency_recovery(False)
                        emergency_recovery = False
                    self.controller.recovery_active = False
                    self.controller.recovery_buffer_level = 0.0
                    self.controller.active_midi_backend.pause()
                    if buffer_lvl >= 60.0:
                        time.sleep(0.1)
                    else:
                        time.sleep(0.005)
                    continue

                if not has_started_playback:
                    if buffer_lvl >= startup_prerender_target:
                        if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                            self.controller.active_midi_backend.set_emergency_recovery(False)
                            emergency_recovery = False
                        self.controller.recovery_active = False
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        self._queue_ui(dpg.set_value, "status_text", "Playing...")
                        self.controller.active_midi_backend.play()
                        has_started_playback = True
                    else:
                        self.controller.recovery_active = False
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        self._queue_ui(
                            dpg.set_value,
                            "status_text",
                            f"Prerendering... {buffer_lvl:.1f}s / {startup_prerender_target:.1f}s",
                        )
                else:
                    if emergency_recovery or buffer_lvl < 0.2:
                        progress = max(0.0, min(buffer_lvl / recovery_target, 1.0))
                        normal_voice_limit = max(
                            1,
                            int(
                                getattr(
                                    self.controller.active_midi_backend,
                                    "normal_voice_limit",
                                    self._CONFIG["audio"].get("voices", 512),
                                )
                            ),
                        )
                        min_voice_limit = min(normal_voice_limit, 32)
                        eased_progress = progress ** 1.65
                        skip_velocity_below = max(0, min(127, int(round(127.0 * (1.0 - eased_progress)))))
                        recovery_voice_limit = max(
                            min_voice_limit,
                            min(
                                normal_voice_limit,
                                int(
                                    round(
                                        min_voice_limit
                                        + ((normal_voice_limit - min_voice_limit) * eased_progress)
                                    )
                                ),
                            ),
                        )
                        skip_note_ons = progress <= 0.04 and recovery_voice_limit <= min_voice_limit
                        if hasattr(self.controller.active_midi_backend, "configure_emergency_recovery"):
                            self.controller.active_midi_backend.configure_emergency_recovery(
                                skip_velocity_below,
                                recovery_voice_limit,
                                skip_note_ons,
                            )
                        elif not emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                            self.controller.active_midi_backend.set_emergency_recovery(True)
                        emergency_recovery = True
                        self.controller.recovery_active = buffer_lvl < recovery_target
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        if not self.controller.paused and not is_active and buffer_lvl > 0.02:
                            self.controller.active_midi_backend.play()
                        if buffer_lvl >= recovery_target:
                            if hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                                self.controller.active_midi_backend.set_emergency_recovery(False)
                            emergency_recovery = False
                            self.controller.recovery_active = False
                            self._queue_ui(dpg.set_value, "status_text", "Playing...")
                        else:
                            self._queue_ui(
                                dpg.set_value,
                                "status_text",
                                f"Recovering buffer... {buffer_lvl:.1f}s / {recovery_target:.1f}s",
                            )
                    else:
                        self.controller.recovery_active = False
                        self.controller.recovery_buffer_level = min(buffer_lvl, recovery_target)
                        if emergency_recovery and hasattr(self.controller.active_midi_backend, "set_emergency_recovery"):
                            self.controller.active_midi_backend.set_emergency_recovery(False)
                            emergency_recovery = False
                            self._queue_ui(dpg.set_value, "status_text", "Playing...")
                        if not self.controller.paused and not is_active and buffer_lvl > 2.0:
                            self.controller.active_midi_backend.play()
                            self._queue_ui(dpg.set_value, "status_text", "Playing...")

                if buffer_lvl >= 60.0:
                    time.sleep(0.1)
                else:
                    time.sleep(0.005)

        except Exception as e:
            print(f"Buffered playback error: {e}")
            traceback.print_exc()
        finally:
            self.controller.recovery_active = False
            self.controller.recovery_buffer_level = 0.0
            if (
                self.controller.active_midi_backend
                and hasattr(self.controller.active_midi_backend, "set_emergency_recovery")
            ):
                try:
                    self.controller.active_midi_backend.set_emergency_recovery(False)
                except Exception:
                    pass
            self._queue_ui(self.playback_finished)


    def play_music_thread_realtime(self):
        try:
            if not self.controller.parsed_midi:
                self._queue_ui(self.playback_finished)
                return

            self._queue_ui(dpg.set_value, "status_text", "Starting playback...")

            if self.controller.active_midi_backend:
                self.set_pitch_bend_range(semitones=12)

            note_events = self.controller.parsed_midi.note_events_for_playback
            audible_mask = note_events["velocity"] >= self._AUDIO_MIN_NOTE_VELOCITY
            note_events = note_events[audible_mask]
            program_change_events = getattr(self.controller.parsed_midi, "program_change_events", [])
            pitch_bend_events = self.controller.parsed_midi.pitch_bend_events
            control_change_events = getattr(self.controller.parsed_midi, "control_change_events", [])

            num_note_events = len(note_events)
            num_program_change_events = len(program_change_events)
            num_pitch_bend_events = len(pitch_bend_events)
            num_control_change_events = len(control_change_events)

            if self.controller.active_midi_backend:
                for channel in range(16):
                    status = 0xE0 + channel
                    param = (0x40 << 8) | 0x00
                    self.controller.active_midi_backend.send_raw_event(status, param)

            start_time = self.get_current_playback_time()
            note_event_index = bisect.bisect_left(note_events["on_time"], start_time)
            program_change_index = bisect.bisect_left(program_change_events, (start_time, -float("inf"), -float("inf")))
            pitch_bend_index = bisect.bisect_left(pitch_bend_events, (start_time, -float("inf"), -float("inf")))
            control_change_index = bisect.bisect_left(control_change_events, (start_time, -float("inf"), -float("inf"), -float("inf")))

            def _apply_program_state(target_time):
                if not self.controller.active_midi_backend:
                    return
                latest_programs = {}
                for change_time, channel, program in program_change_events:
                    if change_time > target_time:
                        break
                    latest_programs[int(channel)] = int(program)
                for channel, program in latest_programs.items():
                    self.controller.active_midi_backend.send_raw_event(0xC0 + channel, program)

            _apply_program_state(start_time)

            with self.playback_lock:
                self.controller.last_processed_event_time = start_time

            note_off_heap = []
            if note_event_index > 0:
                notes_before_now = note_events[:note_event_index]
                active_notes = notes_before_now[notes_before_now["off_time"] > start_time]
                for note in active_notes:
                    heapq.heappush(note_off_heap, (note["off_time"], note["pitch"], note["channel"]))

            while self.controller.playing and (
                note_event_index < num_note_events
                or program_change_index < num_program_change_events
                or pitch_bend_index < num_pitch_bend_events
                or control_change_index < num_control_change_events
                or len(note_off_heap) > 0
            ):
                while self.controller.paused:
                    if not self.controller.playing:
                        break
                    time.sleep(0.01)
                if not self.controller.playing:
                    break

                requested_time = None
                with self.playback_lock:
                    if self.controller.seek_request_time is not None:
                        requested_time = self.controller.seek_request_time
                        self.controller.seek_request_time = None

                if requested_time is not None:
                    with self.playback_lock:
                        self.controller.last_processed_event_time = requested_time
                    note_event_index = bisect.bisect_left(note_events["on_time"], requested_time)
                    program_change_index = bisect.bisect_left(program_change_events, (requested_time, -float("inf"), -float("inf")))
                    pitch_bend_index = bisect.bisect_left(pitch_bend_events, (requested_time, -float("inf"), -float("inf")))
                    control_change_index = bisect.bisect_left(control_change_events, (requested_time, -float("inf"), -float("inf"), -float("inf")))
                    self.controller.playback_start_time = time.monotonic() - (requested_time / max(self.controller.playback_speed, 0.01))
                    self.controller.total_paused_duration = 0.0
                    self.controller.paused_at_time = 0.0
                    self.controller.notes_played_count = note_event_index
                    self.controller.nps_event_timestamps.clear()
                    note_off_heap.clear()
                    if note_event_index > 0:
                        notes_before_now = note_events[:note_event_index]
                        active_notes = notes_before_now[notes_before_now["off_time"] > requested_time]
                        for note in active_notes:
                            heapq.heappush(note_off_heap, (note["off_time"], note["pitch"], note["channel"]))
                    _apply_program_state(requested_time)

                next_note_on_time = note_events[note_event_index]["on_time"] if note_event_index < num_note_events else float("inf")
                next_note_off_time = note_off_heap[0][0] if note_off_heap else float("inf")
                next_program_change_time = program_change_events[program_change_index][0] if program_change_index < num_program_change_events else float("inf")
                next_pitch_bend_time = pitch_bend_events[pitch_bend_index][0] if pitch_bend_index < num_pitch_bend_events else float("inf")
                next_control_change_time = control_change_events[control_change_index][0] if control_change_index < num_control_change_events else float("inf")
                event_time_sec = min(next_note_on_time, next_note_off_time, next_program_change_time, next_pitch_bend_time, next_control_change_time)

                if event_time_sec == float("inf"):
                    break

                with self.playback_lock:
                    self.controller.last_processed_event_time = event_time_sec

                target_wall_time = self.controller.playback_start_time + (event_time_sec / max(self.controller.playback_speed, 0.01)) + self.controller.total_paused_duration
                sleep_duration = target_wall_time - time.monotonic()
                self.controller.current_lag = max(0, -sleep_duration) * self.controller.playback_speed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                if not self.controller.playing:
                    break
                if self.controller.paused:
                    continue
                with self.playback_lock:
                    if self.controller.seek_request_time is not None:
                        continue

                try:
                    batch_events = []

                    while note_off_heap and note_off_heap[0][0] <= event_time_sec:
                        off_time, pitch, channel = heapq.heappop(note_off_heap)
                        batch_events.append((0x80 + channel, pitch))

                    while note_event_index < num_note_events and note_events[note_event_index]["on_time"] <= event_time_sec:
                        note = note_events[note_event_index]
                        note_event_index += 1
                        pitch = int(note["pitch"])
                        vel = int(note["velocity"])
                        channel = int(note["channel"])

                        self.controller.notes_played_count += 1
                        self.controller.nps_event_timestamps.append(note["on_time"])

                        batch_events.append((0x90 + channel, (vel << 8) | pitch))
                        heapq.heappush(note_off_heap, (note["off_time"], pitch, channel))

                    while program_change_index < num_program_change_events and program_change_events[program_change_index][0] <= event_time_sec:
                        _time, channel, program = program_change_events[program_change_index]
                        batch_events.append((0xC0 + int(channel), int(program)))
                        program_change_index += 1

                    while pitch_bend_index < num_pitch_bend_events and pitch_bend_events[pitch_bend_index][0] <= event_time_sec:
                        _time, channel, pitch_value = pitch_bend_events[pitch_bend_index]
                        data1 = pitch_value & 0x7F
                        data2 = (pitch_value >> 7) & 0x7F
                        batch_events.append((0xE0 + channel, (data2 << 8) | data1))
                        pitch_bend_index += 1

                    while control_change_index < num_control_change_events and control_change_events[control_change_index][0] <= event_time_sec:
                        _time, channel, controller, value = control_change_events[control_change_index]
                        batch_events.append((0xB0 + channel, (value << 8) | controller))
                        control_change_index += 1

                    if batch_events and self.controller.active_midi_backend:
                        self.controller.active_midi_backend.send_event_batch(batch_events)
                except Exception as e:
                    print(f"MIDI backend send error: {e}")
                    self._queue_ui(dpg.set_value, "status_text", f"Playback Error: {e}")
                    break
        except Exception as e:
            print(f"Playback thread error: {e}")
            traceback.print_exc()
        finally:
            self._queue_ui(self.playback_finished)


    def get_current_playback_time(self):
        current_time = self.controller.get_current_playback_time()
        if not self.controller.playing:
            return dpg.get_value("seek_slider")
        return current_time


    def get_current_playback_time_thread_safe(self):
        return self.controller.current_playback_time_for_threads


    def _build_buffered_event_arrays(self, parsed_midi):
        note_events = parsed_midi.note_events_for_playback
        audible_note_events = note_events[note_events["velocity"] >= self._AUDIO_MIN_NOTE_VELOCITY]
        program_change_events = getattr(parsed_midi, "program_change_events", [])
        pitch_bend_events = parsed_midi.pitch_bend_events
        control_change_events = getattr(parsed_midi, "control_change_events", [])

        count_notes = len(audible_note_events)
        count_programs = len(program_change_events)
        count_bends = len(pitch_bend_events)
        count_ccs = len(control_change_events)
        total_ops = (count_notes * 2) + count_programs + count_bends + count_ccs

        times = np.empty(total_ops, dtype=np.float64)
        statuses = np.empty(total_ops, dtype=np.uint32)
        params = np.empty(total_ops, dtype=np.uint32)
        priorities = np.empty(total_ops, dtype=np.uint8)

        times[:count_notes] = audible_note_events["on_time"]
        statuses[:count_notes] = 0x90 + audible_note_events["channel"]
        params[:count_notes] = (audible_note_events["velocity"].astype(np.uint32) << 8) | audible_note_events["pitch"].astype(np.uint32)
        priorities[:count_notes] = 2

        times[count_notes : count_notes * 2] = audible_note_events["off_time"]
        statuses[count_notes : count_notes * 2] = 0x80 + audible_note_events["channel"]
        params[count_notes : count_notes * 2] = audible_note_events["pitch"].astype(np.uint32)
        priorities[count_notes : count_notes * 2] = 0

        if count_programs > 0:
            pc_arr = np.array(program_change_events, dtype=[("time", "f8"), ("chan", "u4"), ("program", "u4")])
            start_idx = count_notes * 2
            end_idx = start_idx + count_programs
            times[start_idx:end_idx] = pc_arr["time"]
            statuses[start_idx:end_idx] = 0xC0 + pc_arr["chan"]
            params[start_idx:end_idx] = pc_arr["program"]
            priorities[start_idx:end_idx] = 1

        if count_bends > 0:
            pb_arr = np.array(pitch_bend_events, dtype=[("time", "f8"), ("chan", "u4"), ("val", "u4")])
            start_idx = (count_notes * 2) + count_programs
            end_idx = start_idx + count_bends
            times[start_idx:end_idx] = pb_arr["time"]
            statuses[start_idx:end_idx] = 0xE0 + pb_arr["chan"]
            bend_lsb = pb_arr["val"] & 0x7F
            bend_msb = (pb_arr["val"] >> 7) & 0x7F
            params[start_idx:end_idx] = (bend_msb << 8) | bend_lsb
            priorities[start_idx:end_idx] = 1

        if count_ccs > 0:
            cc_arr = np.array(control_change_events, dtype=[("time", "f8"), ("chan", "u4"), ("cc", "u4"), ("val", "u4")])
            start_idx = (count_notes * 2) + count_programs + count_bends
            end_idx = start_idx + count_ccs
            times[start_idx:end_idx] = cc_arr["time"]
            statuses[start_idx:end_idx] = 0xB0 + cc_arr["chan"]
            params[start_idx:end_idx] = (cc_arr["val"] << 8) | cc_arr["cc"]
            priorities[start_idx:end_idx] = 1

        sort_indices = np.lexsort((np.arange(total_ops, dtype=np.int64), priorities, times))
        return times[sort_indices], statuses[sort_indices], params[sort_indices]


