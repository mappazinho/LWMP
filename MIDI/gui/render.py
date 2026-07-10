"""Auto-extracted mixin for DpgMidiPlayerApp."""
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

class RenderMixin:
    """Methods for render."""

    def _render_cfg(self):
        return self._CONFIG["render"]


    def _build_render_spike_options(self):
        spikes = getattr(self.controller, "max_nps_spikes", [])
        self.render_spike_label_map = {}
        if not spikes:
            return ["No spikes detected"]
        items = ["None"]
        for idx, (spike_time, spike_value) in enumerate(spikes):
            label = f"{idx + 1}. {self.format_time(spike_time)} - {self.format_nps(spike_value)}"
            items.append(label)
            self.render_spike_label_map[label] = (float(spike_time), int(spike_value))
        return items


    def _toggle_render_stats_mod_controls(self, sender=None, app_data=None):
        enabled = bool(dpg.get_value("render_stats_mod_checkbox")) if dpg.does_item_exist("render_stats_mod_checkbox") else False
        for tag in ("render_stats_multiplier", "render_spike_select", "render_spike_intensity"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=enabled)


    def _toggle_render_stats_controls(self, sender=None, app_data=None):
        visible = bool(dpg.get_value("render_stats_overlay_checkbox")) if dpg.does_item_exist("render_stats_overlay_checkbox") else False
        if dpg.does_item_exist("render_stats_mod_group"):
            dpg.configure_item("render_stats_mod_group", show=visible)
        self._toggle_render_stats_mod_controls()


    def show_render_window(self, sender=None, app_data=None):
        if self.controller.parsed_midi is None:
            self._message_warning("No MIDI", "Load a MIDI before starting a render.")
            return
        if self.render_thread and self.render_thread.is_alive():
            self._message_info("Render In Progress", "A video render is already running.")
            return
        current_ffmpeg = dpg.get_value("render_ffmpeg_path").strip()
        if not current_ffmpeg or current_ffmpeg.lower() == "ffmpeg":
            bundled_ffmpeg = self._bundled_ffmpeg_path()
            if bundled_ffmpeg:
                dpg.set_value("render_ffmpeg_path", bundled_ffmpeg)
        # Detect hardware encoders and update codec dropdown
        ffmpeg_for_detect = dpg.get_value("render_ffmpeg_path").strip()
        hw_labels = self._detect_hw_encoders(ffmpeg_for_detect) if ffmpeg_for_detect else []
        codec_items = hw_labels + ["H.264", "H.265", "MPEG-4"]
        saved_codec = str(self._render_cfg().get("codec", "H.264"))
        if saved_codec not in codec_items:
            saved_codec = hw_labels[0] if hw_labels else "H.264"
        dpg.configure_item("render_codec", items=codec_items, default_value=saved_codec)

        if not dpg.get_value("render_output_path") and self.controller.parsed_midi:
            source_name = os.path.splitext(os.path.basename(self.controller.parsed_midi.filename))[0]
            default_path = os.path.join(os.path.dirname(self.controller.parsed_midi.filename), f"{source_name}_render.mp4")
            dpg.set_value("render_output_path", default_path)
        spike_items = self._build_render_spike_options()
        selected_spike = str(self._render_cfg().get("spike_selection", "None") or "None")
        if selected_spike not in spike_items:
            selected_spike = "None" if "None" in spike_items else spike_items[0]
        dpg.configure_item("render_spike_select", items=spike_items)
        dpg.set_value("render_spike_select", selected_spike)
        self._toggle_render_stats_controls()
        self._center_modal("render_window", 520, 520)
        dpg.configure_item("render_window", show=True)


    def _set_render_progress(self, fraction, overlay, detail):
        if dpg.does_item_exist("render_progress_bar"):
            dpg.set_value("render_progress_bar", max(0.0, min(float(fraction), 1.0)))
            dpg.configure_item("render_progress_bar", overlay=str(overlay))
        if dpg.does_item_exist("render_status_text"):
            dpg.set_value("render_status_text", detail)


    def _format_render_eta(self, seconds_remaining):
        seconds_remaining = max(0, int(round(float(seconds_remaining))))
        hours, rem = divmod(seconds_remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


    def _render_progress_with_timing(self, stage_name, stage_start_time, stage_fraction, overall_fraction, overlay, detail):
        stage_fraction = max(0.0, min(float(stage_fraction), 1.0))
        overall_fraction = max(0.0, min(float(overall_fraction), 1.0))
        detail_text = str(detail)
        now = time.monotonic()
        overall_start_time = getattr(self, "render_start_time_monotonic", 0.0) or stage_start_time
        elapsed = max(0.0, now - overall_start_time)
        detail_text = f"{detail_text}\nTime elapsed: {self._format_render_eta(elapsed)}"
        stage_elapsed = max(0.0, now - stage_start_time)

        state = self.render_stage_timing.get(stage_name)
        if state is None or abs(state.get("start_time", 0.0) - stage_start_time) > 1e-6:
            state = {
                "start_time": stage_start_time,
                "last_time": now,
                "last_fraction": stage_fraction,
                "ema_rate": 0.0,
            }
            self.render_stage_timing[stage_name] = state
        else:
            dt = max(0.0, now - state["last_time"])
            df = max(0.0, stage_fraction - state["last_fraction"])
            if dt > 0.05 and df > 0.0:
                instant_rate = df / dt
                if state["ema_rate"] <= 0.0:
                    state["ema_rate"] = instant_rate
                else:
                    state["ema_rate"] = (state["ema_rate"] * 0.65) + (instant_rate * 0.35)
            state["last_time"] = now
            state["last_fraction"] = stage_fraction

        eta_seconds = None
        if stage_fraction > 0.001 and stage_fraction < 1.0 and stage_elapsed > 0.25:
            if state["ema_rate"] > 1e-6:
                eta_seconds = max(0.0, (1.0 - stage_fraction) / state["ema_rate"])
            else:
                estimated_total = stage_elapsed / stage_fraction
                eta_seconds = max(0.0, estimated_total - stage_elapsed)
        if eta_seconds is not None:
            detail_text = f"{detail_text}\n{stage_name} time left: {self._format_render_eta(eta_seconds)}"
        self._set_render_progress(overall_fraction, overlay, detail_text)


    def _bundled_ffmpeg_path(self):
        for candidate in _BUNDLED_FFMPEG_CANDIDATES:
            if os.path.isfile(candidate):
                return candidate
        return ""


    def _resolve_ffmpeg_path(self, requested_path):
        requested_path = (requested_path or "").strip()
        if not requested_path:
            requested_path = "ffmpeg"
        if os.path.isfile(requested_path):
            return requested_path
        if requested_path.lower() == "ffmpeg":
            bundled = self._bundled_ffmpeg_path()
            if bundled:
                return bundled
        resolved = shutil.which(requested_path)
        return resolved


    def _normalize_render_output_path(self, output_path, codec_label):
        output_path = (output_path or "").strip()
        if not output_path:
            return output_path
        _, default_ext = self._codec_settings(codec_label)
        root, ext = os.path.splitext(output_path)
        if not ext:
            return output_path + default_ext
        return output_path


    def _format_ffmpeg_error(self, ffmpeg_cmd, returncode, stderr_output):
        stderr_output = (stderr_output or "").strip()
        if stderr_output:
            stderr_lines = [line.rstrip() for line in stderr_output.splitlines() if line.strip()]
            if len(stderr_lines) > 18:
                stderr_lines = stderr_lines[-18:]
            stderr_text = "\n".join(stderr_lines)
        else:
            stderr_text = f"ffmpeg exited with code {returncode}."
        return (
            f"FFmpeg failed while encoding the video.\n\n"
            f"Command:\n{' '.join(ffmpeg_cmd)}\n\n"
            f"Details:\n{stderr_text}"
        )


    def _build_watermark_filter(self, total_duration):
        start_time = max(0.0, float(total_duration) - 5.0)
        fade_in_end = start_time + 1.0
        hold_end = start_time + 4.0
        end_time = start_time + 5.0
        alpha_expr = (
            f"0.25*if(lt(t,{start_time:.3f}),0,"
            f"if(lt(t,{fade_in_end:.3f}),(t-{start_time:.3f})/1.0,"
            f"if(lt(t,{hold_end:.3f}),1,"
            f"if(lt(t,{end_time:.3f}),({end_time:.3f}-t)/1.0,0))))"
        )
        filter_parts = [
            "drawtext=text='Rendered with LWMP'",
            "fontcolor=white",
            "fontsize=28",
            "x=w-tw-18",
            "y=h-th-16",
            f"alpha='{alpha_expr}'",
        ]
        fontfile = self._preferred_drawtext_font()
        if fontfile:
            filter_parts.insert(1, f"fontfile='{self._escape_drawtext_value(fontfile)}'")
        return ":".join(filter_parts)

    @staticmethod
    def _apply_lookahead_limiter(samples, threshold_db=-0.5, lookahead_ms=5.0, release_ms=50.0, sr=44100):
        """True-peak look-ahead limiter on interleaved float32 stereo samples."""
        from numpy.lib.stride_tricks import sliding_window_view

        threshold = 10.0 ** (threshold_db / 20.0)
        lookahead = max(1, int(sr * lookahead_ms / 1000.0))
        release_coeff = math.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))

        frames = samples.reshape(-1, 2)
        peak = np.abs(frames).max(axis=1).astype(np.float32)

        extended = np.concatenate([peak, np.full(lookahead - 1, peak[-1])])
        lookahead_peak = sliding_window_view(extended, lookahead).max(axis=1)

        gain = np.where(lookahead_peak > threshold, threshold / lookahead_peak, 1.0).astype(np.float32)
        smoothed = np.empty_like(gain)
        smoothed[0] = gain[0]
        for i in range(1, gain.size):
            if gain[i] < smoothed[i - 1]:
                smoothed[i] = gain[i]
            else:
                smoothed[i] = release_coeff * smoothed[i - 1] + (1.0 - release_coeff) * gain[i]
        return (frames * smoothed[:, np.newaxis]).ravel()

    def _render_audio_to_wav(self, wav_path, parsed_midi, audio_limiter=False):
        stage_start_time = time.monotonic()
        bass_cls = self.controller.bass_engine_cls
        if bass_cls is None:
            raise RuntimeError("BASSMIDI engine not available for export.")

        bass_cls = self.controller.bass_engine_cls
        if bass_cls is None:
            raise RuntimeError("BASSMIDI engine not available for export.")

        latest_config = self._load_config()
        audio_cfg = latest_config.get("audio", {})
        soundfont_path = audio_cfg.get("soundfont_path")
        if not soundfont_path or not os.path.exists(soundfont_path):
            raise RuntimeError("A valid SoundFont is required for video rendering.")

        engine = bass_cls({}, soundfont_path=soundfont_path, buffering=True, debug=self._DEBUG)
        try:
            render_volume = float(audio_cfg.get("volume", 0.5))
            render_voices = int(audio_cfg.get("voices", 512))
            engine.set_volume(render_volume)
            if hasattr(engine, "set_voices"):
                engine.set_voices(render_voices)
            if hasattr(engine, "set_emergency_recovery"):
                engine.set_emergency_recovery(False)
                if hasattr(engine, "set_voices"):
                    engine.set_voices(render_voices)
            if hasattr(engine, "set_pitch_bend_range"):
                engine.set_pitch_bend_range(12)

            times, statuses, params = self._build_buffered_event_arrays(parsed_midi)
            engine.upload_events(times, statuses, params)
            engine.set_current_time(0.0)

            total_duration = float(parsed_midi.total_duration_sec)
            chunk_seconds = 1.0 / 60.0
            rendered_audio_time = 0.0
            sample_rate = 44100
            channels = 2

            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                while rendered_audio_time < total_duration:
                    if self._render_cancelled.is_set():
                        return
                    remaining = total_duration - rendered_audio_time
                    requested_seconds = min(chunk_seconds, remaining)
                    target_frames = max(1, int(round(requested_seconds * sample_rate)))
                    target_samples = target_frames * channels
                    pcm_chunk = engine.render_pcm_chunk(requested_seconds)
                    if not pcm_chunk:
                        pcm_int16 = np.zeros(target_samples, dtype=np.int16)
                    else:
                        pcm_float = np.frombuffer(pcm_chunk, dtype=np.float32)
                        if audio_limiter:
                            pcm_float = self._apply_lookahead_limiter(pcm_float, sr=sample_rate)
                        pcm_int16 = np.clip(pcm_float, -1.0, 1.0)
                        pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)
                        sample_delta = target_samples - pcm_int16.size
                        if sample_delta > 0:
                            pcm_int16 = np.pad(pcm_int16, (0, sample_delta), mode="constant")
                        elif sample_delta < 0:
                            pcm_int16 = pcm_int16[:target_samples]
                    wav_file.writeframes(pcm_int16.tobytes())
                    rendered_audio_time = min(total_duration, rendered_audio_time + requested_seconds)
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Audio",
                        stage_start_time,
                        rendered_audio_time / max(total_duration, 0.001),
                        0.45 * (rendered_audio_time / max(total_duration, 0.001)),
                        f"Audio {rendered_audio_time:.1f}s / {total_duration:.1f}s",
                        "Rendering audio...",
                    )
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass

    _HW_CODEC_MAP = {
        "H.264 (NVENC)":   (["-c:v", "h264_nvenc",  "-pix_fmt", "yuv420p"], ".mp4"),
        "H.264 (QSV)":     (["-c:v", "h264_qsv",    "-pix_fmt", "yuv420p"], ".mp4"),
        "H.264 (AMF)":     (["-c:v", "h264_amf",    "-pix_fmt", "yuv420p"], ".mp4"),
        "H.265 (NVENC)":   (["-c:v", "hevc_nvenc",  "-pix_fmt", "yuv420p"], ".mp4"),
        "H.265 (QSV)":     (["-c:v", "hevc_qsv",    "-pix_fmt", "yuv420p"], ".mp4"),
        "H.265 (AMF)":     (["-c:v", "hevc_amf",    "-pix_fmt", "yuv420p"], ".mp4"),
        "H.265":           (["-c:v", "libx265",     "-pix_fmt", "yuv420p"], ".mp4"),
        "MPEG-4":          (["-c:v", "mpeg4"],                                     ".mp4"),
    }


    def _codec_settings(self, codec_label):
        if codec_label in self._HW_CODEC_MAP:
            return self._HW_CODEC_MAP[codec_label]
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p"], ".mp4"
    _HW_ENCODER_PROBES = [
        ("h264_nvenc",  "H.264 (NVENC)"),
        ("hevc_nvenc",  "H.265 (NVENC)"),
        ("h264_qsv",    "H.264 (QSV)"),
        ("hevc_qsv",    "H.265 (QSV)"),
        ("h264_amf",    "H.264 (AMF)"),
        ("hevc_amf",    "H.265 (AMF)"),
    ]


    def _detect_hw_encoders(self, ffmpeg_path):
        """Return list of supported HW codec labels for the given ffmpeg binary."""
        try:
            result = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
            )
            out = result.stdout
        except Exception:
            return []
        found = []
        for encoder_name, label in self._HW_ENCODER_PROBES:
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1] == encoder_name:
                    found.append(label)
                    break
        return found


    def _render_video_stream_only(self, ffmpeg_bin, parsed_midi, settings, video_output_path):
        stage_start_time = time.monotonic()
        codec_args, _ = self._codec_settings(settings["codec"])
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{settings['width']}x{settings['height']}",
            "-r", str(settings["framerate"]),
            "-i", "-",
            *codec_args,
            "-b:v", settings["bitrate"],
            "-an",
            video_output_path,
        ]

        self._queue_ui(
            self._render_progress_with_timing,
            "Video",
            stage_start_time,
            0.0,
            0.48,
            "Video",
            "Rendering piano roll frames...",
        )
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._ffmpeg_processes.append(process)
        piano_roll = self._PianoRoll(settings["width"], settings["height"], self._CONFIG)
        piano_roll.set_export_mode(True)
        piano_roll.set_nps_spikes(getattr(self.controller, "max_nps_spikes", []))
        piano_roll.set_stats_context(
            parsed_midi.note_events_for_playback["on_time"],
            getattr(parsed_midi, "sorted_off_times", None),
            getattr(parsed_midi, "tempo_events", None),
            float(parsed_midi.total_duration_sec),
            int(len(parsed_midi.note_events_for_playback)),
            float(settings.get("stats_multiplier", 1.0)),
            bool(settings.get("enable_stats_modification", False)),
            settings.get("selected_spike"),
            float(settings.get("spike_intensity", 1.0)),
        )
        piano_roll.set_preferred_color_mode(getattr(parsed_midi, "preferred_color_mode", "track"))
        ffmpeg_stderr_chunks = []

        def _drain_ffmpeg_stderr():
            try:
                while process.stderr:
                    chunk = process.stderr.read(4096)
                    if not chunk:
                        break
                    ffmpeg_stderr_chunks.append(chunk)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_ffmpeg_stderr, daemon=True)
        stderr_thread.start()
        live_preview = self._render_live_preview_enabled
        try:
            try:
                piano_roll.init_pygame_and_gl(hidden=not live_preview, disable_vsync=live_preview)
            except Exception:
                piano_roll.init_pygame_and_gl(hidden=False)
            notes_for_gpu = np.ascontiguousarray(parsed_midi.note_data_for_gpu)
            piano_roll.load_midi(notes_for_gpu, lambda: 0.0)
            total_duration = float(parsed_midi.total_duration_sec)
            total_frames = max(1, int(math.ceil(total_duration * settings["framerate"])))


            for frame_idx in range(total_frames):
                if self._render_cancelled.is_set():
                    break
                current_time = min(total_duration, frame_idx / float(settings['framerate']))
                self._render_current_time = current_time
                piano_roll.draw(current_time, present=False)
                try:
                    process.stdin.write(piano_roll.capture_frame_rgb())
                except (BrokenPipeError, OSError, ValueError):
                    break
                if live_preview:
                    self._pygame.display.flip()
                    for _ev in self._pygame.event.get():
                        if _ev.type == self._pygame.QUIT:
                            self._render_cancelled.set()
                            break
                if process.poll() is not None:
                    break
                if frame_idx % max(1, settings['framerate'] // 2) == 0:
                    fraction_start = 0.5 if settings['render_audio'] else 0.02
                    fraction_span = 0.28 if settings['render_audio'] else 0.94
                    stage_fraction = (frame_idx + 1) / float(total_frames)
                    fraction = fraction_start + (fraction_span * ((frame_idx + 1) / float(total_frames)))
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Video",
                        stage_start_time,
                        stage_fraction,
                        fraction,
                        f"Frame {frame_idx + 1:,} / {total_frames:,}",
                        "Rendering piano roll frames...",
                    )
        finally:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                piano_roll.cleanup()
            except Exception:
                pass

        process.wait()
        stderr_thread.join(timeout=2.0)
        if self._render_cancelled.is_set():
            return
        ffmpeg_stderr = b"".join(ffmpeg_stderr_chunks).decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(self._format_ffmpeg_error(ffmpeg_cmd, process.returncode, ffmpeg_stderr))


    def _finalize_render_output(self, ffmpeg_bin, video_path, output_path, settings, total_duration, audio_path=None):
        stage_start_time = time.monotonic()
        codec_args, _ = self._codec_settings(settings["codec"])
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-i", video_path,
        ]
        if audio_path:
            ffmpeg_cmd.extend(["-i", audio_path])
        if settings.get("show_watermark", True):
            ffmpeg_cmd.extend(["-vf", self._build_watermark_filter(total_duration)])
        ffmpeg_cmd.extend([
            *codec_args,
            "-b:v", settings["bitrate"],
        ])
        if audio_path:
            ffmpeg_cmd.extend([
            "-c:a", "aac",
            "-b:a", settings["audio_bitrate"],
            "-shortest",
            ])
        else:
            ffmpeg_cmd.append("-an")
        ffmpeg_cmd.append(output_path)
        self._queue_ui(
            self._render_progress_with_timing,
            "Finalizing",
            stage_start_time,
            0.0,
            0.82,
            "Finalizing",
            "Finalizing and writing video...",
        )
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._ffmpeg_processes.append(process)
        try:
            stderr_output = ""
            if process.stderr:
                try:
                    stderr_output = process.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        finally:
            try:
                self._ffmpeg_processes.remove(process)
            except ValueError:
                pass
        if self._render_cancelled.is_set():
            return
        if process.returncode != 0:
            raise RuntimeError(self._format_ffmpeg_error(ffmpeg_cmd, process.returncode, stderr_output))


    def start_render_video(self, sender=None, app_data=None):
        if self.render_thread and self.render_thread.is_alive():
            self._message_warning("Render Busy", "A render is already in progress.")
            return
        if self.controller.parsed_midi is None:
            self._message_warning("No MIDI", "Load a MIDI before rendering.")
            return
        if self.piano_roll and self.piano_roll.app_running.is_set():
            self._message_warning("Close Piano Roll", "Close the live piano roll before starting a video render.")
            return
        if self._PianoRoll is None:
            self._message_warning("Piano Roll Unavailable", "Piano roll rendering is not available in this build.")
            return

        ffmpeg_path = dpg.get_value("render_ffmpeg_path").strip()
        output_path = self._normalize_render_output_path(dpg.get_value("render_output_path").strip(), dpg.get_value("render_codec"))
        if not output_path:
            self._message_warning("Output Required", "Enter an output video path before rendering.")
            return

        codec_label = dpg.get_value("render_codec")
        resolution = dpg.get_value("render_resolution")
        framerate = max(1, int(dpg.get_value("render_framerate")))
        bitrate = dpg.get_value("render_bitrate").strip() or "20M"
        audio_bitrate = dpg.get_value("render_audio_bitrate").strip() or "320k"
        stats_multiplier_raw = str(dpg.get_value("render_stats_multiplier")).strip()
        try:
            stats_multiplier = float(stats_multiplier_raw.lstrip("xX")) if stats_multiplier_raw else 1.0
        except ValueError:
            self._message_warning("Invalid Multiplier", "Enter a valid numeric stats multiplier, like 2 or 2.5.")
            return
        stats_multiplier = max(0.1, stats_multiplier)
        spike_intensity_raw = str(dpg.get_value("render_spike_intensity")).strip()
        try:
            spike_intensity = float(spike_intensity_raw.lstrip("xX")) if spike_intensity_raw else 1.0
        except ValueError:
            self._message_warning("Invalid Spike Intensity", "Enter a valid numeric spike intensity, like 1.5 or 2.")
            return
        spike_intensity = max(0.0, spike_intensity)
        enable_stats_modification = bool(dpg.get_value("render_stats_mod_checkbox"))
        selected_spike_label = str(dpg.get_value("render_spike_select") or "None")
        selected_spike = self.render_spike_label_map.get(selected_spike_label)
        render_audio = bool(dpg.get_value("render_audio_checkbox"))
        show_stats_overlay = bool(dpg.get_value("render_stats_overlay_checkbox"))
        show_watermark = bool(dpg.get_value("render_watermark_checkbox"))
        audio_limiter = bool(dpg.get_value("render_audio_limiter_checkbox"))

        render_cfg = self._render_cfg()
        render_cfg["ffmpeg_path"] = ffmpeg_path
        render_cfg["output_path"] = output_path
        render_cfg["codec"] = codec_label
        render_cfg["resolution"] = resolution
        render_cfg["framerate"] = framerate
        render_cfg["bitrate"] = bitrate
        render_cfg["audio_bitrate"] = audio_bitrate
        render_cfg["enable_stats_modification"] = enable_stats_modification
        render_cfg["stats_multiplier"] = stats_multiplier
        render_cfg["spike_selection"] = selected_spike_label
        render_cfg["spike_intensity"] = spike_intensity
        render_cfg["render_audio"] = render_audio
        render_cfg["show_stats_overlay"] = show_stats_overlay
        render_cfg["show_watermark"] = show_watermark
        render_cfg["audio_limiter"] = audio_limiter
        self._save_config(self._CONFIG)

        width_str, height_str = resolution.split(" x ")
        settings = {
            "ffmpeg_path": ffmpeg_path,
            "output_path": output_path,
            "codec": codec_label,
            "width": int(width_str),
            "height": int(height_str),
            "framerate": framerate,
            "bitrate": bitrate,
            "audio_bitrate": audio_bitrate,
            "enable_stats_modification": enable_stats_modification,
            "stats_multiplier": stats_multiplier,
            "selected_spike": selected_spike,
            "spike_intensity": spike_intensity,
            "render_audio": render_audio,
            "show_stats_overlay": show_stats_overlay,
            "show_watermark": show_watermark,
            "audio_limiter": audio_limiter,
        }

        dpg.set_value("render_output_path", output_path)
        dpg.set_value("render_stats_multiplier", str(stats_multiplier))
        dpg.set_value("render_spike_intensity", str(spike_intensity))
        self._render_cancelled.clear()
        self._ffmpeg_processes.clear()
        dpg.configure_item("start_render_button", show=False)
        self._render_current_time = 0.0
        self._set_render_progress(0.0, "Preparing", "Preparing video render...")
        self.render_start_time_monotonic = time.monotonic()
        self.render_stage_timing = {}
        self.render_thread = threading.Thread(target=self._render_video_job, args=(settings,), daemon=True)
        self.render_thread.start()


    def _terminate_process(self, proc, timeout=3.0):
        """Gracefully terminate a subprocess; force-kill after timeout seconds."""
        try:
            if proc.poll() is not None:
                return
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            try:
                if os.name == 'nt':
                    subprocess.call(
                        ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000),
                    )
            except Exception:
                pass


    def _render_video_job(self, settings):
        ffmpeg_bin = self._resolve_ffmpeg_path(settings["ffmpeg_path"])
        if not ffmpeg_bin:
            self._queue_ui(self._set_render_progress, 0.0, "FFmpeg missing", "FFmpeg executable not found. Set a valid ffmpeg path.")
            dpg.configure_item("start_render_button", show=True)
            self._queue_ui(dpg.enable_item, "start_render_button")
            return

        parsed_midi = self.controller.parsed_midi
        if parsed_midi is None:
            self._queue_ui(self._set_render_progress, 0.0, "No MIDI", "No parsed MIDI is available for rendering.")
            dpg.configure_item("start_render_button", show=True)
            self._queue_ui(dpg.enable_item, "start_render_button")
            return

        output_path = settings["output_path"]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            with tempfile.TemporaryDirectory(prefix="lwmp_render_") as temp_dir:
                wav_path = os.path.join(temp_dir, "audio.wav")
                video_only_path = os.path.join(temp_dir, "video_only.mp4")

                if settings["render_audio"]:
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Audio",
                        time.monotonic(),
                        0.0,
                        0.02,
                        "Audio",
                        "Rendering full audio track...",
                    )
                    self._render_audio_to_wav(wav_path, parsed_midi, audio_limiter=settings.get("audio_limiter", False))
                    if self._render_cancelled.is_set():
                        raise InterruptedError("Render cancelled by user.")
                else:
                    self._queue_ui(
                        self._render_progress_with_timing,
                        "Video",
                        time.monotonic(),
                        0.0,
                        0.02,
                        "Video",
                        "Skipping audio render.",
                    )

                self._render_video_stream_only(ffmpeg_bin, parsed_midi, settings, video_only_path)

                if self._render_cancelled.is_set():
                    raise InterruptedError("Render cancelled by user.")

                if settings["render_audio"]:
                    self._finalize_render_output(
                        ffmpeg_bin,
                        video_only_path,
                        output_path,
                        settings,
                        parsed_midi.total_duration_sec,
                        audio_path=wav_path,
                    )
                else:
                    self._finalize_render_output(
                        ffmpeg_bin,
                        video_only_path,
                        output_path,
                        settings,
                        parsed_midi.total_duration_sec,
                    )

            if self._render_cancelled.is_set():
                raise InterruptedError("Render cancelled by user.")

            total_elapsed = self._format_render_eta(time.monotonic() - self.render_start_time_monotonic)
            self._queue_ui(
                self._set_render_progress,
                1.0,
                "Done",
                f"Render complete: {output_path}\nTime elapsed: {total_elapsed}",
            )
            self._queue_ui(self.set_status, f"Render complete: {os.path.basename(output_path)}")
            self._queue_ui(self._message_info, "Render Complete", f"Video saved to:\n{output_path}")
        except InterruptedError:
            self._queue_ui(self._set_render_progress, 0.0, "Cancelled", "Render was cancelled.")
            self._queue_ui(self.set_status, "Render cancelled")
            self._queue_ui(dpg.configure_item, "render_window", show=True)
        except Exception as e:
            traceback.print_exc()
            self._queue_ui(self._set_render_progress, 0.0, "Render Failed", str(e))
            self._queue_ui(self._message_error, "Render Failed", str(e))
        finally:
            for proc in list(self._ffmpeg_processes):
                try:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                except Exception:
                    pass
                self._terminate_process(proc)
            self._ffmpeg_processes.clear()
            dpg.configure_item("start_render_button", show=True)
            self._queue_ui(dpg.enable_item, "start_render_button")

    @property


    def _render_live_preview_enabled(self):
        return bool(dpg.does_item_exist("render_live_preview_checkbox") and dpg.get_value("render_live_preview_checkbox"))


