#!/usr/bin/env python3

VERSION_MAJOR = 0    # Major version: 0 = beta/alpha, 1+ = release
VERSION_FEATURE = 1  # Feature update: increment for new features/additions
VERSION_HOTFIX = 1   # Hotfix number: increment for confirmed bug fixes

VERSION = f"{VERSION_MAJOR}.{VERSION_FEATURE}{VERSION_HOTFIX}"
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, ttk
import os
import sys # Need this to exit
import multiprocessing # For parallel parsing AND new piano roll
import atexit # To clean up child processes
import traceback # For logging thread errors
import numpy as np
import subprocess # For GPU detection via wmic
import psutil

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir) # Add the 'MIDI' directory itself
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, 'PARSER')) # Add the 'PARSER' directory

from midi_parser import GPU_NOTE_DTYPE, MidiParser

try: 
    from pianoroll import PianoRoll
    import pygame
except ImportError:
    print("pianoroll.py not found. Piano roll will be disabled.")
    PianoRoll = None

from config import load_config, save_config, setup_omnimidi_preference
from player_controller import PlayerController

CONFIG = load_config()
CONFIG = setup_omnimidi_preference(CONFIG)

DEBUG = False

try:
    if os.name == 'nt':
        from midi_engine import OmniMidiEngine
    else:
        OmniMidiEngine = None
except ImportError:
    OmniMidiEngine = None
    print("midi_engine.py or OmniMIDI.dll not found.")
except Exception as e:
    OmniMidiEngine = None
    print(f"Error importing OmniMidiEngine: {e}")

try:
    from midi_engine_cython import BassMidiEngine
    print("Loaded Cython BASSMIDI engine.")
except ImportError:
    try:
        from bassmidi_engine import BassMidiEngine
        print("Loaded Python BASSMIDI engine.")
    except ImportError:
        BassMidiEngine = None
        print("No BASSMIDI engine found.")
except Exception as e:
    try:
        from bassmidi_engine import BassMidiEngine
        print(f"Cython BASSMIDI engine failed, using Python fallback: {e}")
    except Exception as python_error:
        BassMidiEngine = None
        print(f"Error importing BassMidiEngine: {python_error}")

import threading
import time
import math 
from collections import deque 
import bisect 
import heapq 

class StartupWizard(tk.Toplevel):
    def __init__(self, parent, on_complete):
        super().__init__(parent)
        self.on_complete = on_complete
        
        self.result_mode = 'local' 
        self.recommended_res = (1280, 720)
        self.recommended_note_limit = 0
        self.has_bundled_omnimidi = os.path.exists(os.path.join(script_dir, "OmniMIDI.dll"))
        
        self.title("System Check")
        self.geometry("550x520") 
        self.resizable(False, False)
        self.config(bg="#f0f0f0")
        
        self.transient(parent)
        self.grab_set()
        
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 275
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 260
        self.geometry(f"+{x}+{y}")

        tk.Label(self, text="Performing System Check...", font=("Segoe UI", 12, "bold"), bg="#f0f0f0").pack(pady=10)
        
        self.info_text = tk.Text(self, height=8, width=65, font=("Consolas", 9), relief=tk.FLAT, bg="#e0e0e0")
        self.info_text.pack(pady=5, padx=10)
        self.info_text.insert(tk.END, "Initializing...\n")
        self.info_text.config(state=tk.DISABLED)
        
        self.rec_frame = tk.LabelFrame(self, text="Analysis Results", font=("Segoe UI", 10), bg="#f0f0f0", padx=10, pady=5)
        self.rec_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.rec_label = tk.Label(self.rec_frame, text="Analyzing Hardware...", font=("Segoe UI", 10), justify=tk.LEFT, bg="#f0f0f0", fg="#333")
        self.rec_label.pack(anchor='w')

        self.warn_label = tk.Label(self.rec_frame, text="", font=("Segoe UI", 9, "bold"), justify=tk.LEFT, bg="#f0f0f0", fg="#D00000")
        self.warn_label.pack(anchor='w', pady=(10, 0))
        
        self.btn_frame = tk.Frame(self, bg="#f0f0f0")
        self.btn_frame.pack(side=tk.BOTTOM, pady=15)
        
        self.btn_bass = tk.Button(self.btn_frame, text="Use BASSMIDI (Buffered)", command=self.use_bassmidi, state=tk.DISABLED, width=30)
        self.btn_bass.pack(pady=2)

        self.btn_recommend = tk.Button(self.btn_frame, text="Use Recommended Settings", command=self.accept_recommendation, state=tk.DISABLED, width=30)
        self.btn_recommend.pack(pady=2)
        
        self.btn_default = tk.Button(self.btn_frame, text="Use Defaults (System PATH, 720p)", command=self.use_default_settings, state=tk.DISABLED, width=30)
        self.btn_default.pack(pady=2)
        
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.after(500, self.run_hardware_check)

    def log(self, message):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"> {message}\n")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)
        
    def use_bassmidi(self):
        self.btn_bass.config(text="Initializing...", state=tk.DISABLED)
        self.btn_recommend.config(state=tk.DISABLED)
        self.btn_default.config(state=tk.DISABLED)
        self.update()
        
        self.on_complete('bassmidi', self.recommended_res)
        self.destroy()

    def run_hardware_check(self):
        mode = 'local' if self.has_bundled_omnimidi else 'path'
        
        try:
            mem = psutil.virtual_memory()
            total_ram_gb = mem.total / (1024 ** 3)
            available_ram_gb = mem.available / (1024 ** 3)
            percent_used = mem.percent
            
            self.log(f"RAM: {total_ram_gb:.1f} GB Total ({percent_used}% Used)")
            self.log(f"Available: {available_ram_gb:.1f} GB")
            
            notes_per_gb = 19_000_000 
            
            est_limit = int(available_ram_gb * notes_per_gb)
            if est_limit < 1_000_000: est_limit = 1_000_000

            self.recommended_note_limit = est_limit
            self.log(f"Calc: Free RAM can hold ~{est_limit/1_000_000:.1f}M notes")
            
        except ImportError:
            self.log("RAM: psutil not installed (Skip)")
            self.recommended_note_limit = 20_000_000 # Safe default
        except Exception as e:
            self.log(f"RAM Check Error: {e}")
            self.recommended_note_limit = 20_000_000

        self.recommended_mode = mode

        has_dedicated_gpu = False
        gpu_names = []
        try:
            startupinfo = None
            if os.name == 'nt':
                try:
                    startupinfo = subprocess.STARTUPINFO()
                    useshowwindow = getattr(subprocess, 'STARTUPF_USESHOWWINDOW', 1)
                    startupinfo.dwFlags |= useshowwindow
                except Exception:
                    startupinfo = None
            
            cmd = "wmic path win32_videocontroller get name"
            output = subprocess.check_output(cmd, startupinfo=startupinfo, stderr=subprocess.STDOUT).decode()
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            
            if len(lines) > 1:
                raw_gpu_names = lines[1:]
                ignored_keywords = ['parsec', 'virtualbox', 'vbox', 'vmware', 'remote', 'citrix', 'hyper-v', 'rdp', 'radmin', 'teamviewer']
                filtered_gpus = [name for name in raw_gpu_names if not any(k in name.lower() for k in ignored_keywords)]
                gpu_names = filtered_gpus if filtered_gpus else raw_gpu_names
                
                self.log(f"GPU(s): {', '.join(gpu_names)}")
                
                for name in gpu_names:
                    n = name.lower()
                    if any(x in n for x in ['nvidia', 'geforce', 'radeon rx', 'rtx', 'gtx', 'quadro', 'arc', 'titan']):
                        has_dedicated_gpu = True
                        
        except Exception as e:
            self.log(f"GPU Check skipped: {e}")
            has_dedicated_gpu = True

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.log(f"Screen: {screen_w}x{screen_h}")

        if screen_h <= 768:
            res_str = "640x360"
            self.recommended_res = (640, 360)
        else:
            res_str = "1366x768"
            self.recommended_res = (1366, 768)

        rec_mode_str = "OmniMIDI (Bundled DLL)" if mode == 'local' else "OmniMIDI (System PATH)"
        rec_notes_str = f"{self.recommended_note_limit/1_000_000:.1f} Million"
        
        final_text = (
            f"Recommended Usage: {rec_mode_str}\n"
            f"Recommended Notes: {rec_notes_str} (based on Free RAM)\n"
            f"Recommended Res:   {res_str} (for Piano Roll)"
        )
        self.rec_label.config(text=final_text, fg="blue", font=("Consolas", 10, "bold"))
        
        warning_lines = []

        if not self.has_bundled_omnimidi:
            warning_lines.append("Bundled OmniMIDI DLL not found. System PATH mode is recommended.")
            
        if gpu_names and not has_dedicated_gpu:
            warning_lines.append("WARNING: Integrated GPU detected. Memory usage will be higher than expected!")
            
        if warning_lines:
            self.warn_label.config(text="\n".join(warning_lines))
        
        self.log("Analysis Complete.")
        
        self.btn_recommend.config(state=tk.NORMAL)
        self.btn_default.config(state=tk.NORMAL)
            
        if BassMidiEngine:
            self.btn_bass.config(state=tk.NORMAL)

    def accept_recommendation(self):
        self.btn_recommend.config(text="Initializing...", state=tk.DISABLED)
        self.btn_default.config(state=tk.DISABLED)
        self.update()
        
        self.on_complete(self.recommended_mode, self.recommended_res)
        self.destroy()

    def use_default_settings(self):
        self.btn_default.config(text="Initializing...", state=tk.DISABLED)
        self.btn_recommend.config(state=tk.DISABLED)
        self.update()

        self.on_complete('path', (1280, 720))
        self.destroy()

def run_parser_process(filepath, queue):
    """
    Runs the MIDI parsing in a separate process to avoid blocking the GUI.
    """
    try: 
        print("[DEBUG] run_parser_process: Creating MidiParser...")
        parser = MidiParser(filepath)
        print("[DEBUG] run_parser_process: MidiParser created. Counting events...")
        total_events = parser.count_total_events()
        print(f"[DEBUG] run_parser_process: count_total_events returned {total_events}")
        queue.put(('total_events', total_events))
        
        print("[DEBUG] run_parser_process: Starting parse()...")
        parser.parse(queue, total_events=total_events)
        print("[DEBUG] run_parser_process: parse() completed.")
        
        result_data = {
            'filename': parser.filename,
            'ticks_per_beat': parser.ticks_per_beat,
            'total_duration_sec': parser.total_duration_sec,
            'note_data_for_gpu': parser.note_data_for_gpu,
            'note_events_for_playback': parser.note_events_for_playback,
            'pitch_bend_events': parser.pitch_bend_events,
        }
        print("[DEBUG] run_parser_process: Sending 'success' with raw data dict...")
        queue.put(('success', result_data))
        print("[DEBUG] run_parser_process: 'success' message sent to queue.")
    except Exception as e:
        import traceback
        print(f"[DEBUG] run_parser_process: EXCEPTION: {e}")
        traceback.print_exc()
        queue.put(('error', str(e)))

class MidiPlayerApp:
    CONTROLLER_FIELDS = {
        'parsed_midi',
        'total_song_notes',
        'total_song_duration',
        'active_midi_backend',
        'playing',
        'paused',
        'playback_start_time',
        'paused_at_time',
        'total_paused_duration',
        'notes_played_count',
        'last_processed_event_time',
        'current_lag',
        'buffered_playback_start_offset',
        'current_playback_time_for_threads',
        'nps_event_timestamps',
        'last_nps_graph_update_time',
        'last_lag_update_time',
        'last_lag_value',
        'slowdown_percentage',
        'is_seeking',
        'paused_for_seeking',
        'seek_request_time',
        'parser_process',
        'parser_queue',
    }

    def __getattr__(self, name):
        if name in self.CONTROLLER_FIELDS:
            return getattr(self.controller, name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name != 'controller' and 'controller' in self.__dict__ and name in self.CONTROLLER_FIELDS:
            setattr(self.controller, name, value)
            return
        super().__setattr__(name, value)

    def __init__(self, root):
        self.root = root
        self.root.title("Lightweight MIDI Player")

        self.controller = PlayerController(
            CONFIG,
            script_dir,
            BassMidiEngine,
            OmniMidiEngine,
            save_config,
            debug=DEBUG,
        )

        self.playback_thread = None
        
        self.loading_window = None
        self.playback_lock = threading.Lock()

        self.process = None
        self.cpu_history = deque([0.0] * 100, maxlen=100)
        
        self.piano_roll = None
        self.piano_roll_thread = None
        self.recommended_piano_roll_res = (1280, 720) 
        self.was_piano_roll_open_before_unload = False
        self.last_piano_roll_res = None

        self.create_widgets()
        self.root.after(100, self.start_system_check)

        atexit.register(self.cleanup)

    def start_system_check(self):
        """Launches the startup wizard on Windows, or bypasses on other OS."""
        def on_wizard_complete(mode, resolution):
            print(f"User selected MIDI mode: {mode}, Recommended Res: {resolution}")
            CONFIG['audio']['omnimidi_load_preference'] = mode
            self.recommended_piano_roll_res = resolution
            
            self.init_midi_backends()
            self.finalize_startup()

        if os.name == 'nt':
            StartupWizard(self.root, on_wizard_complete)
        else:
            self.init_midi_backends()
            self.finalize_startup()

    def finalize_startup(self):
        """Starts CPU monitoring, and GUI update loops. Piano Roll is NOT started automatically."""
        print("Finalizing startup...")
        
        try:
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)
        except ImportError:
            print("psutil not found. CPU monitoring will be disabled.")
        except Exception as e:
            print(f"Failed to initialize psutil: {e}")

        self.update_gui_counters()
        self.update_cpu_graph()

    def show_piano_roll_dialog(self):
        """Opens a dialog to select the piano roll resolution."""
        if self.piano_roll and self.piano_roll.app_running.is_set():
            messagebox.showinfo("Piano Roll", "Piano Roll is already running.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Piano Roll Settings")
        dialog.geometry("300x400")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 200
        dialog.geometry(f"+{x}+{y}")

        tk.Label(dialog, text="Select Resolution", font=("Segoe UI", 12, "bold")).pack(pady=10)

        # Common Resolutions
        common_res = [
            (640, 360), (854, 480), (1024, 576), (1280, 720), 
            (1366, 768), (1600, 900), (1920, 1080), 
            (2560, 1440), (3840, 2160)
        ]
        
        native_w = self.root.winfo_screenwidth()
        native_h = self.root.winfo_screenheight()
        
        # Filter resolutions that fit on screen
        available_res = [res for res in common_res if res[0] <= native_w and res[1] <= native_h]
        
        # Ensure native is in the list
        if (native_w, native_h) not in available_res:
            available_res.append((native_w, native_h))
            
        available_res.sort(key=lambda x: x[0]) # Sort by width

        # Create Listbox
        list_frame = tk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        res_listbox = tk.Listbox(list_frame, font=("Consolas", 10), height=10, activestyle='none')
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=res_listbox.yview)
        res_listbox.config(yscrollcommand=scrollbar.set)
        
        res_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        recommended_index = -1

        for i, (w, h) in enumerate(available_res):
            text = f"{w} x {h}"
            if (w, h) == self.recommended_piano_roll_res:
                text += " (Recommended)"
                recommended_index = i
            res_listbox.insert(tk.END, text)

        # Select the recommended one by default
        if recommended_index != -1:
            res_listbox.select_set(recommended_index)
            res_listbox.see(recommended_index)
        elif available_res:
            res_listbox.select_set(0) # Fallback to first

        def launch():
            sel = res_listbox.curselection()
            if not sel: return
            
            selected_res = available_res[sel[0]]
            dialog.destroy()
            self.launch_piano_roll(selected_res[0], selected_res[1])

        tk.Button(dialog, text="Launch", command=launch, width=20, bg="#dddddd").pack(pady=15)

    def launch_piano_roll(self, width, height):
        if not PianoRoll: return

        self.last_piano_roll_res = (width, height)

        print(f"Initializing Piano Roll at {width}x{height}")
        self.piano_roll = PianoRoll(width, height, CONFIG)
        self.piano_roll_thread = threading.Thread(target=self.run_piano_roll, daemon=True)
        self.piano_roll_thread.start()

        # If a MIDI is already loaded, send it to the piano roll immediately
        if self.parsed_midi:
            print("Sending loaded MIDI to Piano Roll...")
            # We need to do this in a thread-safe way or just call it, 
            # since run_piano_roll checks a queue or draws from state.
            # The existing load_file logic does this:
            notes_for_gpu = np.empty(len(self.parsed_midi.note_events_for_playback), dtype=GPU_NOTE_DTYPE)
            notes_for_gpu['on_time'] = self.parsed_midi.note_events_for_playback['on_time']
            notes_for_gpu['off_time'] = self.parsed_midi.note_events_for_playback['off_time']
            notes_for_gpu['pitch'] = self.parsed_midi.note_events_for_playback['pitch']
            notes_for_gpu['velocity'] = self.parsed_midi.note_events_for_playback['velocity']
            notes_for_gpu['track'] = self.parsed_midi.note_events_for_playback['channel']
            
            # Allow a small delay for the window to init
            self.root.after(500, lambda: self.piano_roll.load_midi(notes_for_gpu, self.get_current_playback_time_thread_safe))

    def run_piano_roll(self):
        # Local reference to avoid race condition on unload
        piano_roll_instance = self.piano_roll
        
        piano_roll_instance.init_pygame_and_gl()
        clock = pygame.time.Clock()

        # Set a timer to unblock the event loop periodically
        pygame.time.set_timer(pygame.USEREVENT, 250)

        last_caption_update_time = 0

        while piano_roll_instance.app_running.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    piano_roll_instance.app_running.clear()
                piano_roll_instance.handle_slider_event(event)
                # We can ignore the USEREVENT, its only purpose is to wake the loop
            
            # Check again in case the QUIT event was processed
            if not piano_roll_instance.app_running.is_set():
                break
    
            current_time = self.get_current_playback_time()
            piano_roll_instance.draw(current_time)
            
            now = time.monotonic()
            # Update caption 5 times a second
            if now - last_caption_update_time > 0.2:
                fps = clock.get_fps()
                pygame.display.set_caption(f"Piano Roll - {fps:.1f} FPS - window {piano_roll_instance.window_seconds:.2f}s - scroll {piano_roll_instance.scroll_speed:.0f}")
                last_caption_update_time = now

            # Use the value from the GUI slider to limit FPS
            fps_limit = self.fps_limit_var.get()
            clock.tick(fps_limit)
            
        piano_roll_instance.cleanup()

    def create_widgets(self):
        # --- Main Layout ---
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Left Frame: Controls and Info ---
        self.controls_frame = tk.Frame(self.left_frame)
        self.controls_frame.pack(pady=5, fill=tk.X)

        button_sub_frame = tk.Frame(self.controls_frame)
        button_sub_frame.pack(side=tk.LEFT)
        
        self.load_button = tk.Button(button_sub_frame, text="Load MIDI", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.play_pause_button = tk.Button(button_sub_frame, text="Play", command=self.toggle_play_pause, state=tk.DISABLED)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_sub_frame, text="Stop", command=self.stop_playback, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # [NEW] Piano Roll Button
        self.piano_roll_button = tk.Button(button_sub_frame, text="Piano Roll", command=self.show_piano_roll_dialog, state=tk.DISABLED)
        self.piano_roll_button.pack(side=tk.LEFT, padx=5)

        self.time_var = tk.StringVar(value="00:00 / 00:00")
        self.time_label = tk.Label(self.controls_frame, textvariable=self.time_var, anchor='e', font=("Arial", 12))
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # --- Seek Bar ---
        self.seek_frame = tk.Frame(self.left_frame)
        self.seek_frame.pack(pady=5, fill=tk.X)
        
        self.seek_slider = tk.Scale(self.seek_frame, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=False, state=tk.DISABLED)
        self.seek_slider.pack(fill=tk.X, expand=True, padx=5)
        
        self.seek_slider.bind("<ButtonPress-1>", self.on_seek_press)
        self.seek_slider.bind("<ButtonRelease-1>", self.on_seek_release)
        
        # --- Info display ---
        self.info_frame = tk.Frame(self.left_frame)
        self.info_frame.pack(pady=5, fill=tk.X)
        
        self.filename_var = tk.StringVar(value="No file loaded.")
        self.filename_label = tk.Label(self.info_frame, textvariable=self.filename_var, anchor='w')
        self.filename_label.pack(fill=tk.X)
        
        self.note_count_var = tk.IntVar(value=0)
        self.total_notes_var = tk.StringVar(value="0")
        self.note_count_label = tk.Label(self.info_frame, text="Notes: 0 / 0", anchor='w', font=("Arial", 12))
        self.note_count_label.pack(fill=tk.X)
        
        # --- Performance Frame ---
        self.performance_frame = tk.LabelFrame(self.left_frame, text="Performance")
        self.performance_frame.pack(fill=tk.X, pady=10, padx=5)

        self.nps_frame = tk.Frame(self.performance_frame)
        self.nps_frame.pack(fill=tk.X, pady=5)
        self.nps_var = tk.StringVar(value="NPS: 0")
        tk.Label(self.nps_frame, textvariable=self.nps_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.ultra_mode_var = tk.BooleanVar(value=False)
        self.ultra_mode_check = tk.Checkbutton(self.nps_frame, text="Ultra Mode (60Hz Graph)", variable=self.ultra_mode_var, font=("Arial", 8))
        self.ultra_mode_check.pack(side=tk.LEFT, padx=5)
        
        # --- NPS Graph ---
        nps_graph_frame = tk.Frame(self.performance_frame)
        nps_graph_frame.pack(pady=(0, 5))
        nps_labels_frame = tk.Frame(nps_graph_frame, width=50)
        nps_labels_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2)); nps_labels_frame.pack_propagate(False)
        self.nps_max_var = tk.StringVar(value="Max: 0")
        tk.Label(nps_labels_frame, textvariable=self.nps_max_var, font=("Arial", 8)).pack(side=tk.TOP, anchor='n')
        tk.Label(nps_labels_frame, text="0", font=("Arial", 8)).pack(side=tk.BOTTOM, anchor='s')
        self.nps_canvas = tk.Canvas(nps_graph_frame, width=200, height=80, bg='#111')
        self.nps_canvas.pack(side=tk.LEFT)
        self.nps_history = deque([0] * 100, maxlen=100)
        self.nps_bars = [self.nps_canvas.create_rectangle(i*2, 80, (i+1)*2, 80, fill='#0F0', width=0) for i in range(100)]
        
        # --- CPU Graph ---
        self.cpu_frame = tk.Frame(self.performance_frame)
        self.cpu_frame.pack(fill=tk.X, pady=(10, 5))
        self.cpu_var = tk.StringVar(value="CPU: 0.0%")
        tk.Label(self.cpu_frame, textvariable=self.cpu_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.vbo_lag_label = tk.Label(self.cpu_frame, font=("Arial", 10, "bold"), bg='#111')
        cpu_graph_frame = tk.Frame(self.performance_frame)
        cpu_graph_frame.pack(pady=(0, 5))
        cpu_labels_frame = tk.Frame(cpu_graph_frame, width=50)
        cpu_labels_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2)); cpu_labels_frame.pack_propagate(False)
        tk.Label(cpu_labels_frame, text="100%", font=("Arial", 8)).pack(side=tk.TOP, anchor='n')
        tk.Label(cpu_labels_frame, text="0%", font=("Arial", 8)).pack(side=tk.BOTTOM, anchor='s')
        self.cpu_canvas = tk.Canvas(cpu_graph_frame, width=200, height=80, bg='#111')
        self.cpu_canvas.pack(side=tk.LEFT)
        self.cpu_bars = [self.cpu_canvas.create_rectangle(i*2, 80, (i+1)*2, 80, fill='#0AF', width=0) for i in range(100)]
        
        # --- Slowdown Indicator ---
        self.slowdown_frame = tk.Frame(self.performance_frame)
        self.slowdown_frame.pack(fill=tk.X, pady=5)
        self.slowdown_var = tk.StringVar(value="Slowdown: 0.0%")
        self.slowdown_label = tk.Label(self.slowdown_frame, textvariable=self.slowdown_var, font=("Arial", 10))
        self.slowdown_label.pack(side=tk.LEFT, padx=5)

        # --- Piano Roll FPS Limiter ---
        self.fps_limit_frame = tk.Frame(self.performance_frame)
        self.fps_limit_frame.pack(fill=tk.X, pady=(10, 5), padx=5)
        
        tk.Label(self.fps_limit_frame, text="Piano Roll FPS Limit:", font=("Arial", 9)).pack(side=tk.LEFT)
        
        self.fps_limit_var = tk.IntVar(value=0) # Default to 0 (unlimited)
        
        def format_fps_label(v):
            val = int(v)
            label = str(val) if val > 0 else "Unlimited"
            if hasattr(self, 'fps_limit_display'):
                self.fps_limit_display.config(text=label)

        self.fps_limit_slider = tk.Scale(
            self.fps_limit_frame, 
            from_=0, 
            to=300, # Increased max to 300
            orient=tk.HORIZONTAL, 
            variable=self.fps_limit_var,
            showvalue=False,
            command=format_fps_label
        )
        self.fps_limit_slider.set(0) # Set default to unlimited
        self.fps_limit_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.fps_limit_display = tk.Label(self.fps_limit_frame, text="Unlimited", font=("Arial", 10, "bold"), width=9, anchor='w')
        self.fps_limit_display.pack(side=tk.LEFT)
        
        # --- Audio Settings Frame ---
        self.audio_frame = tk.LabelFrame(self.left_frame, text="Audio Settings")
        self.audio_frame.pack(fill=tk.X, pady=5, padx=5)

        # Volume Slider
        self.volume_frame = tk.Frame(self.audio_frame)
        self.volume_frame.pack(fill=tk.X, pady=2)
        tk.Label(self.volume_frame, text="Volume:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.volume_var = tk.DoubleVar(value=0.5)
        self.volume_slider = tk.Scale(self.volume_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, 
                                      variable=self.volume_var, showvalue=False, command=self.on_volume_change)
        self.volume_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.volume_label = tk.Label(self.volume_frame, text="50%", font=("Arial", 9), width=4)
        self.volume_label.pack(side=tk.LEFT)

        # Voice Limit Slider
        self.voice_limit_frame = tk.Frame(self.audio_frame)
        self.voice_limit_frame.pack(fill=tk.X, pady=2)
        tk.Label(self.voice_limit_frame, text="Max Voices:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.voice_limit_var = tk.IntVar(value=512)
        self.voice_limit_slider = tk.Scale(self.voice_limit_frame, from_=1, to=2000, orient=tk.HORIZONTAL,
                                           variable=self.voice_limit_var, showvalue=False, command=self.on_voice_limit_change)
        self.voice_limit_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.voice_limit_label = tk.Label(self.voice_limit_frame, text="512", font=("Arial", 9), width=4)
        self.voice_limit_label.pack(side=tk.LEFT)

        # --- Status Bar ---
        self.status_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.status_bar, textvariable=self.status_var, anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.about_button = tk.Button(self.status_bar, text="About", command=self.show_about_window, relief=tk.FLAT, font=("Segoe UI", 8))
        self.about_button.pack(side=tk.RIGHT, padx=5)

    def on_volume_change(self, value):
        val = float(value)
        self.volume_label.config(text=f"{int(val*100)}%")
        if self.active_midi_backend and hasattr(self.active_midi_backend, 'set_volume'):
            try:
                self.active_midi_backend.set_volume(val)
            except Exception as e:
                print(f"Failed to set volume: {e}")

    def on_voice_limit_change(self, value):
        val = int(value)
        self.voice_limit_label.config(text=str(val))
        if self.active_midi_backend and hasattr(self.active_midi_backend, 'set_voices'):
             try:
                 self.active_midi_backend.set_voices(val)
             except Exception as e:
                 print(f"Failed to set voices: {e}")

    def show_about_window(self):
        about_window = Toplevel(self.root)
        about_window.title("About Lightweight MIDI Player")
        about_window.geometry("450x300")
        about_window.resizable(False, False)
        about_window.transient(self.root)
        about_window.grab_set()

        # Center the window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 225
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 150
        about_window.geometry(f"+{x}+{y}")

        main_frame = tk.Frame(about_window, padx=15, pady=15)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # About page
        about_text = (
            "Lightweight MIDI Player (LWMP)\n\n"
            "Version: 0.8\n\n"
            "A (sorta) fast MIDI player designed for handling large MIDI files/Black MIDI "
            "... featuring a real-time piano roll visualizer.\n\n"
            "--- Credits ---\n"
            "Proudly vibe coded by: mappa, LexonBlackzz\n"
            "GUI Framework: Tkinter\n"
            "PFA Piano roll Skin: UMP\n"
            "Piano Roll: Pygame & PyOpenGL\n\n"
            "Thank you (and fuck you) Gemini, ChatGPT and Claude for letting me "
            "do the bare minimum - mappa 2025\n\n"
        )

        text_area = tk.Text(main_frame, wrap=tk.WORD, height=10, width=50)
        text_area.insert(tk.END, about_text)
        text_area.config(state=tk.DISABLED, bg="#f0f0f0", relief=tk.FLAT, font=("Segoe UI", 9))
        text_area.pack(pady=5, expand=True, fill=tk.BOTH)

        close_button = tk.Button(main_frame, text="Close", command=about_window.destroy, width=15)
        close_button.pack(pady=(10, 0))

    def init_midi_backends(self):
        self.controller.init_midi_backends(
            volume=self.volume_var.get(),
            voices=self.voice_limit_var.get(),
            set_status=self.status_var.set,
            refresh_status=self.root.update_idletasks,
            prompt_info=messagebox.showinfo,
            prompt_warning=messagebox.showwarning,
            prompt_error=messagebox.showerror,
            pick_soundfont=lambda: filedialog.askopenfilename(
                filetypes=(("SoundFont", "*.sf2"), ("All files", "*.*"))
            ),
            launch_sweep=lambda backend: threading.Thread(
                target=backend.test_piano_sweep, daemon=True
            ).start(),
        )
            
    def reset_graph_history(self):
        self.nps_history.clear(); self.cpu_history.clear()
        for i in range(100):
            self.nps_history.append(0)
            self.cpu_history.append(0.0)
            self.nps_canvas.itemconfig(self.nps_bars[i], fill='#0F0')
            self.cpu_canvas.itemconfig(self.cpu_bars[i], fill='#0AF')
        self.draw_graph(self.nps_canvas, self.nps_bars, self.nps_history, 100, 'nps')
        self.draw_graph(self.cpu_canvas, self.cpu_bars, self.cpu_history, 100, 'cpu')
        self.nps_max_var.set("Max: 0")

    def draw_graph(self, canvas, bars, data, max_val, graph_type='cpu'):
        if max_val == 0: max_val = 1.0
        for i, val in enumerate(data):
            h = (val / max_val) * 80; h = min(80, max(0, h))
            
            color = '#0F0' # default
            if graph_type == 'nps':
                if val >= 400000:
                    color = '#FF0000' # Red
                elif val >= 200000:
                    color = '#FFA500' # Orange
                elif val >= 50000:
                    color = '#FFFF00' # Yellow
                else:
                    color = '#0F0' # Green
            elif graph_type == 'cpu':
                if val >= 95:
                    color = '#FF0000' # Red
                elif val >= 70:
                    color = '#FFA500' # Orange
                elif val >= 30:
                    color = '#FFFF00' # Yellow
                else:
                    color = '#0AF' # Blue

            canvas.coords(bars[i], i*2, 80 - h, (i+1)*2, 80)
            canvas.itemconfig(bars[i], fill=color)
            
    def format_time(self, seconds):
        if seconds < 0: seconds = 0
        minutes = int(seconds // 60); seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def format_nps(self, n):
        n = int(n)
        if n < 1000000: return str(n)
        return f"{n/1000000:.1f}M".replace(".0M", "M")

    def load_file(self):
        if self.playing:
            self.playing = False; self.paused = False
            self.play_pause_button.config(text="Play")
            
        self.reset_playback_state()
        self.parsed_midi = None; self.filename_var.set("No file loaded.")
        self.time_var.set("00:00 / 00:00"); self.note_count_label.config(text="Notes: 0 / 0")
        self.seek_slider.config(state=tk.DISABLED, to=100); self.seek_slider.set(0)
        self.play_pause_button.config(state=tk.DISABLED)
            
        if self.parser_process and self.parser_process.is_alive():
            messagebox.showwarning("Busy", "Already parsing a file. Please wait."); return

        filepath = filedialog.askopenfilename(filetypes=(("MIDI files", "*.mid"), ("All files", "*.*")))
        if not filepath:
            return

        # --- Create Loading Dialog ---
        self.loading_window = Toplevel(self.root)
        self.loading_window.title("Loading MIDI")
        self.loading_window.geometry("400x120")
        self.loading_window.resizable(False, False)
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()
        
        # Center it
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 200
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 60
        self.loading_window.geometry(f"+{x}+{y}")
        
        self.loading_progress_var = tk.DoubleVar()
        self.loading_label_var = tk.StringVar(value="Initializing...")

        tk.Label(self.loading_window, textvariable=self.loading_label_var, font=("Segoe UI", 10)).pack(pady=10)
        
        progress_bar = ttk.Progressbar(self.loading_window, orient="horizontal", length=360, mode="determinate", variable=self.loading_progress_var)
        progress_bar.pack(pady=5, padx=20)
        
        self.loading_window.protocol("WM_DELETE_WINDOW", lambda: None) # Prevent closing

        try:
            self.filename_var.set(f"Parsing {os.path.basename(filepath)}...")
            self.load_button.config(state=tk.DISABLED) 
            self.controller.start_parse_job(
                filepath,
                multiprocessing.Queue,
                multiprocessing.Process,
                run_parser_process,
            )
            self.root.after(100, self.check_parser_status)
        except Exception as e:
            if self.loading_window:
                self.loading_window.destroy()
                self.loading_window = None
            messagebox.showerror("Error", f"Failed to start parser: {e}")
            self.controller.clear_parse_job()
            self.filename_var.set("Failed to start parser."); self.load_button.config(state=tk.NORMAL)

    def check_parser_status(self):
        try:
            for status, payload in self.controller.poll_parser_messages():
                print(f"[DEBUG] check_parser_status: Got status='{status}', payload type={type(payload).__name__}")

                event = self.controller.handle_parser_message(
                    status, payload, start_padding=3.0, end_padding=3.0
                )

                if event['kind'] == 'total_events':
                    if self.loading_window:
                        self.loading_label_var.set(f"Found {event['total_events']:,} events. Parsing...")

                elif event['kind'] == 'progress':
                    if self.loading_window:
                        payload = event['payload']
                        if isinstance(payload, dict):
                            current = payload.get('current', 0)
                            total = payload.get('total', 1)
                            eta = payload.get('eta', 0)
                            
                            progress_percent = (current / total) * 100 if total > 0 else 0
                            self.loading_progress_var.set(progress_percent)
                            
                            eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "Calculating..."
                            self.loading_label_var.set(f"Parsing... {current:,} / {total:,} events ({eta_str})")
                        else:
                            # Fallback for simple string messages
                            self.loading_label_var.set(str(payload))

                elif event['kind'] == 'success':
                    print("[DEBUG] check_parser_status: Received 'success' message!")
                    if self.loading_window:
                        self.loading_window.destroy()
                        self.loading_window = None

                    self.filename_var.set(f"Loaded: {os.path.basename(self.parsed_midi.filename)}")

                    self.time_var.set(f"00:00 / {self.format_time(self.total_song_duration)}")
                    self.note_count_label.config(text=f"Notes: 0 / {self.total_song_notes:,}")
                    self.play_pause_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.NORMAL)
                    self.seek_slider.config(state=tk.NORMAL, to=self.total_song_duration); self.seek_slider.set(0)
                    self.note_count_var.set(0); self.total_notes_var.set(f"{self.total_song_notes:,}") 
                    self.reset_graph_history()

                    # Enable piano roll button now that a MIDI is loaded
                    if PianoRoll:
                        self.piano_roll_button.config(state=tk.NORMAL)

                    # If piano roll was open before, relaunch it with the new MIDI data.
                    # The launch_piano_roll function handles sending the MIDI data internally.
                    if self.was_piano_roll_open_before_unload and self.last_piano_roll_res and PianoRoll:
                        self.launch_piano_roll(self.last_piano_roll_res[0], self.last_piano_roll_res[1])
                    
                    self.was_piano_roll_open_before_unload = False # Reset flag
                    
                    self.load_button.config(text="Unload MIDI", command=self.unload_file, state=tk.NORMAL)
                    return 
                
                else:
                    if self.loading_window:
                        self.loading_window.destroy()
                        self.loading_window = None
                    messagebox.showerror("Parse Error", f"Could not load MIDI file: {event['payload']}")
                    self.filename_var.set("Failed to load file.")
                    self.load_button.config(state=tk.NORMAL)
                    return 

            self.root.after(100, self.check_parser_status)
        except Exception as e:
            if self.loading_window:
                self.loading_window.destroy()
                self.loading_window = None
            messagebox.showerror("Error", f"Error checking parser status: {e}")
            self.filename_var.set("Error during parsing."); self.load_button.config(state=tk.NORMAL)
            self.controller.clear_parse_job()

    def on_seek_press(self, event):
        if not self.parsed_midi: return
        paused_for_seek = self.controller.begin_seek()
        if paused_for_seek:
            self.filename_var.set("Seeking...") # Give user feedback

    def on_seek_release(self, event):
        if not self.parsed_midi: return
        
        seek_time = self.seek_slider.get()
        self.panic_all_notes_off()

        with self.playback_lock:
            resumed = self.controller.complete_seek(seek_time)

        if resumed:
            self.play_pause_button.config(text="Pause")
            self.filename_var.set("Playing...")

        if not self.playing or self.paused:
            self.time_var.set(f"{self.format_time(seek_time)} / {self.format_time(self.total_song_duration)}")

    def toggle_play_pause(self):
        if self.playing:
            if self.paused:
                self.controller.resume_playback()
                self.play_pause_button.config(text="Pause"); self.filename_var.set("Playing...")
            else:
                self.controller.pause_playback()
                self.play_pause_button.config(text="Resume"); self.filename_var.set("Paused")
        else:
            if self.parsed_midi is None: return
            current_time = self.get_current_playback_time() 
            self.controller.start_playback(current_time)
            self.play_pause_button.config(text="Pause"); self.filename_var.set("Playing...")
            
            # Disable voice limit slider during playback
            if hasattr(self, 'voice_limit_slider'):
                self.voice_limit_slider.config(state=tk.DISABLED)

            self.playback_thread = threading.Thread(target=self.play_music_thread, daemon=True)
            self.playback_thread.start()

    def stop_playback(self):
        self.controller.stop_playback()
        self.reset_playback_state()
        self.play_pause_button.config(text="Play")
        
        if self.parsed_midi:
            self.time_var.set(f"00:00 / {self.format_time(self.total_song_duration)}")
            self.filename_var.set(f"Loaded: {os.path.basename(self.parsed_midi.filename)}")

        # Enable voice limit slider
        if hasattr(self, 'voice_limit_slider'):
            self.voice_limit_slider.config(state=tk.NORMAL)

    def unload_file(self):
        if self.playing:
            self.controller.playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(0.1)

        self.controller.unload_midi()
        self.reset_playback_state()
        self.filename_var.set("No file loaded.")
        self.time_var.set("00:00 / 00:00")
        self.note_count_label.config(text="Notes: 0 / 0")
        self.seek_slider.config(state=tk.DISABLED, to=100)
        self.seek_slider.set(0)
        
        self.play_pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)

        if PianoRoll:
            self.piano_roll_button.config(state=tk.DISABLED)

        # Close piano roll if it's open, and remember that it was open
        if self.piano_roll and self.piano_roll.app_running.is_set():
            self.was_piano_roll_open_before_unload = True
            self.piano_roll.app_running.clear()
            if self.piano_roll_thread and self.piano_roll_thread.is_alive():
                self.piano_roll_thread.join(0.2)
            self.piano_roll = None
        else:
            self.was_piano_roll_open_before_unload = False

        self.load_button.config(text="Load MIDI", command=self.load_file)

    def reset_playback_state(self):
        self.controller.reset_playback_state()
        if hasattr(self, 'seek_slider'): self.seek_slider.set(0)
        self.note_count_var.set(0)
        self.reset_graph_history(); self.panic_all_notes_off()

    def playback_finished(self):
        self.controller.finish_playback()
        self.play_pause_button.config(text="Play")
        if self.parsed_midi:
            self.filename_var.set(f"Finished: {os.path.basename(self.parsed_midi.filename)}")
            self.time_var.set(f"{self.format_time(self.total_song_duration)} / {self.format_time(self.total_song_duration)}")
            self.seek_slider.set(self.total_song_duration)
        
        # Enable voice limit slider
        if hasattr(self, 'voice_limit_slider'):
            self.voice_limit_slider.config(state=tk.NORMAL)
            
        self.panic_all_notes_off()
        
    def panic_all_notes_off(self):
        if self.active_midi_backend is None: return 
        try:
            self.controller.panic_all_notes_off()
        except Exception as e:
            print(f"Error during MIDI backend panic: {e}")

    def set_pitch_bend_range(self, semitones=12):
        self.controller.set_pitch_bend_range(semitones=semitones)

    def play_music_thread(self):
        if self.active_midi_backend and getattr(self.active_midi_backend, 'buffering_enabled', False):
            self.play_music_thread_buffered()
        else:
            self.play_music_thread_realtime()

    def play_music_thread_buffered(self):
        try:
            if not self.parsed_midi:
                self.root.after(0, self.playback_finished)
                return

            print("Preparing buffered playback stream...")
            self.root.after(0, lambda: self.filename_var.set("Pre-rendering events..."))
            
            if self.active_midi_backend:
                self.set_pitch_bend_range(semitones=12)
                self.active_midi_backend.stop()

            note_events = self.parsed_midi.note_events_for_playback
            pitch_bend_events = self.parsed_midi.pitch_bend_events
            
            count_notes = len(note_events)
            count_bends = len(pitch_bend_events)
            total_ops = (count_notes * 2) + count_bends
            
            print(f"Flattening {total_ops} events...")
            
            times = np.empty(total_ops, dtype=np.float64)
            statuses = np.empty(total_ops, dtype=np.uint32)
            params = np.empty(total_ops, dtype=np.uint32)
            
            times[:count_notes] = note_events['on_time']
            statuses[:count_notes] = 0x90 + note_events['channel']
            params[:count_notes] = (note_events['velocity'].astype(np.uint32) << 8) | note_events['pitch'].astype(np.uint32)
            
            times[count_notes:count_notes*2] = note_events['off_time']
            statuses[count_notes:count_notes*2] = 0x80 + note_events['channel']
            params[count_notes:count_notes*2] = note_events['pitch'].astype(np.uint32)
            
            if count_bends > 0:
                pb_arr = np.array(pitch_bend_events, dtype=[('time', 'f8'), ('chan', 'u4'), ('val', 'u4')])
                
                start_idx = count_notes * 2
                times[start_idx:] = pb_arr['time']
                statuses[start_idx:] = 0xE0 + pb_arr['chan']
                bend_lsb = pb_arr['val'] & 0x7F
                bend_msb = (pb_arr['val'] >> 7) & 0x7F
                params[start_idx:] = (bend_msb << 8) | bend_lsb

            print("Sorting events...")
            sort_indices = np.argsort(times)
            times = times[sort_indices]
            statuses = statuses[sort_indices]
            params = params[sort_indices]
            
            print("Uploading to Engine...")
            
            is_note_on = (statuses & 0xF0) == 0x90
            if np.any(is_note_on):
                note_on_params = params[is_note_on]
                velocities = (note_on_params >> 8) & 0xFF
                max_vel_count = np.sum(velocities == 127)
                avg_vel = np.mean(velocities)
                print(f"[DEBUG] Velocity Stats: Avg={avg_vel:.1f}, Max(127) Count={max_vel_count}/{len(velocities)}")
                if max_vel_count > len(velocities) * 0.5:
                    print("[WARNING] Over 50% of notes have velocity 127! This indicates a parsing or logic error.")
            self.active_midi_backend.upload_events(times, statuses, params)
            
            start_time = self.get_current_playback_time()
            self.active_midi_backend.set_current_time(start_time)
            
            has_started_playback = False
            
            while self.playing:
                while self.paused:
                    if not self.playing: break
                    time.sleep(0.02)
                if not self.playing: break

                requested_time = None
                with self.playback_lock:
                    if self.seek_request_time is not None:
                        requested_time = self.seek_request_time; self.seek_request_time = None
                
                if requested_time is not None:
                    print(f"Seek -> {requested_time:.2f}s")
                    self.active_midi_backend.stop()
                    self.active_midi_backend.set_current_time(requested_time)
                    self.buffered_playback_start_offset = requested_time
                    has_started_playback = False
                
                buffer_lvl = self.active_midi_backend.fill_buffer(60.0)
                
                is_active = self.active_midi_backend.is_active()
                
                if not has_started_playback:
                    if buffer_lvl > 4.0:
                        self.root.after(0, lambda: self.filename_var.set("Playing..."))
                        self.active_midi_backend.play()
                        has_started_playback = True
                    else:
                         self.root.after(0, lambda lvl=buffer_lvl: self.filename_var.set(f"Prerendering... {lvl:.1f}s / 4.0s"))
                
                elif has_started_playback:
                    if buffer_lvl < 0.2: # Underrun
                         print(f"Buffer Underrun ({buffer_lvl:.2f}s).")
                         self.active_midi_backend.pause()
                         self.root.after(0, lambda: self.filename_var.set("Buffering..."))
                         has_started_playback = False 
                    elif not is_active and buffer_lvl > 2.0:
                         self.active_midi_backend.play()
                         self.root.after(0, lambda: self.filename_var.set("Playing..."))

                if buffer_lvl >= 60.0:
                    time.sleep(0.1)
                else:
                    time.sleep(0.005)

        except Exception as e:
            print(f"Buffered playback error: {e}")
            traceback.print_exc()
        finally:
            print("Buffered playback finished.")
            self.root.after(0, self.playback_finished)

    def play_music_thread_realtime(self):
        try:
            if not self.parsed_midi:
                self.root.after(0, self.playback_finished)
                return

            print("Preparing playback stream...")
            self.root.after(0, lambda: self.filename_var.set("Starting playback..."))
            
            # --- Set a wide pitch bend range for all channels ---
            if self.active_midi_backend:
                self.set_pitch_bend_range(semitones=12)
            
            note_events = self.parsed_midi.note_events_for_playback
            pitch_bend_events = self.parsed_midi.pitch_bend_events
            
            num_note_events = len(note_events)
            num_pitch_bend_events = len(pitch_bend_events)
            
            # --- Reset pitch bend for all channels ---
            if self.active_midi_backend:
                for channel in range(16):
                    status = 0xE0 + channel
                    param = (0x40 << 8) | 0x00
                    self.active_midi_backend.send_raw_event(status, param)
            
            start_time = self.get_current_playback_time()
            
            note_event_index = bisect.bisect_left(note_events['on_time'], start_time)
            
            # Search for a tuple, not a float
            pitch_bend_index = bisect.bisect_left(pitch_bend_events, (start_time, -float('inf'), -float('inf')))

            with self.playback_lock:
                self.last_processed_event_time = start_time

            note_off_heap = []
            if note_event_index > 0:
                print(f"Rebuilding active notes state for seek... (up to index {note_event_index})")
                notes_before_now = note_events[:note_event_index]
                active_notes = notes_before_now[notes_before_now['off_time'] > start_time]
                for note in active_notes:
                    heapq.heappush(note_off_heap, 
                                   (note['off_time'], note['pitch'], note['channel']))
                print(f"Seek rebuild complete, {len(active_notes)} notes active.")


            # --- Main Playback Loop ---
            while self.playing and (note_event_index < num_note_events or 
                                    pitch_bend_index < num_pitch_bend_events or 
                                    len(note_off_heap) > 0):
                
                while self.paused:
                    if not self.playing: break
                    time.sleep(0.01)
                if not self.playing: break 

                # --- Handle Seeking ---
                requested_time = None
                with self.playback_lock:
                    if self.seek_request_time is not None:
                        requested_time = self.seek_request_time; self.seek_request_time = None
                
                if requested_time is not None:
                    print(f"Seek requested to {requested_time:.2f}s")
                    with self.playback_lock:
                        self.last_processed_event_time = requested_time

                    note_event_index = bisect.bisect_left(note_events['on_time'], requested_time)
                    pitch_bend_index = bisect.bisect_left(pitch_bend_events, (requested_time, -float('inf'), -float('inf')))
                    
                    self.playback_start_time = time.monotonic() - requested_time
                    self.total_paused_duration = 0.0; self.paused_at_time = 0.0
                    
                    self.notes_played_count = note_event_index
                    self.nps_event_timestamps.clear()

                    note_off_heap.clear()
                    if note_event_index > 0:
                        notes_before_now = note_events[:note_event_index]
                        active_notes = notes_before_now[notes_before_now['off_time'] > requested_time]
                        for note in active_notes:
                            heapq.heappush(note_off_heap, 
                                           (note['off_time'], note['pitch'], note['channel']))
                    print(f"Seek complete. Index: {note_event_index}, Active notes: {len(note_off_heap)}")
                
                # --- Find the next event time ---
                next_note_on_time = note_events[note_event_index]['on_time'] if note_event_index < num_note_events else float('inf')
                next_note_off_time = note_off_heap[0][0] if note_off_heap else float('inf')
                next_pitch_bend_time = pitch_bend_events[pitch_bend_index][0] if pitch_bend_index < num_pitch_bend_events else float('inf')

                event_time_sec = min(next_note_on_time, next_note_off_time, next_pitch_bend_time)
                
                if event_time_sec == float('inf'):
                    break # No more events, end of song

                with self.playback_lock:
                    self.last_processed_event_time = event_time_sec

                # --- Sleep until the event ---
                target_wall_time = self.playback_start_time + event_time_sec + self.total_paused_duration
                sleep_duration = target_wall_time - time.monotonic()

                self.current_lag = max(0, -sleep_duration)

                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                
                if not self.playing: break
                if self.paused: continue
                with self.playback_lock:
                    if self.seek_request_time is not None:
                        continue 
                
                try:
                    # 1. Process Note-Offs
                    while note_off_heap and note_off_heap[0][0] <= event_time_sec:
                        off_time, pitch, channel = heapq.heappop(note_off_heap)
                        status = 0x80 + channel
                        if self.active_midi_backend:
                            self.active_midi_backend.send_raw_event(status, pitch)

                    # 2. Process Note-Ons
                    while note_event_index < num_note_events and note_events[note_event_index]['on_time'] <= event_time_sec:
                        note = note_events[note_event_index]
                        note_event_index += 1
                        
                        pitch, vel, channel = int(note['pitch']), int(note['velocity']), int(note['channel'])
                        
                        self.notes_played_count += 1
                        self.nps_event_timestamps.append(note['on_time'])

                        if vel >= 20: # Restored velocity check
                            status_on = 0x90 + channel
                            if self.active_midi_backend:
                                param = (vel << 8) | pitch
                                self.active_midi_backend.send_raw_event(status_on, param)
                            
                            heapq.heappush(note_off_heap, (note['off_time'], pitch, channel))

                    # 3. Process Pitch Bends
                    while pitch_bend_index < num_pitch_bend_events and pitch_bend_events[pitch_bend_index][0] <= event_time_sec:
                        _time, channel, pitch_value = pitch_bend_events[pitch_bend_index]
                        status = 0xE0 + channel
                        data1 = pitch_value & 0x7F  # LSB
                        data2 = (pitch_value >> 7) & 0x7F # MSB
                        
                        if self.active_midi_backend:
                            param = (data2 << 8) | data1
                            self.active_midi_backend.send_raw_event(status, param)
                            
                        pitch_bend_index += 1

                except Exception as e:
                    print(f"MIDI backend send error: {e}")
                    self.root.after(0, lambda e=e: self.filename_var.set(f"Playback Error: {e}")); break 
                
        except Exception as e:
            print(f"Playback thread error: {e}")
            traceback.print_exc()
        finally:
            print("Playback thread finished."); self.root.after(0, self.playback_finished)

    def get_current_playback_time_thread_safe(self):
        return self.current_playback_time_for_threads
            
    def get_current_playback_time(self):
        current_time = self.controller.get_current_playback_time()
        if not self.playing and hasattr(self, 'seek_slider'):
            return self.seek_slider.get()
        return current_time

    def update_cpu_graph(self):
        cpu_percent = 0
        if self.process:
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
                # Also include CPU usage of child processes (like the parser)
                for child in self.process.children(recursive=True):
                    try:
                        cpu_percent += child.cpu_percent(interval=None)
                    except psutil.NoSuchProcess:
                        # Child process might have terminated since the list was created
                        continue
                
                self.cpu_history.append(cpu_percent)
                self.cpu_var.set(f"CPU: {cpu_percent:.1f}%")
            except Exception:
                self.cpu_history.append(0)
        
        if cpu_percent > 100:
            self.vbo_lag_label.config(text="OVERLOADED!!", fg='red')
            self.vbo_lag_label.pack(side=tk.LEFT, padx=5)
        else:
            self.vbo_lag_label.pack_forget()

        self.draw_graph(self.cpu_canvas, self.cpu_bars, self.cpu_history, 100, 'cpu')
        self.root.after(1000, self.update_cpu_graph)
    
    def update_gui_counters(self):
        now = time.monotonic()
        current_max_nps = 0; nps = 0
        current_time = self.get_current_playback_time()
        self.current_playback_time_for_threads = current_time
        
        if self.playing and not self.paused:
            if current_time < 0: current_time = 0.0
            if self.parsed_midi and current_time > self.total_song_duration:
                current_time = self.total_song_duration
            if self.parsed_midi:
                self.time_var.set(f"{self.format_time(current_time)} / {self.format_time(self.total_song_duration)}")
            if not self.is_seeking: self.seek_slider.set(current_time)
            
            # accurate NPS and Note Count using bisect
            if self.parsed_midi:
                on_times = self.parsed_midi.note_events_for_playback['on_time']
                
                # Notes played so far (heard)
                played_idx = bisect.bisect_left(on_times, current_time)
                self.note_count_label.config(text=f"Notes: {played_idx:,} / {self.total_song_notes:,}")
                
                # NPS (notes in last second)
                start_nps_time = max(0, current_time - 1.0)
                start_idx = bisect.bisect_left(on_times, start_nps_time)
                nps = played_idx - start_idx
                
                self.nps_history.append(nps); self.nps_var.set(f"NPS: {self.format_nps(nps)}")
        
        elif self.parsed_midi and not self.playing:
            nps = 0; self.nps_history.append(nps); self.nps_var.set("NPS: 0")
            
        current_max_nps = max(self.nps_history) if self.nps_history else 0
        graph_top_value = max(100, (math.ceil(current_max_nps * 1.15 / 50) * 50))
        self.nps_max_var.set(f"Max: {self.format_nps(graph_top_value)}")
        is_ultra = self.ultra_mode_var.get()
        if is_ultra or (now - self.last_nps_graph_update_time > 0.1): # 10Hz
            self.draw_graph(self.nps_canvas, self.nps_bars, self.nps_history, graph_top_value, 'nps')
            if not is_ultra: self.last_nps_graph_update_time = now

        if now - self.last_lag_update_time > 0.5: # Update every 0.5s
            delta_time = now - self.last_lag_update_time
            self.last_lag_update_time = now
            
            # Check mode
            is_buffered = False
            if self.active_midi_backend and hasattr(self.active_midi_backend, 'buffering_enabled'):
                is_buffered = self.active_midi_backend.buffering_enabled
            
            if is_buffered:
                # Show Buffer Level
                try:
                    buf_lvl = self.active_midi_backend.get_buffer_level()
                    self.slowdown_var.set(f"Buffer: {buf_lvl:.1f}s")
                    if buf_lvl > 4.0:
                        self.slowdown_label.config(fg='green')
                    elif buf_lvl > 2.0:
                        self.slowdown_label.config(fg='#AAaa00') # Dark Yellow
                    elif buf_lvl > 0.5:
                        self.slowdown_label.config(fg='orange')
                    else:
                        self.slowdown_label.config(fg='red')
                except:
                    self.slowdown_var.set("Buffer: N/A")
            else:
                # Show Slowdown (Realtime)
                if self.playing and not self.paused:
                    delta_lag = self.current_lag - self.last_lag_value
                    if delta_time > 0:
                        slowdown = delta_lag / delta_time
                        self.slowdown_percentage = max(0, slowdown * 100)
                else:
                    self.slowdown_percentage = 0.0

                self.last_lag_update_time = now
                self.last_lag_value = self.current_lag

                self.slowdown_var.set(f"Slowdown: {self.slowdown_percentage:.1f}%")
                if self.slowdown_percentage > 10: 
                    self.slowdown_label.config(fg='red')
                else:
                    self.slowdown_label.config(fg='black')
        
        self.root.after(16, self.update_gui_counters)

    def cleanup(self):
        print("Cleaning up resources...")
        # Stop playback
        if self.playing:
            self.controller.playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(0.1) # Brief wait for thread to exit

        # Cleanup piano roll process
        if self.piano_roll:
            # This should handle the pygame window closing
            self.piano_roll.app_running.clear() 
            if self.piano_roll_thread and self.piano_roll_thread.is_alive():
                 self.piano_roll_thread.join(0.2)
        
        print("Shutting down controller resources...")
        self.controller.shutdown()


# --- Main execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk() 
    root.geometry("550x580")
    app = MidiPlayerApp(root)
    
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            app.cleanup(); root.destroy(); sys.exit()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, cleaning up...")
        app.cleanup()
        sys.exit()
