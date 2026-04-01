#!/usr/bin/env python3

import ctypes
import multiprocessing
import os
import sys

MidiPlayerApp = None


def _should_enable_log_console():
    if os.name != "nt":
        return False
    if not any(arg.lower() == "-log" for arg in sys.argv[1:]):
        return False
    if any(arg.startswith("--multiprocessing-fork") for arg in sys.argv[1:]):
        return False
    return True


def _enable_log_console():
    kernel32 = ctypes.windll.kernel32
    attached = bool(kernel32.AttachConsole(ctypes.c_uint(-1).value))
    if not attached:
        if not kernel32.AllocConsole():
            return

    try:
        kernel32.SetConsoleTitleW("LWMP Log Console")
    except Exception:
        pass

    try:
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except OSError:
        pass

    try:
        sys.stdin = open("CONIN$", "r", encoding="utf-8", buffering=1)
    except OSError:
        pass

    print("[LWMP] Log console enabled via -log")


def main():
    multiprocessing.freeze_support()
    if _should_enable_log_console():
        _enable_log_console()
        sys.argv = [sys.argv[0], *[arg for arg in sys.argv[1:] if arg.lower() != "-log"]]
    from midiplayer_dpg import DpgMidiPlayerApp as _DpgMidiPlayerApp
    global MidiPlayerApp
    MidiPlayerApp = _DpgMidiPlayerApp
    app = MidiPlayerApp()
    app.run()


if __name__ == "__main__":
    main()
