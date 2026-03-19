#!/usr/bin/env python3

import multiprocessing

from midiplayer_dpg import DpgMidiPlayerApp


# Keep a compatibility alias for any code that still imports the old frontend name.
MidiPlayerApp = DpgMidiPlayerApp


def main():
    multiprocessing.freeze_support()
    app = DpgMidiPlayerApp()
    app.run()


if __name__ == "__main__":
    main()
