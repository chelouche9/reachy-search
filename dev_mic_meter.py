#!/usr/bin/env python3
"""Live mic level meter through the same audio stack the app uses.

Run it (daemon must be running), talk, and watch whether your speech crosses
the gate. If it never does, lower the gate:

    REACHY_SEARCH_SILENCE_RMS=0.001 .venv/bin/python -m reachy_search.main
"""

import time

import numpy as np
from reachy_mini import ReachyMini

from reachy_search import audio_utils, config

with ReachyMini() as mini:
    media = mini.media
    media.start_recording()
    rate = media.get_input_audio_samplerate()
    print(f"gate = {config.SILENCE_RMS}   (Ctrl-C to quit — now talk)")
    buf = []
    try:
        while True:
            chunk = media.get_audio_sample()
            if chunk is None:
                time.sleep(0.01); continue
            buf.append(audio_utils.to_mono(chunk))
            if sum(len(c) for c in buf) >= rate // 4:
                block = np.concatenate(buf); buf = []
                level = audio_utils.rms(block)
                bar = "#" * min(60, int(level / config.SILENCE_RMS * 10))
                mark = "  <-- SPEECH" if level > config.SILENCE_RMS else ""
                print(f"{level:.5f} |{bar:<60}|{mark}")
    except KeyboardInterrupt:
        print()
    finally:
        media.stop_recording()
