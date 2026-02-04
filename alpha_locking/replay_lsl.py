"""
LSL replay script for interactive testing of lock_live_2.

Loads npz recording data and pushes it over LSL as a fake Muse EEG stream
at real-time speed (256 Hz, chunks of 8 samples every 32ms).

Usage:
    Terminal 1:  python alpha_locking/replay_lsl.py
    Terminal 2:  python alpha_locking/lock_live_2.py

Requires: pylsl
Audio playback in lock_live_2 requires: libportaudio2
    sudo apt install libportaudio2
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pylsl import StreamInfo, StreamOutlet

# --- Configuration ---
RECORDING = os.path.join(os.path.dirname(__file__), "..",
                         "recordings", "full_night_350_1000.npz")
FS = 256
N_CHANNELS = 4
CHUNK_SIZE = 8              # samples per push (~31ms at 256 Hz)
CHANNEL_NAMES = ["AF7", "AF8", "TP9", "TP10"]


def main():
    # Load recording
    if not os.path.exists(RECORDING):
        print(f"Recording not found: {RECORDING}")
        sys.exit(1)

    rec = np.load(RECORDING, allow_pickle=True)
    data = rec["data"]  # (channels, samples)
    sample_rate = int(rec["sample_rate"])
    assert sample_rate == FS, f"Expected {FS} Hz, got {sample_rate}"

    n_ch, n_samples = data.shape
    n_ch = min(n_ch, N_CHANNELS)
    duration_s = n_samples / FS
    print(f"Loaded: {n_samples} samples ({duration_s:.1f}s), {n_ch} channels")

    # Create LSL stream
    info = StreamInfo(
        name="Muse EEG Replay",
        type="EEG",
        channel_count=n_ch,
        nominal_srate=FS,
        channel_format="float32",
        source_id="replay_lsl_muse",
    )

    # Add channel metadata
    channels = info.desc().append_child("channels")
    for i in range(n_ch):
        ch = channels.append_child("channel")
        ch.append_child_value("label", CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"Ch{i}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")

    outlet = StreamOutlet(info, chunk_size=CHUNK_SIZE)
    print(f"LSL outlet created: '{info.name()}' ({n_ch}ch @ {FS}Hz)")
    print(f"Streaming {duration_s:.1f}s of data in chunks of {CHUNK_SIZE}...")
    print("Press Ctrl+C to stop.\n")

    chunk_interval = CHUNK_SIZE / FS  # ~0.03125s
    offset = 0
    t_start = time.perf_counter()

    try:
        while offset < n_samples:
            chunk_end = min(offset + CHUNK_SIZE, n_samples)
            # Transpose to (samples, channels) for LSL
            chunk = data[:n_ch, offset:chunk_end].T.astype(np.float32)

            outlet.push_chunk(chunk.tolist())
            offset = chunk_end

            # Pace to real-time
            elapsed = time.perf_counter() - t_start
            expected = offset / FS
            sleep_time = expected - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Progress
            if offset % (FS * 10) < CHUNK_SIZE:
                print(f"  {offset / FS:.0f}s / {duration_s:.0f}s")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"Done. Pushed {offset} samples ({offset / FS:.1f}s).")


if __name__ == "__main__":
    main()
