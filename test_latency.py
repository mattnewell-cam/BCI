"""
Test latency: measure broadband EEG power and emit a sound when it exceeds a threshold.

Designed to measure the end-to-end latency from neural event to audible feedback.
Uses sounddevice for minimal-latency audio output.
"""

import time
from collections import deque

import numpy as np
import sounddevice as sd
from pylsl import StreamInlet
from scipy.signal import welch

from utils import find_eeg_stream

# --- Config ---
POWER_BAND = (1.0, 40.0)       # Hz range for broadband power
WINDOW_S = 0.5                  # analysis window in seconds
HOP_S = 0.02                    # how often to check (seconds)
THRESHOLD = 1000.0                # power threshold (tune to your signal)
COOLDOWN_S = 0.5                # minimum gap between beeps

# Beep params
BEEP_FREQ = 880                 # Hz
BEEP_DURATION = 0.03            # seconds
BEEP_SAMPLERATE = 44100


def make_beep(freq, duration, sr):
    t = np.arange(int(sr * duration)) / sr
    tone = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    # fade in/out to avoid click
    fade = min(64, len(tone) // 4)
    tone[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    tone[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return tone


def main():
    stream_info = find_eeg_stream()
    inlet = StreamInlet(stream_info, max_buflen=2)

    info = inlet.info()
    fs = int(info.nominal_srate())
    n_ch = info.channel_count()
    n_eeg = min(4, n_ch)
    print(f"Connected: fs={fs} Hz, channels={n_ch}, using first {n_eeg}")

    buf_len = int(fs * WINDOW_S)
    buffers = [deque(maxlen=buf_len) for _ in range(n_eeg)]

    beep_tone = make_beep(BEEP_FREQ, BEEP_DURATION, BEEP_SAMPLERATE)

    # Pre-open a persistent output stream so we don't pay device-open cost per beep
    audio_stream = sd.OutputStream(
        samplerate=BEEP_SAMPLERATE, channels=1, dtype="float32",
        blocksize=len(beep_tone),
    )
    audio_stream.start()

    last_check = time.time()
    last_beep = 0.0

    print(f"Threshold: {THRESHOLD}  Window: {WINDOW_S}s  Cooldown: {COOLDOWN_S}s")
    print("Listening...\n")

    while True:
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=int(fs * HOP_S))
        if chunk:
            chunk = np.asarray(chunk, dtype=np.float64)
            for c in range(n_eeg):
                buffers[c].extend(chunk[:, c])

        now = time.time()
        if now - last_check < HOP_S:
            time.sleep(0.001)
            continue
        last_check = now

        if any(len(b) < buf_len for b in buffers):
            continue

        # Compute broadband power averaged across channels
        powers = []
        for b in buffers:
            x = np.asarray(b, dtype=np.float64)
            x = x - x.mean()
            freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(fs)))
            mask = (freqs >= POWER_BAND[0]) & (freqs <= POWER_BAND[1])
            powers.append(np.trapezoid(psd[mask], freqs[mask]))

        avg_power = np.mean(powers)

        if avg_power > THRESHOLD and (now - last_beep) > COOLDOWN_S:
            t_detect = time.perf_counter()
            print(f"THRESHOLD EXCEEDED  power={avg_power:.2f}  (t={now:.3f})")
            audio_stream.write(beep_tone.reshape(-1, 1))
            t_after = time.perf_counter()
            print(f"  sound queued in {(t_after - t_detect)*1000:.1f} ms")
            last_beep = now


if __name__ == "__main__":
    main()
