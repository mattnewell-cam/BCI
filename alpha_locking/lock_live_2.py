"""
Lock Live 2 - Trough-tracking alpha phase prediction with audio feedback.

Every 100ms, pulls the last 1000ms of raw EEG, bandpass-filters it in one
shot, finds the last 3 troughs, averages the two inter-trough gaps to get
the period, and projects forward to schedule a beep at the 2nd predicted
trough.

Includes per-step timing diagnostics printed every 10th round.

Usage:
    python lock_live_2.py
"""

import sys
import os
import time
import threading
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, argrelmin
from pylsl import StreamInlet

from utils import find_eeg_stream

CHANNEL_LABELS = ["AF7", "AF8", "TP9", "TP10"]

# --- Filter parameters ---
BAND_LO = 8.3
BAND_HI = 10.3
FILTER_ORDER = 4

# --- Timing ---
FIT_INTERVAL_S = 0.100   # refit every 100ms
LOOKBACK_S = 1.000       # bandpass this window each round
BEEP_FREQ_HZ = 880
BEEP_MS = 15

# --- Audio (sounddevice) ---
AUDIO_FS = 48000
BEEP_SAMPLES = int(AUDIO_FS * BEEP_MS / 1000)
BEEP_AMP = 0.2

# Precompute one beep waveform
_t = np.arange(BEEP_SAMPLES) / AUDIO_FS
BEEP_WAV = (BEEP_AMP * np.sin(2 * np.pi * BEEP_FREQ_HZ * _t)).astype(np.float32)

_events = deque()
_events_lock = threading.Lock()
_stream_sample = 0


def schedule_beep(delay_s=0.0):
    """Schedule a beep to play delay_s seconds from now in the audio stream."""
    global _stream_sample
    with _events_lock:
        start = _stream_sample + int(delay_s * AUDIO_FS)
        _events.append(start)


def _audio_callback(outdata, frames, time_info, status):
    global _stream_sample
    out = np.zeros(frames, np.float32)

    with _events_lock:
        block_start = _stream_sample
        block_end = _stream_sample + frames

        # drop events that are already finished
        while _events and _events[0] + BEEP_SAMPLES <= block_start:
            _events.popleft()

        for start in list(_events):
            end = start + BEEP_SAMPLES
            if end <= block_start or start >= block_end:
                continue
            # overlap region
            a0 = max(0, block_start - start)
            a1 = min(BEEP_SAMPLES, block_end - start)
            o0 = max(0, start - block_start)
            o1 = o0 + (a1 - a0)
            out[o0:o1] += BEEP_WAV[a0:a1]

    outdata[:, 0] = out
    _stream_sample += frames


def main():
    # Connect to EEG stream
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=2)

    info = inlet.info()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()
    n_eeg = min(4, n_channels)
    print(f"Connected: fs={fs} Hz, channels={n_eeg}, name={info.name()}")

    # --- Bandpass filter (designed once, applied fresh each round) ---
    sos = butter(FILTER_ORDER, [BAND_LO, BAND_HI], btype="band",
                 fs=fs, output="sos")

    # Ring buffer for raw samples (per channel)
    buf_len = int(3.0 * fs)  # 3s is plenty
    raw_bufs = [deque(maxlen=buf_len) for _ in range(n_eeg)]

    # --- Channel selection ---
    best_ch = 3 if n_eeg > 3 else 0

    # --- Timing ---
    sample_count = 0
    t0_wall = time.time()
    last_fit_time = 0.0
    lookback_samples = int(LOOKBACK_S * fs)
    fit_round = 0

    last_fit_wall = None  # perf_counter at last fit, for measuring true interval

    print(f"Filter: {BAND_LO}-{BAND_HI} Hz, interval: {FIT_INTERVAL_S*1000:.0f}ms")
    print(f"Buffering {LOOKBACK_S}s before first fit...")

    # Open persistent low-latency audio stream for beeps
    audio_stream = sd.OutputStream(
        samplerate=AUDIO_FS, channels=1, callback=_audio_callback,
        dtype="float32", latency="low",
    )
    audio_stream.start()
    schedule_beep(0.0)  # test beep
    print("Audio stream started (sounddevice)")

    while True:
        # --- Pull all available samples ---
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=256)
        if chunk:
            chunk_arr = np.array(chunk)
            for ch in range(n_eeg):
                raw_bufs[ch].extend(chunk_arr[:, ch])
            sample_count += len(chunk)

        # --- Check if time to fit ---
        now_wall = time.time()
        elapsed = now_wall - t0_wall

        if sample_count < lookback_samples:
            continue

        time_since_fit = elapsed - last_fit_time
        if time_since_fit < FIT_INTERVAL_S:
            continue

        fit_round += 1
        now_perf = time.perf_counter()
        wall_interval = (now_perf - last_fit_wall) * 1000 if last_fit_wall else 0.0
        last_fit_wall = now_perf
        last_fit_time = elapsed

        # --- Grab last 1000ms of raw signal and bandpass in one shot ---
        t_filt_start = time.perf_counter()
        raw_arr = np.array(raw_bufs[best_ch])
        window = raw_arr[-lookback_samples:]
        filtered = sosfilt(sos, window)
        t_filt = time.perf_counter() - t_filt_start

        # --- Find troughs in the filtered signal ---
        t_trough_start = time.perf_counter()
        # argrelmin with order ~ half-cycle at lower band edge
        half_cycle_samples = max(3, int(fs / BAND_HI / 2))
        trough_indices = argrelmin(filtered, order=half_cycle_samples)[0]

        if len(trough_indices) < 3:
            if fit_round % 10 == 0:
                print(f"[{fit_round}] only {len(trough_indices)} troughs found, skipping")
            continue

        # Last 3 trough positions (in samples from start of window)
        last3 = trough_indices[-3:]
        gap1 = last3[1] - last3[0]
        gap2 = last3[2] - last3[1]
        avg_gap = (gap1 + gap2) / 2.0
        t_trough = time.perf_counter() - t_trough_start

        # --- Project forward from the last trough ---
        t_sched_start = time.perf_counter()
        last_trough_sample = last3[2]
        # How far in the past is the last trough from "now" (end of window)?
        samples_since_last_trough = lookback_samples - last_trough_sample
        seconds_since_last_trough = samples_since_last_trough / fs

        # Next trough: one avg_gap after the last trough
        next1_delay = (avg_gap / fs) - seconds_since_last_trough
        # Second trough: two avg_gaps after the last trough
        next2_delay = (2 * avg_gap / fs) - seconds_since_last_trough

        # Schedule beep at the 2nd predicted trough
        if next2_delay > 0.005:
            schedule_beep(next2_delay)
        t_sched = time.perf_counter() - t_sched_start

        # --- Console output (every 10th round) ---
        if fit_round % 10 == 0:
            freq_est = fs / avg_gap
            print(f"[{fit_round}] freq={freq_est:.1f}Hz  "
                  f"gaps={gap1},{gap2}  avg={avg_gap:.1f}  "
                  f"delay={next2_delay*1000:.0f}ms  |  "
                  f"filt={t_filt*1000:.1f}ms  "
                  f"trough={t_trough*1000:.1f}ms  "
                  f"sched={t_sched*1000:.1f}ms  "
                  f"total={((t_filt+t_trough+t_sched)*1000):.1f}ms  "
                  f"interval={wall_interval:.0f}ms")


if __name__ == "__main__":
    main()
