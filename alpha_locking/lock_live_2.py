"""
Lock Live 2 - Real-time alpha trough prediction with OLS waveform projection.

On startup, trains an OLS model on full_night_350_1000 to predict the next
0.5s of bandpassed alpha waveform from the last 1s (subsampled to 64 points).
In real-time, detects troughs event-driven in the streaming bandpassed signal,
projects forward with the trained model, finds the 2nd predicted trough, and
schedules a beep at that time minus ASSUMED_LATENCY.

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
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi, argrelmin
from pylsl import StreamInlet

from utils import find_eeg_stream

# --- Filter parameters ---
BAND_LO = 7
BAND_HI = 12
FILTER_ORDER = 4

# --- Waveform model ---
WAVEFORM_S = 1.0            # seconds of filtered signal as input
WAVEFORM_SUBSAMPLE = 4      # 256 Hz -> 64 Hz -> 64 input points
WAVEFORM_OUT_NATIVE = 128   # 0.5s at 256 Hz -> 128 output points
CHANNEL_TP10 = 3

# --- Training ---
TRAIN_DATA = os.path.join(os.path.dirname(__file__), "..",
                          "recordings", "full_night_350_1000.npz")
LOOKBACK_S = 1.800           # lookback for trough gating during training
FIT_INTERVAL_S = 0.100       # cursor stride during training
LR_N_GAPS = 10               # min troughs needed in lookback (alpha-active gating)

# --- Live loop ---
ASSUMED_LATENCY_S = 0.120    # assumed audio+processing latency to compensate
MIN_DELAY_S = 0.005          # minimum delay to accept a prediction

# --- Audio ---
AUDIO_FS = 48000
BEEP_FREQ_HZ = 880
BEEP_MS = 15
BEEP_AMP = 0.2

BEEP_SAMPLES = int(AUDIO_FS * BEEP_MS / 1000)
_t = np.arange(BEEP_SAMPLES) / AUDIO_FS
BEEP_WAV = (BEEP_AMP * np.sin(2 * np.pi * BEEP_FREQ_HZ * _t)).astype(np.float32)

_events = deque()
_events_lock = threading.Lock()
_stream_sample = 0

# --- Diagnostics ---
DIAG_ROUNDS = {6, 8, 10}
DIAG_PATH = os.path.join(os.path.dirname(__file__), "..", "media",
                         "lock_live_2_diagnostic.png")

t0 = dt.datetime.now().timestamp()

def schedule_beep(delay_s=0.0):
    """Schedule a beep to play delay_s seconds from now in the audio stream."""
    global _stream_sample
    t = dt.datetime.now().timestamp() - t0
    print("beep", t)
    with _events_lock:
        start = _stream_sample + int(delay_s * AUDIO_FS)
        _events.append(start)


def _audio_callback(outdata, frames, time_info, status):
    global _stream_sample
    out = np.zeros(frames, np.float32)

    with _events_lock:
        block_start = _stream_sample
        block_end = _stream_sample + frames

        while _events and _events[0] + BEEP_SAMPLES <= block_start:
            _events.popleft()

        for start in list(_events):
            end = start + BEEP_SAMPLES
            if end <= block_start or start >= block_end:
                continue
            a0 = max(0, block_start - start)
            a1 = min(BEEP_SAMPLES, block_end - start)
            o0 = max(0, start - block_start)
            o1 = o0 + (a1 - a0)
            out[o0:o1] += BEEP_WAV[a0:a1]

    outdata[:, 0] = out
    _stream_sample += frames


def train_model(fs=256):
    """Load training data, bandpass filter, collect windows, fit OLS.

    Returns dict with keys:
        W            - weight matrix (65, 128)
        feat_mean    - (64,)
        feat_std     - (64,)
        out_mean     - scalar
        out_std      - scalar
        sos          - SOS filter coefficients
        half_cycle   - half-cycle in samples for trough detection
    """
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")

    rec = np.load(TRAIN_DATA, allow_pickle=True)
    data = rec["data"]
    train_fs = int(rec["sample_rate"])
    assert train_fs == fs, f"Expected {fs} Hz training data, got {train_fs}"

    signal = data[CHANNEL_TP10].astype(np.float64)
    total_samples = len(signal)
    print(f"  Training data: {total_samples} samples, {total_samples/fs:.1f}s")

    sos = butter(FILTER_ORDER, [BAND_LO, BAND_HI], btype="band",
                 fs=fs, output="sos")
    full_filtered = sosfilt(sos, signal)

    half_cycle = max(3, int(fs / BAND_HI / 2))
    lookback_samples = int(LOOKBACK_S * fs)
    fit_interval_samples = int(FIT_INTERVAL_S * fs)
    waveform_native = int(WAVEFORM_S * fs)
    n_troughs_needed = LR_N_GAPS + 1  # 11

    waveform_feats = []
    waveform_targets = []

    cursor = lookback_samples
    while cursor < total_samples:
        if cursor + WAVEFORM_OUT_NATIVE > total_samples:
            break

        # Gate: require enough troughs in lookback (alpha-active segments)
        window = signal[cursor - lookback_samples: cursor]
        filt_window = sosfilt(sos, window)
        troughs = argrelmin(filt_window, order=half_cycle)[0]

        if len(troughs) >= n_troughs_needed + 1:
            wave_in = full_filtered[cursor - waveform_native:cursor:WAVEFORM_SUBSAMPLE]
            waveform_feats.append(wave_in.astype(np.float64))

            wave_out = full_filtered[cursor:cursor + WAVEFORM_OUT_NATIVE]
            waveform_targets.append(wave_out.astype(np.float64))

        cursor += fit_interval_samples

    features = np.array(waveform_feats)
    targets = np.array(waveform_targets)
    n_rows = len(features)
    print(f"  Collected {n_rows} training windows "
          f"({features.shape[1]} in -> {targets.shape[1]} out)")

    # Z-score normalize
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    X = (features - feat_mean) / feat_std
    X = np.column_stack([X, np.ones(n_rows)])  # bias -> (N, 65)

    out_mean = targets.mean()
    out_std = targets.std()
    out_std = out_std if out_std > 1e-8 else 1.0
    Y_norm = (targets - out_mean) / out_std

    W = np.linalg.lstsq(X, Y_norm, rcond=None)[0]  # (65, 128)
    print(f"  OLS fit: W shape {W.shape}")

    return {
        "W": W,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "out_mean": out_mean,
        "out_std": out_std,
        "sos": sos,
        "half_cycle": half_cycle,
    }


def detect_trough(filt_arr, half_cycle, last_trough_abs, sample_count):
    """Check if a trough just became confirmable at the tail of filt_arr.

    The candidate is at position len(filt_arr) - 1 - half_cycle (we need
    half_cycle samples after it to confirm it's a local minimum).

    Returns (is_trough, abs_sample_pos).
    """
    n = len(filt_arr)
    candidate = n - 1 - half_cycle
    if candidate < half_cycle:
        return False, -1

    lo = candidate - half_cycle
    hi = candidate + half_cycle + 1
    window = filt_arr[lo:hi]
    center_val = filt_arr[candidate]

    if center_val <= window.min():
        abs_pos = sample_count - (n - candidate)
        if abs_pos - last_trough_abs >= half_cycle * 2:
            return True, abs_pos

    return False, -1


def _save_diagnostic(snapshots):
    """Save diagnostic figure showing input waveform, predicted waveform,
    predicted troughs, and beep target."""
    rounds = sorted(snapshots.keys())
    fig, axes = plt.subplots(len(rounds), 1, figsize=(14, 4 * len(rounds)),
                             sharex=False)
    if len(rounds) == 1:
        axes = [axes]

    for ax, rnd in zip(axes, rounds):
        snap = snapshots[rnd]
        fs = snap["fs"]

        # Input signal (last 1s) -> time axis from -1000ms to 0
        input_sig = snap["input_signal"]
        t_in = np.arange(len(input_sig)) / fs * 1000 - len(input_sig) / fs * 1000

        # Predicted waveform (next 0.5s) -> time axis from 0 to 500ms
        pred_sig = snap["predicted"]
        t_pred = np.arange(len(pred_sig)) / fs * 1000

        ax.plot(t_in, input_sig, color="steelblue", lw=0.8, label="input (1s)")
        ax.plot(t_pred, pred_sig, color="mediumpurple", lw=1.2,
                label="predicted (0.5s)")

        # Predicted troughs
        pred_troughs = snap["pred_troughs"]
        for i, ti in enumerate(pred_troughs):
            t_ms = ti / fs * 1000
            if i == 0:
                ax.axvline(t_ms, color="orange", ls="--", lw=1.2,
                           label="1st pred trough")
            elif i == 1:
                ax.axvline(t_ms, color="red", ls="--", lw=1.5,
                           label="2nd pred trough (target)")
            else:
                ax.axvline(t_ms, color="grey", ls=":", lw=0.8, alpha=0.5)

        # Beep target
        beep_ms = snap["beep_delay_ms"]
        if beep_ms is not None:
            ax.axvline(beep_ms, color="green", ls="-", lw=1.5, alpha=0.8,
                       label=f"beep @ {beep_ms:.0f}ms")

        ax.axvline(0, color="black", ls=":", lw=0.8, alpha=0.5, label="now")

        delay_str = f"{beep_ms:.0f}ms" if beep_ms is not None else "skipped"
        n_tr = len(pred_troughs)
        ax.set_title(f"Round {rnd}:  {n_tr} pred troughs,  "
                     f"beep delay={delay_str}",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("uV (filtered)")
        ax.legend(loc="upper left", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(DIAG_PATH), exist_ok=True)
    fig.savefig(DIAG_PATH, dpi=150)
    plt.close(fig)


def main():
    # ========== PHASE A: TRAIN MODEL ==========
    print("Training OLS model from recording data...")
    t0 = time.perf_counter()
    model = train_model()
    print(f"Training complete in {time.perf_counter() - t0:.1f}s")

    W = model["W"]
    feat_mean = model["feat_mean"]
    feat_std = model["feat_std"]
    out_mean = model["out_mean"]
    out_std = model["out_std"]
    sos = model["sos"]
    half_cycle = model["half_cycle"]

    # ========== PHASE B: CONNECT ==========
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=2)

    info = inlet.info()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()
    n_eeg = min(4, n_channels)
    print(f"Connected: fs={fs} Hz, channels={n_eeg}, name={info.name()}")
    assert fs == 256, f"Expected 256 Hz, got {fs}"

    best_ch = CHANNEL_TP10 if n_eeg > CHANNEL_TP10 else 0
    waveform_native = int(WAVEFORM_S * fs)  # 256 samples

    # Streaming bandpass filter state
    zi = sosfilt_zi(sos) * 0.0

    # Ring buffers
    filt_buf = deque(maxlen=int(2.0 * fs))  # 2s of filtered signal

    # Open persistent low-latency audio stream
    audio_stream = sd.OutputStream(
        samplerate=AUDIO_FS, channels=1, callback=_audio_callback,
        dtype="float32", latency="low",
    )
    audio_stream.start()
    schedule_beep(0.0)  # test beep
    print("Audio stream started")

    # Trough detection state
    last_trough_abs = -1000
    sample_count = 0
    prediction_count = 0

    # Diagnostic state
    diag_snapshots = {}

    print(f"Model: {BAND_LO}-{BAND_HI} Hz, "
          f"input={WAVEFORM_S}s@{fs // WAVEFORM_SUBSAMPLE}Hz, "
          f"output={WAVEFORM_OUT_NATIVE / fs:.2f}s@{fs}Hz, "
          f"latency comp={ASSUMED_LATENCY_S * 1000:.0f}ms")
    print(f"Buffering {WAVEFORM_S}s before predictions...")

    # ========== PHASE C: MAIN LOOP ==========
    while True:
        # --- Pull all available EEG samples ---
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=256)
        if not chunk:
            time.sleep(0.001)
            continue

        chunk_arr = np.array(chunk)
        new_raw = chunk_arr[:, best_ch].astype(np.float64)

        # --- Streaming bandpass filter ---
        new_filtered, zi = sosfilt(sos, new_raw, zi=zi)

        # --- Append to ring buffer ---
        filt_buf.extend(new_filtered)
        sample_count += len(new_raw)

        # --- Need enough samples ---
        if len(filt_buf) < waveform_native + half_cycle:
            continue

        # --- Trough detection ---
        filt_arr = np.array(filt_buf)
        is_trough, trough_abs = detect_trough(
            filt_arr, half_cycle, last_trough_abs, sample_count)

        if not is_trough:
            continue

        last_trough_abs = trough_abs
        prediction_count += 1

        # --- Prediction ---
        t_pred_start = time.perf_counter()

        # Extract last 1s of filtered signal, subsample to 64 points
        input_wave = filt_arr[-waveform_native::WAVEFORM_SUBSAMPLE]  # (64,)

        # Normalize and predict
        x = (input_wave - feat_mean) / feat_std
        x = np.append(x, 1.0)  # bias -> (65,)
        y_norm = x @ W          # (128,)
        y_pred = y_norm * out_std + out_mean

        # Find troughs in predicted waveform
        pred_troughs = argrelmin(y_pred, order=half_cycle)[0]

        if len(pred_troughs) < 2:
            if prediction_count % 50 == 0:
                print(f"[{prediction_count}] <2 troughs in prediction, skipping")
            continue

        # 2nd trough offset in samples from "now" (end of input window)
        trough_2_offset = pred_troughs[1]
        delay_s = (trough_2_offset / fs) - ASSUMED_LATENCY_S

        if delay_s > MIN_DELAY_S:
            schedule_beep(delay_s)

        t_pred_elapsed = time.perf_counter() - t_pred_start

        # --- Console output ---
        if prediction_count % 20 == 0:
            print(f"[{prediction_count}] "
                  f"pred_troughs={pred_troughs[:3].tolist()}  "
                  f"delay={delay_s * 1000:.0f}ms  "
                  f"compute={t_pred_elapsed * 1000:.1f}ms")

        # --- Diagnostic capture ---
        if prediction_count in DIAG_ROUNDS:
            beep_delay_ms = delay_s * 1000 if delay_s > MIN_DELAY_S else None
            diag_snapshots[prediction_count] = {
                "input_signal": filt_arr[-waveform_native:].copy(),
                "predicted": y_pred.copy(),
                "pred_troughs": pred_troughs.copy(),
                "beep_delay_ms": beep_delay_ms,
                "fs": fs,
            }
            print(f"  [diag] captured round {prediction_count}")

            if len(diag_snapshots) == len(DIAG_ROUNDS):
                _save_diagnostic(diag_snapshots)
                print(f"  [diag] saved {DIAG_PATH}")


if __name__ == "__main__":
    main()
