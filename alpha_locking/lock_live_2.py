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
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi, argrelmin

# --- Filter parameters ---
BAND_LO = 8
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
DURATION_S = 20              # run for this many seconds then save diagnostics
ASSUMED_LATENCY_S = 0.0    # assumed audio+processing latency to compensate
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
BEEP_DIAG_ROUNDS = {20, 40, 60}
BEEP_DIAG_PATH_LIVE = os.path.join(os.path.dirname(__file__), "..", "media",
                                   "lock_live_2_beep_diag.png")
BEEP_DIAG_PATH_TEST = os.path.join(os.path.dirname(__file__), "..", "media",
                                   "test_lock_live_2_beep_diag.png")

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
    print("data.shape", data.shape, "dtype", data.dtype, "nbytes", data.nbytes / 1e6, "MB")
    sig = data[CHANNEL_TP10]
    print("sig.shape", sig.shape, "dtype", sig.dtype, "nbytes", sig.nbytes / 1e6, "MB")

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


def detect_troughs_range(filt_arr, half_cycle, last_trough_abs, sample_count,
                         first_candidate_idx, last_candidate_idx):
    """Check a range of candidate positions for troughs.

    Iterates from first_candidate_idx to last_candidate_idx (inclusive) in
    filt_arr, yielding all confirmed troughs.

    Yields (abs_sample_pos,) for each trough found.
    """
    n = len(filt_arr)
    for candidate in range(first_candidate_idx, last_candidate_idx + 1):
        if candidate < half_cycle or candidate + half_cycle >= n:
            continue

        lo = candidate - half_cycle
        hi = candidate + half_cycle + 1
        center_val = filt_arr[candidate]

        if center_val <= filt_arr[lo:hi].min():
            abs_pos = sample_count - (n - candidate)
            if abs_pos - last_trough_abs >= half_cycle * 2:
                last_trough_abs = abs_pos
                yield abs_pos


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


def save_beep_diag(captures, trough_spacings, beep_play_times, chunk_sizes,
                   path, fs=256, half_cycle=10):
    """Save beep diagnostic figure for comparing test vs live behavior.

    captures: dict of {beep_number: snapshot_dict}
    trough_spacings: list of inter-trough intervals in samples
    beep_play_times: list of beep play times in seconds (simulated or real)
    chunk_sizes: list of chunk sizes received
    """
    from matplotlib.gridspec import GridSpec

    beep_nums = sorted(captures.keys())
    n_beep_panels = len(beep_nums)
    # beep panels + 1 row for two side-by-side histograms
    fig = plt.figure(figsize=(14, 4 * (n_beep_panels + 1)))
    gs = GridSpec(n_beep_panels + 1, 2, figure=fig,
                  height_ratios=[1] * n_beep_panels + [1])

    # --- Per-beep waveform panels (span both columns) ---
    for i, bnum in enumerate(beep_nums):
        ax = fig.add_subplot(gs[i, :])
        snap = captures[bnum]

        # Input signal (last 1s) -> time from -1000ms to 0
        sig = snap["filt_signal"]
        t_in = np.arange(len(sig)) / fs * 1000 - len(sig) / fs * 1000
        ax.plot(t_in, sig, "steelblue", lw=0.8, label="input (1s)")

        # Predicted waveform -> time from 0ms to 500ms
        pred = snap["y_pred"]
        t_pred = np.arange(len(pred)) / fs * 1000
        ax.plot(t_pred, pred, "mediumpurple", lw=1.2, label="prediction (0.5s)")

        # Predicted troughs
        for j, ti in enumerate(snap["pred_troughs"]):
            t_ms = ti / fs * 1000
            if j == 0:
                ax.axvline(t_ms, color="orange", ls="--", lw=1.0, alpha=0.7)
            elif j == 1:
                ax.axvline(t_ms, color="red", ls="--", lw=1.5,
                           label=f"2nd trough @ {t_ms:.0f}ms")

        # Beep target
        delay_ms = snap["delay_s"] * 1000
        if delay_ms > MIN_DELAY_S * 1000:
            ax.axvline(delay_ms + ASSUMED_LATENCY_S * 1000, color="green",
                       ls="-", lw=1.5, alpha=0.7,
                       label=f"beep play @ {delay_ms + ASSUMED_LATENCY_S*1000:.0f}ms")

        ax.axvline(0, color="black", ls=":", lw=0.8, alpha=0.5, label="now")

        delta_target = snap["beep_target"] - snap["last_beep_target"]
        title = (f"Beep #{bnum}  |  pred#{snap['prediction_count']}  "
                 f"delay={delay_ms:.0f}ms  "
                 f"target={snap['beep_target']}  "
                 f"\u0394target={delta_target}  "
                 f"chunk={snap['n_new']}  "
                 f"trough_abs={snap['trough_abs']}  "
                 f"SC={snap['sample_count']}")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", ncol=4)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("uV (filtered)")

    # --- Bottom row: two separate histograms ---
    trough_spacings = np.array(trough_spacings) if trough_spacings else np.array([])
    beep_play_times = np.array(beep_play_times) if beep_play_times else np.array([])
    chunk_sizes = np.array(chunk_sizes) if chunk_sizes else np.array([])

    # Left: trough spacing in ms
    ax_ts = fig.add_subplot(gs[n_beep_panels, 0])
    if len(trough_spacings) > 0:
        trough_spacings_ms = trough_spacings / fs * 1000
        bins_ts = np.arange(0, min(500, trough_spacings_ms.max() + 10), 5)
        ax_ts.hist(trough_spacings_ms, bins=bins_ts, alpha=0.7, color="steelblue")
        ax_ts.set_title(f"Trough-to-trough intervals (ms)\n"
                        f"n={len(trough_spacings_ms)}  "
                        f"mean={trough_spacings_ms.mean():.0f}ms  "
                        f"min={trough_spacings_ms.min():.0f}ms",
                        fontsize=9, fontweight="bold")
    else:
        ax_ts.set_title("Trough-to-trough intervals (no data)", fontsize=9)
    ax_ts.set_xlabel("ms")
    ax_ts.set_ylabel("count")
    ax_ts.grid(True, alpha=0.3)

    # Right: beep play-time intervals in ms
    ax_bi = fig.add_subplot(gs[n_beep_panels, 1])
    if len(beep_play_times) > 1:
        intervals_ms = np.diff(beep_play_times) * 1000
        bins_bi = np.arange(0, min(500, intervals_ms.max() + 10), 5)
        ax_bi.hist(intervals_ms, bins=bins_bi, alpha=0.7, color="mediumpurple")
        ax_bi.set_title(f"Beep play-time intervals (ms)\n"
                        f"n={len(intervals_ms)}  "
                        f"mean={intervals_ms.mean():.0f}ms  "
                        f"min={intervals_ms.min():.0f}ms",
                        fontsize=9, fontweight="bold")
    else:
        ax_bi.set_title("Beep play-time intervals (no data)", fontsize=9)
    ax_bi.set_xlabel("ms")
    ax_bi.set_ylabel("count")
    ax_bi.grid(True, alpha=0.3)

    cs_str = (f"chunks: n={len(chunk_sizes)} mean={chunk_sizes.mean():.1f} "
              f"max={chunk_sizes.max()}" if len(chunk_sizes) > 0 else "")
    fig.suptitle(f"half_cycle={half_cycle}  cooldown={half_cycle*2} samples "
                 f"({half_cycle*2/fs*1000:.0f}ms)  |  {cs_str}",
                 fontsize=10, fontweight="bold", y=1.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [beep_diag] saved {path}")


def main():
    import sounddevice as sd
    from pylsl import StreamInlet
    from utils import find_eeg_stream

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

    # Ring buffer: pre-allocated numpy array + write pointer
    _ring_cap = int(2.0 * fs)  # 2s of filtered signal
    _ring = np.zeros(_ring_cap, dtype=np.float64)
    _ring_n = 0  # how many valid samples are in the buffer

    # Open persistent low-latency audio stream
    audio_stream = None
    try:
        audio_stream = sd.OutputStream(
            samplerate=AUDIO_FS, channels=1, callback=_audio_callback,
            dtype="float32", latency="low",
        )
        audio_stream.start()
        schedule_beep(0.0)  # test beep
        print("Audio stream started")
    except Exception as e:
        print(f"Audio unavailable ({e}), running without sound")

    # Trough detection state
    last_trough_abs = -1000
    sample_count = 0
    prediction_count = 0
    last_checked_candidate = -1  # absolute sample index of last checked candidate
    last_beep_target_sample = -1000  # predicted beep time in absolute samples
    last_beep_play_time = -1000.0   # wall-clock play time of last scheduled beep

    # Beep cooldown: minimum samples between predicted beep times
    beep_cooldown_samples = half_cycle * 2
    beep_cooldown_s = beep_cooldown_samples / fs  # same cooldown in seconds

    # Diagnostic state
    diag_snapshots = {}

    # Beep diagnostic tracking
    beep_count = 0
    beep_diag_captures = {}
    trough_spacings = []
    beep_times_wall = []
    chunk_sizes = []
    prev_trough_abs = -1000
    t0_loop = time.perf_counter()
    last_loop_t = t0_loop
    last_loop_end = t0_loop
    last_chunk_ts = None

    # Timing diagnostics
    timing_totals = {
        "pull": 0.0,
        "filter": 0.0,
        "ring": 0.0,
        "candidates": 0.0,
        "trough_scan": 0.0,
        "predict": 0.0,
        "schedule": 0.0,
        "loop": 0.0,
    }
    timing_max = {k: 0.0 for k in timing_totals}
    timing_samples = 0
    slow_loop_threshold_s = 0.200

    # Accumulate time between beeps to explain >200ms gaps
    since_beep = {
        "idle": 0.0,
        "pull": 0.0,
        "sleep": 0.0,
        "filter": 0.0,
        "ring": 0.0,
        "candidates": 0.0,
        "trough_scan": 0.0,
        "predict": 0.0,
        "schedule": 0.0,
        "other": 0.0,
    }
    last_beep_wall = None
    beep_gap_threshold_s = 0.200
    since_beep_counts = {
        "chunks": 0,
        "samples": 0,
        "troughs": 0,
        "predictions": 0,
        "pred_lt2_troughs": 0,
        "delay_too_short": 0,
        "cooldown_sample_block": 0,
        "cooldown_play_block": 0,
        "scheduled": 0,
    }

    print(f"Model: {BAND_LO}-{BAND_HI} Hz, "
          f"input={WAVEFORM_S}s@{fs // WAVEFORM_SUBSAMPLE}Hz, "
          f"output={WAVEFORM_OUT_NATIVE / fs:.2f}s@{fs}Hz, "
          f"latency comp={ASSUMED_LATENCY_S * 1000:.0f}ms, "
          f"half_cycle={half_cycle}")
    print(f"Buffering {WAVEFORM_S}s before predictions...")

    # ========== PHASE C: MAIN LOOP ==========
    while time.perf_counter() - t0_loop < DURATION_S:
        t_loop_start = time.perf_counter()
        iter_times = {
            "pull": 0.0,
            "sleep": 0.0,
            "filter": 0.0,
            "ring": 0.0,
            "candidates": 0.0,
            "trough_scan": 0.0,
            "predict": 0.0,
            "schedule": 0.0,
            "other": 0.0,
        }
        idle_gap = t_loop_start - last_loop_end
        loop_gap = t_loop_start - last_loop_t
        last_loop_t = t_loop_start
        if idle_gap > 0:
            since_beep["idle"] += idle_gap

        # --- Pull all available EEG samples ---
        t_pull_start = time.perf_counter()
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=256)
        t_pull = time.perf_counter() - t_pull_start
        timing_totals["pull"] += t_pull
        if t_pull > timing_max["pull"]:
            timing_max["pull"] = t_pull
        iter_times["pull"] += t_pull
        if not chunk:
            t_sleep_start = time.perf_counter()
            time.sleep(0.001)
            t_sleep = time.perf_counter() - t_sleep_start
            iter_times["sleep"] += t_sleep

            t_loop = time.perf_counter() - t_loop_start
            iter_times["other"] = max(0.0, t_loop - sum(iter_times.values()))
            for k, v in iter_times.items():
                since_beep[k] += v
            last_loop_end = time.perf_counter()
            if t_loop > slow_loop_threshold_s:
                accounted = sum(iter_times.values())
                unaccounted = max(0.0, t_loop - accounted)
                print(f"[slow-loop] total={t_loop*1000:.1f}ms gap={loop_gap*1000:.1f}ms")
                for k in ["pull", "sleep", "other"]:
                    v = iter_times[k]
                    if v > 0:
                        print(f"{k} action: {v*1000:.1f} ms")
                if unaccounted > 0.0005:
                    print(f"unaccounted action: {unaccounted*1000:.1f} ms")
            continue

        chunk_arr = np.array(chunk)
        new_raw = chunk_arr[:, best_ch].astype(np.float64)
        n_new = len(new_raw)
        chunk_sizes.append(n_new)
        since_beep_counts["chunks"] += 1
        since_beep_counts["samples"] += n_new
        if timestamps:
            chunk_start_ts = timestamps[0]
            if last_chunk_ts is not None:
                lsl_gap = chunk_start_ts - last_chunk_ts
                if lsl_gap > 0.200:
                    print(f"[lsl-gap] {lsl_gap*1000:.1f}ms (n_new={n_new})")
            last_chunk_ts = timestamps[-1]

        # --- Streaming bandpass filter ---
        t_filter_start = time.perf_counter()
        new_filtered, zi = sosfilt(sos, new_raw, zi=zi)
        t_filter = time.perf_counter() - t_filter_start
        timing_totals["filter"] += t_filter
        if t_filter > timing_max["filter"]:
            timing_max["filter"] = t_filter
        iter_times["filter"] += t_filter

        # --- Append to ring buffer ---
        t_ring_start = time.perf_counter()
        if n_new >= _ring_cap:
            # Chunk larger than buffer: keep only the last _ring_cap samples
            _ring[:] = new_filtered[-_ring_cap:]
            _ring_n = _ring_cap
        elif _ring_n + n_new <= _ring_cap:
            # Fits without wrapping
            _ring[_ring_n:_ring_n + n_new] = new_filtered
            _ring_n += n_new
        else:
            # Shift out oldest samples to make room
            keep = _ring_cap - n_new
            _ring[:keep] = _ring[_ring_n - keep:_ring_n]
            _ring[keep:keep + n_new] = new_filtered
            _ring_n = _ring_cap
        sample_count += n_new
        t_ring = time.perf_counter() - t_ring_start
        timing_totals["ring"] += t_ring
        if t_ring > timing_max["ring"]:
            timing_max["ring"] = t_ring
        iter_times["ring"] += t_ring

        # --- Need enough samples ---
        if _ring_n < waveform_native + half_cycle:
            t_loop = time.perf_counter() - t_loop_start
            iter_times["other"] = max(0.0, t_loop - sum(iter_times.values()))
            for k, v in iter_times.items():
                since_beep[k] += v
            last_loop_end = time.perf_counter()
            if t_loop > slow_loop_threshold_s:
                accounted = sum(iter_times.values())
                unaccounted = max(0.0, t_loop - accounted)
                print(f"[slow-loop] total={t_loop*1000:.1f}ms gap={loop_gap*1000:.1f}ms")
                for k in ["pull", "sleep", "filter", "ring", "other"]:
                    v = iter_times[k]
                    if v > 0:
                        print(f"{k} action: {v*1000:.1f} ms")
                if unaccounted > 0.0005:
                    print(f"unaccounted action: {unaccounted*1000:.1f} ms")
            continue

        # --- Trough detection: check ALL new candidate positions ---
        t_candidates_start = time.perf_counter()
        filt_arr = _ring[:_ring_n]
        buf_len = _ring_n

        # The rightmost confirmable candidate in the buffer
        rightmost_candidate_idx = buf_len - 1 - half_cycle
        rightmost_candidate_abs = sample_count - 1 - half_cycle

        # The first new candidate to check: either the one after last_checked,
        # or the leftmost valid candidate if we haven't checked any yet
        if last_checked_candidate < 0:
            first_candidate_abs = max(
                sample_count - buf_len + half_cycle,
                rightmost_candidate_abs - n_new + 1
            )
        else:
            first_candidate_abs = last_checked_candidate + 1

        # Clamp to valid range
        first_candidate_abs = max(first_candidate_abs, rightmost_candidate_abs - n_new + 1)
        first_candidate_abs = max(first_candidate_abs, sample_count - buf_len + half_cycle)

        if first_candidate_abs > rightmost_candidate_abs:
            last_checked_candidate = rightmost_candidate_abs
            continue

        # Convert absolute positions to buffer indices
        first_candidate_idx = first_candidate_abs - (sample_count - buf_len)
        last_candidate_idx = rightmost_candidate_idx
        n_candidates = last_candidate_idx - first_candidate_idx + 1
        t_candidates = time.perf_counter() - t_candidates_start
        timing_totals["candidates"] += t_candidates
        if t_candidates > timing_max["candidates"]:
            timing_max["candidates"] = t_candidates
        iter_times["candidates"] += t_candidates

        t_scan_start = time.perf_counter()
        for trough_abs in detect_troughs_range(
                filt_arr, half_cycle, last_trough_abs, sample_count,
                first_candidate_idx, last_candidate_idx):

            # Track trough spacing
            spacing = trough_abs - last_trough_abs if last_trough_abs >= 0 else 0
            if last_trough_abs >= 0:
                trough_spacings.append(spacing)
            if spacing < half_cycle * 2 and last_trough_abs >= 0:
                print(f"  WARNING: trough spacing {spacing} < {half_cycle*2} "
                      f"at abs={trough_abs} (last={last_trough_abs})")

            last_trough_abs = trough_abs
            prev_trough_abs = trough_abs
            prediction_count += 1
            since_beep_counts["troughs"] += 1
            since_beep_counts["predictions"] += 1

            # --- Prediction ---
            t_pred_start = time.perf_counter()

            # Extract last 1s of filtered signal, subsample to 64 points
            input_wave = filt_arr[-waveform_native::WAVEFORM_SUBSAMPLE]  # (64,)

            # Normalize and predict
            x = (input_wave - feat_mean) / feat_std
            x = np.append(x, 1.0)  # bias -> (65,)
            y_norm = x @ W          # (128,)
            y_pred = y_norm * out_std + out_mean

            # Find troughs in predicted waveform.
            # Prepend half_cycle of real signal so argrelmin can detect
            # troughs at/near t=0 (the boundary between history and prediction).
            overlap = filt_arr[-half_cycle:]
            search_sig = np.concatenate([overlap, y_pred])
            raw_troughs = argrelmin(search_sig, order=half_cycle)[0]
            pred_troughs = raw_troughs - half_cycle  # shift to prediction-relative
            pred_troughs = pred_troughs[pred_troughs > 0]  # >0: exclude trigger trough at boundary

            if len(pred_troughs) < 2:
                since_beep_counts["pred_lt2_troughs"] += 1
                if prediction_count % 50 == 0:
                    print(f"[{prediction_count}] <2 troughs in prediction, skipping")
                continue

            # 2nd trough offset in samples from "now" (end of input window)
            trough_2_offset = pred_troughs[1]
            delay_s = (trough_2_offset / fs) - ASSUMED_LATENCY_S

            if delay_s > MIN_DELAY_S:
                # Beep cooldown: check BOTH sample-domain and wall-clock play time
                beep_target_sample = sample_count + int(delay_s * fs)
                beep_play_time = time.perf_counter() + delay_s
                sample_ok = beep_target_sample - last_beep_target_sample >= beep_cooldown_samples
                playtime_ok = beep_play_time - last_beep_play_time >= beep_cooldown_s
                if sample_ok and playtime_ok:
                    t_sched_start = time.perf_counter()
                    schedule_beep(delay_s)
                    t_sched = time.perf_counter() - t_sched_start
                    timing_totals["schedule"] += t_sched
                    if t_sched > timing_max["schedule"]:
                        timing_max["schedule"] = t_sched
                    iter_times["schedule"] += t_sched
                    beep_count += 1
                    beep_times_wall.append(beep_play_time)
                    since_beep_counts["scheduled"] += 1

                    now_wall = time.perf_counter()
                    if last_beep_wall is not None:
                        gap = now_wall - last_beep_wall
                        if gap > beep_gap_threshold_s:
                            accounted = sum(since_beep.values())
                            unaccounted = max(0.0, gap - accounted)
                            print(f"[beep-gap] total={gap*1000:.1f}ms")
                            print("counts "
                                  f"chunks={since_beep_counts['chunks']} "
                                  f"samples={since_beep_counts['samples']} "
                                  f"troughs={since_beep_counts['troughs']} "
                                  f"pred={since_beep_counts['predictions']} "
                                  f"lt2={since_beep_counts['pred_lt2_troughs']} "
                                  f"delay_short={since_beep_counts['delay_too_short']} "
                                  f"cd_sample={since_beep_counts['cooldown_sample_block']} "
                                  f"cd_play={since_beep_counts['cooldown_play_block']} "
                                  f"scheduled={since_beep_counts['scheduled']}")
                            for k in ["idle", "pull", "sleep", "filter", "ring",
                                      "candidates", "trough_scan", "predict",
                                      "schedule", "other"]:
                                v = since_beep[k]
                                if v > 0:
                                    print(f"{k} action: {v*1000:.1f} ms")
                            if unaccounted > 0.0005:
                                print(f"unaccounted action: {unaccounted*1000:.1f} ms")
                    last_beep_wall = now_wall
                    for k in since_beep:
                        since_beep[k] = 0.0
                    for k in since_beep_counts:
                        since_beep_counts[k] = 0
                else:
                    if not sample_ok:
                        since_beep_counts["cooldown_sample_block"] += 1
                    if not playtime_ok:
                        since_beep_counts["cooldown_play_block"] += 1
            else:
                since_beep_counts["delay_too_short"] += 1

            # Capture beep diagnostic
            if delay_s > MIN_DELAY_S and sample_ok and playtime_ok:
                if beep_count in BEEP_DIAG_ROUNDS:
                    beep_diag_captures[beep_count] = {
                        "filt_signal": filt_arr[-waveform_native:].copy(),
                        "y_pred": y_pred.copy(),
                        "pred_troughs": pred_troughs.copy(),
                        "trough_abs": trough_abs,
                        "sample_count": sample_count,
                        "delay_s": delay_s,
                        "beep_target": beep_target_sample,
                        "last_beep_target": last_beep_target_sample,
                        "prediction_count": prediction_count,
                        "n_new": n_new,
                        "n_candidates": n_candidates,
                        "fs": fs,
                    }

                last_beep_target_sample = beep_target_sample
                last_beep_play_time = beep_play_time

            t_pred_elapsed = time.perf_counter() - t_pred_start
            timing_totals["predict"] += t_pred_elapsed
            if t_pred_elapsed > timing_max["predict"]:
                timing_max["predict"] = t_pred_elapsed
            iter_times["predict"] += t_pred_elapsed

            # --- Console output ---
            if prediction_count % 100 == 0:
                elapsed = time.perf_counter() - t0_loop
                trough_rate = prediction_count / elapsed if elapsed > 0 else 0
                beep_rate = beep_count / elapsed if elapsed > 0 else 0
                print(f"[{prediction_count}] "
                      f"pred_troughs={pred_troughs[:3].tolist()}  "
                      f"delay={delay_s * 1000:.0f}ms  "
                      f"compute={t_pred_elapsed * 1000:.1f}ms  "
                      f"troughs/s={trough_rate:.1f}  "
                      f"beeps/s={beep_rate:.1f}  "
                      f"chunk={n_new}  "
                      f"SC={sample_count}")

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

        last_checked_candidate = rightmost_candidate_abs
        t_scan = time.perf_counter() - t_scan_start
        timing_totals["trough_scan"] += t_scan
        if t_scan > timing_max["trough_scan"]:
            timing_max["trough_scan"] = t_scan
        iter_times["trough_scan"] += t_scan

        t_loop = time.perf_counter() - t_loop_start
        timing_totals["loop"] += t_loop
        if t_loop > timing_max["loop"]:
            timing_max["loop"] = t_loop
        timing_samples += 1
        iter_times["other"] = max(0.0, t_loop - sum(iter_times.values()))
        for k, v in iter_times.items():
            since_beep[k] += v
        last_loop_end = time.perf_counter()

        if t_loop > slow_loop_threshold_s:
            accounted = sum(iter_times.values())
            unaccounted = max(0.0, t_loop - accounted)
            print(f"[slow-loop] total={t_loop*1000:.1f}ms gap={loop_gap*1000:.1f}ms")
            for k in ["pull", "sleep", "filter", "ring", "candidates",
                      "trough_scan", "predict", "schedule", "other"]:
                v = iter_times[k]
                if v > 0:
                    print(f"{k} action: {v*1000:.1f} ms")
            if unaccounted > 0.0005:
                print(f"unaccounted action: {unaccounted*1000:.1f} ms")

        if timing_samples % 200 == 0:
            avg = {k: (timing_totals[k] / timing_samples) * 1000 for k in timing_totals}
            mx = {k: timing_max[k] * 1000 for k in timing_max}
            print("[timing] avg_ms "
                  f"pull={avg['pull']:.2f} filter={avg['filter']:.2f} ring={avg['ring']:.2f} "
                  f"candidates={avg['candidates']:.2f} scan={avg['trough_scan']:.2f} "
                  f"predict={avg['predict']:.2f} schedule={avg['schedule']:.2f} "
                  f"loop={avg['loop']:.2f} | "
                  f"max_ms pull={mx['pull']:.2f} filter={mx['filter']:.2f} ring={mx['ring']:.2f} "
                  f"candidates={mx['candidates']:.2f} scan={mx['trough_scan']:.2f} "
                  f"predict={mx['predict']:.2f} schedule={mx['schedule']:.2f} "
                  f"loop={mx['loop']:.2f}")

    # ========== PHASE D: SAVE DIAGNOSTICS ==========
    elapsed = time.perf_counter() - t0_loop
    print(f"\nLoop finished after {elapsed:.1f}s  "
          f"({prediction_count} predictions, {beep_count} beeps)")

    if diag_snapshots:
        _save_diagnostic(diag_snapshots)
        print(f"  [diag] saved {DIAG_PATH}")

    if beep_diag_captures:
        save_beep_diag(
            beep_diag_captures, trough_spacings,
            beep_times_wall, chunk_sizes,
            BEEP_DIAG_PATH_LIVE, fs, half_cycle)

    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()


if __name__ == "__main__":
    main()
