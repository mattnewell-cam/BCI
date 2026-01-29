"""
Lock Live 2 - Sinusoid-fit based alpha phase tracking with audio feedback.

Every 300ms, fits a sinusoid to the last 400ms of continuously-bandpassed
EEG (using 1000ms of filter warm-up), projects forward, and schedules
beeps at the next 3 troughs (skipping the very first projected trough).

On the first 3 fitting rounds, captures diagnostic data and saves a
validation plot to media/.

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
from scipy.signal import butter, sosfilt, sosfilt_zi
from pylsl import StreamInlet

from utils import find_eeg_stream
from alpha_lock_logic import beep

CHANNEL_LABELS = ["AF7", "AF8", "TP9", "TP10"]

# --- Filter parameters (hardcoded for now) ---
BAND_LO = 8.3
BAND_HI = 10.3
FILTER_ORDER = 4
FIT_FREQ_LO = 8.8
FIT_FREQ_HI = 9.8
FIT_FREQ_STEP = 0.02

# --- Timing ---
FIT_INTERVAL_S = 0.300   # refit every 300ms
FIT_WINDOW_S = 0.400     # fit sinusoid to last 400ms
LOOKBACK_S = 1.000       # bandpassed buffer needed (warm-up + fit window)
BEEP_FREQ_HZ = 880
BEEP_MS = 15

# --- Diagnostics ---
N_DIAGNOSTIC_ROUNDS = 3
DIAGNOSTIC_FUTURE_S = 0.400  # extra data to capture after fit


# ---------------------------------------------------------------------------
# Sinusoid fitting (same as alpha_lock_sections)
# ---------------------------------------------------------------------------

def fit_sinusoid(signal_section, fs, freq_lo, freq_hi, freq_step):
    """Fit A*sin(2*pi*f*t + phi), grid-searching frequency for best RMSE."""
    n = len(signal_section)
    t = np.arange(n) / fs

    best_rmse = np.inf
    best_f = (freq_lo + freq_hi) / 2

    for f in np.arange(freq_lo, freq_hi + freq_step / 2, freq_step):
        w = 2 * np.pi * f
        X = np.column_stack([np.sin(w * t), np.cos(w * t)])
        coeffs, _, _, _ = np.linalg.lstsq(X, signal_section, rcond=None)
        resid = signal_section - X @ coeffs
        rmse = np.sqrt(np.mean(resid ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_f = f
            best_coeffs = coeffs

    a, b = best_coeffs
    amplitude = np.sqrt(a ** 2 + b ** 2)
    phase = np.arctan2(b, a)
    return best_f, amplitude, phase, best_rmse


def synth_sinusoid(t, freq, amplitude, phase):
    """Generate A*sin(2*pi*f*t + phi)."""
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def find_troughs(freq, amplitude, phase, t_start, n_troughs=4):
    """Find the next n trough times after t_start.

    Trough of A*sin(2*pi*f*t + phi) occurs when 2*pi*f*t + phi = -pi/2 + 2*pi*k.
    So t = (-pi/2 - phi + 2*pi*k) / (2*pi*f).
    """
    w = 2 * np.pi * freq
    troughs = []
    # Start searching from a k that puts us near t_start
    k_start = int(np.floor((w * t_start + phase + np.pi / 2) / (2 * np.pi)))
    for k in range(k_start, k_start + n_troughs + 5):
        t_trough = (-np.pi / 2 - phase + 2 * np.pi * k) / w
        if t_trough > t_start:
            troughs.append(t_trough)
            if len(troughs) == n_troughs:
                break
    return troughs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Connect to EEG stream
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=2)

    info = inlet.info()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()
    n_eeg = min(4, n_channels)
    print(f"Connected: fs={fs} Hz, channels={n_eeg}, name={info.name()}")

    # --- Bandpass filter (continuous, state-preserving) ---
    sos = butter(FILTER_ORDER, [BAND_LO, BAND_HI], btype="band",
                 fs=fs, output="sos")
    zi = sosfilt_zi(sos)
    # One filter state per channel
    filter_states = [zi.copy() for _ in range(n_eeg)]

    # Ring buffers: raw and filtered (per channel)
    # Need enough to hold all data from round 8 through plot time (~5s margin)
    buf_len = int(10.0 * fs)
    raw_bufs = [deque(maxlen=buf_len) for _ in range(n_eeg)]
    filt_bufs = [deque(maxlen=buf_len) for _ in range(n_eeg)]

    # --- Channel selection ---
    # For now, use channel 3 (TP10) as default; could add auto-selection later
    best_ch = 3 if n_eeg > 3 else 0

    # --- Beep scheduling ---
    pending_timers = []

    def schedule_beep(delay_s):
        """Schedule a beep delay_s seconds from now."""
        if delay_s < 0.005:
            return  # too close, skip
        t = threading.Timer(delay_s, beep, args=[BEEP_FREQ_HZ, BEEP_MS])
        t.daemon = True
        t.start()
        pending_timers.append(t)

    # --- Diagnostic capture (rounds 8-10) ---
    DIAG_START_ROUND = 8   # 1-indexed: capture rounds 8, 9, 10
    diagnostics = []
    fit_round = 0

    # --- Timing ---
    sample_count = 0
    t0_wall = time.time()  # wall clock at start
    last_fit_time = 0.0
    min_samples_for_fit = int(LOOKBACK_S * fs)

    print(f"Filter: {BAND_LO}-{BAND_HI} Hz, fit range: {FIT_FREQ_LO}-{FIT_FREQ_HI} Hz")
    print(f"Buffering {LOOKBACK_S}s before first fit...")

    while True:
        # --- Pull all available samples ---
        chunk, timestamps = inlet.pull_chunk(timeout=0.01, max_samples=256)
        if chunk:
            for sample in chunk:
                for ch in range(n_eeg):
                    x = np.array([sample[ch]])
                    y, filter_states[ch] = sosfilt(sos, x, zi=filter_states[ch])
                    raw_bufs[ch].append(sample[ch])
                    filt_bufs[ch].append(y[0])
                sample_count += 1

        # --- Check if time to fit ---
        now_wall = time.time()
        elapsed = now_wall - t0_wall

        if sample_count < min_samples_for_fit:
            continue

        if elapsed - last_fit_time < FIT_INTERVAL_S:
            time.sleep(0.005)
            continue

        last_fit_time = elapsed

        # --- Grab last 1000ms of filtered signal, fit last 400ms ---
        lookback_samples = int(LOOKBACK_S * fs)
        fit_samples = int(FIT_WINDOW_S * fs)

        filt_arr = np.array(filt_bufs[best_ch])
        if len(filt_arr) < lookback_samples:
            continue

        # Last 1000ms of filtered output
        lookback_chunk = filt_arr[-lookback_samples:]
        # Last 400ms for fitting
        fit_chunk = lookback_chunk[-fit_samples:]

        # --- Fit sinusoid ---
        # t=0 is the START of the fit window
        freq, amp, phase, rmse = fit_sinusoid(
            fit_chunk, fs, FIT_FREQ_LO, FIT_FREQ_HI, FIT_FREQ_STEP,
        )

        # Time reference: t=0 at start of fit window,
        # t=FIT_WINDOW_S at "now" (end of fit window)
        t_now = FIT_WINDOW_S  # "now" relative to fit window start

        # Find next 4 troughs after "now"
        troughs = find_troughs(freq, amp, phase, t_now, n_troughs=4)

        # Schedule beeps at troughs 2, 3, 4 (skip trough 1)
        for trough_t in troughs[1:4]:
            delay = trough_t - t_now  # seconds from now
            schedule_beep(delay)

        # Clean up expired timers
        pending_timers[:] = [t for t in pending_timers if t.is_alive()]

        # --- Console output ---
        trough_delays = [f"{(t - t_now)*1000:.0f}ms" for t in troughs[:4]]
        print(f"fit: {freq:.2f}Hz  amp={amp:.2f}  rmse={rmse:.3f}  "
              f"troughs=[{', '.join(trough_delays)}]  "
              f"beeps at [{', '.join(trough_delays[1:4])}]")

        # --- Diagnostic capture for rounds 8-10 ---
        fit_round += 1
        if DIAG_START_ROUND <= fit_round < DIAG_START_ROUND + N_DIAGNOSTIC_ROUNDS:
            diagnostics.append({
                "round": fit_round,
                "lookback": lookback_chunk.copy(),
                "freq": freq,
                "amp": amp,
                "phase": phase,
                "rmse": rmse,
                "fit_samples": fit_samples,
                "lookback_samples": lookback_samples,
                "wall_time": now_wall,
                "sample_idx": sample_count,
            })

            if fit_round == DIAG_START_ROUND + N_DIAGNOSTIC_ROUNDS - 1:
                # Schedule the diagnostic plot after future data arrives
                def capture_and_plot():
                    time.sleep(DIAGNOSTIC_FUTURE_S + 0.2)
                    _save_diagnostic_plot(diagnostics, filt_bufs[best_ch],
                                         fs, sample_count)
                t = threading.Thread(target=capture_and_plot, daemon=True)
                t.start()


def _save_diagnostic_plot(diagnostics, filt_buf, fs, current_sample_count):
    """Save diagnostic plot for captured fitting rounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(diagnostics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    filt_arr = np.array(filt_buf)
    # The deque's last element corresponds to current_sample_count.
    # Position in array = sample_idx - (current_sample_count - len(filt_arr))
    deque_offset = current_sample_count - len(filt_arr)

    for i, diag in enumerate(diagnostics):
        ax = axes[i]

        lookback_samples = diag["lookback_samples"]
        fit_samples = diag["fit_samples"]
        future_samples = int(DIAGNOSTIC_FUTURE_S * fs)

        # (a) Blue: the 1000ms lookback (bandpassed), saved at capture time
        lookback = diag["lookback"]
        t_lookback = np.arange(len(lookback)) / fs

        # (b) Green: full 1400ms from the continuous filter buffer
        end_abs = diag["sample_idx"] + future_samples
        start_abs = diag["sample_idx"] - lookback_samples
        # Convert absolute indices to deque positions
        start_pos = max(0, start_abs - deque_offset)
        end_pos = min(len(filt_arr), end_abs - deque_offset)
        full_window = filt_arr[start_pos:end_pos]
        t_full = np.arange(len(full_window)) / fs

        # (c) Red: fitted sinusoid projected across the full 1400ms
        # Fit's t=0 is the start of the 400ms fit window, which is
        # (lookback_samples - fit_samples) into the lookback
        fit_offset_s = (lookback_samples - fit_samples) / fs
        t_sin_shifted = t_full - fit_offset_s
        fitted_sin = synth_sinusoid(
            t_sin_shifted, diag["freq"], diag["amp"], diag["phase"],
        )

        now_t = lookback_samples / fs

        ax.plot(t_full, full_window, color="green", lw=0.8, alpha=0.7,
                label="Full 1400ms (bandpassed)")
        ax.plot(t_lookback, lookback, color="blue", lw=1.0, alpha=0.9,
                label="1000ms lookback (used)")
        ax.plot(t_full, fitted_sin, color="red", lw=1.2, alpha=0.8,
                label=f"Fitted sin: {diag['freq']:.2f}Hz, RMSE={diag['rmse']:.3f}")

        ax.axvline(fit_offset_s, color="gray", ls=":", lw=1,
                   label="Fit window start")
        ax.axvline(now_t, color="black", ls="--", lw=1, label='"Now"')

        # Mark projected troughs
        troughs = find_troughs(
            diag["freq"], diag["amp"], diag["phase"],
            FIT_WINDOW_S, n_troughs=4,
        )
        for j, tr in enumerate(troughs):
            tr_plot = tr + fit_offset_s  # convert to plot time
            if tr_plot <= t_full[-1]:
                color = "gray" if j == 0 else "orange"
                ax.axvline(tr_plot, color=color, ls=":", lw=0.8, alpha=0.7)
                val = synth_sinusoid(np.array([tr]), diag["freq"],
                                     diag["amp"], diag["phase"])[0]
                ax.plot(tr_plot, val, "v", color=color, ms=8, zorder=5)

        ax.set_title(f"Round {diag['round']}", fontsize=11)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Lock Live 2 — Diagnostic: Fit Validation", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "..", "media",
                            "lock_live_2_diagnostic.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nDiagnostic plot saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
