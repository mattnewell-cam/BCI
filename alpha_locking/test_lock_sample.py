"""
Offline accuracy test for lock_live_2 trough-prediction logic.

Loads a pre-recorded sample, runs the same bandpass + trough-detection +
forward-prediction logic that lock_live_2 uses, and measures how far each
predicted beep lands from the nearest actual trough in the ground-truth signal.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.signal import butter, sosfilt, argrelmin
import matplotlib.pyplot as plt

# --- Parameters (must match lock_live_2) ---
BAND_LO = 7
BAND_HI = 12
FILTER_ORDER = 4
FIT_INTERVAL_S = 0.100
CHANNEL_TP10 = 3

# --- Tunable ---
LOOKBACK_S = 1.800       # seconds of signal to bandpass each round
N_GAPS = 1               # number of trough-trough gaps to average (gap-avg method)

# --- Linear regression ---
LR_N_GAPS = 10           # number of gap features for the regression
TRAIN_FRAC = 0.70

# --- Raw waveform features (predict_alpha style) ---
WAVEFORM_S = 1.0         # seconds of signal to use as features
WAVEFORM_SUBSAMPLE = 4   # 256 Hz -> 64 Hz -> 64 points
WAVEFORM_OUT_S = 0.50    # seconds of future signal to predict
WAVEFORM_OUT_NATIVE = 128 # 0.5s at 256 Hz


def load_signal():
    """Load sample and return signals, fs, filter sos, ground-truth troughs."""
    sample_path = os.path.join(os.path.dirname(__file__), "..",
                               "recordings", "full_night_350_1000.npz")
    rec = np.load(sample_path, allow_pickle=True)
    data = rec["data"]
    fs = int(rec["sample_rate"])
    print(f"Loaded: {data.shape[1]} samples, fs={fs}, "
          f"duration={data.shape[1]/fs:.1f}s, channels={data.shape[0]}")

    signal_tp10 = data[CHANNEL_TP10].astype(np.float64)

    sos = butter(FILTER_ORDER, [BAND_LO, BAND_HI], btype="band",
                 fs=fs, output="sos")

    full_filtered = sosfilt(sos, signal_tp10)
    half_cycle_samples = max(3, int(fs / BAND_HI / 2))
    all_troughs = argrelmin(full_filtered, order=half_cycle_samples)[0]
    print(f"Ground truth: {len(all_troughs)} troughs found (TP10)")

    return signal_tp10, full_filtered, fs, sos, all_troughs, half_cycle_samples


def find_nearest_trough_error(predicted_sample, all_troughs, fs, total_samples):
    """Return signed error in ms from predicted_sample to nearest ground-truth trough, or None."""
    if predicted_sample >= total_samples:
        return None
    diffs = all_troughs - predicted_sample
    nearest_idx = np.argmin(np.abs(diffs))
    return diffs[nearest_idx] / fs * 1000


def report(name, errors_ms, n_rounds):
    """Print a results block."""
    errors = np.array(errors_ms)
    abs_errors = np.abs(errors)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  {len(errors)} predictions over {n_rounds} rounds")
    print(f"{'='*50}")
    print(f"Signed error (+ = late, - = early):")
    print(f"  mean:   {errors.mean():+.1f} ms")
    print(f"  median: {np.median(errors):+.1f} ms")
    print(f"  std:    {errors.std():.1f} ms")
    print(f"Absolute error:")
    print(f"  mean:   {abs_errors.mean():.1f} ms")
    print(f"  median: {np.median(abs_errors):.1f} ms")
    print(f"  p95:    {np.percentile(abs_errors, 95):.1f} ms")
    print(f"  max:    {abs_errors.max():.1f} ms")
    print(f"Within 10ms: {(abs_errors < 10).sum()}/{len(errors)} "
          f"({(abs_errors < 10).mean()*100:.0f}%)")
    print(f"Within 20ms: {(abs_errors < 20).sum()}/{len(errors)} "
          f"({(abs_errors < 20).mean()*100:.0f}%)")
    print(f"Within 50ms: {(abs_errors < 50).sum()}/{len(errors)} "
          f"({(abs_errors < 50).mean()*100:.0f}%)")


def test_gap_average(signal, fs, sos, all_troughs, half_cycle_samples):
    """Current lock_live_2 method: average last N_GAPS gaps."""
    lookback_samples = int(LOOKBACK_S * fs)
    fit_interval_samples = int(FIT_INTERVAL_S * fs)
    total_samples = len(signal)

    cursor = lookback_samples
    all_errors = []
    fit_round = 0

    while cursor < total_samples:
        fit_round += 1

        window = signal[cursor - lookback_samples: cursor]
        filtered = sosfilt(sos, window)
        trough_indices = argrelmin(filtered, order=half_cycle_samples)[0]

        n_troughs_needed = N_GAPS + 1
        if len(trough_indices) < n_troughs_needed:
            cursor += fit_interval_samples
            continue

        last_n = trough_indices[-n_troughs_needed:]
        gaps = np.diff(last_n)
        avg_gap = gaps.mean()

        last_trough_sample = last_n[-1]
        samples_since = lookback_samples - last_trough_sample
        next2_delay = (2 * avg_gap / fs) - samples_since / fs

        if next2_delay > 0.005:
            predicted_sample = cursor + int(next2_delay * fs)
            err = find_nearest_trough_error(predicted_sample, all_troughs, fs, total_samples)
            if err is not None:
                all_errors.append(err)

        cursor += fit_interval_samples

    # Only report on the last 30% to match the linreg test split
    n_test = int(len(all_errors) * (1 - TRAIN_FRAC))
    errors_ms = all_errors[-n_test:]

    report(f"Gap Average  (LOOKBACK_S={LOOKBACK_S}, N_GAPS={N_GAPS}, last {100*(1-TRAIN_FRAC):.0f}%)",
           errors_ms, fit_round)


def fit_and_eval_ols(features, targets, meta, n_train, fs, all_troughs, total_samples, label):
    """Fit OLS on train split, evaluate on test split, print report.
    Returns per-test-row predicted trough sample positions (NaN if no prediction)."""
    n_total = len(features)
    n_test = n_total - n_train
    print(f"\n{label}: {n_total} rows, {features.shape[1]} features, "
          f"train={n_train}, test={n_test}")

    feat_mean = features[:n_train].mean(axis=0)
    feat_std = features[:n_train].std(axis=0)
    feat_std[feat_std == 0] = 1.0

    X = (features - feat_mean) / feat_std
    X = np.column_stack([X, np.ones(n_total)])

    X_train, y_train = X[:n_train], targets[:n_train]
    w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    y_pred = X[n_train:] @ w

    errors_ms = evaluate_predictions(y_pred, meta, n_train, fs, all_troughs, total_samples)
    report(label, errors_ms, n_test)

    # Collect predicted sample positions for plotting
    pred_samples = np.full(n_test, np.nan)
    for i in range(n_test):
        cursor, last_trough_sample, samples_since = meta[n_train + i]
        pred_gap = y_pred[i]
        next2_delay = (2 * pred_gap / fs) - samples_since / fs
        if next2_delay > 0.005:
            pred_samples[i] = cursor + int(next2_delay * fs)

    return pred_samples


def fit_waveform_to_waveform(features, targets_wave, meta, n_train, fs, all_troughs,
                             total_samples, half_cycle_samples, label):
    """Fit OLS: waveform input -> waveform output, find 2nd trough in prediction, report.
    Returns (pred_samples, Y_pred) — predicted trough positions and predicted waveforms."""
    n_total = len(features)
    n_test = n_total - n_train
    n_out = targets_wave.shape[1]
    print(f"\n{label}: {n_total} rows, {features.shape[1]} in -> {n_out} out, "
          f"train={n_train}, test={n_test}")

    feat_mean = features[:n_train].mean(axis=0)
    feat_std = features[:n_train].std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X = (features - feat_mean) / feat_std
    X = np.column_stack([X, np.ones(n_total)])

    out_mean = targets_wave[:n_train].mean()
    out_std = targets_wave[:n_train].std()
    out_std = out_std if out_std > 1e-8 else 1.0
    Y_norm = (targets_wave - out_mean) / out_std

    X_train = X[:n_train]
    Y_train = Y_norm[:n_train]
    W = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]

    Y_pred_norm = X[n_train:] @ W
    Y_pred = Y_pred_norm * out_std + out_mean

    errors_ms = evaluate_waveform_trough(Y_pred, meta, n_train, fs, all_troughs,
                                         total_samples, half_cycle_samples)
    report(label, errors_ms, n_test)

    # Collect predicted sample positions for plotting
    pred_samples = np.full(n_test, np.nan)
    for i in range(n_test):
        cursor = meta[n_train + i][0]
        pred = Y_pred[i]
        troughs = argrelmin(pred, order=half_cycle_samples)[0]
        if len(troughs) >= 2:
            pred_samples[i] = cursor + int(troughs[1])

    return pred_samples, Y_pred


def test_linear_regressions(signal_tp10, full_filtered, fs, sos, all_troughs, half_cycle_samples):
    """Run linear regression variants and plot comparison."""
    data = collect_features(signal_tp10, full_filtered, fs, sos, half_cycle_samples)
    total_samples = len(signal_tp10)

    targets = data['targets']
    meta = data['meta']
    n_total = len(targets)
    n_train = int(n_total * TRAIN_FRAC)

    # 1) TP10 gaps + amplitude -> next gap
    pred_gaps_amp = fit_and_eval_ols(
        data['gaps'], targets, meta, n_train, fs, all_troughs, total_samples,
        f"LinReg Gaps+Amp  ({LR_N_GAPS} gaps + 1 amp = {LR_N_GAPS+1} feats)")

    # 2) Raw waveform -> next gap
    pred_wave_gap = fit_and_eval_ols(
        data['waveform'], targets, meta, n_train, fs, all_troughs, total_samples,
        f"LinReg Waveform->Gap  ({data['waveform'].shape[1]} feats -> 1 gap)")

    # 3) Raw waveform -> next 0.5s waveform -> find 2nd trough
    pred_wave_wave, Y_pred_waveforms = fit_waveform_to_waveform(
        data['waveform'], data['waveform_target'], meta, n_train, fs,
        all_troughs, total_samples, half_cycle_samples,
        f"LinReg Waveform->Waveform  ({data['waveform'].shape[1]} -> {WAVEFORM_OUT_NATIVE} pts, 2nd trough)")

    # --- Plot 8 examples ---
    plot_examples(full_filtered, meta, n_train, fs, all_troughs, half_cycle_samples,
                  pred_gaps_amp, pred_wave_gap, pred_wave_wave, Y_pred_waveforms)


def plot_examples(full_filtered, meta, n_train, fs, all_troughs, half_cycle_samples,
                  pred_gaps_amp, pred_wave_gap, pred_wave_wave, Y_pred_waveforms):
    """Plot 8 example windows showing input, actual future, predicted future, and trough markers."""
    n_test = len(pred_gaps_amp)
    waveform_native = int(WAVEFORM_S * fs)

    # Pick 8 examples where all methods made a prediction
    valid = ~np.isnan(pred_gaps_amp) & ~np.isnan(pred_wave_gap) & ~np.isnan(pred_wave_wave)
    valid_idx = np.where(valid)[0]
    pick = np.linspace(0, len(valid_idx) - 1, 8, dtype=int)
    examples = valid_idx[pick]

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle("Trough Prediction Comparison — 2nd trough ahead",
                 fontsize=14, fontweight="bold")

    for ax_i, test_i in enumerate(examples):
        ax = axes[ax_i // 2, ax_i % 2]
        cursor = meta[n_train + test_i][0]

        # Time axes: input (last 1s) then future (next 0.5s)
        input_start = cursor - waveform_native
        total_start = input_start
        total_end = cursor + WAVEFORM_OUT_NATIVE
        t = np.arange(total_start, total_end) / fs * 1000  # ms
        t0 = total_start / fs * 1000  # offset for ms axis

        # Actual filtered signal: input + future
        actual = full_filtered[total_start:total_end]
        t_rel = t - cursor / fs * 1000  # relative to cursor, 0 = now

        # Plot actual signal
        n_input = waveform_native
        ax.plot(t_rel[:n_input], actual[:n_input], "k-", lw=1.0, alpha=0.5, label="Input")
        ax.plot(t_rel[n_input:], actual[n_input:], "k-", lw=1.5, label="Actual future")

        # Plot predicted waveform (waveform->waveform method)
        pred_wave = Y_pred_waveforms[test_i]
        t_pred = np.arange(WAVEFORM_OUT_NATIVE) / fs * 1000  # starts at 0
        ax.plot(t_pred, pred_wave, "m-", lw=1.2, alpha=0.7, label="Predicted wave")

        # Find the actual 2nd trough in the future window for ground truth marker
        future_troughs = all_troughs[(all_troughs >= cursor) & (all_troughs < total_end)]
        if len(future_troughs) >= 2:
            gt_sample = future_troughs[1]
            gt_ms = (gt_sample - cursor) / fs * 1000
            ax.axvline(gt_ms, color="black", ls="-", lw=2, alpha=0.6, label="Actual 2nd trough")

        # Method markers
        methods = [
            (pred_gaps_amp[test_i], "Gaps+Amp", "tab:blue", "--"),
            (pred_wave_gap[test_i], "Wave->Gap", "tab:orange", "--"),
            (pred_wave_wave[test_i], "Wave->Wave", "tab:red", ":"),
        ]
        for pred_sample, name, color, ls in methods:
            if not np.isnan(pred_sample):
                ms = (pred_sample - cursor) / fs * 1000
                ax.axvline(ms, color=color, ls=ls, lw=1.5, label=name)

        ax.axvline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
        ax.set_xlabel("ms relative to cursor")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Test #{n_train + test_i}", fontsize=10)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def collect_features(signal_tp10, full_filtered, fs, sos, half_cycle_samples):
    """Collect feature/target/meta rows for all regression variants.

    Returns dict with keys:
        'gaps':            (N, LR_N_GAPS+1) — TP10 gap features + amplitude
        'waveform':        (N, 64) — last 1s of filtered TP10, subsampled to 64 pts
        'waveform_target': (N, 64) — next 0.25s of filtered TP10 at 256 Hz
        'targets':         (N,) — next TP10 gap
        'meta':            list of (cursor, last_trough_sample, samples_since)
    """
    lookback_samples = int(LOOKBACK_S * fs)
    fit_interval_samples = int(FIT_INTERVAL_S * fs)
    total_samples = len(signal_tp10)
    n_troughs_needed = LR_N_GAPS + 1
    waveform_native = int(WAVEFORM_S * fs)  # 256 samples

    gap_feats = []
    waveform_feats = []
    waveform_targets = []
    targets = []
    meta = []

    cursor = lookback_samples
    while cursor < total_samples:
        # Need room for future waveform output
        if cursor + WAVEFORM_OUT_NATIVE > total_samples:
            break

        window_tp10 = signal_tp10[cursor - lookback_samples: cursor]
        filt_tp10 = sosfilt(sos, window_tp10)
        troughs_tp10 = argrelmin(filt_tp10, order=half_cycle_samples)[0]

        has_tp10 = len(troughs_tp10) >= n_troughs_needed + 1

        if has_tp10:
            last_n_plus = troughs_tp10[-(n_troughs_needed + 1):]
            all_gaps = np.diff(last_n_plus)
            tp10_gaps = all_gaps[:-1].astype(np.float64)
            target_gap = float(all_gaps[-1])

            last_tr = troughs_tp10[-1]
            prev_tr = troughs_tp10[-2]
            segment = filt_tp10[prev_tr:last_tr + 1]
            amplitude = float(segment.max() - segment.min())

            last_trough_sample = last_n_plus[-1]
            samples_since = lookback_samples - last_trough_sample

            # TP10 gaps + amplitude (11 features)
            gap_feats.append(np.append(tp10_gaps, amplitude))

            # Waveform input: last 1s from full_filtered, subsampled to 64 pts
            wave_in = full_filtered[cursor - waveform_native:cursor:WAVEFORM_SUBSAMPLE]
            waveform_feats.append(wave_in.astype(np.float64))

            # Waveform output target: next 0.25s at native 256 Hz (64 pts)
            wave_out = full_filtered[cursor:cursor + WAVEFORM_OUT_NATIVE]
            waveform_targets.append(wave_out.astype(np.float64))

            targets.append(target_gap)
            meta.append((cursor, last_trough_sample, samples_since))

        cursor += fit_interval_samples

    return {
        'gaps': np.array(gap_feats),
        'waveform': np.array(waveform_feats),
        'waveform_target': np.array(waveform_targets),
        'targets': np.array(targets),
        'meta': meta,
    }


def evaluate_predictions(y_pred, meta, n_train, fs, all_troughs, total_samples):
    """Given predicted gaps for the test split, compute error list."""
    errors_ms = []
    for i in range(len(y_pred)):
        idx = n_train + i
        cursor, last_trough_sample, samples_since = meta[idx]

        pred_gap = y_pred[i]
        next2_delay = (2 * pred_gap / fs) - samples_since / fs

        if next2_delay > 0.005:
            predicted_sample = cursor + int(next2_delay * fs)
            err = find_nearest_trough_error(predicted_sample, all_troughs, fs, total_samples)
            if err is not None:
                errors_ms.append(err)
    return errors_ms


def evaluate_waveform_trough(predicted_waveforms, meta, n_train, fs, all_troughs,
                             total_samples, half_cycle_samples):
    """Given predicted future waveforms, find the second trough and compute error."""
    errors_ms = []
    for i in range(len(predicted_waveforms)):
        idx = n_train + i
        cursor = meta[idx][0]

        # Find local minima in the predicted waveform
        pred = predicted_waveforms[i]
        troughs = argrelmin(pred, order=half_cycle_samples)[0]

        # We want the second trough (one after next)
        if len(troughs) < 2:
            continue
        trough_offset = troughs[1]
        predicted_sample = cursor + int(trough_offset)

        err = find_nearest_trough_error(predicted_sample, all_troughs, fs, total_samples)
        if err is not None:
            errors_ms.append(err)
    return errors_ms


def main():
    signal_tp10, full_filtered, fs, sos, all_troughs, half_cycle_samples = load_signal()

    test_gap_average(signal_tp10, fs, sos, all_troughs, half_cycle_samples)
    test_linear_regressions(signal_tp10, full_filtered, fs, sos, all_troughs, half_cycle_samples)


if __name__ == "__main__":
    main()
