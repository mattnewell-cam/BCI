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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- Parameters (must match lock_live_2) ---
BAND_LO = 7
BAND_HI = 12
FILTER_ORDER = 4
FIT_INTERVAL_S = 0.100
CHANNEL = 3  # TP10

# --- Tunable ---
LOOKBACK_S = 1.800       # seconds of signal to bandpass each round
N_GAPS = 1               # number of trough-trough gaps to average (gap-avg method)

# --- Linear regression ---
LR_N_GAPS = 10           # number of gap features for the regression
TRAIN_FRAC = 0.70


def load_signal():
    """Load sample and return signal, fs, filter sos, ground-truth troughs."""
    sample_path = os.path.join(os.path.dirname(__file__), "..",
                               "recordings", "full_night_350_1000.npz")
    rec = np.load(sample_path, allow_pickle=True)
    data = rec["data"]
    fs = int(rec["sample_rate"])
    print(f"Loaded: {data.shape[1]} samples, fs={fs}, "
          f"duration={data.shape[1]/fs:.1f}s, channels={data.shape[0]}")

    signal = data[CHANNEL].astype(np.float64)

    sos = butter(FILTER_ORDER, [BAND_LO, BAND_HI], btype="band",
                 fs=fs, output="sos")

    full_filtered = sosfilt(sos, signal)
    half_cycle_samples = max(3, int(fs / BAND_HI / 2))
    all_troughs = argrelmin(full_filtered, order=half_cycle_samples)[0]
    print(f"Ground truth: {len(all_troughs)} troughs found")

    return signal, fs, sos, all_troughs, half_cycle_samples


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


def test_linear_regression(signal, fs, sos, all_troughs, half_cycle_samples):
    """Linear regression on last LR_N_GAPS gaps + amplitude -> next gap."""
    features, targets, meta = collect_features(signal, fs, sos, half_cycle_samples)
    total_samples = len(signal)
    n_total = len(features)
    n_train = int(n_total * TRAIN_FRAC)

    print(f"\nLinReg dataset: {n_total} rows, {features.shape[1]} features, "
          f"train={n_train}, test={n_total - n_train}")

    # Normalise (fit on train only)
    feat_mean = features[:n_train].mean(axis=0)
    feat_std = features[:n_train].std(axis=0)
    feat_std[feat_std == 0] = 1.0

    X = (features - feat_mean) / feat_std
    X = np.column_stack([X, np.ones(n_total)])  # bias term

    # Fit OLS on train split
    X_train, y_train = X[:n_train], targets[:n_train]
    w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    y_pred = X[n_train:] @ w

    errors_ms = evaluate_predictions(y_pred, meta, n_train, fs, all_troughs, total_samples)

    report(f"Linear Regression  (LR_N_GAPS={LR_N_GAPS}, "
           f"train={n_train}, test={n_total - n_train})",
           errors_ms, n_total - n_train)


def collect_features(signal, fs, sos, half_cycle_samples):
    """Collect feature/target/meta rows shared by linreg and NN."""
    lookback_samples = int(LOOKBACK_S * fs)
    fit_interval_samples = int(FIT_INTERVAL_S * fs)
    total_samples = len(signal)
    n_troughs_needed = LR_N_GAPS + 1

    features = []
    targets = []
    meta = []

    cursor = lookback_samples
    while cursor < total_samples:
        window = signal[cursor - lookback_samples: cursor]
        filtered = sosfilt(sos, window)
        trough_indices = argrelmin(filtered, order=half_cycle_samples)[0]

        if len(trough_indices) >= n_troughs_needed + 1:
            last_n_plus = trough_indices[-(n_troughs_needed + 1):]
            all_gaps = np.diff(last_n_plus)
            gap_features = all_gaps[:-1].astype(np.float64)
            target_gap = float(all_gaps[-1])

            last_tr = trough_indices[-1]
            prev_tr = trough_indices[-2]
            segment = filtered[prev_tr:last_tr + 1]
            amplitude = float(segment.max() - segment.min())

            last_trough_sample = last_n_plus[-1]
            samples_since = lookback_samples - last_trough_sample

            features.append(np.append(gap_features, amplitude))
            targets.append(target_gap)
            meta.append((cursor, last_trough_sample, samples_since))

        cursor += fit_interval_samples

    return np.array(features), np.array(targets), meta


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


def test_neural_network(signal, fs, sos, all_troughs, half_cycle_samples):
    """MLP neural network on last LR_N_GAPS gaps + amplitude -> next gap."""
    features, targets, meta = collect_features(signal, fs, sos, half_cycle_samples)
    total_samples = len(signal)
    n_total = len(features)
    n_train = int(n_total * TRAIN_FRAC)

    # Normalise (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[:n_train])
    X_test = scaler.transform(features[n_train:])
    y_train = targets[:n_train]

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    best_loss = mlp.best_loss_ if mlp.best_loss_ is not None else mlp.loss_
    print(f"\nMLP: {n_total} rows, train={n_train}, test={n_total - n_train}, "
          f"iters={mlp.n_iter_}, loss={best_loss:.4f}")

    errors_ms = evaluate_predictions(y_pred, meta, n_train, fs, all_troughs, total_samples)

    report(f"Neural Network  (LR_N_GAPS={LR_N_GAPS}, "
           f"layers=(64,32), train={n_train}, test={n_total - n_train})",
           errors_ms, n_total - n_train)


def main():
    signal, fs, sos, all_troughs, half_cycle_samples = load_signal()

    test_gap_average(signal, fs, sos, all_troughs, half_cycle_samples)
    test_linear_regression(signal, fs, sos, all_troughs, half_cycle_samples)
    test_neural_network(signal, fs, sos, all_troughs, half_cycle_samples)


if __name__ == "__main__":
    main()
