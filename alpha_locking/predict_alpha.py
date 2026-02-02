"""
Predict Alpha - Linear regression to predict the next 0.25s of alpha-band EEG
from the previous 1.0s. Tests multiple channel combinations to find the best
predictor set and visualizes learned weights.

Usage:
    python alpha_locking/predict_alpha.py                          # interactive selection
    python alpha_locking/predict_alpha.py recordings/recording.npz # specify recording
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt

# ---- Constants ----
FS_NATIVE = 256           # Hz (Muse sample rate)
FS_SUB = 64               # Subsampled rate
SUBSAMPLE = 4             # FS_NATIVE // FS_SUB
BAND = (8, 12)            # Alpha band Hz
INPUT_LEN = 64            # 1.0s at 64 Hz
OUTPUT_LEN = 16           # 0.25s at 64 Hz
STRIDE = 16               # Window stride in native samples
TRAIN_FRAC = 0.80

# Muse channel order: TP9=0, AF7=1, AF8=2, TP10=3
CHANNEL_NAMES = {0: "TP9", 1: "AF7", 2: "AF8", 3: "TP10"}
TARGET_CH = 3             # Always predict TP10

# Channel combos to test: (label, channel indices)
COMBOS = [
    ("TP10",                [3]),
    ("TP10 + TP9",          [3, 0]),
    ("TP10 + AF8",          [3, 2]),
    ("TP10 + AF7",          [3, 1]),
    ("TP10 + TP9 + AF8",    [3, 0, 2]),
    ("TP10 + TP9 + AF7",    [3, 0, 1]),
    ("All 4",               [3, 0, 1, 2]),
]


# ---- Recording loading (matches alpha_lock_sample.py pattern) ----

def list_recordings(recordings_dir="recordings"):
    """List available recordings."""
    path = Path(recordings_dir)
    if not path.exists():
        print(f"No recordings directory found at: {recordings_dir}")
        return []

    recordings = sorted(path.glob("*.npz"))
    if not recordings:
        print(f"No .npz files found in: {recordings_dir}")
        return []

    print(f"\nAvailable recordings in {recordings_dir}/:")
    for i, rec in enumerate(recordings):
        try:
            data = np.load(rec)
            duration = float(data.get("duration_seconds", 0))
            fs = float(data.get("sample_rate", 0))
            name = str(data.get("recording_name", ""))
            print(f"  {i + 1}. {rec.name}")
            print(f"       Name: {name}, Duration: {duration:.1f}s, Sample rate: {fs}Hz")
        except Exception as e:
            print(f"  {i + 1}. {rec.name} (error loading: {e})")

    return recordings


def select_recording():
    """Interactive recording selection, returns filepath or None."""
    recordings = list_recordings()
    if not recordings:
        print("\nUsage: python alpha_locking/predict_alpha.py <recording.npz>")
        return None

    print("\nEnter recording number (or path to .npz file):")
    choice = input("> ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(recordings):
            return recordings[idx]
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        return choice


# ---- Data pipeline ----

def bandpass_filter(signal, lo, hi, fs, order=4):
    """Apply bandpass filter using SOS for stability."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, signal)


def build_windows(filtered_channels, target_filtered):
    """
    Build sliding windows from filtered signals.

    Args:
        filtered_channels: list of 1D arrays, one per input channel
        target_filtered: 1D array for the target channel (TP10)

    Returns:
        X: (N, C, INPUT_LEN)
        Y: (N, OUTPUT_LEN)
    """
    n_samples = len(target_filtered)
    input_native = INPUT_LEN * SUBSAMPLE
    output_native = OUTPUT_LEN * SUBSAMPLE
    window_native = input_native + output_native

    X_lists = [[] for _ in filtered_channels]
    Y = []

    start = 0
    while start + window_native <= n_samples:
        for i, ch in enumerate(filtered_channels):
            X_lists[i].append(ch[start:start + input_native:SUBSAMPLE])
        Y.append(target_filtered[start + input_native:start + window_native:SUBSAMPLE])
        start += STRIDE

    X_arrs = [np.array(xl, dtype=np.float32) for xl in X_lists]
    Y = np.array(Y, dtype=np.float32)

    X = np.stack(X_arrs, axis=1)  # (N, C, INPUT_LEN)
    return X, Y


# ---- Linear regression ----

def train_linear_regression(X_train, Y_train, X_test):
    """
    OLS linear regression. X shape: (N, C, INPUT_LEN), Y shape: (N, OUTPUT_LEN).

    Returns:
        preds: (N_test, OUTPUT_LEN)
        W: weight matrix (n_features + 1, OUTPUT_LEN) — last row is bias
    """
    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_te_flat = X_test.reshape(len(X_test), -1)

    X_tr = np.column_stack([X_tr_flat, np.ones(len(X_tr_flat))])
    X_te = np.column_stack([X_te_flat, np.ones(len(X_te_flat))])

    W = np.linalg.lstsq(X_tr, Y_train, rcond=None)[0]
    preds = X_te @ W

    return preds, W


# ---- Evaluation ----

def eval_predictions(preds, test_Y):
    """Compute MSE and mean Pearson correlation from numpy arrays."""
    mse = np.mean((preds - test_Y) ** 2)

    correlations = []
    for i in range(len(test_Y)):
        y_true = test_Y[i]
        y_pred = preds[i]
        if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
            r = np.corrcoef(y_true, y_pred)[0, 1]
            correlations.append(r)

    mean_corr = np.mean(correlations) if correlations else 0.0
    return mse, mean_corr


# ---- Plotting ----

def plot_results(test_Y, results, best_key):
    """
    3-row figure:
      Row 1: 4 example predictions from single-channel (TP10 only)
      Row 2: 4 example predictions from best multi-channel combo
      Row 3: weight profile for best multi-channel + channel contribution bar + MSE/corr comparison
    """
    single = results[0]  # TP10 only is always first
    best = results[best_key]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Alpha Prediction: Channel Combination Comparison",
                 fontsize=14, fontweight="bold")

    t_out = np.arange(OUTPUT_LEN) / FS_SUB * 1000  # ms
    t_in = np.arange(INPUT_LEN) / FS_SUB * 1000    # ms

    n_test = len(test_Y)
    example_idx = np.linspace(0, n_test - 1, 4, dtype=int)

    # Row 1: Single-channel (TP10) predictions
    for col, idx in enumerate(example_idx):
        ax = fig.add_subplot(3, 4, col + 1)
        ax.plot(t_out, test_Y[idx], "k-", lw=1.5, label="Actual")
        ax.plot(t_out, single["preds"][idx], "b--", lw=1.2, label="Predicted")
        r = np.corrcoef(test_Y[idx], single["preds"][idx])[0, 1] \
            if np.std(test_Y[idx]) > 1e-8 and np.std(single["preds"][idx]) > 1e-8 else 0.0
        ax.set_title(f"TP10 only #{idx}  r={r:.2f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=7)
            ax.set_xlabel("ms")
        ax.tick_params(labelsize=7)

    # Row 2: Best multi-channel predictions
    for col, idx in enumerate(example_idx):
        ax = fig.add_subplot(3, 4, col + 5)
        ax.plot(t_out, test_Y[idx], "k-", lw=1.5, label="Actual")
        ax.plot(t_out, best["preds"][idx], "r--", lw=1.2, label="Predicted")
        r = np.corrcoef(test_Y[idx], best["preds"][idx])[0, 1] \
            if np.std(test_Y[idx]) > 1e-8 and np.std(best["preds"][idx]) > 1e-8 else 0.0
        ax.set_title(f"{best['label']} #{idx}  r={r:.2f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=7)
            ax.set_xlabel("ms")
        ax.tick_params(labelsize=7)

    # Row 3, left: Weight profile per channel for best combo
    W = best["W"]
    ch_names = [CHANNEL_NAMES[c] for c in best["channels"]]
    n_ch = len(ch_names)

    ax_wp = fig.add_subplot(3, 4, 9)
    colors = ["steelblue", "seagreen", "coral", "mediumpurple"]
    for i, name in enumerate(ch_names):
        w_ch = W[i * INPUT_LEN:(i + 1) * INPUT_LEN, :]
        w_avg = np.mean(np.abs(w_ch), axis=1)
        ax_wp.plot(t_in, w_avg, color=colors[i % len(colors)], lw=1.2, label=name)
    ax_wp.set_xlabel("Input time (ms)")
    ax_wp.set_ylabel("Mean |weight|")
    ax_wp.set_title(f"Weights: {best['label']}", fontsize=10)
    ax_wp.legend(fontsize=8)
    ax_wp.tick_params(labelsize=7)

    # Row 3, middle-left: Channel contribution bar chart for best combo
    ax_bar = fig.add_subplot(3, 4, 10)
    totals = []
    for i in range(n_ch):
        w_ch = W[i * INPUT_LEN:(i + 1) * INPUT_LEN, :]
        totals.append(np.sum(np.abs(w_ch)))
    grand_total = sum(totals)
    pcts = [t / grand_total * 100 for t in totals]

    bars = ax_bar.bar(ch_names, pcts, color=colors[:n_ch])
    ax_bar.set_ylabel("% of total |weight|")
    ax_bar.set_title("Channel contribution", fontsize=10)
    for bar, v in zip(bars, pcts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}%", ha="center", fontsize=9)
    ax_bar.set_ylim(0, max(pcts) * 1.2)
    ax_bar.tick_params(labelsize=7)

    # Row 3, middle-right: MSE bar chart for all combos
    ax_mse = fig.add_subplot(3, 4, 11)
    labels = [r["label"] for r in results]
    mses = [r["mse"] for r in results]
    bar_colors = ["steelblue" if i == 0 else "indianred" if i == best_key else "lightgray"
                  for i in range(len(results))]
    bars = ax_mse.barh(range(len(results)), mses, color=bar_colors)
    ax_mse.set_yticks(range(len(results)))
    ax_mse.set_yticklabels(labels, fontsize=7)
    ax_mse.set_xlabel("MSE")
    ax_mse.set_title("Test MSE (lower=better)", fontsize=10)
    for bar, v in zip(bars, mses):
        ax_mse.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=7)
    ax_mse.invert_yaxis()
    ax_mse.tick_params(labelsize=7)

    # Row 3, right: Correlation bar chart for all combos
    ax_corr = fig.add_subplot(3, 4, 12)
    corrs = [r["corr"] for r in results]
    bars = ax_corr.barh(range(len(results)), corrs, color=bar_colors)
    ax_corr.set_yticks(range(len(results)))
    ax_corr.set_yticklabels(labels, fontsize=7)
    ax_corr.set_xlabel("Pearson r")
    ax_corr.set_title("Mean Pearson r (higher=better)", fontsize=10)
    for bar, v in zip(bars, corrs):
        ax_corr.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{v:.4f}", va="center", fontsize=7)
    ax_corr.invert_yaxis()
    ax_corr.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---- Main ----

def main():
    # Select recording
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = select_recording()
        if filepath is None:
            return

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Load
    rec = np.load(filepath, allow_pickle=True)
    data = rec["data"]
    fs = int(rec["sample_rate"])
    print(f"Loaded: {data.shape[1]} samples, fs={fs}, "
          f"duration={data.shape[1]/fs:.1f}s, channels={data.shape[0]}")

    assert fs == FS_NATIVE, f"Expected {FS_NATIVE} Hz, got {fs}"

    # Bandpass filter all 4 channels
    filtered = {}
    for ch_idx, ch_name in CHANNEL_NAMES.items():
        raw = data[ch_idx].astype(np.float64)
        filtered[ch_idx] = bandpass_filter(raw, BAND[0], BAND[1], fs)

    target_filt = filtered[TARGET_CH]

    # Run each channel combination
    results = []

    for label, channels in COMBOS:
        ch_signals = [filtered[c] for c in channels]
        X, Y = build_windows(ch_signals, target_filt)

        n_windows = len(Y)
        n_train = int(n_windows * TRAIN_FRAC)

        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]

        # Z-score normalize per channel, fit on train only
        ch_mean = X_train.mean(axis=(0, 2), keepdims=True)
        ch_std = X_train.std(axis=(0, 2), keepdims=True)
        ch_std = np.where(ch_std > 1e-8, ch_std, 1.0)
        X_train = (X_train - ch_mean) / ch_std
        X_test = (X_test - ch_mean) / ch_std

        y_mean = Y_train.mean()
        y_std = Y_train.std()
        y_std = y_std if y_std > 1e-8 else 1.0
        Y_train_norm = (Y_train - y_mean) / y_std
        Y_test_norm = (Y_test - y_mean) / y_std

        preds, W = train_linear_regression(X_train, Y_train_norm, X_test)
        mse, corr = eval_predictions(preds, Y_test_norm)

        results.append({
            "label": label,
            "channels": channels,
            "mse": mse,
            "corr": corr,
            "preds": preds,
            "W": W,
        })

    # Print results table
    Y_test_norm_ref = results[0]["preds"]  # just need length — use any
    baseline_mse = results[0]["mse"]
    baseline_corr = results[0]["corr"]

    print(f"\nWindows: {n_windows} total, {n_train} train, {n_windows - n_train} test  "
          f"(stride={STRIDE} native samples)")
    print()
    print("=" * 75)
    print("  Results")
    print("=" * 75)
    print(f"  {'Channels':<25s} {'MSE':>10s} {'vs TP10':>10s} {'Pearson r':>10s} {'vs TP10':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        mse_delta = (baseline_mse - r["mse"]) / baseline_mse * 100 if baseline_mse > 0 else 0
        corr_delta = (r["corr"] - baseline_corr) / abs(baseline_corr) * 100 if abs(baseline_corr) > 1e-8 else 0
        print(f"  {r['label']:<25s} {r['mse']:>10.6f} {mse_delta:>+9.1f}% {r['corr']:>10.4f} {corr_delta:>+9.1f}%")

    # Best combo by MSE
    best_key = min(range(len(results)), key=lambda i: results[i]["mse"])
    best = results[best_key]
    print(f"\n  Best: {best['label']}  (MSE={best['mse']:.6f}, r={best['corr']:.4f})")

    # Weight analysis for best multi-channel combo
    if len(best["channels"]) > 1:
        W = best["W"]
        ch_names = [CHANNEL_NAMES[c] for c in best["channels"]]
        totals = []
        for i in range(len(ch_names)):
            w_ch = W[i * INPUT_LEN:(i + 1) * INPUT_LEN, :]
            totals.append(np.sum(np.abs(w_ch)))
        grand_total = sum(totals)
        print(f"\n  Weight share ({best['label']}):")
        for name, t in zip(ch_names, totals):
            print(f"    {name}: {t/grand_total*100:.1f}%")

    # Plot
    plot_results(Y_test_norm, results, best_key)
    plt.show()


if __name__ == "__main__":
    main()
