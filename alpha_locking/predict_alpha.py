"""
Predict Alpha - Train a 1D CNN to predict the next 0.25s of alpha-band EEG
from the previous 1.0s. Compares single-channel (TP10) vs dual-channel
(TP10 + TP9) performance.

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---- Constants ----
FS_NATIVE = 256           # Hz (Muse sample rate)
FS_SUB = 64               # Subsampled rate
SUBSAMPLE = 4             # FS_NATIVE // FS_SUB
BAND = (8, 12)            # Alpha band Hz
INPUT_LEN = 64            # 1.0s at 64 Hz
OUTPUT_LEN = 16            # 0.25s at 64 Hz
STRIDE = 16               # Window stride in native samples
TRAIN_FRAC = 0.80
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
PATIENCE = 3              # Stop after this many epochs of rising test loss
CHANNEL_TP10 = 3
CHANNEL_TP9 = 0


# ---- Recording loading (matches alpha_lock_sample.py pattern) ----

def list_recordings(recordings_dir="../recordings"):
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


def build_windows(tp10_filtered, tp9_filtered):
    """
    Build sliding windows from filtered signals.

    Returns:
        X_single: (N, 1, INPUT_LEN) - TP10 only
        X_dual:   (N, 2, INPUT_LEN) - TP10 + TP9
        Y:        (N, OUTPUT_LEN)   - next 0.25s of TP10
    """
    n_samples = len(tp10_filtered)
    # Each window needs INPUT_LEN*SUBSAMPLE input + OUTPUT_LEN*SUBSAMPLE output native samples
    input_native = INPUT_LEN * SUBSAMPLE    # 256
    output_native = OUTPUT_LEN * SUBSAMPLE  # 64
    window_native = input_native + output_native  # 320

    X_tp10 = []
    X_tp9 = []
    Y = []

    start = 0
    while start + window_native <= n_samples:
        inp_tp10 = tp10_filtered[start:start + input_native:SUBSAMPLE]
        inp_tp9 = tp9_filtered[start:start + input_native:SUBSAMPLE]
        out_tp10 = tp10_filtered[start + input_native:start + window_native:SUBSAMPLE]

        X_tp10.append(inp_tp10)
        X_tp9.append(inp_tp9)
        Y.append(out_tp10)

        start += STRIDE

    X_tp10 = np.array(X_tp10, dtype=np.float32)
    X_tp9 = np.array(X_tp9, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # (N, 1, INPUT_LEN) and (N, 2, INPUT_LEN)
    X_single = X_tp10[:, np.newaxis, :]
    X_dual = np.stack([X_tp10, X_tp9], axis=1)

    return X_single, X_dual, Y


# ---- CNN model ----

class AlphaCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # After conv: (batch, 64, 64) -> flatten -> 4096
        self.fc = nn.Sequential(
            nn.Linear(64 * INPUT_LEN, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_LEN),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ---- Training ----

def train_model(model, train_loader, test_X, test_Y, device, label=""):
    """Train model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.to(device)
    test_X_t = torch.tensor(test_X, dtype=torch.float32, device=device)
    test_Y_t = torch.tensor(test_Y, dtype=torch.float32, device=device)

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            test_pred = model(test_X_t)
            test_loss = criterion(test_pred, test_Y_t).item()

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"  [{label}] Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={avg_train_loss:.6f}  test_loss={test_loss:.6f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"  [{label}] Early stop at epoch {epoch} "
                  f"(test loss did not improve for {PATIENCE} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, test_losses


# ---- Evaluation ----

def evaluate(model, test_X, test_Y, device):
    """Compute MSE and mean Pearson correlation on test set."""
    model.eval()
    test_X_t = torch.tensor(test_X, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(test_X_t).cpu().numpy()

    mse = np.mean((preds - test_Y) ** 2)

    correlations = []
    for i in range(len(test_Y)):
        y_true = test_Y[i]
        y_pred = preds[i]
        if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
            r = np.corrcoef(y_true, y_pred)[0, 1]
            correlations.append(r)

    mean_corr = np.mean(correlations) if correlations else 0.0

    return mse, mean_corr, preds


# ---- Plotting ----

def plot_results(test_Y, preds_single, preds_dual,
                 train_losses_s, test_losses_s,
                 train_losses_d, test_losses_d,
                 mse_s, corr_s, mse_d, corr_d):
    """3-row figure: example predictions, loss curves, bar chart."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Alpha Prediction: Single-Channel vs Dual-Channel CNN",
                 fontsize=14, fontweight="bold")

    t_out = np.arange(OUTPUT_LEN) / FS_SUB * 1000  # ms

    # Pick 4 evenly spaced examples from the test set
    n_test = len(test_Y)
    example_idx = np.linspace(0, n_test - 1, 4, dtype=int)

    # Row 1: Single-channel predictions
    for col, idx in enumerate(example_idx):
        ax = fig.add_subplot(3, 4, col + 1)
        ax.plot(t_out, test_Y[idx], "k-", lw=1.5, label="Actual")
        ax.plot(t_out, preds_single[idx], "b--", lw=1.2, label="Predicted")
        r = np.corrcoef(test_Y[idx], preds_single[idx])[0, 1] \
            if np.std(test_Y[idx]) > 1e-8 and np.std(preds_single[idx]) > 1e-8 else 0.0
        ax.set_title(f"Single #{idx}  r={r:.2f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=7)
        ax.set_xlabel("ms") if col == 0 else None
        ax.tick_params(labelsize=7)

    # Row 2: Dual-channel predictions
    for col, idx in enumerate(example_idx):
        ax = fig.add_subplot(3, 4, col + 5)
        ax.plot(t_out, test_Y[idx], "k-", lw=1.5, label="Actual")
        ax.plot(t_out, preds_dual[idx], "r--", lw=1.2, label="Predicted")
        r = np.corrcoef(test_Y[idx], preds_dual[idx])[0, 1] \
            if np.std(test_Y[idx]) > 1e-8 and np.std(preds_dual[idx]) > 1e-8 else 0.0
        ax.set_title(f"Dual #{idx}  r={r:.2f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=7)
        ax.set_xlabel("ms") if col == 0 else None
        ax.tick_params(labelsize=7)

    # Row 3, left: Loss curves
    ax_loss = fig.add_subplot(3, 2, 5)
    epochs_s = range(1, len(train_losses_s) + 1)
    epochs_d = range(1, len(train_losses_d) + 1)
    ax_loss.plot(epochs_s, train_losses_s, "b-", alpha=0.7, label="Single train")
    ax_loss.plot(epochs_s, test_losses_s, "b--", alpha=0.7, label="Single test")
    ax_loss.plot(epochs_d, train_losses_d, "r-", alpha=0.7, label="Dual train")
    ax_loss.plot(epochs_d, test_losses_d, "r--", alpha=0.7, label="Dual test")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.set_title("Training Curves")
    ax_loss.legend(fontsize=8)
    ax_loss.set_yscale("log")

    # Row 3, right: Bar chart comparing MSE and correlation
    ax_bar = fig.add_subplot(3, 2, 6)
    x = np.arange(2)
    width = 0.3

    ax_bar.bar(x - width / 2, [mse_s, corr_s], width, label="Single", color="steelblue")
    ax_bar.bar(x + width / 2, [mse_d, corr_d], width, label="Dual", color="indianred")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["MSE (lower=better)", "Correlation (higher=better)"])
    ax_bar.set_title("Model Comparison")
    ax_bar.legend(fontsize=8)

    # Add value labels on bars
    for i, (vs, vd) in enumerate([(mse_s, mse_d), (corr_s, corr_d)]):
        ax_bar.text(i - width / 2, vs + 0.01, f"{vs:.4f}", ha="center", fontsize=8)
        ax_bar.text(i + width / 2, vd + 0.01, f"{vd:.4f}", ha="center", fontsize=8)

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

    # Extract and bandpass filter
    tp10_raw = data[CHANNEL_TP10].astype(np.float64)
    tp9_raw = data[CHANNEL_TP9].astype(np.float64)

    tp10_filt = bandpass_filter(tp10_raw, BAND[0], BAND[1], fs)
    tp9_filt = bandpass_filter(tp9_raw, BAND[0], BAND[1], fs)

    # Build windows
    X_single, X_dual, Y = build_windows(tp10_filt, tp9_filt)
    n_windows = len(Y)
    n_train = int(n_windows * TRAIN_FRAC)
    n_test = n_windows - n_train
    print(f"Windows: {n_windows} total, {n_train} train, {n_test} test  "
          f"(stride={STRIDE} native samples)")

    # Temporal split
    X_single_train, X_single_test = X_single[:n_train], X_single[n_train:]
    X_dual_train, X_dual_test = X_dual[:n_train], X_dual[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    # Z-score normalize using train stats only
    # Single channel: compute over (N_train, 1, INPUT_LEN)
    single_mean = X_single_train.mean()
    single_std = X_single_train.std()
    single_std = single_std if single_std > 1e-8 else 1.0
    X_single_train = (X_single_train - single_mean) / single_std
    X_single_test = (X_single_test - single_mean) / single_std

    # Dual channel: per-channel stats from train
    dual_mean = X_dual_train.mean(axis=(0, 2), keepdims=True)
    dual_std = X_dual_train.std(axis=(0, 2), keepdims=True)
    dual_std = np.where(dual_std > 1e-8, dual_std, 1.0)
    X_dual_train = (X_dual_train - dual_mean) / dual_std
    X_dual_test = (X_dual_test - dual_mean) / dual_std

    # Normalize Y using train stats
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    y_std = y_std if y_std > 1e-8 else 1.0
    Y_train_norm = (Y_train - y_mean) / y_std
    Y_test_norm = (Y_test - y_mean) / y_std

    # DataLoaders
    train_ds_single = TensorDataset(
        torch.tensor(X_single_train), torch.tensor(Y_train_norm))
    train_ds_dual = TensorDataset(
        torch.tensor(X_dual_train), torch.tensor(Y_train_norm))

    train_loader_single = DataLoader(train_ds_single, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_dual = DataLoader(train_ds_dual, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Train single-channel model ---
    print("=" * 50)
    print("  Training Single-Channel Model (TP10)")
    print("=" * 50)
    model_single = AlphaCNN(in_channels=1)
    train_losses_s, test_losses_s = train_model(
        model_single, train_loader_single, X_single_test, Y_test_norm, device,
        label="Single")

    # --- Train dual-channel model ---
    print()
    print("=" * 50)
    print("  Training Dual-Channel Model (TP10 + TP9)")
    print("=" * 50)
    model_dual = AlphaCNN(in_channels=2)
    train_losses_d, test_losses_d = train_model(
        model_dual, train_loader_dual, X_dual_test, Y_test_norm, device,
        label="Dual")

    # --- Evaluate ---
    mse_s, corr_s, preds_single = evaluate(model_single, X_single_test, Y_test_norm, device)
    mse_d, corr_d, preds_dual = evaluate(model_dual, X_dual_test, Y_test_norm, device)

    # --- Print comparison ---
    print()
    print("=" * 60)
    print("  Results Comparison")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'Single (TP10)':>15s} {'Dual (TP10+TP9)':>15s} {'Improvement':>12s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    mse_improvement = (mse_s - mse_d) / mse_s * 100 if mse_s > 0 else 0
    corr_improvement = (corr_d - corr_s) / abs(corr_s) * 100 if abs(corr_s) > 1e-8 else 0

    print(f"  {'Test MSE':<25s} {mse_s:>15.6f} {mse_d:>15.6f} {mse_improvement:>+11.1f}%")
    print(f"  {'Mean Pearson r':<25s} {corr_s:>15.4f} {corr_d:>15.4f} {corr_improvement:>+11.1f}%")
    print()

    if mse_d < mse_s:
        print("  Dual-channel model has LOWER MSE (better).")
    else:
        print("  Single-channel model has LOWER MSE (better).")

    if corr_d > corr_s:
        print("  Dual-channel model has HIGHER correlation (better).")
    else:
        print("  Single-channel model has HIGHER correlation (better).")

    # --- Plot ---
    plot_results(Y_test_norm, preds_single, preds_dual,
                 train_losses_s, test_losses_s,
                 train_losses_d, test_losses_d,
                 mse_s, corr_s, mse_d, corr_d)
    plt.show()


if __name__ == "__main__":
    main()
