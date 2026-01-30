"""
Test Correlation - Inter-channel correlation heatmap across frequency sub-bands.

Loads a recording and displays a heatmap of correlation between all EEG channel
pairs for each 0.5 Hz sub-band within the specified frequency range.

Usage:
    python test_correlation.py                   # interactive selection
    python test_correlation.py recording.npz     # specify recording
"""

import sys
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

# ---- Configuration ----
FREQ_RANGE = (7, 12)   # Hz
BIN_WIDTH = 0.5         # Hz

CHANNEL_LABELS = ["AF7", "AF8", "TP9", "TP10"]


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


def load_recording(filepath):
    """Load a recording from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    return {
        "data": data["data"],
        "timestamps": data.get("timestamps", None),
        "sample_rate": float(data["sample_rate"]),
        "channel_labels": list(data.get("channel_labels", [])),
        "recording_name": str(data.get("recording_name", "")),
        "duration_seconds": float(data.get("duration_seconds", 0)),
    }


def bandpass_filter(data, lo, hi, fs, order=4):
    """Apply bandpass filter using SOS for stability."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def compute_correlation_matrix(recording):
    """Compute inter-channel correlation for each frequency sub-band."""
    fs = recording["sample_rate"]
    eeg = recording["data"]
    n_channels = eeg.shape[0]

    pairs = list(combinations(range(n_channels), 2))
    labels = recording["channel_labels"] if recording["channel_labels"] else CHANNEL_LABELS
    pair_labels = [f"{labels[a]}-{labels[b]}" for a, b in pairs]

    # Build frequency bins
    bin_edges = np.arange(FREQ_RANGE[0], FREQ_RANGE[1] + BIN_WIDTH / 2, BIN_WIDTH)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]
    n_bins = len(bin_labels)

    corr_matrix = np.zeros((len(pairs), n_bins))

    for bi in range(n_bins):
        lo = bin_edges[bi]
        hi = bin_edges[bi + 1]

        # Filter each channel into this sub-band
        filtered = np.array([bandpass_filter(eeg[ch], lo, hi, fs) for ch in range(n_channels)])

        for pi, (a, b) in enumerate(pairs):
            r = np.corrcoef(filtered[a], filtered[b])[0, 1]
            corr_matrix[pi, bi] = r

    return corr_matrix, pair_labels, bin_labels


def print_correlation(corr_matrix, pair_labels, bin_labels):
    """Print the correlation matrix to the terminal."""
    # Header
    col_width = 10
    header = f"{'Pair':<12}" + "".join(f"{bl:>{col_width}}" for bl in bin_labels)
    print(f"\nInter-channel correlation ({FREQ_RANGE[0]}-{FREQ_RANGE[1]} Hz, {BIN_WIDTH} Hz bins):")
    print(header)
    print("-" * len(header))

    for i, label in enumerate(pair_labels):
        row = f"{label:<12}" + "".join(f"{corr_matrix[i, j]:>{col_width}.4f}" for j in range(len(bin_labels)))
        print(row)


def plot_correlation(corr_matrix, pair_labels, bin_labels, recording_name):
    """Display correlation heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, len(bin_labels) * 1.2), 5))

    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels)

    ax.set_xlabel("Frequency bin (Hz)")
    ax.set_ylabel("Channel pair")
    ax.set_title(f"Inter-channel Correlation: {recording_name}\n({FREQ_RANGE[0]}-{FREQ_RANGE[1]} Hz, {BIN_WIDTH} Hz bins)")

    # Annotate cells
    for i in range(len(pair_labels)):
        for j in range(len(bin_labels)):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig


def main():
    # Select recording
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        recordings = list_recordings()
        if not recordings:
            print("\nUsage: python test_correlation.py <recording.npz>")
            return
        print("\nEnter recording number (or path to .npz file):")
        choice = input("> ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(recordings):
                filepath = recordings[idx]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            filepath = choice

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Frequency range: {FREQ_RANGE[0]}-{FREQ_RANGE[1]} Hz")
    print(f"Bin width: {BIN_WIDTH} Hz")

    recording = load_recording(filepath)
    corr_matrix, pair_labels, bin_labels = compute_correlation_matrix(recording)

    print_correlation(corr_matrix, pair_labels, bin_labels)
    plot_correlation(corr_matrix, pair_labels, bin_labels, recording["recording_name"])
    plt.show()


if __name__ == "__main__":
    main()
