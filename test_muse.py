"""
Muse EEG Contact Quality Checker

Tests that all electrode contacts are working by analyzing signal characteristics:
- Real EEG has 1/f spectral slope (power decreases with frequency)
- Disconnected electrodes show flat spectrum or excessive noise
- Good contacts show appropriate amplitude and frequency content

Visual indicators:
- GREEN: Good contact, looks like real EEG
- YELLOW: Marginal contact, signal quality questionable
- RED: Poor/no contact, noise or flat signal
"""

import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import welch, butter, sosfilt, sosfilt_zi

from pylsl import StreamInlet
from utils import (
    find_eeg_stream,
    get_channel_labels,
    assess_channel_quality,
    get_quality_color,
)


# Muse channel positions (approximate 10-20 layout for display)
CHANNEL_POSITIONS = {
    "TP9": (-0.8, -0.3),   # Left ear
    "AF7": (-0.4, 0.6),    # Left forehead
    "AF8": (0.4, 0.6),     # Right forehead
    "TP10": (0.8, -0.3),   # Right ear
}


def main():
    # Connect to EEG stream
    print("Searching for EEG stream...")
    s = find_eeg_stream()
    # Small buffer to prevent lag accumulation
    inlet_eeg = StreamInlet(s, max_buflen=2)

    info = inlet_eeg.info()
    fs = float(info.nominal_srate())
    n_channels = int(info.channel_count())
    stream_name = info.name()

    ch_labels = get_channel_labels(info, n_channels)
    print(f"Connected: {stream_name}")
    print(f"Sample rate: {fs:.1f} Hz, Channels: {n_channels}")
    print(f"Channel labels: {', '.join(ch_labels)}")

    # Use first 4 channels (Muse EEG channels)
    n_eeg = min(4, n_channels)

    # Rolling buffer for each channel (2 seconds of data)
    window_s = 2.0
    buf_len = int(fs * window_s)
    buffers = [deque(maxlen=buf_len) for _ in range(n_eeg)]

    # Quality history for smoothing (30 frames @ 200ms = 6 seconds)
    quality_history = [deque(maxlen=15) for _ in range(n_eeg)]

    # Notch filter for 50Hz (common in many regions)
    notch_freq = 50.0
    notch_bw = 2.0
    low = notch_freq - notch_bw
    high = notch_freq + notch_bw
    sos_notch = butter(2, [low, high], btype="bandstop", fs=fs, output="sos")
    notch_states = [sosfilt_zi(sos_notch) * 0.0 for _ in range(n_eeg)]

    # Setup matplotlib figure
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Muse EEG Contact Quality Check", fontsize=14, fontweight="bold")

    # Create grid: head diagram on left, waveforms on right, PSD at bottom
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.5], width_ratios=[1, 2],
                          hspace=0.3, wspace=0.3)

    # Head diagram axis
    ax_head = fig.add_subplot(gs[0:2, 0])
    ax_head.set_xlim(-1.2, 1.2)
    ax_head.set_ylim(-1.0, 1.2)
    ax_head.set_aspect("equal")
    ax_head.axis("off")
    ax_head.set_title("Contact Quality", fontsize=12)

    # Draw head outline
    head_circle = plt.Circle((0, 0), 0.9, fill=False, linewidth=2, color="gray")
    ax_head.add_patch(head_circle)
    # Nose indicator
    ax_head.plot([0, 0], [0.9, 1.05], color="gray", linewidth=2)
    # Ears
    ax_head.plot([-0.95, -1.05], [-0.1, -0.1], color="gray", linewidth=2)
    ax_head.plot([0.95, 1.05], [-0.1, -0.1], color="gray", linewidth=2)

    # Create electrode circles and labels
    electrode_circles = {}
    electrode_texts = {}
    quality_texts = {}

    for i, label in enumerate(ch_labels[:n_eeg]):
        # Try to match known positions, otherwise place in grid
        if label in CHANNEL_POSITIONS:
            x, y = CHANNEL_POSITIONS[label]
        else:
            # Default grid placement
            x = -0.5 + (i % 2) * 1.0
            y = 0.3 - (i // 2) * 0.6

        circle = plt.Circle((x, y), 0.15, color="#e74c3c", ec="black", linewidth=2)
        ax_head.add_patch(circle)
        electrode_circles[i] = circle

        # Channel label
        txt = ax_head.text(x, y + 0.02, label, ha="center", va="center",
                          fontsize=10, fontweight="bold", color="white")
        electrode_texts[i] = txt

        # Quality percentage below
        qtxt = ax_head.text(x, y - 0.28, "0%", ha="center", va="top",
                           fontsize=9, color="black")
        quality_texts[i] = qtxt

    # Waveform axes (one per channel)
    ax_waves = []
    for i in range(n_eeg):
        ax = fig.add_subplot(gs[i // 2, 1] if n_eeg <= 2 else gs[i // 2, 1])
        if i < 2:
            ax = fig.add_subplot(gs[0, 1])
            break

    # Actually create 4 subplots for waveforms
    gs_waves = gs[0:2, 1].subgridspec(n_eeg, 1, hspace=0.1)
    ax_waves = [fig.add_subplot(gs_waves[i]) for i in range(n_eeg)]

    wave_lines = []
    for i, ax in enumerate(ax_waves):
        line, = ax.plot([], [], lw=0.8, color="#3498db")
        wave_lines.append(line)
        ax.set_ylabel(ch_labels[i] if i < len(ch_labels) else f"Ch{i}", fontsize=9)
        ax.set_xlim(-window_s, 0)
        ax.set_ylim(-100, 100)
        ax.grid(True, alpha=0.3)
        if i < n_eeg - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")

    # PSD axis
    ax_psd = fig.add_subplot(gs[2, :])
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power (dB)")
    ax_psd.set_xlim(0, 60)
    ax_psd.set_ylim(-40, 20)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_title("Power Spectral Density", fontsize=10)

    psd_lines = []
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for i in range(n_eeg):
        line, = ax_psd.plot([], [], lw=1.2, color=colors[i % len(colors)],
                           label=ch_labels[i] if i < len(ch_labels) else f"Ch{i}")
        psd_lines.append(line)
    ax_psd.legend(loc="upper right", fontsize=8)

    # Add 1/f reference line
    ref_freqs = np.linspace(1, 60, 100)
    ref_power = -10 - 10 * np.log10(ref_freqs)  # 1/f reference
    ax_psd.plot(ref_freqs, ref_power, "--", color="gray", alpha=0.5, label="1/f ref")

    # Status text
    status_text = fig.text(0.02, 0.02, "", fontsize=10, va="bottom",
                          family="monospace")

    # Animation update function
    def update(_frame):
        nonlocal notch_states

        # Pull ALL available samples to prevent lag accumulation
        # Keep pulling until buffer is drained
        while True:
            chunk, _ = inlet_eeg.pull_chunk(timeout=0.0, max_samples=256)
            if not chunk:
                break
            chunk = np.array(chunk)
            for i in range(n_eeg):
                if i < chunk.shape[1]:
                    # Apply notch filter
                    filtered, notch_states[i] = sosfilt(
                        sos_notch, chunk[:, i], zi=notch_states[i]
                    )
                    buffers[i].extend(filtered)

        # Check if we have enough data
        if any(len(buf) < buf_len // 2 for buf in buffers):
            return wave_lines + psd_lines + list(electrode_circles.values())

        status_lines = []

        for i in range(n_eeg):
            data = np.array(buffers[i])

            # Assess quality
            quality, status, metrics = assess_channel_quality(data, fs)
            quality_history[i].append(quality)

            # Smoothed quality
            smooth_quality = np.mean(quality_history[i])

            # Update electrode circle color (based on smoothed quality)
            color = get_quality_color(smooth_quality)
            electrode_circles[i].set_facecolor(color)
            quality_texts[i].set_text(f"{int(smooth_quality * 100)}%")

            # Update waveform
            t = np.linspace(-len(data) / fs, 0, len(data))
            wave_lines[i].set_data(t, data)

            # Auto-scale y-axis based on signal
            std = np.std(data)
            ylim = max(50, min(500, std * 4))
            ax_waves[i].set_ylim(-ylim, ylim)

            # Set waveform color based on quality
            wave_lines[i].set_color(color)

            # Update PSD
            if len(data) >= fs:
                freqs, psd = welch(data, fs=fs, nperseg=min(len(data), int(fs * 2)))
                psd_db = 10 * np.log10(psd + 1e-12)
                psd_lines[i].set_data(freqs, psd_db)

            # Build status line (derive status from smoothed quality)
            label = ch_labels[i] if i < len(ch_labels) else f"Ch{i}"
            smooth_status = "GOOD" if smooth_quality >= 0.7 else "MARGINAL" if smooth_quality >= 0.4 else "POOR"
            slope_str = f"{metrics.get('slope', 0):.2f}" if 'slope' in metrics else "N/A"
            var_str = f"{metrics.get('variance', 0):.1f}"
            status_lines.append(
                f"{label}: {smooth_status:8s} (q={smooth_quality:.0%}, "
                f"slope={slope_str}, var={var_str})"
            )

        # Update status text
        status_text.set_text("\n".join(status_lines))

        return wave_lines + psd_lines + list(electrode_circles.values())

    # Run animation (200ms interval - slower updates but no lag)
    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    print("\nContact Quality Indicators:")
    print("  GREEN  = Good contact, signal looks like real EEG")
    print("  YELLOW = Marginal contact, may need adjustment")
    print("  RED    = Poor/no contact, check electrode placement")
    print("\nKey metrics:")
    print("  - Spectral slope: Real EEG has 1/f characteristic (slope < -0.8)")
    print("  - Variance: Too low = no contact, too high = artifacts")
    print("  - High-freq ratio: Excessive = noise/movement")
    print("\nClose the window to exit.")

    plt.show()


if __name__ == "__main__":
    main()
