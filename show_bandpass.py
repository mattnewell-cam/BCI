"""
Show Bandpass - Compare raw EEG with two bandpass-filtered versions.

Displays three panels (raw + two bandpass ranges) with position/zoom sliders.
Prints power distribution in 0.5 Hz bins within each band.

Usage:
    python show_bandpass.py                   # interactive selection
    python show_bandpass.py recording.npz     # specify recording
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import butter, sosfilt, welch

# ---- Band ranges (edit these) ----
BAND1 = (5, 20)    # Hz
BAND2 = (7, 12)     # Hz
BIN_WIDTH = 0.5     # Hz per bin for power breakdown


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


def print_band_power(signal, fs, band, label):
    """Compute PSD via Welch and print power in 0.5 Hz bins across the band."""
    lo, hi = band
    # Use long segments for fine frequency resolution (~0.125 Hz)
    nperseg = min(len(signal), int(fs * 8))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    df = freqs[1] - freqs[0]

    print(f"\n  {label} ({lo}-{hi} Hz) power by {BIN_WIDTH} Hz bins  (df={df:.3f} Hz):")

    bin_lo = lo
    total = 0.0
    while bin_lo < hi:
        bin_hi = min(bin_lo + BIN_WIDTH, hi)
        mask = (freqs >= bin_lo) & (freqs < bin_hi)
        power = psd[mask].sum() * df
        total += power
        print(f"    {bin_lo:5.1f} - {bin_hi:5.1f} Hz : {power:.4g}")
        bin_lo = bin_hi

    print(f"    {'total':>13s}      : {total:.4g}")


def plot_bandpass(recording, channel=0):
    """Plot raw + two bandpass filtered signals with interactive scroll/zoom."""
    fs = recording["sample_rate"]
    raw = recording["data"][channel]
    n_samples = len(raw)
    t = np.arange(n_samples) / fs
    total_duration = t[-1]

    labels = recording["channel_labels"]
    ch_label = labels[channel] if channel < len(labels) else f"Ch{channel}"

    filt1 = bandpass_filter(raw, BAND1[0], BAND1[1], fs)
    filt2 = bandpass_filter(raw, BAND2[0], BAND2[1], fs)

    # Print power breakdown
    print(f"\nChannel: {ch_label}")
    print_band_power(filt1, fs, BAND1, "Band 1")
    print_band_power(filt2, fs, BAND2, "Band 2")

    default_window = min(5.0, total_duration)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Bandpass Comparison: {recording['recording_name']} ({ch_label})",
        fontsize=12, fontweight="bold",
    )

    panel_data = [
        (raw, f"Raw ({ch_label})", None),
        (filt1, f"Bandpass {BAND1[0]}-{BAND1[1]} Hz", "tab:orange"),
        (filt2, f"Bandpass {BAND2[0]}-{BAND2[1]} Hz", "tab:green"),
    ]

    axes = []
    for i in range(3):
        ax = fig.add_axes([0.08, 0.30 + (2 - i) * 0.22, 0.88, 0.19])
        signal, label, color = panel_data[i]
        kwargs = {"lw": 0.8, "alpha": 0.8, "label": label}
        if color:
            kwargs["color"] = color
        ax.plot(t, signal, **kwargs)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(label, fontsize=10)
        if i < 2:
            ax.set_xticklabels([])
        axes.append(ax)

    axes[-1].set_xlabel("Time (s)")

    # Sliders
    ax_pos = fig.add_axes([0.15, 0.14, 0.7, 0.03])
    ax_zoom = fig.add_axes([0.15, 0.08, 0.7, 0.03])

    slider_pos = Slider(
        ax_pos, "Position (s)", 0, max(0.1, total_duration - default_window),
        valinit=0, valstep=0.1,
    )
    slider_zoom = Slider(
        ax_zoom, "Window (s)", 1.0, total_duration,
        valinit=default_window, valstep=0.5,
    )

    # Channel switching state
    state = {"channel": channel}

    def replot_channel(ch):
        state["channel"] = ch
        new_raw = recording["data"][ch]
        new_filt1 = bandpass_filter(new_raw, BAND1[0], BAND1[1], fs)
        new_filt2 = bandpass_filter(new_raw, BAND2[0], BAND2[1], fs)
        ch_lbl = labels[ch] if ch < len(labels) else f"Ch{ch}"

        # Print power breakdown for new channel
        print(f"\nChannel: {ch_lbl}")
        print_band_power(new_filt1, fs, BAND1, "Band 1")
        print_band_power(new_filt2, fs, BAND2, "Band 2")

        signals = [new_raw, new_filt1, new_filt2]
        for i, ax in enumerate(axes):
            ax.lines[0].set_ydata(signals[i])
        axes[0].set_title(f"Raw ({ch_lbl})", fontsize=10)
        axes[0].legend_.texts[0].set_text(f"Raw ({ch_lbl})")
        fig.suptitle(
            f"Bandpass Comparison: {recording['recording_name']} ({ch_lbl})",
            fontsize=12, fontweight="bold",
        )
        update()

    def update(val=None):
        window = slider_zoom.val
        pos = slider_pos.val

        new_max = max(0.1, total_duration - window)
        slider_pos.valmax = new_max
        slider_pos.ax.set_xlim(0, new_max)
        if pos > new_max:
            slider_pos.set_val(new_max)
            pos = new_max

        x_min = pos
        x_max = pos + window

        for ax in axes:
            ax.set_xlim(x_min, x_max)
            line = ax.lines[0]
            ydata = line.get_ydata()
            mask = (t >= x_min) & (t <= x_max)
            if np.any(mask):
                visible = ydata[mask]
                y_lo, y_hi = visible.min(), visible.max()
                pad = 0.1 * (y_hi - y_lo + 1e-9)
                ax.set_ylim(y_lo - pad, y_hi + pad)

        fig.canvas.draw_idle()

    slider_pos.on_changed(update)
    slider_zoom.on_changed(update)

    def on_scroll(event):
        if event.inaxes in axes:
            current_window = slider_zoom.val
            if event.button == "up":
                new_window = max(1.0, current_window * 0.8)
            else:
                new_window = min(total_duration, current_window * 1.25)
            slider_zoom.set_val(new_window)

    def on_key(event):
        n_ch = recording["data"].shape[0]
        if event.key == "right":
            replot_channel((state["channel"] + 1) % n_ch)
        elif event.key == "left":
            replot_channel((state["channel"] - 1) % n_ch)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update()

    fig.text(
        0.5, 0.02,
        "Scroll wheel to zoom | Drag sliders to navigate | Left/Right arrows to switch channels",
        ha="center", fontsize=9, style="italic", color="gray",
    )

    return fig


def main():
    # Select recording
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        recordings = list_recordings()
        if not recordings:
            print("\nUsage: python show_bandpass.py <recording.npz>")
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

    print(f"Band 1: {BAND1[0]}-{BAND1[1]} Hz")
    print(f"Band 2: {BAND2[0]}-{BAND2[1]} Hz")

    recording = load_recording(filepath)
    plot_bandpass(recording)
    plt.show()


if __name__ == "__main__":
    main()
