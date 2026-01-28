"""
Alpha Lock Sample - Apply PLL-based alpha tracking to recorded EEG data.

Uses alpha_lock_logic for all processing - edit that file to tune the algorithm.

Usage:
    python alpha_lock_sample.py recordings/my_recording.npz
    python alpha_lock_sample.py  # lists available recordings
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from alpha_lock_logic import AlphaLockProcessor, beep


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
        "data": data["data"],  # (n_channels, n_samples)
        "timestamps": data.get("timestamps", None),
        "sample_rate": float(data["sample_rate"]),
        "channel_labels": list(data.get("channel_labels", [])),
        "recording_name": str(data.get("recording_name", "")),
        "duration_seconds": float(data.get("duration_seconds", 0)),
    }


def process_recording(recording, playback_audio=False):
    """
    Process a recording through the alpha lock algorithm.

    Args:
        recording: dict from load_recording()
        playback_audio: if True, play beeps in real-time (slows down processing)

    Returns:
        dict with processing results
    """
    data = recording["data"]  # (n_channels, n_samples)
    fs = int(recording["sample_rate"])
    n_channels, n_samples = data.shape

    print(f"\nProcessing: {recording['recording_name']}")
    print(f"  Duration: {recording['duration_seconds']:.1f}s")
    print(f"  Sample rate: {fs}Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Samples: {n_samples}")

    # Create processor
    processor = AlphaLockProcessor(
        fs=fs,
        n_channels=n_channels,
        buffer_seconds=10,
        reselect_every_s=3.0,
    )

    # Storage for results
    xf_out = []
    theta_out = []
    freq_out = []
    lock_out = []
    nco_out = []
    beep_times = []
    best_channel_out = []

    # Process sample by sample
    print("  Processing...")
    for i in range(n_samples):
        sample = data[:, i]
        result = processor.process_sample(sample)

        if result["ready"]:
            xf_out.append(result["xf"])
            theta_out.append(result["theta"])
            freq_out.append(result["freq"])
            lock_out.append(result["lock"])
            nco_out.append(result["nco"])
            best_channel_out.append(result["best_channel"])

            if result["beep"]:
                beep_times.append(len(xf_out) / fs)
                if playback_audio:
                    beep(880, 15)

        # Progress
        if (i + 1) % (fs * 10) == 0:
            print(f"    {(i + 1) / fs:.0f}s / {n_samples / fs:.0f}s")

    print(f"  Done. Found {len(beep_times)} phase crossings.")

    status = processor.get_status()
    print(f"  Final IAF: {status['iaf']:.2f}Hz, Lock: {status['lock']:.3f}")

    return {
        "xf": np.array(xf_out),
        "theta": np.array(theta_out),
        "freq": np.array(freq_out),
        "lock": np.array(lock_out),
        "nco": np.array(nco_out),
        "beep_times": np.array(beep_times),
        "best_channel": np.array(best_channel_out),
        "fs": fs,
        "final_iaf": status["iaf"],
        "final_lock": status["lock"],
    }


def plot_results(results, recording_name=""):
    """Plot the processing results with interactive zoom/scroll."""
    from matplotlib.widgets import Slider

    fs = results["fs"]
    n_samples = len(results["xf"])
    t = np.arange(n_samples) / fs
    total_duration = t[-1]

    # Default view: 3 seconds
    default_window = min(3.0, total_duration)

    # Create figure with space for sliders at bottom
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Alpha Lock Analysis: {recording_name}", fontsize=12, fontweight="bold")

    # Create axes with space for sliders
    axes = []
    for i in range(4):
        ax = fig.add_axes([0.08, 0.28 + (3 - i) * 0.17, 0.88, 0.15])
        axes.append(ax)

    # Plot 1: Filtered signal + NCO
    ax1 = axes[0]
    ax1.plot(t, results["xf"], label="xf (filtered)", alpha=0.8, lw=0.8)
    nco_scaled = results["nco"] * np.std(results["xf"]) * 2
    ax1.plot(t, nco_scaled, label="NCO (scaled)", alpha=0.7, lw=0.8)
    beep_lines_1 = [ax1.axvline(bt, color="red", alpha=0.4, lw=1) for bt in results["beep_times"]]
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Filtered EEG and NCO phase reference", fontsize=10)
    ax1.set_xticklabels([])

    # Plot 2: Instantaneous frequency
    ax2 = axes[1]
    ax2.plot(t, results["freq"], lw=0.8)
    ax2.axhline(results["final_iaf"], color="red", linestyle="--",
                label=f"Final IAF: {results['final_iaf']:.2f}Hz", alpha=0.7)
    ax2.set_ylabel("Freq (Hz)")
    ax2.set_ylim(7, 14)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("PLL instantaneous frequency", fontsize=10)
    ax2.set_xticklabels([])

    # Plot 3: Lock metric
    ax3 = axes[2]
    ax3.plot(t, results["lock"], lw=0.8)
    ax3.axhline(0.5, color="orange", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Lock")
    ax3.set_ylim(0, 1)
    ax3.set_title("PLL lock quality", fontsize=10)
    ax3.set_xticklabels([])

    # Plot 4: Phase (wrapped)
    ax4 = axes[3]
    phase_wrapped = np.mod(results["theta"], 2 * np.pi)
    ax4.plot(t, phase_wrapped, lw=0.8, alpha=0.8)
    beep_lines_4 = [ax4.axvline(bt, color="red", alpha=0.4, lw=1) for bt in results["beep_times"]]
    ax4.set_ylabel("Phase (rad)")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Phase (mod 2π) - red = beep times", fontsize=10)

    # Slider axes
    ax_pos = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_zoom = fig.add_axes([0.15, 0.06, 0.7, 0.03])

    # Position slider (0 to end - window)
    slider_pos = Slider(
        ax_pos, "Position (s)", 0, max(0.1, total_duration - default_window),
        valinit=0, valstep=0.1
    )

    # Zoom slider (window size: 1s to full duration)
    slider_zoom = Slider(
        ax_zoom, "Window (s)", 1.0, total_duration,
        valinit=default_window, valstep=0.5
    )

    def update(val=None):
        window = slider_zoom.val
        pos = slider_pos.val

        # Update position slider max based on zoom
        new_max = max(0.1, total_duration - window)
        slider_pos.valmax = new_max
        slider_pos.ax.set_xlim(0, new_max)
        if pos > new_max:
            slider_pos.set_val(new_max)
            pos = new_max

        # Set x limits for all axes
        x_min = pos
        x_max = pos + window
        for ax in axes:
            ax.set_xlim(x_min, x_max)

        # Auto-scale y for first axis based on visible data
        mask = (t >= x_min) & (t <= x_max)
        if np.any(mask):
            visible_xf = results["xf"][mask]
            visible_nco = nco_scaled[mask]
            y_min = min(visible_xf.min(), visible_nco.min())
            y_max = max(visible_xf.max(), visible_nco.max())
            pad = 0.1 * (y_max - y_min + 1e-9)
            ax1.set_ylim(y_min - pad, y_max + pad)

        fig.canvas.draw_idle()

    slider_pos.on_changed(update)
    slider_zoom.on_changed(update)

    # Scroll wheel zoom
    def on_scroll(event):
        if event.inaxes in axes:
            current_window = slider_zoom.val
            if event.button == 'up':
                new_window = max(1.0, current_window * 0.8)
            else:
                new_window = min(total_duration, current_window * 1.25)
            slider_zoom.set_val(new_window)

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Initialize view
    update()

    # Instructions
    fig.text(0.5, 0.01, "Scroll wheel to zoom | Drag sliders to navigate",
             ha="center", fontsize=9, style="italic", color="gray")

    return fig


def main():
    if len(sys.argv) < 2:
        # No argument - list recordings and prompt
        recordings = list_recordings()
        if not recordings:
            print("\nUsage: python alpha_lock_sample.py <recording.npz>")
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
    else:
        filepath = sys.argv[1]

    # Load and process
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    recording = load_recording(filepath)
    results = process_recording(recording, playback_audio=False)

    # Plot
    fig = plot_results(results, recording["recording_name"])
    plt.show()


if __name__ == "__main__":
    main()
