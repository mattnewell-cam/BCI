"""
Spectrogram - Time-frequency map viewer for recorded EEG data.

Computes and displays spectrograms (short-time FFT power spectral density)
for all EEG channels from a .npz recording. Useful for visualising how
frequency content evolves over time.

Usage:
    python spectrogram.py recordings/my_recording.npz
    python spectrogram.py  # lists available recordings
"""

import sys
import os

# Add parent directory so imports work when running from alpha_locking/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
import matplotlib.pyplot as plt

from alpha_lock_sample import load_recording, list_recordings

CHANNEL_LABELS = ["AF7", "AF8", "TP9", "TP10"]


def compute_spectrogram(data_1ch, fs, f_min=0, f_max=30):
    """Compute a spectrogram for a single channel.

    Parameters
    ----------
    data_1ch : ndarray, shape (n_samples,)
        Raw EEG samples for one channel.
    fs : float
        Sampling rate (Hz).
    f_min, f_max : float
        Frequency range to return (Hz).

    Returns
    -------
    times : ndarray
        Time axis (seconds).
    freqs : ndarray
        Frequency axis (Hz), clipped to [f_min, f_max].
    power_dB : ndarray, shape (n_freqs, n_times)
        Power spectral density in dB (10 * log10).
    """
    nperseg = int(2 * fs)        # 2-second windows -> 0.5 Hz resolution
    noverlap = int(1.5 * fs)     # 75% overlap

    freqs, times, Sxx = scipy_spectrogram(
        data_1ch, fs=fs, nperseg=nperseg, noverlap=noverlap,
    )

    # Clip to requested frequency range
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]

    # Convert to dB
    power_dB = 10 * np.log10(Sxx + 1e-20)

    return times, freqs, power_dB


def plot_spectrogram(recording, f_min=0, f_max=30):
    """Plot spectrograms for all channels with interactive sliders.

    Parameters
    ----------
    recording : dict
        From load_recording().
    f_min, f_max : float
        Frequency display range (Hz).

    Returns
    -------
    fig : matplotlib Figure
    """
    from matplotlib.widgets import Slider

    data = recording["data"]         # (n_channels, n_samples)
    fs = recording["sample_rate"]
    n_channels = data.shape[0]
    labels = recording.get("channel_labels", [])
    if not labels or len(labels) < n_channels:
        labels = CHANNEL_LABELS[:n_channels]

    # Compute spectrograms for all channels
    specs = []
    for ch in range(n_channels):
        times, freqs, power_dB = compute_spectrogram(data[ch], fs, f_min, f_max)
        specs.append((times, freqs, power_dB))

    # Global colour limits so all channels share the same scale
    all_power = np.concatenate([s[2].ravel() for s in specs])
    vmin = np.percentile(all_power, 5)
    vmax = np.percentile(all_power, 99)

    total_duration = times[-1]
    default_window = min(10.0, total_duration)

    # --- Figure layout ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Spectrogram: {recording.get('recording_name', '')}",
        fontsize=12, fontweight="bold",
    )

    axes = []       # spectrogram axes
    psd_axes = []   # mean-PSD side axes
    meshes = []
    panel_height = 0.15
    panel_gap = 0.17
    spec_width = 0.68
    psd_width = 0.12
    psd_left = 0.08 + spec_width + 0.01  # small gap between spectrogram and PSD

    for i in range(n_channels):
        bottom = 0.28 + (n_channels - 1 - i) * panel_gap

        # Spectrogram panel
        ax = fig.add_axes([0.08, bottom, spec_width, panel_height])
        t, f, pdb = specs[i]
        mesh = ax.pcolormesh(t, f, pdb, shading="gouraud", cmap="viridis",
                             vmin=vmin, vmax=vmax)
        ax.set_ylabel(f"{labels[i]}\nFreq (Hz)", fontsize=9)
        if i < n_channels - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")
        axes.append(ax)
        meshes.append(mesh)

        # Mean PSD side panel (frequency on Y to align with spectrogram)
        ax_psd = fig.add_axes([psd_left, bottom, psd_width, panel_height],
                              sharey=ax)
        mean_psd = pdb.mean(axis=1)  # average dB across time
        ax_psd.plot(mean_psd, f, color="white", lw=1.2)
        ax_psd.fill_betweenx(f, vmin, mean_psd, alpha=0.3, color="cyan")
        ax_psd.set_xlim(vmin, vmax)
        ax_psd.set_facecolor("#1a1a2e")
        ax_psd.tick_params(labelleft=False)
        if i < n_channels - 1:
            ax_psd.set_xticklabels([])
        else:
            ax_psd.set_xlabel("dB", fontsize=8)

        # Flag alpha peak (8-12 Hz)
        alpha_mask = (f >= 8) & (f <= 12)
        if np.any(alpha_mask):
            alpha_freqs = f[alpha_mask]
            alpha_psd = mean_psd[alpha_mask]
            peak_idx = np.argmax(alpha_psd)
            peak_freq = alpha_freqs[peak_idx]
            peak_db = alpha_psd[peak_idx]
            ax_psd.plot(peak_db, peak_freq, "ro", ms=6, zorder=5)
            ax_psd.annotate(
                f"{peak_freq:.1f} Hz",
                xy=(peak_db, peak_freq),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7, color="red", fontweight="bold",
                va="center",
            )

        # Shade alpha band background
        ax_psd.axhspan(8, 12, alpha=0.15, color="red")

        psd_axes.append(ax_psd)

    # Colour bar
    cbar_left = psd_left + psd_width + 0.015
    cbar_ax = fig.add_axes([cbar_left, 0.28, 0.012,
                            panel_gap * n_channels - 0.02])
    fig.colorbar(meshes[0], cax=cbar_ax, label="Power (dB)")

    # --- Sliders ---
    ax_pos = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_zoom = fig.add_axes([0.15, 0.06, 0.7, 0.03])

    slider_pos = Slider(
        ax_pos, "Position (s)", 0, max(0.1, total_duration - default_window),
        valinit=0, valstep=0.1,
    )
    slider_zoom = Slider(
        ax_zoom, "Window (s)", 1.0, total_duration,
        valinit=default_window, valstep=0.5,
    )

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
        fig.canvas.draw_idle()

    slider_pos.on_changed(update)
    slider_zoom.on_changed(update)

    def on_scroll(event):
        if event.inaxes in axes or event.inaxes in psd_axes:
            cur = slider_zoom.val
            if event.button == "up":
                slider_zoom.set_val(max(1.0, cur * 0.8))
            else:
                slider_zoom.set_val(min(total_duration, cur * 1.25))

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    update()

    fig.text(
        0.5, 0.01,
        "Scroll wheel to zoom | Drag sliders to navigate",
        ha="center", fontsize=9, style="italic", color="gray",
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        recordings = list_recordings()
        if not recordings:
            print("\nUsage: python spectrogram.py <recording.npz>")
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

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    recording = load_recording(filepath)

    print(f"\nSpectrogram: {recording['recording_name']}")
    print(f"  Duration: {recording['duration_seconds']:.1f}s")
    print(f"  Sample rate: {recording['sample_rate']}Hz")
    print(f"  Channels: {recording['data'].shape[0]}")

    fig = plot_spectrogram(recording)
    plt.show()


if __name__ == "__main__":
    main()
