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
from scipy.signal import spectrogram as scipy_spectrogram, welch
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


def compute_mean_psd(data_1ch, fs, f_min=0, f_max=30, bw=0.5, step=0.1):
    """Compute mean PSD with rolling frequency windows.

    Uses Welch's method at high resolution, then averages within sliding
    windows of width *bw* Hz, stepped at *step* Hz.

    Parameters
    ----------
    data_1ch : ndarray
        Raw samples for one channel.
    fs : float
        Sampling rate (Hz).
    f_min, f_max : float
        Frequency range.
    bw : float
        Rolling window width (Hz).
    step : float
        Step between window centres (Hz).

    Returns
    -------
    centers : ndarray
        Frequency centres (Hz).
    mean_db : ndarray
        Mean power (dB) at each centre.
    """
    # Welch PSD at 0.1 Hz native resolution (nperseg = 10 * fs)
    nperseg = min(int(10 * fs), len(data_1ch))
    freqs, psd = welch(data_1ch, fs=fs, nperseg=nperseg)

    psd_db = 10 * np.log10(psd + 1e-20)

    half = bw / 2
    centers = np.arange(f_min, f_max + step / 2, step)
    mean_db = np.empty(len(centers))
    for i, fc in enumerate(centers):
        mask = (freqs >= fc - half) & (freqs <= fc + half)
        if np.any(mask):
            mean_db[i] = psd_db[mask].mean()
        else:
            # No bins in window — interpolate from nearest
            idx = np.argmin(np.abs(freqs - fc))
            mean_db[i] = psd_db[idx]

    return centers, mean_db


def compute_best_channel_spectrogram(specs, period_s=3.0):
    """Build a composite spectrogram by picking the best-alpha channel per period.

    For each *period_s*-second block, selects the channel with the highest
    alpha (8-12 Hz) to total power ratio, then splices that channel's
    spectrogram columns into the output.

    Parameters
    ----------
    specs : list of (times, freqs, power_dB)
        Per-channel spectrograms from compute_spectrogram().
    period_s : float
        Selection period in seconds.

    Returns
    -------
    composite_dB : ndarray, shape (n_freqs, n_times)
        Composite spectrogram in dB.
    best_per_col : ndarray, shape (n_times,)
        Which channel was selected for each time column.
    """
    times = specs[0][0]
    freqs = specs[0][1]
    n_times = len(times)
    n_channels = len(specs)

    # Stack all channels: (n_channels, n_freqs, n_times)
    all_dB = np.stack([s[2] for s in specs], axis=0)

    # Convert to linear for ratio computation
    all_linear = 10 ** (all_dB / 10)

    alpha_mask = (freqs >= 8) & (freqs <= 12)
    total_mask = (freqs >= 1) & (freqs <= 30)

    composite_dB = np.empty_like(all_dB[0])
    best_per_col = np.empty(n_times, dtype=int)

    # Group columns into periods
    t0 = times[0]
    period_start = t0
    col = 0
    while col < n_times:
        # Find columns in this period
        period_end = period_start + period_s
        block_mask = (times >= period_start) & (times < period_end)
        if not np.any(block_mask):
            period_start = period_end
            continue

        block_idx = np.where(block_mask)[0]

        # Compute alpha:total ratio per channel for this block
        best_ratio, best_ch = -1.0, 0
        for ch in range(n_channels):
            block_power = all_linear[ch][:, block_idx]  # (n_freqs, n_block)
            alpha_power = block_power[alpha_mask, :].sum()
            total_power = block_power[total_mask, :].sum() + 1e-20
            ratio = alpha_power / total_power
            if ratio > best_ratio:
                best_ratio = ratio
                best_ch = ch

        composite_dB[:, block_idx] = all_dB[best_ch][:, block_idx]
        best_per_col[block_idx] = best_ch

        period_start = period_end
        col = block_idx[-1] + 1

    return composite_dB, best_per_col


def compute_mean_psd_from_spectrogram(freqs, power_dB, f_min=0, f_max=30,
                                      bw=0.5, step=0.1):
    """Rolling mean PSD from a spectrogram's time-averaged power.

    Like compute_mean_psd but works from spectrogram data directly (used
    for the composite best-channel panel).

    Parameters
    ----------
    freqs : ndarray
        Spectrogram frequency axis (Hz).
    power_dB : ndarray, shape (n_freqs, n_times)
        Spectrogram power in dB.
    f_min, f_max, bw, step : float
        Same as compute_mean_psd.

    Returns
    -------
    centers, mean_db : ndarrays
    """
    raw_mean = power_dB.mean(axis=1)  # average dB across time

    half = bw / 2
    centers = np.arange(f_min, f_max + step / 2, step)
    mean_db = np.empty(len(centers))
    for i, fc in enumerate(centers):
        mask = (freqs >= fc - half) & (freqs <= fc + half)
        if np.any(mask):
            mean_db[i] = raw_mean[mask].mean()
        else:
            idx = np.argmin(np.abs(freqs - fc))
            mean_db[i] = raw_mean[idx]

    return centers, mean_db


def plot_spectrogram(recording, f_min=0, f_max=30):
    """Plot one spectrogram + PSD at a time; left/right arrows switch channel.

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
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

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

    # Composite best-channel spectrogram
    composite_dB, best_per_col = compute_best_channel_spectrogram(specs)

    # Global colour limits so all channels share the same scale
    all_power = np.concatenate([s[2].ravel() for s in specs])
    vmin = np.percentile(all_power, 5)
    vmax = np.percentile(all_power, 99)

    total_duration = times[-1]
    default_window = min(10.0, total_duration)

    # Pre-compute PSD for each panel so channel and PSD always match
    panel_labels = list(labels) + [u"Best \u03b1"]
    n_panels = n_channels + 1
    psd_data = []
    for i in range(n_panels):
        if i < n_channels:
            pf, mp = compute_mean_psd(
                data[i], fs, f_min, f_max, bw=0.5, step=0.1)
        else:
            pf, mp = compute_mean_psd_from_spectrogram(
                specs[0][1], composite_dB, f_min, f_max, bw=0.5, step=0.1)
        psd_data.append((pf, mp))

    # --- Figure layout: single spectrogram + PSD ---
    fig = plt.figure(figsize=(16, 6))
    title_text = fig.suptitle(
        f"Spectrogram: {recording.get('recording_name', '')} "
        + u"\u2014 " + f"{panel_labels[0]}",
        fontsize=12, fontweight="bold",
    )

    spec_ax = fig.add_axes([0.08, 0.25, 0.63, 0.60])
    psd_ax = fig.add_axes([0.73, 0.25, 0.12, 0.60], sharey=spec_ax)
    cbar_ax = fig.add_axes([0.87, 0.25, 0.015, 0.60])

    # Colour bar from a fixed ScalarMappable (independent of mesh lifecycle)
    sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label="Power (dB)")

    # Placeholder mesh (immediately replaced by first update())
    t0, f0, p0 = specs[0]
    mesh = [spec_ax.pcolormesh(t0[:2], f0, p0[:, :2], shading="gouraud",
                               cmap="viridis", vmin=vmin, vmax=vmax)]

    current_panel = [0]
    t_full = specs[0][0]

    def draw_psd(panel_idx):
        """Redraw the PSD side-panel for the given channel."""
        psd_ax.clear()
        pf, mp = psd_data[panel_idx]
        psd_ax.plot(mp, pf, color="white", lw=1.2)
        psd_ax.fill_betweenx(pf, vmin, mp, alpha=0.3, color="cyan")
        psd_ax.set_xlim(vmin, vmax)
        psd_ax.set_facecolor("#1a1a2e")
        psd_ax.tick_params(labelleft=False)
        psd_ax.set_xlabel("dB", fontsize=8)

        # Alpha peak marker (8-12 Hz)
        a_mask = (pf >= 8) & (pf <= 12)
        if np.any(a_mask):
            a_freqs = pf[a_mask]
            a_psd = mp[a_mask]
            pk = np.argmax(a_psd)
            psd_ax.plot(a_psd[pk], a_freqs[pk], "ro", ms=6, zorder=5)
            psd_ax.annotate(
                f"{a_freqs[pk]:.1f} Hz",
                xy=(a_psd[pk], a_freqs[pk]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7, color="red", fontweight="bold", va="center",
            )
        psd_ax.axhspan(8, 12, alpha=0.15, color="red")

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

        idx = current_panel[0]
        if idx < n_channels:
            t_src, f_src, p_src = specs[idx]
        else:
            t_src, f_src, p_src = specs[0][0], specs[0][1], composite_dB

        # ------- DYNAMIC FIDELITY: comment out this block for full resolution,
        #         and uncomment the two lines after "END DYNAMIC FIDELITY". ------
        margin = (x_max - x_min) * 0.02
        vis_mask = (t_src >= x_min - margin) & (t_src <= x_max + margin)
        vis_idx = np.where(vis_mask)[0]
        if len(vis_idx) == 0:
            return
        max_cols = 800
        if len(vis_idx) > max_cols:
            ds_step = max(1, len(vis_idx) // max_cols)
            vis_idx = vis_idx[::ds_step]
        t_render = t_src[vis_idx]
        p_render = p_src[:, vis_idx]
        # ------- END DYNAMIC FIDELITY (uncomment below for full resolution) -----
        # t_render = t_src
        # p_render = p_src

        mesh[0].remove()
        mesh[0] = spec_ax.pcolormesh(t_render, f_src, p_render,
                                      shading="gouraud", cmap="viridis",
                                      vmin=vmin, vmax=vmax)
        spec_ax.set_xlim(x_min, x_max)
        spec_ax.set_ylabel("Freq (Hz)", fontsize=9)
        spec_ax.set_xlabel("Time (s)")

        fig.canvas.draw_idle()

    def switch_panel(new_idx):
        current_panel[0] = new_idx % n_panels
        title_text.set_text(
            f"Spectrogram: {recording.get('recording_name', '')} "
            + u"\u2014 " + f"{panel_labels[current_panel[0]]}")
        draw_psd(current_panel[0])
        update()

    def on_key(event):
        if event.key == "right":
            switch_panel(current_panel[0] + 1)
        elif event.key == "left":
            switch_panel(current_panel[0] - 1)

    def on_scroll(event):
        if event.inaxes in (spec_ax, psd_ax):
            cur = slider_zoom.val
            if event.button == "up":
                slider_zoom.set_val(max(1.0, cur * 0.8))
            else:
                slider_zoom.set_val(min(total_duration, cur * 1.25))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    slider_pos.on_changed(update)
    slider_zoom.on_changed(update)

    draw_psd(0)
    update()

    fig.text(
        0.5, 0.01,
        u"Left/Right: switch channel \u2502 Scroll: zoom \u2502 "
        u"Sliders: navigate",
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
