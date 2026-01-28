"""
Alpha Lock Sections - Identify strongest alpha sections and fit sinusoids.

Selects the channel with the sharpest alpha peak, identifies the best
2-second sections by in-band power, bandpass-filters the signal, and fits
a pure sinusoid to each section to measure how "clean" the alpha is.

Usage:
    python alpha_lock_sections.py recordings/my_recording.npz
    python alpha_lock_sections.py  # lists available recordings
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.signal import butter, sosfilt, welch, spectrogram as scipy_spectrogram
import matplotlib.pyplot as plt

from alpha_lock_sample import load_recording, list_recordings

CHANNEL_LABELS = ["AF7", "AF8", "TP9", "TP10"]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def rolling_alpha_profile(data_1ch, fs):
    """Rolling 0.5 Hz mean-of-log PSD at 0.1 Hz steps across 8-12 Hz.

    Computes a high-res spectrogram (0.1 Hz bins), converts to dB,
    averages across time, then applies a 5-point (0.5 Hz) rolling mean.

    Returns
    -------
    centers : ndarray
        Frequency centres, 0.1 Hz apart, from 8.2 to 11.8 Hz.
    rolling_db : ndarray
        Mean dB at each centre.
    """
    nperseg = min(int(10 * fs), len(data_1ch))
    noverlap = nperseg // 2
    freqs, _, Sxx = scipy_spectrogram(
        data_1ch, fs=fs, nperseg=nperseg, noverlap=noverlap,
    )
    mean_db = (10 * np.log10(Sxx + 1e-20)).mean(axis=1)

    # Extract 8-12 Hz at 0.1 Hz resolution
    mask = (freqs >= 8.0) & (freqs <= 12.0)
    f_alpha = freqs[mask]
    db_alpha = mean_db[mask]

    # 5-point rolling mean (0.5 Hz window)
    kernel = np.ones(5) / 5
    rolled = np.convolve(db_alpha, kernel, mode="valid")
    centers = f_alpha[2:2 + len(rolled)]

    return centers, rolled


def select_channel(data, fs):
    """Pick the channel with the most concentrated alpha peak.

    Uses rolling_alpha_profile to find the 0.5 Hz window with the
    highest mean dB above the alpha-band average.

    Returns
    -------
    best_ch : int
    peak_freq : float
        Centre of the best 0.5 Hz band.
    best_score : float
        dB above alpha-band mean.
    """
    n_channels = data.shape[0]
    best_ch, best_score, best_center = 0, -np.inf, 10.0

    for ch in range(n_channels):
        centers, rolling_db = rolling_alpha_profile(data[ch], fs)
        alpha_mean = rolling_db.mean()
        pk = np.argmax(rolling_db)
        score = rolling_db[pk] - alpha_mean
        if score > best_score:
            best_score = score
            best_ch = ch
            best_center = float(centers[pk])

    return best_ch, best_center, best_score


def select_sections(data_1ch, fs, peak_freq, section_s=2.0, fraction=0.3):
    """Select the top non-overlapping 2-second sections by in-band power.

    Parameters
    ----------
    data_1ch : ndarray
        Single-channel raw data.
    fs : float
        Sampling rate.
    peak_freq : float
        Centre of the 0.5 Hz peak band.
    section_s : float
        Section length in seconds.
    fraction : float
        Fraction of total sections to keep (0.3 = top 30%).

    Returns
    -------
    sections : list of (start_sample, end_sample)
        Sorted by time.
    powers : ndarray
        In-band power for each section (same order as sections).
    """
    n_samples = len(data_1ch)
    section_len = int(section_s * fs)

    # Build non-overlapping sections
    all_sections = []
    start = 0
    while start + section_len <= n_samples:
        all_sections.append((start, start + section_len))
        start += section_len

    if not all_sections:
        return [], np.array([])

    # Compute in-band power for each section (log before time-averaging)
    band_lo = peak_freq - 0.25
    band_hi = peak_freq + 0.25
    section_powers = []
    nperseg_sec = int(fs)  # 1s windows within 2s section -> 2 averages in log
    noverlap_sec = int(0.5 * fs)
    for s_start, s_end in all_sections:
        chunk = data_1ch[s_start:s_end]
        freqs, _, Sxx = scipy_spectrogram(
            chunk, fs=fs, nperseg=nperseg_sec, noverlap=noverlap_sec,
        )
        log_power = 10 * np.log10(Sxx + 1e-20)
        mean_db = log_power.mean(axis=1)  # log first, then average across time
        band_mask = (freqs >= band_lo) & (freqs <= band_hi)
        if np.any(band_mask):
            section_powers.append(mean_db[band_mask].mean())
        else:
            section_powers.append(-200.0)

    section_powers = np.array(section_powers)

    # Keep top fraction
    n_keep = max(1, int(len(all_sections) * fraction))
    top_idx = np.argsort(section_powers)[::-1][:n_keep]
    top_idx_sorted = np.sort(top_idx)  # re-sort by time

    sections = [all_sections[i] for i in top_idx_sorted]
    powers = section_powers[top_idx_sorted]

    return sections, powers


def bandpass_filter(data_1ch, fs, center_freq, bw=2.0, order=4):
    """Causal Butterworth bandpass using SOS form.

    Parameters
    ----------
    data_1ch : ndarray
    fs : float
    center_freq : float
    bw : float
        Total bandwidth (center ± bw/2).
    order : int
        Filter order.

    Returns
    -------
    filtered : ndarray
    """
    lo = center_freq - bw / 2
    hi = center_freq + bw / 2
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data_1ch)


def _fit_sinusoid_fixed(signal_section, fs, freq):
    """Fit A*sin(2*pi*freq*t + phi) at a fixed frequency via least squares.

    The model A*sin(wt + phi) = a*sin(wt) + b*cos(wt) is linear in (a, b).
    """
    n = len(signal_section)
    t = np.arange(n) / fs
    w = 2 * np.pi * freq

    X = np.column_stack([np.sin(w * t), np.cos(w * t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, signal_section, rcond=None)
    a, b = coeffs

    fitted = X @ coeffs
    rmse = np.sqrt(np.mean((signal_section - fitted) ** 2))
    amplitude = np.sqrt(a ** 2 + b ** 2)
    phase = np.arctan2(b, a)

    return fitted, rmse, amplitude, phase


def fit_sinusoid(signal_section, fs, freq, search_range=0.5, search_step=0.02):
    """Fit a sinusoid, searching over freq ± search_range for best RMSE.

    Grid-searches frequencies in [freq - search_range, freq + search_range]
    at *search_step* Hz resolution, picks the frequency with lowest RMSE,
    then returns the fit at that frequency.

    Returns
    -------
    fitted : ndarray
    rmse : float
    amplitude : float
    phase : float (radians)
    best_freq : float
        The frequency that minimised RMSE.
    """
    best_rmse = np.inf
    best_f = freq

    for f in np.arange(freq - search_range, freq + search_range + search_step / 2,
                       search_step):
        _, rmse, _, _ = _fit_sinusoid_fixed(signal_section, fs, f)
        if rmse < best_rmse:
            best_rmse = rmse
            best_f = f

    fitted, rmse, amplitude, phase = _fit_sinusoid_fixed(signal_section, fs, best_f)
    return fitted, rmse, amplitude, phase, best_f


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(recording, best_ch, peak_freq, sections, filtered,
                 fitted_sections, rmses):
    """Three-panel figure: spectrogram, PSD profile, bandpassed signal."""
    from matplotlib.widgets import Slider
    from matplotlib.patches import Rectangle

    data = recording["data"]
    fs = recording["sample_rate"]
    n_samples = data.shape[1]
    labels = list(recording.get("channel_labels", []))
    if not labels or len(labels) < data.shape[0]:
        labels = CHANNEL_LABELS[:data.shape[0]]

    total_duration = n_samples / fs
    t_full = np.arange(n_samples) / fs
    default_window = min(10.0, total_duration)

    # Spectrogram of selected channel
    nperseg_spec = int(2 * fs)
    noverlap_spec = int(1.5 * fs)
    spec_freqs, spec_times, Sxx = scipy_spectrogram(
        data[best_ch], fs=fs, nperseg=nperseg_spec, noverlap=noverlap_spec,
    )
    freq_mask = (spec_freqs >= 0) & (spec_freqs <= 30)
    spec_freqs = spec_freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]
    power_dB = 10 * np.log10(Sxx + 1e-20)
    vmin = np.percentile(power_dB, 5)
    vmax = np.percentile(power_dB, 99)

    # Rolling PSD for diagnostic panel — same function as channel selection
    centers, rolling_db = rolling_alpha_profile(data[best_ch], fs)

    overall_rmse = rmses[-1]  # last entry is overall

    # --- Figure ---
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"Alpha Sections: {recording.get('recording_name', '')}  |  "
        f"Ch: {labels[best_ch]}  |  Peak: {peak_freq:.1f} Hz  |  "
        f"Overall RMSE: {overall_rmse:.4f}",
        fontsize=11, fontweight="bold",
    )

    ax_spec = fig.add_axes([0.08, 0.58, 0.88, 0.32])
    ax_psd = fig.add_axes([0.08, 0.42, 0.88, 0.13])
    ax_sig = fig.add_axes([0.08, 0.20, 0.88, 0.18])

    # -- Spectrogram --
    ax_spec.pcolormesh(spec_times, spec_freqs, power_dB, shading="gouraud",
                       cmap="viridis", vmin=vmin, vmax=vmax)
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_xticklabels([])
    ax_spec.set_title(f"Spectrogram — {labels[best_ch]}", fontsize=10)

    # Highlight sections on spectrogram
    for idx, (s_start, s_end) in enumerate(sections):
        t0 = s_start / fs
        t1 = s_end / fs
        rect = Rectangle(
            (t0, spec_freqs[0]), t1 - t0, spec_freqs[-1] - spec_freqs[0],
            linewidth=1.5, edgecolor="orange", facecolor="orange", alpha=0.15,
        )
        ax_spec.add_patch(rect)
        ax_spec.text(
            (t0 + t1) / 2, spec_freqs[-1] - 1, f"S{idx + 1}",
            ha="center", va="top", fontsize=7, color="orange",
            fontweight="bold",
        )

    # Peak band lines
    ax_spec.axhline(peak_freq - 0.25, color="red", ls="--", lw=0.8, alpha=0.7)
    ax_spec.axhline(peak_freq + 0.25, color="red", ls="--", lw=0.8, alpha=0.7)

    # -- PSD diagnostic panel --
    ax_psd.plot(centers, rolling_db, color="white", lw=1.5)
    ax_psd.fill_between(centers, rolling_db.min() - 1, rolling_db,
                        alpha=0.3, color="cyan")
    ax_psd.axvspan(peak_freq - 0.25, peak_freq + 0.25,
                   alpha=0.3, color="red", label=f"Selected: {peak_freq:.2f} Hz")
    pk_idx = np.argmax(rolling_db)
    ax_psd.plot(centers[pk_idx], rolling_db[pk_idx], "ro", ms=8, zorder=5)
    ax_psd.annotate(
        f"{centers[pk_idx]:.1f} Hz",
        xy=(centers[pk_idx], rolling_db[pk_idx]),
        xytext=(10, 5), textcoords="offset points",
        fontsize=8, color="red", fontweight="bold",
    )
    ax_psd.set_xlim(8, 12)
    ax_psd.set_ylabel("Mean dB")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_facecolor("#1a1a2e")
    ax_psd.set_title(
        "Rolling 0.5 Hz window power (0.1 Hz steps) — 8–12 Hz",
        fontsize=10,
    )
    ax_psd.legend(loc="upper right", fontsize=8)

    # -- Bandpassed signal --
    ax_sig.plot(t_full, filtered, color="#4488cc", lw=0.6, alpha=0.8,
                label="Bandpassed")

    for idx, (s_start, s_end) in enumerate(sections):
        t_sec = np.arange(s_start, s_end) / fs
        label = "Fitted sinusoid" if idx == 0 else None
        ax_sig.plot(t_sec, fitted_sections[idx], color="orange", lw=1.2,
                    alpha=0.9, label=label)

    ax_sig.set_ylabel("Amplitude")
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_title(
        f"Bandpassed {peak_freq - 1:.1f}–{peak_freq + 1:.1f} Hz "
        f"(blue) + sinusoid fits (orange)",
        fontsize=10,
    )
    ax_sig.legend(loc="upper right", fontsize=8)

    axes = [ax_spec, ax_sig]

    # -- Sliders --
    ax_pos = fig.add_axes([0.15, 0.09, 0.7, 0.03])
    ax_zoom = fig.add_axes([0.15, 0.03, 0.7, 0.03])

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

        x_min, x_max = pos, pos + window
        for ax in axes:
            ax.set_xlim(x_min, x_max)

        # Auto-scale signal axis
        mask = (t_full >= x_min) & (t_full <= x_max)
        if np.any(mask):
            vis = filtered[mask]
            pad = 0.1 * (vis.max() - vis.min() + 1e-9)
            ax_sig.set_ylim(vis.min() - pad, vis.max() + pad)

        fig.canvas.draw_idle()

    slider_pos.on_changed(update)
    slider_zoom.on_changed(update)

    def on_scroll(event):
        if event.inaxes in axes:
            cur = slider_zoom.val
            if event.button == "up":
                slider_zoom.set_val(max(1.0, cur * 0.8))
            else:
                slider_zoom.set_val(min(total_duration, cur * 1.25))

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    update()

    fig.text(
        0.5, 0.005,
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
            print("\nUsage: python alpha_lock_sections.py <recording.npz>")
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
    data = recording["data"]
    fs = recording["sample_rate"]
    n_channels = data.shape[0]
    labels = list(recording.get("channel_labels", []))
    if not labels or len(labels) < n_channels:
        labels = CHANNEL_LABELS[:n_channels]

    print(f"\nAlpha Lock Sections: {recording['recording_name']}")
    print(f"  Duration: {recording['duration_seconds']:.1f}s")
    print(f"  Sample rate: {fs}Hz")
    print(f"  Channels: {n_channels}")

    # 1. Select best channel
    best_ch, peak_freq, score = select_channel(data, fs)
    print(f"\n  Best channel: {labels[best_ch]} (ch {best_ch})")
    print(f"  Peak band: {peak_freq - 0.25:.2f} – {peak_freq + 0.25:.2f} Hz "
          f"(centre {peak_freq:.2f} Hz)")
    print(f"  Concentration score: {score:.3f} "
          f"(fraction of 8-12 Hz power in best 0.5 Hz band)")

    # 2. Select top sections
    sections, powers = select_sections(data[best_ch], fs, peak_freq)
    n_total = int(data.shape[1] / (2 * fs))
    print(f"\n  Sections: {len(sections)} of {n_total} "
          f"({len(sections)/max(1,n_total)*100:.0f}% of 2s chunks)")

    # 3. Bandpass filter
    filtered = bandpass_filter(data[best_ch], fs, peak_freq, bw=2.0, order=4)

    # 4. Fit sinusoids and compute RMSE
    print(f"\n  {'Section':>8}  {'Start':>7}  {'End':>7}  {'RMSE':>10}  "
          f"{'Amplitude':>10}  {'Freq':>8}  {'Phase':>8}")
    print(f"  {'—' * 70}")

    fitted_sections = []
    rmses = []
    all_residuals = []

    for idx, (s_start, s_end) in enumerate(sections):
        segment = filtered[s_start:s_end]
        fitted, rmse, amplitude, phase, fit_freq = fit_sinusoid(
            segment, fs, peak_freq,
        )
        fitted_sections.append(fitted)
        rmses.append(rmse)
        all_residuals.append(segment - fitted)

        t0 = s_start / fs
        t1 = s_end / fs
        print(f"  S{idx + 1:>6}  {t0:7.2f}s  {t1:7.2f}s  {rmse:10.6f}  "
              f"{amplitude:10.6f}  {fit_freq:7.2f}Hz  {phase:+8.3f}")

    # Overall RMSE across all selected sections
    if all_residuals:
        overall_rmse = np.sqrt(np.mean(np.concatenate(all_residuals) ** 2))
    else:
        overall_rmse = 0.0
    rmses.append(overall_rmse)

    print(f"  {'—' * 70}")
    print(f"  {'Overall':>8}  {'':>7}  {'':>7}  {overall_rmse:10.6f}")

    # 5. Plot
    fig = plot_results(
        recording, best_ch, peak_freq, sections, filtered,
        fitted_sections, rmses,
    )
    plt.show()


if __name__ == "__main__":
    main()
