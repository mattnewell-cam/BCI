"""
ecHT - Endpoint-Corrected Hilbert Transform for alpha phase tracking.

Implements the ecHT method described in:
  "A wearable EEG system for closed-loop neuromodulation of sleep-related
   oscillations" (Bressler, Neely et al., J. Neural Eng., 2023)

The ecHT reduces Gibbs-phenomenon boundary artifacts in the standard Hilbert
transform by applying a causal narrow-band Butterworth filter in the frequency
domain after zeroing negative frequencies.  This preserves the phase estimate
at the most recent (endpoint) sample, making it suitable for real-time
closed-loop stimulation.

Algorithm (per sliding window):
  1. FFT the window
  2. Zero negative frequencies (analytic spectrum)
  3. fftshift, multiply by Butterworth bandpass H(f), ifftshift
  4. IFFT back to time domain
  5. Extract phase & amplitude from the last sample

Usage:
    python ecHT.py recordings/my_recording.npz
    python ecHT.py  # lists available recordings
"""

from collections import deque

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import butter, freqz, iirnotch, lfilter, lfilter_zi, welch
import matplotlib.pyplot as plt

try:
    from alpha_lock_sample import load_recording, list_recordings
except ImportError:
    from path_setup import add_repo_root
    add_repo_root()
    from alpha_lock_sample import load_recording, list_recordings

try:
    from alpha_lock_logic import beep
except ImportError:
    from path_setup import add_repo_root
    add_repo_root()
    from alpha_lock_logic import beep


# ---------------------------------------------------------------------------
# Core ecHT
# ---------------------------------------------------------------------------

class EcHT:
    """Endpoint-Corrected Hilbert Transform.

    Computes the analytic signal via DFT with a Hilbert mask and a
    narrow-band Butterworth filter applied in the frequency domain.
    Endpoint (last-sample) phase and amplitude are returned for real-time use.

    Reference implementations:
      - Schreglmann et al. (2021), Nature Communications 12, 363  (MATLAB)
      - MEEGkit ECHT class (Python)

    Parameters
    ----------
    l_freq : float
        Low cutoff of the bandpass (Hz).
    h_freq : float
        High cutoff of the bandpass (Hz).
    fs : float
        Sampling rate (Hz).
    n_fft : int, optional
        Window / FFT length in samples.  Default: ``int(fs)`` (1 second).
    filt_order : int, optional
        Butterworth filter order.  Default: 2.
    """

    def __init__(self, l_freq, h_freq, fs, n_fft=None, filt_order=2):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.fs = fs
        self.filt_order = filt_order

        if n_fft is None:
            n_fft = int(fs)  # 1-second window
        self.n_fft = n_fft

        # --- Hilbert mask: zero negative freqs, double positive ----------
        h = np.zeros(n_fft)
        if n_fft > 0 and n_fft % 2 == 0:
            h[0] = 1
            h[n_fft // 2] = 1
            h[1 : n_fft // 2] = 2
        elif n_fft > 0:
            h[0] = 1
            h[1 : (n_fft + 1) // 2] = 2
        self._hilbert_mask = h

        # --- Butterworth bandpass evaluated at FFT bin frequencies --------
        Wn = [l_freq / (fs / 2), h_freq / (fs / 2)]
        b, a = butter(filt_order, Wn, btype="band")
        T = n_fft / fs  # window duration (seconds)
        filt_freq = np.ceil(np.arange(-n_fft / 2, n_fft / 2)) / T
        _, self._filt_coeff = freqz(b, a, worN=filt_freq, fs=fs)

    def analytic(self, x):
        """Return the ecHT analytic signal for window *x*.

        Parameters
        ----------
        x : array-like, shape (n_fft,)

        Returns
        -------
        z : ndarray, complex, shape (n_fft,)
        """
        X = fft(np.asarray(x, dtype=np.float64), self.n_fft)
        X *= self._hilbert_mask              # analytic spectrum
        X = fftshift(X)                      # DC to centre
        X *= self._filt_coeff                # bandpass in freq domain
        X = ifftshift(X)                     # DC back to bin 0
        return ifft(X)

    def endpoint(self, x):
        """Phase and amplitude at the last sample of window *x*.

        Returns
        -------
        phase : float   (-pi, pi]
        amplitude : float
        """
        z = self.analytic(x)
        z_end = z[-1]
        return float(np.angle(z_end)), float(np.abs(z_end))


# ---------------------------------------------------------------------------
# Streaming processor
# ---------------------------------------------------------------------------

class EcHTProcessor:
    """Multi-channel EEG processor using ecHT for phase-locked stimulation.

    Handles 50 Hz notch filtering, automatic channel selection (best alpha),
    IAF estimation, and target-phase crossing detection.

    Parameters
    ----------
    fs : int
        Sampling rate (Hz).
    n_channels : int
        Number of EEG channels (capped at 4 for Muse).
    alpha_low, alpha_high : float
        Alpha band edges (Hz).
    window_s : float
        ecHT sliding-window length in seconds.
    buffer_seconds : int
        History buffer for channel selection / Welch PSD (seconds).
    reselect_every_s : float
        How often to re-evaluate best channel (seconds).
    target_phase : float
        Phase at which to trigger a beep (radians).
    refractory_s : float
        Minimum interval between consecutive beeps (seconds).
    """

    def __init__(
        self,
        fs,
        n_channels=4,
        alpha_low=8.0,
        alpha_high=12.0,
        window_s=1.0,
        buffer_seconds=10,
        reselect_every_s=3.0,
        target_phase=0.0,
        refractory_s=0.06,
    ):
        self.fs = fs
        self.n_channels = min(n_channels, 4)
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.target_phase = target_phase
        self.refractory_s = refractory_s

        # -- channel-selection buffers (after notch) --
        buf_len = int(fs * buffer_seconds)
        self.buffers = [deque(maxlen=buf_len) for _ in range(self.n_channels)]

        # -- ecHT --
        self.window_len = int(round(window_s * fs))
        f0 = (alpha_low + alpha_high) / 2
        self.ecHT = EcHT(alpha_low, alpha_high, fs, n_fft=self.window_len)
        self.signal_buf = deque(maxlen=self.window_len)

        # -- 50 Hz notch (IIR with streaming state per channel) --
        b50, a50 = iirnotch(50, 30, fs)
        self.b50, self.a50 = b50, a50
        zi_template = lfilter_zi(b50, a50) * 0.0
        self.zi50 = [zi_template.copy() for _ in range(self.n_channels)]

        # -- channel selection state --
        self.best_channel = 0
        self.iaf = f0
        self._samples_since_reselect = 0
        self._reselect_interval = int(fs * reselect_every_s)

        # -- phase tracking --
        self._prev_phase = 0.0
        self._cumulative_phase = 0.0
        self._next_target = None
        self._last_beep_sample = -int(fs * refractory_s) - 1
        self._sample_count = 0

    # ----- channel selection -------------------------------------------------

    def _select_best_channel(self):
        """Pick channel with highest alpha / total power ratio; estimate IAF."""
        if len(self.buffers[0]) < self.fs * 2:
            return

        best_ratio, best_ch = -1.0, 0
        for ch in range(self.n_channels):
            data = np.asarray(self.buffers[ch], dtype=np.float64)
            freqs, psd = welch(
                data, fs=self.fs, nperseg=min(len(data), int(self.fs * 2))
            )
            total_mask = (freqs >= 1.0) & (freqs <= 40.0)
            alpha_mask = (freqs >= self.alpha_low) & (freqs <= self.alpha_high)
            total = np.trapezoid(psd[total_mask], freqs[total_mask])
            alpha = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
            ratio = alpha / (total + 1e-12)
            if ratio > best_ratio:
                best_ratio = ratio
                best_ch = ch

        self.best_channel = best_ch

        # Estimate Individual Alpha Frequency from best channel
        data = np.asarray(self.buffers[best_ch], dtype=np.float64)
        freqs, psd = welch(
            data,
            fs=self.fs,
            nperseg=min(len(data), int(self.fs * 4)),
            noverlap=int(self.fs * 2),
        )
        alpha_mask = (freqs >= self.alpha_low) & (freqs <= self.alpha_high)
        if np.any(alpha_mask):
            peak_idx = np.argmax(psd[alpha_mask])
            self.iaf = float(freqs[alpha_mask][peak_idx])
            # Rebuild ecHT centred on IAF
            bw = self.alpha_high - self.alpha_low
            self.ecHT = EcHT(
                max(1.0, self.iaf - bw / 2),
                self.iaf + bw / 2,
                self.fs,
                n_fft=self.window_len,
            )

    # ----- sample-by-sample processing ---------------------------------------

    def process_sample(self, sample):
        """Process one multi-channel sample.

        Parameters
        ----------
        sample : array-like, shape (n_channels,)

        Returns
        -------
        dict
            ready, phase, phase_wrapped, amplitude, freq, beep,
            best_channel, iaf.
        """
        use_ch = min(len(sample), self.n_channels)

        # Notch-filter and store per channel
        for ch in range(use_ch):
            y, self.zi50[ch] = lfilter(
                self.b50, self.a50, np.array([sample[ch]]), zi=self.zi50[ch]
            )
            self.buffers[ch].append(float(y[0]))

        # Periodic channel re-evaluation
        self._samples_since_reselect += 1
        if self._samples_since_reselect >= self._reselect_interval:
            self._select_best_channel()
            self._samples_since_reselect = 0

        # Feed best-channel notched sample into ecHT sliding buffer
        self.signal_buf.append(
            self.buffers[self.best_channel][-1]
            if self.buffers[self.best_channel]
            else 0.0
        )
        self._sample_count += 1

        # Need a full window before ecHT can run
        if len(self.signal_buf) < self.window_len:
            return {
                "ready": False,
                "phase": 0.0,
                "phase_wrapped": 0.0,
                "amplitude": 0.0,
                "freq": 0.0,
                "beep": False,
                "best_channel": self.best_channel,
                "iaf": self.iaf,
            }

        # --- ecHT endpoint phase estimation ---
        window = np.asarray(self.signal_buf, dtype=np.float64)
        phase, amplitude = self.ecHT.endpoint(window)

        # Unwrap phase for cumulative tracking
        delta = phase - self._prev_phase
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi
        self._cumulative_phase += delta
        self._prev_phase = phase

        # Instantaneous frequency from phase increment
        inst_freq = abs(delta) * self.fs / (2 * np.pi)
        inst_freq = float(np.clip(inst_freq, self.alpha_low - 2, self.alpha_high + 2))

        # First-time target initialisation
        if self._next_target is None:
            self._next_target = (
                np.ceil(
                    (self._cumulative_phase - self.target_phase) / (2 * np.pi)
                )
                * 2 * np.pi
                + self.target_phase
            )

        # Phase-crossing detection
        beep_now = False
        if self._cumulative_phase >= self._next_target:
            elapsed = self._sample_count - self._last_beep_sample
            if elapsed >= int(self.fs * self.refractory_s):
                beep_now = True
                self._last_beep_sample = self._sample_count
            self._next_target += 2 * np.pi

        return {
            "ready": True,
            "phase": self._cumulative_phase,
            "phase_wrapped": phase,
            "amplitude": amplitude,
            "freq": inst_freq,
            "beep": beep_now,
            "best_channel": self.best_channel,
            "iaf": self.iaf,
        }

    def get_status(self):
        return {"best_channel": self.best_channel, "iaf": self.iaf}


# ---------------------------------------------------------------------------
# Recording processing
# ---------------------------------------------------------------------------

def process_recording(recording, playback_audio=False):
    """Run a recorded .npz file through the ecHT processor.

    Args:
        recording: dict from load_recording()
        playback_audio: if True, play beeps in real time (slow)

    Returns:
        dict with arrays: amplitude, freq, phase, phase_wrapped,
        beep_times, best_channel, fs, final_iaf.
    """
    data = recording["data"]  # (n_channels, n_samples)
    fs = int(recording["sample_rate"])
    n_channels, n_samples = data.shape

    print(f"\nProcessing (ecHT): {recording['recording_name']}")
    print(f"  Duration: {recording['duration_seconds']:.1f}s")
    print(f"  Sample rate: {fs}Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Samples: {n_samples}")

    processor = EcHTProcessor(
        fs=fs,
        n_channels=n_channels,
        buffer_seconds=10,
        reselect_every_s=3.0,
    )

    # Storage
    amplitude_out = []
    freq_out = []
    phase_out = []
    phase_wrapped_out = []
    beep_times = []
    beep_phases = []
    best_channel_out = []

    print("  Processing...")
    for i in range(n_samples):
        result = processor.process_sample(data[:, i])

        if result["ready"]:
            amplitude_out.append(result["amplitude"])
            freq_out.append(result["freq"])
            phase_out.append(result["phase"])
            phase_wrapped_out.append(result["phase_wrapped"])
            best_channel_out.append(result["best_channel"])

            if result["beep"]:
                beep_times.append(len(amplitude_out) / fs)
                beep_phases.append(result["phase_wrapped"])
                if playback_audio:
                    beep(880, 15)

        # Progress
        if (i + 1) % (fs * 10) == 0:
            print(f"    {(i + 1) / fs:.0f}s / {n_samples / fs:.0f}s")

    print(f"  Done. Found {len(beep_times)} phase crossings.")
    status = processor.get_status()
    print(f"  Final IAF: {status['iaf']:.2f}Hz")

    # Phase Locking Value: consistency of phase at stimulation times
    beep_phases = np.array(beep_phases)
    if len(beep_phases) > 0:
        plv = float(np.abs(np.mean(np.exp(1j * beep_phases))))
    else:
        plv = 0.0
    print(f"  PLV: {plv:.3f}  ({len(beep_phases)} events)")

    return {
        "amplitude": np.array(amplitude_out),
        "freq": np.array(freq_out),
        "phase": np.array(phase_out),
        "phase_wrapped": np.array(phase_wrapped_out),
        "beep_times": np.array(beep_times),
        "beep_phases": beep_phases,
        "plv": plv,
        "best_channel": np.array(best_channel_out),
        "fs": fs,
        "final_iaf": status["iaf"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, recording_name=""):
    """Interactive 4-panel plot of ecHT results."""
    from matplotlib.widgets import Slider

    fs = results["fs"]
    n = len(results["amplitude"])
    if n == 0:
        print("No data to plot.")
        return None
    t = np.arange(n) / fs
    total_duration = t[-1]
    default_window = min(3.0, total_duration)

    fig = plt.figure(figsize=(14, 10))
    plv = results.get("plv", 0.0)
    n_events = len(results.get("beep_phases", []))
    fig.suptitle(
        f"ecHT Alpha Analysis: {recording_name}    "
        f"PLV = {plv:.3f} ({n_events} events)",
        fontsize=12,
        fontweight="bold",
    )

    axes = []
    for i in range(4):
        ax = fig.add_axes([0.08, 0.28 + (3 - i) * 0.17, 0.88, 0.15])
        axes.append(ax)

    # 1 - Amplitude envelope
    ax1 = axes[0]
    ax1.plot(t, results["amplitude"], lw=0.8, label="ecHT amplitude")
    for bt in results["beep_times"]:
        ax1.axvline(bt, color="red", alpha=0.4, lw=1)
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Alpha amplitude envelope (ecHT)", fontsize=10)
    ax1.set_xticklabels([])

    # 2 - Instantaneous frequency
    ax2 = axes[1]
    ax2.plot(t, results["freq"], lw=0.8)
    ax2.axhline(
        results["final_iaf"],
        color="red",
        linestyle="--",
        label=f"Final IAF: {results['final_iaf']:.2f}Hz",
        alpha=0.7,
    )
    ax2.set_ylabel("Freq (Hz)")
    ax2.set_ylim(6, 14)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("Instantaneous frequency (ecHT)", fontsize=10)
    ax2.set_xticklabels([])

    # 3 - Reconstructed alpha signal
    ax3 = axes[2]
    reconstructed = results["amplitude"] * np.cos(results["phase_wrapped"])
    ax3.plot(t, reconstructed, lw=0.8, alpha=0.8)
    for bt in results["beep_times"]:
        ax3.axvline(bt, color="red", alpha=0.4, lw=1)
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Reconstructed alpha signal", fontsize=10)
    ax3.set_xticklabels([])

    # 4 - Wrapped phase
    ax4 = axes[3]
    ax4.plot(t, results["phase_wrapped"], lw=0.8, alpha=0.8)
    for bt in results["beep_times"]:
        ax4.axvline(bt, color="red", alpha=0.4, lw=1)
    ax4.set_ylabel("Phase (rad)")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Instantaneous phase (mod 2\u03c0) \u2013 red = beep times", fontsize=10)

    # -- Sliders --
    ax_pos = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_zoom = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slider_pos = Slider(
        ax_pos,
        "Position (s)",
        0,
        max(0.1, total_duration - default_window),
        valinit=0,
        valstep=0.1,
    )
    slider_zoom = Slider(
        ax_zoom,
        "Window (s)",
        1.0,
        total_duration,
        valinit=default_window,
        valstep=0.5,
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
        # auto-scale amplitude panel
        mask = (t >= x_min) & (t <= x_max)
        if np.any(mask):
            vis = results["amplitude"][mask]
            if len(vis) > 0:
                pad = 0.1 * (vis.max() - vis.min() + 1e-9)
                ax1.set_ylim(vis.min() - pad, vis.max() + pad)
            vis_r = reconstructed[mask]
            if len(vis_r) > 0:
                pad = 0.1 * (vis_r.max() - vis_r.min() + 1e-9)
                ax3.set_ylim(vis_r.min() - pad, vis_r.max() + pad)
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
        0.5,
        0.01,
        "Scroll wheel to zoom | Drag sliders to navigate",
        ha="center",
        fontsize=9,
        style="italic",
        color="gray",
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        recordings = list_recordings()
        if not recordings:
            print("\nUsage: python ecHT.py <recording.npz>")
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
    results = process_recording(recording, playback_audio=False)
    fig = plot_results(results, recording["recording_name"])
    plt.show()


if __name__ == "__main__":
    main()
