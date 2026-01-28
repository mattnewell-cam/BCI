import time
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, lfilter, welch, lfilter_zi, iirnotch
from utils import find_eeg_stream, butter_bandpass, bandpower_welch

def beep(f_hz=880, ms=20):
    # Windows: winsound is simplest
    try:
        import winsound  # type: ignore
        winsound.Beep(int(f_hz), int(ms))
        return
    except Exception:
        pass

    # Cross-platform fallback: try simpleaudio if installed
    try:
        import simpleaudio as sa  # type: ignore
        fs = 44100
        t = np.linspace(0, ms / 1000.0, int(fs * ms / 1000.0), endpoint=False)
        x = (0.2 * np.sin(2 * np.pi * f_hz * t)).astype(np.float32)
        audio = (x * 32767).astype(np.int16)
        sa.play_buffer(audio, 1, 2, fs)
        return
    except Exception:
        pass

    # Last resort: terminal bell (may be ignored by OS)
    print("\a", end="", flush=True)


# -------------------------
# Simple sinusoid-tracking PLL (alpha band)
# -------------------------
class AlphaPLL:
    """
    Tracks the phase of an approximately-sinusoidal component near center_hz.
    Not magic: it will behave poorly when alpha is weak / noisy.

    Implementation idea:
      - bandpassed x is mixed with internal NCO (cos/sin)
      - lowpass I/Q to estimate phase error
      - PI loop updates frequency + phase
      - provides unwrapped phase for stable cycle timing
    """

    def __init__(
        self,
        fs,
        center_hz=10.0,
        iq_lp_hz=2.0,     # low-pass cutoff for I/Q smoothing (Hz)
        kp=15.0,         # proportional gain (tune)
        ki=500.0,        # integral gain (tune)
    ):
        self.fs = float(fs)
        self.center_hz = float(center_hz)

        # NCO state
        self.omega = 2.0 * math.pi * self.center_hz  # rad/s
        self.theta = 0.0                             # phase used for sin/cos (wrapped)
        self.theta_unwrapped = 0.0                   # monotonically increasing phase (rad)

        # Loop filter state
        self.int_err = 0.0
        self.kp = float(kp)
        self.ki = float(ki)

        # I/Q low-pass (1st order)
        # y[n] = a*y[n-1] + (1-a)*x[n], where a = exp(-2πfc/fs)
        self.iq_a = math.exp(-2.0 * math.pi * float(iq_lp_hz) / self.fs)
        self.i_lp = 0.0
        self.q_lp = 0.0
        self.x2_lp = 0.0   # smoothed x² for amplitude normalization

    def step(self, x):
        """
        Process one sample, return (theta_unwrapped, inst_freq_hz, phase_error, lock_metric).
        """
        # NCO
        c = math.cos(self.theta)
        s = math.sin(self.theta)

        # Mix to baseband
        i = x * c
        q = x * s

        # Low-pass I/Q and signal power (same smoothing constant)
        a = self.iq_a
        self.i_lp = a * self.i_lp + (1.0 - a) * i
        self.q_lp = a * self.q_lp + (1.0 - a) * q
        self.x2_lp = a * self.x2_lp + (1.0 - a) * (x * x)

        # Phase error estimate:
        # If perfectly aligned with cos(), Q should be ~0. Use atan2 for robustness.
        err = math.atan2(self.q_lp, self.i_lp)

        # PI loop (discrete)
        dt = 1.0 / self.fs
        self.int_err += err * dt

        # Update omega (rad/s) and phase increment
        omega_correction = (self.kp * err) + (self.ki * self.int_err)
        omega = (2.0 * math.pi * self.center_hz) + omega_correction

        # Tight guard rails around current center (outer loop sets center_hz)
        omega_min = 2.0 * math.pi * (self.center_hz - 1.5)
        omega_max = 2.0 * math.pi * (self.center_hz + 1.5)
        omega = float(np.clip(omega, omega_min, omega_max))

        # Anti-windup: if clamped, back-calculate integrator to stay at the limit
        if omega != (2.0 * math.pi * self.center_hz) + omega_correction:
            self.int_err = (omega - 2.0 * math.pi * self.center_hz - self.kp * err) / self.ki

        self.omega = omega

        dtheta = self.omega * dt
        self.theta_unwrapped += dtheta
        self.theta = (self.theta + dtheta) % (2.0 * math.pi)

        inst_freq_hz = self.omega / (2.0 * math.pi)

        # Lock metric: I/Q magnitude normalized by signal RMS → 0..1
        # 1.0 = perfect sinusoid at NCO frequency, 0.0 = no coherent component
        iq_mag = math.sqrt(self.i_lp * self.i_lp + self.q_lp * self.q_lp)
        x_rms = math.sqrt(self.x2_lp) if self.x2_lp > 0.0 else 1e-12
        lock_metric = min(iq_mag / x_rms, 1.0)

        return self.theta_unwrapped, inst_freq_hz, err, lock_metric


def main():
    import matplotlib.pyplot as plt
    from scipy.signal import lfilter_zi  # add here if you didn't import at top

    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=60)

    info = inlet.info()
    fs = int(info.nominal_srate())
    ch = info.channel_count()
    print(f"Connected: fs={fs} Hz, channels={ch}, name={info.name()}")

    # Perf counter
    n_samples = 0
    t0 = time.time()

    # Muse via BlueMuse often has 5 channels (4 EEG + REF). Use first 4.
    use_ch = list(range(min(4, ch)))

    # Creating a notch at 50 Hz to remove mains hum
    b50, a50 = iirnotch(w0=50.0, Q=30.0, fs=fs)
    zi50_0 = lfilter_zi(b50, a50) * 0.0
    zi50_by_ch = {c: zi50_0.copy() for c in use_ch}

    # Buffer used to re-evaluate "best" channel + estimate IAF
    window_seconds = 10
    buf_len = fs * window_seconds
    buffers = [deque(maxlen=buf_len) for _ in use_ch]

    # Bandpass for alpha tracking (streamed)
    b, a = butter_bandpass(8.0, 12.0, fs, order=4)
    zi0 = lfilter_zi(b, a) * 0.0
    zi_by_ch = {c: zi0.copy() for c in use_ch}

    # PLL settings
    pll = AlphaPLL(
        fs=fs,
        center_hz=10.0,
        iq_lp_hz=1.0,      # tighter I/Q smoothing → less noise into loop
        kp=4.0,             # gentle proportional — won't overshoot to rail
        ki=40.0,            # slow integrator — tracks drift, not noise
    )

    # Beep timing settings
    target_phase = 0.0
    refractory_s = 0.06
    beep_freq_hz = 880
    beep_ms = 15

    best_channel = use_ch[0]
    last_reselect = time.time()
    reselect_every_s = 3.0

    two_pi = 2.0 * math.pi
    next_target = None
    last_beep_t = 0.0

    # -------------------------
    # Real-time plot (5s window): PLL input (xf) + internal cos()
    # -------------------------
    plot_window_s = 3.0
    plot_len = int(fs * plot_window_s)
    sig_plot = deque(maxlen=plot_len)   # xf (filtered EEG fed into PLL)
    nco_plot = deque(maxlen=plot_len)   # cos(theta) (PLL internal reference)

    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    (ln_sig,) = ax1.plot([], [], label="xf (PLL input)", color="tab:blue")
    (ln_nco,) = ax2.plot([], [], label="cos(theta) (PLL internal)", color="tab:orange")

    ax1.set_title("PLL input vs internal oscillator (rolling 5s)")
    ax1.set_xlabel("time (s, last 5s)")
    ax1.set_ylabel("xf (EEG units)")
    ax2.set_ylabel("cos(theta)")

    lines = [ln_sig, ln_nco]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    last_plot = time.time()
    plot_every_s = 0.5  # ~20 Hz UI refresh

    print("Buffering initial data...")

    while True:
        chunk, ts = inlet.pull_chunk(timeout=1.0, max_samples=max(1, int(fs * 0.05)))
        if not chunk:
            continue

        for sample in chunk:
            # Append to buffers
            for i, cidx in enumerate(use_ch):
                buffers[i].append(sample[cidx])

            if not all(len(buf) >= buf_len for buf in buffers):
                continue

            # Use current sample from best channel (raw)
            x = float(sample[best_channel])

            # 50 Hz notch (streaming, per-channel state)
            x_arr, zi50_by_ch[best_channel] = lfilter(
                b50, a50, np.array([x], dtype=np.float64), zi=zi50_by_ch[best_channel]
            )
            x = float(x_arr[0])

            # Bandpass 8-12 Hz (streaming, per-channel state)
            xf_arr, zi_by_ch[best_channel] = lfilter(
                b, a, np.array([x], dtype=np.float64), zi=zi_by_ch[best_channel]
            )
            xf = float(xf_arr[0])

            # PLL step uses filtered sample
            theta_u, f_est, err, lock = pll.step(xf)

            # Perf counter
            n_samples += 1
            if time.time() - t0 >= 1.0:
                # print("processed_hz≈", n_samples / (time.time() - t0))
                n_samples = 0
                t0 = time.time()

            # Record for plotting
            sig_plot.append(xf)
            nco_plot.append(math.cos(pll.theta) * 0.25)  # Scaling


            # Set first target just ahead
            if next_target is None:
                k = math.floor((theta_u - target_phase) / two_pi) + 1
                next_target = (k * two_pi) + target_phase

            # Beep once per cycle at target phase crossing
            now = time.time()
            if theta_u >= next_target and (now - last_beep_t) >= refractory_s:
                beep(beep_freq_hz, beep_ms)
                last_beep_t = now
                next_target += two_pi

            # Update plot
            if now - last_plot >= plot_every_s and len(sig_plot) > 5:
                last_plot = now
                y_sig = np.asarray(sig_plot, dtype=np.float64)
                y_nco = np.asarray(nco_plot, dtype=np.float64)

                # Plot "last 5s" axis (assumes you're keeping up; good enough for debug)
                t = np.linspace(-len(y_sig) / fs, 0.0, len(y_sig), endpoint=False)

                ln_sig.set_data(t, y_sig)
                ln_nco.set_data(t, y_nco)

                ax1.set_xlim(-plot_window_s, 0.0)

                ymin1, ymax1 = float(y_sig.min()), float(y_sig.max())
                pad1 = 0.1 * (ymax1 - ymin1 + 1e-9)
                ax1.set_ylim(ymin1 - pad1, ymax1 + pad1)

                ax2.set_ylim(-1.1, 1.1)

                fig.canvas.draw_idle()
                plt.pause(0.001)

        # Reselect best channel + estimate IAF every few seconds
        now = time.time()
        if now - last_reselect >= reselect_every_s and all(len(buf) >= buf_len for buf in buffers):
            last_reselect = now
            ratios = []
            for buf in buffers:
                xw = np.asarray(buf, dtype=np.float64)
                alpha_p = bandpower_welch(xw, fs, 8.0, 12.0)
                total_p = bandpower_welch(xw, fs, 1.0, 30.0)
                ratios.append(alpha_p / total_p if total_p > 0 else 0.0)

            best_idx = int(np.argmax(ratios))
            best_channel = use_ch[best_idx]

            # Estimate IAF: peak frequency in 8-12 Hz from best channel's 10s buffer
            xw = np.asarray(buffers[best_idx], dtype=np.float64)
            freqs, psd = welch(xw, fs=fs, nperseg=min(len(xw), fs * 4), noverlap=fs * 2)
            alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
            if np.any(alpha_mask):
                peak_idx = np.argmax(psd[alpha_mask])
                iaf = float(freqs[alpha_mask][peak_idx])
                # Recenter PLL on the detected IAF
                pll.center_hz = iaf

            iq_mag = math.sqrt(pll.i_lp**2 + pll.q_lp**2)
            x_rms = math.sqrt(pll.x2_lp) if pll.x2_lp > 0 else 1e-12
            lock_norm = min(iq_mag / x_rms, 1.0)
            print(f"best_ch=ch{best_channel}  iaf≈{pll.center_hz:.2f}Hz  pll_f≈{pll.omega/(2*math.pi):.2f}Hz  lock≈{lock_norm:.3f}")


if __name__ == "__main__":
    main()
