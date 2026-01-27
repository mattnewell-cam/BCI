import time
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, lfilter, welch, lfilter_zi
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
        kp=150.0,         # proportional gain (tune)
        ki=5000.0,        # integral gain (tune)
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

        # Low-pass I/Q
        a = self.iq_a
        self.i_lp = a * self.i_lp + (1.0 - a) * i
        self.q_lp = a * self.q_lp + (1.0 - a) * q

        # Phase error estimate:
        # If perfectly aligned with cos(), Q should be ~0. Use atan2 for robustness.
        err = math.atan2(self.q_lp, self.i_lp)

        # PI loop (discrete)
        dt = 1.0 / self.fs
        self.int_err += err * dt

        # Update omega (rad/s) and phase increment
        omega_correction = (self.kp * err) + (self.ki * self.int_err)
        omega = (2.0 * math.pi * self.center_hz) + omega_correction

        # Keep omega within a sane alpha-ish range (prevents runaway on garbage input)
        omega = float(np.clip(omega, 2.0 * math.pi * 6.0, 2.0 * math.pi * 14.0))
        self.omega = omega

        dtheta = self.omega * dt
        self.theta_unwrapped += dtheta
        self.theta = (self.theta + dtheta) % (2.0 * math.pi)

        inst_freq_hz = self.omega / (2.0 * math.pi)

        # A crude "lock metric": magnitude of baseband vector (bigger tends to mean more coherent)
        lock_metric = math.sqrt(self.i_lp * self.i_lp + self.q_lp * self.q_lp)

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

    # Buffer used to re-evaluate "best" channel
    window_seconds = 4
    buf_len = fs * window_seconds
    buffers = [deque(maxlen=buf_len) for _ in use_ch]

    # Bandpass for alpha tracking (streamed)
    b, a = butter_bandpass(8.0, 12.0, fs, order=4)
    zi = lfilter_zi(b, a) * 0.0  # bandpass filter state

    # PLL settings
    pll = AlphaPLL(
        fs=fs,
        center_hz=10.0,
        iq_lp_hz=2.0,
        kp=150.0,
        ki=5000.0,
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
    plot_window_s = 5.0
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
    plot_every_s = 0.05  # ~20 Hz UI refresh

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

            # --- streaming bandpass: one-sample update, no tail refiltering ---
            xf_arr, zi = lfilter(b, a, np.array([x], dtype=np.float64), zi=zi)
            xf = float(xf_arr[0])

            # PLL step uses filtered sample
            theta_u, f_est, err, lock = pll.step(xf)

            # Perf counter
            n_samples += 1
            if time.time() - t0 >= 1.0:
                print("processed_hz≈", n_samples / (time.time() - t0))
                n_samples = 0
                t0 = time.time()

            # Record for plotting
            sig_plot.append(xf)
            nco_plot.append(math.cos(pll.theta))

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

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

        # Reselect best channel every few seconds (alpha ratio in 8–12 vs 1–30)
        now = time.time()
        if now - last_reselect >= reselect_every_s and all(len(buf) >= buf_len for buf in buffers):
            last_reselect = now
            ratios = []
            for buf in buffers:
                xw = np.asarray(buf, dtype=np.float64)
                x_alpha = lfilter(b, a, xw)  # fine (only every few seconds)
                alpha_p = bandpower_welch(x_alpha, fs, 8.0, 12.0)
                total_p = bandpower_welch(xw, fs, 1.0, 30.0)
                ratios.append(alpha_p / total_p if total_p > 0 else 0.0)

            best_idx = int(np.argmax(ratios))
            best_channel = use_ch[best_idx]
            print(f"best_ch=ch{best_channel}  pll_f≈{pll.omega/(2*math.pi):.2f}Hz  lock≈{pll.i_lp*pll.i_lp+pll.q_lp*pll.q_lp:.6f}")


if __name__ == "__main__":
    main()
