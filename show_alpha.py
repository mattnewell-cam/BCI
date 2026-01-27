import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, sosfilt, sosfilt_zi, welch, hilbert
from utils import find_eeg_stream, bandpower_welch


def main():
    # ---- LSL connect ----
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=60)

    info = inlet.info()
    fs = float(info.nominal_srate())
    ch = int(info.channel_count())
    name = info.name()
    print(f"Connected: {name} | fs={fs:.1f} Hz | channels={ch}")

    # Pick channel 0 by default (Muse: 0–3 are EEG, often 4 is REF)
    use_channel = 0 if ch > 0 else None
    if use_channel is None:
        raise RuntimeError("No channels in stream?")

    # ---- Filter (streaming IIR bandpass) ----
    # Butterworth bandpass 8–12 Hz implemented as SOS for stability
    low, high = 8.0, 12.0
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    zi = sosfilt_zi(sos)  # shape (n_sections, 2)
    filt_state = zi * 0.0  # start at rest

    # ---- Buffers (plot window) ----
    window_s = 5.0
    maxlen = int(fs * window_s)

    raw_buf = deque(maxlen=maxlen)
    alpha_buf = deque(maxlen=maxlen)

    # We’ll also keep timestamps for x-axis
    t_buf = deque(maxlen=maxlen)

    # ---- Matplotlib setup ----
    plt.rcParams["toolbar"] = "toolmanager"  # nicer UX if supported
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.set_title(f"Filtered alpha (8–12 Hz) — channel {use_channel}")
    ax1.set_ylabel("Amplitude (a.u.)")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Alpha magnitude (envelope + bandpower)")
    ax2.set_xlabel("Time (s, relative)")
    ax2.set_ylabel("Magnitude / Power")
    ax2.grid(True, alpha=0.3)

    # Lines
    line_alpha, = ax1.plot([], [], lw=1.5, label="alpha (bandpassed)")
    line_env, = ax1.plot([], [], lw=1.0, label="envelope (Hilbert)", alpha=0.9)
    ax1.legend(loc="upper right")

    line_env2, = ax2.plot([], [], lw=1.5, label="envelope (same)", alpha=0.9)
    ax2.legend(loc="upper right")

    txt = ax2.text(
        0.02, 0.95, "", transform=ax2.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # ---- Update loop ----
    last_frame_t = time.time()

    def update(_frame_idx):
        nonlocal filt_state, last_frame_t

        # Pull a small chunk (non-blocking-ish)
        chunk, ts = inlet.pull_chunk(timeout=0.0, max_samples=max(1, int(fs * 0.1)))
        if not chunk:
            # still update axes slowly so UI doesn’t feel frozen
            return line_alpha, line_env, line_env2, txt

        # For each sample, filter streaming and append
        for sample in chunk:
            x = float(sample[use_channel])

            # Streaming SOS filter: one sample at a time to preserve exact state
            y, filt_state = sosfilt(sos, [x], zi=filt_state)
            y = float(y[0])

            raw_buf.append(x)
            alpha_buf.append(y)

            # time axis: relative seconds in the past
            t_buf.append(time.time())

        # Build arrays for plotting
        t = np.asarray(t_buf, dtype=np.float64)
        if len(t) < 5:
            return line_alpha, line_env, line_env2, txt

        # Convert absolute time -> relative (last sample = 0s)
        t_rel = t - t[-1]

        alpha = np.asarray(alpha_buf, dtype=np.float64)

        # Envelope magnitude (computed over current plot window)
        # Hilbert gives analytic signal; abs() gives instantaneous amplitude
        env = np.abs(hilbert(alpha)) if len(alpha) >= 16 else np.zeros_like(alpha)

        # Scalar alpha power via Welch (over same window)
        alpha_power = bandpower_welch(alpha, fs, 8.0, 12.0)

        # Update lines
        line_alpha.set_data(t_rel, alpha)
        line_env.set_data(t_rel, env)

        line_env2.set_data(t_rel, env)

        # Auto-scale x and y
        ax1.set_xlim(t_rel[0], 0.0)

        # Give some padding on y
        amax = max(1e-9, float(np.max(np.abs(alpha))))
        emax = max(1e-9, float(np.max(env)))
        ax1.set_ylim(-1.2 * amax, 1.2 * max(amax, emax))

        ax2.set_xlim(t_rel[0], 0.0)
        ax2.set_ylim(0.0, 1.2 * emax)

        # Update text
        txt.set_text(
            f"fs: {fs:.1f} Hz\n"
            f"window: {window_s:.1f} s ({len(alpha)} samples)\n"
            f"envelope peak: {np.max(env):.3g}\n"
            f"alpha bandpower (Welch): {alpha_power:.3g}"
        )

        last_frame_t = time.time()
        return line_alpha, line_env, line_env2, txt

    ani = FuncAnimation(fig, update, interval=50, blit=False)  # ~20 FPS
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()