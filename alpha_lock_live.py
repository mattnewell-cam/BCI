"""
Alpha Lock Live - Real-time PLL-based alpha phase tracking with audio feedback.

Uses alpha_lock_logic for all processing - edit that file to tune the algorithm.
"""

import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet

from utils import find_eeg_stream
from alpha_lock_logic import AlphaLockProcessor, beep


def main():
    # Connect to EEG stream
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=2)

    info = inlet.info()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()
    print(f"Connected: fs={fs} Hz, channels={n_channels}, name={info.name()}")

    # Use first 4 channels
    n_eeg = min(4, n_channels)

    # Create processor
    processor = AlphaLockProcessor(
        fs=fs,
        n_channels=n_eeg,
        buffer_seconds=10,
        reselect_every_s=3.0,
    )

    # Beep settings
    beep_freq_hz = 880
    beep_ms = 15

    # Plot setup
    plot_window_s = 3.0
    plot_len = int(fs * plot_window_s)
    sig_plot = deque(maxlen=plot_len)
    nco_plot = deque(maxlen=plot_len)

    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ln_sig, = ax1.plot([], [], label="xf (PLL input)", color="tab:blue")
    ln_nco, = ax2.plot([], [], label="cos(theta) (NCO)", color="tab:orange")

    ax1.set_title("Alpha Lock - PLL tracking")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("xf (filtered EEG)")
    ax2.set_ylabel("NCO")

    ax1.legend([ln_sig, ln_nco], [ln_sig.get_label(), ln_nco.get_label()], loc="upper right")

    status_text = ax1.text(0.02, 0.98, "", transform=ax1.transAxes,
                           va="top", ha="left", fontsize=9,
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    last_plot = time.time()
    last_status_print = time.time()

    print("Buffering initial data...")

    while True:
        # Pull all available samples
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=256)
            if not chunk:
                break

            for sample in chunk:
                # Process sample
                result = processor.process_sample(sample[:n_eeg])

                if not result["ready"]:
                    continue

                # Record for plotting
                sig_plot.append(result["xf"])
                nco_plot.append(result["nco"] * 0.25)

                # Beep on phase crossing
                if result["beep"]:
                    beep(beep_freq_hz, beep_ms)

        # Update plot at ~5Hz
        now = time.time()
        if now - last_plot >= 0.2 and len(sig_plot) > 5:
            last_plot = now

            y_sig = np.asarray(sig_plot, dtype=np.float64)
            y_nco = np.asarray(nco_plot, dtype=np.float64)
            t = np.linspace(-len(y_sig) / fs, 0.0, len(y_sig), endpoint=False)

            ln_sig.set_data(t, y_sig)
            ln_nco.set_data(t, y_nco)

            ax1.set_xlim(-plot_window_s, 0.0)

            ymin1, ymax1 = float(y_sig.min()), float(y_sig.max())
            pad1 = 0.1 * (ymax1 - ymin1 + 1e-9)
            ax1.set_ylim(ymin1 - pad1, ymax1 + pad1)
            ax2.set_ylim(-0.5, 0.5)

            # Update status text
            status = processor.get_status()
            status_text.set_text(
                f"Ch: {status['best_channel']}  "
                f"IAF: {status['iaf']:.2f}Hz  "
                f"PLL: {status['pll_freq']:.2f}Hz  "
                f"Lock: {status['lock']:.2f}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # Print status periodically
        if now - last_status_print >= 3.0:
            last_status_print = now
            status = processor.get_status()
            if status["buffers_ready"]:
                print(f"ch={status['best_channel']}  "
                      f"iaf={status['iaf']:.2f}Hz  "
                      f"pll={status['pll_freq']:.2f}Hz  "
                      f"lock={status['lock']:.3f}")

        plt.pause(0.001)


if __name__ == "__main__":
    main()
