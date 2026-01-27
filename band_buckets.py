# realtime_eeg_band_buckets.py
# Requires: pip install pylsl numpy scipy matplotlib

import time
from collections import deque

import numpy as np
from pylsl import resolve_streams, StreamInlet
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from utils import find_eeg_stream, get_channel_labels


BANDS = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 12.0),
    ("Beta", 12.0, 30.0),
    ("Gamma", 30.0, 45.0),
    ("45-100", 45.0, 100.0),
    ("100-200", 100.0, 200.0)
]

def bandpower_from_psd(f, pxx, f_lo, f_hi):
    m = (f >= f_lo) & (f <= f_hi)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(pxx[m], f[m]))


def main():
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=60)

    info = inlet.info()
    fs = float(info.nominal_srate())
    ch = int(info.channel_count())
    name = info.name()
    print(f"Connected: fs={fs:.1f} Hz, channels={ch}, name={name}")

    # Default: first 4 channels (Muse often: 4 EEG + REF)
    use_ch = list(range(min(4, ch)))
    ch_names = get_channel_labels(info, ch)
    print("Channel labels:", ", ".join(ch_names))
    active_mask = np.zeros(ch, dtype=bool)
    active_mask[use_ch] = True

    window_s = 2.0
    hop_s = 0.25
    buf_len = int(fs * window_s)

    bufs = [deque(maxlen=buf_len) for _ in range(ch)]
    last = time.time()

    plt.ion()
    fig = plt.figure(figsize=(9, 5))

    # Main bar chart axis
    ax = fig.add_axes([0.08, 0.15, 0.62, 0.75])
    labels = [b[0] for b in BANDS]
    bars = ax.bar(labels, [0]*len(BANDS))
    ax.set_ylabel("Band power (log10 a.u.)")
    ax.set_title("EEG band buckets (avg across selected channels)")
    ax.set_ylim(-12, 1)

    # Channel selector (CheckButtons)
    cax = fig.add_axes([0.74, 0.18, 0.23, 0.70])
    checks = CheckButtons(cax, ch_names, active_mask.tolist())
    cax.set_title("Channels", fontsize=10)

    def on_toggle(label):
        idx = ch_names.index(label)
        active_mask[idx] = not active_mask[idx]

    checks.on_clicked(on_toggle)

    while True:
        chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=int(fs * hop_s))
        if chunk:
            chunk = np.asarray(chunk, dtype=np.float64)  # (n, ch)
            for c in range(min(ch, chunk.shape[1])):
                bufs[c].extend(chunk[:, c])

        now = time.time()
        if now - last < hop_s:
            plt.pause(0.001)
            continue
        last = now

        active_idxs = np.where(active_mask)[0]
        if active_idxs.size == 0:
            plt.pause(0.001)
            continue

        if any(len(bufs[c]) < buf_len for c in active_idxs):
            plt.pause(0.001)
            continue

        x = np.vstack([np.asarray(bufs[c]) for c in active_idxs])  # (n_active, buf_len)
        x = x - x.mean(axis=1, keepdims=True)

        f, pxx = welch(x, fs=fs, nperseg=min(buf_len, int(fs)), axis=1)
        pxx_mean = pxx.mean(axis=0)

        vals = np.array([bandpower_from_psd(f, pxx_mean, lo, hi) for _, lo, hi in BANDS])
        vals_plot = np.log10(vals + 1e-12)

        for bar, v in zip(bars, vals_plot):
            bar.set_height(float(v))

        y_min = min(-12.0, float(vals_plot.min()) * 1.2)
        y_max = max(1.0, float(vals_plot.max()) * 1.2)
        ax.set_ylim(y_min, y_max)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

if __name__ == "__main__":
    main()