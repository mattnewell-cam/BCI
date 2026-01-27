import time
from collections import deque

import numpy as np
from pylsl import StreamInlet
from scipy.signal import lfilter
from utils import find_eeg_stream, butter_bandpass, bandpower_welch



def main():
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=60)

    info = inlet.info()
    fs = int(info.nominal_srate())
    ch = info.channel_count()
    print(f"Connected: fs={fs} Hz, channels={ch}, name={info.name()}")

    # Muse via BlueMuse often has 5 channels (4 EEG + REF). Use first 4.
    use_ch = list(range(min(4, ch)))

    window_seconds = 4
    buf_len = fs * window_seconds
    buffers = [deque(maxlen=buf_len) for _ in use_ch]

    b, a = butter_bandpass(1.0, 40.0, fs, order=4)

    last_report = time.time()

    while True:
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue

        for i, c in enumerate(use_ch):
            buffers[i].append(sample[c])

        now = time.time()
        if now - last_report >= 1:
            last_report = now

            if any(len(buf) < buf_len for buf in buffers):
                print("Buffering...")
                continue

            ratios = []
            for buf in buffers:
                x = np.asarray(buf, dtype=np.float64)
                xf = lfilter(b, a, x)

                alpha = bandpower_welch(xf, fs, 8.0, 12.0)
                total = bandpower_welch(xf, fs, 1.0, 30.0)
                ratios.append(alpha / total if total > 0 else 0.0)

            best_idx = int(np.argmax(ratios))
            ratios_str = " ".join([f"ch{use_ch[i]}:{ratios[i]:.3f}" for i in range(len(ratios))])
            print(f"alpha/total ratios: {ratios_str} | best=ch{use_ch[best_idx]}:{ratios[best_idx]:.3f}")


if __name__ == "__main__":
    main()