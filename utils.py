from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, lfilter, welch
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def bandpower_welch(x, fs, fmin, fmax):
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapezoid(psd[idx], freqs[idx])


def find_eeg_stream(name_contains=" EEG", type_equals="EEG"):
    streams = resolve_streams()
    # Prefer exact type match, fallback to name contains
    candidates = [s for s in streams if s.type() == type_equals]
    if not candidates:
        candidates = [s for s in streams if name_contains in s.name()]
    if not candidates:
        raise RuntimeError(f"No EEG stream found. Streams seen: {[s.name() for s in streams]}")
    return candidates[0]


def get_channel_labels(info, ch):
    labels = []
    try:
        node = info.desc().child("channels").child("channel")
        for _ in range(ch):
            lab = node.child_value("label") if not node.empty() else ""
            labels.append(lab if lab else f"Ch{len(labels)}")
            node = node.next_sibling()
    except Exception:
        labels = [f"Ch{i}" for i in range(ch)]
    return labels