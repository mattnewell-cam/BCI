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


# --- Contact Quality Assessment ---

QUALITY_THRESHOLDS = {
    "variance_min": 5.0,
    "variance_max": 10000.0,
    "slope_good": -0.8,
    "slope_marginal": -0.3,
    "hf_ratio_max": 0.4,
    "line_noise_max": 0.5,
}


def compute_spectral_slope(freqs, psd, f_low=2, f_high=40):
    """
    Compute spectral slope in log-log space.
    Real EEG has 1/f characteristic with negative slope (-1 to -2).
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if np.sum(mask) < 5:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-12)
    slope, _ = np.polyfit(log_f, log_p, 1)
    return slope


def compute_hf_ratio(freqs, psd, cutoff=30):
    """Ratio of high-frequency power (>cutoff) to total power."""
    total = np.trapezoid(psd, freqs)
    if total < 1e-12:
        return 1.0
    hf_mask = freqs > cutoff
    hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
    return hf_power / total


def compute_line_noise_ratio(freqs, psd):
    """Check for excessive 50/60 Hz line noise."""
    for lf in [50, 60]:
        mask_line = (freqs >= lf - 2) & (freqs <= lf + 2)
        mask_neighbor = ((freqs >= lf - 10) & (freqs < lf - 2)) | \
                        ((freqs > lf + 2) & (freqs <= lf + 10))
        if np.sum(mask_line) > 0 and np.sum(mask_neighbor) > 0:
            line_power = np.mean(psd[mask_line])
            neighbor_power = np.mean(psd[mask_neighbor])
            if neighbor_power > 1e-12 and line_power / neighbor_power > 3:
                return (line_power / neighbor_power) / 10
    return 0.0


def assess_channel_quality(data, fs):
    """
    Assess signal quality for a single channel.

    Returns:
        quality: float 0-1 (0=bad, 1=good)
        status: str ('good', 'marginal', 'poor')
        metrics: dict of computed metrics
    """
    metrics = {"variance": np.var(data)}

    if metrics["variance"] < QUALITY_THRESHOLDS["variance_min"]:
        return 0.0, "poor", metrics
    if metrics["variance"] > QUALITY_THRESHOLDS["variance_max"]:
        return 0.2, "poor", metrics

    if len(data) < fs:
        return 0.5, "marginal", metrics

    freqs, psd = welch(data, fs=fs, nperseg=min(len(data), int(fs * 2)))

    metrics["slope"] = compute_spectral_slope(freqs, psd)
    metrics["hf_ratio"] = compute_hf_ratio(freqs, psd)
    metrics["line_noise"] = compute_line_noise_ratio(freqs, psd)

    score = 0.0

    if metrics["slope"] < QUALITY_THRESHOLDS["slope_good"]:
        score += 0.5
    elif metrics["slope"] < QUALITY_THRESHOLDS["slope_marginal"]:
        score += 0.25

    if QUALITY_THRESHOLDS["variance_min"] * 10 < metrics["variance"] < QUALITY_THRESHOLDS["variance_max"] / 10:
        score += 0.2
    elif QUALITY_THRESHOLDS["variance_min"] < metrics["variance"] < QUALITY_THRESHOLDS["variance_max"]:
        score += 0.1

    if metrics["hf_ratio"] < QUALITY_THRESHOLDS["hf_ratio_max"] / 2:
        score += 0.2
    elif metrics["hf_ratio"] < QUALITY_THRESHOLDS["hf_ratio_max"]:
        score += 0.1

    if metrics["line_noise"] < QUALITY_THRESHOLDS["line_noise_max"] / 2:
        score += 0.1
    elif metrics["line_noise"] < QUALITY_THRESHOLDS["line_noise_max"]:
        score += 0.05

    score = min(score, 1.0)
    status = "good" if score >= 0.7 else "marginal" if score >= 0.4 else "poor"

    return score, status, metrics


def get_quality_color(quality):
    """Return color based on quality score (0-1)."""
    if quality >= 0.7:
        return "#2ecc71"  # Green
    elif quality >= 0.4:
        return "#f1c40f"  # Yellow
    else:
        return "#e74c3c"  # Red