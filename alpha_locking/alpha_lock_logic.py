"""
Alpha Lock Logic

Core PLL-based alpha phase tracking that can be used for both
live streaming and offline sample processing.
"""

import math
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch, welch
from utils import butter_bandpass, bandpower_welch


class AlphaPLL:
    """
    Tracks the phase of an approximately-sinusoidal component near center_hz.

    Implementation:
      - Bandpassed signal is mixed with internal NCO (cos/sin)
      - Lowpass I/Q to estimate phase error
      - PI loop updates frequency + phase
      - Provides unwrapped phase for stable cycle timing
    """

    def __init__(
        self,
        fs,
        center_hz=10.0,
        iq_lp_hz=1.0,
        kp=4.0,
        ki=40.0,
    ):
        self.fs = float(fs)
        self.center_hz = float(center_hz)

        # NCO state
        self.omega = 2.0 * math.pi * self.center_hz
        self.theta = 0.0
        self.theta_unwrapped = 0.0

        # Loop filter state
        self.int_err = 0.0
        self.kp = float(kp)
        self.ki = float(ki)

        # I/Q low-pass (1st order)
        self.iq_a = math.exp(-2.0 * math.pi * float(iq_lp_hz) / self.fs)
        self.i_lp = 0.0
        self.q_lp = 0.0
        self.x2_lp = 0.0

    def step(self, x):
        """
        Process one sample.
        Returns: (theta_unwrapped, inst_freq_hz, phase_error, lock_metric)
        """
        c = math.cos(self.theta)
        s = math.sin(self.theta)

        i = x * c
        q = x * s

        a = self.iq_a
        self.i_lp = a * self.i_lp + (1.0 - a) * i
        self.q_lp = a * self.q_lp + (1.0 - a) * q
        self.x2_lp = a * self.x2_lp + (1.0 - a) * (x * x)

        err = math.atan2(self.q_lp, self.i_lp)

        dt = 1.0 / self.fs
        self.int_err += err * dt

        omega_correction = (self.kp * err) + (self.ki * self.int_err)
        omega = (2.0 * math.pi * self.center_hz) + omega_correction

        omega_min = 2.0 * math.pi * (self.center_hz - 1.5)
        omega_max = 2.0 * math.pi * (self.center_hz + 1.5)
        omega = float(np.clip(omega, omega_min, omega_max))

        if omega != (2.0 * math.pi * self.center_hz) + omega_correction:
            self.int_err = (omega - 2.0 * math.pi * self.center_hz - self.kp * err) / self.ki

        self.omega = omega

        dtheta = self.omega * dt
        self.theta_unwrapped += dtheta
        self.theta = (self.theta + dtheta) % (2.0 * math.pi)

        inst_freq_hz = self.omega / (2.0 * math.pi)

        iq_mag = math.sqrt(self.i_lp * self.i_lp + self.q_lp * self.q_lp)
        x_rms = math.sqrt(self.x2_lp) if self.x2_lp > 0.0 else 1e-12
        lock_metric = min(iq_mag / x_rms, 1.0)

        return self.theta_unwrapped, inst_freq_hz, err, lock_metric

    def get_lock_metric(self):
        """Get current lock quality (0-1)."""
        iq_mag = math.sqrt(self.i_lp ** 2 + self.q_lp ** 2)
        x_rms = math.sqrt(self.x2_lp) if self.x2_lp > 0 else 1e-12
        return min(iq_mag / x_rms, 1.0)


class AlphaLockProcessor:
    """
    Complete alpha lock processing pipeline.

    Handles:
    - Multi-channel buffering
    - Notch filtering (50Hz)
    - Bandpass filtering (8-12Hz)
    - Best channel selection based on alpha power
    - IAF (Individual Alpha Frequency) estimation
    - PLL tracking
    - Beep timing detection
    """

    def __init__(
        self,
        fs,
        n_channels=4,
        buffer_seconds=10,
        reselect_every_s=3.0,
        target_phase=0.0,
        refractory_s=0.06,
    ):
        self.fs = int(fs)
        self.n_channels = n_channels
        self.buffer_seconds = buffer_seconds
        self.reselect_every_s = reselect_every_s
        self.target_phase = target_phase
        self.refractory_s = refractory_s

        self.buf_len = self.fs * buffer_seconds
        self.buffers = [[] for _ in range(n_channels)]

        # 50Hz notch filter
        self.b50, self.a50 = iirnotch(w0=50.0, Q=30.0, fs=self.fs)
        zi50_0 = lfilter_zi(self.b50, self.a50) * 0.0
        self.zi50_by_ch = [zi50_0.copy() for _ in range(n_channels)]

        # 8-12Hz bandpass
        self.b_bp, self.a_bp = butter_bandpass(8.0, 12.0, self.fs, order=4)
        zi_bp_0 = lfilter_zi(self.b_bp, self.a_bp) * 0.0
        self.zi_bp_by_ch = [zi_bp_0.copy() for _ in range(n_channels)]

        # PLL
        self.pll = AlphaPLL(fs=self.fs, center_hz=10.0)

        # State
        self.best_channel = 0
        self.samples_since_reselect = 0
        self.reselect_samples = int(self.fs * reselect_every_s)

        self.two_pi = 2.0 * math.pi
        self.next_target = None
        self.last_beep_sample = -int(self.fs * refractory_s)
        self.current_sample = 0

        # Track if buffers are filled
        self.buffers_ready = False

    def _update_buffers(self, sample):
        """Add sample to all channel buffers."""
        for i in range(min(len(sample), self.n_channels)):
            self.buffers[i].append(sample[i])
            if len(self.buffers[i]) > self.buf_len:
                self.buffers[i] = self.buffers[i][-self.buf_len:]

        if not self.buffers_ready:
            self.buffers_ready = all(len(buf) >= self.buf_len for buf in self.buffers)

    def _select_best_channel(self):
        """Select channel with highest alpha/total power ratio."""
        if not self.buffers_ready:
            return

        ratios = []
        for buf in self.buffers:
            xw = np.asarray(buf, dtype=np.float64)
            alpha_p = bandpower_welch(xw, self.fs, 8.0, 12.0)
            total_p = bandpower_welch(xw, self.fs, 1.0, 30.0)
            ratios.append(alpha_p / total_p if total_p > 0 else 0.0)

        self.best_channel = int(np.argmax(ratios))

        # Estimate IAF from best channel
        xw = np.asarray(self.buffers[self.best_channel], dtype=np.float64)
        freqs, psd = welch(xw, fs=self.fs, nperseg=min(len(xw), self.fs * 4), noverlap=self.fs * 2)
        alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
        if np.any(alpha_mask):
            peak_idx = np.argmax(psd[alpha_mask])
            iaf = float(freqs[alpha_mask][peak_idx])
            self.pll.center_hz = iaf

    def process_sample(self, sample):
        """
        Process a single multi-channel sample.

        Args:
            sample: array-like of shape (n_channels,)

        Returns:
            dict with:
                - ready: bool, whether buffers are filled
                - xf: filtered signal (PLL input)
                - theta: unwrapped phase
                - freq: instantaneous frequency
                - lock: lock metric (0-1)
                - nco: NCO output (cos(theta))
                - beep: bool, whether to beep this sample
                - best_channel: current best channel index
                - iaf: current IAF estimate
        """
        self._update_buffers(sample)

        if not self.buffers_ready:
            return {"ready": False}

        # Reselect channel periodically
        self.samples_since_reselect += 1
        if self.samples_since_reselect >= self.reselect_samples:
            self.samples_since_reselect = 0
            self._select_best_channel()

        # Get sample from best channel
        x = float(sample[self.best_channel])

        # 50Hz notch
        x_arr, self.zi50_by_ch[self.best_channel] = lfilter(
            self.b50, self.a50, np.array([x], dtype=np.float64),
            zi=self.zi50_by_ch[self.best_channel]
        )
        x = float(x_arr[0])

        # 8-12Hz bandpass
        xf_arr, self.zi_bp_by_ch[self.best_channel] = lfilter(
            self.b_bp, self.a_bp, np.array([x], dtype=np.float64),
            zi=self.zi_bp_by_ch[self.best_channel]
        )
        xf = float(xf_arr[0])

        # PLL step
        theta_u, f_est, err, lock = self.pll.step(xf)

        # Initialize target phase
        if self.next_target is None:
            k = math.floor((theta_u - self.target_phase) / self.two_pi) + 1
            self.next_target = (k * self.two_pi) + self.target_phase

        # Check for beep
        beep = False
        samples_since_beep = self.current_sample - self.last_beep_sample
        refractory_samples = int(self.fs * self.refractory_s)

        if theta_u >= self.next_target and samples_since_beep >= refractory_samples:
            beep = True
            self.last_beep_sample = self.current_sample
            self.next_target += self.two_pi

        self.current_sample += 1

        return {
            "ready": True,
            "xf": xf,
            "theta": theta_u,
            "freq": f_est,
            "lock": lock,
            "nco": math.cos(self.pll.theta),
            "beep": beep,
            "best_channel": self.best_channel,
            "iaf": self.pll.center_hz,
        }

    def process_chunk(self, chunk):
        """
        Process multiple samples at once.

        Args:
            chunk: array of shape (n_samples, n_channels)

        Returns:
            list of result dicts from process_sample
        """
        results = []
        for sample in chunk:
            results.append(self.process_sample(sample))
        return results

    def get_status(self):
        """Get current processor status."""
        return {
            "best_channel": self.best_channel,
            "iaf": self.pll.center_hz,
            "pll_freq": self.pll.omega / (2.0 * math.pi),
            "lock": self.pll.get_lock_metric(),
            "buffers_ready": self.buffers_ready,
        }


_BEEP_WAV_CACHE = {}


def _make_wav_bytes(freq_hz, duration_ms, sample_rate=44100, volume=0.2):
    import io
    import wave
    import struct

    n = int(sample_rate * duration_ms / 1000.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n):
            t = i / sample_rate
            sample = int(32767 * volume * math.sin(2 * math.pi * freq_hz * t))
            wf.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


def beep(f_hz=880, ms=20):
    """Play a beep sound (cross-platform, non-blocking).

    Prefer winsound.PlaySound (async) on Windows to avoid blocking; fall back to
    simpleaudio, then winsound.Beep if needed.
    """
    try:
        import winsound
        key = (int(f_hz), int(ms))
        wav = _BEEP_WAV_CACHE.get(key)
        if wav is None:
            wav = _make_wav_bytes(key[0], key[1])
            _BEEP_WAV_CACHE[key] = wav
        winsound.PlaySound(wav, winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NOSTOP)
        return
    except Exception:
        pass

    try:
        import simpleaudio as sa
        fs = 44100
        t = np.linspace(0, ms / 1000.0, int(fs * ms / 1000.0), endpoint=False)
        x = (0.2 * np.sin(2 * np.pi * f_hz * t)).astype(np.float32)
        audio = (x * 32767).astype(np.int16)
        sa.play_buffer(audio, 1, 2, fs)
        return
    except Exception:
        pass

    try:
        import winsound
        winsound.Beep(int(f_hz), int(ms))
        return
    except Exception:
        pass

    print("\a", end="", flush=True)
