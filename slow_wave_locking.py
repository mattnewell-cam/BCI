"""
Slow Wave Activity (SWA) Locking - Real-time slow wave detection with pink noise bursts.

Monitors TP10 for slow waves (0.6-1.1 Hz), and when the negative magnitude exceeds
a threshold, plays pink noise bursts (50% of the time via coin flip) and records
7 seconds of raw data for later analysis.

Reconnects to BlueMuse automatically if the Muse disconnects.
"""

import io
import json
import math
import os
import random
import subprocess
import struct
import time
import wave
from collections import deque
from datetime import datetime

import numpy as np
from pylsl import StreamInlet
from scipy.signal import butter, sosfilt, sosfilt_zi

from utils import find_eeg_stream, get_channel_labels
from create_sample import wait_for_good_contact

# --- Configuration ---
BANDPASS_LOW = 0.6  # Hz
BANDPASS_HIGH = 1.1  # Hz
NEGATIVE_THRESHOLD = -25  # Trigger when filtered signal goes below this
POWER_THRESHOLD = 500  # Broadband variance threshold for facial movement detection
POWER_WINDOW_S = 1.0  # Window for computing broadband power
COOLDOWN_S = 3.0  # Cooldown after facial movement detected
REFRACTORY_S = 7.0  # Time after trigger before next trigger (data collection period)
DATA_COLLECTION_S = 7.0  # Seconds of raw data to save after trigger

# Pink noise burst settings
BURST_DURATION_MS = 50
BURST_FADE_MS = 5
BURST_COUNT = 3
BURST_INTERVAL_MS = 120  # Time between start of each burst

# Reconnection settings (from build_long_sample.py)
DISCONNECT_TIMEOUT_S = 10.0
RECONNECT_PAUSE_S = 5.0
RECONNECT_STREAM_TIMEOUT_S = 1200
MAX_RECONNECT_ATTEMPTS = 50
MUSE_MAC_ADDRESS = "00:55:da:b5:f6:a4"

# Output
OUTPUT_DIR = "recordings/swa"


def generate_pink_noise(duration_ms, sample_rate=44100, fade_ms=5, volume=0.3):
    """
    Generate pink noise (1/f spectrum) with fade in/out.

    Uses the Voss-McCartney algorithm for efficient 1/f noise generation.
    """
    n_samples = int(sample_rate * duration_ms / 1000.0)
    fade_samples = int(sample_rate * fade_ms / 1000.0)

    # Voss-McCartney algorithm for pink noise
    # Use multiple rows of white noise at different update rates
    n_rows = 16
    max_key = (1 << n_rows) - 1

    pink = np.zeros(n_samples)
    running_sum = 0.0
    rows = np.zeros(n_rows)

    for i in range(n_samples):
        # Determine which rows to update based on trailing zeros in binary representation
        key = i & max_key
        changed_bits = (key ^ (key + 1)) if i > 0 else max_key

        for row in range(n_rows):
            if changed_bits & (1 << row):
                running_sum -= rows[row]
                rows[row] = random.gauss(0, 1)
                running_sum += rows[row]

        # Add white noise component
        white = random.gauss(0, 1)
        pink[i] = (running_sum + white) / (n_rows + 1)

    # Normalize
    pink = pink / (np.abs(pink).max() + 1e-10)

    # Apply fade in/out
    if fade_samples > 0 and fade_samples < n_samples // 2:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        pink[:fade_samples] *= fade_in
        pink[-fade_samples:] *= fade_out

    # Scale by volume
    pink = pink * volume

    return pink


def make_pink_noise_wav(duration_ms, sample_rate=44100, fade_ms=5, volume=0.3):
    """Create WAV bytes for pink noise burst."""
    pink = generate_pink_noise(duration_ms, sample_rate, fade_ms, volume)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        audio = (pink * 32767).astype(np.int16)
        wf.writeframes(audio.tobytes())

    return buf.getvalue()


# Cache for pink noise WAV
_PINK_NOISE_CACHE = None


def play_pink_burst():
    """Play a single pink noise burst (non-blocking)."""
    global _PINK_NOISE_CACHE

    if _PINK_NOISE_CACHE is None:
        _PINK_NOISE_CACHE = make_pink_noise_wav(
            BURST_DURATION_MS,
            fade_ms=BURST_FADE_MS,
            volume=0.3
        )

    try:
        import winsound
        winsound.PlaySound(
            _PINK_NOISE_CACHE,
            winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NOSTOP
        )
        return True
    except Exception:
        pass

    try:
        import simpleaudio as sa
        # Decode WAV for simpleaudio
        buf = io.BytesIO(_PINK_NOISE_CACHE)
        with wave.open(buf, "rb") as wf:
            audio_data = wf.readframes(wf.getnframes())
            fs = wf.getframerate()
        sa.play_buffer(audio_data, 1, 2, fs)
        return True
    except Exception:
        pass

    print("\a", end="", flush=True)
    return False


def schedule_bursts(n_bursts, interval_ms):
    """
    Schedule multiple pink noise bursts with specified interval.
    Returns a list of scheduled times (relative to now).
    """
    return [i * interval_ms / 1000.0 for i in range(n_bursts)]


def _restart_bluemuse():
    """Restart BlueMuse streaming via its URI protocol."""
    if MUSE_MAC_ADDRESS:
        uri = f"bluemuse://start?addresses={MUSE_MAC_ADDRESS}"
    else:
        uri = "bluemuse:"
    print(f"  Launching BlueMuse: {uri}")
    try:
        subprocess.run(
            ["cmd.exe", "/c", "start", "", uri],
            timeout=10,
            capture_output=True,
        )
    except FileNotFoundError:
        subprocess.run(
            ["start", "", uri],
            shell=True,
            timeout=10,
            capture_output=True,
        )


def _reconnect():
    """Restart BlueMuse and wait for a new LSL EEG stream. Returns a new StreamInlet."""
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        print(f"\nReconnect attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}...")
        _restart_bluemuse()
        print(f"  Waiting up to {RECONNECT_STREAM_TIMEOUT_S:.0f}s for LSL stream...")

        deadline = time.time() + RECONNECT_STREAM_TIMEOUT_S
        while time.time() < deadline:
            try:
                s = find_eeg_stream()
                inlet = StreamInlet(s, max_buflen=2)
                inlet.pull_chunk(timeout=0.0, max_samples=4096)
                info = inlet.info()
                print(f"  Reconnected: {info.name()}")
                return inlet, info
            except RuntimeError:
                time.sleep(RECONNECT_PAUSE_S)

        print(f"  Attempt {attempt} failed, no stream found.")

    raise RuntimeError(f"Could not reconnect after {MAX_RECONNECT_ATTEMPTS} attempts.")


def save_trial(timestamp, played_sounds, raw_data, fs):
    """Save a trial to JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"swa_beeps_{date_str}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Load existing data or create new
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = {
            "sample_rate": fs,
            "channel": "TP10",
            "bandpass": [BANDPASS_LOW, BANDPASS_HIGH],
            "threshold": NEGATIVE_THRESHOLD,
            "trials": []
        }

    # Add new trial
    trial = {
        "timestamp": timestamp,
        "played_sounds": played_sounds,
        "raw_data": raw_data.tolist() if isinstance(raw_data, np.ndarray) else raw_data
    }
    data["trials"].append(trial)

    # Save
    with open(filepath, "w") as f:
        json.dump(data, f)

    return filepath


class SlowWaveProcessor:
    """
    Processes EEG for slow wave detection.

    Handles:
    - Bandpass filtering (0.6-1.1 Hz)
    - Threshold detection
    - Power-based artifact rejection
    - Cooldown/refractory periods
    """

    def __init__(self, fs, tp10_idx=3):
        self.fs = int(fs)
        self.tp10_idx = tp10_idx

        # Bandpass filter (0.6-1.1 Hz) using SOS for stability
        self.sos = butter(4, [BANDPASS_LOW, BANDPASS_HIGH], btype='band', fs=self.fs, output='sos')
        self.zi = sosfilt_zi(self.sos) * 0.0

        # Power computation buffer (1 second of raw data)
        self.power_window_samples = int(POWER_WINDOW_S * self.fs)
        self.raw_buffer = deque(maxlen=self.power_window_samples)

        # State
        self.cooldown_until = 0.0
        self.refractory_until = 0.0
        self.current_time = 0.0

        # Data collection
        self.collecting = False
        self.collection_start = 0.0
        self.collection_data = []
        self.collection_played_sounds = False
        self.collection_samples_needed = int(DATA_COLLECTION_S * self.fs)

        # Burst scheduling
        self.pending_bursts = []

    def process_sample(self, sample, wall_time):
        """
        Process a single sample.

        Args:
            sample: Full multi-channel sample
            wall_time: Current wall clock time

        Returns:
            dict with:
                - filtered: bandpassed value
                - power: current broadband power
                - triggered: whether threshold was crossed
                - play_sounds: whether to play sounds (coin flip result)
                - artifact: whether artifact was detected
                - collecting: whether we're collecting data
                - trial_complete: dict with trial data if collection just finished, else None
        """
        self.current_time = wall_time

        # Get TP10 value
        x = float(sample[self.tp10_idx])

        # Store raw for power computation and data collection
        self.raw_buffer.append(x)

        if self.collecting:
            self.collection_data.append(x)

        # Apply bandpass filter
        x_arr = np.array([x], dtype=np.float64)
        filtered_arr, self.zi = sosfilt(self.sos, x_arr, zi=self.zi)
        filtered = float(filtered_arr[0])

        # Compute broadband power
        power = 0.0
        if len(self.raw_buffer) >= self.power_window_samples:
            power = np.var(list(self.raw_buffer))

        result = {
            "filtered": filtered,
            "power": power,
            "triggered": False,
            "play_sounds": False,
            "artifact": False,
            "collecting": self.collecting,
            "trial_complete": None,
        }

        # Check if data collection is complete
        if self.collecting and len(self.collection_data) >= self.collection_samples_needed:
            result["trial_complete"] = {
                "timestamp": datetime.fromtimestamp(self.collection_start).isoformat(),
                "played_sounds": self.collection_played_sounds,
                "raw_data": np.array(self.collection_data[:self.collection_samples_needed])
            }
            self.collecting = False
            self.collection_data = []

        # Process pending bursts
        bursts_to_play = []
        remaining_bursts = []
        for burst_time in self.pending_bursts:
            if wall_time >= burst_time:
                bursts_to_play.append(burst_time)
            else:
                remaining_bursts.append(burst_time)
        self.pending_bursts = remaining_bursts

        for _ in bursts_to_play:
            play_pink_burst()

        # Check for cooldown/refractory
        if wall_time < self.cooldown_until or wall_time < self.refractory_until:
            return result

        # Check artifact (high power)
        if power > POWER_THRESHOLD:
            result["artifact"] = True
            self.cooldown_until = wall_time + COOLDOWN_S
            # Cancel any pending bursts
            self.pending_bursts = []
            return result

        # Check threshold crossing
        if filtered < NEGATIVE_THRESHOLD:
            result["triggered"] = True

            # Coin flip - play sounds 50% of the time
            play_sounds = random.random() < 0.5
            result["play_sounds"] = play_sounds

            if play_sounds:
                # Schedule bursts
                burst_times = schedule_bursts(BURST_COUNT, BURST_INTERVAL_MS)
                self.pending_bursts = [wall_time + t for t in burst_times]

            # Start data collection
            self.collecting = True
            self.collection_start = wall_time
            self.collection_data = [x]  # Include triggering sample
            self.collection_played_sounds = play_sounds

            # Set refractory period
            self.refractory_until = wall_time + REFRACTORY_S

        return result

    def check_artifact_after_trigger(self, power):
        """
        Check if artifact occurred after trigger (for the first burst case).
        If so, cancel remaining bursts and set cooldown.
        """
        if power > POWER_THRESHOLD and self.pending_bursts:
            self.pending_bursts = []
            self.cooldown_until = self.current_time + COOLDOWN_S
            return True
        return False


def main():
    print("=" * 60)
    print("Slow Wave Activity (SWA) Locking")
    print("=" * 60)
    print(f"Bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"Threshold: {NEGATIVE_THRESHOLD}")
    print(f"Power threshold (artifact): {POWER_THRESHOLD}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Connect to EEG stream
    print("Connecting to BlueMuse...")
    inlet, info = _reconnect()
    fs = float(info.nominal_srate())
    n_channels = int(info.channel_count())
    ch_labels = get_channel_labels(info, n_channels)

    print(f"Connected: {info.name()}")
    print(f"Sample rate: {fs:.1f} Hz, Channels: {n_channels}")
    print(f"Channel labels: {', '.join(ch_labels)}")

    # Find TP10 index
    tp10_idx = 3  # Default
    if "TP10" in ch_labels:
        tp10_idx = ch_labels.index("TP10")
    print(f"Using TP10 (channel index {tp10_idx})")

    n_eeg = min(4, n_channels)

    # Wait for good electrode contact
    wait_for_good_contact(inlet, fs, n_eeg, ch_labels[:n_eeg])

    # Create processor
    processor = SlowWaveProcessor(fs, tp10_idx)

    # Stats
    n_triggers = 0
    n_sounds_played = 0
    n_artifacts = 0
    n_trials_saved = 0
    last_status_time = time.time()
    last_sample_time = time.time()

    print("\nMonitoring for slow waves... (Ctrl+C to stop)")
    print("-" * 60)

    try:
        while True:
            now = time.time()

            # Pull samples
            chunk, timestamps = inlet.pull_chunk(timeout=0.5, max_samples=256)

            if chunk:
                last_sample_time = now

                for i, sample in enumerate(chunk):
                    # Use LSL timestamp if available, else wall clock
                    wall_time = timestamps[i] if timestamps else now

                    result = processor.process_sample(sample, now)

                    if result["artifact"]:
                        n_artifacts += 1
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Artifact detected (power={result['power']:.0f}), cooldown...")

                    if result["triggered"]:
                        n_triggers += 1
                        action = "PLAYING SOUNDS" if result["play_sounds"] else "no sounds (control)"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Triggered! filtered={result['filtered']:.1f}, {action}")
                        if result["play_sounds"]:
                            n_sounds_played += 1

                    # Check for artifact after trigger (cancel remaining bursts)
                    if result["collecting"] and result["power"] > POWER_THRESHOLD:
                        if processor.check_artifact_after_trigger(result["power"]):
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Artifact during collection, cancelled remaining bursts")

                    if result["trial_complete"]:
                        trial = result["trial_complete"]
                        filepath = save_trial(
                            trial["timestamp"],
                            trial["played_sounds"],
                            trial["raw_data"],
                            fs
                        )
                        n_trials_saved += 1
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trial saved to {filepath} (total: {n_trials_saved})")

            else:
                # Check for disconnect
                if now - last_sample_time >= DISCONNECT_TIMEOUT_S:
                    print("\nDisconnect detected, attempting to reconnect...")
                    inlet, info = _reconnect()
                    fs = float(info.nominal_srate())
                    processor = SlowWaveProcessor(fs, tp10_idx)
                    last_sample_time = now

            # Status update every 30 seconds
            if now - last_status_time >= 30.0:
                last_status_time = now
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: triggers={n_triggers}, sounds={n_sounds_played}, artifacts={n_artifacts}, saved={n_trials_saved}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Stopped by user")
        print(f"Total triggers: {n_triggers}")
        print(f"Sounds played: {n_sounds_played}")
        print(f"Artifacts detected: {n_artifacts}")
        print(f"Trials saved: {n_trials_saved}")
        print("=" * 60)


if __name__ == "__main__":
    main()
