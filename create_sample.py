"""
EEG Recording Script

1. Prompts for recording name and duration
2. Waits for all 4 channels to show good contact for 2+ seconds
3. Records EEG data with live waveform and countdown
4. Saves to recordings/ directory as .npz
"""

import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi

from pylsl import StreamInlet
from utils import (
    find_eeg_stream,
    get_channel_labels,
    assess_channel_quality,
    get_quality_color,
)


def wait_for_good_contact(inlet, fs, n_eeg, ch_labels):
    """
    Wait until all channels show good contact for 2+ seconds.
    Shows a simple visual indicator.
    """
    print("\nWaiting for good contact on all channels...")
    print("Ensure all electrodes are properly seated.\n")

    buf_len = int(fs * 2)
    buffers = [deque(maxlen=buf_len) for _ in range(n_eeg)]
    quality_history = [deque(maxlen=15) for _ in range(n_eeg)]

    # Track how long all channels have been green
    all_green_since = None
    required_green_duration = 2.0

    # Notch filter
    sos_notch = butter(2, [48, 52], btype="bandstop", fs=fs, output="sos")
    notch_states = [sosfilt_zi(sos_notch) * 0.0 for _ in range(n_eeg)]

    # Simple matplotlib display
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(-0.5, n_eeg - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(n_eeg))
    ax.set_xticklabels(ch_labels[:n_eeg])
    ax.set_ylabel("Quality")
    ax.set_title("Contact Quality - Waiting for all GREEN for 2 seconds...")
    ax.axhline(0.7, color="green", linestyle="--", alpha=0.5, label="Good threshold")

    bars = ax.bar(range(n_eeg), [0] * n_eeg, color=["#e74c3c"] * n_eeg)
    status_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center", va="top",
                          fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.ion()
    plt.show()

    ready = False
    last_update = time.time()

    while not ready:
        # Pull all available samples
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=256)
            if not chunk:
                break
            chunk = np.array(chunk)
            for i in range(n_eeg):
                if i < chunk.shape[1]:
                    filtered, notch_states[i] = sosfilt(sos_notch, chunk[:, i], zi=notch_states[i])
                    buffers[i].extend(filtered)

        # Update display at 200ms intervals
        now = time.time()
        if now - last_update < 0.2:
            plt.pause(0.01)
            continue
        last_update = now

        # Check if buffers have enough data
        if any(len(buf) < buf_len // 2 for buf in buffers):
            plt.pause(0.01)
            continue

        # Assess each channel
        all_good = True
        for i in range(n_eeg):
            data = np.array(buffers[i])
            quality, _, _ = assess_channel_quality(data, fs)
            quality_history[i].append(quality)
            smooth_quality = np.mean(quality_history[i])

            bars[i].set_height(smooth_quality)
            bars[i].set_color(get_quality_color(smooth_quality))

            if smooth_quality < 0.7:
                all_good = False

        # Track green duration
        if all_good:
            if all_green_since is None:
                all_green_since = time.time()
            green_duration = time.time() - all_green_since
            remaining = max(0, required_green_duration - green_duration)
            status_text.set_text(f"All good! Hold steady... {remaining:.1f}s")
            status_text.set_color("#2ecc71")

            if green_duration >= required_green_duration:
                ready = True
        else:
            all_green_since = None
            status_text.set_text("Adjust electrodes...")
            status_text.set_color("#e74c3c")

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.close(fig)
    print("Contact quality confirmed!")
    return notch_states


def record_eeg(inlet, fs, n_eeg, ch_labels, duration, notch_states):
    """
    Record EEG data for specified duration with live waveform display.
    Returns recorded data and timestamps.
    """
    print(f"\nRecording for {duration} seconds...")

    # Storage for recorded data
    recorded_data = [[] for _ in range(n_eeg)]
    recorded_timestamps = []

    # Display buffer (2 seconds rolling)
    display_buf_len = int(fs * 2)
    display_buffers = [deque(maxlen=display_buf_len) for _ in range(n_eeg)]

    # Notch filter
    sos_notch = butter(2, [48, 52], btype="bandstop", fs=fs, output="sos")

    # Setup figure
    fig, axes = plt.subplots(n_eeg, 1, figsize=(10, 6), sharex=True)
    if n_eeg == 1:
        axes = [axes]

    fig.suptitle("Recording EEG", fontsize=14, fontweight="bold")

    lines = []
    for i, ax in enumerate(axes):
        line, = ax.plot([], [], lw=0.8, color="#3498db")
        lines.append(line)
        ax.set_ylabel(ch_labels[i] if i < len(ch_labels) else f"Ch{i}")
        ax.set_xlim(-2, 0)
        ax.set_ylim(-100, 100)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    # Countdown and progress text
    countdown_text = fig.text(0.5, 0.02, "", ha="center", fontsize=14, fontweight="bold")
    progress_bar_ax = fig.add_axes([0.1, 0.93, 0.8, 0.02])
    progress_bar_ax.set_xlim(0, 1)
    progress_bar_ax.set_ylim(0, 1)
    progress_bar_ax.axis("off")
    progress_rect = plt.Rectangle((0, 0), 0, 1, color="#3498db")
    progress_bar_ax.add_patch(progress_rect)
    progress_bar_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="gray", linewidth=2))

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.ion()
    plt.show()

    start_time = time.time()
    last_update = start_time

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        # Pull all available samples
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=256)
            if not chunk:
                break
            chunk = np.array(chunk)

            # Store raw data
            for i in range(n_eeg):
                if i < chunk.shape[1]:
                    recorded_data[i].extend(chunk[:, i])
                    # Filter for display only
                    filtered, notch_states[i] = sosfilt(sos_notch, chunk[:, i], zi=notch_states[i])
                    display_buffers[i].extend(filtered)

            if timestamps:
                recorded_timestamps.extend(timestamps)

        # Update display at 200ms intervals
        now = time.time()
        if now - last_update < 0.2:
            plt.pause(0.01)
            continue
        last_update = now

        # Update waveforms
        for i in range(n_eeg):
            if len(display_buffers[i]) > 10:
                data = np.array(display_buffers[i])
                t = np.linspace(-len(data) / fs, 0, len(data))
                lines[i].set_data(t, data)

                # Auto-scale
                std = np.std(data)
                ylim = max(50, min(500, std * 4))
                axes[i].set_ylim(-ylim, ylim)

        # Update countdown
        remaining = max(0, duration - elapsed)
        countdown_text.set_text(f"Recording: {remaining:.1f}s remaining")

        # Update progress bar
        progress = elapsed / duration
        progress_rect.set_width(progress)

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.close(fig)

    # Convert to numpy arrays
    data_array = np.array([np.array(ch) for ch in recorded_data])
    timestamps_array = np.array(recorded_timestamps)

    print(f"Recording complete! Captured {data_array.shape[1]} samples per channel.")
    return data_array, timestamps_array


def save_recording(data, timestamps, name, fs, ch_labels, recordings_dir="recordings"):
    """Save recording to .npz file with metadata."""
    os.makedirs(recordings_dir, exist_ok=True)

    # Create filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    filename = f"{safe_name}_{timestamp_str}.npz"
    filepath = os.path.join(recordings_dir, filename)

    # Save with metadata
    np.savez(
        filepath,
        data=data,                    # Shape: (n_channels, n_samples)
        timestamps=timestamps,         # LSL timestamps
        sample_rate=fs,
        channel_labels=ch_labels,
        recording_name=name,
        recording_date=datetime.now().isoformat(),
        duration_seconds=data.shape[1] / fs if data.shape[1] > 0 else 0,
    )

    print(f"\nSaved to: {filepath}")
    print(f"  Channels: {len(ch_labels)}")
    print(f"  Samples: {data.shape[1]}")
    print(f"  Duration: {data.shape[1] / fs:.2f}s")
    print(f"  Sample rate: {fs} Hz")

    return filepath


def main():
    # Get recording parameters from user
    print("=" * 50)
    print("EEG Recording Tool")
    print("=" * 50)

    name = input("\nRecording name: ").strip()
    if not name:
        name = "unnamed"

    while True:
        try:
            duration = float(input("Duration (seconds): ").strip())
            if duration <= 0:
                print("Duration must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Connect to EEG stream
    print("\nSearching for EEG stream...")
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=2)

    info = inlet.info()
    fs = float(info.nominal_srate())
    n_channels = int(info.channel_count())
    ch_labels = get_channel_labels(info, n_channels)

    print(f"Connected: {info.name()}")
    print(f"Sample rate: {fs:.1f} Hz, Channels: {n_channels}")
    print(f"Channel labels: {', '.join(ch_labels)}")

    n_eeg = min(4, n_channels)
    ch_labels = ch_labels[:n_eeg]

    # Phase 1: Wait for good contact
    notch_states = wait_for_good_contact(inlet, fs, n_eeg, ch_labels)

    # Small pause before recording
    print("\nStarting recording in 1 second...")
    time.sleep(1)

    # Phase 2: Record
    data, timestamps = record_eeg(inlet, fs, n_eeg, ch_labels, duration, notch_states)

    # Phase 3: Save
    save_recording(data, timestamps, name, fs, ch_labels)

    print("\nDone!")


if __name__ == "__main__":
    main()
