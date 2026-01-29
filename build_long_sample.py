"""
Long EEG recording tool.

- Waits for good contact on all channels
- Records indefinitely in 5-minute chunks
- Appends to recordings/full_night.npz after each chunk
- Exits cleanly if the Muse disconnects
"""

import os
import time
import tempfile
from datetime import datetime

import numpy as np
from pylsl import StreamInlet

from utils import find_eeg_stream, get_channel_labels
from create_sample import wait_for_good_contact


SAVE_INTERVAL_S = 20.0  # 5 minutes
DISCONNECT_TIMEOUT_S = 10.0
OUTPUT_DIR = "recordings"
OUTPUT_NAME = "full_night.npz"


def _append_and_save(out_path, data_chunk, ts_chunk, fs, ch_labels, start_iso):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        existing = np.load(out_path)
        try:
            existing_data = existing["data"]
            existing_ts = existing["timestamps"]
            existing_fs = float(existing.get("sample_rate", fs))
            existing_labels = list(existing.get("channel_labels", ch_labels))

            if int(existing_data.shape[0]) != int(data_chunk.shape[0]):
                raise RuntimeError("Channel count mismatch with existing file.")
            if abs(existing_fs - fs) > 1e-6:
                raise RuntimeError("Sample rate mismatch with existing file.")

            data_all = np.concatenate([existing_data, data_chunk], axis=1)
            ts_all = np.concatenate([existing_ts, ts_chunk], axis=0)
            ch_labels = existing_labels
        finally:
            existing.close()
    else:
        data_all = data_chunk
        ts_all = ts_chunk

    tmp_dir = os.path.dirname(out_path)
    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".npz", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    np.savez(
        tmp_path,
        data=data_all,
        timestamps=ts_all,
        sample_rate=fs,
        channel_labels=ch_labels,
        recording_name="full_night",
        recording_date=start_iso,
        duration_seconds=data_all.shape[1] / fs if data_all.shape[1] > 0 else 0,
    )
    os.replace(tmp_path, out_path)


def _record_interval(inlet, fs, n_eeg, duration_s):
    recorded_data = [[] for _ in range(n_eeg)]
    recorded_ts = []

    start_time = time.time()
    last_sample_time = start_time

    while True:
        now = time.time()
        if now - start_time >= duration_s:
            break

        chunk, timestamps = inlet.pull_chunk(timeout=0.5, max_samples=256)
        if chunk:
            last_sample_time = now
            chunk = np.asarray(chunk)
            for i in range(n_eeg):
                if i < chunk.shape[1]:
                    recorded_data[i].extend(chunk[:, i])
            if timestamps:
                recorded_ts.extend(timestamps)
        else:
            if now - last_sample_time >= DISCONNECT_TIMEOUT_S:
                raise RuntimeError("No samples received; device likely disconnected.")

    data_array = np.array([np.asarray(ch, dtype=np.float32) for ch in recorded_data])
    ts_array = np.asarray(recorded_ts, dtype=np.float64)
    return data_array, ts_array


def main():
    print("=" * 50)
    print("Long EEG Recording Tool")
    print("=" * 50)

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
    wait_for_good_contact(inlet, fs, n_eeg, ch_labels)

    # Phase 2: Record indefinitely in chunks
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    start_iso = datetime.now().isoformat()
    chunk_idx = 0

    print("\nStarting long recording. Saving every 5 minutes...")

    try:
        while True:
            chunk_idx += 1
            print(f"\nChunk {chunk_idx}: recording {SAVE_INTERVAL_S:.0f}s...")
            data_chunk, ts_chunk = _record_interval(inlet, fs, n_eeg, SAVE_INTERVAL_S)
            _append_and_save(out_path, data_chunk, ts_chunk, fs, ch_labels, start_iso)
            print(f"Saved chunk {chunk_idx} -> {out_path}")
    except Exception as exc:
        print(f"\nStopping: {exc}")
        # Best-effort save on exit if we have partial data
        try:
            if "data_chunk" in locals() and data_chunk.shape[1] > 0:
                _append_and_save(out_path, data_chunk, ts_chunk, fs, ch_labels, start_iso)
                print(f"Saved final partial chunk -> {out_path}")
        except Exception as save_exc:
            print(f"Failed to save final chunk: {save_exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
