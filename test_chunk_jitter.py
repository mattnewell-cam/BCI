"""
Measure LSL chunk delivery jitter.

Pulls EEG chunks in a tight loop (similar to lock_live_2) and records
the size and wall-clock timing of each chunk.  After DURATION_S seconds,
prints statistics and a histogram showing how bursty the stream really is.

Usage:
    python test_chunk_jitter.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pylsl import StreamInlet
from utils import find_eeg_stream

DURATION_S = 10
MAX_SAMPLES = 256  # same as lock_live_2


def main():
    s = find_eeg_stream()
    inlet = StreamInlet(s, max_buflen=2)

    info = inlet.info()
    fs = int(info.nominal_srate())
    print(f"Connected: fs={fs} Hz, name={info.name()}")

    chunk_sizes = []
    pull_times = []       # wall-clock time of each pull
    pull_intervals = []   # wall-clock gap between successive pulls

    t_start = time.perf_counter()
    last_pull = t_start

    print(f"Recording chunk stats for {DURATION_S}s...")

    while time.perf_counter() - t_start < DURATION_S:
        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=MAX_SAMPLES)
        now = time.perf_counter()

        if not chunk:
            time.sleep(0.001)
            continue

        n = len(chunk)
        chunk_sizes.append(n)
        pull_times.append(now - t_start)
        pull_intervals.append(now - last_pull)
        last_pull = now

    chunk_sizes = np.array(chunk_sizes)
    pull_intervals = np.array(pull_intervals) * 1000  # ms

    total_samples = chunk_sizes.sum()
    effective_rate = total_samples / DURATION_S

    print(f"\n{'='*60}")
    print(f"Results  ({DURATION_S}s, {len(chunk_sizes)} pulls, "
          f"{total_samples} samples, effective {effective_rate:.1f} Hz)")
    print(f"{'='*60}")

    print(f"\nChunk sizes (samples):")
    print(f"  mean:   {chunk_sizes.mean():.1f}")
    print(f"  median: {np.median(chunk_sizes):.0f}")
    print(f"  std:    {chunk_sizes.std():.1f}")
    print(f"  min:    {chunk_sizes.min()}")
    print(f"  max:    {chunk_sizes.max()}")

    print(f"\nInter-pull intervals (ms):")
    print(f"  mean:   {pull_intervals.mean():.2f}")
    print(f"  median: {np.median(pull_intervals):.2f}")
    print(f"  std:    {pull_intervals.std():.2f}")
    print(f"  min:    {pull_intervals.min():.2f}")
    print(f"  max:    {pull_intervals.max():.2f}")

    # Chunk size histogram
    bins_cs = [1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 257]
    counts_cs, _ = np.histogram(chunk_sizes, bins=bins_cs)
    print(f"\nChunk size histogram:")
    for i in range(len(bins_cs) - 1):
        pct = counts_cs[i] / len(chunk_sizes) * 100
        bar = "#" * int(pct / 2)
        print(f"  {bins_cs[i]:>3d}-{bins_cs[i+1]-1:>3d}: "
              f"{counts_cs[i]:>5d} ({pct:5.1f}%) {bar}")

    # Inter-pull interval histogram
    bins_ip = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    counts_ip, _ = np.histogram(pull_intervals, bins=bins_ip)
    print(f"\nInter-pull interval histogram (ms):")
    for i in range(len(bins_ip) - 1):
        pct = counts_ip[i] / len(pull_intervals) * 100
        bar = "#" * int(pct / 2)
        print(f"  {bins_ip[i]:>3d}-{bins_ip[i+1]-1:>3d}ms: "
              f"{counts_ip[i]:>5d} ({pct:5.1f}%) {bar}")

    # Stalls: pulls with gap > 50ms
    stalls = pull_intervals[pull_intervals > 50]
    print(f"\nStalls (>50ms gap): {len(stalls)} "
          f"({len(stalls)/len(pull_intervals)*100:.1f}%)")
    if len(stalls) > 0:
        print(f"  mean: {stalls.mean():.1f}ms  max: {stalls.max():.1f}ms")
        # samples lost per stall
        stall_mask = pull_intervals > 50
        stall_chunks = chunk_sizes[stall_mask]
        print(f"  chunk sizes during stalls: "
              f"mean={stall_chunks.mean():.0f} max={stall_chunks.max()}")


if __name__ == "__main__":
    main()
