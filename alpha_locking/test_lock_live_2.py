"""
Real-time replay test for lock_live_2 trough detection + beep scheduling.

Replays npz data through the same streaming bandpass filter, trough detection,
prediction, and beep scheduling used in lock_live_2 — with actual audio output
and real-time pacing.  Chunk delivery mimics LSL: each iteration delivers
however many samples have accumulated since the last pull, producing realistic
burst patterns.

Usage:
    python alpha_locking/test_lock_live_2.py

Requires libportaudio2 for audio (sudo apt install libportaudio2).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sounddevice as sd
from collections import deque
from scipy.signal import sosfilt, sosfilt_zi, argrelmin

try:
    from alpha_locking.lock_live_2 import (
        WAVEFORM_S, WAVEFORM_SUBSAMPLE, WAVEFORM_OUT_NATIVE,
        CHANNEL_TP10, ASSUMED_LATENCY_S, MIN_DELAY_S,
        AUDIO_FS, BEEP_DIAG_ROUNDS, BEEP_DIAG_PATH_TEST,
        schedule_beep, _audio_callback,
        train_model, detect_troughs_range, save_beep_diag,
    )
except ImportError:
    from lock_live_2 import (
        WAVEFORM_S, WAVEFORM_SUBSAMPLE, WAVEFORM_OUT_NATIVE,
        CHANNEL_TP10, ASSUMED_LATENCY_S, MIN_DELAY_S,
        AUDIO_FS, BEEP_DIAG_ROUNDS, BEEP_DIAG_PATH_TEST,
        schedule_beep, _audio_callback,
        train_model, detect_troughs_range, save_beep_diag,
    )

TEST_DURATION_S = 30


def run_realtime_test():
    """Replay recording data in real-time with actual audio beeps."""
    print("=" * 60)
    print("Real-time replay test: lock_live_2 with audio")
    print("=" * 60)

    # --- Train model ---
    print("\nTraining model...")
    model = train_model()
    W = model["W"]
    feat_mean = model["feat_mean"]
    feat_std = model["feat_std"]
    out_mean = model["out_mean"]
    out_std = model["out_std"]
    sos = model["sos"]
    half_cycle = model["half_cycle"]

    fs = 256
    waveform_native = int(WAVEFORM_S * fs)
    beep_cooldown_samples = half_cycle * 2

    # --- Load signal ---
    train_data = os.path.join(os.path.dirname(__file__), "..",
                              "recordings", "full_night_350_1000.npz")
    rec = np.load(train_data, allow_pickle=True)
    signal = rec["data"][CHANNEL_TP10].astype(np.float64)

    test_samples = min(TEST_DURATION_S * fs, len(signal))
    signal = signal[:test_samples]
    print(f"Test signal: {test_samples} samples ({test_samples / fs:.1f}s)")
    print(f"half_cycle={half_cycle}, cooldown={beep_cooldown_samples} samples "
          f"({beep_cooldown_samples / fs * 1000:.0f}ms)")

    # --- Open audio stream (same as lock_live_2) ---
    audio_stream = sd.OutputStream(
        samplerate=AUDIO_FS, channels=1, callback=_audio_callback,
        dtype="float32", latency="low",
    )
    audio_stream.start()
    schedule_beep(0.0)  # test beep
    print("Audio stream started")

    # --- Streaming filter state ---
    zi = sosfilt_zi(sos) * 0.0
    filt_buf = deque(maxlen=int(2.0 * fs))

    # --- Trough detection state ---
    last_trough_abs = -1000
    sample_count = 0
    prediction_count = 0
    last_checked_candidate = -1
    last_beep_target_sample = -1000
    last_beep_play_time = -1000.0
    beep_cooldown_s = beep_cooldown_samples / fs

    # --- Diagnostic tracking ---
    beep_count = 0
    beep_diag_captures = {}
    trough_spacings = []
    beep_play_times = []   # time.perf_counter() + delay_s
    chunk_sizes = []
    t0_loop = time.perf_counter()
    last_loop_end = t0_loop

    # Beep-gap diagnostics (same style as lock_live_2)
    since_beep = {
        "idle": 0.0,
        "sleep": 0.0,
        "filter": 0.0,
        "ring": 0.0,
        "candidates": 0.0,
        "trough_scan": 0.0,
        "predict": 0.0,
        "schedule": 0.0,
        "other": 0.0,
    }
    since_beep_counts = {
        "chunks": 0,
        "samples": 0,
        "troughs": 0,
        "predictions": 0,
        "pred_lt2_troughs": 0,
        "delay_too_short": 0,
        "cooldown_sample_block": 0,
        "cooldown_play_block": 0,
        "scheduled": 0,
    }
    last_beep_wall = None
    beep_gap_threshold_s = 0.200

    # --- Real-time replay: deliver samples at 256 Hz pace ---
    samples_delivered = 0
    t_start = time.perf_counter()

    print(f"\nReplaying {test_samples / fs:.0f}s of data in real-time...")
    print(f"(You should hear beeps)\n")

    while samples_delivered < test_samples:
        t_loop_start = time.perf_counter()
        iter_times = {
            "sleep": 0.0,
            "filter": 0.0,
            "ring": 0.0,
            "candidates": 0.0,
            "trough_scan": 0.0,
            "predict": 0.0,
            "schedule": 0.0,
            "other": 0.0,
        }
        idle_gap = t_loop_start - last_loop_end
        if idle_gap > 0:
            since_beep["idle"] += idle_gap

        # How many samples should have arrived by now?
        elapsed = time.perf_counter() - t_start
        target_delivered = min(int(elapsed * fs), test_samples)
        n_available = target_delivered - samples_delivered

        if n_available <= 0:
            t_sleep_start = time.perf_counter()
            time.sleep(0.001)
            t_sleep = time.perf_counter() - t_sleep_start
            iter_times["sleep"] += t_sleep
            t_loop = time.perf_counter() - t_loop_start
            iter_times["other"] = max(0.0, t_loop - sum(iter_times.values()))
            for k, v in iter_times.items():
                since_beep[k] += v
            last_loop_end = time.perf_counter()
            continue

        # Deliver chunk (mimics pull_chunk behavior)
        new_raw = signal[samples_delivered:samples_delivered + n_available]
        n_new = len(new_raw)
        samples_delivered += n_new
        chunk_sizes.append(n_new)
        since_beep_counts["chunks"] += 1
        since_beep_counts["samples"] += n_new

        # --- Streaming bandpass filter ---
        t_filter_start = time.perf_counter()
        new_filtered, zi = sosfilt(sos, new_raw, zi=zi)
        iter_times["filter"] += time.perf_counter() - t_filter_start

        # --- Append to ring buffer ---
        t_ring_start = time.perf_counter()
        filt_buf.extend(new_filtered)
        sample_count += n_new
        iter_times["ring"] += time.perf_counter() - t_ring_start

        # --- Need enough samples ---
        if len(filt_buf) < waveform_native + half_cycle:
            t_loop = time.perf_counter() - t_loop_start
            iter_times["other"] = max(0.0, t_loop - sum(iter_times.values()))
            for k, v in iter_times.items():
                since_beep[k] += v
            last_loop_end = time.perf_counter()
            continue

        # --- Trough detection: check ALL new candidate positions ---
        t_candidates_start = time.perf_counter()
        filt_arr = np.array(filt_buf)
        buf_len = len(filt_arr)

        rightmost_candidate_idx = buf_len - 1 - half_cycle
        rightmost_candidate_abs = sample_count - 1 - half_cycle

        if last_checked_candidate < 0:
            first_candidate_abs = max(
                sample_count - buf_len + half_cycle,
                rightmost_candidate_abs - n_new + 1
            )
        else:
            first_candidate_abs = last_checked_candidate + 1

        first_candidate_abs = max(first_candidate_abs, rightmost_candidate_abs - n_new + 1)
        first_candidate_abs = max(first_candidate_abs, sample_count - buf_len + half_cycle)

        if first_candidate_abs > rightmost_candidate_abs:
            last_checked_candidate = rightmost_candidate_abs
            continue

        first_candidate_idx = first_candidate_abs - (sample_count - buf_len)
        last_candidate_idx = rightmost_candidate_idx
        n_candidates = last_candidate_idx - first_candidate_idx + 1
        iter_times["candidates"] += time.perf_counter() - t_candidates_start

        t_scan_start = time.perf_counter()
        for trough_abs in detect_troughs_range(
                filt_arr, half_cycle, last_trough_abs, sample_count,
                first_candidate_idx, last_candidate_idx):

            # Track trough spacing
            spacing = trough_abs - last_trough_abs if last_trough_abs >= 0 else 0
            if last_trough_abs >= 0:
                trough_spacings.append(spacing)

            last_trough_abs = trough_abs
            prediction_count += 1
            since_beep_counts["troughs"] += 1
            since_beep_counts["predictions"] += 1

            # --- Prediction ---
            t_pred_start = time.perf_counter()
            input_wave = filt_arr[-waveform_native::WAVEFORM_SUBSAMPLE]
            x = (input_wave - feat_mean) / feat_std
            x = np.append(x, 1.0)
            y_norm = x @ W
            y_pred = y_norm * out_std + out_mean
            iter_times["predict"] += time.perf_counter() - t_pred_start

            pred_troughs = argrelmin(y_pred, order=half_cycle)[0]
            if len(pred_troughs) < 2:
                since_beep_counts["pred_lt2_troughs"] += 1
                continue

            trough_2_offset = pred_troughs[1]
            delay_s = (trough_2_offset / fs) - ASSUMED_LATENCY_S

            if delay_s > MIN_DELAY_S:
                beep_target_sample = sample_count + int(delay_s * fs)
                beep_play_time = time.perf_counter() + delay_s
                sample_ok = beep_target_sample - last_beep_target_sample >= beep_cooldown_samples
                playtime_ok = beep_play_time - last_beep_play_time >= beep_cooldown_s
                if sample_ok and playtime_ok:
                    t_sched_start = time.perf_counter()
                    schedule_beep(delay_s)
                    iter_times["schedule"] += time.perf_counter() - t_sched_start
                    beep_count += 1
                    beep_play_times.append(beep_play_time)
                    since_beep_counts["scheduled"] += 1

                    now_wall = time.perf_counter()
                    if last_beep_wall is not None:
                        gap = now_wall - last_beep_wall
                        if gap > beep_gap_threshold_s:
                            accounted = sum(since_beep.values())
                            unaccounted = max(0.0, gap - accounted)
                            print(f"[beep-gap] total={gap*1000:.1f}ms")
                            print("counts "
                                  f"chunks={since_beep_counts['chunks']} "
                                  f"samples={since_beep_counts['samples']} "
                                  f"troughs={since_beep_counts['troughs']} "
                                  f"pred={since_beep_counts['predictions']} "
                                  f"lt2={since_beep_counts['pred_lt2_troughs']} "
                                  f"delay_short={since_beep_counts['delay_too_short']} "
                                  f"cd_sample={since_beep_counts['cooldown_sample_block']} "
                                  f"cd_play={since_beep_counts['cooldown_play_block']} "
                                  f"scheduled={since_beep_counts['scheduled']}")
                            for k in ["idle", "sleep", "filter", "ring",
                                      "candidates", "trough_scan", "predict",
                                      "schedule", "other"]:
                                v = since_beep[k]
                                if v > 0:
                                    print(f"{k} action: {v*1000:.1f} ms")
                            if unaccounted > 0.0005:
                                print(f"unaccounted action: {unaccounted*1000:.1f} ms")
                    last_beep_wall = now_wall
                    for k in since_beep:
                        since_beep[k] = 0.0
                    for k in since_beep_counts:
                        since_beep_counts[k] = 0

                    # Capture beep diagnostic
                    if beep_count in BEEP_DIAG_ROUNDS:
                        beep_diag_captures[beep_count] = {
                            "filt_signal": filt_arr[-waveform_native:].copy(),
                            "y_pred": y_pred.copy(),
                            "pred_troughs": pred_troughs.copy(),
                            "trough_abs": trough_abs,
                            "sample_count": sample_count,
                            "delay_s": delay_s,
                            "beep_target": beep_target_sample,
                            "last_beep_target": last_beep_target_sample,
                            "prediction_count": prediction_count,
                            "n_new": n_new,
                            "n_candidates": n_candidates,
                            "fs": fs,
                        }
                        print(f"  [beep_diag] captured beep #{beep_count}")

                    last_beep_target_sample = beep_target_sample
                    last_beep_play_time = beep_play_time
                else:
                    if not sample_ok:
                        since_beep_counts["cooldown_sample_block"] += 1
                    if not playtime_ok:
                        since_beep_counts["cooldown_play_block"] += 1
            else:
                since_beep_counts["delay_too_short"] += 1

            # Console output
            if prediction_count % 20 == 0:
                loop_elapsed = time.perf_counter() - t0_loop
                trough_rate = prediction_count / loop_elapsed if loop_elapsed > 0 else 0
                beep_rate = beep_count / loop_elapsed if loop_elapsed > 0 else 0
                print(f"[{prediction_count}] "
                      f"pred_troughs={pred_troughs[:3].tolist()}  "
                      f"delay={delay_s * 1000:.0f}ms  "
                      f"troughs/s={trough_rate:.1f}  "
                      f"beeps/s={beep_rate:.1f}  "
                      f"chunk={n_new}  "
                      f"SC={sample_count}")

        last_checked_candidate = rightmost_candidate_abs
        t_scan = time.perf_counter() - t_scan_start
        iter_times["trough_scan"] += t_scan

        t_loop = time.perf_counter() - t_loop_start
        iter_times["other"] = max(0.0, t_loop - sum(iter_times.values()))
        for k, v in iter_times.items():
            since_beep[k] += v
        last_loop_end = time.perf_counter()

    # --- Wait for final beeps to play ---
    time.sleep(0.5)
    audio_stream.stop()
    audio_stream.close()

    # --- Save diagnostic plot ---
    if beep_diag_captures:
        save_beep_diag(
            beep_diag_captures, trough_spacings,
            beep_play_times, chunk_sizes,
            BEEP_DIAG_PATH_TEST, fs, half_cycle)

    # --- Analysis ---
    print(f"\nResults:")
    print(f"  Troughs detected: {prediction_count}")
    print(f"  Beeps scheduled:  {beep_count}")

    if len(trough_spacings) > 0:
        ts = np.array(trough_spacings)
        print(f"  Trough spacings: mean={ts.mean():.1f} min={ts.min()} max={ts.max()}")

    if len(chunk_sizes) > 0:
        cs = np.array(chunk_sizes)
        print(f"  Chunk sizes: mean={cs.mean():.1f} min={cs.min()} max={cs.max()}")

    if len(beep_play_times) < 2:
        print("\n  FAIL: Too few beeps to analyze intervals")
        return False

    beep_play_times = np.array(beep_play_times)
    intervals_ms = np.diff(beep_play_times) * 1000

    print(f"\nInter-beep play-time intervals (ms):")
    print(f"  Count: {len(intervals_ms)}")
    print(f"  Mean:  {intervals_ms.mean():.1f}")
    print(f"  Std:   {intervals_ms.std():.1f}")
    print(f"  Min:   {intervals_ms.min():.1f}")
    print(f"  Max:   {intervals_ms.max():.1f}")
    print(f"  Median:{np.median(intervals_ms):.1f}")

    # Histogram
    bins = [0, 50, 80, 100, 120, 150, 200, 500, 1000, float("inf")]
    counts, _ = np.histogram(intervals_ms, bins=bins)
    print(f"\nHistogram:")
    for i in range(len(bins) - 1):
        hi = f"{bins[i+1]:.0f}" if bins[i + 1] != float("inf") else "inf"
        pct = counts[i] / len(intervals_ms) * 100
        bar = "#" * int(pct / 2)
        print(f"  {bins[i]:>5.0f}-{hi:>5s}ms: {counts[i]:>4d} ({pct:5.1f}%) {bar}")

    # Checks
    in_wide = np.sum((intervals_ms >= 50) & (intervals_ms <= 200))
    in_wide_pct = in_wide / len(intervals_ms) * 100
    max_gap_s = intervals_ms.max() / 1000
    no_long_gaps = max_gap_s < 2.0
    no_doubles = intervals_ms.min() >= 30

    print(f"\nChecks:")
    print(f"  50-200ms range:  {in_wide}/{len(intervals_ms)} ({in_wide_pct:.1f}%)")
    print(f"  No gaps > 2s:    {'PASS' if no_long_gaps else 'FAIL'} (max={max_gap_s:.2f}s)")
    print(f"  No doubles <30ms:{'PASS' if no_doubles else 'FAIL'} (min={intervals_ms.min():.1f}ms)")

    passed = in_wide_pct >= 50 and no_long_gaps and no_doubles
    print(f"\n{'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    success = run_realtime_test()
    sys.exit(0 if success else 1)
