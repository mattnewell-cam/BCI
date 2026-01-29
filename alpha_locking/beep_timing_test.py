"""
Test that the sounddevice beep system from lock_live_2 works correctly.

Plays a series of beeps at different intervals to verify:
1. Audio stream opens and produces sound
2. schedule_beep() works with various delays
3. Rapid scheduling doesn't drop beeps
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the audio machinery from lock_live_2
from lock_live_2 import (
    schedule_beep, _audio_callback,
    AUDIO_FS, BEEP_MS, BEEP_FREQ_HZ,
)
import sounddevice as sd

print(f"Audio: {AUDIO_FS} Hz, beep: {BEEP_FREQ_HZ} Hz x {BEEP_MS}ms")
print(f"sounddevice version: {sd.__version__}")
print(f"Default output device: {sd.query_devices(kind='output')['name']}")
print()

audio_stream = sd.OutputStream(
    samplerate=AUDIO_FS, channels=1, callback=_audio_callback,
    dtype="float32", latency="low",
)
audio_stream.start()

# Test 1: immediate beep
print("Test 1: Single immediate beep")
schedule_beep(0.0)
time.sleep(0.5)

# Test 2: beep with 200ms delay
print("Test 2: Single beep with 200ms delay")
schedule_beep(0.2)
time.sleep(0.5)

# Test 3: rapid beeps at ~10 Hz (like alpha rhythm)
print("Test 3: 10 rapid beeps at ~100ms intervals (alpha rhythm rate)")
for i in range(10):
    schedule_beep(i * 0.1)
time.sleep(1.5)

# Test 4: two beeps close together (overlapping)
print("Test 4: Two overlapping beeps 10ms apart")
schedule_beep(0.0)
schedule_beep(0.01)
time.sleep(0.5)

audio_stream.stop()
audio_stream.close()

print()
print("All tests completed. If you heard beeps for each test, sounddevice is working.")
