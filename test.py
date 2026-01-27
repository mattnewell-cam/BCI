import numpy as np
from scipy.signal import butter, group_delay


fs = 256.0
b, a = butter(3, [8, 12], btype="bandpass", fs=fs, output="ba")
w, gd = group_delay((b, a), w=8192, fs=fs) # w in Hz, gd in samples


for f in [8, 10, 12]:
    gd_ms = np.interp(f, w, gd) / fs * 1000
    print(f"{f} Hz: {gd_ms:.1f} ms")