# Muse EEG Processing

Real-time EEG signal processing and analysis for the Muse headband, with focus on alpha wave detection and neurofeedback.

## Tech Stack

- **pylsl** - Lab Streaming Layer protocol for real-time EEG streaming
- **muselsl** - Muse device discovery and streaming
- **scipy** - Signal processing (Butterworth filters, Welch PSD, Hilbert transform)
- **numpy** - Numerical computations
- **matplotlib** - Real-time visualization

## Key Files

| File | Purpose |
|------|---------|
| `alpha_lock.py` | PLL-based alpha detection with phase-locked audio beeps |
| `alpha_live.py` | Real-time alpha power calculation using Welch's method |
| `show_alpha.py` | Alpha visualization with envelope and bandpower metrics |
| `band_buckets.py` | Multi-band frequency analyzer (Delta through Gamma) |
| `list_streams.py` | Utility to discover LSL streams |
| `utils.py` | Shared functions: stream discovery, filters, bandpower |

## Architecture

```
Muse Device → LSL Stream → Notch Filter (50Hz) → Bandpass → Analysis → Visualization
```

All scripts follow a similar pattern:
1. Find EEG stream via `find_eeg_stream()` from utils
2. Pull samples from StreamInlet
3. Apply streaming IIR filters with state preservation
4. Analyze (Welch PSD, PLL, Hilbert envelope)
5. Update real-time matplotlib plots

## EEG Frequency Bands

- Delta: 1-4 Hz
- Theta: 4-8 Hz
- Alpha: 8-12 Hz (primary focus)
- Beta: 12-30 Hz
- Gamma: 30-45+ Hz

## Conventions

- Muse channels: AF7, AF8, TP9, TP10 (4 EEG + reference)
- Sample rate: typically 256 Hz
- Use SOS (second-order sections) filters for numerical stability
- Preserve filter state (`zi`) between sample chunks for streaming
