"""
Biomedical EEG digital filtering: zero-phase bandpass and notch filtering.
"""

from __future__ import annotations
import numpy as np
from scipy import signal
from typing import Optional, Union

from config import DEFAULT_SAMPLING_RATE, DEFAULT_LOW_PASS, DEFAULT_HIGH_PASS, DEFAULT_NOTCH_FREQ, FILTER_ORDER


def apply_bandpass_filter(
    data: np.ndarray,
    sfreq: float = DEFAULT_SAMPLING_RATE,
    lowcut: float = DEFAULT_HIGH_PASS,
    highcut: float = DEFAULT_LOW_PASS,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter along the last axis of data.
    data: (n_channels, n_samples) or (n_samples,)
    """
    nyquist = 0.5 * sfreq
    if nyquist <= 0:
        raise ValueError(f"Invalid sampling rate: {sfreq}")

    low = max(0.05, float(lowcut)) / nyquist
    high = min(float(highcut), nyquist - 0.5) / nyquist

    if low >= high:
        raise ValueError(f"Invalid filter band: lowcut {lowcut} Hz >= highcut {highcut} Hz for sfreq {sfreq} Hz")

    b, a = signal.butter(order, [low, high], btype="bandpass")

    # Ensure signal length is greater than filtfilt padlen
    padlen = 3 * (max(len(a), len(b)) + 1)
    if data.shape[-1] <= padlen:
        # Signal too short for filtfilt; return original
        return data.copy()

    # Apply along last axis
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered.astype(np.float32)


def apply_notch_filter(
    data: np.ndarray,
    sfreq: float = DEFAULT_SAMPLING_RATE,
    notch_freq: float = DEFAULT_NOTCH_FREQ,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Apply zero-phase IIR notch filter to suppress powerline interference (e.g. 60 Hz or 50 Hz).
    """
    nyquist = 0.5 * sfreq
    if notch_freq >= nyquist or notch_freq <= 0:
        # Notch frequency outside reachable band; return unaltered
        return data.copy()

    w0 = notch_freq / nyquist
    b, a = signal.iirnotch(w0, quality_factor)

    padlen = 3 * (max(len(a), len(b)) + 1)
    if data.shape[-1] <= padlen:
        return data.copy()

    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered.astype(np.float32)


def preprocess_eeg_signal(
    data: np.ndarray,
    sfreq: float = DEFAULT_SAMPLING_RATE,
    lowcut: float = DEFAULT_HIGH_PASS,
    highcut: float = DEFAULT_LOW_PASS,
    notch_freq: Optional[float] = DEFAULT_NOTCH_FREQ,
    demean: bool = True,
) -> np.ndarray:
    """
    Standard EEG preprocessing pipeline:
    1. Demean / remove DC offset
    2. Zero-phase bandpass filter (0.5 - 40 Hz)
    3. Zero-phase notch filter (60 Hz)
    """
    out = data.astype(np.float32)
    if demean:
        # Subtract mean along sample axis
        out = out - np.mean(out, axis=-1, keepdims=True)

    out = apply_bandpass_filter(out, sfreq=sfreq, lowcut=lowcut, highcut=highcut)

    if notch_freq is not None and notch_freq > 0:
        out = apply_notch_filter(out, sfreq=sfreq, notch_freq=notch_freq)

    return out
