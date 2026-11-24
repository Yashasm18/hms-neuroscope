"""
Synthetic EEG generator strictly for software testing, offline development,
and interactive UI demonstrations when real data downloads are unavailable.

IMPORTANT: All synthetic signals are explicitly labeled. They must NEVER be
used as evidence of real clinical or algorithmic seizure-detection performance.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

from config import DEFAULT_SAMPLING_RATE, STANDARD_CHANNELS
from .edf_loader import StandardizedEEG
from .annotation_parser import RecordingSummary, SeizureAnnotation


def generate_synthetic_eeg_record(
    patient_id: str = "chb_synthetic",
    file_name: str = "synthetic_demo_01.edf",
    duration_sec: float = 120.0,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    channel_names: Optional[List[str]] = None,
    seizure_intervals: Optional[List[Tuple[float, float]]] = None,
    random_seed: int = 42,
) -> Tuple[StandardizedEEG, RecordingSummary]:
    """
    Generate a multichannel synthetic EEG record with realistic spectral background
    and injected epileptiform spike-and-wave discharges during specified seizure intervals.

    Returns:
        (StandardizedEEG, RecordingSummary)
    """
    rng = np.random.default_rng(random_seed)
    channels = channel_names or STANDARD_CHANNELS
    n_channels = len(channels)
    n_samples = int(duration_sec * sampling_rate)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)

    # If no intervals specified, insert one 25-second seizure in the middle
    if seizure_intervals is None:
        sz_start = max(10.0, duration_sec * 0.4)
        sz_end = min(duration_sec - 10.0, sz_start + 25.0)
        seizure_intervals = [(sz_start, sz_end)]

    data = np.zeros((n_channels, n_samples), dtype=np.float32)

    # 1. Background physiological EEG per channel
    for ch_idx, ch_name in enumerate(channels):
        # 1/f pink noise approximation via cumulative sum of white noise with decay
        white = rng.normal(0, 15.0, n_samples)
        # AR(1) smoothing for biological baseline
        pink = np.zeros(n_samples)
        decay = 0.96
        for i in range(1, n_samples):
            pink[i] = decay * pink[i - 1] + white[i]
        
        # Physiological Alpha rhythm (8-12 Hz) - stronger in posterior/occipital channels
        is_occipital = any(reg in ch_name for reg in ["O1", "O2", "P7", "P8", "P3", "P4"])
        alpha_amp = 18.0 if is_occipital else 7.0
        alpha_freq = rng.uniform(9.5, 11.0)
        alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2 * np.pi))

        # Theta rhythm (4-7 Hz)
        theta_amp = rng.uniform(6.0, 12.0)
        theta_freq = rng.uniform(4.5, 6.5)
        theta = theta_amp * np.sin(2 * np.pi * theta_freq * t + rng.uniform(0, 2 * np.pi))

        # Beta rhythm (13-22 Hz)
        beta_amp = rng.uniform(3.0, 6.0)
        beta_freq = rng.uniform(15.0, 20.0)
        beta = beta_amp * np.sin(2 * np.pi * beta_freq * t + rng.uniform(0, 2 * np.pi))

        # Frontal eye blink artifact (delta transients in FP1, FP2)
        blinks = np.zeros(n_samples)
        if "FP" in ch_name or "F7" in ch_name or "F8" in ch_name:
            blink_times = np.arange(15.0, duration_sec, 20.0)
            for bt in blink_times:
                b_idx = int(bt * sampling_rate)
                b_width = int(0.3 * sampling_rate)
                if 0 <= b_idx < n_samples - b_width:
                    blinks[b_idx : b_idx + b_width] += 70.0 * np.hanning(b_width)

        data[ch_idx] = pink + alpha + theta + beta + blinks

    # 2. Inject Ictal Seizure Activity (3 Hz Spike-and-Wave + paroxysmal fast activity)
    for sz_start, sz_end in seizure_intervals:
        mask = (t >= sz_start) & (t <= sz_end)
        n_sz_samples = np.sum(mask)
        if n_sz_samples > 0:
            sz_t = t[mask] - sz_start
            
            # Generalized 3 Hz spike-and-slow-wave discharge:
            # Slow wave (3 Hz sine) + sharp spike (harmonic 9-15 Hz pulse)
            spike_wave = 65.0 * np.sin(2 * np.pi * 3.0 * sz_t)
            spike_wave += 45.0 * np.sin(2 * np.pi * 9.0 * sz_t + np.pi / 4)
            spike_wave += 30.0 * np.sin(2 * np.pi * 15.0 * sz_t + np.pi / 2)
            
            # Paroxysmal high frequency activity (20-30 Hz fast onset)
            fast_activity = 25.0 * np.sin(2 * np.pi * 22.0 * sz_t) * np.exp(-sz_t / 8.0)

            # High amplitude burst envelope
            envelope = np.minimum(1.0, sz_t / 2.0) * np.minimum(1.0, (sz_end - t[mask]) / 2.0)
            ictal_signal = (spike_wave + fast_activity) * envelope

            for ch_idx in range(n_channels):
                # Channel-specific amplitude modulation
                ch_factor = rng.uniform(0.7, 1.3)
                data[ch_idx, mask] += (ictal_signal * ch_factor).astype(np.float32)

    annotations = [
        SeizureAnnotation(index=i + 1, start_sec=float(s), end_sec=float(e))
        for i, (s, e) in enumerate(seizure_intervals)
    ]

    summary = RecordingSummary(
        file_name=file_name,
        patient_id=patient_id,
        start_time_str="00:00:00",
        end_time_str=f"{int(duration_sec // 3600):02d}:{int((duration_sec % 3600) // 60):02d}:{int(duration_sec % 60):02d}",
        num_seizures=len(annotations),
        seizures=annotations,
    )

    std_eeg = StandardizedEEG(
        file_name=file_name,
        patient_id=patient_id,
        sampling_rate=sampling_rate,
        duration_sec=duration_sec,
        total_samples=n_samples,
        channel_names=channels,
        data=data,
        is_synthetic=True,
        warning_messages=["SYNTHETIC EEG DATA: Generated strictly for software testing & UI demo."],
    )

    return std_eeg, summary
