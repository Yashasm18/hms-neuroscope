"""
Unit tests for time-domain and frequency-domain EEG feature extraction.
"""

import numpy as np
import pytest
from src.features.time_frequency import (
    compute_time_domain_features,
    compute_frequency_domain_features,
    extract_single_channel_features,
)
from src.features.extractor import FeatureExtractor
from src.data.synthetic import generate_synthetic_eeg_record
from src.preprocessing.windowing import segment_eeg_windows


def test_time_domain_features():
    sfreq = 256.0
    t = np.linspace(0, 5, int(5 * sfreq), endpoint=False)
    sig = 20.0 * np.sin(2 * np.pi * 5.0 * t)  # 5 Hz sine wave

    feats = compute_time_domain_features(sig)

    assert "mean" in feats
    assert "std" in feats
    assert "variance" in feats
    assert "line_length" in feats
    assert "hjorth_activity" in feats
    assert "hjorth_mobility" in feats
    assert "hjorth_complexity" in feats

    assert abs(feats["mean"]) < 1.0  # Zero mean sine wave
    assert feats["variance"] > 0
    assert feats["ptp"] > 35.0  # Peak to peak for amplitude 20 is ~40
    assert feats["line_length"] > 0
    assert not any(np.isnan(v) or np.isinf(v) for v in feats.values())


def test_frequency_domain_features():
    sfreq = 256.0
    t = np.linspace(0, 5, int(5 * sfreq), endpoint=False)
    # 10 Hz pure sine wave (Alpha band: 8 - 13 Hz)
    sig = 25.0 * np.sin(2 * np.pi * 10.0 * t)

    feats = compute_frequency_domain_features(sig, sfreq=sfreq)

    assert "abs_power_alpha" in feats
    assert "rel_power_alpha" in feats
    assert "spectral_entropy" in feats
    assert "sef95" in feats
    assert "peak_frequency" in feats

    # For 10 Hz tone, peak frequency should be ~10 Hz and alpha should dominate
    assert abs(feats["peak_frequency"] - 10.0) <= 1.0
    assert feats["rel_power_alpha"] > 0.70  # Dominates spectral power
    assert not any(np.isnan(v) or np.isinf(v) for v in feats.values())


def test_feature_extractor_on_dataset():
    eeg, summary = generate_synthetic_eeg_record(
        duration_sec=20.0,
        seizure_intervals=[(5.0, 15.0)],
        random_seed=42,
    )
    dataset = segment_eeg_windows(
        data=eeg.data,
        sfreq=eeg.sampling_rate,
        window_duration_sec=5.0,
        window_step_sec=5.0,
        annotation_summary=summary,
        patient_id=eeg.patient_id,
        file_name=eeg.file_name,
    )

    extractor = FeatureExtractor(sfreq=eeg.sampling_rate, target_channels=eeg.channel_names)
    df = extractor.transform_dataset(dataset, channel_names=eeg.channel_names)

    assert len(df) == len(dataset)
    assert "target_label" in df.columns
    assert "patient_id" in df.columns
    assert "global_mean__abs_power_delta" in df.columns
    assert not df.isna().any().any()
