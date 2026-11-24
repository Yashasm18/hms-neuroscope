"""
Unit tests for filtering, montage verification, and boundary-aware window segmentation.
"""

import numpy as np
import pytest
from src.preprocessing.filter import apply_bandpass_filter, apply_notch_filter, preprocess_eeg_signal
from src.preprocessing.montage import verify_channel_montage, standardize_montage
from src.preprocessing.windowing import segment_eeg_windows
from src.data.annotation_parser import RecordingSummary, SeizureAnnotation
from config import STANDARD_CHANNELS


def test_bandpass_filter():
    sfreq = 256.0
    t = np.linspace(0, 5, int(5 * sfreq), endpoint=False)
    # 0.1 Hz slow drift + 10 Hz alpha wave + 60 Hz line noise + 100 Hz high freq noise
    sig = (
        50.0 * np.sin(2 * np.pi * 0.1 * t)
        + 20.0 * np.sin(2 * np.pi * 10.0 * t)
        + 30.0 * np.sin(2 * np.pi * 60.0 * t)
        + 15.0 * np.sin(2 * np.pi * 100.0 * t)
    )

    filtered = apply_bandpass_filter(sig, sfreq=sfreq, lowcut=0.5, highcut=40.0)

    assert filtered.shape == sig.shape
    # Slow baseline and high-frequency components should be attenuated
    assert np.std(filtered) < np.std(sig)
    assert not np.isnan(filtered).any()


def test_notch_filter():
    sfreq = 256.0
    t = np.linspace(0, 5, int(5 * sfreq), endpoint=False)
    line_noise = 50.0 * np.sin(2 * np.pi * 60.0 * t)

    filtered = apply_notch_filter(line_noise, sfreq=sfreq, notch_freq=60.0)
    # 60 Hz should be heavily suppressed
    assert np.std(filtered) < 0.15 * np.std(line_noise)


def test_montage_verification_and_standardization():
    input_channels = ["FP1-F7", "F7-T7", "EXTRA_CH"]
    matched, missing, extra = verify_channel_montage(input_channels, STANDARD_CHANNELS)

    assert "FP1-F7" in matched
    assert "F7-T7" in matched
    assert "EXTRA_CH" in extra
    assert "T7-P7" in missing

    # Standardize data matrix
    dummy_data = np.ones((len(input_channels), 100), dtype=np.float32)
    aligned_data, final_channels, warnings = standardize_montage(
        dummy_data, input_channels, STANDARD_CHANNELS
    )

    assert aligned_data.shape == (len(STANDARD_CHANNELS), 100)
    assert final_channels == STANDARD_CHANNELS
    assert len(warnings) > 0  # Missing channels were reported


def test_window_segmentation_and_boundary_labeling():
    sfreq = 256.0
    duration_sec = 60.0
    n_samples = int(duration_sec * sfreq)
    data = np.zeros((2, n_samples), dtype=np.float32)

    # Place a seizure from t=20.0s to t=35.0s (15s duration)
    summary = RecordingSummary(
        file_name="test_rec.edf",
        patient_id="chb_test",
        num_seizures=1,
        seizures=[SeizureAnnotation(index=1, start_sec=20.0, end_sec=35.0)],
    )

    # 5-second non-overlapping windows:
    # Win 0: [0 - 5] -> 0
    # Win 1: [5 - 10] -> 0
    # Win 2: [10 - 15] -> 0
    # Win 3: [15 - 20] -> -1 (Within pre-ictal 5s boundary buffer: [15, 20])
    # Win 4: [20 - 25] -> 1 (Seizure: 100% overlap)
    # Win 5: [25 - 30] -> 1 (Seizure: 100% overlap)
    # Win 6: [30 - 35] -> 1 (Seizure: 100% overlap)
    # Win 7: [35 - 40] -> -1 (Within post-ictal 5s boundary buffer: [35, 40])
    # Win 8: [40 - 45] -> 0 (Normal)

    dataset = segment_eeg_windows(
        data=data,
        sfreq=sfreq,
        window_duration_sec=5.0,
        window_step_sec=5.0,
        annotation_summary=summary,
        overlap_ratio_threshold=0.5,
        boundary_buffer_sec=5.0,
        patient_id="chb_test",
        file_name="test_rec.edf",
    )

    assert len(dataset) == 12  # 60s / 5s = 12 windows

    labels = dataset.labels
    # Check normal windows
    assert labels[0] == 0
    assert labels[1] == 0
    # Check boundary exclusion window before seizure
    assert labels[3] == -1
    # Check seizure windows
    assert labels[4] == 1
    assert labels[5] == 1
    assert labels[6] == 1
    # Check boundary exclusion window after seizure
    assert labels[7] == -1
    # Check post-boundary normal window
    assert labels[8] == 0

    # Test clean subset (ambiguous windows removed)
    clean = dataset.get_clean_training_subset()
    assert -1 not in clean.labels
    assert len(clean) == 10  # 12 - 2 boundary windows
