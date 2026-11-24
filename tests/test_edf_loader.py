"""
Unit tests for EDF loading, channel cleaning, and validation logic.
"""

import numpy as np
import pytest
from src.data.edf_loader import clean_channel_name, StandardizedEEG
from src.data.synthetic import generate_synthetic_eeg_record
from config import STANDARD_CHANNELS


def test_clean_channel_name():
    assert clean_channel_name("EEG FP1-F7") == "FP1-F7"
    assert clean_channel_name("fp1-f7.") == "FP1-F7"
    assert clean_channel_name("  c3-p3  ") == "C3-P3"
    assert clean_channel_name("EEG T8-P8-0") == "T8-P8"


def test_synthetic_record_structure():
    eeg, summary = generate_synthetic_eeg_record(
        duration_sec=30.0,
        seizure_intervals=[(10.0, 20.0)],
        random_seed=123,
    )

    assert isinstance(eeg, StandardizedEEG)
    assert eeg.is_synthetic is True
    assert eeg.sampling_rate == 256.0
    assert eeg.data.shape[0] == len(STANDARD_CHANNELS)
    assert eeg.data.shape[1] == int(30.0 * 256.0)
    assert summary.has_seizure is True
    assert len(summary.seizures) == 1
    assert summary.seizures[0].start_sec == 10.0
    assert summary.seizures[0].end_sec == 20.0
    assert not np.isnan(eeg.data).any()
    assert not np.isinf(eeg.data).any()
