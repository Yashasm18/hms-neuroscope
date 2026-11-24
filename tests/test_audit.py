"""
Unit tests for the audit of CHB-MIT experiment and chronological partitioning.
Mathematically verifies window timestamps, boundary exclusions, and class counts.
"""

import pytest
import numpy as np
from config import RAW_DATA_DIR, ANNOTATIONS_DIR
from src.data.edf_loader import load_edf_recording
from src.data.annotation_parser import load_verified_recording_annotations
from src.preprocessing.windowing import segment_eeg_windows


def test_audit_chb01_03_timestamps_and_window_counts():
    edf_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
    summary_path = ANNOTATIONS_DIR / "chb01-summary.txt"
    seizures_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures"

    eeg = load_edf_recording(edf_path)
    summary = load_verified_recording_annotations(
        "chb01_03.edf",
        summary_path=summary_path,
        seizures_path=seizures_path,
    )

    dataset = segment_eeg_windows(
        data=eeg.data,
        sfreq=eeg.sampling_rate,
        window_duration_sec=5.0,
        window_step_sec=5.0,
        annotation_summary=summary,
        boundary_buffer_sec=5.0,
        patient_id="chb01",
        file_name="chb01_03.edf",
    )

    assert len(dataset.windows) == 610, "Total windows must be 610"

    pos_windows = [w for w in dataset.windows if w.label == 1]
    neg_windows = [w for w in dataset.windows if w.label == 0]
    amb_windows = [w for w in dataset.windows if w.label == -1]

    # Reconcile counts
    assert len(pos_windows) == 8, f"Expected 8 seizure windows, got {len(pos_windows)}"
    assert len(neg_windows) == 599, f"Expected 599 non-seizure windows, got {len(neg_windows)}"
    assert len(amb_windows) == 3, f"Expected 3 ambiguous boundary windows, got {len(amb_windows)}"
    assert len(pos_windows) + len(neg_windows) + len(amb_windows) == 610, "Window sum reconciliation failed"

    # Verify exact seizure indices and timestamps
    pos_indices = [w.window_idx for w in pos_windows]
    assert pos_indices == list(range(599, 607)), "Seizure windows must be indices 599 to 606"
    assert pos_windows[0].start_sec == 2995.0
    assert pos_windows[-1].end_sec == 3035.0

    # Verify ambiguous buffer windows
    amb_indices = [w.window_idx for w in amb_windows]
    assert amb_indices == [598, 607, 608], "Ambiguous buffer windows must be indices 598, 607, 608"


def test_chronological_split_zero_seizures_in_train_and_val():
    edf_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
    summary_path = ANNOTATIONS_DIR / "chb01-summary.txt"
    seizures_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures"

    eeg = load_edf_recording(edf_path)
    summary = load_verified_recording_annotations(
        "chb01_03.edf",
        summary_path=summary_path,
        seizures_path=seizures_path,
    )

    dataset = segment_eeg_windows(
        data=eeg.data,
        sfreq=eeg.sampling_rate,
        window_duration_sec=5.0,
        window_step_sec=5.0,
        annotation_summary=summary,
        boundary_buffer_sec=5.0,
    )

    # Chronological partition of the recording:
    # 60% Train: 0s to 1830s (windows with end_sec <= 1830.0)
    # 20% Val:   1830s to 2440s (windows with start_sec >= 1830.0 and end_sec <= 2440.0)
    # 20% Test:  2440s to 3050s (windows with start_sec >= 2440.0)
    train_windows = [w for w in dataset.windows if w.end_sec <= 1830.0]
    val_windows = [w for w in dataset.windows if w.start_sec >= 1830.0 and w.end_sec <= 2440.0]
    test_windows = [w for w in dataset.windows if w.start_sec >= 2440.0]

    assert len(train_windows) == 366
    assert len(val_windows) == 122
    assert len(test_windows) == 122
    assert len(train_windows) + len(val_windows) + len(test_windows) == 610

    # CRITICAL AUDIT PROOF:
    # No seizure windows appear in the first 60% or middle 20% chronological partitions!
    train_seizures = sum(1 for w in train_windows if w.label == 1)
    val_seizures = sum(1 for w in val_windows if w.label == 1)
    test_seizures = sum(1 for w in test_windows if w.label == 1)

    assert train_seizures == 0, "Train partition must contain 0 seizure windows"
    assert val_seizures == 0, "Validation partition must contain 0 seizure windows"
    assert test_seizures == 8, "Test partition must contain all 8 seizure windows"


def test_multi_recording_train_and_val_have_positives():
    edf_path1 = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
    edf_path2 = RAW_DATA_DIR / "chb01" / "chb01_04.edf"

    assert edf_path1.exists(), "chb01_03.edf must exist"
    assert edf_path2.exists(), "chb01_04.edf must exist"

    eeg1 = load_edf_recording(edf_path1)
    sum1 = load_verified_recording_annotations("chb01_03.edf", summary_path=ANNOTATIONS_DIR / "chb01-summary.txt", seizures_path=RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures")
    ds1 = segment_eeg_windows(eeg1.data, eeg1.sampling_rate, 5.0, 5.0, sum1, boundary_buffer_sec=5.0)

    eeg2 = load_edf_recording(edf_path2)
    sum2 = load_verified_recording_annotations("chb01_04.edf", summary_path=ANNOTATIONS_DIR / "chb01-summary.txt", seizures_path=RAW_DATA_DIR / "chb01" / "chb01_04.edf.seizures")
    ds2 = segment_eeg_windows(eeg2.data, eeg2.sampling_rate, 5.0, 5.0, sum2, boundary_buffer_sec=5.0)

    # In a multi-recording setup:
    # Train: chb01_03.edf has 8 seizure windows, 599 non-seizure
    # Val:   chb01_04.edf has 6 seizure windows, 294 non-seizure
    assert sum(w.label == 1 for w in ds1.windows) == 8
    assert sum(w.label == 0 for w in ds1.windows) == 599
    assert sum(w.label == 1 for w in ds2.windows) == 6
    assert sum(w.label == 0 for w in ds2.windows) == 294

    # Both partitions contain genuine positives, enabling real supervised training and valid threshold tuning!
