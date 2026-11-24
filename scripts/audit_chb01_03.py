"""
Audit of single-recording chronological split for chb01_03.edf.
Mathematical verification of window timestamps, boundary buffer exclusions,
and partition class counts.
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, ANNOTATIONS_DIR, BASE_DIR
from src.data.edf_loader import load_edf_recording
from src.data.annotation_parser import load_verified_recording_annotations
from src.preprocessing.windowing import segment_eeg_windows


def audit_chb01_03_chronological_split():
    edf_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
    summary_path = ANNOTATIONS_DIR / "chb01-summary.txt"
    seizures_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures"

    # 1. Load data & annotations
    eeg = load_edf_recording(edf_path)
    summary = load_verified_recording_annotations(
        "chb01_03.edf",
        summary_path=summary_path,
        seizures_path=seizures_path,
    )

    # 2. Segment windows (5.0s window, 5.0s step, 5.0s buffer)
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

    total_windows = len(dataset.windows)
    pos_windows = [w for w in dataset.windows if w.label == 1]
    neg_windows = [w for w in dataset.windows if w.label == 0]
    amb_windows = [w for w in dataset.windows if w.label == -1]

    # Reconciliation check
    assert len(pos_windows) + len(neg_windows) + len(amb_windows) == total_windows, "Window count reconciliation failed"

    # 3. Chronological partitions:
    # 0% to 60%: [0.0s, 1830.0s] (windows with end_sec <= 1830.0)
    # 60% to 80%: [1830.0s, 2440.0s] (windows with start_sec >= 1830.0 and end_sec <= 2440.0)
    # 80% to 100%: [2440.0s, 3050.0s] (windows with start_sec >= 2440.0)
    train_windows = [w for w in dataset.windows if w.end_sec <= 1830.0]
    val_windows = [w for w in dataset.windows if w.start_sec >= 1830.0 and w.end_sec <= 2440.0]
    test_windows = [w for w in dataset.windows if w.start_sec >= 2440.0]

    assert len(train_windows) + len(val_windows) + len(test_windows) == total_windows, "Partition sum mismatch"

    # 4. Critical Mathematical Assertions
    train_pos = sum(1 for w in train_windows if w.label == 1)
    train_neg = sum(1 for w in train_windows if w.label == 0)
    train_amb = sum(1 for w in train_windows if w.label == -1)

    val_pos = sum(1 for w in val_windows if w.label == 1)
    val_neg = sum(1 for w in val_windows if w.label == 0)
    val_amb = sum(1 for w in val_windows if w.label == -1)

    test_pos = sum(1 for w in test_windows if w.label == 1)
    test_neg = sum(1 for w in test_windows if w.label == 0)
    test_amb = sum(1 for w in test_windows if w.label == -1)

    # CRITICAL AUDIT ASSERTIONS:
    # Under a strict chronological split of chb01_03.edf, ZERO seizure windows appear in Train or Val.
    assert train_pos == 0, f"Expected 0 seizure windows in Train, found {train_pos}"
    assert val_pos == 0, f"Expected 0 seizure windows in Val, found {val_pos}"
    assert test_pos == 8, f"Expected 8 seizure windows in Test, found {test_pos}"
    assert len(train_windows) == 366, f"Expected 366 train windows, got {len(train_windows)}"
    assert len(val_windows) == 122, f"Expected 122 val windows, got {len(val_windows)}"
    assert len(test_windows) == 122, f"Expected 122 test windows, got {len(test_windows)}"

    audit_report = {
        "recording": "chb01_03.edf",
        "duration_sec": float(eeg.duration_sec),
        "sampling_rate_hz": float(eeg.sampling_rate),
        "seizure_intervals_sec": [(s.start_sec, s.end_sec) for s in summary.seizures],
        "window_duration_sec": 5.0,
        "window_step_sec": 5.0,
        "boundary_buffer_sec": 5.0,
        "total_windows": total_windows,
        "class_counts_overall": {
            "seizure_windows": len(pos_windows),
            "non_seizure_windows": len(neg_windows),
            "ambiguous_boundary_excluded": len(amb_windows),
            "reconciliation_sum": len(pos_windows) + len(neg_windows) + len(amb_windows),
        },
        "chronological_partitions": {
            "train": {
                "span_sec": [0.0, 1830.0],
                "span_percentage": "60%",
                "total_windows": len(train_windows),
                "seizure_windows": train_pos,
                "non_seizure_windows": train_neg,
                "ambiguous_windows": train_amb,
            },
            "validation": {
                "span_sec": [1830.0, 2440.0],
                "span_percentage": "20%",
                "total_windows": len(val_windows),
                "seizure_windows": val_pos,
                "non_seizure_windows": val_neg,
                "ambiguous_windows": val_amb,
            },
            "test": {
                "span_sec": [2440.0, 3050.0],
                "span_percentage": "20%",
                "total_windows": len(test_windows),
                "seizure_windows": test_pos,
                "non_seizure_windows": test_neg,
                "ambiguous_windows": test_amb,
            },
        },
        "seizure_windows_detail": [
            {
                "window_idx": w.window_idx,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "label": w.label,
                "partition": "test",
            }
            for w in pos_windows
        ],
        "ambiguous_windows_detail": [
            {
                "window_idx": w.window_idx,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "label": w.label,
                "partition": "test",
                "reason": "buffer zone (within 5s of seizure onset 2996s or offset 3036s)",
            }
            for w in amb_windows
        ],
        "audit_finding": (
            "CRITICAL BUG CONFIRMED: In chb01_03.edf, the single verified seizure occurs from 2996s to 3036s. "
            "A chronological 60%/20%/20% split places the seizure ENTIRELY inside the test partition (2440s to 3050s). "
            "The training partition has 0 seizure windows (366 non-seizure), and the validation partition has 0 seizure windows "
            "(122 non-seizure). Therefore, a supervised binary model CANNOT be trained or threshold-calibrated on this split. "
            "The previous run artificially split the 8 positive windows 4/2/2, causing intra-event temporal leakage across partitions."
        ),
    }

    out_file = BASE_DIR / "results" / "audit_chb01_03.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    return audit_report


if __name__ == "__main__":
    rep = audit_chb01_03_chronological_split()
    print("Audit successful! Report written to results/audit_chb01_03.json")
    print(json.dumps(rep["chronological_partitions"], indent=2))
