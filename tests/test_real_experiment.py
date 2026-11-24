"""
Tests for Real CHB-MIT Scalp EEG Experiment pipeline and artifacts.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from config import RAW_DATA_DIR, ANNOTATIONS_DIR, BASE_DIR
from src.data.annotation_parser import (
    load_verified_recording_annotations,
    parse_wfdb_seizures,
    parse_summary_file,
)
from src.data.edf_loader import load_edf_recording


def test_verified_chb01_annotations():
    summary_path = ANNOTATIONS_DIR / "chb01-summary.txt"
    seizures_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures"

    assert summary_path.exists(), "chb01-summary.txt must exist"
    assert seizures_path.exists(), "chb01_03.edf.seizures must exist"

    # Test load_verified_recording_annotations
    rec_sum = load_verified_recording_annotations(
        edf_file_name_or_path="chb01_03.edf",
        summary_path=summary_path,
        seizures_path=seizures_path,
    )

    assert rec_sum is not None
    assert rec_sum.has_seizure is True
    assert len(rec_sum.seizures) == 1
    assert rec_sum.seizures[0].start_sec == 2996.0
    assert rec_sum.seizures[0].end_sec == 3036.0


def test_real_edf_loading():
    edf_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
    assert edf_path.exists(), "chb01_03.edf must exist"

    eeg = load_edf_recording(edf_path, max_duration_sec=60.0)
    assert eeg.sampling_rate == 256.0
    assert eeg.duration_sec == 60.0
    assert eeg.is_synthetic is False
    assert len(eeg.channel_names) == 18
    assert not np.isnan(eeg.data).any(), "Real EEG data must not contain NaNs"


def test_results_artifacts_integrity():
    results_dir = BASE_DIR / "results"
    assert results_dir.exists(), "results/ directory must exist"

    metrics_file = results_dir / "metrics.json"
    assert metrics_file.exists(), "results/metrics.json must exist"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["dataset"] == "CHB-MIT Scalp EEG Database (v1.0.0)"
    assert "partitions" in metrics
    assert metrics["partitions"]["train"]["recording"] == "chb01_03.edf"
    assert metrics["partitions"]["validation"]["recording"] == "chb01_04.edf"
    assert metrics["partitions"]["test"]["recording"] == "chb02_16.edf"
    assert metrics["partitions"]["test"]["patient"] == "chb02"
    assert "test_metrics" in metrics
    assert "validation_metrics" in metrics
    assert "train_metrics" in metrics

    tm = metrics["test_metrics"]
    assert 0.0 <= tm["accuracy"] <= 1.0
    assert 0.0 <= tm["sensitivity"] <= 1.0
    assert 0.0 <= tm["specificity"] <= 1.0
    assert 0.0 <= tm["f1_score"] <= 1.0

    # Verify other required artifact files
    assert (results_dir / "classification_report.txt").exists()
    assert (results_dir / "confusion_matrix.png").exists()
    assert (results_dir / "seizure_timeline.png").exists()
    assert (results_dir / "experiment_config.json").exists()
    assert (results_dir / "run_log.txt").exists()
    assert (results_dir / "trained_model.joblib").exists()
