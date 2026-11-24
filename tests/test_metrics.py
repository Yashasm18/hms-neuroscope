"""
Unit tests for evaluation metrics, event clustering, and edge cases.
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    calculate_window_metrics,
    cluster_positive_windows,
    calculate_event_metrics,
    evaluate_model_performance,
)


def test_calculate_window_metrics_standard():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.7, 0.4, 0.9, 0.85, 0.95])

    metrics, roc_data = calculate_window_metrics(y_true, y_pred, y_prob)

    assert metrics.total_samples == 8
    assert metrics.positives == 4
    assert metrics.negatives == 4
    assert metrics.tp == 3
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.tn == 3

    assert metrics.accuracy == 6 / 8
    assert metrics.sensitivity == 3 / 4
    assert metrics.specificity == 3 / 4
    assert metrics.precision == 3 / 4
    assert metrics.roc_auc is not None
    assert metrics.roc_auc > 0.5


def test_window_metrics_zero_positives_warning():
    # Only negatives present
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics, roc_data = calculate_window_metrics(y_true, y_pred)
    assert metrics.positives == 0
    assert len(metrics.warnings) > 0
    assert metrics.roc_auc is None  # Cannot compute ROC-AUC with single class


def test_cluster_positive_windows():
    preds = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
    starts = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    ends = starts + 5.0

    events = cluster_positive_windows(preds, starts, ends, max_gap_sec=6.0, min_duration_sec=5.0)

    assert len(events) == 2
    # First event: windows 1 and 2 -> [5.0, 15.0] (duration 10s)
    assert events[0] == (5.0, 15.0)
    # Second event: windows 5, 6, 7 -> [25.0, 40.0] (duration 15s)
    assert events[1] == (25.0, 40.0)


def test_event_metrics_and_far_per_hour():
    ref_seizures = [(10.0, 30.0)]  # 1 reference seizure
    pred_events = [(12.0, 25.0), (80.0, 95.0)]  # 1 matched, 1 false alarm
    total_hours = 0.5  # 30 minutes

    ev_metrics = calculate_event_metrics(ref_seizures, pred_events, total_monitoring_hours=total_hours)

    assert ev_metrics.total_true_events == 1
    assert ev_metrics.detected_true_events == 1
    assert ev_metrics.event_sensitivity == 1.0
    assert ev_metrics.false_alarm_events == 1
    # 1 false alarm in 0.5 hours = 2.0 FAR/h
    assert ev_metrics.false_alarms_per_hour == 2.0
