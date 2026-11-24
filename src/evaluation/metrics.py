"""
Evaluation metric computations: window-level, event-level, and false alarm rate (FAR/h).
Never invents metrics. Clearly differentiates window-level vs event-level analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class WindowMetrics:
    """Standard binary classification metrics computed at the window level."""
    accuracy: float
    balanced_accuracy: float
    precision: float
    sensitivity: float  # Recall (True Positive Rate)
    specificity: float  # True Negative Rate
    f1_score: float
    roc_auc: Optional[float]
    tp: int
    fp: int
    tn: int
    fn: int
    total_samples: int
    positives: int
    negatives: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class EventMetrics:
    """Clinical event-level seizure detection metrics."""
    total_true_events: int
    detected_true_events: int
    event_sensitivity: float
    false_alarm_events: int
    total_monitoring_hours: float
    false_alarms_per_hour: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation bundle across window and event levels."""
    model_name: str
    decision_threshold: float
    window_metrics: WindowMetrics
    event_metrics: Optional[EventMetrics]
    roc_fpr: Optional[np.ndarray] = None
    roc_tpr: Optional[np.ndarray] = None
    roc_thresholds: Optional[np.ndarray] = None
    num_patients: int = 0
    patient_ids: List[str] = field(default_factory=list)
    num_recordings: int = 0
    recording_files: List[str] = field(default_factory=list)
    split_name: str = "Test"
    warnings: List[str] = field(default_factory=list)


def calculate_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Tuple[WindowMetrics, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Calculate window-level accuracy, precision, sensitivity, specificity, F1, and ROC-AUC.
    Returns (WindowMetrics, (fpr, tpr, roc_thresholds)).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    warnings: List[str] = []

    if n_total == 0:
        return (
            WindowMetrics(
                accuracy=0.0,
                balanced_accuracy=0.0,
                precision=0.0,
                sensitivity=0.0,
                specificity=0.0,
                f1_score=0.0,
                roc_auc=None,
                tp=0,
                fp=0,
                tn=0,
                fn=0,
                total_samples=0,
                positives=0,
                negatives=0,
                warnings=["Evaluation dataset is empty."],
            ),
            None,
        )

    if n_pos == 0:
        warnings.append("Test set contains 0 positive (seizure) windows. Recall and ROC-AUC cannot be calculated.")
    if n_neg == 0:
        warnings.append("Test set contains 0 negative (non-seizure) windows. Specificity cannot be calculated.")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float((sens + spec) / 2.0)

    # ROC-AUC calculation
    auc_val = None
    roc_curve_data = None
    if y_prob is not None and n_pos > 0 and n_neg > 0:
        try:
            auc_val = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_curve_data = (fpr, tpr, thresholds)
        except Exception as e:
            warnings.append(f"Could not compute ROC-AUC: {e}")

    metrics = WindowMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        sensitivity=sens,
        specificity=spec,
        f1_score=f1,
        roc_auc=auc_val,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        total_samples=n_total,
        positives=n_pos,
        negatives=n_neg,
        warnings=warnings,
    )
    return metrics, roc_curve_data


def cluster_positive_windows(
    window_preds: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    max_gap_sec: float = 10.0,
    min_duration_sec: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Cluster contiguous or near-contiguous positive window predictions into discrete seizure events.
    """
    events: List[Tuple[float, float]] = []
    in_event = False
    event_start = 0.0
    event_end = 0.0

    for pred, t_start, t_end in zip(window_preds, start_times, end_times):
        if pred == 1:
            if not in_event:
                in_event = True
                event_start = float(t_start)
                event_end = float(t_end)
            else:
                # Extend event
                if t_start - event_end <= max_gap_sec:
                    event_end = float(t_end)
                else:
                    # Save previous event and start new
                    if (event_end - event_start) >= min_duration_sec:
                        events.append((event_start, event_end))
                    event_start = float(t_start)
                    event_end = float(t_end)
        else:
            if in_event and (t_start - event_end) > max_gap_sec:
                in_event = False
                if (event_end - event_start) >= min_duration_sec:
                    events.append((event_start, event_end))

    if in_event and (event_end - event_start) >= min_duration_sec:
        events.append((event_start, event_end))

    return events


def calculate_event_metrics(
    reference_seizure_intervals: List[Tuple[float, float]],
    predicted_events: List[Tuple[float, float]],
    total_monitoring_hours: float,
) -> EventMetrics:
    """
    Compute event-level sensitivity and False Alarm Rate per hour (FAR/h).
    """
    warnings: List[str] = []
    total_ref = len(reference_seizure_intervals)

    if total_monitoring_hours <= 0:
        total_monitoring_hours = 1.0  # Avoid zero division
        warnings.append("Total monitoring hours was <= 0; defaulted to 1.0 hr.")

    detected_ref_count = 0
    matched_pred_indices = set()

    for ref_start, ref_end in reference_seizure_intervals:
        # Check if any predicted event overlaps this reference seizure
        matched = False
        for pred_idx, (p_start, p_end) in enumerate(predicted_events):
            if max(ref_start, p_start) < min(ref_end, p_end):
                matched = True
                matched_pred_indices.add(pred_idx)
        if matched:
            detected_ref_count += 1

    event_sens = (detected_ref_count / total_ref) if total_ref > 0 else 0.0
    false_alarm_count = len(predicted_events) - len(matched_pred_indices)
    far_per_hour = false_alarm_count / total_monitoring_hours

    if total_ref == 0:
        warnings.append("No reference seizures present in this interval; event sensitivity not applicable.")

    return EventMetrics(
        total_true_events=total_ref,
        detected_true_events=detected_ref_count,
        event_sensitivity=event_sens,
        false_alarm_events=false_alarm_count,
        total_monitoring_hours=total_monitoring_hours,
        false_alarms_per_hour=far_per_hour,
        warnings=warnings,
    )


def evaluate_model_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float = 0.5,
    patient_ids: Optional[List[str]] = None,
    recording_files: Optional[List[str]] = None,
    reference_events: Optional[List[Tuple[float, float]]] = None,
    window_start_times: Optional[np.ndarray] = None,
    window_end_times: Optional[np.ndarray] = None,
    total_hours: Optional[float] = None,
    model_name: str = "Random Forest",
    split_name: str = "Test",
) -> EvaluationReport:
    """Generate complete window-level and event-level evaluation report."""
    y_pred = (y_prob >= decision_threshold).astype(int)
    win_metrics, roc_data = calculate_window_metrics(y_true, y_pred, y_prob)

    event_metrics = None
    if reference_events is not None and window_start_times is not None and window_end_times is not None:
        pred_events = cluster_positive_windows(y_pred, window_start_times, window_end_times)
        if total_hours is None or total_hours <= 0:
            total_duration_sec = (window_end_times[-1] - window_start_times[0]) if len(window_end_times) > 0 else 0.0
            total_hours = total_duration_sec / 3600.0
        event_metrics = calculate_event_metrics(reference_events, pred_events, total_hours)

    pats = list(set(patient_ids)) if patient_ids else []
    files = list(set(recording_files)) if recording_files else []

    fpr = roc_data[0] if roc_data else None
    tpr = roc_data[1] if roc_data else None
    thresh = roc_data[2] if roc_data else None

    return EvaluationReport(
        model_name=model_name,
        decision_threshold=decision_threshold,
        window_metrics=win_metrics,
        event_metrics=event_metrics,
        roc_fpr=fpr,
        roc_tpr=tpr,
        roc_thresholds=thresh,
        num_patients=len(pats),
        patient_ids=pats,
        num_recordings=len(files),
        recording_files=files,
        split_name=split_name,
        warnings=win_metrics.warnings,
    )
