"""Evaluation metrics, confusion matrices, ROC-AUC, and clinical false alarm rates."""
from .metrics import (
    calculate_window_metrics,
    calculate_event_metrics,
    evaluate_model_performance,
    EvaluationReport,
)

__all__ = [
    "calculate_window_metrics",
    "calculate_event_metrics",
    "evaluate_model_performance",
    "EvaluationReport",
]
