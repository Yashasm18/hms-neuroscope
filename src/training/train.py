"""
Model training orchestrator, validation threshold calibration, and artifact persistence.
"""

from __future__ import annotations
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR, MEDICAL_DISCLAIMER
from src.evaluation.metrics import evaluate_model_performance, EvaluationReport
from .split import create_patient_train_test_split, SplitResult
from .pipeline import build_classification_pipeline


@dataclass
class TrainedModelBundle:
    """Complete bundle containing trained pipeline, threshold, features, and evaluation metadata."""
    model_name: str
    pipeline: Any
    feature_names: List[str]
    optimal_threshold: float
    train_evaluation: EvaluationReport
    val_evaluation: EvaluationReport
    test_evaluation: Optional[EvaluationReport]
    dummy_baseline_evaluation: Optional[EvaluationReport]
    train_patients: List[str]
    val_patients: List[str]
    test_patients: List[str]
    trained_at_utc: str
    config_dict: Dict[str, Any] = field(default_factory=dict)
    medical_disclaimer: str = MEDICAL_DISCLAIMER


def safe_predict_proba(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """Safely extract positive class (1) probabilities even if only one class was seen."""
    probs = pipeline.predict_proba(X)
    clf = pipeline.named_steps.get("classifier", pipeline)
    classes = getattr(clf, "classes_", np.array([0, 1]))
    if probs.shape[1] > 1:
        class_1_indices = np.where(classes == 1)[0]
        if len(class_1_indices) > 0:
            return probs[:, class_1_indices[0]]
        return probs[:, 1]
    else:
        only_class = classes[0]
        return np.full(len(X), 1.0 if only_class == 1 else 0.0, dtype=np.float32)


def tune_decision_threshold(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    candidate_thresholds: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Search candidate classification thresholds on validation data to maximize F1-score.
    Returns: (optimal_threshold, best_f1, threshold_scores_dict)
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if len(y_val) == 0 or np.sum(y_val == 1) == 0:
        return 0.5, 0.0, {t: 0.0 for t in candidate_thresholds}

    y_probs = safe_predict_proba(pipeline, X_val)
    best_thresh = 0.5
    best_score = -1.0
    scores = {}

    for t in candidate_thresholds:
        preds = (y_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        scores[t] = f1
        if f1 > best_score:
            best_score = f1
            best_thresh = t

    return best_thresh, best_score, scores


def train_and_evaluate_models(
    features_df: pd.DataFrame,
    model_type: str = "random_forest",
    random_state: int = 42,
    test_ratio: float = 0.3,
    val_ratio: float = 0.2,
    save_bundle: bool = True,
    bundle_name: Optional[str] = None,
) -> TrainedModelBundle:
    """
    Train and evaluate model using patient-isolated cross-validation.
    1. Splits data by patient identity.
    2. Fits Dummy baseline classifier.
    3. Fits target classifier pipeline (scaler fit only on Train).
    4. Calibrates threshold on Validation set.
    5. Evaluates on Test set with optimal threshold.
    6. Persists model bundle to disk.
    """
    # 1. Patient-level split
    split: SplitResult = create_patient_train_test_split(
        features_df,
        test_patient_ratio=test_ratio,
        val_patient_ratio=val_ratio,
        random_state=random_state,
    )

    feature_cols = list(split.X_train.columns)

    # 2. Train Dummy Baseline
    dummy_pipe = build_classification_pipeline("dummy", random_state=random_state)
    dummy_pipe.fit(split.X_train, split.y_train)
    dummy_val_prob = safe_predict_proba(dummy_pipe, split.X_val)
    dummy_val_eval = evaluate_model_performance(
        y_true=split.y_val,
        y_prob=dummy_val_prob,
        decision_threshold=0.5,
        patient_ids=split.val_patients,
        recording_files=split.val_files,
        model_name="Dummy Baseline",
        split_name="Validation",
    )

    # 3. Build & Fit Target Model Pipeline
    target_pipe = build_classification_pipeline(model_type, random_state=random_state)
    target_pipe.fit(split.X_train, split.y_train)

    # 4. Tune Threshold on Validation Split
    opt_threshold, best_val_f1, _ = tune_decision_threshold(
        target_pipe, split.X_val, split.y_val
    )

    # 5. Evaluate on Train, Val, and Test
    train_probs = safe_predict_proba(target_pipe, split.X_train)
    train_eval = evaluate_model_performance(
        y_true=split.y_train,
        y_prob=train_probs,
        decision_threshold=opt_threshold,
        patient_ids=split.train_patients,
        recording_files=split.train_files,
        model_name=f"{model_type} (Train)",
        split_name="Train",
    )

    val_probs = safe_predict_proba(target_pipe, split.X_val)
    val_eval = evaluate_model_performance(
        y_true=split.y_val,
        y_prob=val_probs,
        decision_threshold=opt_threshold,
        patient_ids=split.val_patients,
        recording_files=split.val_files,
        model_name=f"{model_type} (Val)",
        split_name="Validation",
    )

    test_eval = None
    if split.X_test is not None and len(split.X_test) > 0:
        test_probs = safe_predict_proba(target_pipe, split.X_test)
        test_eval = evaluate_model_performance(
            y_true=split.y_test,
            y_prob=test_probs,
            decision_threshold=opt_threshold,
            patient_ids=split.test_patients,
            recording_files=split.test_files,
            model_name=f"{model_type} (Test)",
            split_name="Test",
        )

    bundle = TrainedModelBundle(
        model_name=model_type,
        pipeline=target_pipe,
        feature_names=feature_cols,
        optimal_threshold=opt_threshold,
        train_evaluation=train_eval,
        val_evaluation=val_eval,
        test_evaluation=test_eval,
        dummy_baseline_evaluation=dummy_val_eval,
        train_patients=split.train_patients,
        val_patients=split.val_patients,
        test_patients=split.test_patients,
        trained_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_dict={
            "model_type": model_type,
            "random_state": random_state,
            "n_features": len(feature_cols),
            "n_train_samples": len(split.X_train),
            "n_val_samples": len(split.X_val),
            "n_test_samples": len(split.X_test) if split.X_test is not None else 0,
            "warnings": split.warnings,
        },
    )

    if save_bundle:
        save_model_bundle(bundle, bundle_name=bundle_name)

    return bundle


def save_model_bundle(
    bundle: TrainedModelBundle,
    target_path: Optional[Path] = None,
    bundle_name: Optional[str] = None,
) -> Path:
    """Serialize and save trained model bundle with joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if target_path is None:
        name = bundle_name or f"{bundle.model_name}_seizure_model.joblib"
        target_path = MODELS_DIR / name

    joblib.dump(bundle, target_path)
    return target_path


def load_model_bundle(path_or_name: str | Path) -> TrainedModelBundle:
    """Load persisted model bundle from disk."""
    p = Path(path_or_name)
    if not p.is_absolute() and not p.exists():
        p = MODELS_DIR / path_or_name
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found: {p}")
    bundle = joblib.load(p)
    return bundle
