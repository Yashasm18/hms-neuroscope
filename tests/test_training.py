"""
Unit tests for patient-level splitting, training pipelines, and threshold calibration.
"""

import numpy as np
import pandas as pd
import pytest
from src.training.split import create_patient_train_test_split
from src.training.pipeline import build_classification_pipeline
from src.training.train import train_and_evaluate_models, save_model_bundle, load_model_bundle


@pytest.fixture
def multi_patient_feature_df():
    """Create mock multi-patient feature dataframe."""
    rng = np.random.default_rng(42)
    rows = []
    # 4 distinct patients
    for pat in ["chb01", "chb02", "chb03", "chb04"]:
        for f_idx in [1, 2]:
            file_name = f"{pat}_{f_idx:02d}.edf"
            for w in range(15):
                # 20% seizure windows
                label = 1 if w in [5, 6, 7] else 0
                feats = {
                    "feat_1": rng.normal(10 if label == 1 else 2, 1.0),
                    "feat_2": rng.normal(5 if label == 1 else 1, 1.0),
                    "feat_3": rng.normal(0, 1.0),
                    "target_label": label,
                    "is_ambiguous": 0,
                    "patient_id": pat,
                    "file_name": file_name,
                    "window_idx": w,
                    "start_sec": float(w * 5),
                    "end_sec": float((w + 1) * 5),
                }
                rows.append(feats)
    return pd.DataFrame(rows)


def test_patient_level_split_isolation(multi_patient_feature_df):
    """Verify that train and test splits have 0% patient leakage."""
    split = create_patient_train_test_split(
        multi_patient_feature_df,
        test_patient_ratio=0.25,
        val_patient_ratio=0.25,
        random_state=42,
    )

    train_set = set(split.train_patients)
    val_set = set(split.val_patients)
    test_set = set(split.test_patients)

    # Assert complete patient independence
    assert len(train_set.intersection(test_set)) == 0
    assert len(train_set.intersection(val_set)) == 0
    assert len(split.X_train) > 0
    assert len(split.X_val) > 0
    assert len(split.X_test) > 0


def test_pipeline_construction():
    pipe_rf = build_classification_pipeline("random_forest", random_state=42)
    assert "imputer" in pipe_rf.named_steps
    assert "scaler" in pipe_rf.named_steps
    assert "classifier" in pipe_rf.named_steps

    pipe_dummy = build_classification_pipeline("dummy", random_state=42)
    assert pipe_dummy.named_steps["classifier"].__class__.__name__ == "DummyClassifier"


def test_train_and_evaluate_workflow(multi_patient_feature_df, tmp_path):
    bundle = train_and_evaluate_models(
        features_df=multi_patient_feature_df,
        model_type="random_forest",
        random_state=42,
        save_bundle=False,
    )

    assert bundle.pipeline is not None
    assert 0.0 <= bundle.optimal_threshold <= 1.0
    assert bundle.train_evaluation.window_metrics.total_samples > 0
    assert bundle.val_evaluation.window_metrics.total_samples > 0
    assert bundle.dummy_baseline_evaluation is not None

    # Test save and load
    save_path = tmp_path / "test_model_bundle.joblib"
    saved = save_model_bundle(bundle, target_path=save_path)
    loaded = load_model_bundle(saved)

    assert loaded.model_name == bundle.model_name
    assert loaded.optimal_threshold == bundle.optimal_threshold
    assert loaded.feature_names == bundle.feature_names
