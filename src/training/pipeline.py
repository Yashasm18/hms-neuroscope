"""
Scikit-learn pipeline construction with class imbalance handling.
"""

from __future__ import annotations
from typing import Any, Dict
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classification_pipeline(
    model_type: str = "random_forest",
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Construct an end-to-end scikit-learn pipeline:
    1. SimpleImputer (median)
    2. StandardScaler (fitted ONLY on training split)
    3. Classifier (Dummy baseline or Random Forest / Gradient Boosting)
    """
    model_type = model_type.lower().strip()

    if model_type == "dummy":
        clf = DummyClassifier(strategy="prior", random_state=random_state)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced_subsample" if class_weight == "balanced" else None,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type in ("gradient_boosting", "hist_gradient_boosting"):
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced" if class_weight == "balanced" else None,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'dummy', 'random_forest', or 'gradient_boosting'.")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ]
    )
    return pipeline


def get_classifier_options() -> Dict[str, str]:
    """Available model choices for UI and training routines."""
    return {
        "random_forest": "Random Forest (Balanced Subsample)",
        "gradient_boosting": "Hist-Gradient Boosting (Balanced)",
        "dummy": "Dummy Baseline (Class Prior)",
    }
