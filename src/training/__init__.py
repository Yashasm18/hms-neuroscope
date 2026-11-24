"""Model training, patient-level splitting, and pipeline persistence."""
from .split import create_patient_train_test_split, get_patient_grouped_cv, SplitResult
from .pipeline import build_classification_pipeline, get_classifier_options
from .train import train_and_evaluate_models, TrainedModelBundle, load_model_bundle, save_model_bundle

__all__ = [
    "create_patient_train_test_split",
    "get_patient_grouped_cv",
    "SplitResult",
    "build_classification_pipeline",
    "get_classifier_options",
    "train_and_evaluate_models",
    "TrainedModelBundle",
    "load_model_bundle",
    "save_model_bundle",
]
