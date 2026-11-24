"""EEG feature extraction: time-domain statistics and Welch spectral power."""
from .time_frequency import (
    compute_time_domain_features,
    compute_frequency_domain_features,
    extract_single_channel_features,
)
from .extractor import FeatureExtractor, extract_features_from_dataset

__all__ = [
    "compute_time_domain_features",
    "compute_frequency_domain_features",
    "extract_single_channel_features",
    "FeatureExtractor",
    "extract_features_from_dataset",
]
