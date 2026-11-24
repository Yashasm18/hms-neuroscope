"""Signal preprocessing, filtering, montage standardizing, and window segmentation."""
from .filter import apply_bandpass_filter, apply_notch_filter, preprocess_eeg_signal
from .montage import standardize_montage, verify_channel_montage
from .windowing import segment_eeg_windows, WindowSegment, WindowDataset

__all__ = [
    "apply_bandpass_filter",
    "apply_notch_filter",
    "preprocess_eeg_signal",
    "standardize_montage",
    "verify_channel_montage",
    "segment_eeg_windows",
    "WindowSegment",
    "WindowDataset",
]
