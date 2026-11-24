"""
Montage standardization and channel mapping for CHB-MIT EEG.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import STANDARD_CHANNELS
from src.data.edf_loader import clean_channel_name


def verify_channel_montage(
    channel_names: List[str],
    reference_channels: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Check input channel names against reference standard montage.
    Returns:
        (matched_channels, missing_channels, extra_channels)
    """
    ref = reference_channels or STANDARD_CHANNELS
    cleaned_input = [clean_channel_name(c) for c in channel_names]

    matched = [c for c in ref if c in cleaned_input]
    missing = [c for c in ref if c not in cleaned_input]
    extra = [c for c in cleaned_input if c not in ref]

    return matched, missing, extra


def standardize_montage(
    data: np.ndarray,
    current_channels: List[str],
    target_channels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Reorder and align EEG matrix rows to target standard montage.
    If a target channel is missing from input, it is filled with zeros and noted in warnings.

    Returns:
        (aligned_data, final_channels, warnings)
    """
    targets = target_channels or STANDARD_CHANNELS
    cleaned_current = [clean_channel_name(c) for c in current_channels]
    channel_to_idx = {name: idx for idx, name in enumerate(cleaned_current)}

    n_samples = data.shape[1]
    aligned_data = np.zeros((len(targets), n_samples), dtype=np.float32)
    warnings = []

    for i, target_ch in enumerate(targets):
        if target_ch in channel_to_idx:
            src_idx = channel_to_idx[target_ch]
            aligned_data[i] = data[src_idx]
        else:
            warnings.append(f"Channel {target_ch} was missing and imputed with zeros.")

    return aligned_data, targets, warnings
