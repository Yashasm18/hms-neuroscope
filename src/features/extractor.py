"""
Multichannel EEG feature extraction pipeline.
Computes channel-specific and cross-channel summary statistics across segmented windows.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config import DEFAULT_SAMPLING_RATE, STANDARD_CHANNELS
from src.preprocessing.windowing import WindowDataset, WindowSegment
from .time_frequency import extract_single_channel_features


class FeatureExtractor:
    """Coordinated extractor for multichannel EEG feature matrices."""

    def __init__(
        self,
        sfreq: float = DEFAULT_SAMPLING_RATE,
        target_channels: Optional[List[str]] = None,
        include_channel_specific: bool = True,
        include_cross_channel_aggregates: bool = True,
    ):
        self.sfreq = sfreq
        self.target_channels = target_channels or STANDARD_CHANNELS
        self.include_channel_specific = include_channel_specific
        self.include_cross_channel_aggregates = include_cross_channel_aggregates
        self.feature_names_: List[str] = []

    def extract_from_window_data(
        self,
        window_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Extract features from a single window array of shape (n_channels, n_samples).
        """
        channels = channel_names or self.target_channels
        n_channels = len(channels)

        # Dictionary mapping feat_name -> list of values across channels
        channel_feature_collector: Dict[str, List[float]] = {}
        row_dict: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(channels):
            if ch_idx < window_data.shape[0]:
                sig = window_data[ch_idx]
            else:
                sig = np.zeros(window_data.shape[1], dtype=np.float32)

            ch_feats = extract_single_channel_features(sig, sfreq=self.sfreq)

            for feat_name, val in ch_feats.items():
                if feat_name not in channel_feature_collector:
                    channel_feature_collector[feat_name] = []
                channel_feature_collector[feat_name].append(val)

                if self.include_channel_specific:
                    col_name = f"{ch_name}__{feat_name}"
                    row_dict[col_name] = val

        # Cross-channel summary statistics (mean, max, std across all channels)
        if self.include_cross_channel_aggregates:
            for feat_name, vals in channel_feature_collector.items():
                arr = np.array(vals, dtype=np.float32)
                row_dict[f"global_mean__{feat_name}"] = float(np.mean(arr))
                row_dict[f"global_max__{feat_name}"] = float(np.max(arr))
                row_dict[f"global_std__{feat_name}"] = float(np.std(arr))

        return row_dict

    def transform_dataset(
        self,
        dataset: WindowDataset,
        channel_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract features for all windows in a WindowDataset.
        Returns a DataFrame with features plus metadata columns ('target_label', 'patient_id', 'file_name', 'window_idx', 'start_sec', 'end_sec').
        """
        records = []
        for w in dataset.windows:
            feats = self.extract_from_window_data(w.data, channel_names=channel_names)
            # Add metadata
            feats["target_label"] = w.label
            feats["is_ambiguous"] = int(w.is_ambiguous)
            feats["patient_id"] = w.patient_id
            feats["file_name"] = w.file_name
            feats["window_idx"] = w.window_idx
            feats["start_sec"] = w.start_sec
            feats["end_sec"] = w.end_sec
            records.append(feats)

        df = pd.DataFrame(records)
        
        # Save feature column names (excluding metadata)
        meta_cols = {"target_label", "is_ambiguous", "patient_id", "file_name", "window_idx", "start_sec", "end_sec"}
        self.feature_names_ = [c for c in df.columns if c not in meta_cols]
        return df


def extract_features_from_dataset(
    dataset: WindowDataset,
    sfreq: float = DEFAULT_SAMPLING_RATE,
    channel_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience helper to extract features from a WindowDataset."""
    extractor = FeatureExtractor(sfreq=sfreq)
    return extractor.transform_dataset(dataset, channel_names=channel_names)
