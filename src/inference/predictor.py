"""
Continuous EEG seizure inference, window scoring, and timeline episode clustering.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    DEFAULT_SAMPLING_RATE,
    DEFAULT_WINDOW_DURATION_SEC,
    DEFAULT_WINDOW_STEP_SEC,
    STANDARD_CHANNELS,
    MEDICAL_DISCLAIMER,
)
from src.data.edf_loader import StandardizedEEG
from src.preprocessing.filter import preprocess_eeg_signal
from src.preprocessing.montage import standardize_montage, verify_channel_montage
from src.preprocessing.windowing import segment_eeg_windows
from src.features.extractor import FeatureExtractor
from src.evaluation.metrics import cluster_positive_windows
from src.training.train import TrainedModelBundle, load_model_bundle


@dataclass
class PredictedEpisode:
    """A detected continuous seizure event."""
    episode_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    max_score: float
    mean_score: float


@dataclass
class PredictionResult:
    """Complete inference results over an EEG recording."""
    file_name: str
    patient_id: str
    sampling_rate: float
    duration_sec: float
    decision_threshold: float
    timeline_df: pd.DataFrame
    episodes: List[PredictedEpisode]
    num_episodes: int
    total_seizure_time_sec: float
    is_synthetic: bool
    warnings: List[str] = field(default_factory=list)
    medical_disclaimer: str = MEDICAL_DISCLAIMER


class SeizurePredictor:
    """Runs window-by-window seizure classification on EEG recordings."""

    def __init__(self, model_bundle: Union[TrainedModelBundle, str, Path]):
        if isinstance(model_bundle, (str, Path)):
            self.bundle = load_model_bundle(model_bundle)
        else:
            self.bundle = model_bundle

        self.pipeline = self.bundle.pipeline
        self.expected_features = self.bundle.feature_names
        self.default_threshold = self.bundle.optimal_threshold
        self.extractor = FeatureExtractor(
            sfreq=DEFAULT_SAMPLING_RATE,
            target_channels=STANDARD_CHANNELS,
        )

    def predict_recording(
        self,
        eeg: StandardizedEEG,
        threshold: Optional[float] = None,
        window_duration_sec: float = DEFAULT_WINDOW_DURATION_SEC,
        window_step_sec: float = DEFAULT_WINDOW_STEP_SEC,
    ) -> PredictionResult:
        """
        Run continuous seizure inference across a StandardizedEEG recording.
        """
        active_threshold = threshold if threshold is not None else self.default_threshold
        warnings = list(eeg.warning_messages)

        # 1. Montage verification and alignment
        matched, missing, extra = verify_channel_montage(eeg.channel_names, STANDARD_CHANNELS)
        if missing:
            warnings.append(
                f"Montage discrepancy: {len(missing)} expected channels missing ({missing[:4]}...). "
                "Channels were imputed with zero baseline, which may degrade predictive sensitivity."
            )

        aligned_data, final_channels, align_warns = standardize_montage(
            eeg.data, eeg.channel_names, STANDARD_CHANNELS
        )
        warnings.extend(align_warns)

        # 2. Digital Filtering
        filtered_data = preprocess_eeg_signal(
            aligned_data,
            sfreq=eeg.sampling_rate,
            lowcut=0.5,
            highcut=40.0,
            notch_freq=60.0,
        )

        # 3. Window Segmentation
        dataset = segment_eeg_windows(
            data=filtered_data,
            sfreq=eeg.sampling_rate,
            window_duration_sec=window_duration_sec,
            window_step_sec=window_step_sec,
            patient_id=eeg.patient_id,
            file_name=eeg.file_name,
        )

        if len(dataset) == 0:
            warnings.append("Recording duration was too short to yield any windows.")
            empty_df = pd.DataFrame(
                columns=["window_idx", "start_sec", "end_sec", "seizure_score", "predicted_class", "status"]
            )
            return PredictionResult(
                file_name=eeg.file_name,
                patient_id=eeg.patient_id,
                sampling_rate=eeg.sampling_rate,
                duration_sec=eeg.duration_sec,
                decision_threshold=active_threshold,
                timeline_df=empty_df,
                episodes=[],
                num_episodes=0,
                total_seizure_time_sec=0.0,
                is_synthetic=eeg.is_synthetic,
                warnings=warnings,
            )

        # 4. Feature Extraction
        features_df = self.extractor.transform_dataset(dataset, channel_names=final_channels)

        # Align columns cleanly to match model expected features
        X = features_df.reindex(columns=self.expected_features, fill_value=0.0)

        # 5. Model Probability Scoring
        from src.training.train import safe_predict_proba
        y_probs = safe_predict_proba(self.pipeline, X)
        y_preds = (y_probs >= active_threshold).astype(int)

        # Build timeline DataFrame
        timeline_records = []
        for i, w in enumerate(dataset.windows):
            prob = float(y_probs[i])
            pred = int(y_preds[i])
            timeline_records.append(
                {
                    "window_idx": w.window_idx,
                    "start_sec": round(w.start_sec, 2),
                    "end_sec": round(w.end_sec, 2),
                    "seizure_score": round(prob, 4),
                    "predicted_class": pred,
                    "status": "SEIZURE DETECTED" if pred == 1 else "Normal",
                }
            )

        timeline_df = pd.DataFrame(timeline_records)

        # 6. Cluster into Episodes
        starts = timeline_df["start_sec"].to_numpy()
        ends = timeline_df["end_sec"].to_numpy()
        clustered = cluster_positive_windows(y_preds, starts, ends, max_gap_sec=10.0, min_duration_sec=window_duration_sec)

        episodes = []
        for ep_idx, (ep_start, ep_end) in enumerate(clustered):
            mask = (timeline_df["start_sec"] >= ep_start) & (timeline_df["end_sec"] <= ep_end)
            ep_scores = timeline_df.loc[mask, "seizure_score"]
            max_sc = float(ep_scores.max()) if not ep_scores.empty else float(active_threshold)
            mean_sc = float(ep_scores.mean()) if not ep_scores.empty else float(active_threshold)

            episodes.append(
                PredictedEpisode(
                    episode_id=ep_idx + 1,
                    start_sec=ep_start,
                    end_sec=ep_end,
                    duration_sec=ep_end - ep_start,
                    max_score=round(max_sc, 4),
                    mean_score=round(mean_sc, 4),
                )
            )

        total_sz_time = sum(ep.duration_sec for ep in episodes)

        return PredictionResult(
            file_name=eeg.file_name,
            patient_id=eeg.patient_id,
            sampling_rate=eeg.sampling_rate,
            duration_sec=eeg.duration_sec,
            decision_threshold=active_threshold,
            timeline_df=timeline_df,
            episodes=episodes,
            num_episodes=len(episodes),
            total_seizure_time_sec=total_sz_time,
            is_synthetic=eeg.is_synthetic,
            warnings=warnings,
        )
