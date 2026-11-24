"""
Configurable EEG window segmentation, timeline alignment, and boundary-aware labeling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from config import (
    DEFAULT_SAMPLING_RATE,
    DEFAULT_WINDOW_DURATION_SEC,
    DEFAULT_WINDOW_STEP_SEC,
    DEFAULT_SEIZURE_OVERLAP_RATIO,
    DEFAULT_BOUNDARY_BUFFER_SEC,
)
from src.data.annotation_parser import RecordingSummary, SeizureAnnotation


@dataclass
class WindowSegment:
    """Metadata and signal slice for a single segmented EEG window."""
    window_idx: int
    start_sec: float
    end_sec: float
    label: int  # 1: Seizure, 0: Non-seizure, -1: Ambiguous boundary
    is_ambiguous: bool
    patient_id: str
    file_name: str
    data: np.ndarray  # Shape: (n_channels, n_window_samples)


@dataclass
class WindowDataset:
    """Collection of segmented windows with split/metadata tracking."""
    windows: List[WindowSegment] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def labels(self) -> np.ndarray:
        return np.array([w.label for w in self.windows], dtype=int)

    @property
    def patient_ids(self) -> List[str]:
        return [w.patient_id for w in self.windows]

    @property
    def file_names(self) -> List[str]:
        return [w.file_name for w in self.windows]

    def get_clean_training_subset(self) -> WindowDataset:
        """Return dataset excluding ambiguous boundary windows (label != -1)."""
        clean_windows = [w for w in self.windows if not w.is_ambiguous and w.label in (0, 1)]
        return WindowDataset(windows=clean_windows)

    def summary_stats(self) -> dict:
        total = len(self.windows)
        seizures = sum(1 for w in self.windows if w.label == 1)
        non_seizures = sum(1 for w in self.windows if w.label == 0)
        ambiguous = sum(1 for w in self.windows if w.is_ambiguous or w.label == -1)
        unique_patients = list(set(self.patient_ids))
        unique_files = list(set(self.file_names))
        return {
            "total_windows": total,
            "seizure_windows": seizures,
            "non_seizure_windows": non_seizures,
            "ambiguous_windows": ambiguous,
            "num_patients": len(unique_patients),
            "patients": unique_patients,
            "num_files": len(unique_files),
        }


def segment_eeg_windows(
    data: np.ndarray,
    sfreq: float = DEFAULT_SAMPLING_RATE,
    window_duration_sec: float = DEFAULT_WINDOW_DURATION_SEC,
    window_step_sec: float = DEFAULT_WINDOW_STEP_SEC,
    annotation_summary: Optional[RecordingSummary] = None,
    overlap_ratio_threshold: float = DEFAULT_SEIZURE_OVERLAP_RATIO,
    boundary_buffer_sec: float = DEFAULT_BOUNDARY_BUFFER_SEC,
    patient_id: str = "unknown",
    file_name: str = "recording.edf",
) -> WindowDataset:
    """
    Segment continuous multichannel EEG into fixed duration windows.
    Assign ground-truth binary labels (1: seizure, 0: non-seizure, -1: ambiguous boundary)
    based strictly on clinician seizure annotations in annotation_summary.

    Labels are NEVER inferred from spectral features, adhering to clinical protocol.
    """
    n_channels, n_samples = data.shape
    win_samples = int(round(window_duration_sec * sfreq))
    step_samples = int(round(window_step_sec * sfreq))

    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("Window duration and step must be positive.")

    seizures: List[SeizureAnnotation] = []
    if annotation_summary is not None and annotation_summary.has_seizure:
        seizures = annotation_summary.seizures

    window_list: List[WindowSegment] = []
    curr_start_sample = 0
    win_idx = 0

    while curr_start_sample + win_samples <= n_samples:
        win_end_sample = curr_start_sample + win_samples
        t_start = curr_start_sample / sfreq
        t_end = win_end_sample / sfreq

        label = 0
        is_ambiguous = False

        if seizures:
            # Check overlap with any verified seizure
            max_overlap_duration = 0.0
            is_near_boundary = False

            for sz in seizures:
                overlap = sz.overlaps_window(t_start, t_end)
                if overlap > max_overlap_duration:
                    max_overlap_duration = overlap

                # Check proximity to seizure boundary within buffer zone
                in_pre_ictal_buffer = (sz.start_sec - boundary_buffer_sec) < t_end <= sz.start_sec
                in_post_ictal_buffer = sz.end_sec <= t_start < (sz.end_sec + boundary_buffer_sec)
                if in_pre_ictal_buffer or in_post_ictal_buffer:
                    is_near_boundary = True

            overlap_ratio = max_overlap_duration / window_duration_sec

            if overlap_ratio >= overlap_ratio_threshold:
                label = 1
                is_ambiguous = False
            elif overlap_ratio > 0.0 or is_near_boundary:
                # Partial overlap below threshold or inside transition buffer
                label = -1
                is_ambiguous = True
            else:
                label = 0
                is_ambiguous = False

        window_data = data[:, curr_start_sample:win_end_sample]

        window_list.append(
            WindowSegment(
                window_idx=win_idx,
                start_sec=t_start,
                end_sec=t_end,
                label=label,
                is_ambiguous=is_ambiguous,
                patient_id=patient_id,
                file_name=file_name,
                data=window_data,
            )
        )

        win_idx += 1
        curr_start_sample += step_samples

    return WindowDataset(windows=window_list)
