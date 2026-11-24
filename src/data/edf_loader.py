"""
MNE-based EDF EEG loader, channel validator, and montage standardizer.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mne
import numpy as np

from config import DEFAULT_SAMPLING_RATE, STANDARD_CHANNELS


@dataclass
class StandardizedEEG:
    """Container for standardized multichannel EEG recording data and metadata."""
    file_name: str
    patient_id: str
    sampling_rate: float
    duration_sec: float
    total_samples: int
    channel_names: List[str]
    data: np.ndarray  # Shape: (n_channels, n_samples) in microvolts (uV)
    is_synthetic: bool = False
    warning_messages: List[str] = None

    def __post_init__(self):
        if self.warning_messages is None:
            self.warning_messages = []


def clean_channel_name(ch_name: str) -> str:
    """
    Standardize EEG channel names by removing prefixes (e.g. 'EEG '),
    trailing punctuation, and normalizing hyphenation.
    Example: 'EEG FP1-F7' -> 'FP1-F7'
    Also removes MNE running duplicate suffixes like '-0' or '-1' (e.g. 'T8-P8-0' -> 'T8-P8').
    """
    cleaned = re.sub(r"^EEG\s*", "", ch_name.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.rstrip(".-_ ")
    cleaned = cleaned.upper()
    # Strip running duplicate suffix from MNE (e.g. 'T8-P8-0' -> 'T8-P8')
    cleaned = re.sub(r"-\d+$", "", cleaned)
    return cleaned


def validate_eeg_raw(
    raw: mne.io.Raw,
    expected_sfreq: float = DEFAULT_SAMPLING_RATE,
    required_channels: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate an MNE Raw object for sampling rate, channel count, and signal duration.
    Returns: (is_valid, list_of_warning_or_error_messages)
    """
    messages = []
    is_valid = True

    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - expected_sfreq) > 1.0:
        messages.append(
            f"Sampling rate {sfreq:.1f} Hz deviates from expected CHB-MIT standard {expected_sfreq:.1f} Hz."
        )

    n_samples = raw.n_times
    duration = n_samples / sfreq if sfreq > 0 else 0.0
    if duration < 5.0:
        is_valid = False
        messages.append(f"Recording is too short ({duration:.2f} s). Minimum required is 5.0 s.")

    raw_clean_names = [clean_channel_name(c) for c in raw.ch_names]
    target_channels = required_channels or STANDARD_CHANNELS
    present_channels = [c for c in target_channels if c in raw_clean_names]
    missing_channels = [c for c in target_channels if c not in raw_clean_names]

    if len(present_channels) < 4:
        is_valid = False
        messages.append(
            f"Only {len(present_channels)}/{len(target_channels)} standard channels present. "
            f"Present: {present_channels}. Missing: {missing_channels}"
        )
    elif missing_channels:
        messages.append(
            f"Warning: {len(missing_channels)} standard channels missing: {missing_channels[:5]}..."
        )

    return is_valid, messages


def load_edf_recording(
    edf_path: Union[str, Path],
    target_channels: Optional[List[str]] = None,
    resample_to: Optional[float] = None,
    max_duration_sec: Optional[float] = None,
    preload: bool = True,
) -> StandardizedEEG:
    """
    Load an EDF file via MNE-Python, validate channels, and return standardized microvolt signals.
    """
    path = Path(edf_path)
    if not path.exists():
        raise FileNotFoundError(f"EDF file does not exist: {path}")

    # Read raw with MNE
    # Read the header only; signal data is loaded after channel picking and cropping
    # so a long or wide file cannot exhaust memory before the limits apply.
    raw = mne.io.read_raw_edf(
        str(path),
        preload=False,
        verbose="ERROR",
    )

    # Rename channels to standard cleaned names, dropping duplicates if any
    rename_dict = {}
    seen_clean = set()
    channels_to_drop = []
    for ch in raw.ch_names:
        clean = clean_channel_name(ch)
        if clean in seen_clean:
            channels_to_drop.append(ch)
        else:
            seen_clean.add(clean)
            rename_dict[ch] = clean

    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
    raw.rename_channels(rename_dict)

    is_valid, warnings = validate_eeg_raw(raw, required_channels=target_channels)
    if not is_valid:
        raise ValueError(f"EDF validation failed for {path.name}: {'; '.join(warnings)}")

    # Target channels selection
    selected_targets = target_channels or STANDARD_CHANNELS
    available_targets = [c for c in selected_targets if c in raw.ch_names]

    if not available_targets:
        # Fall back to whatever channels exist
        available_targets = raw.ch_names[: min(18, len(raw.ch_names))]
        warnings.append("No standard bipolar channels found. Falling back to existing channels.")

    raw.pick(available_targets)

    # Optional duration clipping (for fast testing / memory safety), before loading data
    if max_duration_sec is not None and raw.times[-1] > max_duration_sec:
        raw.crop(tmin=0.0, tmax=max_duration_sec)

    raw.load_data(verbose="ERROR")

    # Optional resampling
    if resample_to is not None and abs(raw.info["sfreq"] - resample_to) > 1.0:
        raw.resample(resample_to, npad="auto", verbose="ERROR")

    # Extract data in microvolts (MNE stores in Volts by default: 1 V = 1e6 uV)
    data = raw.get_data() * 1e6  # (n_channels, n_samples)

    # Handle NaNs and Infs if any exist
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=1000.0, neginf=-1000.0)
        warnings.append("Invalid NaN or Inf values detected and replaced with zeros.")

    from .annotation_parser import extract_patient_id
    patient_id = extract_patient_id(path.name)

    return StandardizedEEG(
        file_name=path.name,
        patient_id=patient_id,
        sampling_rate=float(raw.info["sfreq"]),
        duration_sec=float(raw.times[-1]),
        total_samples=data.shape[1],
        channel_names=available_targets,
        data=data,
        is_synthetic=False,
        warning_messages=warnings,
    )
