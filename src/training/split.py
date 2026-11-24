"""
Patient-level splitting and group cross-validation utilities to prevent data leakage.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass
class SplitResult:
    """Container for partitioned features, labels, and patient metadata."""
    X_train: pd.DataFrame
    y_train: np.ndarray
    train_patients: List[str]
    train_files: List[str]

    X_val: pd.DataFrame
    y_val: np.ndarray
    val_patients: List[str]
    val_files: List[str]

    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[np.ndarray] = None
    test_patients: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)


def create_patient_train_test_split(
    df: pd.DataFrame,
    test_patient_ratio: float = 0.3,
    val_patient_ratio: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """
    Split DataFrame strictly by patient ID so that no patient's windows appear
    in both training and evaluation splits.

    If only 1 patient is present in the data, a recording-level or chronological
    split is performed with an explicit academic warning regarding single-subject generalizability.
    """
    meta_cols = ["target_label", "is_ambiguous", "patient_id", "file_name", "window_idx", "start_sec", "end_sec"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Filter out ambiguous boundary windows for training
    clean_df = df[df["target_label"].isin([0, 1])].copy()
    if clean_df.empty:
        raise ValueError("No valid labeled windows available after filtering ambiguous boundaries.")

    unique_patients = list(clean_df["patient_id"].unique())
    unique_files = list(clean_df["file_name"].unique())
    warnings: List[str] = []

    if len(unique_patients) >= 3:
        # Multi-patient split: train / val / test by patient
        rng = np.random.default_rng(random_state)
        shuffled_patients = list(unique_patients)
        rng.shuffle(shuffled_patients)

        n_test = max(1, int(round(len(unique_patients) * test_patient_ratio)))
        n_val = max(1, int(round(len(unique_patients) * val_patient_ratio)))
        # Ensure at least 1 patient remains for train
        if len(shuffled_patients) - n_test - n_val < 1:
            n_val = 0

        test_pats = shuffled_patients[:n_test]
        val_pats = shuffled_patients[n_test : n_test + n_val]
        train_pats = shuffled_patients[n_test + n_val :]

        train_mask = clean_df["patient_id"].isin(train_pats)
        val_mask = clean_df["patient_id"].isin(val_pats)
        test_mask = clean_df["patient_id"].isin(test_pats)

    elif len(unique_patients) == 2:
        # 2 patients: 1 train, 1 val/test
        train_pats = [unique_patients[0]]
        val_pats = [unique_patients[1]]
        test_pats = [unique_patients[1]]
        train_mask = clean_df["patient_id"].isin(train_pats)
        val_mask = clean_df["patient_id"].isin(val_pats)
        test_mask = val_mask
        warnings.append(
            "Only 2 patients available. Patient-level split is evaluated with 1 patient for training and 1 for testing."
        )

    else:
        # Single patient available
        single_pat = unique_patients[0]
        train_pats = [single_pat]
        val_pats = [single_pat]
        test_pats = [single_pat]

        warnings.append(
            f"SINGLE-PATIENT CAUTION: Only 1 patient ('{single_pat}') is available in this dataset. "
            "Partitioning by recording files to prevent intra-window temporal leakage. "
            "Cross-patient generalization cannot be proven without additional patients."
        )

        if len(unique_files) >= 2:
            # Split by files
            rng = np.random.default_rng(random_state)
            shuffled_files = list(unique_files)
            rng.shuffle(shuffled_files)
            split_idx = max(1, int(len(shuffled_files) * 0.7))
            train_file_subset = set(shuffled_files[:split_idx])
            val_file_subset = set(shuffled_files[split_idx:])

            train_mask = clean_df["file_name"].isin(train_file_subset)
            val_mask = clean_df["file_name"].isin(val_file_subset)
            test_mask = val_mask
        else:
            # Single file: chronological split (first 70% train, last 30% val/test)
            n_rows = len(clean_df)
            split_idx = int(n_rows * 0.7)
            train_mask = np.zeros(n_rows, dtype=bool)
            train_mask[:split_idx] = True
            val_mask = ~train_mask
            test_mask = val_mask
            warnings.append(
                "Single recording file: performing chronological temporal split (70% train / 30% test)."
            )

    train_df = clean_df[train_mask]
    val_df = clean_df[val_mask]
    test_df = clean_df[test_mask] if test_mask is not None else val_df

    return SplitResult(
        X_train=train_df[feature_cols].copy(),
        y_train=train_df["target_label"].to_numpy(dtype=int),
        train_patients=list(train_df["patient_id"].unique()),
        train_files=list(train_df["file_name"].unique()),
        X_val=val_df[feature_cols].copy(),
        y_val=val_df["target_label"].to_numpy(dtype=int),
        val_patients=list(val_df["patient_id"].unique()),
        val_files=list(val_df["file_name"].unique()),
        X_test=test_df[feature_cols].copy(),
        y_test=test_df["target_label"].to_numpy(dtype=int),
        test_patients=list(test_df["patient_id"].unique()),
        test_files=list(test_df["file_name"].unique()),
        warnings=warnings,
    )


def get_patient_grouped_cv(
    df: pd.DataFrame,
    n_splits: int = 3,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate patient-grouped cross-validation splits."""
    clean_df = df[df["target_label"].isin([0, 1])]
    groups = clean_df["patient_id"].to_numpy()
    unique_groups = np.unique(groups)

    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 distinct patients for patient-grouped CV, got {len(unique_groups)}.")

    gkf = GroupKFold(n_splits=actual_splits)
    X_dummy = np.zeros((len(clean_df), 1))
    return gkf.split(X_dummy, clean_df["target_label"].to_numpy(), groups=groups)
