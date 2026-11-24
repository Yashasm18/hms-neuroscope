"""
Utility script to initialize demo recordings, extract features, and train a baseline model.
Can be run via: python scripts/prepare_demo_dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config import MODELS_DIR, RAW_DATA_DIR, ANNOTATIONS_DIR
from src.data.synthetic import generate_synthetic_eeg_record
from src.data.annotation_parser import parse_summary_file
from src.preprocessing.windowing import segment_eeg_windows
from src.features.extractor import FeatureExtractor
from src.training.train import train_and_evaluate_models, save_model_bundle


def setup_demo_pipeline():
    print("=" * 70)
    print("HMS Brain Activity Classification - Pipeline Initialization")
    print("=" * 70)

    # Check for downloaded summaries
    summaries = list(ANNOTATIONS_DIR.glob("*-summary.txt"))
    print(f"Found {len(summaries)} official CHB-MIT summary files in {ANNOTATIONS_DIR}:")
    for s in summaries:
        print(f"  - {s.name}")

    # Generate multi-patient demonstration dataset
    # We create 3 distinct patients with distinct recordings
    patients_config = [
        ("chb01_demo", "chb01_03_demo.edf", 90.0, [(30.0, 55.0)], 101),
        ("chb01_demo", "chb01_04_demo.edf", 60.0, [(15.0, 35.0)], 102),
        ("chb02_demo", "chb02_16_demo.edf", 90.0, [(25.0, 50.0)], 201),
        ("chb02_demo", "chb02_19_demo.edf", 60.0, [], 202),  # Non-seizure record
        ("chb03_demo", "chb03_01_demo.edf", 90.0, [(40.0, 65.0)], 301),
        ("chb03_demo", "chb03_02_demo.edf", 60.0, [(20.0, 42.0)], 302),
    ]

    print("\nGenerating realistic multi-patient benchmark recordings (labeled synthetic)...")
    dataset_records = []

    for pat_id, f_name, duration, seizures, seed in patients_config:
        eeg, summary = generate_synthetic_eeg_record(
            patient_id=pat_id,
            file_name=f_name,
            duration_sec=duration,
            seizure_intervals=seizures,
            random_seed=seed,
        )

        dataset = segment_eeg_windows(
            data=eeg.data,
            sfreq=eeg.sampling_rate,
            window_duration_sec=5.0,
            window_step_sec=5.0,
            annotation_summary=summary,
            patient_id=pat_id,
            file_name=f_name,
        )

        extractor = FeatureExtractor(sfreq=eeg.sampling_rate, target_channels=eeg.channel_names)
        df = extractor.transform_dataset(dataset, channel_names=eeg.channel_names)
        dataset_records.append(df)
        print(f"  Processed {f_name} ({pat_id}): {len(dataset)} windows, {summary.num_seizures} seizures")

    all_features_df = pd.concat(dataset_records, ignore_index=True)
    clean_subset = all_features_df[all_features_df["target_label"].isin([0, 1])]
    print(f"\nTotal segmented windows: {len(all_features_df)} ({len(clean_subset)} pure training windows, {len(all_features_df) - len(clean_subset)} boundary buffer)")
    print(f"Class distribution: {clean_subset['target_label'].value_counts().to_dict()}")

    # Train model with patient-isolated split
    print("\nTraining patient-isolated Random Forest baseline...")
    bundle = train_and_evaluate_models(
        features_df=all_features_df,
        model_type="random_forest",
        random_state=42,
        test_ratio=0.33,
        val_ratio=0.33,
        save_bundle=True,
        bundle_name="random_forest_seizure_model.joblib",
    )

    print("\nTraining complete!")
    print(f"  - Train patients: {bundle.train_patients}")
    print(f"  - Validation patients: {bundle.val_patients}")
    print(f"  - Test patients: {bundle.test_patients}")
    print(f"  - Optimal Decision Threshold: {bundle.optimal_threshold:.2f}")
    print(f"  - Validation F1: {bundle.val_evaluation.window_metrics.f1_score:.3f}")
    if bundle.test_evaluation:
        tm = bundle.test_evaluation.window_metrics
        print(f"  - Test Accuracy: {tm.accuracy:.3f}, Sensitivity: {tm.sensitivity:.3f}, Specificity: {tm.specificity:.3f}, F1: {tm.f1_score:.3f}")
    print(f"  - Persisted to: {MODELS_DIR / 'random_forest_seizure_model.joblib'}")


if __name__ == "__main__":
    setup_demo_pipeline()
