"""
CLI Utility to test any new or arbitrary CHB-MIT EEG recording file with explicit input.

Usage examples:
  # Test with explicit command line arguments:
  python scripts/predict_recording.py --edf data/raw/chb02/chb02_16.edf
  
  # Test with optional ground truth annotation verification:
  python scripts/predict_recording.py --edf data/raw/chb02/chb02_16.edf --summary data/annotations/chb02-summary.txt

  # Run interactively (will prompt for file path):
  python scripts/predict_recording.py
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_WINDOW_DURATION_SEC, DEFAULT_WINDOW_STEP_SEC, MODELS_DIR
from src.data.edf_loader import load_edf_recording
from src.data.annotation_parser import load_verified_recording_annotations
from src.inference.predictor import SeizurePredictor


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run continuous EEG seizure prediction on an explicit EDF recording file."
    )
    parser.add_argument(
        "--edf",
        type=str,
        default=None,
        help="Path to the .edf EEG recording file to test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/trained_model.joblib",
        help="Path to the trained model bundle .joblib (default: results/trained_model.joblib).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (0.05 - 0.95). Defaults to optimal calibrated threshold from model bundle.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional path to clinician *-summary.txt file to evaluate accuracy and sensitivity.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save window-by-window prediction scores as a CSV file.",
    )

    args = parser.parse_args()

    # If no EDF provided via arguments, prompt interactively
    edf_input = args.edf
    if not edf_input:
        print("\n" + "=" * 70)
        print("HMS EEG Seizure Detection -- Explicit Input Inference")
        print("=" * 70)
        print("Available verified local recordings:")
        print("  1. data/raw/chb01/chb01_03.edf (Patient 01, 3050s, seizure @ 2996s-3036s)")
        print("  2. data/raw/chb01/chb01_04.edf (Patient 01, 1510s, seizure @ 1467s-1494s)")
        print("  3. data/raw/chb02/chb02_16.edf (Patient 02, 196s, seizure @ 130s-196s)")
        print("-" * 70)
        try:
            edf_input = input("Enter path to .edf file [default: data/raw/chb02/chb02_16.edf]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return

        if not edf_input:
            edf_input = "data/raw/chb02/chb02_16.edf"

    edf_path = Path(edf_input)
    if not edf_path.is_absolute():
        edf_path = PROJECT_ROOT / edf_path

    if not edf_path.exists():
        print(f"\n[ERROR] Specified EDF file not found at: {edf_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        print(f"\n[ERROR] Model bundle not found at: {model_path}")
        print("Please run 'python scripts/run_real_experiment.py' first to train the model.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"LOADING RECORDING: {edf_path.name}")
    print("=" * 70)

    # 1. Load EDF recording
    print("Reading EDF header and channels via MNE-Python...")
    eeg = load_edf_recording(edf_path)
    print(f"  - Sampling Rate: {eeg.sampling_rate:.1f} Hz")
    print(f"  - Duration:      {eeg.duration_sec:.1f} seconds ({eeg.duration_sec / 60.0:.1f} minutes)")
    print(f"  - Channels:      {len(eeg.channel_names)} standardized channels")
    if eeg.warning_messages:
        for w in eeg.warning_messages:
            print(f"  - [WARNING] {w}")

    # 2. Load Predictor and Model Bundle
    print(f"\nLoading trained model bundle from: {model_path.name}...")
    predictor = SeizurePredictor(model_path)
    threshold = args.threshold if args.threshold is not None else predictor.default_threshold
    print(f"  - Active Decision Threshold: {threshold:.2f}")

    # 3. Optional Summary Annotation Alignment
    ref_intervals = []
    if args.summary:
        sum_path = Path(args.summary)
        if not sum_path.is_absolute():
            sum_path = PROJECT_ROOT / sum_path
        if sum_path.exists():
            ann_summary = load_verified_recording_annotations(edf_path.name, summary_path=sum_path)
            if ann_summary and ann_summary.has_seizure:
                for sz in ann_summary.seizures:
                    ref_intervals.append((sz.start_sec, sz.end_sec))
                print(f"  - Clinician Annotations Loaded: {len(ref_intervals)} seizure interval(s)")
                for s_start, s_end in ref_intervals:
                    print(f"      Seizure: {s_start:.1f}s - {s_end:.1f}s")
            else:
                print("  - Clinician Annotations Loaded: 0 seizures in this recording (baseline)")

    # 4. Run Continuous Inference
    print("\nRunning continuous inference (5.0s non-overlapping windows, 0.5-40 Hz filter)...")
    result = predictor.predict_recording(
        eeg=eeg,
        threshold=threshold,
        window_duration_sec=DEFAULT_WINDOW_DURATION_SEC,
        window_step_sec=DEFAULT_WINDOW_STEP_SEC,
    )

    # 5. Output Summary Results
    print("\n" + "=" * 70)
    print("INFERENCE DETECTION SUMMARY")
    print("=" * 70)
    print(f"  - Total Analyzed Windows:  {len(result.timeline_df)}")
    print(f"  - Detected Seizure Windows: {int((result.timeline_df['predicted_class'] == 1).sum())}")
    print(f"  - Detected Episodes:       {result.num_episodes}")
    print(f"  - Total Seizure Duration:  {result.total_seizure_time_sec:.1f} seconds")

    if result.episodes:
        print("\nDetected Seizure Episodes:")
        print(f"  {'Episode #':<10} {'Onset':<12} {'Offset':<12} {'Duration':<12} {'Peak Score':<12} {'Ground Truth Alignment'}")
        print("  " + "-" * 75)
        for ep in result.episodes:
            if ref_intervals:
                matches_gt = any((ep.start_sec < r_end and ep.end_sec > r_start) for r_start, r_end in ref_intervals)
                alignment = "[MATCH: TRUE POSITIVE]" if matches_gt else "[WARNING: FALSE ALARM]"
            else:
                alignment = "N/A (No annotations provided)"
            print(
                f"  {ep.episode_id:<10} {ep.start_sec:>6.1f}s     {ep.end_sec:>6.1f}s     {ep.duration_sec:>6.1f}s     {ep.max_score:>8.3f}     {alignment}"
            )
    else:
        print("\n  [OK] No seizure episodes detected across this recording (all windows scored below threshold).")

    # Optional Evaluation against ground truth if provided
    if ref_intervals and len(result.timeline_df) > 0:
        print("\n" + "-" * 70)
        print("CLINICAL EVALUATION AGAINST ANNOTATIONS:")
        # Calculate window labels
        true_labels = []
        for _, row in result.timeline_df.iterrows():
            w_start = row["start_sec"]
            w_end = row["end_sec"]
            w_dur = w_end - w_start
            overlap = 0.0
            for r_s, r_e in ref_intervals:
                ov = max(0.0, min(w_end, r_e) - max(w_start, r_s))
                overlap = max(overlap, ov)
            true_labels.append(1 if (overlap / w_dur) >= 0.5 else 0)

        pred_labels = result.timeline_df["predicted_class"].to_numpy()
        true_labels = np.array(true_labels)

        tp = int(np.sum((true_labels == 1) & (pred_labels == 1)))
        tn = int(np.sum((true_labels == 0) & (pred_labels == 0)))
        fp = int(np.sum((true_labels == 0) & (pred_labels == 1)))
        fn = int(np.sum((true_labels == 1) & (pred_labels == 0)))

        acc = (tp + tn) / max(1, len(true_labels))
        sens = tp / max(1, tp + fn) if (tp + fn) > 0 else 1.0
        spec = tn / max(1, tn + fp) if (tn + fp) > 0 else 1.0

        print(f"  - Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  - Window Accuracy:  {acc:.2%}")
        print(f"  - Sensitivity:      {sens:.2%} ({tp}/{tp+fn} seizure windows)")
        print(f"  - Specificity:      {spec:.2%} ({tn}/{tn+fp} normal windows)")

    # 6. Save output CSV if requested
    if args.output_csv:
        out_csv = Path(args.output_csv)
        result.timeline_df.to_csv(out_csv, index=False)
        print(f"\nSaved detailed timeline scores to: {out_csv.resolve()}")

    print("\n" + "=" * 70)
    print("Inference completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
