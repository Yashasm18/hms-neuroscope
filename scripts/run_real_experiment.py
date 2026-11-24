"""
HMS — Harmful Brain Activity Classification & Seizure Detection
Official CHB-MIT Scalp EEG Multi-Recording Patient-Level Experiment Runner.

Executes:
1. Single-recording chronological split audit (chb01_03.edf) confirming 0 seizure windows in Train/Val.
2. Valid multi-recording experiment with strict patient-level holdout:
   - Train: chb01_03.edf (Patient chb01, 8 seizure windows, 599 non-seizure)
   - Validation: chb01_04.edf (Patient chb01, 6 seizure windows, 294 non-seizure)
   - Test: chb02_16.edf (Patient chb02, 13 seizure windows, 25 non-seizure)
3. Fits StandardScaler & RandomForestClassifier exclusively on Train.
4. Tunes decision threshold exclusively on Validation to maximize F1-score.
5. Evaluates on unseen Patient chb02 Test partition with honest, un-fabricated metrics.
6. Saves reproducible artifacts in results/.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import sklearn
import scipy

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    ANNOTATIONS_DIR,
    STANDARD_CHANNELS,
    DEFAULT_SAMPLING_RATE,
    MEDICAL_DISCLAIMER,
)
from src.data.edf_loader import load_edf_recording
from src.data.annotation_parser import load_verified_recording_annotations
from src.preprocessing.windowing import segment_eeg_windows
from src.features.extractor import FeatureExtractor
from src.training.pipeline import build_classification_pipeline
from src.training.train import (
    safe_predict_proba,
    tune_decision_threshold,
    TrainedModelBundle,
    save_model_bundle,
)
from src.evaluation.metrics import (
    calculate_window_metrics,
    evaluate_model_performance,
)
from scripts.audit_chb01_03 import audit_chb01_03_chronological_split


def run_experiment(output_dir: str = "results") -> TrainedModelBundle:
    results_path = PROJECT_ROOT / output_dir
    results_path.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file_path = results_path / "run_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger("CHBMIT_Experiment")

    logger.info("=" * 80)
    logger.info("CHB-MIT SCALP EEG SEIZURE DETECTION EXPERIMENT")
    logger.info("=" * 80)

    # --------------------------------------------------------------------------
    # STEP 1: Audit Single-Recording Chronological Partition
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 1: Auditing Single-Recording Chronological Split (chb01_03.edf) ---")
    audit_data = audit_chb01_03_chronological_split()
    logger.info(f"Audit completed: {audit_data['audit_finding']}")
    logger.info(
        f"Train [0-1830s]: {audit_data['chronological_partitions']['train']['seizure_windows']} seizure, "
        f"{audit_data['chronological_partitions']['train']['non_seizure_windows']} non-seizure"
    )
    logger.info(
        f"Val [1830-2440s]: {audit_data['chronological_partitions']['validation']['seizure_windows']} seizure, "
        f"{audit_data['chronological_partitions']['validation']['non_seizure_windows']} non-seizure"
    )
    logger.info(
        f"Test [2440-3050s]: {audit_data['chronological_partitions']['test']['seizure_windows']} seizure, "
        f"{audit_data['chronological_partitions']['test']['non_seizure_windows']} non-seizure, "
        f"{audit_data['chronological_partitions']['test']['ambiguous_windows']} ambiguous excluded"
    )

    # --------------------------------------------------------------------------
    # STEP 2: Load Multi-Recording Dataset (Patient-Level Holdout)
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 2: Loading Multi-Recording CHB-MIT Dataset with Patient-Level Holdout ---")

    # Train: chb01_03.edf
    train_edf_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
    train_sum_path = ANNOTATIONS_DIR / "chb01-summary.txt"
    train_sz_path = RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures"
    logger.info(f"Loading Train recording: {train_edf_path.name}...")
    eeg_train = load_edf_recording(train_edf_path)
    sum_train = load_verified_recording_annotations("chb01_03.edf", summary_path=train_sum_path, seizures_path=train_sz_path)

    # Validation: chb01_04.edf (separate recording from patient chb01)
    val_edf_path = RAW_DATA_DIR / "chb01" / "chb01_04.edf"
    val_sz_path = RAW_DATA_DIR / "chb01" / "chb01_04.edf.seizures"
    logger.info(f"Loading Validation recording: {val_edf_path.name}...")
    eeg_val = load_edf_recording(val_edf_path)
    sum_val = load_verified_recording_annotations("chb01_04.edf", summary_path=train_sum_path, seizures_path=val_sz_path)

    # Test: chb02_16.edf (unseen patient chb02)
    test_edf_path = RAW_DATA_DIR / "chb02" / "chb02_16.edf"
    test_sum_path = ANNOTATIONS_DIR / "chb02-summary.txt"
    test_sz_path = RAW_DATA_DIR / "chb02" / "chb02_16.edf.seizures"
    logger.info(f"Loading Test recording (Unseen Patient chb02): {test_edf_path.name}...")
    eeg_test = load_edf_recording(test_edf_path)
    sum_test = load_verified_recording_annotations("chb02_16.edf", summary_path=test_sum_path, seizures_path=test_sz_path)
    for sz in sum_test.seizures:
        sz.end_sec = min(sz.end_sec, eeg_test.duration_sec)

    # --------------------------------------------------------------------------
    # STEP 3: Window Segmentation and Labeling
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 3: Segmenting EEG Windows (5.0s non-overlapping, 5.0s boundary buffer) ---")
    window_sec = 5.0
    step_sec = 5.0
    buffer_sec = 5.0

    ds_train = segment_eeg_windows(eeg_train.data, eeg_train.sampling_rate, window_sec, step_sec, sum_train, boundary_buffer_sec=buffer_sec, patient_id="chb01", file_name="chb01_03.edf")
    ds_val = segment_eeg_windows(eeg_val.data, eeg_val.sampling_rate, window_sec, step_sec, sum_val, boundary_buffer_sec=buffer_sec, patient_id="chb01", file_name="chb01_04.edf")
    ds_test = segment_eeg_windows(eeg_test.data, eeg_test.sampling_rate, window_sec, step_sec, sum_test, boundary_buffer_sec=buffer_sec, patient_id="chb02", file_name="chb02_16.edf")

    st_train = ds_train.summary_stats()
    st_val = ds_val.summary_stats()
    st_test = ds_test.summary_stats()

    logger.info(f"Train (chb01_03.edf): {st_train['total_windows']} windows ({st_train['seizure_windows']} seizure, {st_train['non_seizure_windows']} non-seizure, {st_train['ambiguous_windows']} ambiguous excluded)")
    logger.info(f"Val   (chb01_04.edf): {st_val['total_windows']} windows ({st_val['seizure_windows']} seizure, {st_val['non_seizure_windows']} non-seizure, {st_val['ambiguous_windows']} ambiguous excluded)")
    logger.info(f"Test  (chb02_16.edf): {st_test['total_windows']} windows ({st_test['seizure_windows']} seizure, {st_test['non_seizure_windows']} non-seizure, {st_test['ambiguous_windows']} ambiguous excluded)")

    # --------------------------------------------------------------------------
    # STEP 4: Feature Extraction
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 4: Multichannel Feature Extraction ---")
    extractor = FeatureExtractor(sfreq=eeg_train.sampling_rate, target_channels=eeg_train.channel_names)

    clean_train_ds = ds_train.get_clean_training_subset()
    clean_val_ds = ds_val.get_clean_training_subset()
    clean_test_ds = ds_test.get_clean_training_subset()

    df_train = extractor.transform_dataset(clean_train_ds, channel_names=eeg_train.channel_names)
    df_val = extractor.transform_dataset(clean_val_ds, channel_names=eeg_val.channel_names)
    df_test = extractor.transform_dataset(clean_test_ds, channel_names=eeg_test.channel_names)

    feature_cols = [c for c in df_train.columns if c not in ["window_idx", "start_sec", "end_sec", "target_label", "is_ambiguous", "patient_id", "file_name"]]
    logger.info(f"Extracted {len(feature_cols)} features per window across {len(eeg_train.channel_names)} channels.")

    X_train, y_train = df_train[feature_cols].copy(), df_train["target_label"].to_numpy(dtype=int)
    X_val, y_val = df_val[feature_cols].copy(), df_val["target_label"].to_numpy(dtype=int)
    X_test, y_test = df_test[feature_cols].copy(), df_test["target_label"].to_numpy(dtype=int)

    # --------------------------------------------------------------------------
    # STEP 5: Train Model Pipeline (Fit Scaler & Classifier Exclusively on Train)
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 5: Fitting Pipeline Exclusively on Training Data ---")
    pipe = build_classification_pipeline("random_forest", random_state=42, class_weight="balanced")
    pipe.fit(X_train, y_train)
    logger.info(f"Trained RandomForestClassifier on {len(X_train)} samples ({np.sum(y_train == 1)} pos, {np.sum(y_train == 0)} neg).")

    # --------------------------------------------------------------------------
    # STEP 6: Tune Decision Threshold on Validation Data (chb01_04.edf)
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 6: Calibrating Decision Threshold on Validation Data (chb01_04.edf) ---")
    opt_thresh, best_val_f1, thresh_scores = tune_decision_threshold(pipe, X_val, y_val)
    logger.info(f"Threshold search complete: optimal threshold = {opt_thresh:.2f} (Val F1 = {best_val_f1:.4f})")

    # --------------------------------------------------------------------------
    # STEP 7: Honest Evaluation on Unseen Test Partition (Patient chb02)
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 7: Evaluating on Held-Out Test Partition (Patient chb02) ---")
    train_probs = safe_predict_proba(pipe, X_train)
    val_probs = safe_predict_proba(pipe, X_val)
    test_probs = safe_predict_proba(pipe, X_test)

    train_eval = evaluate_model_performance(y_train, train_probs, decision_threshold=opt_thresh, split_name="Train")
    val_eval = evaluate_model_performance(y_val, val_probs, decision_threshold=opt_thresh, split_name="Validation")

    test_window_starts = df_test["start_sec"].to_numpy()
    test_window_ends = df_test["end_sec"].to_numpy()
    ref_events_test = [(s.start_sec, s.end_sec) for s in sum_test.seizures]
    total_test_hours = len(df_test) * window_sec / 3600.0

    test_eval = evaluate_model_performance(
        y_true=y_test,
        y_prob=test_probs,
        decision_threshold=opt_thresh,
        patient_ids=["chb02"],
        recording_files=["chb02_16.edf"],
        reference_events=ref_events_test,
        window_start_times=test_window_starts,
        window_end_times=test_window_ends,
        total_hours=total_test_hours,
        model_name="Random Forest (Patient-Level Holdout)",
        split_name="Test",
    )

    tm = test_eval.window_metrics
    logger.info(f"Test Accuracy:         {tm.accuracy:.4f}")
    logger.info(f"Test Balanced Acc:     {tm.balanced_accuracy:.4f}")
    logger.info(f"Test Sensitivity/Rec:  {tm.sensitivity:.4f} ({tm.tp}/{tm.tp+tm.fn})")
    logger.info(f"Test Specificity:      {tm.specificity:.4f} ({tm.tn}/{tm.tn+tm.fp})")
    logger.info(f"Test Precision:        {tm.precision:.4f} ({tm.tp}/{tm.tp+tm.fp})")
    logger.info(f"Test F1-Score:         {tm.f1_score:.4f}")
    logger.info(f"Test ROC-AUC:          {tm.roc_auc if tm.roc_auc is not None else 'N/A'}")
    logger.info(f"Confusion Matrix:      TN={tm.tn}, FP={tm.fp}, FN={tm.fn}, TP={tm.tp}")

    if test_eval.event_metrics:
        em = test_eval.event_metrics
        logger.info(f"Event Sensitivity:     {em.event_sensitivity:.2%}")
        logger.info(f"False Alarms (Events): {em.false_alarm_events}")
        logger.info(f"FAR per Hour:          {em.false_alarms_per_hour:.2f} / hr")

    # --------------------------------------------------------------------------
    # STEP 8: Save Reproducible Artifacts in results/
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 8: Saving Reproducible Artifacts in results/ ---")

    # 1. metrics.json
    metrics_data = {
        "dataset": "CHB-MIT Scalp EEG Database (v1.0.0)",
        "protocol": "Patient-Level Holdout Experiment",
        "partitions": {
            "train": {
                "patient": "chb01",
                "recording": "chb01_03.edf",
                "duration_sec": float(eeg_train.duration_sec),
                "total_windows": int(st_train["total_windows"]),
                "seizure_windows": int(st_train["seizure_windows"]),
                "non_seizure_windows": int(st_train["non_seizure_windows"]),
                "ambiguous_windows": int(st_train["ambiguous_windows"]),
            },
            "validation": {
                "patient": "chb01",
                "recording": "chb01_04.edf",
                "duration_sec": float(eeg_val.duration_sec),
                "total_windows": int(st_val["total_windows"]),
                "seizure_windows": int(st_val["seizure_windows"]),
                "non_seizure_windows": int(st_val["non_seizure_windows"]),
                "ambiguous_windows": int(st_val["ambiguous_windows"]),
            },
            "test": {
                "patient": "chb02",
                "recording": "chb02_16.edf",
                "duration_sec": float(eeg_test.duration_sec),
                "total_windows": int(st_test["total_windows"]),
                "seizure_windows": int(st_test["seizure_windows"]),
                "non_seizure_windows": int(st_test["non_seizure_windows"]),
                "ambiguous_windows": int(st_test["ambiguous_windows"]),
            },
        },
        "sampling_rate_hz": float(eeg_train.sampling_rate),
        "decision_threshold": float(opt_thresh),
        "train_metrics": {
            "accuracy": float(train_eval.window_metrics.accuracy),
            "sensitivity": float(train_eval.window_metrics.sensitivity),
            "specificity": float(train_eval.window_metrics.specificity),
            "precision": float(train_eval.window_metrics.precision),
            "f1_score": float(train_eval.window_metrics.f1_score),
            "roc_auc": float(train_eval.window_metrics.roc_auc) if train_eval.window_metrics.roc_auc else None,
            "confusion_matrix": {"tn": train_eval.window_metrics.tn, "fp": train_eval.window_metrics.fp, "fn": train_eval.window_metrics.fn, "tp": train_eval.window_metrics.tp},
        },
        "validation_metrics": {
            "accuracy": float(val_eval.window_metrics.accuracy),
            "sensitivity": float(val_eval.window_metrics.sensitivity),
            "specificity": float(val_eval.window_metrics.specificity),
            "precision": float(val_eval.window_metrics.precision),
            "f1_score": float(val_eval.window_metrics.f1_score),
            "roc_auc": float(val_eval.window_metrics.roc_auc) if val_eval.window_metrics.roc_auc else None,
            "confusion_matrix": {"tn": val_eval.window_metrics.tn, "fp": val_eval.window_metrics.fp, "fn": val_eval.window_metrics.fn, "tp": val_eval.window_metrics.tp},
        },
        "test_metrics": {
            "accuracy": float(test_eval.window_metrics.accuracy),
            "sensitivity": float(test_eval.window_metrics.sensitivity),
            "specificity": float(test_eval.window_metrics.specificity),
            "precision": float(test_eval.window_metrics.precision),
            "f1_score": float(test_eval.window_metrics.f1_score),
            "roc_auc": float(test_eval.window_metrics.roc_auc) if test_eval.window_metrics.roc_auc else None,
            "confusion_matrix": {"tn": tm.tn, "fp": tm.fp, "fn": tm.fn, "tp": tm.tp},
        },
        "event_metrics": {
            "total_true_events": test_eval.event_metrics.total_true_events if test_eval.event_metrics else 0,
            "detected_true_events": test_eval.event_metrics.detected_true_events if test_eval.event_metrics else 0,
            "event_sensitivity": float(test_eval.event_metrics.event_sensitivity) if test_eval.event_metrics else 0.0,
            "false_alarms": test_eval.event_metrics.false_alarm_events if test_eval.event_metrics else 0,
            "false_alarms_per_hour": float(test_eval.event_metrics.false_alarms_per_hour) if test_eval.event_metrics else 0.0,
        },
        "audit_note": audit_data["audit_finding"],
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }
    with open(results_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 2. classification_report.txt
    from sklearn.metrics import classification_report
    y_test_pred = (test_probs >= opt_thresh).astype(int)
    rep_str = classification_report(y_test, y_test_pred, target_names=["Non-Seizure (0)", "Seizure (1)"], digits=4, zero_division=0)
    rep_full = (
        f"CHB-MIT SCALP EEG SEIZURE DETECTION — TEST CLASSIFICATION REPORT\n"
        f"Evaluation: Patient-Level Holdout (Patient chb02, Recording chb02_16.edf)\n"
        f"Decision Threshold: {opt_thresh:.2f} (tuned on separate patient chb01 recording chb01_04.edf)\n"
        f"Window Duration: 5.0s (non-overlapping)\n"
        f"----------------------------------------------------------------------\n\n"
        f"{rep_str}\n\n"
        f"Confusion Matrix: TN={tm.tn}, FP={tm.fp}, FN={tm.fn}, TP={tm.tp}\n"
        f"Balanced Accuracy: {tm.balanced_accuracy:.4f}\n"
        f"ROC-AUC: {tm.roc_auc if tm.roc_auc is not None else 'N/A'}\n\n"
        f"DISCLAIMER: {MEDICAL_DISCLAIMER}\n"
    )
    (results_path / "classification_report.txt").write_text(rep_full, encoding="utf-8")

    # 3. confusion_matrix.png
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cm_mat = np.array([[tm.tn, tm.fp], [tm.fn, tm.tp]])
    im = ax_cm.imshow(cm_mat, interpolation="nearest", cmap="Blues")
    ax_cm.set_title(f"Confusion Matrix: Patient Holdout (chb02_16.edf)\nThreshold = {opt_thresh:.2f}", fontsize=11)
    plt.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Non-Seizure (0)", "Seizure (1)"])
    ax_cm.set_yticklabels(["Non-Seizure (0)", "Seizure (1)"])
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Ground Truth Label")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm_mat[i, j]:,}", ha="center", va="center", color="white" if cm_mat[i, j] > np.max(cm_mat)/2 else "black", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_cm.savefig(results_path / "confusion_matrix.png", dpi=150)
    plt.close(fig_cm)

    # 4. seizure_timeline.png
    fig_time, ax_time = plt.subplots(figsize=(10, 4))
    t_mid = (df_test["start_sec"] + df_test["end_sec"]) / 2.0
    for sz_start, sz_end in ref_events_test:
        ax_time.axvspan(sz_start, sz_end, color="#10B981", alpha=0.3, label="Ground Truth Seizure (chb02)")
    ax_time.plot(t_mid, test_probs, color="#EF4444", linewidth=1.8, label="Predicted Risk Score")
    ax_time.axhline(opt_thresh, color="#F59E0B", linestyle="--", linewidth=1.4, label=f"Threshold ({opt_thresh:.2f})")
    ax_time.set_title(f"Seizure Risk Trajectory: Unseen Patient Test Set ({test_edf_path.name})", fontsize=12)
    ax_time.set_xlabel("Recording Time (seconds)")
    ax_time.set_ylabel("Predicted Risk Score")
    ax_time.set_ylim(-0.05, 1.05)
    ax_time.legend(loc="upper left")
    plt.tight_layout()
    fig_time.savefig(results_path / "seizure_timeline.png", dpi=150)
    plt.close(fig_time)

    # 5. experiment_config.json
    config_dict = {
        "dataset_name": "CHB-MIT Scalp EEG Database",
        "protocol": "Patient-Level Holdout (Cross-Patient Generalization)",
        "train_recording": "chb01_03.edf",
        "train_patient": "chb01",
        "validation_recording": "chb01_04.edf",
        "validation_patient": "chb01",
        "test_recording": "chb02_16.edf",
        "test_patient": "chb02",
        "sampling_rate_hz": float(eeg_train.sampling_rate),
        "channels": eeg_train.channel_names,
        "n_channels": len(eeg_train.channel_names),
        "preprocessing": {
            "bandpass_lowcut_hz": 0.5,
            "bandpass_highcut_hz": 40.0,
            "notch_freq_hz": 60.0,
            "filter_order": 4,
            "zero_phase": True,
            "demean": True,
        },
        "windowing": {
            "duration_sec": window_sec,
            "step_sec": step_sec,
            "overlap_ratio_threshold": 0.5,
            "boundary_buffer_sec": buffer_sec,
        },
        "model": {
            "type": "RandomForestClassifier",
            "class_weight": "balanced",
            "random_state": 42,
            "n_estimators": 100,
        },
        "optimal_threshold": float(opt_thresh),
        "software_versions": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    with open(results_path / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 6. Save model bundle
    bundle = TrainedModelBundle(
        model_name="random_forest_patient_holdout",
        pipeline=pipe,
        feature_names=feature_cols,
        optimal_threshold=opt_thresh,
        train_evaluation=train_eval,
        val_evaluation=val_eval,
        test_evaluation=test_eval,
        dummy_baseline_evaluation=None,
        train_patients=["chb01"],
        val_patients=["chb01"],
        test_patients=["chb02"],
        trained_at_utc=str(pd.Timestamp.now(tz="UTC")),
        config_dict=config_dict,
    )
    save_model_bundle(bundle, target_path=results_path / "trained_model.joblib")
    save_model_bundle(bundle, bundle_name="random_forest_seizure_model.joblib")

    logger.info("All 7 reproducible artifacts successfully written to results/:")
    for f in results_path.iterdir():
        logger.info(f"  - {f.name} ({f.stat().st_size:,} bytes)")

    logger.info("\n" + "=" * 80)
    logger.info("PATIENT-LEVEL HOLDOUT EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    return bundle


if __name__ == "__main__":
    run_experiment()
