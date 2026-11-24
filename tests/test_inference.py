"""
Unit tests for continuous seizure inference and episode clustering.
"""

import numpy as np
import pytest
from src.data.synthetic import generate_synthetic_eeg_record
from src.preprocessing.windowing import segment_eeg_windows
from src.features.extractor import FeatureExtractor
from src.training.train import train_and_evaluate_models
from src.inference.predictor import SeizurePredictor


@pytest.fixture
def trained_synthetic_model():
    # Generate 2 synthetic recordings for 2 mock patients
    records = []
    for pat_id in ["chb_syn1", "chb_syn2"]:
        eeg, summary = generate_synthetic_eeg_record(
            patient_id=pat_id,
            file_name=f"{pat_id}_01.edf",
            duration_sec=30.0,
            seizure_intervals=[(10.0, 20.0)],
            random_seed=42,
        )
        dataset = segment_eeg_windows(
            data=eeg.data,
            sfreq=eeg.sampling_rate,
            window_duration_sec=5.0,
            window_step_sec=5.0,
            annotation_summary=summary,
            patient_id=pat_id,
            file_name=eeg.file_name,
        )
        extractor = FeatureExtractor(sfreq=eeg.sampling_rate, target_channels=eeg.channel_names)
        df = extractor.transform_dataset(dataset, channel_names=eeg.channel_names)
        records.append(df)

    import pandas as pd
    combined_df = pd.concat(records, ignore_index=True)

    bundle = train_and_evaluate_models(
        features_df=combined_df,
        model_type="random_forest",
        random_state=42,
        save_bundle=False,
    )
    return bundle


def test_inference_predictor_pipeline(trained_synthetic_model):
    predictor = SeizurePredictor(trained_synthetic_model)

    test_eeg, test_summary = generate_synthetic_eeg_record(
        patient_id="chb_test_infer",
        file_name="infer_sample.edf",
        duration_sec=30.0,
        seizure_intervals=[(10.0, 20.0)],
        random_seed=99,
    )

    result = predictor.predict_recording(test_eeg, threshold=0.5)

    assert result.file_name == "infer_sample.edf"
    assert len(result.timeline_df) > 0
    assert "seizure_score" in result.timeline_df.columns
    assert "status" in result.timeline_df.columns
    assert result.is_synthetic is True
    assert result.total_seizure_time_sec >= 0.0
    assert not any(np.isnan(result.timeline_df["seizure_score"]))
