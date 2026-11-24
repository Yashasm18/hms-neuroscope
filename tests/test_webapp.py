"""Tests for the FastAPI web app (webapp/) that replaced the Streamlit dashboard."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import RAW_DATA_DIR
from src.inference.predictor import SeizurePredictor
from webapp import service as S
from webapp.server import app

client = TestClient(app, headers={"X-NeuroScope": "1"})
HAS_TEST_EDF = (RAW_DATA_DIR / "chb02" / "chb02_16.edf").exists()


def test_config_and_frontend_served():
    cfg = client.get("/api/config").json()
    assert cfg["default_recording"] == "chb01_03"
    page = client.get("/")
    assert page.status_code == 200 and "Harmful Brain Activity" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/vendor/plotly.min.js").status_code == 200


def test_unknown_recording_and_artifact_paths_rejected():
    assert client.get("/api/recording/does_not_exist").status_code == 404
    assert client.get("/artifacts/metrics.json").status_code == 404
    assert client.get("/artifacts/..%2Fconfig.py").status_code == 404


def test_post_without_app_header_is_refused():
    bare = TestClient(app)
    assert bare.post("/api/detect", json={"rec": "demo"}).status_code == 403
    assert bare.post("/api/train/real").status_code == 403


def test_upload_rejects_non_edf():
    r = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_training_blocked_in_public_demo(monkeypatch):
    monkeypatch.setattr(S, "PUBLIC_DEMO", True)
    assert client.post("/api/train/real").status_code == 403
    assert client.post("/api/train/synthetic", json={}).status_code == 403
    assert client.post("/api/summary/chb03").status_code == 403


def test_synthetic_demo_detection_and_explorer():
    r = client.post("/api/detect", json={"rec": "demo"})
    assert r.status_code == 200
    body = r.json()
    assert body["windows"] == 18  # 90 s / 5 s windows
    assert body["timeline"]["data"]

    ex = client.get("/api/explore", params={"rec": "demo", "start": 25, "span": 10, "channels": "FP1-F7,F7-T7"})
    assert ex.status_code == 200
    assert len(ex.json()["eeg"]["data"]) >= 2


@pytest.mark.skipif(not HAS_TEST_EDF, reason="chb02_16.edf not present")
def test_chunked_scores_match_full_predictor():
    """Chunked, padded filtering must give the same window scores as the original predictor."""
    rec = S.STORE.get("chb02_16")
    bundle = S.MODEL.get()
    chunked = S.score_recording(rec.eeg, bundle, chunk_sec=60.0)  # forces several chunks
    full = SeizurePredictor(bundle).predict_recording(rec.eeg)
    assert len(chunked.probs) == len(full.timeline_df)
    np.testing.assert_allclose(chunked.probs, full.timeline_df["seizure_score"].to_numpy(), atol=1e-3)
    np.testing.assert_allclose(chunked.starts, full.timeline_df["start_sec"].to_numpy())


@pytest.mark.skipif(not HAS_TEST_EDF, reason="chb02_16.edf not present")
def test_threshold_changes_reuse_cached_scores():
    lo = client.post("/api/detect", json={"rec": "chb02_16", "threshold": 0.05}).json()
    hi = client.post("/api/detect", json={"rec": "chb02_16", "threshold": 0.95}).json()
    assert lo["windows"] == hi["windows"]
    assert lo["total_seizure_sec"] >= hi["total_seizure_sec"]
    assert any(e["matches_truth"] for e in lo["episodes"])


def test_bundled_score_caches_match_current_model():
    """Shipped score caches must exist for the shipped model and match a fresh computation."""
    bundle = S.MODEL.get()
    path = S._disk_cache_path("chb02_16")
    assert path is not None and path.exists(), "run: python -m webapp.precompute"
    cached = S._read_disk_scores(path)
    fresh = S.score_recording(S.STORE.get("chb02_16").eeg, bundle)
    np.testing.assert_array_equal(cached.probs, fresh.probs)
