"""
Backend services for the Harmful Brain Activity Classification web app.

Keeps memory low enough for small hosts (512 MB):
- each recording is held once, as float32, in a small LRU cache shared by every visitor;
- detection scores a recording in padded chunks instead of filtering the full hour at once;
- window scores are cached, so moving the decision threshold only re-clusters episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ANNOTATIONS_DIR,
    BASE_DIR,
    DEFAULT_WINDOW_DURATION_SEC,
    DEFAULT_WINDOW_STEP_SEC,
    MODELS_DIR,
    RAW_DATA_DIR,
    STANDARD_CHANNELS,
)
from src.data.annotation_parser import RecordingSummary, load_verified_recording_annotations
from src.data.edf_loader import StandardizedEEG, load_edf_recording
from src.data.synthetic import generate_synthetic_eeg_record
from src.evaluation.metrics import cluster_positive_windows
from src.inference.predictor import PredictedEpisode
from src.preprocessing.filter import preprocess_eeg_signal
from src.preprocessing.montage import standardize_montage, verify_channel_montage
from src.preprocessing.windowing import segment_eeg_windows
from src.features.extractor import FeatureExtractor
from src.training.train import TrainedModelBundle, load_model_bundle, safe_predict_proba

PUBLIC_DEMO = os.environ.get("PUBLIC_DEMO") == "1"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_RECORDING = "chb01_03"
UPLOAD_MAX_SECONDS = 300.0
MAX_UPLOADS = 3

# Heavy work (EDF parsing, scoring, training) runs one job at a time to bound peak memory.
HEAVY_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Recording registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BundledRecording:
    rec_id: str
    patient: str
    file_name: str
    role: str
    note: str

    @property
    def edf_path(self) -> Path:
        return RAW_DATA_DIR / self.patient / self.file_name


BUNDLED: Dict[str, BundledRecording] = {
    r.rec_id: r
    for r in [
        BundledRecording("chb01_03", "chb01", "chb01_03.edf", "Train", "Patient 01, female, age 11"),
        BundledRecording("chb01_04", "chb01", "chb01_04.edf", "Validation", "Patient 01, threshold tuning"),
        BundledRecording("chb02_16", "chb02", "chb02_16.edf", "Test", "Unseen patient 02"),
    ]
}

# (patient_id, file_name, duration, seizure intervals, seed, label)
DEMOS: Dict[str, Tuple[str, str, float, List[Tuple[float, float]], int, str]] = {
    "demo": ("chb01_demo", "chb01_03_demo.edf", 90.0, [(30.0, 55.0)], 42, "Quick demo · 90 s · seizure 30–55 s"),
    "demo_chb01_03": ("chb01_demo", "chb01_03_demo.edf", 90.0, [(30.0, 55.0)], 101, "chb01_03_demo · 90 s · seizure 30–55 s"),
    "demo_chb01_04": ("chb01_demo", "chb01_04_demo.edf", 60.0, [(15.0, 35.0)], 102, "chb01_04_demo · 60 s · seizure 15–35 s"),
    "demo_chb02_16": ("chb02_demo", "chb02_16_demo.edf", 90.0, [(25.0, 50.0)], 201, "chb02_16_demo · 90 s · seizure 25–50 s"),
    "demo_chb02_19": ("chb02_demo", "chb02_19_demo.edf", 60.0, [], 202, "chb02_19_demo · 60 s · no seizure"),
    "demo_chb03_01": ("chb03_demo", "chb03_01_demo.edf", 90.0, [(40.0, 65.0)], 301, "chb03_01_demo · 90 s · seizure 40–65 s"),
}


@dataclass
class LoadedRecording:
    rec_id: str
    eeg: StandardizedEEG
    summary: Optional[RecordingSummary]
    kind: str  # "real" | "synthetic" | "upload"

    def info(self) -> dict:
        e = self.eeg
        return {
            "id": self.rec_id,
            "kind": self.kind,
            "file_name": e.file_name,
            "patient_id": e.patient_id,
            "duration_sec": round(float(e.duration_sec), 2),
            "sampling_rate": float(e.sampling_rate),
            "channels": list(e.channel_names),
            "is_synthetic": bool(e.is_synthetic),
            "seizures": seizure_intervals(self.summary),
            "warnings": list(e.warning_messages),
        }


def seizure_intervals(summary: Optional[RecordingSummary]) -> List[dict]:
    if not summary or not summary.has_seizure:
        return []
    return [{"index": s.index, "start": float(s.start_sec), "end": float(s.end_sec)} for s in summary.seizures]


class RecordingStore:
    """Small LRU of loaded recordings plus a registry of uploads."""

    def __init__(self, max_real: int):
        self._max_real = max_real
        self._real: "OrderedDict[str, LoadedRecording]" = OrderedDict()
        self._demos: Dict[str, LoadedRecording] = {}
        self._uploads: "OrderedDict[str, LoadedRecording]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, rec_id: str) -> LoadedRecording:
        with self._lock:
            if rec_id in self._real:
                self._real.move_to_end(rec_id)
                return self._real[rec_id]
            if rec_id in self._demos:
                return self._demos[rec_id]
            if rec_id in self._uploads:
                return self._uploads[rec_id]

        if rec_id in BUNDLED:
            return self._load_bundled(BUNDLED[rec_id])
        if rec_id in DEMOS:
            return self._load_demo(rec_id)
        raise KeyError(rec_id)

    def _load_bundled(self, rec: BundledRecording) -> LoadedRecording:
        if not rec.edf_path.exists():
            raise FileNotFoundError(f"{rec.file_name} is not in data/raw. See DATA_SETUP.md.")
        with HEAVY_LOCK:
            with self._lock:
                if rec.rec_id in self._real:
                    return self._real[rec.rec_id]
                # Drop older recordings before loading so two never sit in memory at once.
                while len(self._real) >= self._max_real:
                    self._real.popitem(last=False)
            eeg = load_edf_recording(rec.edf_path, dtype=np.float32)
            summary_path = ANNOTATIONS_DIR / f"{rec.patient}-summary.txt"
            seizures_path = RAW_DATA_DIR / rec.patient / f"{rec.file_name}.seizures"
            summary = load_verified_recording_annotations(
                rec.file_name,
                summary_path=summary_path if summary_path.exists() else None,
                seizures_path=seizures_path if seizures_path.exists() else None,
            )
            loaded = LoadedRecording(rec.rec_id, eeg, summary, "real")
            with self._lock:
                self._real[rec.rec_id] = loaded
            return loaded

    def _load_demo(self, rec_id: str) -> LoadedRecording:
        patient, fname, dur, seizures, seed, _ = DEMOS[rec_id]
        eeg, summary = generate_synthetic_eeg_record(
            patient_id=patient, file_name=fname, duration_sec=dur,
            seizure_intervals=seizures, random_seed=seed,
        )
        eeg.data = eeg.data.astype(np.float32, copy=False)
        loaded = LoadedRecording(rec_id, eeg, summary, "synthetic")
        with self._lock:
            self._demos[rec_id] = loaded
        return loaded

    def add_upload(self, tmp_path: Path, original_name: str) -> LoadedRecording:
        with HEAVY_LOCK:
            eeg = load_edf_recording(tmp_path, max_duration_sec=UPLOAD_MAX_SECONDS, dtype=np.float32)
        eeg.file_name = original_name
        rec_id = f"upload_{uuid.uuid4().hex[:10]}"
        loaded = LoadedRecording(rec_id, eeg, None, "upload")
        with self._lock:
            self._uploads[rec_id] = loaded
            while len(self._uploads) > MAX_UPLOADS:
                old_id, _ = self._uploads.popitem(last=False)
                SCORES.forget(old_id)
        return loaded


STORE = RecordingStore(max_real=1 if PUBLIC_DEMO else 3)


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

class ModelHolder:
    def __init__(self):
        self.bundle: Optional[TrainedModelBundle] = None
        self.version = 0
        self.source = ""
        self.fingerprint: Optional[str] = None  # hash of the model file, keys the on-disk score cache
        self._lock = threading.Lock()

    def get(self) -> Optional[TrainedModelBundle]:
        with self._lock:
            if self.bundle is None:
                self._load_default()
            return self.bundle

    def _load_default(self):
        for path in (RESULTS_DIR / "trained_model.joblib", MODELS_DIR / "random_forest_seizure_model.joblib"):
            if path.exists():
                try:
                    self.bundle = load_model_bundle(path)
                    self.source = str(path.relative_to(BASE_DIR))
                    self.fingerprint = file_fingerprint(path)
                    self.version += 1
                    return
                except Exception as exc:  # pragma: no cover - corrupt artifact
                    print(f"Could not load model {path}: {exc}")

    def snapshot(self):
        """(bundle, version, fingerprint) read together, so they always belong to the same model."""
        with self._lock:
            if self.bundle is None:
                self._load_default()
            return self.bundle, self.version, self.fingerprint

    def set(self, bundle: TrainedModelBundle, source: str, model_path: Optional[Path] = None):
        with self._lock:
            self.bundle = bundle
            self.source = source
            self.fingerprint = file_fingerprint(model_path) if model_path and model_path.exists() else None
            self.version += 1
        SCORES.clear()


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


MODEL = ModelHolder()


def model_info(bundle: Optional[TrainedModelBundle]) -> Optional[dict]:
    if bundle is None:
        return None
    name = bundle.model_name
    return {
        "name": name,
        "is_real": MODEL.source.startswith("results") or "real" in name.lower() or "chb01" in name.lower(),
        "threshold": float(bundle.optimal_threshold),
        "n_features": len(bundle.feature_names),
        "train_patients": list(bundle.train_patients),
        "val_patients": list(bundle.val_patients),
        "test_patients": list(bundle.test_patients),
        "trained_at_utc": bundle.trained_at_utc,
        "source": MODEL.source,
    }


# ---------------------------------------------------------------------------
# Detection (chunked scoring + threshold clustering)
# ---------------------------------------------------------------------------

@dataclass
class WindowScores:
    starts: np.ndarray
    ends: np.ndarray
    probs: np.ndarray
    warnings: List[str]


class ScoreCache:
    def __init__(self):
        self._items: "OrderedDict[Tuple[str, int], WindowScores]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            return self._items.get(key)

    def put(self, key, value):
        with self._lock:
            self._items[key] = value
            while len(self._items) > 16:
                self._items.popitem(last=False)

    def forget(self, rec_id: str):
        with self._lock:
            for k in [k for k in self._items if k[0] == rec_id]:
                del self._items[k]

    def clear(self):
        with self._lock:
            self._items.clear()


SCORES = ScoreCache()

CHUNK_SEC = 600.0
FILTER_PAD_SEC = 30.0


def score_recording(
    eeg: StandardizedEEG,
    bundle: TrainedModelBundle,
    window_sec: float = DEFAULT_WINDOW_DURATION_SEC,
    step_sec: float = DEFAULT_WINDOW_STEP_SEC,
    chunk_sec: float = CHUNK_SEC,
    pad_sec: float = FILTER_PAD_SEC,
) -> WindowScores:
    """
    Same steps as SeizurePredictor.predict_recording (montage alignment, 0.5–40 Hz band-pass,
    60 Hz notch, 5 s windows, features, model probabilities), but the zero-phase filters run on
    padded chunks so peak memory stays flat for hour-long recordings. The padding is far longer
    than the filters' settling time, so window scores match a whole-recording run.
    """
    warnings = list(eeg.warning_messages)
    _, missing, _ = verify_channel_montage(eeg.channel_names, STANDARD_CHANNELS)
    if missing:
        warnings.append(
            f"Montage discrepancy: {len(missing)} expected channels missing ({missing[:4]}...). "
            "Channels were imputed with zero baseline, which may degrade predictive sensitivity."
        )
    # Alignment warnings depend only on channel names, so check them once on a tiny slice.
    _, final_channels, align_warns = standardize_montage(eeg.data[:, :1], eeg.channel_names, STANDARD_CHANNELS)
    warnings.extend(align_warns)

    sf = float(eeg.sampling_rate)
    n = eeg.data.shape[1]
    win = int(round(window_sec * sf))
    step = int(round(step_sec * sf))
    n_windows = 0 if n < win else (n - win) // step + 1
    if n_windows == 0:
        warnings.append("Recording duration was too short to yield any windows.")
        empty = np.array([], dtype=float)
        return WindowScores(empty, empty, empty, warnings)

    extractor = FeatureExtractor(sfreq=sf, target_channels=STANDARD_CHANNELS)
    per_chunk = max(1, int(chunk_sec // step_sec))
    pad = int(round(pad_sec * sf))

    starts, ends, probs = [], [], []
    for k0 in range(0, n_windows, per_chunk):
        k1 = min(n_windows, k0 + per_chunk)
        a = k0 * step
        b = (k1 - 1) * step + win
        pa, pb = max(0, a - pad), min(n, b + pad)
        aligned, _, _ = standardize_montage(eeg.data[:, pa:pb], eeg.channel_names, STANDARD_CHANNELS)
        filtered = preprocess_eeg_signal(aligned, sfreq=sf, lowcut=0.5, highcut=40.0, notch_freq=60.0)
        segment = filtered[:, a - pa:b - pa]
        dataset = segment_eeg_windows(
            data=segment, sfreq=sf, window_duration_sec=window_sec, window_step_sec=step_sec,
            patient_id=eeg.patient_id, file_name=eeg.file_name,
        )
        feats = extractor.transform_dataset(dataset, channel_names=final_channels)
        X = feats.reindex(columns=bundle.feature_names, fill_value=0.0)
        p = safe_predict_proba(bundle.pipeline, X)
        offset = a / sf
        for i, w in enumerate(dataset.windows):
            starts.append(round(w.start_sec + offset, 2))
            ends.append(round(w.end_sec + offset, 2))
            probs.append(round(float(p[i]), 4))
        del aligned, filtered, segment, dataset, feats, X

    return WindowScores(np.array(starts), np.array(ends), np.array(probs), warnings)


SCORE_CACHE_DIR = RESULTS_DIR / "score_cache"


def _disk_cache_path(rec_id: str, fingerprint: Optional[str] = None) -> Optional[Path]:
    # Only bundled recordings with a model loaded from a file get a reusable on-disk cache.
    fingerprint = fingerprint if fingerprint is not None else MODEL.fingerprint
    if rec_id not in BUNDLED or not fingerprint:
        return None
    return SCORE_CACHE_DIR / f"{rec_id}-{fingerprint}.npz"


def _read_disk_scores(path: Optional[Path]) -> Optional[WindowScores]:
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return WindowScores(z["starts"], z["ends"], z["probs"], json.loads(str(z["warnings"])))
    except Exception:
        return None


def _write_disk_scores(path: Optional[Path], scores: WindowScores) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.npz")
        np.savez(tmp, starts=scores.starts, ends=scores.ends, probs=scores.probs,
                 warnings=np.array(json.dumps(scores.warnings)))
        os.replace(tmp, path)  # atomic: readers never see a half-written file
    except OSError:
        pass


def get_scores(rec: LoadedRecording, bundle: Optional[TrainedModelBundle] = None) -> WindowScores:
    """
    Window scores for a recording under the active model. Looked up in memory, then in
    results/score_cache (precomputed for the bundled recordings and keyed by the model
    file's hash, so a retrained model never reuses stale scores), else computed.
    """
    model, version, fingerprint = MODEL.snapshot()  # one consistent view of the active model
    key = (rec.rec_id, version)
    cached = SCORES.get(key)
    if cached is not None:
        return cached
    disk = _disk_cache_path(rec.rec_id, fingerprint)
    scores = _read_disk_scores(disk)
    if scores is None:
        with HEAVY_LOCK:
            cached = SCORES.get(key)
            if cached is not None:
                return cached
            scores = score_recording(rec.eeg, model)
        if MODEL.version != version:
            return scores  # model changed while scoring: return, but cache nothing under either key
        _write_disk_scores(disk, scores)
    SCORES.put(key, scores)
    return scores


def precompute_bundled_scores() -> List[str]:
    """Score every bundled recording with the active model and store the results on disk."""
    bundle = MODEL.get()
    written = []
    for rec_id in BUNDLED:
        rec = STORE.get(rec_id)
        get_scores(rec)
        path = _disk_cache_path(rec_id)
        if path and path.exists():
            written.append(str(path.relative_to(BASE_DIR)))
    return written


def apply_threshold(scores: WindowScores, threshold: float, window_sec: float = DEFAULT_WINDOW_DURATION_SEC):
    """Turn cached window scores into the timeline and episodes for one threshold."""
    preds = (scores.probs >= threshold).astype(int)
    timeline = pd.DataFrame({
        "window_idx": np.arange(len(scores.probs)),
        "start_sec": scores.starts,
        "end_sec": scores.ends,
        "seizure_score": scores.probs,
        "predicted_class": preds,
    })
    timeline["status"] = np.where(preds == 1, "SEIZURE DETECTED", "Normal")

    episodes: List[PredictedEpisode] = []
    if len(preds):
        clustered = cluster_positive_windows(preds, scores.starts, scores.ends, max_gap_sec=10.0, min_duration_sec=window_sec)
        for idx, (s, e) in enumerate(clustered):
            mask = (timeline["start_sec"] >= s) & (timeline["end_sec"] <= e)
            sc = timeline.loc[mask, "seizure_score"]
            episodes.append(PredictedEpisode(
                episode_id=idx + 1, start_sec=s, end_sec=e, duration_sec=e - s,
                max_score=round(float(sc.max()) if not sc.empty else threshold, 4),
                mean_score=round(float(sc.mean()) if not sc.empty else threshold, 4),
            ))
    return timeline, episodes


# ---------------------------------------------------------------------------
# Saved experiment artifacts
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def real_metrics() -> Optional[dict]:
    return read_json(RESULTS_DIR / "metrics.json")
