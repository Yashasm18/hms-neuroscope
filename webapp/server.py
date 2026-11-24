"""
Harmful Brain Activity Classification web server (FastAPI).

Serves the single-page frontend in web/ and a JSON API on /api. Run locally with:

    python run_app.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    ANNOTATIONS_DIR,
    BASE_DIR,
    DEFAULT_SAMPLING_RATE,
    MEDICAL_DISCLAIMER,
    MODELS_DIR,
    RAW_DATA_DIR,
)
from src.data.downloader import PhysioNetDownloader
from src.preprocessing.filter import preprocess_eeg_signal
from src.training.pipeline import get_classifier_options
from src.visualization.plots import (
    plot_confusion_matrix_figure,
    plot_multichannel_eeg,
    plot_psd_spectrum,
    plot_roc_curve,
    plot_seizure_score_timeline,
)
from webapp import service as S

WEB_DIR = BASE_DIR / "web"
IMAGES_DIR = BASE_DIR / "assets" / "images"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
UPLOAD_LIMIT_BYTES = 60 * 1024 * 1024
ARTIFACT_IMAGES = {"confusion_matrix.png", "seizure_timeline.png"}
SUMMARY_PATIENTS = ["chb01", "chb02", "chb03", "chb06", "chb10"]

app = FastAPI(title="Harmful Brain Activity Classification", docs_url="/api/docs", redoc_url=None)

# Every state-changing request must carry this header. Browsers cannot add a custom
# header to a cross-site request without a CORS preflight (which this server never
# approves), so other websites cannot trigger training or uploads on localhost.
CSRF_HEADER = "x-neuroscope"


@app.middleware("http")
async def require_app_header(request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(CSRF_HEADER) != "1":
        return JSONResponse(status_code=403, content={"detail": "Missing application header."})
    return await call_next(request)


def fig_json(fig) -> dict:
    return json.loads(fig.to_json())


def get_recording(rec_id: str) -> S.LoadedRecording:
    try:
        return S.STORE.get(rec_id)
    except KeyError:
        raise HTTPException(404, f"Recording '{rec_id}' is not loaded. It may have expired; choose another.")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


def require_writable():
    if S.PUBLIC_DEMO:
        raise HTTPException(403, "Training is disabled on the public demo. Run the project locally to retrain.")


def highlight(rec: S.LoadedRecording, label: str = "Seizure") -> list:
    return [(s["start"], s["end"], f"{label} #{s['index']}") for s in S.seizure_intervals(rec.summary)]


# ---------------------------------------------------------------------------
# Status and overview
# ---------------------------------------------------------------------------

_WARM = {"ready": False, "error": None}


def _warm_up():
    """Load the model, default recording and its cached scores before serving traffic."""
    try:
        S.MODEL.get()
        rec = S.STORE.get(S.DEFAULT_RECORDING)
        if S.MODEL.get() is not None:
            S.get_scores(rec)
        api_overview(S.DEFAULT_RECORDING)
        psd_figure(rec, rec.eeg.channel_names[0], True)
    except Exception as exc:  # keep serving even if warm-up fails
        _WARM["error"] = str(exc)
    _WARM["ready"] = True


@app.on_event("startup")
def start_warm_up():
    import threading
    threading.Thread(target=_warm_up, daemon=True).start()


@app.get("/api/health")
def api_health():
    """Render health check: only reports healthy once warm-up is done, so new deploys
    receive traffic only after the data is loaded (zero-downtime switch-over)."""
    if not _WARM["ready"]:
        return JSONResponse(status_code=503, content={"status": "warming up"})
    return {"status": "ok", "warm_up_error": _WARM["error"]}


@app.get("/api/config")
def api_config():
    return {
        "public_demo": S.PUBLIC_DEMO,
        "default_recording": S.DEFAULT_RECORDING,
        "disclaimer": MEDICAL_DISCLAIMER,
        "upload_limit_mb": UPLOAD_LIMIT_BYTES // (1024 * 1024),
        "upload_max_seconds": S.UPLOAD_MAX_SECONDS,
    }


@app.get("/api/recording/{rec_id}")
def api_recording(rec_id: str):
    return get_recording(rec_id).info()


@app.get("/api/model")
def api_model():
    return S.model_info(S.MODEL.get())


_OVERVIEW_CACHE: "dict[tuple, dict]" = {}


@app.get("/api/overview")
def api_overview(rec: str = S.DEFAULT_RECORDING):
    """Dashboard payload; cached per (recording, model version) because it only changes with those."""
    key = (rec, S.MODEL.version)
    if key not in _OVERVIEW_CACHE:
        payload = _build_overview(rec)
        if len(_OVERVIEW_CACHE) > 16:
            _OVERVIEW_CACHE.clear()
        _OVERVIEW_CACHE[key] = payload
    return _OVERVIEW_CACHE[key]


def _build_overview(rec: str):
    metrics = S.real_metrics()
    bundle = S.MODEL.get()
    cards = None
    if metrics and "test_metrics" in metrics:
        tm = metrics["test_metrics"]
        cm = tm["confusion_matrix"]
        part = metrics.get("partitions", {}).get("test", {})
        cards = {
            "accuracy": tm["accuracy"], "sensitivity": tm["sensitivity"], "specificity": tm["specificity"],
            "test_patient": part.get("patient", "chb02"), "test_recording": part.get("recording", "chb02_16.edf"),
            "tp": cm["tp"], "fn": cm["fn"], "tn": cm["tn"], "fp": cm["fp"],
        }
    elif bundle and bundle.test_evaluation:
        wm = bundle.test_evaluation.window_metrics
        cards = {"accuracy": wm.accuracy, "sensitivity": wm.sensitivity, "specificity": wm.specificity,
                 "test_patient": ", ".join(bundle.test_patients), "test_recording": "active bundle",
                 "tp": wm.tp, "fn": wm.fn, "tn": wm.tn, "fp": wm.fp}

    summaries = list(ANNOTATIONS_DIR.glob("*-summary.txt"))
    _, free_gb, total_gb = PhysioNetDownloader.check_disk_space(RAW_DATA_DIR)

    recording = get_recording(rec)
    eeg = recording.eeg
    start = 2990.0 if (recording.kind == "real" and eeg.duration_sec >= 3000) else 0.0
    span = min(15.0, eeg.duration_sec)
    preview = plot_multichannel_eeg(
        data=eeg.data, channel_names=eeg.channel_names, sfreq=eeg.sampling_rate,
        t_start_sec=start, duration_sec=span, highlight_intervals=highlight(recording),
        title=f"Waveform preview [{start:.0f}–{start + span:.0f} s] · {eeg.file_name}",
    )
    return {
        "records_indexed": 141 if (ANNOTATIONS_DIR / "RECORDS-WITH-SEIZURES").exists() else len(summaries),
        "cards": cards,
        "model": S.model_info(bundle),
        "recording": recording.info(),
        "system": {
            "free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1),
            "annotation_summaries": len(summaries), "sampling_rate": DEFAULT_SAMPLING_RATE,
        },
        "preview": fig_json(preview),
    }


# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------

@app.get("/api/data")
def api_data():
    bundled = []
    for r in S.BUNDLED.values():
        p = r.edf_path
        bundled.append({
            "id": r.rec_id, "file_name": r.file_name, "patient": r.patient, "role": r.role, "note": r.note,
            "present": p.exists(), "size_mb": round(p.stat().st_size / 1048576, 1) if p.exists() else None,
            "has_seizure_file": (RAW_DATA_DIR / r.patient / f"{r.file_name}.seizures").exists(),
            "has_summary": (ANNOTATIONS_DIR / f"{r.patient}-summary.txt").exists(),
        })
    records_file = ANNOTATIONS_DIR / "RECORDS-WITH-SEIZURES"
    records = []
    if records_file.exists():
        records = [l.strip() for l in records_file.read_text(encoding="utf-8", errors="replace").splitlines()
                   if l.strip() and not l.strip().startswith("<")]
    return {
        "bundled": bundled,
        "records": records,
        "demos": [{"id": k, "label": v[5]} for k, v in S.DEMOS.items() if k != "demo"],
        "summary_patients": [
            {"id": p, "cached": (ANNOTATIONS_DIR / f"{p}-summary.txt").exists()} for p in SUMMARY_PATIENTS
        ],
    }


@app.post("/api/summary/{patient}")
def api_fetch_summary(patient: str):
    require_writable()
    if patient not in SUMMARY_PATIENTS:
        raise HTTPException(400, "Unknown patient")
    path = PhysioNetDownloader().download_summary(patient)
    if not path:
        raise HTTPException(502, "Could not fetch the summary from PhysioNet. Check the network connection.")
    return {"file": path.name}


@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.edf").name
    if not name.lower().endswith(".edf"):
        raise HTTPException(400, "Please upload an .edf file.")
    tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    tmp_path = Path(tmp.name)
    try:
        written = 0
        with tmp:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > UPLOAD_LIMIT_BYTES:
                    raise HTTPException(413, f"File is larger than {UPLOAD_LIMIT_BYTES // 1048576} MB.")
                tmp.write(chunk)
        try:
            loaded = S.STORE.add_upload(tmp_path, name)
        except Exception as exc:
            raise HTTPException(422, f"Failed to parse EDF file: {exc}")
        info = loaded.info()
        info["size_mb"] = round(written / 1048576, 2)
        return info
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Signal explorer
# ---------------------------------------------------------------------------

@app.get("/api/explore")
def api_explore(
    rec: str,
    start: float = 0.0,
    span: float = Query(10.0, ge=1.0, le=60.0),
    channels: str = "",
    filtered: bool = True,
    psd_channel: str = "",
):
    recording = get_recording(rec)
    eeg = recording.eeg
    names = [c for c in channels.split(",") if c] or eeg.channel_names[:10]
    unknown = [c for c in names if c not in eeg.channel_names]
    if unknown:
        raise HTTPException(400, f"Unknown channels: {unknown}")
    idx = [eeg.channel_names.index(c) for c in names]
    sf = eeg.sampling_rate
    start = float(min(max(0.0, start), max(0.0, eeg.duration_sec - span)))

    # Filter only the visible window plus padding, so edges match a full-recording filter.
    pad = 10.0
    a = max(0, int((start - pad) * sf))
    b = min(eeg.data.shape[1], int((start + span + pad) * sf))
    segment = eeg.data[idx, a:b]
    if filtered:
        segment = preprocess_eeg_signal(segment, sfreq=sf, lowcut=0.5, highcut=40.0, notch_freq=60.0)

    fig = plot_multichannel_eeg(
        data=segment, channel_names=names, sfreq=sf, t_start_sec=start, duration_sec=span,
        highlight_intervals=highlight(recording, "Verified seizure"),
        title=f"{eeg.file_name} · {start:.1f}–{start + span:.1f} s",
        data_offset_sec=a / sf,
    )

    psd_name = psd_channel if psd_channel in names else names[0]
    return {"start": start, "eeg": fig_json(fig), "psd": psd_figure(recording, psd_name, filtered), "psd_channel": psd_name}


_PSD_CACHE: "dict[tuple, dict]" = {}


def psd_figure(recording: S.LoadedRecording, channel: str, filtered: bool) -> dict:
    """Welch PSD over the whole channel; cached because the slider does not change it."""
    key = (recording.rec_id, channel, filtered, recording.eeg.data.shape[1])
    if key not in _PSD_CACHE:
        eeg = recording.eeg
        sig = eeg.data[eeg.channel_names.index(channel)]
        if filtered:
            sig = preprocess_eeg_signal(sig[np.newaxis, :], sfreq=eeg.sampling_rate, lowcut=0.5, highcut=40.0, notch_freq=60.0)[0]
        if len(_PSD_CACHE) > 128:
            _PSD_CACHE.clear()
        _PSD_CACHE[key] = fig_json(plot_psd_spectrum(sig=sig, sfreq=eeg.sampling_rate, max_freq=45.0,
                                                     title=f"Power spectral density · {channel}"))
    return _PSD_CACHE[key]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    rec: str
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


@app.post("/api/detect")
def api_detect(req: DetectRequest):
    bundle = S.MODEL.get()
    if bundle is None:
        raise HTTPException(409, "No trained model is available. Train one on the Model Training page.")
    recording = get_recording(req.rec)
    threshold = float(req.threshold if req.threshold is not None else bundle.optimal_threshold)
    scores = S.get_scores(recording)
    timeline, episodes = S.apply_threshold(scores, threshold)
    refs = [(s["start"], s["end"]) for s in S.seizure_intervals(recording.summary)]

    fig = None
    if len(timeline):
        fig = fig_json(plot_seizure_score_timeline(
            timeline_df=timeline, threshold=threshold, reference_intervals=refs,
            title=f"Seizure risk profile · {recording.eeg.file_name}",
        ))
    return {
        "recording": recording.info(),
        "threshold": threshold,
        "calibrated_threshold": float(bundle.optimal_threshold),
        "windows": int(len(timeline)),
        "num_episodes": len(episodes),
        "total_seizure_sec": float(sum(e.duration_sec for e in episodes)),
        "episodes": [
            {
                "id": e.episode_id, "start": e.start_sec, "end": e.end_sec, "duration": e.duration_sec,
                "max_score": e.max_score, "mean_score": e.mean_score,
                "matches_truth": any(e.start_sec < r1 and e.end_sec > r0 for r0, r1 in refs) if refs else None,
            }
            for e in episodes
        ],
        "warnings": scores.warnings,
        "timeline": fig,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def report_payload(report) -> Optional[dict]:
    if report is None:
        return None
    wm = report.window_metrics
    roc = None
    if report.roc_fpr is not None and report.roc_tpr is not None:
        roc = fig_json(plot_roc_curve(fpr=report.roc_fpr, tpr=report.roc_tpr, auc_score=wm.roc_auc,
                                      title=f"ROC curve · {report.split_name}"))
    return {
        "split": report.split_name,
        "accuracy": wm.accuracy, "sensitivity": wm.sensitivity, "specificity": wm.specificity,
        "f1_score": wm.f1_score, "roc_auc": wm.roc_auc, "warnings": list(wm.warnings),
        "confusion": fig_json(plot_confusion_matrix_figure(tn=wm.tn, fp=wm.fp, fn=wm.fn, tp=wm.tp,
                                                           title=f"Confusion matrix · {report.split_name}")),
        "roc": roc,
    }


@app.get("/api/evaluation")
def api_evaluation():
    metrics = S.real_metrics()
    real = None
    if metrics:
        splits = {}
        for key, label in (("test_metrics", "Test"), ("validation_metrics", "Validation"), ("train_metrics", "Train")):
            m = metrics.get(key)
            if not m:
                continue
            cm = m["confusion_matrix"]
            splits[label] = {
                **{k: m.get(k) for k in ("accuracy", "sensitivity", "specificity", "precision", "f1_score", "roc_auc")},
                "confusion": fig_json(plot_confusion_matrix_figure(
                    tn=cm["tn"], fp=cm["fp"], fn=cm["fn"], tp=cm["tp"],
                    title=f"{label} partition · threshold {metrics.get('decision_threshold', 0.5):.2f}")),
            }
        report_path = S.RESULTS_DIR / "classification_report.txt"
        real = {
            "partitions": metrics.get("partitions", {}),
            "threshold": metrics.get("decision_threshold"),
            "sampling_rate": metrics.get("sampling_rate_hz"),
            "event_metrics": metrics.get("event_metrics", {}),
            "splits": splits,
            "audit": S.read_json(S.RESULTS_DIR / "audit_chb01_03.json"),
            "report_text": report_path.read_text(encoding="utf-8") if report_path.exists() else None,
            "images": sorted(n for n in ARTIFACT_IMAGES if (S.RESULTS_DIR / n).exists()),
        }
    bundle = S.MODEL.get()
    active = None
    if bundle:
        active = {
            "model": S.model_info(bundle),
            "Validation": report_payload(bundle.val_evaluation),
            "Test": report_payload(bundle.test_evaluation or bundle.val_evaluation),
            "Dummy baseline": report_payload(bundle.dummy_baseline_evaluation),
        }
    return {"real": real, "active": active}


@app.get("/artifacts/{name}")
def artifact(name: str):
    if name not in ARTIFACT_IMAGES or not (S.RESULTS_DIR / name).exists():
        raise HTTPException(404)
    return FileResponse(S.RESULTS_DIR / name)


# ---------------------------------------------------------------------------
# Training (local only)
# ---------------------------------------------------------------------------

@app.get("/api/classifiers")
def api_classifiers():
    return get_classifier_options()


@app.post("/api/train/real")
def api_train_real():
    require_writable()
    from scripts.run_real_experiment import run_experiment

    with S.HEAVY_LOCK:
        bundle = run_experiment("results")
    S.MODEL.set(bundle, "results/trained_model.joblib", S.RESULTS_DIR / "trained_model.joblib")
    return {"model": S.model_info(bundle), "metrics": S.real_metrics()}


class SyntheticTrainRequest(BaseModel):
    model_type: str = "random_forest"
    window_sec: float = Field(5.0, ge=1.0, le=20.0)
    step_sec: float = Field(5.0, ge=1.0, le=20.0)
    buffer_sec: float = Field(5.0, ge=0.0, le=15.0)
    test_ratio: float = Field(0.33, ge=0.2, le=0.5)
    val_ratio: float = Field(0.33, ge=0.2, le=0.5)
    seed: int = 42


@app.post("/api/train/synthetic")
def api_train_synthetic(req: SyntheticTrainRequest):
    require_writable()
    if req.model_type not in get_classifier_options():
        raise HTTPException(400, "Unknown classifier")
    from src.data.synthetic import generate_synthetic_eeg_record
    from src.features.extractor import FeatureExtractor
    from src.preprocessing.windowing import segment_eeg_windows
    from src.training.train import train_and_evaluate_models

    patients = [
        ("chb01_demo", "chb01_03_demo.edf", 90.0, [(30.0, 55.0)], 101),
        ("chb01_demo", "chb01_04_demo.edf", 60.0, [(15.0, 35.0)], 102),
        ("chb02_demo", "chb02_16_demo.edf", 90.0, [(25.0, 50.0)], 201),
        ("chb02_demo", "chb02_19_demo.edf", 60.0, [], 202),
        ("chb03_demo", "chb03_01_demo.edf", 90.0, [(40.0, 65.0)], 301),
        ("chb03_demo", "chb03_02_demo.edf", 60.0, [(20.0, 42.0)], 302),
    ]
    with S.HEAVY_LOCK:
        frames = []
        for pid, fname, dur, seizures, seed in patients:
            eeg, summary = generate_synthetic_eeg_record(
                patient_id=pid, file_name=fname, duration_sec=dur, seizure_intervals=seizures, random_seed=seed)
            ds = segment_eeg_windows(
                data=eeg.data, sfreq=eeg.sampling_rate, window_duration_sec=req.window_sec,
                window_step_sec=req.step_sec, annotation_summary=summary, boundary_buffer_sec=req.buffer_sec,
                patient_id=pid, file_name=fname)
            frames.append(FeatureExtractor(sfreq=eeg.sampling_rate, target_channels=eeg.channel_names)
                          .transform_dataset(ds, channel_names=eeg.channel_names))
        features = pd.concat(frames, ignore_index=True)
        bundle = train_and_evaluate_models(
            features_df=features, model_type=req.model_type, random_state=req.seed,
            test_ratio=req.test_ratio, val_ratio=req.val_ratio, save_bundle=True)
    S.MODEL.set(bundle, f"models/ ({bundle.model_name})")
    return {"model": S.model_info(bundle), "windows": int(len(features)), "saved_to": str(MODELS_DIR.name)}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

@app.get("/vendor/plotly.min.js")
def plotly_js():
    return FileResponse(PLOTLY_JS, media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=604800"})


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):  # pragma: no cover - safety net
    return JSONResponse(status_code=500, content={"detail": f"Server error: {exc}"})


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
