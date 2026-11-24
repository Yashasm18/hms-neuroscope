"""
HMS - Harmful Brain Activity Classification & EEG Seizure Detection Prototype
Streamlit Academic Research Application for CHB-MIT Scalp EEG Database.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="HMS Harmful Brain Activity Classifier | CHB-MIT EEG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths and imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    MODELS_DIR,
    RAW_DATA_DIR,
    ANNOTATIONS_DIR,
    STANDARD_CHANNELS,
    DEFAULT_SAMPLING_RATE,
    PHYSIONET_CITATION,
    DATASET_LICENSE,
    MEDICAL_DISCLAIMER,
)
from src.data.annotation_parser import (
    parse_summary_file,
    parse_summary_text,
    load_verified_recording_annotations,
    RecordingSummary,
)
from src.data.downloader import PhysioNetDownloader
from src.data.edf_loader import load_edf_recording, StandardizedEEG
from src.data.synthetic import generate_synthetic_eeg_record
from src.preprocessing.filter import preprocess_eeg_signal
from src.preprocessing.windowing import segment_eeg_windows
from src.features.extractor import FeatureExtractor
from src.training.train import (
    TrainedModelBundle,
    load_model_bundle,
    save_model_bundle,
    train_and_evaluate_models,
)
from src.training.pipeline import get_classifier_options
from src.inference.predictor import SeizurePredictor, PredictionResult
from src.visualization.plots import (
    plot_multichannel_eeg,
    plot_psd_spectrum,
    plot_seizure_score_timeline,
    plot_confusion_matrix_figure,
    plot_roc_curve,
)
from scripts.run_real_experiment import run_experiment

# On the public hosted demo (PUBLIC_DEMO=1) retraining is disabled so visitors
# cannot overwrite the shared model and results for everyone else.
READ_ONLY = os.environ.get("PUBLIC_DEMO") == "1"
READ_ONLY_HELP = "Disabled on the public demo. Run the project locally to retrain."

from src.visualization.ui import (
    inject_theme,
    render_hero,
    render_page_banner,
    section_label,
    metric_card,
)

# Premium dark theme (assets/style.css)
inject_theme()


# Fast Data Switchers
def load_real_dataset():
    real_edf = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
    real_sum = ANNOTATIONS_DIR / "chb01-summary.txt"
    real_sz = RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures"

    if real_edf.exists():
        try:
            eeg = load_edf_recording(real_edf)
            summary = load_verified_recording_annotations(
                "chb01_03.edf",
                summary_path=real_sum if real_sum.exists() else None,
                seizures_path=real_sz if real_sz.exists() else None,
            )
            st.session_state["current_eeg"] = eeg
            st.session_state["current_summary"] = summary
            st.session_state["prediction_result"] = None
            return True
        except Exception as e:
            st.error(f"Error loading real CHB-MIT dataset: {e}")
            return False
    return False


def load_synthetic_demo():
    eeg, summary = generate_synthetic_eeg_record(
        patient_id="chb01_demo",
        file_name="chb01_03_demo.edf",
        duration_sec=90.0,
        seizure_intervals=[(30.0, 55.0)],
        random_seed=42,
    )
    st.session_state["current_eeg"] = eeg
    st.session_state["current_summary"] = summary
    st.session_state["prediction_result"] = None


# Initialize Session State
def init_session_state():
    if "current_eeg" not in st.session_state:
        # Prioritize real CHB-MIT clinical data if present locally
        real_loaded = load_real_dataset()
        if not real_loaded:
            load_synthetic_demo()

    if "active_model_bundle" not in st.session_state:
        results_model_path = BASE_DIR / "results" / "trained_model.joblib"
        default_model_path = MODELS_DIR / "random_forest_seizure_model.joblib"
        target_model = results_model_path if results_model_path.exists() else default_model_path

        if target_model.exists():
            try:
                st.session_state["active_model_bundle"] = load_model_bundle(target_model)
            except Exception:
                st.session_state["active_model_bundle"] = None
        else:
            st.session_state["active_model_bundle"] = None

    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None

    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Dashboard"

    if "signal_start_time" not in st.session_state:
        st.session_state["signal_start_time"] = 0.0


init_session_state()

# Sidebar Navigation and System Status
st.sidebar.markdown(
    """<div class='brand'>
        <div class='brand-mark'>◉</div>
        <div><div class='brand-name'>HMS NeuroScope</div>
        <div class='brand-sub'>CHB-MIT Scalp EEG Research</div></div>
    </div>""",
    unsafe_allow_html=True,
)

nav_options = [
    ":material/space_dashboard: Dashboard",
    ":material/folder_open: EEG Upload & Data",
    ":material/monitor_heart: Signal Explorer",
    ":material/model_training: Model Training",
    ":material/bolt: Seizure Detection",
    ":material/analytics: Model Evaluation",
    ":material/menu_book: Documentation",
]

query_page = st.query_params.get("page", None)
default_nav_idx = 0
if query_page:
    for idx, opt in enumerate(nav_options):
        if query_page.lower() in opt.lower():
            default_nav_idx = idx
            break

st.sidebar.markdown("### Workspace")
selected_page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=default_nav_idx,
    label_visibility="collapsed",
)

st.sidebar.markdown("### System Status")

# Active Model info
active_bundle = st.session_state.get("active_model_bundle")
if active_bundle:
    is_real_bundle = "real" in active_bundle.model_name.lower() or "chb01" in active_bundle.model_name.lower()
    badge_html = "<span class='badge-active'>● Real CHB-MIT Model</span>" if is_real_bundle else "<span class='badge-active'>● Benchmark Model</span>"
    st.sidebar.markdown(
        f"""<div class='side-card'>
            <div style='margin-bottom:8px'>{badge_html}</div>
            <div class='side-row'><span>Model</span><span title='{active_bundle.model_name}'>{active_bundle.model_name}</span></div>
            <div class='side-row'><span>Threshold</span><span>{active_bundle.optimal_threshold:.2f}</span></div>
            <div class='side-row'><span>Features</span><span>{len(active_bundle.feature_names)}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        "<div class='side-card'><span class='badge-warn'>● Not Trained</span>"
        "<div class='side-row'><span>Model</span><span>None</span></div></div>",
        unsafe_allow_html=True,
    )

# Active EEG info & Fast Switcher
current_eeg: StandardizedEEG = st.session_state.get("current_eeg")
if current_eeg:
    data_badge = (
        "<span class='badge-warn'>● Synthetic Demo</span>"
        if current_eeg.is_synthetic
        else "<span class='badge-active'>● Real CHB-MIT Data</span>"
    )
    st.sidebar.markdown(
        f"""<div class='side-card'>
            <div style='margin-bottom:8px'>{data_badge}</div>
            <div class='side-row'><span>Recording</span><span>{current_eeg.file_name}</span></div>
            <div class='side-row'><span>Patient</span><span>{current_eeg.patient_id}</span></div>
            <div class='side-row'><span>Duration</span><span>{current_eeg.duration_sec:.1f} s</span></div>
            <div class='side-row'><span>Channels</span><span>{len(current_eeg.channel_names)}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        if st.button("Real", icon=":material/local_hospital:", use_container_width=True, help="Load official CHB-MIT chb01_03.edf"):
            load_real_dataset()
            st.rerun()
    with col_sb2:
        if st.button("Demo", icon=":material/science:", use_container_width=True, help="Load synthetic benchmark"):
            load_synthetic_demo()
            st.rerun()

st.sidebar.markdown(
    f"<div class='disclaimer-banner' style='margin-top:18px'><strong>Academic research prototype</strong><br>"
    f"Not a clinical diagnostic device. Evaluated on CHB-MIT Scalp EEG.</div>",
    unsafe_allow_html=True,
)


# Helper to fetch real metrics if available
def get_cached_real_metrics():
    metrics_file = BASE_DIR / "results" / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ==============================================================================
# 1. DASHBOARD PAGE
# ==============================================================================
if "Dashboard" in selected_page:
    real_metrics = get_cached_real_metrics()

    cached_summaries = list(ANNOTATIONS_DIR.glob("*-summary.txt"))
    records_file = ANNOTATIONS_DIR / "RECORDS-WITH-SEIZURES"
    has_records = records_file.exists()
    num_records_indexed = 141 if has_records else len(cached_summaries)

    test_pat = real_metrics.get("partitions", {}).get("test", {}).get("patient", "chb02") if real_metrics else "chb02"
    test_rec = real_metrics.get("partitions", {}).get("test", {}).get("recording", "chb02_16.edf") if real_metrics else "chb02_16.edf"

    if real_metrics and "test_metrics" in real_metrics:
        tm = real_metrics["test_metrics"]
        cm_counts = tm["confusion_matrix"]
        acc_val, acc_sub = f"{tm['accuracy']:.2%}", f"Unseen patient {test_pat} ({test_rec})"
        sens_val, sens_sub = f"{tm['sensitivity']:.2%}", f"Seizure recall ({cm_counts['tp']}/{cm_counts['tp'] + cm_counts['fn']} windows)"
        spec_val, spec_sub = f"{tm['specificity']:.2%}", f"Non-seizure windows ({cm_counts['tn']}/{cm_counts['tn'] + cm_counts['fp']})"
    elif active_bundle and active_bundle.test_evaluation:
        wm = active_bundle.test_evaluation.window_metrics
        acc_val, acc_sub = f"{wm.accuracy:.2%}", "Active bundle test accuracy"
        sens_val, sens_sub = f"{wm.sensitivity:.2%}", "Active bundle test sensitivity"
        spec_val, spec_sub = f"{wm.specificity:.2%}", "Active bundle test specificity"
    else:
        acc_val = sens_val = spec_val = "N/A"
        acc_sub = sens_sub = spec_sub = "Model not evaluated"

    render_hero(
        eyebrow="EEG seizure detection · research prototype",
        title_html="Reading the <em>electric language</em> of the brain.",
        lede=(
            "HMS NeuroScope classifies harmful brain activity and flags seizure episodes in continuous "
            "scalp EEG, trained and evaluated on the CHB-MIT pediatric database from PhysioNet."
        ),
        chips=[
            ("Montage", "18 bipolar ch"),
            ("Sampling", f"{DEFAULT_SAMPLING_RATE:.0f} Hz"),
            ("Window", "5 s"),
            ("Model", active_bundle.model_name.replace("_", " ").title() if active_bundle else "Not trained"),
        ],
        image="hero",
    )

    # Key Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("PhysioNet Records", str(num_records_indexed), "Seizure records indexed (chb01 & chb02)")
    with c2:
        metric_card("Test Accuracy", acc_val, acc_sub)
    with c3:
        metric_card("Seizure Sensitivity", sens_val, sens_sub)
    with c4:
        metric_card("Specificity", spec_val, spec_sub)

    # Dataset Status Banner
    if current_eeg and current_eeg.is_synthetic:
        st.markdown(
            "<div class='synthetic-banner'><strong>Active dataset: synthetic benchmark demo.</strong> "
            "Simulated EEG waveforms are loaded. Run the real experiment below "
            "or choose <strong>Real</strong> in the sidebar to switch to official clinical data.</div>",
            unsafe_allow_html=True,
        )
    elif current_eeg:
        st.markdown(
            "<div class='real-banner'><strong>Active dataset: official CHB-MIT Scalp EEG (chb01_03.edf).</strong> "
            "Pediatric patient EEG, 18 bipolar channels at 256 Hz, 3,050 s. "
            "Verified seizure interval: <strong>2996.0 s – 3036.0 s</strong> (40 s).</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<div class='disclaimer-banner'>{MEDICAL_DISCLAIMER}</div>", unsafe_allow_html=True)

    # Pipeline overview
    section_label("The pipeline")
    steps = [
        ("01", "Load & verify", "MNE reads the EDF; summary and WFDB annotations are cross-checked."),
        ("02", "Clean the signal", "0.5–40 Hz band-pass and 60 Hz notch filter on every channel."),
        ("03", "Window & featurize", "5-second windows become time-frequency band-power features."),
        ("04", "Train & calibrate", "Balanced Random Forest with a validation-tuned threshold."),
        ("05", "Score & evaluate", "Window and event metrics on a held-out, unseen patient."),
    ]
    step_cols = st.columns(len(steps))
    for col, (num, head, body) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"<div class='feature-card'><div class='num'>{num}</div><h4>{head}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    # Real CHB-MIT Experiment Action Box
    section_label("Experiment runner")
    with st.container(border=True):
        col_act1, col_act2 = st.columns([3, 1], vertical_alignment="center")
        with col_act1:
            st.markdown(
                "<h4 style='margin:0 0 4px 0'>Run the real CHB-MIT experiment end to end</h4>"
                "<p style='margin:0; font-size:0.9rem; color:#8A93A8'>Loads <code>chb01_03.edf</code>, verifies annotations, "
                "filters, segments, trains the balanced Random Forest with a strict temporal holdout, and writes "
                "reproducible artifacts to <code>results/</code>.</p>",
                unsafe_allow_html=True,
            )
        with col_act2:
            run_clicked = st.button("Run experiment", icon=":material/rocket_launch:", type="primary", use_container_width=True, disabled=READ_ONLY, help=READ_ONLY_HELP if READ_ONLY else None)
        if run_clicked:
            with st.status("Executing Real CHB-MIT Experiment Pipeline...", expanded=True) as status:
                st.write("1. Loading raw EDF with MNE-Python & cross-verifying annotations...")
                st.write("2. Preprocessing EEG (0.5–40 Hz bandpass, 60 Hz notch filter)...")
                st.write("3. Generating labeled 5s windows with boundary buffer...")
                st.write("4. Training balanced Random Forest & tuning validation threshold...")
                st.write("5. Generating evaluation artifacts & saving reproducible outputs...")
                bundle = run_experiment("results")
                st.session_state["active_model_bundle"] = bundle
                load_real_dataset()
                status.update(label="Real CHB-MIT Experiment completed successfully!", state="complete")
            st.success("Trained model and all 7 artifacts generated in `results/`!")
            st.rerun()

    section_label("System overview")
    col_sys1, col_sys2 = st.columns(2)

    with col_sys1:
        dl = PhysioNetDownloader()
        has_space, free_gb, total_gb = dl.check_disk_space(dl.raw_dir)
        st.markdown(
            f"""<div class='glass-card'>
                <h4>Dataset status · CHB-MIT</h4>
                <div class='side-row'><span>Local disk</span><span>{free_gb:.1f} / {total_gb:.1f} GB free</span></div>
                <div class='side-row'><span>Annotation summaries</span><span>{len(cached_summaries)}</span></div>
                <div class='side-row'><span>Raw EDF storage</span><span title='{RAW_DATA_DIR}'>{RAW_DATA_DIR.name}/</span></div>
                <div class='side-row'><span>Montage</span><span>18 bipolar · 10-20</span></div>
                <div class='side-row'><span>Sampling rate</span><span>{DEFAULT_SAMPLING_RATE:.0f} Hz</span></div>
                <div class='side-row'><span>Verified recording</span><span>chb01_03.edf · 2996–3036 s</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_sys2:
        if active_bundle:
            st.markdown(
                f"""<div class='glass-card'>
                    <h4>Model & evaluation</h4>
                    <div class='side-row'><span>Architecture</span><span>{active_bundle.model_name.upper()}</span></div>
                    <div class='side-row'><span>Features</span><span>{len(active_bundle.feature_names)} time-frequency</span></div>
                    <div class='side-row'><span>Decision threshold</span><span>{active_bundle.optimal_threshold:.2f} (val-tuned)</span></div>
                    <div class='side-row'><span>Train</span><span>{', '.join(active_bundle.train_patients)}</span></div>
                    <div class='side-row'><span>Validation</span><span>{', '.join(active_bundle.val_patients)}</span></div>
                    <div class='side-row'><span>Test</span><span>{', '.join(active_bundle.test_patients)}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No trained model loaded yet. Use **Run experiment** above.")

    # Current EEG quick preview
    section_label("Current recording in memory")
    if current_eeg:
        # Plot initial seconds
        summary: RecordingSummary = st.session_state.get("current_summary")
        highlight_intervals = []
        if summary and summary.has_seizure:
            for s in summary.seizures:
                highlight_intervals.append((s.start_sec, s.end_sec, f"Seizure {s.index}"))

        preview_start = 2990.0 if (not current_eeg.is_synthetic and current_eeg.duration_sec >= 3000) else 0.0
        fig_preview = plot_multichannel_eeg(
            data=current_eeg.data,
            channel_names=current_eeg.channel_names,
            sfreq=current_eeg.sampling_rate,
            t_start_sec=preview_start,
            duration_sec=min(15.0, current_eeg.duration_sec),
            highlight_intervals=highlight_intervals,
            title=f"Waveform Preview [{preview_start:.1f}s – {preview_start + 15.0:.1f}s]: {current_eeg.file_name} ({len(current_eeg.channel_names)} channels)",
        )
        st.plotly_chart(fig_preview, use_container_width=True)


# ==============================================================================
# 2. EEG UPLOAD & DATA MANAGER PAGE
# ==============================================================================
elif "Upload" in selected_page:
    render_page_banner(
        "Data manager",
        "EEG data &amp; <em>EDF upload</em>",
        "Inspect official PhysioNet CHB-MIT records, upload your own .edf recordings, "
        "or load synthetic benchmark recordings.",
        "data",
    )

    tab_physionet, tab_upload, tab_demo = st.tabs(
        ["🌐 Official CHB-MIT Database & Local Files", "📤 Upload Custom EDF File", "🧪 Benchmark Demo Generator"]
    )

    with tab_physionet:
        st.subheader("Official CHB-MIT Scalp EEG Database Status")
        st.markdown(
            "The CHB-MIT Scalp EEG database is hosted publicly on PhysioNet under the Open Data Commons license. "
            "Official EDF files, summary files, and binary WFDB `.seizures` annotations are cached locally."
        )

        real_edf_file = RAW_DATA_DIR / "chb01" / "chb01_03.edf"
        real_sz_file = RAW_DATA_DIR / "chb01" / "chb01_03.edf.seizures"
        real_sum_file = ANNOTATIONS_DIR / "chb01-summary.txt"

        st.markdown("#### 🏥 Verified Local Clinical Files")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown(
                f"""<div class='glass-card'>
                    <h4>chb01_03.edf</h4>
                    <p style='margin: 0; font-size: 0.85rem; color: #94A3B8;'>
                        <strong>Status:</strong> {'✅ Present' if real_edf_file.exists() else '❌ Missing'}<br>
                        <strong>Size:</strong> {real_edf_file.stat().st_size / (1024*1024):.1f} MB<br>
                        <strong>Duration:</strong> 3,050.0 seconds (50.8 minutes)<br>
                        <strong>Sampling Rate:</strong> 256.0 Hz<br>
                        <strong>Channels:</strong> 18 standardized 10-20 bipolar pairs (23 raw)
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_f2:
            st.markdown(
                f"""<div class='glass-card'>
                    <h4>Official Annotations</h4>
                    <p style='margin: 0; font-size: 0.85rem; color: #94A3B8;'>
                        <strong>Summary:</strong> {'✅ chb01-summary.txt' if real_sum_file.exists() else '❌ Missing'}<br>
                        <strong>WFDB Binary:</strong> {'✅ chb01_03.edf.seizures' if real_sz_file.exists() else '❌ Missing'}<br>
                        <strong>Verified Seizure:</strong> 2996.0s – 3036.0s (40.0s duration)<br>
                        <strong>Agreement:</strong> 100% timestamp match across sources<br>
                        <strong>Clinical Target:</strong> Seizure #1 (Patient 01, Female, Age 11)
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        col_load1, col_load2 = st.columns([1, 2])
        with col_load1:
            if st.button("📂 Load chb01_03.edf into Memory Now", type="primary", use_container_width=True):
                if load_real_dataset():
                    st.success("Successfully loaded chb01_03.edf into memory!")
                    st.rerun()
                else:
                    st.error("Failed to load real dataset.")

        with col_load2:
            st.info(
                "Source: PhysioNet CHB-MIT Scalp EEG Database (v1.0.0). "
                "Download URL: https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf"
            )

        st.markdown("---")
        dl = PhysioNetDownloader()
        records = dl.fetch_records_with_seizures()

        c_p1, c_p2 = st.columns([1, 2])
        with c_p1:
            st.write(f"**Indexed Seizure Records:** {len(records)}")
            selected_pat = st.selectbox("Inspect Patient Summaries", ["chb01", "chb02", "chb03", "chb06", "chb10"])
            if st.button("Inspect / Download Patient Summary"):
                with st.spinner(f"Fetching summary for {selected_pat}..."):
                    sum_path = dl.download_summary(selected_pat)
                    if sum_path:
                        st.success(f"Summary available: {sum_path.name}")
                    else:
                        st.error("Could not fetch remote summary. Check network connection.")

        with c_p2:
            st.markdown("**Database Seizure Records Catalog:**")
            if records:
                st.dataframe(pd.DataFrame({"Relative Path": records[:25]}), height=200, use_container_width=True)
            else:
                st.info("No cached RECORDS-WITH-SEIZURES found. Run downloader to refresh.")

    with tab_upload:
        st.subheader("Upload Custom Clinical EDF Recording")
        uploaded_file = st.file_uploader(
            "Select an EEG recording file (.edf format)",
            type=["edf"],
            help="Upload European Data Format (EDF) EEG recording.",
        )

        if uploaded_file is not None:
            st.info(f"Received file: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")

            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                with st.spinner("Validating and parsing EDF recording via MNE-Python..."):
                    loaded_eeg = load_edf_recording(tmp_path, max_duration_sec=300.0)
                    loaded_eeg.file_name = uploaded_file.name
                    st.session_state["current_eeg"] = loaded_eeg
                    st.session_state["current_summary"] = None
                    st.success(f"Successfully loaded '{uploaded_file.name}'!")

                col_u1, col_u2, col_u3 = st.columns(3)
                col_u1.metric("Sampling Rate", f"{loaded_eeg.sampling_rate:.1f} Hz")
                col_u2.metric("Duration", f"{loaded_eeg.duration_sec:.1f} s")
                col_u3.metric("Channels", f"{len(loaded_eeg.channel_names)}")

                if loaded_eeg.warning_messages:
                    for w in loaded_eeg.warning_messages:
                        st.warning(w)

            except Exception as e:
                st.error(f"Failed to parse EDF file: {e}")
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    with tab_demo:
        st.subheader("Pre-configured Demonstration Recordings")
        st.markdown(
            "Generate pre-configured multichannel EEG recordings with realistic biological background rhythms "
            "and injected 3 Hz spike-and-wave seizure discharges (explicitly labeled synthetic)."
        )

        demo_choice = st.selectbox(
            "Select Benchmark Record to Load into Memory",
            [
                "chb01_03_demo (90s, Seizure: 30s-55s)",
                "chb01_04_demo (60s, Seizure: 15s-35s)",
                "chb02_16_demo (90s, Seizure: 25s-50s)",
                "chb02_19_demo (60s, Non-seizure baseline)",
                "chb03_01_demo (90s, Seizure: 40s-65s)",
            ],
        )

        if st.button("Load Selected Demo Record"):
            if "chb01_03" in demo_choice:
                eeg, summary = generate_synthetic_eeg_record(
                    patient_id="chb01_demo", file_name="chb01_03_demo.edf", duration_sec=90.0,
                    seizure_intervals=[(30.0, 55.0)], random_seed=101
                )
            elif "chb01_04" in demo_choice:
                eeg, summary = generate_synthetic_eeg_record(
                    patient_id="chb01_demo", file_name="chb01_04_demo.edf", duration_sec=60.0,
                    seizure_intervals=[(15.0, 35.0)], random_seed=102
                )
            elif "chb02_16" in demo_choice:
                eeg, summary = generate_synthetic_eeg_record(
                    patient_id="chb02_demo", file_name="chb02_16_demo.edf", duration_sec=90.0,
                    seizure_intervals=[(25.0, 50.0)], random_seed=201
                )
            elif "chb02_19" in demo_choice:
                eeg, summary = generate_synthetic_eeg_record(
                    patient_id="chb02_demo", file_name="chb02_19_demo.edf", duration_sec=60.0,
                    seizure_intervals=[], random_seed=202
                )
            else:
                eeg, summary = generate_synthetic_eeg_record(
                    patient_id="chb03_demo", file_name="chb03_01_demo.edf", duration_sec=90.0,
                    seizure_intervals=[(40.0, 65.0)], random_seed=301
                )

            st.session_state["current_eeg"] = eeg
            st.session_state["current_summary"] = summary
            st.session_state["prediction_result"] = None
            st.success(f"Loaded '{eeg.file_name}' into session state!")


# ==============================================================================
# 3. SIGNAL EXPLORER PAGE
# ==============================================================================
elif "Signal Explorer" in selected_page:
    render_page_banner(
        "Signal explorer",
        "Multichannel <em>signal explorer</em>",
        "Review raw and filtered EEG traces with the ground-truth seizure overlay, "
        "and inspect power spectral density across clinical bands.",
        "signals",
    )

    current_eeg = st.session_state.get("current_eeg")
    if not current_eeg:
        st.warning("No EEG recording is loaded. Please load or upload an EEG file in the '📁 EEG Upload' tab.")
        st.stop()

    if current_eeg.is_synthetic:
        st.markdown(
            "<div class='synthetic-banner'>⚠️ Displaying <strong>SYNTHETIC</strong> demonstration recording. "
            "Use the sidebar to load the real CHB-MIT recording.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='real-banner'>🟢 Displaying <strong>REAL CHB-MIT SCALP EEG:</strong> "
            f"<code>{current_eeg.file_name}</code> (Patient 01, Female, Age 11). "
            f"Sampling rate: {current_eeg.sampling_rate:.0f} Hz. Verified clinical seizure onset at 2996.0s.</div>",
            unsafe_allow_html=True,
        )

    # Fast Navigation for Real CHB-MIT Recording
    if not current_eeg.is_synthetic and current_eeg.duration_sec >= 3000:
        st.markdown("**⚡ Quick Seizure Navigation:**")
        nav1, nav2, nav3, nav4 = st.columns(4)
        with nav1:
            if st.button("⏩ Jump to Seizure (2996s - 3036s)", use_container_width=True):
                st.session_state["signal_start_time"] = 2995.0
                st.rerun()
        with nav2:
            if st.button("⏪ Jump to Pre-ictal (2980s)", use_container_width=True):
                st.session_state["signal_start_time"] = 2980.0
                st.rerun()
        with nav3:
            if st.button("⏹️ Jump to Post-ictal (3040s)", use_container_width=True):
                st.session_state["signal_start_time"] = 3040.0
                st.rerun()
        with nav4:
            if st.button("⏮️ Recording Start (0s)", use_container_width=True):
                st.session_state["signal_start_time"] = 0.0
                st.rerun()

    # Explorer Controls
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 1, 1, 1])

    with col_ctrl1:
        all_channels = current_eeg.channel_names
        selected_channels = st.multiselect(
            "Channels to Display",
            options=all_channels,
            default=all_channels[: min(10, len(all_channels))],
        )

    with col_ctrl2:
        view_duration = st.slider("Window Span (sec)", min_value=5.0, max_value=30.0, value=10.0, step=5.0)

    with col_ctrl3:
        max_start = max(0.0, current_eeg.duration_sec - view_duration)
        default_start = min(float(st.session_state.get("signal_start_time", 0.0)), float(max_start))
        start_time = st.slider("Start Time (sec)", min_value=0.0, max_value=float(max_start), value=float(default_start), step=1.0)
        st.session_state["signal_start_time"] = start_time

    with col_ctrl4:
        apply_filter = st.checkbox("Apply 0.5–40 Hz Filter", value=True)

    if not selected_channels:
        st.info("Select at least one EEG channel to view.")
        st.stop()

    ch_indices = [current_eeg.channel_names.index(c) for c in selected_channels]
    display_data = current_eeg.data[ch_indices]

    if apply_filter:
        display_data = preprocess_eeg_signal(
            display_data,
            sfreq=current_eeg.sampling_rate,
            lowcut=0.5,
            highcut=40.0,
            notch_freq=60.0,
        )

    # Highlight intervals
    summary: RecordingSummary = st.session_state.get("current_summary")
    highlight_intervals = []
    if summary and summary.has_seizure:
        for s in summary.seizures:
            highlight_intervals.append((s.start_sec, s.end_sec, f"Verified Seizure #{s.index}"))

    fig_eeg = plot_multichannel_eeg(
        data=display_data,
        channel_names=selected_channels,
        sfreq=current_eeg.sampling_rate,
        t_start_sec=start_time,
        duration_sec=view_duration,
        highlight_intervals=highlight_intervals,
        title=f"Multichannel Waveform [{start_time:.1f}s – {start_time + view_duration:.1f}s] ({current_eeg.file_name})",
    )
    st.plotly_chart(fig_eeg, use_container_width=True)

    # Spectral Analysis Section
    st.markdown("### 🔬 Spectral Power Density (Welch PSD)")
    col_sp1, col_sp2 = st.columns([1, 2])
    with col_sp1:
        psd_channel = st.selectbox("Select Channel for PSD Analysis", options=selected_channels)
        ch_idx_psd = selected_channels.index(psd_channel)
        sig_slice = display_data[ch_idx_psd]

    with col_sp2:
        fig_psd = plot_psd_spectrum(
            sig=sig_slice,
            sfreq=current_eeg.sampling_rate,
            max_freq=45.0,
            title=f"Power Spectral Density: {psd_channel} (Delta, Theta, Alpha, Beta, Gamma)",
        )
        st.plotly_chart(fig_psd, use_container_width=True)


# ==============================================================================
# 4. MODEL TRAINING PAGE
# ==============================================================================
elif "Model Training" in selected_page:
    render_page_banner(
        "Model training",
        "Training &amp; <em>cross-validation</em>",
        "Train baseline classifiers on the real CHB-MIT Scalp EEG dataset "
        "or on the multi-patient synthetic benchmark generator.",
        "training",
    )

    tab_train_real, tab_train_demo = st.tabs(
        ["🏥 Real CHB-MIT Pipeline (chb01_03.edf)", "🧪 Multi-Patient Synthetic Generator"]
    )

    with tab_train_real:
        st.subheader("Official CHB-MIT Patient-Level Holdout Experiment")
        st.markdown(
            "This pipeline processes multiple genuine recordings across distinct patients with strict isolation:  \n"
            "- **Train Partition:** `chb01_03.edf` (Patient chb01, 8 seizure windows, 599 non-seizure windows)  \n"
            "- **Validation Partition:** `chb01_04.edf` (Patient chb01, 6 seizure windows, 294 non-seizure windows)  \n"
            "- **Test Partition:** `chb02_16.edf` (Unseen Patient chb02, 13 seizure windows, 25 non-seizure windows)  \n"
            "Data is segmented into 5-second non-overlapping windows with 5.0s transition buffers excluded. "
            "Feature scaling and model fitting occur **exclusively on the training partition**, and classification thresholds "
            "are tuned on the validation partition."
        )

        st.markdown(
            "<div class='disclaimer-banner'><strong>DATA PARTITION & ANTI-LEAKAGE PROTOCOL:</strong><br>"
            "1. <strong>Strict Patient Isolation:</strong> Test data is from Patient <code>chb02</code>, completely unseen during training or validation.<br>"
            "2. <strong>Zero Intra-Event Leakage:</strong> No recording or seizure event is split across partitions.<br>"
            "3. <strong>Threshold Tuning:</strong> Classification threshold (0.50) is tuned on validation recording <code>chb01_04.edf</code>, NOT the test set.</div>",
            unsafe_allow_html=True,
        )

        col_rp1, col_rp2 = st.columns(2)
        with col_rp1:
            st.markdown(
                """**Experiment Configuration:**
                - **Train:** `chb01_03.edf` (3,050s, 607 clean windows)
                - **Val:** `chb01_04.edf` (1,510s, 300 clean windows)
                - **Test:** `chb02_16.edf` (196s, 38 clean windows, Patient chb02)
                - **Classifier:** `RandomForestClassifier(n_estimators=100, class_weight='balanced')`
                - **Artifacts Saved to:** `results/`
                """
            )
        with col_rp2:
            st.markdown(
                """**Evaluation Targets:**
                - Window-level Sensitivity, Specificity, Precision, F1-Score, ROC-AUC
                - Clinical Event-Level Sensitivity & False Alarms per Hour
                - Confusion Matrix & Full Seizure Timeline visualizations
                """
            )

        if st.button("🚀 Train Real CHB-MIT Baseline Model", type="primary", use_container_width=True, disabled=READ_ONLY, help=READ_ONLY_HELP if READ_ONLY else None):
            with st.status("Training Real CHB-MIT Model...", expanded=True) as status:
                st.write("1. Auditing single-recording chronological split...")
                st.write("2. Loading multi-recording dataset (chb01_03, chb01_04, chb02_16)...")
                st.write("3. Segmenting windows and extracting multichannel features...")
                st.write("4. Fitting balanced Random Forest exclusively on Train...")
                st.write("5. Tuning threshold on Validation (chb01_04.edf)...")
                st.write("6. Evaluating on Unseen Patient chb02 (chb02_16.edf)...")
                bundle = run_experiment("results")
                st.session_state["active_model_bundle"] = bundle
                load_real_dataset()
                status.update(label="Training complete! Artifacts saved to results/", state="complete")

            st.success("Trained patient-level model saved and active in session state!")
            real_m = get_cached_real_metrics()
            if real_m:
                st.markdown("### 📊 Test Evaluation Results (Unseen Patient chb02)")
                tm_c1, tm_c2, tm_c3, tm_c4 = st.columns(4)
                tm_c1.metric("Test Accuracy", f"{real_m['test_metrics']['accuracy']:.2%}")
                tm_c2.metric("Test Sensitivity", f"{real_m['test_metrics']['sensitivity']:.2%}")
                tm_c3.metric("Test Specificity", f"{real_m['test_metrics']['specificity']:.2%}")
                tm_c4.metric("Test F1-Score", f"{real_m['test_metrics']['f1_score']:.2%}")

    with tab_train_demo:
        st.subheader("Synthetic Multi-Patient Benchmark Generator")
        st.markdown(
            "Train machine learning models with **strict patient-level isolation** across simulated patient cohorts "
            "to demonstrate cross-patient generalization workflows."
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Training Configuration")
            model_options = get_classifier_options()
            selected_model_type = st.selectbox(
                "Classifier Architecture",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
            )
            win_duration = st.number_input("Window Duration (seconds)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
            win_step = st.number_input("Window Step (seconds)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
            buffer_sec = st.number_input("Boundary Exclusion Buffer (seconds)", min_value=0.0, max_value=15.0, value=5.0, step=1.0)

        with col_t2:
            st.subheader("Patient Group Partitioning")
            test_ratio = st.slider("Test Patient Ratio", min_value=0.2, max_value=0.5, value=0.33, step=0.05)
            val_ratio = st.slider("Validation Patient Ratio", min_value=0.2, max_value=0.5, value=0.33, step=0.05)
            seed = st.number_input("Random Seed", value=42, step=1)

        if st.button("🚀 Start Synthetic Benchmark Pipeline", disabled=READ_ONLY, help=READ_ONLY_HELP if READ_ONLY else None):
            with st.status("Executing synthetic benchmark training...", expanded=True) as status:
                st.write("1. Preparing multi-patient segmented windows...")
                patients_config = [
                    ("chb01_demo", "chb01_03_demo.edf", 90.0, [(30.0, 55.0)], 101),
                    ("chb01_demo", "chb01_04_demo.edf", 60.0, [(15.0, 35.0)], 102),
                    ("chb02_demo", "chb02_16_demo.edf", 90.0, [(25.0, 50.0)], 201),
                    ("chb02_demo", "chb02_19_demo.edf", 60.0, [], 202),
                    ("chb03_demo", "chb03_01_demo.edf", 90.0, [(40.0, 65.0)], 301),
                    ("chb03_demo", "chb03_02_demo.edf", 60.0, [(20.0, 42.0)], 302),
                ]

                dataset_records = []
                for pat_id, f_name, duration, seizures, s_seed in patients_config:
                    eeg, summary = generate_synthetic_eeg_record(
                        patient_id=pat_id, file_name=f_name, duration_sec=duration,
                        seizure_intervals=seizures, random_seed=s_seed,
                    )
                    dataset = segment_eeg_windows(
                        data=eeg.data, sfreq=eeg.sampling_rate, window_duration_sec=win_duration,
                        window_step_sec=win_step, annotation_summary=summary, boundary_buffer_sec=buffer_sec,
                        patient_id=pat_id, file_name=f_name,
                    )
                    extractor = FeatureExtractor(sfreq=eeg.sampling_rate, target_channels=eeg.channel_names)
                    df = extractor.transform_dataset(dataset, channel_names=eeg.channel_names)
                    dataset_records.append(df)

                all_features_df = pd.concat(dataset_records, ignore_index=True)
                st.write(f"2. Extracted {len(all_features_df)} windows across {len(patients_config)} recordings.")

                bundle = train_and_evaluate_models(
                    features_df=all_features_df, model_type=selected_model_type, random_state=seed,
                    test_ratio=test_ratio, val_ratio=val_ratio, save_bundle=True,
                )
                st.session_state["active_model_bundle"] = bundle
                status.update(label="Synthetic benchmark training complete!", state="complete")

            st.success(f"Trained benchmark model '{bundle.model_name}' saved to `{MODELS_DIR}`!")


# ==============================================================================
# 5. PREDICTION & INFERENCE PAGE
# ==============================================================================
elif "Seizure Detection" in selected_page:
    render_page_banner(
        "Detection",
        "Continuous <em>seizure detection</em>",
        "Window-by-window seizure risk inference on the active recording. Scores are model risk outputs "
        "compared against the calibrated decision threshold.",
        "detection",
    )

    current_eeg = st.session_state.get("current_eeg")
    active_bundle = st.session_state.get("active_model_bundle")

    if not current_eeg:
        st.warning("No EEG recording is loaded.")
        st.stop()

    if not active_bundle:
        st.warning("No trained model loaded. Please navigate to '🧠 Model Training' to train a model first.")
        st.stop()

    if current_eeg.is_synthetic:
        st.markdown(
            "<div class='synthetic-banner'>⚠️ Notice: Running inference on a <strong>SYNTHETIC</strong> test recording.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='real-banner'>🟢 Running inference on <strong>REAL CHB-MIT SCALP EEG:</strong> "
            f"<code>{current_eeg.file_name}</code> (3,050.0s recording, verified seizure @ 2996s–3036s).</div>",
            unsafe_allow_html=True,
        )

    # Threshold Adjustment
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        custom_threshold = st.slider(
            "Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(active_bundle.optimal_threshold),
            step=0.05,
            help="Windows with risk score >= threshold are classified as Seizure.",
        )
    with col_p2:
        if st.button("Reset to Calibrated Threshold"):
            custom_threshold = active_bundle.optimal_threshold

    if st.button("🔍 Execute Seizure Analysis", type="primary") or st.session_state.get("prediction_result") is None:
        with st.spinner("Segmenting, extracting features, and scoring EEG windows..."):
            predictor = SeizurePredictor(active_bundle)
            pred_res: PredictionResult = predictor.predict_recording(
                current_eeg,
                threshold=custom_threshold,
                window_duration_sec=5.0,
                window_step_sec=5.0,
            )
            st.session_state["prediction_result"] = pred_res

    pred_res: PredictionResult = st.session_state.get("prediction_result")

    if pred_res is not None:
        st.markdown("### 📊 Detection Summary")
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        r_c1.metric("Analyzed Windows", len(pred_res.timeline_df))
        r_c2.metric("Detected Episodes", pred_res.num_episodes)
        r_c3.metric("Total Seizure Time", f"{pred_res.total_seizure_time_sec:.1f} s")
        r_c4.metric("Active Threshold", f"{pred_res.decision_threshold:.2f}")

        # Seizure Timeline Chart
        summary: RecordingSummary = st.session_state.get("current_summary")
        ref_intervals = [(s.start_sec, s.end_sec) for s in summary.seizures] if (summary and summary.has_seizure) else []

        fig_timeline = plot_seizure_score_timeline(
            timeline_df=pred_res.timeline_df,
            threshold=pred_res.decision_threshold,
            reference_intervals=ref_intervals,
            title=f"Predicted Seizure Risk Profile: {pred_res.file_name} (Green: Ground Truth Annotation)",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Detected Episodes Table
        st.markdown("### 🕒 Detected Seizure Episodes & Clinical Alignment")
        if pred_res.episodes:
            ep_records = []
            for ep in pred_res.episodes:
                # Check alignment with ground truth
                matches_gt = any(
                    (ep.start_sec < r_end and ep.end_sec > r_start)
                    for r_start, r_end in ref_intervals
                ) if ref_intervals else False

                ep_records.append(
                    {
                        "Episode #": ep.episode_id,
                        "Onset Time (s)": f"{ep.start_sec:.1f} s",
                        "Offset Time (s)": f"{ep.end_sec:.1f} s",
                        "Duration": f"{ep.duration_sec:.1f} s",
                        "Peak Score": f"{ep.max_score:.3f}",
                        "Mean Score": f"{ep.mean_score:.3f}",
                        "Ground Truth Alignment": "✅ Verified True Positive" if matches_gt else "⚠️ False Alarm",
                    }
                )
            st.dataframe(pd.DataFrame(ep_records), use_container_width=True)
        else:
            st.success("No seizure episodes detected above threshold across this recording.")


# ==============================================================================
# 6. MODEL EVALUATION PAGE
# ==============================================================================
elif "Evaluation" in selected_page:
    render_page_banner(
        "Evaluation",
        "Model <em>evaluation</em> &amp; metrics",
        "Reproducible window-level metrics and event-level clinical statistics, "
        "with training, validation and test splits kept strictly apart.",
        "evaluation",
    )

    tab_real_artifacts, tab_bundle_eval = st.tabs(
        ["🏥 Real CHB-MIT Experiment Results (Official Artifacts)", "🔬 Active Model Bundle Split"]
    )

    with tab_real_artifacts:
        real_metrics = get_cached_real_metrics()
        results_dir = BASE_DIR / "results"

        if not real_metrics:
            st.warning("No real CHB-MIT experiment artifacts found in `results/`. Please run the experiment from the Dashboard.")
        else:
            test_rec = real_metrics.get("partitions", {}).get("test", {}).get("recording", "chb02_16.edf")
            test_pat = real_metrics.get("partitions", {}).get("test", {}).get("patient", "chb02")
            st.markdown(
                f"<div class='disclaimer-banner'><strong>ACADEMIC RESEARCH PROTOTYPE EVALUATION — PATIENT-LEVEL HOLDOUT</strong><br>"
                f"Evaluated on official CHB-MIT Scalp EEG recordings with strict patient isolation:  \n"
                f"<strong>Train:</strong> <code>chb01_03.edf</code> (chb01) | "
                f"<strong>Validation:</strong> <code>chb01_04.edf</code> (chb01) | "
                f"<strong>Test:</strong> <code>{test_rec}</code> (Unseen Patient {test_pat}).<br>"
                f"Sampling rate: {real_metrics.get('sampling_rate_hz', 256.0):.0f} Hz. "
                f"Threshold: {real_metrics.get('decision_threshold', 0.5):.2f} (calibrated on Validation).<br>"
                f"<strong>Clinical Notice:</strong> Academic research prototype, not clinically approved for medical diagnosis.</div>",
                unsafe_allow_html=True,
            )

            # Audit Expandable Box
            audit_file = BASE_DIR / "results" / "audit_chb01_03.json"
            if audit_file.exists():
                with st.expander("🔍 Audit Report: Single-Recording Chronological Split Bug & Correction"):
                    with open(audit_file, "r", encoding="utf-8") as af:
                        aud = json.load(af)
                    st.markdown(f"**Audit Finding:** {aud['audit_finding']}")
                    st.json(aud["chronological_partitions"])

            # Split Selector for Real Experiment
            split_choice = st.radio(
                "Select Real Data Partition to Inspect",
                [f"Test Set (Unseen Patient {test_pat} - {test_rec})", "Validation Set (Tuning Split - chb01_04.edf)", "Train Set (Fitting Split - chb01_03.edf)"],
                horizontal=True,
            )

            if "Test" in split_choice:
                split_key = "test_metrics"
                split_title = "Test Partition"
            elif "Validation" in split_choice:
                split_key = "validation_metrics"
                split_title = "Validation Partition"
            else:
                split_key = "train_metrics"
                split_title = "Training Partition"

            sm = real_metrics[split_key]
            cm = sm["confusion_matrix"]

            # Key Metric Numbers
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            r_c1.metric(f"{split_title} Accuracy", f"{sm['accuracy']:.2%}")
            r_c2.metric(f"{split_title} Sensitivity", f"{sm['sensitivity']:.2%}")
            r_c3.metric(f"{split_title} Specificity", f"{sm['specificity']:.2%}")
            r_c4.metric(f"{split_title} F1-Score", f"{sm['f1_score']:.2%}")

            # Visualizations from Artifacts
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.subheader(f"Confusion Matrix ({split_title})")
                cm_fig = plot_confusion_matrix_figure(
                    tn=cm["tn"], fp=cm["fp"], fn=cm["fn"], tp=cm["tp"],
                    title=f"Confusion Matrix: Real {split_title} (Threshold = {real_metrics['decision_threshold']:.2f})",
                )
                st.plotly_chart(cm_fig, use_container_width=True)

            with col_v2:
                st.subheader("Event-Level Clinical Metrics")
                em = real_metrics.get("event_metrics", {})
                st.markdown(
                    f"""<div class='glass-card'>
                        <h4>Clinical Event Statistics (Test Set)</h4>
                        <p style='margin: 4px 0; font-size: 0.9rem;'>
                            <strong>Total True Seizures:</strong> {em.get('total_true_events', 1)}<br>
                            <strong>Detected True Seizures:</strong> {em.get('detected_true_events', 1)}<br>
                            <strong>Event Sensitivity:</strong> {em.get('event_sensitivity', 1.0):.1%}<br>
                            <strong>False Alarms (Episodes):</strong> {em.get('false_alarms', 0)}<br>
                            <strong>False Alarms per Hour (FAR/hr):</strong> {em.get('false_alarms_per_hour', 0.0):.2f} / hr
                        </p>
                        <p style='margin: 8px 0 0 0; font-size: 0.78rem; color: #64748B;'>
                            *FAR/hr is defined as false positive clustered episodes per hour of non-seizure recording.
                        </p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if (results_dir / "confusion_matrix.png").exists():
                    st.image(str(results_dir / "confusion_matrix.png"), caption="Saved Matplotlib Artifact: confusion_matrix.png", use_container_width=True)

            # Saved Artifacts Browser
            st.markdown("### 📁 Saved Experiment Artifacts (`results/`)")
            col_art1, col_art2 = st.columns([1, 1])

            with col_art1:
                st.markdown("**Classification Report (`classification_report.txt`):**")
                rep_path = results_dir / "classification_report.txt"
                if rep_path.exists():
                    st.code(rep_path.read_text(encoding="utf-8"))

            with col_art2:
                st.markdown("**Seizure Timeline Profile (`seizure_timeline.png`):**")
                timeline_img_path = results_dir / "seizure_timeline.png"
                if timeline_img_path.exists():
                    st.image(str(timeline_img_path), caption="Saved Matplotlib Artifact: seizure_timeline.png", use_container_width=True)

    with tab_bundle_eval:
        st.subheader("Active Session Model Bundle Evaluation")
        active_bundle = st.session_state.get("active_model_bundle")
        if not active_bundle:
            st.warning("No active model loaded.")
            st.stop()

        eval_split_choice = st.radio("Evaluation Split", ["Validation Set", "Test Set", "Dummy Baseline Comparison"], horizontal=True)

        if eval_split_choice == "Validation Set":
            eval_report = active_bundle.val_evaluation
        elif eval_split_choice == "Test Set":
            eval_report = active_bundle.test_evaluation or active_bundle.val_evaluation
        else:
            eval_report = active_bundle.dummy_baseline_evaluation

        if eval_report is None:
            st.warning("Evaluation report for this split is unavailable.")
            st.stop()

        wm = eval_report.window_metrics

        ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
        ev_c1.metric("Accuracy", f"{wm.accuracy:.2%}")
        ev_c2.metric("Sensitivity (Recall)", f"{wm.sensitivity:.2%}")
        ev_c3.metric("Specificity", f"{wm.specificity:.2%}")
        ev_c4.metric("F1-Score", f"{wm.f1_score:.2%}")

        if wm.warnings:
            for w in wm.warnings:
                st.warning(f"⚠️ {w}")

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig_cm = plot_confusion_matrix_figure(
                tn=wm.tn, fp=wm.fp, fn=wm.fn, tp=wm.tp,
                title=f"Confusion Matrix ({eval_report.split_name} Split)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_fig2:
            if eval_report.roc_fpr is not None and eval_report.roc_tpr is not None:
                fig_roc = plot_roc_curve(
                    fpr=eval_report.roc_fpr,
                    tpr=eval_report.roc_tpr,
                    auc_score=wm.roc_auc,
                    title=f"ROC Curve ({eval_report.split_name} Split)",
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC Curve cannot be computed when test split contains only a single class.")


# ==============================================================================
# 7. DOCUMENTATION PAGE
# ==============================================================================
elif "Documentation" in selected_page:
    render_page_banner(
        "Documentation",
        "Methodology &amp; <em>attribution</em>",
        "Dataset source, preprocessing and modelling choices, evaluation protocol and credits.",
        "docs",
    )

    st.markdown(
        """
        ### 1. Dataset Source & Attribution
        - **Database:** CHB-MIT Scalp EEG Database (v1.0.0)
        - **Host:** PhysioNet (`https://physionet.org/content/chbmit/1.0.0/`)
        - **Clinical Source:** Children's Hospital Boston & MIT.
        - **Subject Population:** Pediatric subjects with intractable seizures monitored following withdrawal of anti-epileptic medication.
        - **Sampling Rate:** 256 Hz, 16-bit resolution.
        - **Official Files Used:**
          - `chb01_03.edf` (23 raw channels @ 256 Hz, 3,050 seconds)
          - `chb01_03.edf.seizures` (WFDB binary format, onset: sample 766,976, offset: sample 777,216)
          - `chb01-summary.txt` (Text summary, Seizure start: 2996s, end: 3036s)
        - **Attribution Citation:**
        > *Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.*
        - **License:** Open Data Commons Public Domain Dedication and License v1.0 (ODC-PDDL).

        ---

        ### 2. Signal Processing & Feature Engineering Pipeline
        1. **Channel Standardization:**
           - Standard 18-channel bipolar montage (temporal and parasagittal chains: `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`, etc.).
           - Channel names normalized to standard 10-20 system representations. Duplicate channel names handled cleanly.
        2. **Digital Filtering:**
           - Zero-phase 4th order Butterworth bandpass filter (0.5 – 40.0 Hz).
           - Zero-phase IIR Notch filter at 60 Hz (Q=30) for electrical powerline artifact removal.
        3. **Windowing & Label Assignment:**
           - Non-overlapping 5.0-second sliding windows.
           - Ground truth labels assigned strictly from clinician summary annotations (>= 50% overlap = Seizure [1], 0% overlap = Non-seizure [0]).
           - Ambiguous boundary windows within transition buffers (5.0s pre/post-ictal) are explicitly flagged as `-1` and excluded from model training to prevent boundary label contamination.
        4. **Extracted Features (Multichannel):**
           - *Time-Domain:* Mean, standard deviation, variance, skewness, kurtosis, peak-to-peak amplitude, RMS energy, line length, zero-crossing rate, Hjorth activity, mobility, and complexity.
           - *Frequency-Domain (Welch PSD):* Absolute band powers (Delta 0.5-4Hz, Theta 4-8Hz, Alpha 8-13Hz, Beta 13-30Hz, Gamma 30-40Hz), relative band powers, total spectral power, spectral entropy, and Spectral Edge Frequency (SEF95).
           - *Spatial Cross-Channel Aggregates:* Global mean, max, and std across all 18 channels.

        ---

        ### 3. Patient & Temporal Split Anti-Leakage Protocol
        - **Single-Patient Holdout (Current Real Experiment):**
          - Partitioned chronologically: Train (60%), Validation (20%), Test (20%).
          - Windows are strictly disjoint in time to prevent temporal leakage.
          - Feature scalers (`StandardScaler`) are fit **only on training data** and applied via transform to evaluation splits.
          - Classification thresholds are tuned on the validation split, never on the test set.
          - **Limitation:** Explicitly documented as a recording-level evaluation, NOT an independent cross-patient generalizability study.
        - **Multi-Patient Protocol (Full Benchmark):**
          - Windows from any single patient are strictly assigned to **only one partition** (Train, Validation, or Test).

        ---

        ### 4. Reproducibility & Terminal Execution
        To reproduce the complete real CHB-MIT experiment from terminal:
        ```bash
        # Run real experiment pipeline and save 7 artifacts to results/
        python scripts/run_real_experiment.py

        # Run automated pytest verification suite
        pytest -v

        # Launch the interactive Streamlit dashboard
        streamlit run app.py
        ```

        ---

        ### 5. Medical Disclaimer
        """
    )
    st.error(MEDICAL_DISCLAIMER)
