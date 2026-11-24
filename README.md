# HMS — Harmful Brain Activity Classification & EEG Seizure Detection Prototype

An end-to-end continuous EEG seizure detection research prototype developed using the **CHB-MIT Scalp EEG Database** from PhysioNet. Includes standardized EDF ingestion, zero-phase filtering, 546-dimensional time-frequency feature extraction, balanced Random Forest classification, threshold tuning, and a premium dark web dashboard (HMS NeuroScope) built with FastAPI, a hand-crafted HTML/CSS/JS frontend and interactive Plotly charts, with photo banners on every page.

> **CRITICAL MEDICAL DISCLAIMER**  
> **RESEARCH PROTOTYPE ONLY:** This software is developed strictly for academic research and educational demonstration on the CHB-MIT EEG database. It is **NOT** a medical device, is **NOT** clinically validated or FDA-cleared, and must **NEVER** be used for clinical diagnosis, patient monitoring, triage, or treatment decisions. Predicted scores reflect academic model risk outputs and are not calibrated clinical probabilities.

---

## ⚡ 60-Second Quick Start (Beginner Friendly)

If you are running this project on Windows:

**Before you start:** install **Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/windows/) and tick **"Add python.exe to PATH"**. Then **extract the zip** (right-click → *Extract All…*) — don't run the files from inside the zip.

### Option 1: One-Click Launcher (Easiest)
1. Double-click **`START_PROJECT.bat`**. The first run installs packages and takes 3–5 minutes; later runs start in seconds.
2. The launcher will automatically detect Python, create a local `.venv`, install all dependencies, verify model files, and launch the web dashboard.
3. Your browser will open automatically to **`http://localhost:8501`**.

---

### Option 2: Manual Terminal Commands (3 Steps)
Open **PowerShell** or Command Prompt in this folder:

```powershell
# 1. (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install verified dependencies
pip install -r requirements.txt

# 3. Launch the dashboard (opens your browser automatically)
python run_app.py
```
If it does not open by itself, go to **`http://localhost:8501`** in Google Chrome or Microsoft Edge.

### Troubleshooting on Windows
- **"Python was not found"** → reinstall Python with *Add python.exe to PATH* ticked, or use the `py` launcher.
- **PowerShell blocks `Activate.ps1`** → run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or just use `START_PROJECT.bat`.
- **Port 8501 busy** → `.venv\Scripts\python run_app.py --port 8502`.
- **Website fonts** load from Google Fonts; offline, the app falls back to system fonts. Photos are bundled in `assets/images` and work offline.

---

## 🗂️ Included Documentation Files

| Guide | Description |
|---|---|
| [`USER_MANUAL.md`](USER_MANUAL.md) | Complete tab-by-tab dashboard instructions, custom EDF upload guide, and metric definitions. |
| [`DATA_SETUP.md`](DATA_SETUP.md) | Details on the pre-packaged sample EEG recordings, PhysioNet origin, and directory layout. |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Solutions for port conflicts, missing files, memory limits, and test issues. |
| [`PROFESSOR_DEMO_CHECKLIST.md`](PROFESSOR_DEMO_CHECKLIST.md) | Concise 5-minute presentation script and viva checklist. |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Detailed directory tree and description of every Python module. |
| [`HANDOVER_CHECKLIST.md`](HANDOVER_CHECKLIST.md) | Step-by-step verification checklist for new recipients. |

---

## 🔬 Project Overview & Key Engineering Highlights

1. **EDF Ingestion & Channel Standardization:**
   - Standardized European Data Format (`.edf`) loading using MNE-Python.
   - Alignment against the international 10-20 combinatorial bipolar montage (18 channels: `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`, `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`, `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`, `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2`, `FZ-CZ`, `CZ-PZ`).
2. **Digital Signal Preprocessing:**
   - Zero-phase 4th order Butterworth bandpass filter ($0.5 - 40.0\text{ Hz}$).
   - Zero-phase IIR notch filter at $60\text{ Hz}$ ($Q=30$) to suppress powerline electrical interference.
3. **Clinician Annotation-Based Windowing:**
   - 5.0-second non-overlapping window segmentation.
   - Ground truth labels derived from clinician annotations (`*-summary.txt` and `.seizures`).
   - Boundary buffer zones (5.0s pre- and post-ictal) explicitly flagged as `-1` (ambiguous) and excluded from training to prevent label contamination.
4. **Biomedical Time-Frequency Feature Extraction (546 Features):**
   - *Time-Domain:* Variance, skewness, kurtosis, peak-to-peak amplitude, RMS energy, line length, zero-crossing rate.
   - *Hjorth Parameters:* Activity, mobility, complexity.
   - *Frequency-Domain (Welch PSD):* Delta ($0.5-4\text{ Hz}$), Theta ($4-8\text{ Hz}$), Alpha ($8-13\text{ Hz}$), Beta ($13-30\text{ Hz}$), Gamma ($30-40\text{ Hz}$), total power, spectral edge frequency (SEF95), and Shannon spectral entropy.
   - *Spatial Cross-Channel Aggregates:* Global mean, standard deviation, and maximum.
5. **Multi-Recording Patient-Level Holdout Protocol:**
   - **Training:** `chb01_03.edf` (Patient `chb01`, 607 clean windows)
   - **Validation:** `chb01_04.edf` (Patient `chb01`, 300 clean windows for threshold calibration)
   - **Test:** `chb02_16.edf` (Unseen Patient `chb02`, 38 clean windows for unbiased cross-patient testing)
   - **Performance:** **94.74% Accuracy**, **84.62% Seizure Sensitivity**, **100.00% Specificity**, **100.00% Event Sensitivity**, **0.00 False Alarms/hr**.

---

## 🛠️ Automated Scripts & One-Click Tools

| Script | Command | Purpose |
|---|---|---|
| **One-Click App Launch** | `START_PROJECT.bat` | Sets up venv, installs deps, and starts the web app |
| **Run All Unit Tests** | `RUN_TESTS.bat` or `pytest -v` | Executes 34 automated tests (incl. web API) in ~5s |
| **Re-Run Experiment** | `RUN_EXPERIMENT.bat` | Re-fits model, tunes threshold, updates `results/` |
| **Test Any Custom EDF** | `python scripts/predict_recording.py --edf <path>` | CLI explicit inference on any new recording |

---

## 🔬 Dataset & Scientific Attribution

The dataset utilized is the **CHB-MIT Scalp EEG Database (v1.0.0)**, collected at Children's Hospital Boston and hosted by PhysioNet:
- **PhysioNet Record:** [https://physionet.org/content/chbmit/1.0.0/](https://physionet.org/content/chbmit/1.0.0/)
- **License:** Open Data Commons Public Domain Dedication and License v1.0 (ODC-PDDL).
- **Official Citation:**
  > Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.

---

## 🛡️ Limitations & Research Boundaries

1. **Cohort Size:** Evaluated on a single held-out test patient (`chb02`) and one test recording (`chb02_16.edf`).
2. **Sample Positives:** The test recording contains one clinical seizure event (13 seizure windows, 25 non-seizure windows).
3. **Generalization Requirement:** More patients and recordings across diverse age groups and seizure types are required before clinical translation.
4. **Homomorphic Encryption:** Privacy-preserving encrypted inference is an unverified / unimplemented future research concept; it is **not** part of this codebase.
