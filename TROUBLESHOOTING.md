# Troubleshooting Guide & Common Error Solutions
## HMS Harmful Brain Activity Classifier & EEG Seizure Detection Prototype

This guide provides diagnostic steps and fixes for common technical issues encountered when setting up, running, or presenting the project.

---

### Table of Contents
1. [Streamlit Port Conflict (Port 8501 already in use)](#1-streamlit-port-conflict)
2. [Incorrect Working Directory](#2-incorrect-working-directory)
3. [Missing Python Packages or ImportError](#3-missing-python-packages)
4. [Missing EDF Recordings or Annotation Files](#4-missing-edf-recordings)
5. [Missing Trained Model Artifacts](#5-missing-trained-model)
6. [Feature Dimension Mismatch](#6-feature-dimension-mismatch)
7. [High Memory Usage or Processing Freezes](#7-memory-and-processing-delays)
8. [Automated Unit Tests Failing](#8-automated-unit-tests-failing)
9. [UI Metrics Differ from `results/metrics.json`](#9-ui-metrics-discrepancy)
10. [Network / PhysioNet Download Throttling](#10-physionet-download-throttling)

---

### 1. Streamlit Port Conflict

#### Symptom:
Terminal displays:
```text
OSError: [Errno 10048] error while attempting to bind on address ('localhost', 8501): only one usage of each socket address is normally permitted
```
or Streamlit automatically jumps to `http://localhost:8502`.

#### Cause:
Another Streamlit instance is already running in the background.

#### Fix:
**Option A: Run on a different port:**
```powershell
streamlit run app.py --server.port=8502
```

**Option B: Terminate existing Streamlit processes (Windows PowerShell):**
```powershell
Get-Process -Name python | Where-Object { $_.MainWindowTitle -like "*streamlit*" } | Stop-Process
```
Or kill all Python tasks if safe:
```powershell
taskkill /F /IM python.exe
```
Then relaunch:
```powershell
streamlit run app.py
```

---

### 2. Incorrect Working Directory

#### Symptom:
```text
FileNotFoundError: [Errno 2] No such file or directory: 'config.py'
ModuleNotFoundError: No module named 'src'
```

#### Cause:
Terminal was opened in `C:\Users\...` or a subdirectory instead of the project root.

#### Fix:
Verify your current directory and navigate to the project workspace:
```powershell
cd d:\akshay_project
pwd
```
Ensure you see `app.py`, `config.py`, and `src/`.

---

### 3. Missing Python Packages

#### Symptom:
```text
ModuleNotFoundError: No module named 'mne'
ModuleNotFoundError: No module named 'plotly'
```

#### Fix:
Run dependency installation against [`requirements.txt`](requirements.txt):
```powershell
pip install -r requirements.txt
```

Verify that all required packages load without error:
```powershell
python -c "import mne, streamlit, sklearn, scipy, plotly, joblib, matplotlib, requests, pytest; print('All Core Packages OK')"
```

---

### 4. Missing EDF Recordings or Annotation Files

#### Symptom:
- Dashboard sidebar displays `chb01_03.edf: ❌ Missing`.
- Terminal shows: `FileNotFoundError: data/raw/chb01/chb01_03.edf`.

#### Fix:
Confirm that local data directories are intact:
```powershell
Test-Path data/raw/chb01/chb01_03.edf
Test-Path data/raw/chb01/chb01_04.edf
Test-Path data/raw/chb02/chb02_16.edf
Test-Path data/annotations/chb01-summary.txt
```
If any file is missing, do **not** download the full 43 GB dataset. The script downloads only the required files with range requests:
```powershell
python -c "from src.data.downloader import PhysioNetDownloader; dl = PhysioNetDownloader(); dl.download_summary('chb01'); dl.download_summary('chb02')"
```

---

### 5. Missing Trained Model

#### Symptom:
Prediction tab displays:
```text
⚠️ No trained model loaded. Please navigate to '🧠 Model Training' to train a model first.
```

#### Fix:
Execute the real experiment script to generate the serialized pipeline:
```powershell
python scripts/run_real_experiment.py
```
This generates [`results/trained_model.joblib`](results/trained_model.joblib) in ~18 seconds.

---

### 6. Feature Dimension Mismatch

#### Symptom:
```text
ValueError: X has 520 features, but RandomForestClassifier is expecting 546 features as input.
```

#### Cause:
A non-standard channel montage was loaded, or spatial feature computation was skipped.

#### Fix:
The pipeline requires 18 standard 10-20 bipolar channels:
- $18\text{ channels} \times 30\text{ features} = 540\text{ channel features}$.
- $+ 6\text{ spatial aggregates}$ (global mean, std, and max across time and frequency).
- Total = **$546$ features**.

Ensure the EEG data is passed through `src.preprocessing.montage.standardize_bipolar_channels()` prior to feature extraction.

---

### 7. Memory and Processing Delays

#### Symptom:
Streamlit UI shows `"Running..."` for a prolonged period, or memory spikes above 4 GB.

#### Cause:
Attempting to process or plot an entire 1-hour raw EEG recording at once ($23\text{ channels} \times 256\text{ Hz} \times 3600\text{ s} \approx 21.2\text{ million datapoints}$).

#### Fix:
1. In the **Signal Explorer**, keep the *Window Span* slider between **5.0s and 30.0s**.
2. When loading custom EDF files via the upload tab, duration is automatically clamped to 300 seconds to preserve UI responsiveness.
3. If memory is constrained, close other open browser tabs or restart Streamlit.

---

### 8. Automated Unit Tests Failing

#### Symptom:
`pytest -v` reports failures.

#### Fix:
Run pytest with detailed tracebacks to isolate the failing component:
```powershell
pytest -v -tb=short
```
Check specific modules individually:
```powershell
# Test mathematical audit of timestamps
pytest -v tests/test_audit.py

# Test signal preprocessing & 5s windowing
pytest -v tests/test_preprocessing.py

# Test real dataset integrity
pytest -v tests/test_real_experiment.py
```
All 27 tests in `tests/` are verified to pass. If an error occurs, ensure that `results/metrics.json` and `results/audit_chb01_03.json` are present.

---

### 9. UI Metrics Discrepancy

#### Symptom:
Metrics shown on the Dashboard differ from [`results/metrics.json`](results/metrics.json).

#### Cause:
The Streamlit session is displaying a cached or synthetic demo model instead of the real experiment bundle.

#### Fix:
1. In the left sidebar, click the **`🏥 Load Real`** button.
2. In the top-right of the Dashboard page, click **`🚀 Run Real CHB-MIT Experiment`** (or refresh the page with `F5`).
3. Verify that the badge in the sidebar reads **`🟢 REAL CHB-MIT DATA`**.

---

### 10. PhysioNet Download Throttling

#### Symptom:
Terminal hangs when attempting to download EDF files from PhysioNet, or downloads stall at ~3 KB/s.

#### Cause:
The public PhysioNet web server throttles single persistent HTTP connections.

#### Fix:
Do **not** use `curl` or single-stream `wget` for entire recordings. The included [`src/data/downloader.py`](src/data/downloader.py) uses HTTP byte-range requests and chunked verification to assemble EDF recordings reliably without socket drops.
