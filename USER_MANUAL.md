# HMS Harmful Brain Activity Classifier & EEG Seizure Detection Prototype
## Comprehensive Operating Manual & User Guide

> **CRITICAL ACADEMIC RESEARCH DISCLAIMER**  
> **RESEARCH PROTOTYPE ONLY:** This software is developed strictly for academic research and educational demonstration on the CHB-MIT EEG database. It is **NOT** a medical device, is **NOT** clinically validated or FDA-cleared, and must **NEVER** be used for clinical diagnosis, patient monitoring, triage, or treatment decisions. Predicted scores reflect academic model risk outputs and are not calibrated clinical probabilities.

---

## 1. Environment Setup

### 1.1 System & Software Requirements
- **Operating System:** Windows 10/11 (PowerShell), Linux, or macOS.
- **Python Version:** Python 3.10, 3.11, or 3.12 (Tested and verified on **Python 3.12.4**).
- **RAM:** Minimum 8 GB recommended (16 GB optimal for holding multiple multi-channel EDF recordings in memory).
- **Disk Space:** ~200 MB for core code and model artifacts; ~60 MB for cached sample EDF recordings. (Do **not** download the full 43 GB PhysioNet archive).
- **Hardware Acceleration:** CPU-only execution; no dedicated GPU or CUDA setup is required.

### 1.2 Opening the Project Directory
Open **PowerShell** (or your preferred terminal) and navigate to the project directory:

```powershell
cd d:\akshay_project
```

Verify that you are in the correct root directory by running:
```powershell
dir app.py, config.py, requirements.txt
```
All three files must be listed.

### 1.3 Virtual Environment Setup (Optional but Recommended)
To prevent package conflicts with global Python packages:

```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run the activation script).*

### 1.4 Installing Dependencies
Install all verified dependencies from [`requirements.txt`](requirements.txt):

```powershell
pip install -r requirements.txt
```

#### Core Installed Libraries & Exact Verified Versions:
| Package | Verified Version | Purpose |
|---|---|---|
| `mne` | `1.13.2` | European Data Format (EDF) loading and raw signal container |
| `streamlit` | `1.52.0` | Dark scientific interactive web dashboard |
| `scikit-learn` | `1.7.2` | `RandomForestClassifier`, `StandardScaler`, metrics, pipelines |
| `scipy` | `1.16.3` | Butterworth bandpass filter, IIR notch filter, Welch PSD |
| `plotly` | `5.24.1` | Interactive dark-themed multi-channel waveform & PSD charts |
| `joblib` | `1.5.2` | Model pipeline serialization (`trained_model.joblib`) |
| `matplotlib` | `3.9.3` | Static figure generation (`confusion_matrix.png`, `seizure_timeline.png`) |
| `pandas` | `2.2.3` | Window tabular tracking, episode alignment, and metrics dataframes |
| `numpy` | `2.1.3` | High-performance multi-dimensional array operations |
| `pytest` | `9.1.1` | Automated test suite execution |
| `requests` | `2.32.3` | Resumable PhysioNet annotation and header range downloading |

---

## 2. Launching the Application

### 2.1 Launch Command
From the project root `d:\akshay_project`, run:

```powershell
streamlit run app.py
```

### 2.2 Accessing the Dashboard
Streamlit will initialize the local server and display output in the terminal:

```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open your web browser (Google Chrome, Microsoft Edge, or Mozilla Firefox) and navigate to:
**`http://localhost:8501`**

To stop the server at any time, switch to the terminal and press `Ctrl + C`.

---

## 3. Dashboard Navigation & Tab-by-Tab Operations

The application features a dark scientific UI with 7 navigation pages accessible via the sidebar radio selector.

```
+-------------------------------------------------------------------------+
| Sidebar Navigation Menu:                                                |
|  (o) 📊 Dashboard                                                       |
|  ( ) 📁 EEG Upload & Data Manager                                       |
|  ( ) 📈 Multichannel Signal Explorer                                    |
|  ( ) 🧠 Model Training & Cross-Validation                               |
|  ( ) ⚡ Seizure Detection & Prediction                                  |
|  ( ) 📋 Model Evaluation & Metrics                                      |
|  ( ) 📖 Documentation & Methodology                                     |
+-------------------------------------------------------------------------+
```

### Tab 1: 📊 Dashboard
- **Purpose:** System overview, storage health, active dataset indicator, summary metric cards, and one-click experiment execution.
- **Inputs Required:** None.
- **Key Buttons & Order:**
  1. Click **`🚀 Run Real CHB-MIT Experiment`** (top right action box) if you wish to run the full training and evaluation pipeline directly from the UI.
  2. In the sidebar, use **`🏥 Load Real`** to switch memory to `chb01_03.edf`, or **`🧪 Load Demo`** to switch to the synthetic benchmark.
- **Expected Output:**
  - 4 Metric Cards: *PhysioNet Records (141)*, *Test Accuracy (94.74%)*, *Seizure Sensitivity (84.62%)*, *Specificity (100.00%)*.
  - System Health & Infrastructure Overview (Disk space, standard montage, channel count).
  - Waveform Preview of the active recording.
- **Troubleshooting:** If metrics show `"N/A"`, click the `🚀 Run Real CHB-MIT Experiment` button or run `python scripts/run_real_experiment.py` in the terminal to generate `results/metrics.json`.

---

### Tab 2: 📁 EEG Upload & Data Manager
- **Purpose:** Manage local CHB-MIT files, inspect PhysioNet database records, upload custom `.edf` recordings, or load synthetic benchmarks.
- **Sub-Tabs:**
  1. **`🌐 Official CHB-MIT Database & Local Files`**:
     - Shows verified status for `chb01_03.edf`, `chb01-summary.txt`, and `chb01_03.edf.seizures`.
     - Button: **`📂 Load chb01_03.edf into Memory Now`** loads the 3,050s recording into RAM.
     - Dropdown: Inspect Patient Summaries (`chb01` to `chb10`).
  2. **`📤 Upload Custom EDF File`**:
     - Drag-and-drop or browse for any local `.edf` file.
     - Automatically parses sampling rate, duration, and channel names with MNE-Python.
  3. **`🧪 Benchmark Demo Generator`**:
     - Dropdown: Select from 5 simulated recordings (`chb01_03_demo`, `chb01_04_demo`, `chb02_16_demo`, `chb02_19_demo`, `chb03_01_demo`).
     - Button: **`Load Selected Demo Record`**.
- **Expected Output:** Success banner displaying sampling rate, recording duration, and standardized 18 bipolar channels.

---

### Tab 3: 📈 Multichannel Signal Explorer
- **Purpose:** Clinical visual review of raw and filtered multichannel EEG traces with ground-truth seizure overlay and spectral power density analysis.
- **Inputs Required:** Ensure an EEG recording is loaded in memory.
- **Key Controls:**
  - **Quick Seizure Navigation Buttons:**
    - `⏩ Jump to Seizure (2996s - 3036s)`: Immediately shifts the view window to the seizure onset.
    - `⏪ Jump to Pre-ictal (2980s)`: Displays normal baseline prior to the event.
    - `⏹️ Jump to Post-ictal (3040s)`: Displays recovery rhythms after the event.
    - `⏮️ Recording Start (0s)`: Returns to recording onset.
  - **Controls Bar:**
    - *Channels to Display:* Multiselect from 18 bipolar channels (e.g., `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`).
    - *Window Span (sec):* Slider from 5.0s to 30.0s (default 10.0s).
    - *Start Time (sec):* Slider across recording duration.
    - *Apply 0.5–40 Hz Filter:* Checkbox (default checked).
  - **Channel for PSD Analysis:** Selectbox to pick a single channel for Welch Power Spectral Density.
- **Expected Output:**
  - Interactive Plotly stacked waveform showing 18 channels with green shaded regions highlighting clinical seizure intervals.
  - Welch PSD chart displaying power across Delta ($0.5-4\text{ Hz}$), Theta ($4-8\text{ Hz}$), Alpha ($8-13\text{ Hz}$), Beta ($13-30\text{ Hz}$), and Gamma ($30-40\text{ Hz}$).

---

### Tab 4: 🧠 Model Training & Cross-Validation
- **Purpose:** Train and calibrate the machine learning seizure detection model.
- **Sub-Tabs:**
  1. **`🏥 Real CHB-MIT Pipeline (chb01_03.edf)`**:
     - Explains the multi-recording patient-level holdout protocol:
       - **Train Partition:** `chb01_03.edf` (Patient `chb01`, 607 clean windows, 8 seizure windows).
       - **Validation Partition:** `chb01_04.edf` (Patient `chb01`, 300 clean windows, 6 seizure windows).
       - **Test Partition:** `chb02_16.edf` (Unseen Patient `chb02`, 38 clean windows, 13 seizure windows).
     - Button: **`🚀 Train Real CHB-MIT Baseline Model`**.
     - Executes feature extraction, model fitting, and validation threshold tuning.
     - Displays resulting test metrics: Test Accuracy (94.74%), Sensitivity (84.62%), Specificity (100.00%), F1-Score (91.67%).
  2. **`🧪 Multi-Patient Synthetic Generator`**:
     - Select classifier architecture: Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, Linear SVM, or MLP.
     - Button: **`⚡ Generate Benchmark & Train Models`**.

---

### Tab 5: ⚡ Seizure Detection & Prediction
- **Purpose:** Continuous window-by-window inference and episode clustering on the loaded recording.
- **Inputs Required:** A loaded EEG recording and a loaded/trained model.
- **Key Controls & Buttons:**
  1. **Decision Threshold Slider:** Adjust between 0.05 and 0.95 (default is the calibrated optimal threshold: `0.50`).
  2. **`Reset to Calibrated Threshold`** Button: Resets slider back to `0.50`.
  3. **`🔍 Execute Seizure Analysis`** Button: Starts feature extraction and scoring.
- **Expected Output:**
  - **Detection Summary Metrics:** Analyzed Windows ($610$), Detected Episodes ($1$), Total Seizure Time ($40.0\text{ s}$), Active Threshold ($0.50$).
  - **Predicted Seizure Risk Profile Plot:** Interactive timeline comparing model probability against threshold (red dashed line) with ground truth annotations (green shaded area).
  - **Detected Seizure Episodes Table:** Lists episode ID, onset time, offset time, duration, peak risk score, mean risk score, and alignment: `"✅ Verified True Positive"` or `"⚠️ False Alarm"`.

---

### Tab 6: 📋 Model Evaluation & Metrics
- **Purpose:** Comprehensive academic evaluation reports, confusion matrices, event statistics, and audit verification.
- **Sub-Tabs:**
  1. **`🏥 Real CHB-MIT Experiment Results (Official Artifacts)`**:
     - **Audit Report Expander:** Click `🔍 Audit Report: Single-Recording Chronological Split Bug & Correction` to view the window counts and mathematical proof.
     - **Partition Selector Radio:** Toggle between:
       - `Test Set (Unseen Patient chb02 - chb02_16.edf)`
       - `Validation Set (Tuning Split - chb01_04.edf)`
       - `Train Set (Fitting Split - chb01_03.edf)`
     - Displays corresponding Accuracy, Sensitivity, Specificity, F1-Score, and Confusion Matrix.
     - **Clinical Event Statistics:** Total True Seizures ($1$), Detected True Seizures ($1$), Event Sensitivity ($100.0\%$), False Alarms ($0$), False Alarm Rate ($0.00\text{ FA/hr}$).
     - Displays raw text of [`results/classification_report.txt`](results/classification_report.txt) and embedded image [`results/seizure_timeline.png`](results/seizure_timeline.png).
  2. **`🔬 Active Model Bundle Split`**:
     - Evaluates active session model against the validation set, test set, or `DummyClassifier` baseline.

---

### Tab 7: 📖 Documentation & Methodology
- **Purpose:** Documents dataset source, citations, license, signal processing pipeline, 18-channel bipolar montage, mathematical feature formulas, anti-leakage split protocols, and the medical disclaimer.
- **Inputs Required:** None.

---

## 4. Understanding Data & File Structure

### 4.1 File Distinctions
1. **Raw EEG Recordings (`.edf`):**  
   Standard European Data Format files containing continuous multi-channel voltage samples digitized at 256 Hz. (e.g., `data/raw/chb01/chb01_03.edf`).
2. **Seizure Annotation Files (`*-summary.txt` & `*.seizures`):**  
   - Text summaries (`chb01-summary.txt`): Human-readable clinical logs with patient demographics, channels, and seizure start/end times in seconds.
   - WFDB binary files (`chb01_03.edf.seizures`): Exact sample indices of seizure onset and offset.
3. **Extracted Feature Matrices:**  
   $546$-dimensional numerical feature vectors ($30$ time/frequency features $\times 18$ channels $+ 6$ spatial aggregates) computed over $5.0$-second non-overlapping windows.
4. **Saved Model Artifacts (`.joblib`):**  
   Serialized Python objects storing the scikit-learn pipeline (Imputer + `StandardScaler` + `RandomForestClassifier`), optimal decision threshold, and feature names.
5. **Experiment Metrics & Visual Artifacts (`results/`):**  
   JSON metrics, classification reports, and PNG plots produced by genuine execution.

### 4.2 Required Local CHB-MIT Files
Do **not** download the entire 43 GB PhysioNet dataset. The system requires only:
- [`data/raw/chb01/chb01_03.edf`](data/raw/chb01/chb01_03.edf) ($35.9\text{ MB}$, Patient `chb01` Train)
- [`data/raw/chb01/chb01_04.edf`](data/raw/chb01/chb01_04.edf) ($17.8\text{ MB}$, Patient `chb01` Validation)
- [`data/raw/chb02/chb02_16.edf`](data/raw/chb02/chb02_16.edf) ($2.3\text{ MB}$, Patient `chb02` Test Holdout)
- Summary files: [`data/annotations/chb01-summary.txt`](data/annotations/chb01-summary.txt) and [`data/annotations/chb02-summary.txt`](data/annotations/chb02-summary.txt)
- WFDB binary seizure files: `chb01_03.edf.seizures`, `chb01_04.edf.seizures`, and `chb02_16.edf.seizures`.

All of these are already cached and verified locally in the workspace.

---

## 5. Terminal Commands & CLI Execution

### Command 1: Run the Complete Real Experiment
Executes window segmentation, multichannel feature extraction, training on `chb01_03.edf`, threshold calibration on `chb01_04.edf`, and evaluation on unseen patient `chb02_16.edf`.

```powershell
python scripts/run_real_experiment.py
```
- **Execution Time:** ~18 seconds.
- **Generated Artifacts:** Regenerates all 8 files in `results/`.
- **Expected Terminal Output:**
  ```text
  Test Accuracy:         0.9474
  Test Sensitivity/Rec:  0.8462 (11/13)
  Test Specificity:      1.0000 (25/25)
  Test Precision:        1.0000 (11/11)
  Test F1-Score:         0.9167
  Test ROC-AUC:          1.0
  Confusion Matrix:      TN=25, FP=0, FN=2, TP=11
  Event Sensitivity:     100.00%
  False Alarms (Events): 0
  FAR per Hour:          0.00 / hr
  ```

### Command 2: Mathematical Audit of Single-Recording Split
Verifies the exact timestamps and window assignments for `chb01_03.edf`:

```powershell
python scripts/audit_chb01_03.py
```
- **Execution Time:** ~0.4 seconds.
- **Generated Artifact:** [`results/audit_chb01_03.json`](results/audit_chb01_03.json).
- **Verification Result:** Confirms 0 seizure windows in Train (0–1830s), 0 in Val (1830–2440s), and all 8 in Test (2440–3050s).

### Command 3: Automated PyTest Suite
Runs 27 unit tests validating data loading, window boundary buffers, feature extraction math, patient-level split isolation, and metric computations:

```powershell
pytest -v
```
- **Execution Time:** ~4.0 seconds.
- **Expected Result:** `27 passed in 4.04s` with zero errors or warnings.

### Command 4: Test Any Recording by Explicit Input
Pass an explicit path to any `.edf` recording file to test it with the trained model:

```powershell
# Test an arbitrary EDF recording:
python scripts/predict_recording.py --edf data/raw/chb02/chb02_16.edf

# Test with clinician ground truth evaluation:
python scripts/predict_recording.py --edf data/raw/chb02/chb02_16.edf --summary data/annotations/chb02-summary.txt

# Save window-by-window risk scores to a CSV:
python scripts/predict_recording.py --edf data/raw/chb02/chb02_16.edf --output-csv results/prediction_scores.csv
```
- **Interactive Mode:** If run with no arguments (`python scripts/predict_recording.py`), it prompts interactively for the file path.
- **Output:** Outputs window counts, detected seizure episode timestamps, peak risk scores, and (if annotations are supplied) the test confusion matrix, window accuracy, sensitivity, and specificity.

---

## 6. Interpretation of Evaluation Metrics

| Metric | Formula | Value | Plain Language Meaning |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{N}$ | **94.74%** | Percentage of all test windows correctly identified ($36/38$). |
| **Sensitivity (Recall)** | $\frac{TP}{TP + FN}$ | **84.62%** | Percentage of true seizure windows detected ($11/13$). Two windows were missed. |
| **Specificity** | $\frac{TN}{TN + FP}$ | **100.00%** | Percentage of non-seizure windows correctly classified as normal ($25/25$). Zero false positives. |
| **Precision** | $\frac{TP}{TP + FP}$ | **100.00%** | When the model flagged a window as seizure, it was correct $100\%$ of the time ($11/11$). |
| **F1-Score** | $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ | **0.9167** | Balanced harmonic mean of precision and sensitivity on seizure windows. |
| **ROC-AUC** | Area under ROC | **1.000** | Model assigned higher risk probabilities to all seizure windows than to non-seizure windows. |
| **Event Sensitivity** | $\frac{\text{Detected Seizures}}{\text{True Seizures}}$ | **100.00%** | The single clinical seizure event in the test recording was successfully flagged. |
| **False Alarm Rate** | $\frac{\text{False Alarms}}{\text{Hours}}$ | **0.00 / hr** | Clustered false positive episodes per hour of non-seizure monitoring. |

### Test Confusion Matrix
```
                         Predicted Non-Seizure (0)    Predicted Seizure (1)      Total
Actual Non-Seizure (0)              25 (TN)                      0 (FP)            25
Actual Seizure (1)                   2 (FN)                     11 (TP)            13
Total Predictions                   27                          11                 38
```

---

## 7. Critical Limitations & Academic Scope

1. **Cohort Size:** Evaluated on a single held-out test patient (`chb02`) and one test recording (`chb02_16.edf`).
2. **Limited Event Sample:** Contains only one seizure event and 38 valid test windows.
3. **Training Positives:** The training recording (`chb01_03.edf`) contains only 8 positive seizure windows (40.0 seconds total seizure duration).
4. **Generalization Requirement:** More patients and recordings across diverse age groups and seizure types (focal, generalized, tonic-clonic) are required before claiming general clinical efficacy.
5. **Unverified / Unimplemented Features:**  
   > ⚠️ **Notice on Homomorphic Encryption:** Privacy-preserving homomorphic encryption is **NOT implemented** in this repository. It is a proposed future research concept. Do **not** claim or attempt to demonstrate encrypted inference.
