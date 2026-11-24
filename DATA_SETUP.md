# CHB-MIT EEG Data Setup & Management Guide
## HMS Harmful Brain Activity Classifier & EEG Seizure Detection Prototype

> **ACADEMIC DISCLAIMER:** This dataset guide is provided solely for academic research and educational experimentation on scalp EEG recordings. This prototype is **NOT** a medical diagnostic device.

---

## 1. Dataset Origin & PhysioNet Source

The project uses the publicly available **CHB-MIT Scalp EEG Database (v1.0.0)** collected at Children's Hospital Boston and hosted by PhysioNet:
- **Official Database URL:** [https://physionet.org/content/chbmit/1.0.0/](https://physionet.org/content/chbmit/1.0.0/)
- **Direct Files Directory:** [https://physionet.org/files/chbmit/1.0.0/](https://physionet.org/files/chbmit/1.0.0/)
- **License:** Open Data Commons Public Domain Dedication and License v1.0 (ODC-PDDL).
- **Format:** European Data Format (`.edf`) continuous recordings digitized at 256 Hz with 16-bit resolution.

---

## 2. Pre-Packaged Sample Recordings (Immediate Demonstration)

To ensure your friend can run the project immediately without waiting hours to download large files over throttled connections, **a curated set of sample recordings is already included in this release package**:

| File Path | Patient | Duration | Clinical Seizure Interval | Role in Experiment |
|---|---|---|---|---|
| `data/raw/chb01/chb01_03.edf` | `chb01` | 3,050.0s (50.8 min) | **2996.0s – 3036.0s** (40.0s) | **Training Partition** (Fitting Random Forest) |
| `data/raw/chb01/chb01_04.edf` | `chb01` | 1,510.0s (25.2 min) | **1467.0s – 1494.0s** (27.0s) | **Validation Partition** (Threshold Calibration) |
| `data/raw/chb02/chb02_16.edf` | `chb02` | 196.0s (3.3 min) | **130.0s – 196.0s** (66.0s) | **Test Partition** (Unseen Patient Evaluation) |

Corresponding clinician annotations (`chb01-summary.txt`, `chb02-summary.txt`, and binary `.seizures` files) are pre-placed in `data/annotations/` and `data/raw/`.

---

## 3. Expected Directory Structure

Ensure the files remain organized as follows:

```text
data/
├── annotations/
│   ├── chb01-summary.txt          # Clinician time logs for patient 01
│   ├── chb02-summary.txt          # Clinician time logs for patient 02
│   └── RECORDS-WITH-SEIZURES      # Index of all 141 seizure records in CHB-MIT
├── raw/
│   ├── chb01/
│   │   ├── chb01_03.edf           # Genuine EEG recording (Train)
│   │   ├── chb01_03.edf.seizures  # Binary WFDB annotation
│   │   ├── chb01_04.edf           # Genuine EEG recording (Validation)
│   │   └── chb01_04.edf.seizures  # Binary WFDB annotation
│   └── chb02/
│       ├── chb02_16.edf           # Genuine EEG recording (Held-out Test)
│       └── chb02_16.edf.seizures  # Binary WFDB annotation
└── processed/                     # Scratch folder for temporary caches
```

---

## 4. How to Download Additional CHB-MIT Files (Optional)

> ⚠️ **DO NOT download the entire 43 GB dataset.** The application only needs individual files for analysis.

If you wish to test other subjects (e.g., `chb03`):
1. Navigate to the PhysioNet folder in your browser:  
   `https://physionet.org/files/chbmit/1.0.0/chb03/`
2. Download the summary file: `chb03-summary.txt` and save it to `data/annotations/`.
3. Open `chb03-summary.txt` to find which recordings contain seizures (e.g., `chb03_01.edf`).
4. Download `chb03_01.edf` and save it into a new folder: `data/raw/chb03/`.

---

## 5. How to Upload a Custom EDF via the Dashboard

To test an arbitrary EDF recording without modifying folders:
1. Start the app: double-click `START_PROJECT.bat` (or run `streamlit run app.py`).
2. In the sidebar, select **`📁 EEG Upload & Data Manager`**.
3. Select the tab: **`Upload Custom EDF File`**.
4. Drag and drop any `.edf` file.
5. The system will inspect:
   - Sampling frequency (expected: 256 Hz; resamples or reports if different).
   - Standard 10-20 bipolar montage (18 channels).
   - Total duration.
6. Navigate to **`⚡ Seizure Detection & Prediction`** to run inference.

---

## 6. Prediction Input vs. Training Labels

It is important to understand the difference between inference and training:
- **Prediction / Inference Input (Unsupervised):**
  Requires **only** the raw `.edf` file. The model processes the voltage signals, computes spectral powers, and outputs continuous seizure risk scores ($0.0$ to $1.0$). No clinician annotation is needed to run predictions.
- **Model Training / Supervised Evaluation:**
  Requires both the `.edf` file **and** the verified seizure start/end times from `*-summary.txt`. This allows the windowing pipeline to label 5-second segments as Class 1 (Seizure) or Class 0 (Non-seizure) and flag transition buffer zones as Class -1 (Ambiguous/Excluded).

---

## 7. How to Verify Data Integrity

Run the automated verification script from PowerShell:

```powershell
python scripts/predict_recording.py --edf data/raw/chb02/chb02_16.edf --summary data/annotations/chb02-summary.txt
```

If the files are in the right place, you will see `[SUCCESS] Inference completed successfully!` and a printed confusion matrix showing $TN=26$, $FP=0$, $FN=3$, $TP=10$.
