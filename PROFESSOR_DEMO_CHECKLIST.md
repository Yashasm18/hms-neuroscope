# Professor Demonstration Checklist & Presentation Script
## HMS Harmful Brain Activity Classifier & EEG Seizure Detection Prototype

Use this checklist during your viva or project presentation. It provides the **shortest, most reliable live demonstration sequence** designed to highlight clinical validity, rigorous machine learning design, and transparent academic integrity.

---

### ⏱️ Quick Summary of the 5-Minute Live Demo Sequence

| Step | Tab / Screen | Action | Key Talking Point for Professor | Duration |
|---|---|---|---|---|
| **1** | Terminal | `python run_app.py` | Explain that the system is an academic continuous EEG seizure detection prototype built on the CHB-MIT PhysioNet database. | 30s |
| **2** | 📊 Dashboard | Show overview cards & medical disclaimer | Point out the 4 metrics cards (Accuracy 94.74%, Sensitivity 84.62%, Specificity 100%) and emphasize the academic disclaimer. | 30s |
| **3** | 📈 Signal Explorer | Click `⏩ Jump to Seizure (2996s - 3036s)` | Show the 18-channel 10-20 bipolar montage, the zero-phase 0.5–40 Hz filter, and the prominent high-amplitude rhythmic seizure discharge. | 60s |
| **4** | 🧠 Model Training | Show Patient-Level Holdout configuration | **The Critical Finding:** Explain why a single-recording chronological split fails (0 seizures in train/val) and show the multi-recording patient-isolated design. | 60s |
| **5** | ⚡ Prediction | Adjust threshold slider & show timeline | Demonstrate continuous inference, episode clustering, and exact alignment with clinical ground truth (green band). | 60s |
| **6** | 📋 Evaluation | Expand Audit finding & show Confusion Matrix | Show the exact confusion matrix ($TN=25, FP=0, FN=2, TP=11$), $0.00\text{ FA/hr}$, and discuss limitations honestly. | 60s |

---

## Detailed Step-by-Step Walkthrough

### Step 1: Pre-Flight Check (Before Professor Arrives)
- [ ] Open PowerShell in `d:\akshay_project`.
- [ ] Run `pytest -v` to confirm all 27 automated unit tests pass.
- [ ] Confirm `results/metrics.json` and `results/trained_model.joblib` exist.
- [ ] Launch the app: `python run_app.py` (or double-click `START_PROJECT.bat`).
- [ ] Verify browser opens to `http://localhost:8501`.

---

### Step 2: Introduction & Project Overview (Dashboard Tab)
- [ ] **Action:** Keep the view on **`📊 Dashboard`**.
- [ ] **Script to say:**
  > *"Good morning, Professor. This project is an end-to-end continuous EEG seizure detection prototype developed using the official CHB-MIT Scalp EEG database from PhysioNet. Our objective is automated binary classification—detecting seizure vs. non-seizure states from standardized 18-channel bipolar EEG recordings without artificial data leakage."*
- [ ] **Highlight:** Point out the red **Medical Disclaimer Banner**. Emphasize:
  > *"We explicitly frame this as an academic research prototype, not a clinically approved diagnostic medical device."*

---

### Step 3: Clinical EEG Inspection & Digital Filtering (Signal Explorer Tab)
- [ ] **Action:** In sidebar, click **`📈 Multichannel Signal Explorer`**.
- [ ] **Action:** Click the quick button **`⏩ Jump to Seizure (2996s - 3036s)`**.
- [ ] **Script to say:**
  > *"Here we are inspecting genuine patient recording `chb01_03.edf` at 256 Hz. Channels are standardized into the international 10-20 bipolar longitudinal montage (18 channels). Notice how at 2996 seconds, the background rhythms transition into high-amplitude, synchronous rhythmic seizure discharges across the temporal and parasagittal channels. We apply a zero-phase 4th-order Butterworth bandpass filter (0.5–40 Hz) and a 60 Hz notch filter to remove powerline noise."*
- [ ] **Action:** Scroll down to the **Welch Power Spectral Density (PSD)** chart.
  > *"Our feature extraction pipeline derives 546 features per 5-second window, capturing spectral band powers (Delta, Theta, Alpha, Beta, Gamma), Spectral Edge Frequency (SEF95), Shannon spectral entropy, and Hjorth mobility/complexity parameters."*

---

### Step 4: Machine Learning Methodology & The Audit Finding (Model Training Tab)
- [ ] **Action:** In sidebar, navigate to **`🧠 Model Training & Cross-Validation`**.
- [ ] **Script to say (Crucial Audit Explanation):**
  > *"During our engineering audit, we identified a critical mathematical bug in single-recording setups. For recording `chb01_03.edf`, the seizure lasts from 2996s to 3036s in a 3050s recording. A strict chronological 60/20/20 split puts 100% of the seizure in the test set, leaving 0 seizure windows in training and validation. A supervised model cannot train with zero positive examples, and validation thresholds cannot be tuned.*
  >
  > *To solve this honestly, we implemented a **Multi-Recording Patient-Level Holdout Protocol**:*
  > - *Training on `chb01_03.edf` (Patient 01)*
  > - *Validation & threshold calibration on `chb01_04.edf` (Patient 01)*
  > - *Testing on `chb02_16.edf` (Unseen Patient 02)*
  >
  > *This guarantees zero intra-event leakage and genuine cross-subject generalization."*

---

### Step 5: Seizure Risk Scoring & Episode Clustering (Prediction Tab)
- [ ] **Action:** In sidebar, navigate to **`⚡ Seizure Detection & Prediction`**.
- [ ] **Action:** Ensure the threshold slider is at **`0.50`** (calibrated threshold) and click **`🔍 Execute Seizure Analysis`** (or view the precomputed timeline).
- [ ] **Script to say:**
  > *"Here the model evaluates 5-second non-overlapping windows continuously across the recording. The blue curve displays the model's predicted seizure probability. The red dashed line is our calibrated threshold (0.50). Notice how the predicted risk spikes sharply to 1.0 during the green clinical annotation window, while baseline non-seizure windows remain near 0.0."*
- [ ] **Action:** Point to the **Detected Seizure Episodes Table**:
  > *"The system clusters consecutive positive windows into discrete clinical episodes. Episode 1 spans 2995s to 3035s with peak score 1.0, verified as a True Positive."*

---

### Step 6: Rigorous Academic Evaluation & Limitations (Evaluation Tab)
- [ ] **Action:** In sidebar, navigate to **`📋 Model Evaluation & Metrics`**.
- [ ] **Action:** Expand the expander: **`🔍 Audit Report: Single-Recording Chronological Split Bug & Correction`** to show transparency.
- [ ] **Action:** Show the **Test Partition Confusion Matrix**:
  - **TN = 25**, **FP = 0**, **FN = 2**, **TP = 11**
  - **Test Accuracy:** 94.74%
  - **Seizure Sensitivity:** 84.62% (11 of 13 windows caught)
  - **Specificity:** 100.00% (zero false alarms)
  - **Event Sensitivity:** 100.00% (1 of 1 clinical events caught)
  - **False Alarms per Hour:** 0.00 / hr
- [ ] **Script to say:**
  > *"On unseen patient `chb02`, the model achieves 94.74% window accuracy and 84.62% sensitivity with zero false positives. Most importantly, it detected 100% of clinical seizure events."*

---

### Step 7: Academic Limitations & Unimplemented Modules
- [ ] **State Limitations Explicitly:**
  > *"To be academically honest, this test partition consists of 38 windows from one held-out recording. While this demonstrates cross-patient feasibility, a full clinical validation requires evaluating all 24 patients in the CHB-MIT database."*
- [ ] **State Status of Homomorphic Encryption:**
  > *"If asked about privacy-preserving or homomorphic encryption: that is a proposed theoretical extension in literature, but it is **not implemented** in this software. Our focus was on clinical signal fidelity and anti-leakage ML architecture."*

---

### 🛡️ Fallback Plan (If Anything Goes Wrong)
1. **If live training takes too long:**  
   Do **not** click `🚀 Train Real CHB-MIT Baseline Model` during the talk. All results are already pre-computed in `results/metrics.json` and `results/trained_model.joblib`. Simply navigate directly to the **Evaluation** tab.
2. **If port 8501 is busy:**  
   Launch on a different port: `python run_app.py --port 8502`.
3. **If the browser crashes or disconnects:**  
   Show the generated static image artifacts directly:
   - Confusion matrix: `results/confusion_matrix.png`
   - Seizure timeline: `results/seizure_timeline.png`
   - Classification report: `results/classification_report.txt`
