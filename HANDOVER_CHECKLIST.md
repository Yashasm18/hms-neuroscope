# Project Handover & Recipient Verification Checklist
## HMS Harmful Brain Activity Classifier & EEG Seizure Detection Prototype

Give this checklist to your friend or teammate. Following these steps verifies that the project works end-to-end on a new Windows computer without manual code changes.

---

### Phase 1: Unpacking & Prerequisites Check

- [ ] **1. Extract Archive:**  
  Extract `akshay_project_release.zip` into a clean directory (e.g., `C:\Users\<username>\Desktop\akshay_project_release` or `D:\akshay_project_release`).
- [ ] **2. Check Python Installation:**  
  Open PowerShell or Command Prompt and run:
  ```powershell
  python --version
  ```
  - Required: **Python 3.10, 3.11, or 3.12** (verified on Python 3.12.4).
  - *If Python is not recognized:* Download and install Python from [python.org](https://www.python.org/downloads/) and ensure **"Add Python to PATH"** is checked during setup.
- [ ] **3. Verify Working Directory:**  
  Make sure you open the project in the extracted root folder where `run_app.py` and `START_PROJECT.bat` are located.

---

### Phase 2: One-Click Launch & Dependency Installation

- [ ] **4. Run the One-Click Launcher:**  
  Double-click **`START_PROJECT.bat`** (or run `.\START_PROJECT.bat` in PowerShell).
- [ ] **5. Watch the Automated Setup:**  
  The script automatically:
  - Creates a local isolated virtual environment (`.venv\`).
  - Installs all dependencies from `requirements.txt` (MNE-Python, FastAPI, Scikit-learn, etc.).
  - Confirms that `results\trained_model.joblib` and `data\raw\` files exist.
  - Launches the web server on port 8501 and opens your browser.
- [ ] **6. Open Dashboard:**  
  Confirm your browser opens automatically to:
  **`http://localhost:8501`**

---

### Phase 3: Dashboard Verification

- [ ] **7. Dashboard Overview Tab:**  
  Check the 4 top metric cards:
  - *PhysioNet Records:* `141`
  - *Test Accuracy:* `94.74%`
  - *Seizure Sensitivity:* `84.62%`
  - *Specificity:* `100.00%`
- [ ] **8. Multichannel Signal Explorer Tab:**  
  - Select `📈 Multichannel Signal Explorer` from the sidebar.
  - Click `⏩ Jump to Seizure (2996s - 3036s)`.
  - Confirm the 18 bipolar EEG channels display with the green shaded seizure overlay.
- [ ] **9. Seizure Detection & Prediction Tab:**  
  - Select `⚡ Seizure Detection & Prediction`.
  - Click `🔍 Execute Seizure Analysis`.
  - Confirm the predicted risk curve spikes to 1.0 during the seizure and that Episode 1 appears in the Detected Episodes table.
- [ ] **10. Model Evaluation Tab:**  
  - Select `📋 Model Evaluation & Metrics`.
  - Open the expander: `🔍 Audit Report: Single-Recording Chronological Split Bug & Correction`.
  - Verify the test confusion matrix: $TN=25, FP=0, FN=2, TP=11$.

---

### Phase 4: Automated Test Suite & CLI Verification

- [ ] **11. Run Automated Unit Tests:**  
  Double-click **`RUN_TESTS.bat`** (or run `pytest -v`).  
  - **Expected Result:** `27 passed in ~4.0s` with zero errors.
- [ ] **12. Run Explicit Input CLI Inference:**  
  Open PowerShell in the project directory and run:
  ```powershell
  python scripts/predict_recording.py --edf data/raw/chb02/chb02_16.edf --summary data/annotations/chb02-summary.txt
  ```
  - **Expected Result:** Outputs detection summary and confirms $TN=26, FP=0, FN=3, TP=10$ on the test recording.

---

### Phase 5: Academic & Medical Integrity Confirmation

- [ ] **13. Disclaimer Visibility:**  
  Confirm that the red academic research disclaimer appears across the dashboard.
- [ ] **14. No Fabricated Data:**  
  Confirm that all metrics originate directly from `results/metrics.json` and actual model execution on CHB-MIT recordings.
- [ ] **15. Ready for Presentation:**  
  Review `PROFESSOR_DEMO_CHECKLIST.md` for the 5-minute presentation script.
