# Project Architecture & Directory Structure Guide
## HMS Harmful Brain Activity Classifier & EEG Seizure Detection Prototype

```text
akshay_project_release/
│
├── START_PROJECT.bat              # One-click Windows launcher (creates venv, installs deps, runs app)
├── RUN_TESTS.bat                  # Runs the 27-test automated verification suite
├── RUN_EXPERIMENT.bat             # Executes real CHB-MIT experiment pipeline and updates results/
│
├── app.py                         # Streamlit multi-tab scientific web application
├── config.py                      # Central constants, montage definitions, and file paths
├── requirements.txt               # Pinned Python package dependencies (compatible with Python 3.10-3.12)
├── .env.example                   # Environment variable template
├── .gitignore                     # Git exclusions for venv, caches, and large archives
│
├── README.md                      # Project overview, quick-start guide, and core instructions
├── USER_MANUAL.md                 # Detailed dashboard operations, CLI commands, and metrics guide
├── DATA_SETUP.md                  # Dataset source, folder layout, and verification guide
├── TROUBLESHOOTING.md             # Common Windows errors, port conflicts, and fixes
├── PROJECT_STRUCTURE.md           # This architecture and directory explanation file
├── HANDOVER_CHECKLIST.md          # Step-by-step verification checklist for the project recipient
├── PROFESSOR_DEMO_CHECKLIST.md    # 5-minute presentation script and viva checklist
│
├── data/
│   ├── annotations/               # Clinician ground truth summaries
│   │   ├── chb01-summary.txt      # Text summary for Patient 01
│   │   ├── chb02-summary.txt      # Text summary for Patient 02
│   │   └── RECORDS-WITH-SEIZURES  # Index of all 141 seizure recordings in CHB-MIT
│   ├── raw/                       # Curated clinical EDF recordings
│   │   ├── chb01/                 # Patient 01 recordings
│   │   │   ├── chb01_03.edf       # Training recording (3,050s, seizure @ 2996s-3036s)
│   │   │   ├── chb01_03.edf.seizures # Binary sample annotation
│   │   │   ├── chb01_04.edf       # Validation recording (1,510s, seizure @ 1467s-1494s)
│   │   │   └── chb01_04.edf.seizures # Binary sample annotation
│   │   └── chb02/                 # Patient 02 recordings (Unseen holdout)
│   │       ├── chb02_16.edf       # Test recording (196s, seizure @ 130s-196s)
│   │       └── chb02_16.edf.seizures # Binary sample annotation
│   └── processed/                 # Scratch storage for temporary caches
│
├── models/                        # Serialized model bundles directory
│
├── results/                       # Verified, reproducible experiment artifacts
│   ├── metrics.json               # Full JSON record of train, val, test confusion matrices & metrics
│   ├── audit_chb01_03.json        # Timestamp audit proving single-recording split bug & correction
│   ├── classification_report.txt  # Precision, recall, F1, and support text report
│   ├── confusion_matrix.png       # High-resolution Seaborn/Matplotlib confusion matrix
│   ├── seizure_timeline.png       # Test timeline: predicted risk profile vs. clinical ground truth
│   ├── experiment_config.json     # Hyperparameters, filters, and feature extraction configuration
│   ├── run_log.txt                # Timestamped execution trace of the real experiment
│   └── trained_model.joblib       # Serialized scikit-learn pipeline (Imputer + StandardScaler + RF)
│
├── scripts/
│   ├── run_real_experiment.py     # Main CLI script: extracts features, fits model, tunes threshold
│   ├── audit_chb01_03.py          # Mathematical timestamp auditor for chb01_03.edf
│   ├── predict_recording.py       # Explicit-input CLI inference tool for testing any new EDF file
│   └── prepare_demo_dataset.py    # Benchmark initialization utility
│
├── src/                           # Modular Python package source tree
│   ├── __init__.py
│   ├── data/                      # Data ingestion & parsing
│   │   ├── edf_loader.py          # MNE-Python EDF reader and StandardizedEEG container
│   │   ├── annotation_parser.py   # Clinician summary and WFDB binary annotation parser
│   │   ├── downloader.py          # Resumable range-request PhysioNet downloader
│   │   └── synthetic.py           # Biological baseline EEG generator with injected spikes
│   │
│   ├── preprocessing/             # Digital signal processing
│   │   ├── filter.py              # Zero-phase Butterworth bandpass (0.5-40Hz) and 60Hz notch
│   │   ├── montage.py             # 18-channel longitudinal bipolar montage standardization
│   │   └── windowing.py           # 5.0s window segmentation with 5.0s boundary buffer exclusion
│   │
│   ├── features/                  # Biomedical feature engineering
│   │   ├── time_frequency.py      # Statistical, Hjorth, Welch PSD, and Shannon spectral entropy
│   │   └── extractor.py           # 546-dimensional coordinated multichannel feature extractor
│   │
│   ├── training/                  # Model training & anti-leakage splitting
│   │   ├── split.py               # Patient-level group splitting (zero cross-subject leakage)
│   │   ├── pipeline.py            # Imputer + StandardScaler + Balanced Classifier pipeline
│   │   └── train.py               # Threshold calibration and joblib bundle serialization
│   │
│   ├── evaluation/                # Performance metrics
│   │   └── metrics.py             # Accuracy, sensitivity, specificity, F1, event sensitivity, FAR/hr
│   │
│   ├── inference/                 # Continuous scoring & clustering
│   │   └── predictor.py           # SeizurePredictor class, episode clustering, risk scoring
│   │
│   └── visualization/             # Interactive plotting
│       └── plots.py               # Dark Plotly multichannel waveforms, PSD, ROC, timelines
│
└── tests/                         # Automated unit test suite (27 tests)
    ├── test_annotations.py        # Validates summary parsing and timestamp overlap math
    ├── test_audit.py              # Validates single-recording audit and window reconciliation
    ├── test_edf_loader.py         # Validates EDF loading and channel cleaning
    ├── test_features.py           # Validates statistical and spectral feature math
    ├── test_inference.py          # Validates SeizurePredictor continuous inference
    ├── test_metrics.py            # Validates confusion matrix, accuracy, sensitivity, and FAR/hr
    ├── test_preprocessing.py      # Validates Butterworth filters, notch filters, and windowing
    ├── test_real_experiment.py    # Validates real CHB-MIT loading and results artifact integrity
    └── test_training.py           # Validates patient-level split isolation and pipeline fitting
```
