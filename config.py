"""
Centralized Configuration and Constants for CHB-MIT EEG Seizure Detection.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = BASE_DIR / "models"

for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ANNOTATIONS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# PhysioNet CHB-MIT Database details
PHYSIONET_BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
PHYSIONET_RECORDS_WITH_SEIZURES_URL = (
    "https://physionet.org/files/chbmit/1.0.0/RECORDS-WITH-SEIZURES"
)
PHYSIONET_CITATION = (
    "Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). "
    "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. "
    "Circulation, 101(23), e215-e220."
)
DATASET_LICENSE = "Open Data Commons Public Domain Dedication and License v1.0 (ODC-PDDL)"

# Signal and Preprocessing Defaults
DEFAULT_SAMPLING_RATE = 256.0  # Hz (CHB-MIT standard)
DEFAULT_LOW_PASS = 40.0        # Hz
DEFAULT_HIGH_PASS = 0.5        # Hz
DEFAULT_NOTCH_FREQ = 60.0      # Hz powerline interference in US recordings
FILTER_ORDER = 4

# Windowing Defaults
DEFAULT_WINDOW_DURATION_SEC = 5.0
DEFAULT_WINDOW_STEP_SEC = 5.0   # Non-overlapping by default, or configurable
DEFAULT_SEIZURE_OVERLAP_RATIO = 0.5  # Fraction of window that must be seizure to label as 1
DEFAULT_BOUNDARY_BUFFER_SEC = 5.0    # Exclusion buffer around seizure boundaries

# Standard 18-channel bipolar montage commonly preserved across CHB-MIT recordings
STANDARD_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]

# Frequency bands (Hz) for Spectral Power Feature Extraction
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# Medical Disclaimer String
MEDICAL_DISCLAIMER = (
    "RESEARCH PROTOTYPE ONLY: This software is developed solely for academic research and "
    "educational experimentation on the CHB-MIT EEG database. It is NOT a medical device, is "
    "NOT clinically validated or FDA-approved, and must NEVER be used for clinical diagnosis, "
    "treatment decisions, or emergency patient monitoring. Predicted scores reflect academic model "
    "outputs and are not calibrated clinical probabilities."
)
