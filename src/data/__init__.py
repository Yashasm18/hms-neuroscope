"""Data loading, downloading, annotation parsing, and synthetic signal generation."""
from .annotation_parser import parse_summary_file, parse_summary_text, SeizureAnnotation, RecordingSummary
from .downloader import PhysioNetDownloader
from .edf_loader import load_edf_recording, validate_eeg_raw, StandardizedEEG
from .synthetic import generate_synthetic_eeg_record

__all__ = [
    "parse_summary_file",
    "parse_summary_text",
    "SeizureAnnotation",
    "RecordingSummary",
    "PhysioNetDownloader",
    "load_edf_recording",
    "validate_eeg_raw",
    "StandardizedEEG",
    "generate_synthetic_eeg_record",
]
