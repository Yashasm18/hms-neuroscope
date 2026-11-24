"""
Unit tests for CHB-MIT annotation and summary file parsing.
"""

import pytest
from src.data.annotation_parser import (
    parse_summary_text,
    extract_patient_id,
    SeizureAnnotation,
    RecordingSummary,
)

SAMPLE_SUMMARY_TEXT = """
Data Sampling Rate: 256 Hz
*************************

Channels in EDF Files:
**********************
Channel 1: FP1-F7
Channel 2: F7-T7

File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb06_01.edf
File Start Time: 17:23:00
File End Time: 18:23:00
Number of Seizures in File: 2
Seizure 1 Start Time: 1724 seconds
Seizure 1 End Time: 1738 seconds
Seizure 2 Start Time: 1862 seconds
Seizure 2 End Time: 1963 seconds
"""


def test_extract_patient_id():
    assert extract_patient_id("chb01_03.edf") == "chb01"
    assert extract_patient_id("D:/path/to/chb15_22.edf") == "chb15"
    assert extract_patient_id("CHB02_01.EDF") == "chb02"
    assert extract_patient_id("random_sample.edf") == "unknown"


def test_parse_summary_text():
    recordings = parse_summary_text(SAMPLE_SUMMARY_TEXT)

    assert len(recordings) == 3
    assert "chb01_01.edf" in recordings
    assert "chb01_03.edf" in recordings
    assert "chb06_01.edf" in recordings

    # Check non-seizure file
    rec1 = recordings["chb01_01.edf"]
    assert rec1.num_seizures == 0
    assert not rec1.has_seizure
    assert len(rec1.seizures) == 0

    # Check single seizure file
    rec3 = recordings["chb01_03.edf"]
    assert rec3.num_seizures == 1
    assert rec3.has_seizure
    assert len(rec3.seizures) == 1
    assert rec3.seizures[0].start_sec == 2996.0
    assert rec3.seizures[0].end_sec == 3036.0
    assert rec3.seizures[0].duration_sec == 40.0
    assert rec3.patient_id == "chb01"

    # Check multi-seizure file
    rec6 = recordings["chb06_01.edf"]
    assert rec6.num_seizures == 2
    assert len(rec6.seizures) == 2
    assert rec6.seizures[0].start_sec == 1724.0
    assert rec6.seizures[0].end_sec == 1738.0
    assert rec6.seizures[1].start_sec == 1862.0
    assert rec6.seizures[1].end_sec == 1963.0
    assert rec6.patient_id == "chb06"


def test_seizure_overlap_calculation():
    sz = SeizureAnnotation(index=1, start_sec=100.0, end_sec=130.0)

    # Window completely before
    assert sz.overlaps_window(50.0, 60.0) == 0.0

    # Window completely inside
    assert sz.overlaps_window(105.0, 115.0) == 10.0

    # Window spanning start boundary
    assert sz.overlaps_window(95.0, 105.0) == 5.0

    # Window spanning end boundary
    assert sz.overlaps_window(125.0, 135.0) == 5.0

    # Window completely enclosing seizure
    assert sz.overlaps_window(90.0, 140.0) == 30.0


def test_real_wfdb_seizures_parsing():
    from pathlib import Path
    from src.data.annotation_parser import parse_wfdb_seizures, load_verified_recording_annotations

    seizures_file = Path("data/raw/chb01/chb01_03.edf.seizures")
    if seizures_file.exists():
        seizures = parse_wfdb_seizures(seizures_file, sfreq=256.0)
        assert len(seizures) == 1
        assert seizures[0].start_sec == 2996.0
        assert seizures[0].end_sec == 3036.0
        assert seizures[0].duration_sec == 40.0

    summary_file = Path("data/annotations/chb01-summary.txt")
    if summary_file.exists() and seizures_file.exists():
        verified = load_verified_recording_annotations(
            edf_file_name_or_path="chb01_03.edf",
            summary_path=summary_file,
            seizures_path=seizures_file,
        )
        assert verified.num_seizures == 1
        assert verified.seizures[0].start_sec == 2996.0
        assert verified.seizures[0].end_sec == 3036.0

