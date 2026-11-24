"""
Parser for CHB-MIT EEG summary files (*-summary.txt).
Extracts sampling rate, channel names, file timelines, and seizure annotations.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SeizureAnnotation:
    """Represents a single verified seizure event within an EDF recording."""
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def overlaps_window(self, win_start: float, win_end: float) -> float:
        """Calculate overlap duration in seconds between seizure and window."""
        overlap_start = max(self.start_sec, win_start)
        overlap_end = min(self.end_sec, win_end)
        return max(0.0, overlap_end - overlap_start)


@dataclass
class RecordingSummary:
    """Metadata and seizure annotations for a single EDF recording."""
    file_name: str
    patient_id: str
    start_time_str: Optional[str] = None
    end_time_str: Optional[str] = None
    num_seizures: int = 0
    seizures: List[SeizureAnnotation] = field(default_factory=list)

    @property
    def has_seizure(self) -> bool:
        return self.num_seizures > 0 and len(self.seizures) > 0

    @property
    def total_seizure_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.seizures)


def extract_patient_id(file_name_or_path: str) -> str:
    """Extract standard patient ID (e.g., 'chb01' from 'chb01_03.edf' or full path)."""
    base = Path(file_name_or_path).name
    match = re.match(r"^(chb\d+)", base, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown"


def parse_summary_text(summary_text: str) -> Dict[str, RecordingSummary]:
    """
    Parse the contents of a CHB-MIT *-summary.txt file into a dictionary of RecordingSummary.
    Keys are lowercased file names (e.g. 'chb01_03.edf').
    """
    recordings: Dict[str, RecordingSummary] = {}
    lines = summary_text.splitlines()

    current_file: Optional[str] = None
    current_start_time: Optional[str] = None
    current_end_time: Optional[str] = None
    current_num_seizures = 0
    current_seizures: List[SeizureAnnotation] = []

    # Seizure regex patterns
    # Matches: "Seizure Start Time: 2996 seconds" or "Seizure 1 Start Time: 2996 seconds"
    start_pattern = re.compile(
        r"Seizure\s*(?:(\d+))?\s*Start\s*(?:Time)?:\s*(\d+(?:\.\d+)?)\s*sec",
        re.IGNORECASE,
    )
    end_pattern = re.compile(
        r"Seizure\s*(?:(\d+))?\s*End\s*(?:Time)?:\s*(\d+(?:\.\d+)?)\s*sec",
        re.IGNORECASE,
    )
    file_name_pattern = re.compile(r"^File\s*Name\s*:\s*(\S+)", re.IGNORECASE)
    start_time_pattern = re.compile(r"^File\s*Start\s*Time\s*:\s*(.+)$", re.IGNORECASE)
    end_time_pattern = re.compile(r"^File\s*End\s*Time\s*:\s*(.+)$", re.IGNORECASE)
    num_seizures_pattern = re.compile(
        r"Number\s*of\s*Seizures(?:\s*in\s*File)?\s*:\s*(\d+)", re.IGNORECASE
    )

    pending_start: Optional[Tuple[int, float]] = None

    def commit_current_recording():
        nonlocal current_file, current_start_time, current_end_time, current_num_seizures, current_seizures
        if current_file:
            cleaned_name = current_file.strip().lower()
            if not cleaned_name.endswith(".edf"):
                cleaned_name += ".edf"
            patient_id = extract_patient_id(cleaned_name)
            recordings[cleaned_name] = RecordingSummary(
                file_name=cleaned_name,
                patient_id=patient_id,
                start_time_str=current_start_time,
                end_time_str=current_end_time,
                num_seizures=max(current_num_seizures, len(current_seizures)),
                seizures=list(current_seizures),
            )
        current_file = None
        current_start_time = None
        current_end_time = None
        current_num_seizures = 0
        current_seizures = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        file_match = file_name_pattern.match(line_clean)
        if file_match:
            commit_current_recording()
            current_file = file_match.group(1).strip()
            continue

        if current_file is None:
            continue

        st_match = start_time_pattern.match(line_clean)
        if st_match:
            current_start_time = st_match.group(1).strip()
            continue

        et_match = end_time_pattern.match(line_clean)
        if et_match:
            current_end_time = et_match.group(1).strip()
            continue

        num_match = num_seizures_pattern.match(line_clean)
        if num_match:
            current_num_seizures = int(num_match.group(1))
            continue

        sz_start_match = start_pattern.search(line_clean)
        if sz_start_match:
            idx_str, sec_str = sz_start_match.groups()
            sz_idx = int(idx_str) if idx_str else len(current_seizures) + 1
            sz_start_sec = float(sec_str)
            pending_start = (sz_idx, sz_start_sec)
            continue

        sz_end_match = end_pattern.search(line_clean)
        if sz_end_match:
            idx_str, sec_str = sz_end_match.groups()
            sz_idx = int(idx_str) if idx_str else (pending_start[0] if pending_start else len(current_seizures) + 1)
            sz_end_sec = float(sec_str)
            if pending_start and pending_start[0] == sz_idx:
                sz_start_sec = pending_start[1]
            else:
                # Fallback if start wasn't matched with same index
                sz_start_sec = pending_start[1] if pending_start else 0.0

            if sz_end_sec > sz_start_sec:
                current_seizures.append(
                    SeizureAnnotation(
                        index=sz_idx,
                        start_sec=sz_start_sec,
                        end_sec=sz_end_sec,
                    )
                )
            pending_start = None

    commit_current_recording()
    return recordings


def parse_wfdb_seizures(seizures_path: str | Path, sfreq: float = 256.0) -> List[SeizureAnnotation]:
    """
    Parse a binary WFDB .seizures annotation file (e.g. chb01_03.edf.seizures).
    Pairs '[' (onset) and ']' (offset) annotations into SeizureAnnotation instances.
    """
    import wfdb
    path = Path(seizures_path)
    record_path_str = str(path)
    if record_path_str.endswith(".seizures"):
        record_base = record_path_str[:-9]
    else:
        record_base = record_path_str

    ann = wfdb.rdann(record_base, "seizures")
    fs = float(ann.fs) if ann.fs else sfreq

    seizures: List[SeizureAnnotation] = []
    onset_sample: Optional[int] = None
    sz_idx = 1

    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol == "[":
            onset_sample = int(sample)
        elif symbol == "]" and onset_sample is not None:
            offset_sample = int(sample)
            start_s = onset_sample / fs
            end_s = offset_sample / fs
            if end_s > start_s:
                seizures.append(
                    SeizureAnnotation(
                        index=sz_idx,
                        start_sec=float(start_s),
                        end_sec=float(end_s),
                    )
                )
                sz_idx += 1
            onset_sample = None

    return seizures


def load_verified_recording_annotations(
    edf_file_name_or_path: str | Path,
    summary_path: Optional[str | Path] = None,
    seizures_path: Optional[str | Path] = None,
) -> RecordingSummary:
    """
    Load annotations for an EDF recording supporting both *-summary.txt and *.seizures formats.
    Cross-verifies onset/end timestamps if both sources are available.
    """
    edf_name = Path(edf_file_name_or_path).name.lower()
    patient_id = extract_patient_id(edf_name)

    # 1. Try parsing from summary file
    summary_rec: Optional[RecordingSummary] = None
    if summary_path and Path(summary_path).exists():
        parsed = parse_summary_file(summary_path)
        summary_rec = parsed.get(edf_name)

    # 2. Try parsing from .seizures file
    wfdb_seizures: Optional[List[SeizureAnnotation]] = None
    if seizures_path and Path(seizures_path).exists():
        try:
            wfdb_seizures = parse_wfdb_seizures(seizures_path)
        except Exception as e:
            print(f"Warning: wfdb parsing error on {seizures_path}: {e}")

    # If neither found, stop and raise informative error
    if summary_rec is None and wfdb_seizures is None:
        raise FileNotFoundError(
            f"No valid annotation file found for {edf_name}. "
            f"Provide either official summary text ({patient_id}-summary.txt) "
            f"or binary annotation ({edf_name}.seizures)."
        )

    # Cross-verify if both are present
    if summary_rec and wfdb_seizures is not None:
        if len(summary_rec.seizures) != len(wfdb_seizures):
            print(
                f"Notice: Count difference between summary ({len(summary_rec.seizures)}) "
                f"and WFDB .seizures ({len(wfdb_seizures)}) for {edf_name}."
            )
        # Verify first seizure start
        if summary_rec.seizures and wfdb_seizures:
            s1 = summary_rec.seizures[0]
            w1 = wfdb_seizures[0]
            if abs(s1.start_sec - w1.start_sec) > 1.0 or abs(s1.end_sec - w1.end_sec) > 1.0:
                print(
                    f"Warning: Timing offset between summary [{s1.start_sec}s-{s1.end_sec}s] "
                    f"and WFDB [{w1.start_sec}s-{w1.end_sec}s]."
                )

    final_seizures = (
        summary_rec.seizures if summary_rec and summary_rec.has_seizure else (wfdb_seizures or [])
    )
    start_str = summary_rec.start_time_str if summary_rec else None
    end_str = summary_rec.end_time_str if summary_rec else None

    return RecordingSummary(
        file_name=edf_name,
        patient_id=patient_id,
        start_time_str=start_str,
        end_time_str=end_str,
        num_seizures=len(final_seizures),
        seizures=final_seizures,
    )


def parse_summary_file(file_path: str | Path) -> Dict[str, RecordingSummary]:
    """Read and parse a CHB-MIT *-summary.txt file from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_summary_text(text)
