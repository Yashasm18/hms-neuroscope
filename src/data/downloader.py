"""
Resumable downloader utility for CHB-MIT PhysioNet dataset.
Supports range requests, disk space checking, and progress tracking.
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import requests

from config import (
    PHYSIONET_BASE_URL,
    PHYSIONET_RECORDS_WITH_SEIZURES_URL,
    RAW_DATA_DIR,
    ANNOTATIONS_DIR,
)


class PhysioNetDownloader:
    """Manages selective, resumable downloads of CHB-MIT recordings and summaries."""

    def __init__(
        self,
        base_url: str = PHYSIONET_BASE_URL,
        raw_dir: Path = RAW_DATA_DIR,
        annotations_dir: Path = ANNOTATIONS_DIR,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.raw_dir = Path(raw_dir)
        self.annotations_dir = Path(annotations_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_disk_space(target_path: Path, min_free_gb: float = 1.0) -> Tuple[bool, float, float]:
        """
        Check if sufficient disk space is available.
        Returns: (has_space, free_gb, total_gb)
        """
        try:
            total, used, free = shutil.disk_usage(str(target_path.resolve()))
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            return (free_gb >= min_free_gb, free_gb, total_gb)
        except Exception:
            return (True, 0.0, 0.0)

    @staticmethod
    def check_internet_access(timeout: float = 12.0) -> bool:
        """Verify reachability of PhysioNet server using streaming GET."""
        try:
            r = requests.get(PHYSIONET_RECORDS_WITH_SEIZURES_URL, stream=True, timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def fetch_records_with_seizures(self) -> List[str]:
        """Fetch list of all recordings known to contain seizures (with local cache fallback)."""
        cached_file = self.annotations_dir / "RECORDS-WITH-SEIZURES"
        try:
            r = requests.get(PHYSIONET_RECORDS_WITH_SEIZURES_URL, timeout=15)
            if r.status_code == 200 and not r.text.strip().startswith("<"):
                cached_file.write_text(r.text, encoding="utf-8")
                lines = [line.strip() for line in r.text.splitlines() if line.strip()]
                return lines
        except Exception as e:
            print(f"Warning: Remote fetch of RECORDS-WITH-SEIZURES failed: {e}")

        # Check local cache fallback
        if cached_file.exists():
            text = cached_file.read_text(encoding="utf-8", errors="replace")
            return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("<")]

        return []

    def download_summary(self, patient_id: str) -> Optional[Path]:
        """
        Download patient summary file (e.g. chb01-summary.txt).
        Returns Path to downloaded file or None on failure.
        """
        patient_id = patient_id.lower()
        file_name = f"{patient_id}-summary.txt"
        dest_path = self.annotations_dir / file_name

        # Return existing if already downloaded
        if dest_path.exists() and dest_path.stat().st_size > 100:
            return dest_path

        url = f"{self.base_url}{patient_id}/{file_name}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                dest_path.write_bytes(r.content)
                return dest_path
        except Exception as e:
            print(f"Failed to download summary for {patient_id}: {e}")
        return None

    def download_edf(
        self,
        relative_record_path: str,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        chunk_size: int = 1024 * 256,
        max_retries: int = 8,
    ) -> Optional[Path]:
        """
        Resumably download an EDF recording from PhysioNet with automatic retry loop.
        relative_record_path: e.g. 'chb01/chb01_03.edf' or 'chb01_03.edf'
        progress_callback: callable(downloaded_bytes, total_bytes, percent)
        """
        rel_path = relative_record_path.strip().replace("\\", "/")
        parts = rel_path.split("/")
        if len(parts) == 1:
            patient_id = parts[0].split("_")[0].lower()
            file_name = parts[0]
            url = f"{self.base_url}{patient_id}/{file_name}"
        else:
            patient_id = parts[0].lower()
            file_name = parts[1]
            url = f"{self.base_url}{patient_id}/{file_name}"

        dest_folder = self.raw_dir / patient_id
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_path = dest_folder / file_name

        # Query HEAD for total size
        total_size = 0
        accept_ranges = True
        try:
            head_resp = requests.head(url, timeout=15)
            if head_resp.status_code == 200:
                total_size = int(head_resp.headers.get("content-length", 0))
                accept_ranges = "bytes" in head_resp.headers.get("accept-ranges", "").lower()
        except Exception as e:
            print(f"HEAD check failed: {e}. Will attempt streaming GET directly.")

        # Check existing file size
        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        if total_size > 0 and existing_size == total_size:
            if progress_callback:
                progress_callback(total_size, total_size, 100.0)
            return dest_path

        # Check disk space
        needed_bytes = max(0, total_size - existing_size)
        has_space, free_gb, _ = self.check_disk_space(
            dest_folder, min_free_gb=max(0.2, needed_bytes / (1024**3))
        )
        if not has_space:
            raise OSError(f"Insufficient disk space on {dest_folder}. Free: {free_gb:.2f} GB")

        # Retry loop for resumable download
        for attempt in range(1, max_retries + 1):
            existing_size = dest_path.stat().st_size if dest_path.exists() else 0
            if total_size > 0 and existing_size >= total_size:
                if progress_callback:
                    progress_callback(total_size, total_size, 100.0)
                return dest_path

            headers = {}
            mode = "wb"
            downloaded = 0
            if existing_size > 0 and accept_ranges:
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"
                downloaded = existing_size

            try:
                with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                    if r.status_code not in (200, 206):
                        print(f"Server returned HTTP {r.status_code} for {url}")
                        continue
                    if total_size == 0:
                        total_size = int(r.headers.get("content-length", 0)) + downloaded

                    with open(dest_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if progress_callback and total_size > 0:
                                    pct = min(100.0, (downloaded / total_size) * 100.0)
                                    progress_callback(downloaded, total_size, pct)

                # If reached here and file complete
                final_size = dest_path.stat().st_size
                if total_size > 0 and final_size >= total_size:
                    return dest_path
                elif total_size == 0 and final_size > 0:
                    return dest_path

            except Exception as e:
                print(f"[Attempt {attempt}/{max_retries}] Download interrupted at {downloaded/(1024*1024):.1f} MB: {e}. Resuming...")

        if dest_path.exists() and dest_path.stat().st_size > 1000:
            return dest_path
        return None
