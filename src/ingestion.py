"""
ingestion.py — Image folder ingestion for the perception pipeline.

Developer 2 owns this file (PROJECT_CONTRACT.md Section 18).

Responsibilities:
- Recursively scan a directory for image files
- Compute file hash (SHA-256) per image
- Read sidecar metadata.csv if present (station_id, lat, lon, timestamp, camera_status)
- Fallback: assign synthetic demo metadata and mark data_mode="demo"
- NEVER delete originals
"""

from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.schemas import CameraStatus, DataMode, ImageRecord

logger = logging.getLogger(__name__)

# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file. Reads in 64 KB chunks to handle
    large files without excessive memory use."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_metadata_csv(csv_path: Path) -> dict[str, dict]:
    """Load a sidecar metadata.csv into a dict keyed by image_id.
    Returns empty dict if file doesn't exist or can't be parsed."""
    metadata: dict[str, dict] = {}
    if not csv_path.exists():
        return metadata
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_id", "").strip()
                if not image_id:
                    continue
                metadata[image_id] = {
                    "station_id": row.get("station_id", "").strip() or None,
                    "latitude": _safe_float(row.get("latitude")),
                    "longitude": _safe_float(row.get("longitude")),
                    "timestamp": _safe_datetime(row.get("timestamp")),
                    "camera_status": _safe_camera_status(row.get("camera_status")),
                    "data_mode": _safe_data_mode(row.get("data_mode")),
                }
    except Exception as e:
        logger.warning("Failed to parse metadata CSV %s: %s", csv_path, e)
    return metadata


def _safe_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _safe_datetime(val: Optional[str]) -> Optional[datetime]:
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    # Try ISO format first, then common camera-trap formats
    for fmt in (None, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            if fmt is None:
                return datetime.fromisoformat(val).replace(tzinfo=timezone.utc)
            return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    logger.warning("Could not parse timestamp: %s", val)
    return None


def _safe_camera_status(val: Optional[str]) -> CameraStatus:
    if val is None:
        return CameraStatus.UNKNOWN
    val = val.strip().lower()
    try:
        return CameraStatus(val)
    except ValueError:
        return CameraStatus.UNKNOWN


def _safe_data_mode(val: Optional[str]) -> DataMode:
    if val is None:
        return DataMode.DEMO
    val = val.strip().lower()
    try:
        return DataMode(val)
    except ValueError:
        return DataMode.DEMO


# --------------------------------------------------------------------------
# Synthetic demo metadata — used when no metadata.csv is found.
# Covers Pench Tiger Reserve approximate coordinates.
# --------------------------------------------------------------------------

_DEMO_STATIONS = [
    {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290},
    {"station_id": "STATION_B2", "latitude": 21.702, "longitude": 79.315},
    {"station_id": "STATION_C3", "latitude": 21.664, "longitude": 79.278},
    {"station_id": "STATION_D4_BUFFER", "latitude": 21.640, "longitude": 79.260},
]


def _generate_demo_metadata(image_id: str, index: int) -> dict:
    """Generate synthetic metadata for a single image when no CSV is available.
    Uses a deterministic assignment so results are reproducible."""
    station = _DEMO_STATIONS[index % len(_DEMO_STATIONS)]
    base_ts = datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc)
    offset_hours = index * 36  # spread observations across time
    return {
        "station_id": station["station_id"],
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "timestamp": base_ts.replace(hour=(6 + index * 3) % 24)
                     .__add__(__import__("datetime").timedelta(days=index * 2)),
        "camera_status": CameraStatus.ACTIVE,
        "data_mode": DataMode.DEMO,
    }


def ingest_folder(path: str) -> list[ImageRecord]:
    """Ingest all images from a directory, returning an ImageRecord per file.

    Parameters
    ----------
    path : str
        Path to the directory to scan (recursively).

    Returns
    -------
    list[ImageRecord]
        One record per image file found, enriched with metadata from
        a sidecar CSV if present, otherwise demo-generated metadata.
    """
    folder = Path(path)
    if not folder.exists():
        logger.warning("Ingestion path does not exist: %s — returning empty list", path)
        return []

    # Discover image files (sorted for determinism)
    image_files = sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.info("No image files found in %s", path)
        return []

    # Try to load sidecar metadata
    csv_candidates = [folder / "metadata.csv", folder / "meta.csv"]
    metadata_map: dict[str, dict] = {}
    for csv_path in csv_candidates:
        if csv_path.exists():
            metadata_map = _load_metadata_csv(csv_path)
            logger.info("Loaded metadata from %s (%d entries)", csv_path, len(metadata_map))
            break

    has_metadata = len(metadata_map) > 0

    records: list[ImageRecord] = []
    for idx, img_path in enumerate(image_files):
        image_id = img_path.stem
        file_hash = _compute_file_hash(img_path)

        if image_id in metadata_map:
            meta = metadata_map[image_id]
            record = ImageRecord(
                image_id=image_id,
                image_path=str(img_path),
                file_hash=file_hash,
                station_id=meta.get("station_id"),
                latitude=meta.get("latitude"),
                longitude=meta.get("longitude"),
                timestamp=meta.get("timestamp"),
                camera_status=meta.get("camera_status", CameraStatus.UNKNOWN),
                processing_status="ingested",
                data_mode=meta.get("data_mode", DataMode.DEMO),
            )
        elif has_metadata:
            # Metadata CSV exists but this image isn't in it
            logger.warning("Image %s not found in metadata CSV — marking as demo", image_id)
            record = ImageRecord(
                image_id=image_id,
                image_path=str(img_path),
                file_hash=file_hash,
                processing_status="ingested",
                data_mode=DataMode.DEMO,
            )
        else:
            # No metadata CSV at all — generate synthetic metadata
            demo_meta = _generate_demo_metadata(image_id, idx)
            record = ImageRecord(
                image_id=image_id,
                image_path=str(img_path),
                file_hash=file_hash,
                station_id=demo_meta["station_id"],
                latitude=demo_meta["latitude"],
                longitude=demo_meta["longitude"],
                timestamp=demo_meta["timestamp"],
                camera_status=demo_meta["camera_status"],
                processing_status="ingested",
                data_mode=DataMode.DEMO,
            )

        records.append(record)

    logger.info("Ingested %d images from %s (metadata source: %s)",
                len(records), path, "CSV" if has_metadata else "synthetic demo")
    return records
