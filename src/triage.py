"""
triage.py — Blank / nonblank / uncertain image triage.

Developer 2 owns this file (PROJECT_CONTRACT.md Section 18).

Responsibilities:
- Classify each image as blank, nonblank, or uncertain using a
  brightness + edge-density heuristic.
- Blank images are flagged (quarantined), NEVER deleted.
- Demo fallback: deterministic triage based on filename or simple
  pixel statistics when OpenCV is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.schemas import ImageRecord, TriageRecord, TriageStatus

logger = logging.getLogger(__name__)

# Configurable thresholds for the brightness+edge heuristic.
# These are prototype heuristics, NOT scientifically validated parameters.
BLANK_EDGE_THRESHOLD = 5.0       # Laplacian std below this → likely blank
BLANK_BRIGHTNESS_STD = 15.0      # Brightness std below this → likely blank
UNCERTAIN_EDGE_THRESHOLD = 15.0  # Between blank and this → uncertain
UNCERTAIN_BRIGHTNESS_STD = 30.0  # Between blank and this → uncertain

# Try to import image processing libraries
_HAS_CV2 = False
_HAS_PIL = False

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    logger.info("OpenCV not available — triage will use PIL fallback or demo mode")

if not _HAS_CV2:
    try:
        from PIL import Image as PILImage
        import numpy as np  # noqa: F811 — same import, different branch
        _HAS_PIL = True
    except ImportError:
        logger.info("PIL not available — triage will use demo fallback only")


def _compute_image_stats_cv2(image_path: str) -> tuple[float, float]:
    """Compute edge density (Laplacian std) and brightness std using OpenCV."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, 0.0

    # Edge density via Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    edge_density = float(laplacian.std())

    # Brightness variation
    brightness_std = float(img.std())

    return edge_density, brightness_std


def _compute_image_stats_pil(image_path: str) -> tuple[float, float]:
    """Fallback: compute approximate edge density and brightness std using PIL."""
    try:
        img = PILImage.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float64)

        # Approximate edge density using simple difference filter
        if arr.shape[0] > 1 and arr.shape[1] > 1:
            dx = np.diff(arr, axis=1)
            dy = np.diff(arr, axis=0)
            edge_density = float((dx.std() + dy.std()) / 2.0)
        else:
            edge_density = 0.0

        brightness_std = float(arr.std())
        return edge_density, brightness_std
    except Exception as e:
        logger.warning("PIL stats computation failed for %s: %s", image_path, e)
        return 0.0, 0.0


def _classify_from_stats(edge_density: float, brightness_std: float) -> tuple[TriageStatus, float, float]:
    """Classify image as blank/nonblank/uncertain from statistics.

    Returns (status, blank_probability, subject_probability).
    """
    is_low_edge = edge_density < BLANK_EDGE_THRESHOLD
    is_low_brightness = brightness_std < BLANK_BRIGHTNESS_STD

    if is_low_edge and is_low_brightness:
        # Very likely a blank frame
        blank_prob = min(0.95, 1.0 - (edge_density / BLANK_EDGE_THRESHOLD) * 0.3
                         - (brightness_std / BLANK_BRIGHTNESS_STD) * 0.2)
        return TriageStatus.BLANK, blank_prob, 1.0 - blank_prob

    is_moderate_edge = edge_density < UNCERTAIN_EDGE_THRESHOLD
    is_moderate_brightness = brightness_std < UNCERTAIN_BRIGHTNESS_STD

    if is_moderate_edge or is_moderate_brightness:
        # Uncertain — borderline case
        # Scale uncertainty inversely with edge/brightness
        edge_ratio = min(1.0, edge_density / UNCERTAIN_EDGE_THRESHOLD)
        bright_ratio = min(1.0, brightness_std / UNCERTAIN_BRIGHTNESS_STD)
        subject_prob = (edge_ratio + bright_ratio) / 2.0
        blank_prob = 1.0 - subject_prob
        return TriageStatus.UNCERTAIN, blank_prob, subject_prob

    # Likely contains a subject
    # Higher edge density and brightness variance → more likely nonblank
    edge_conf = min(1.0, edge_density / (UNCERTAIN_EDGE_THRESHOLD * 2))
    bright_conf = min(1.0, brightness_std / (UNCERTAIN_BRIGHTNESS_STD * 2))
    subject_prob = min(0.95, (edge_conf + bright_conf) / 2.0 + 0.4)
    return TriageStatus.NONBLANK, 1.0 - subject_prob, subject_prob


def _demo_triage(image_id: str) -> tuple[TriageStatus, float, float]:
    """Deterministic demo triage based on image_id for when no image
    processing libraries are available."""
    # Use hash of image_id for deterministic assignment
    import hashlib
    h = int(hashlib.sha256(image_id.encode()).hexdigest(), 16)
    mod = h % 10

    if mod < 2:  # 20% blank
        return TriageStatus.BLANK, 0.90, 0.10
    elif mod < 3:  # 10% uncertain
        return TriageStatus.UNCERTAIN, 0.45, 0.55
    else:  # 70% nonblank
        return TriageStatus.NONBLANK, 0.10, 0.90


def triage_image(record: ImageRecord) -> TriageRecord:
    """Classify a single image as blank, nonblank, or uncertain.

    Uses brightness + edge-density heuristic when OpenCV or PIL is
    available. Falls back to deterministic demo classification otherwise.

    Parameters
    ----------
    record : ImageRecord
        The image record (must have image_path set).

    Returns
    -------
    TriageRecord
        Contains blank_probability, subject_probability, triage_status.
    """
    image_path = record.image_path

    # Check if image file exists
    if not Path(image_path).exists():
        logger.warning("Image file not found: %s — using demo triage", image_path)
        status, blank_prob, subj_prob = _demo_triage(record.image_id)
        return TriageRecord(
            image_id=record.image_id,
            blank_probability=blank_prob,
            subject_probability=subj_prob,
            triage_status=status,
        )

    # Try real image analysis
    edge_density = 0.0
    brightness_std = 0.0
    used_real_analysis = False

    if _HAS_CV2:
        edge_density, brightness_std = _compute_image_stats_cv2(image_path)
        used_real_analysis = True
    elif _HAS_PIL:
        edge_density, brightness_std = _compute_image_stats_pil(image_path)
        used_real_analysis = True

    if used_real_analysis and (edge_density > 0 or brightness_std > 0):
        status, blank_prob, subj_prob = _classify_from_stats(edge_density, brightness_std)
        logger.debug(
            "Triage %s: edge=%.2f, bright_std=%.2f -> %s (blank=%.2f, subj=%.2f)",
            record.image_id, edge_density, brightness_std, status, blank_prob, subj_prob,
        )
    else:
        # Fallback to demo triage
        status, blank_prob, subj_prob = _demo_triage(record.image_id)
        logger.debug("Triage %s: demo fallback -> %s", record.image_id, status)

    return TriageRecord(
        image_id=record.image_id,
        blank_probability=round(blank_prob, 4),
        subject_probability=round(subj_prob, 4),
        triage_status=status,
    )
