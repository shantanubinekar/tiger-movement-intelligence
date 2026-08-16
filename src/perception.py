"""
perception.py — Detection, quality assessment, and embedding generation.

Developer 2 owns this file (PROJECT_CONTRACT.md Section 18).

Responsibilities:
- detect_subject(record) -> DetectionRecord
  - Try pretrained torchvision model for detection confidence; fallback
    to deterministic demo crop.
  - Compute quality features: blur (Laplacian variance), brightness,
    contrast, crop area fraction, flank visibility proxy.
  - Poor images -> low quality_score, NOT a forced identity.

- generate_embedding(crop_path) -> list[float]
  - Try pretrained ResNet-50 feature extraction; fallback to deterministic
    pseudo-embedding seeded from file hash (512-dim).
  - Embedding is normalized to unit length.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from src.schemas import DetectionRecord, ImageRecord

logger = logging.getLogger(__name__)

# Embedding dimension for the demo fallback
EMBEDDING_DIM = 512

# Try to import perception libraries
_HAS_TORCH = False
_HAS_CV2 = False
_HAS_PIL = False
_resnet_model = None
_resnet_transform = None

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    pass

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    pass

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    _HAS_TORCH = True
except ImportError:
    logger.info("PyTorch/torchvision not available — perception will use demo fallback")


def _get_resnet_model():
    """Lazy-load a pretrained ResNet-50 for feature extraction.
    Returns (model, transform) or (None, None) if unavailable."""
    global _resnet_model, _resnet_transform
    if _resnet_model is not None:
        return _resnet_model, _resnet_transform
    if not _HAS_TORCH:
        return None, None
    try:
        # Use ResNet-50 pretrained on ImageNet — remove final FC layer
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        # Remove the final classification layer; we want features
        # We'll use avgpool output (2048-dim) and project to EMBEDDING_DIM
        _resnet_model = torch.nn.Sequential(*list(model.children())[:-1])
        _resnet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("Loaded pretrained ResNet-50 for feature extraction")
        return _resnet_model, _resnet_transform
    except Exception as e:
        logger.warning("Failed to load ResNet-50: %s — using demo fallback", e)
        return None, None


# ---------------------------------------------------------------------------
# Quality feature computation
# ---------------------------------------------------------------------------

def _seeded_quality_features(seed: str) -> dict[str, float]:
    """Demo fallback: deterministic quality from seed string hash."""
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    return {
        "blur_score": round(((h >> 0) % 100) / 100.0 * 0.5 + 0.3, 4),
        "brightness": round(((h >> 8) % 100) / 100.0 * 0.4 + 0.3, 4),
        "contrast": round(((h >> 16) % 100) / 100.0 * 0.4 + 0.3, 4),
        "crop_area_fraction": 0.7,
        "flank_visibility": round(((h >> 24) % 100) / 100.0 * 0.5 + 0.2, 4),
    }


def _compute_quality_features_raw(image_path: str) -> Optional[dict[str, float]]:
    """Compute image quality features from a real image file using OpenCV or PIL.
    Returns dict with keys: blur_score, brightness, contrast, crop_area_fraction,
    flank_visibility, or None if image loading / processing fails.
    All values normalized to [0, 1].
    """
    if not image_path:
        return None

    path = Path(image_path)
    if not path.is_file():
        return None

    if _HAS_CV2:
        try:
            img = cv2.imread(str(path))
            if img is not None and img.size > 0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                if h > 0 and w > 0:
                    # Blur score: Laplacian variance — higher means sharper
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    # Normalize: typical camera-trap values range 0-500+
                    blur_score = min(1.0, max(0.0, lap_var / 500.0))

                    # Brightness: mean pixel intensity / 255
                    brightness = min(1.0, max(0.0, float(gray.mean()) / 255.0))

                    # Contrast: std of pixel intensities / 128 (half of max range)
                    contrast = min(1.0, max(0.0, float(gray.std()) / 128.0))

                    # Crop area fraction: ratio of image dimensions to standard camera trap (640x480)
                    standard_area = 640.0 * 480.0
                    actual_area = float(h * w)
                    crop_area_fraction = min(1.0, max(0.0, actual_area / standard_area))

                    # Flank visibility proxy: percentage of center region with edge features
                    edges = cv2.Canny(gray, 50, 150)
                    edge_fraction = float(np.count_nonzero(edges)) / float(h * w)
                    center_h, center_w = h // 4, w // 4
                    center_crop = edges[center_h:3*center_h, center_w:3*center_w]
                    if center_crop.size > 0:
                        center_edge_fraction = float(np.count_nonzero(center_crop)) / float(center_crop.size)
                    else:
                        center_edge_fraction = edge_fraction
                    flank_visibility = min(1.0, max(0.0, center_edge_fraction * 5.0))

                    return {
                        "blur_score": round(blur_score, 4),
                        "brightness": round(brightness, 4),
                        "contrast": round(contrast, 4),
                        "crop_area_fraction": round(crop_area_fraction, 4),
                        "flank_visibility": round(flank_visibility, 4),
                    }
        except Exception as e:
            logger.warning("CV2 quality computation failed for %s: %s", image_path, e)

    if _HAS_PIL:
        try:
            with PILImage.open(str(path)) as img:
                gray_img = img.convert("L")
                arr = np.array(gray_img, dtype=np.float64)
                if arr.size > 0 and arr.shape[0] > 0 and arr.shape[1] > 0:
                    h, w = arr.shape
                    brightness = min(1.0, max(0.0, float(arr.mean()) / 255.0))
                    contrast = min(1.0, max(0.0, float(arr.std()) / 128.0))
                    crop_area_fraction = min(1.0, max(0.0, float(arr.size) / (640.0 * 480.0)))

                    blur_score = 0.5
                    flank_visibility = 0.5
                    if h > 1 and w > 1:
                        dx = np.diff(arr, axis=1)
                        dy = np.diff(arr, axis=0)
                        grad_mag = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)
                        blur_score = min(1.0, max(0.0, float(grad_mag.std()) / 50.0))
                        flank_visibility = min(1.0, max(0.0, float(grad_mag.mean()) / 30.0))

                    return {
                        "blur_score": round(blur_score, 4),
                        "brightness": round(brightness, 4),
                        "contrast": round(contrast, 4),
                        "crop_area_fraction": round(crop_area_fraction, 4),
                        "flank_visibility": round(flank_visibility, 4),
                    }
        except Exception as e:
            logger.warning("PIL quality computation failed for %s: %s", image_path, e)

    return None


def _compute_quality_features(image_path: str) -> dict[str, float]:
    """Compute image quality features. Returns dict with keys:
    blur_score, brightness, contrast, crop_area_fraction, flank_visibility.
    All values normalized to [0, 1].

    If a real image file exists and can be loaded, computes real quality features
    (actual Laplacian variance for blur, actual mean/std for brightness/contrast).
    Falls back to deterministic seeded quality if missing or unreadable.
    """
    features = _compute_quality_features_raw(image_path)
    if features is not None:
        return features
    return _seeded_quality_features(image_path or "fallback_image")


def _compute_overall_quality(features: dict[str, float]) -> float:
    """Combine individual quality features into a single score [0, 1].
    Weights are prototype heuristics, NOT scientifically validated."""
    return (
        0.30 * features.get("blur_score", 0.5)
        + 0.15 * features.get("brightness", 0.5)
        + 0.20 * features.get("contrast", 0.5)
        + 0.15 * features.get("crop_area_fraction", 0.5)
        + 0.20 * features.get("flank_visibility", 0.5)
    )


# ---------------------------------------------------------------------------
# Subject detection
# ---------------------------------------------------------------------------

def detect_subject(record: ImageRecord) -> DetectionRecord:
    """Detect a subject in the image and produce quality assessment.

    Parameters
    ----------
    record : ImageRecord
        Must have image_path set.

    Returns
    -------
    DetectionRecord
        With detection confidence, quality score, flank visibility,
        and (optionally) bounding box and crop path.
    """
    image_path = record.image_path
    quality_features = None
    path_exists = False

    if image_path:
        p = Path(image_path)
        if p.is_file():
            path_exists = True
            quality_features = _compute_quality_features_raw(image_path)

    # Only fall back to seeded pseudo-random values if the file genuinely
    # can't be loaded (missing file, corrupt image, etc.)
    if quality_features is None:
        quality_features = _seeded_quality_features(record.image_id)

    overall_quality = _compute_overall_quality(quality_features)
    flank_vis = quality_features.get("flank_visibility", 0.5)

    # Detection confidence: for Phase 1, use quality as a proxy for
    # detection confidence (a real detector like YOLO would replace this).
    # If image quality is very low, detection confidence is low.
    detection_conf = min(0.95, overall_quality * 0.8 + 0.15) if overall_quality > 0.2 else 0.1

    # For crop path: in Phase 1, use the original image as the "crop"
    # (a real detector would produce a bounding-box crop here).
    crop_path = image_path if path_exists else None

    # Bounding box: demo placeholder centered in image
    bbox = (0.1, 0.1, 0.8, 0.8) if detection_conf > 0.3 else None

    return DetectionRecord(
        image_id=record.image_id,
        species="tiger" if detection_conf > 0.3 else None,
        bbox=bbox,
        detection_confidence=round(detection_conf, 4),
        quality_score=round(overall_quality, 4),
        flank_visibility=round(flank_vis, 4),
        crop_path=crop_path,
    )


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def _deterministic_embedding(seed: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Generate a deterministic pseudo-embedding seeded from a string.
    Used as demo fallback when no pretrained model is available.
    The embedding is normalized to unit length."""
    h = hashlib.sha256(seed.encode()).digest()
    # Expand hash to fill the embedding dimension
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    vec = rng.randn(dim).astype(np.float64)
    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def generate_embedding(crop_path: str) -> list[float]:
    """Generate a feature embedding for a crop image.

    Attempts to use a pretrained ResNet-50. Falls back to a deterministic
    pseudo-embedding seeded from the file hash if the model is unavailable
    or the file doesn't exist.

    Parameters
    ----------
    crop_path : str
        Path to the cropped image file.

    Returns
    -------
    list[float]
        Unit-normalized embedding vector (EMBEDDING_DIM dimensions).
    """
    path = Path(crop_path) if crop_path else None

    # Try real model
    if _HAS_TORCH and _HAS_PIL and path and path.exists():
        model, transform = _get_resnet_model()
        if model is not None and transform is not None:
            try:
                img = PILImage.open(str(path)).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    features = model(tensor).squeeze()
                # features is 2048-dim from ResNet-50 avgpool
                vec = features.numpy().astype(np.float64)
                # Project to EMBEDDING_DIM if needed
                if len(vec) > EMBEDDING_DIM:
                    # Simple truncation + renormalization
                    vec = vec[:EMBEDDING_DIM]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()
            except Exception as e:
                logger.warning(
                    "ResNet-50 embedding failed for %s: %s — using demo fallback",
                    crop_path, e,
                )

    # Demo fallback: deterministic embedding from file path hash
    seed = crop_path if crop_path else "unknown_crop"
    logger.debug("Using deterministic demo embedding for %s", seed)
    return _deterministic_embedding(seed)
