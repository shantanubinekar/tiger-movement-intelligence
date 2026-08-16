"""
tests/test_perception.py — Perception module tests.

Developer 2 owns this file (per MASTER_PROMPTS.md testing requirements).

Tests:
- Quality features return sane ranges (0–1) on a real and a synthetic image.
- Embedding generation produces unit-normalized vectors of correct dimension.
- Detection produces valid DetectionRecord.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.schemas import DataMode, ImageRecord
from src.perception import (
    EMBEDDING_DIM,
    _compute_quality_features,
    _deterministic_embedding,
    detect_subject,
    generate_embedding,
    generate_local_embedding,
)


DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"


class TestQualityFeatures:
    """Quality features return sane ranges (0–1)."""

    def test_quality_features_on_real_image(self):
        """Test quality features on a real demo image (if available)."""
        img_path = DEMO_DIR / "demo_img_000.jpg"
        if not img_path.exists():
            pytest.skip("Demo image not available")

        features = _compute_quality_features(str(img_path))

        for key in ["blur_score", "brightness", "contrast",
                     "crop_area_fraction", "flank_visibility"]:
            assert key in features, f"Missing feature: {key}"
            assert 0.0 <= features[key] <= 1.0, (
                f"Feature {key} = {features[key]} out of [0, 1] range"
            )

    def test_quality_features_on_blank_image(self):
        """Test quality features on the blank/dark demo image."""
        img_path = DEMO_DIR / "demo_img_004.jpg"
        if not img_path.exists():
            pytest.skip("Blank demo image not available")

        features = _compute_quality_features(str(img_path))

        for key in ["blur_score", "brightness", "contrast",
                     "crop_area_fraction", "flank_visibility"]:
            assert 0.0 <= features[key] <= 1.0

        # Blank image should have low brightness and low edge density
        assert features["brightness"] < 0.3, (
            f"Blank image brightness unexpectedly high: {features['brightness']}"
        )

    def test_quality_features_on_missing_file(self):
        """Quality features should return defaults for missing files."""
        features = _compute_quality_features("/nonexistent/path.jpg")
        for key in ["blur_score", "brightness", "contrast",
                     "crop_area_fraction", "flank_visibility"]:
            assert 0.0 <= features[key] <= 1.0

    def test_real_vs_fallback_quality_scoring(self, tmp_path):
        """Confirm real image features (Laplacian blur, brightness/contrast)
        differ based on pixel data and fall back cleanly when file is missing or corrupt."""
        import cv2

        # 1. Create a sharp image (high gradient edges)
        sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
        sharp_img[::4, :] = 255
        sharp_path = tmp_path / "sharp.jpg"
        cv2.imwrite(str(sharp_path), sharp_img)

        # 2. Create a blurred version of the same image
        blur_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
        blur_path = tmp_path / "blurred.jpg"
        cv2.imwrite(str(blur_path), blur_img)

        # 3. Create a bright image
        bright_img = np.full((200, 200, 3), 240, dtype=np.uint8)
        bright_path = tmp_path / "bright.jpg"
        cv2.imwrite(str(bright_path), bright_img)

        sharp_feat = _compute_quality_features(str(sharp_path))
        blur_feat = _compute_quality_features(str(blur_path))
        bright_feat = _compute_quality_features(str(bright_path))

        # Real OpenCV Laplacian variance should make sharp image have higher blur score than blurred image
        assert sharp_feat["blur_score"] > blur_feat["blur_score"], (
            f"Expected sharp blur_score ({sharp_feat['blur_score']}) > blurred ({blur_feat['blur_score']})"
        )

        # Real brightness feature should reflect pixel intensity
        assert bright_feat["brightness"] > blur_feat["brightness"]

        # 4. Fallback test: missing image path
        missing_record = ImageRecord(
            image_id="img_missing_fallback",
            image_path=str(tmp_path / "does_not_exist.jpg"),
            file_hash="missing_hash",
            data_mode=DataMode.DEMO,
        )
        missing_detection = detect_subject(missing_record)
        assert missing_detection.crop_path is None
        assert 0.0 <= missing_detection.quality_score <= 1.0

        # 5. Fallback test: corrupt file (not an image)
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a valid image content")
        corrupt_record = ImageRecord(
            image_id="img_corrupt_fallback",
            image_path=str(corrupt_path),
            file_hash="corrupt_hash",
            data_mode=DataMode.DEMO,
        )
        corrupt_detection = detect_subject(corrupt_record)
        assert 0.0 <= corrupt_detection.quality_score <= 1.0



class TestDetection:
    """detect_subject produces valid DetectionRecord."""

    def test_detect_on_real_image(self):
        """Detection on a real demo image should produce valid record."""
        img_path = DEMO_DIR / "demo_img_000.jpg"
        if not img_path.exists():
            pytest.skip("Demo image not available")

        record = ImageRecord(
            image_id="demo_img_000",
            image_path=str(img_path),
            file_hash="test_hash",
            data_mode=DataMode.DEMO,
        )
        detection = detect_subject(record)

        assert detection.image_id == "demo_img_000"
        assert 0.0 <= detection.detection_confidence <= 1.0
        assert 0.0 <= detection.quality_score <= 1.0
        assert 0.0 <= detection.flank_visibility <= 1.0

    def test_detect_on_missing_image(self):
        """Detection on a missing image should use fallback (not crash)."""
        record = ImageRecord(
            image_id="nonexistent",
            image_path="/nonexistent/path.jpg",
            file_hash="test_hash",
            data_mode=DataMode.DEMO,
        )
        detection = detect_subject(record)

        assert detection.image_id == "nonexistent"
        assert 0.0 <= detection.detection_confidence <= 1.0
        assert 0.0 <= detection.quality_score <= 1.0


class TestEmbedding:
    """Embedding generation produces unit-normalized vectors."""

    def test_deterministic_embedding_correct_dim(self):
        """Deterministic embedding has correct dimension."""
        emb = _deterministic_embedding("test_seed")
        assert len(emb) == EMBEDDING_DIM

    def test_deterministic_embedding_unit_normalized(self):
        """Deterministic embedding is unit-normalized."""
        emb = _deterministic_embedding("test_seed")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-6, f"Embedding norm = {norm}, expected ~1.0"

    def test_deterministic_embedding_reproducible(self):
        """Same seed produces same embedding."""
        emb1 = _deterministic_embedding("same_seed")
        emb2 = _deterministic_embedding("same_seed")
        assert emb1 == emb2

    def test_deterministic_embedding_different_seeds(self):
        """Different seeds produce different embeddings."""
        emb1 = _deterministic_embedding("seed_a")
        emb2 = _deterministic_embedding("seed_b")
        assert emb1 != emb2

    def test_generate_embedding_on_real_image(self):
        """generate_embedding on a real image returns valid vector."""
        img_path = DEMO_DIR / "demo_img_000.jpg"
        if not img_path.exists():
            pytest.skip("Demo image not available")

        emb = generate_embedding(str(img_path))
        assert len(emb) > 0
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4, f"Embedding norm = {norm}, expected ~1.0"

    def test_generate_embedding_on_missing_file(self):
        """generate_embedding on missing file uses deterministic fallback."""
        emb = generate_embedding("/nonexistent/path.jpg")
        assert len(emb) == EMBEDDING_DIM
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-6

    def test_generate_local_embedding(self):
        """generate_local_embedding returns unit-normalized local vector."""
        emb = generate_local_embedding("/nonexistent/path.jpg")
        assert len(emb) == EMBEDDING_DIM
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-6

