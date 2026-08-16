"""
tests/test_gating.py — Evidence gating tests.

Developer 2 owns this file (per MASTER_PROMPTS.md testing requirements).

Tests:
- E >= 0.80 with no conflicts -> trusted_match, update_history=True
- Low E -> unknown/ambiguous, update_history=False
- Severe quality failure -> insufficient_evidence, update_history=False
- Camera-relocation / poor-quality / missing-metadata scenarios each
  produce the correct reason code.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.schemas import (
    CameraStatus,
    IdentityCandidate,
    IdentityDecision,
    IdentityDecisionState,
    ReasonCode,
)
from src.gating import (
    THRESHOLD_AMBIGUOUS,
    THRESHOLD_TRUSTED,
    check_calibration,
    compute_evidence,
    make_identity_decision,
)


def _make_candidate(
    visual_score=0.9,
    quality_score=0.8,
    spatial_feasibility=0.8,
    temporal_feasibility=0.8,
    history_consistency=0.7,
    image_id="test_img",
    candidate_identity="T01",
    rank=1,
) -> IdentityCandidate:
    """Helper to create a candidate with configurable scores."""
    total = (
        0.55 * visual_score
        + 0.15 * quality_score
        + 0.15 * spatial_feasibility
        + 0.10 * temporal_feasibility
        + 0.05 * history_consistency
    )
    return IdentityCandidate(
        image_id=image_id,
        candidate_identity=candidate_identity,
        rank=rank,
        visual_score=visual_score,
        quality_score=quality_score,
        spatial_feasibility=spatial_feasibility,
        temporal_feasibility=temporal_feasibility,
        history_consistency=history_consistency,
        total_evidence=round(min(1.0, max(0.0, total)), 4),
    )


class TestEvidenceComputation:
    """compute_evidence returns weighted combination."""

    def test_perfect_scores(self):
        c = _make_candidate(1.0, 1.0, 1.0, 1.0, 1.0)
        E = compute_evidence(c)
        assert abs(E - 1.0) < 0.01

    def test_zero_scores(self):
        c = _make_candidate(0.0, 0.0, 0.0, 0.0, 0.0)
        E = compute_evidence(c)
        assert abs(E - 0.0) < 0.01

    def test_mixed_scores(self):
        c = _make_candidate(0.8, 0.6, 0.7, 0.5, 0.4)
        E = compute_evidence(c)
        expected = 0.55 * 0.8 + 0.15 * 0.6 + 0.15 * 0.7 + 0.10 * 0.5 + 0.05 * 0.4
        assert abs(E - expected) < 0.01

    def test_evidence_in_valid_range(self):
        """Evidence score should always be in [0, 1]."""
        import random
        random.seed(42)
        for _ in range(50):
            c = _make_candidate(
                visual_score=random.random(),
                quality_score=random.random(),
                spatial_feasibility=random.random(),
                temporal_feasibility=random.random(),
                history_consistency=random.random(),
            )
            E = compute_evidence(c)
            assert 0.0 <= E <= 1.0


class TestTrustedMatch:
    """E >= 0.80 with no major conflict -> trusted_match, update_history=True."""

    def test_high_evidence_no_conflict_is_trusted(self):
        """High scores across all dimensions -> trusted_match."""
        top = _make_candidate(0.95, 0.85, 0.9, 0.85, 0.8)
        second = _make_candidate(0.5, 0.5, 0.5, 0.5, 0.5, candidate_identity="T02", rank=2)

        decision = make_identity_decision(
            [top, second],
            context={
                "quality_score": 0.85,
                "flank_visibility": 0.7,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert decision.decision == IdentityDecisionState.TRUSTED_MATCH
        assert decision.update_history is True
        assert decision.identity_id == "T01"
        assert ReasonCode.HIGH_CONFIDENCE_MATCH in decision.reason_codes

    def test_trusted_match_has_required_fields(self):
        """Every trusted_match decision has confidence, reason_codes,
        evidence_summary, top_candidates, and update_history."""
        top = _make_candidate(0.95, 0.85, 0.9, 0.85, 0.8)
        second = _make_candidate(0.5, 0.5, 0.5, 0.5, 0.5, candidate_identity="T02", rank=2)

        decision = make_identity_decision(
            [top, second],
            context={
                "quality_score": 0.85,
                "flank_visibility": 0.7,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert decision.confidence > 0
        assert len(decision.reason_codes) > 0
        assert len(decision.evidence_summary) > 0
        assert len(decision.top_candidates) > 0
        assert decision.update_history is True


class TestAmbiguousAndUnknown:
    """Low E -> unknown/ambiguous, update_history=False."""

    def test_moderate_evidence_is_ambiguous(self):
        """Evidence between thresholds -> ambiguous_review."""
        top = _make_candidate(0.65, 0.6, 0.6, 0.6, 0.5)
        second = _make_candidate(0.55, 0.5, 0.5, 0.5, 0.4, candidate_identity="T02", rank=2)

        decision = make_identity_decision(
            [top, second],
            context={
                "quality_score": 0.6,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert decision.decision in (
            IdentityDecisionState.AMBIGUOUS_REVIEW,
            IdentityDecisionState.UNKNOWN,
        )
        assert decision.update_history is False

    def test_low_evidence_is_unknown(self):
        """Very low evidence -> unknown."""
        top = _make_candidate(0.2, 0.2, 0.2, 0.2, 0.1)

        decision = make_identity_decision(
            [top],
            context={
                "quality_score": 0.2,
                "flank_visibility": 0.3,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert decision.decision == IdentityDecisionState.UNKNOWN
        assert decision.update_history is False

    def test_no_candidates_is_unknown(self):
        """Empty candidate list -> unknown."""
        decision = make_identity_decision([], context={"image_id": "empty_test"})
        assert decision.decision == IdentityDecisionState.UNKNOWN
        assert decision.update_history is False


class TestInsufficientEvidence:
    """Severe quality failure -> insufficient_evidence, update_history=False."""

    def test_camera_inactive_forces_insufficient(self):
        """Inactive camera -> insufficient_evidence."""
        top = _make_candidate(0.9, 0.8, 0.8, 0.8, 0.7)

        decision = make_identity_decision(
            [top],
            context={
                "quality_score": 0.8,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.INACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert decision.decision == IdentityDecisionState.INSUFFICIENT_EVIDENCE
        assert decision.update_history is False

    def test_very_low_quality_forces_insufficient(self):
        """Extremely low quality -> insufficient_evidence."""
        top = _make_candidate(0.9, 0.01, 0.8, 0.8, 0.7)

        decision = make_identity_decision(
            [top],
            context={
                "quality_score": 0.01,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert decision.decision == IdentityDecisionState.INSUFFICIENT_EVIDENCE
        assert decision.update_history is False

    def test_missing_location_and_timestamp_forces_insufficient(self):
        """Missing both location AND timestamp -> insufficient_evidence."""
        top = _make_candidate(0.9, 0.8, 0.8, 0.8, 0.7)

        decision = make_identity_decision(
            [top],
            context={
                "quality_score": 0.8,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.ACTIVE,
                # No timestamp, no lat/lon
            },
        )

        assert decision.decision == IdentityDecisionState.INSUFFICIENT_EVIDENCE
        assert decision.update_history is False


class TestReasonCodes:
    """Camera-relocation / poor-quality / missing-metadata scenarios each
    produce the correct reason code."""

    def test_camera_relocation_produces_reason_code(self):
        """Camera relocated -> CAMERA_RELOCATED reason code."""
        top = _make_candidate(0.85, 0.7, 0.7, 0.7, 0.6)
        second = _make_candidate(0.4, 0.5, 0.5, 0.5, 0.4, candidate_identity="T02", rank=2)

        decision = make_identity_decision(
            [top, second],
            context={
                "quality_score": 0.7,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.RELOCATED,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert ReasonCode.CAMERA_RELOCATED in decision.reason_codes
        # Camera relocation is a major conflict — should downgrade from trusted
        assert decision.update_history is False

    def test_poor_quality_produces_reason_code(self):
        """Very poor quality -> POOR_IMAGE_QUALITY reason code."""
        top = _make_candidate(0.85, 0.05, 0.7, 0.7, 0.6)
        second = _make_candidate(0.4, 0.5, 0.5, 0.5, 0.4, candidate_identity="T02", rank=2)

        decision = make_identity_decision(
            [top, second],
            context={
                "quality_score": 0.05,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert ReasonCode.POOR_IMAGE_QUALITY in decision.reason_codes

    def test_missing_timestamp_produces_reason_code(self):
        """Missing timestamp -> MISSING_TIMESTAMP reason code."""
        top = _make_candidate(0.7, 0.7, 0.7, 0.7, 0.6)

        decision = make_identity_decision(
            [top],
            context={
                "quality_score": 0.7,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.ACTIVE,
                # No timestamp
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert ReasonCode.MISSING_TIMESTAMP in decision.reason_codes

    def test_missing_location_produces_reason_code(self):
        """Missing location -> MISSING_LOCATION reason code."""
        top = _make_candidate(0.7, 0.7, 0.7, 0.7, 0.6)

        decision = make_identity_decision(
            [top],
            context={
                "quality_score": 0.7,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                # No lat/lon
            },
        )

        assert ReasonCode.MISSING_LOCATION in decision.reason_codes

    def test_low_visual_margin_produces_reason_code(self):
        """Close top-1/top-2 scores -> LOW_VISUAL_MARGIN reason code."""
        top = _make_candidate(0.80, 0.7, 0.7, 0.7, 0.6)
        second = _make_candidate(0.78, 0.7, 0.7, 0.7, 0.6, candidate_identity="T02", rank=2)

        decision = make_identity_decision(
            [top, second],
            context={
                "quality_score": 0.7,
                "flank_visibility": 0.5,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )

        assert ReasonCode.LOW_VISUAL_MARGIN in decision.reason_codes
        # Low margin is a major conflict — should not be trusted
        assert decision.update_history is False


class TestUpdateHistoryInvariant:
    """update_history is NEVER True unless decision == trusted_match.
    This is the one rule we must never weaken."""

    def test_update_history_only_for_trusted(self):
        """Exhaustive: for various evidence levels, update_history
        should only be True when decision is trusted_match."""
        import random
        random.seed(42)

        for _ in range(100):
            vs = random.random()
            qs = random.random()
            ss = random.random()
            ts = random.random()
            hs = random.random()

            top = _make_candidate(vs, qs, ss, ts, hs)

            decision = make_identity_decision(
                [top],
                context={
                    "quality_score": qs,
                    "flank_visibility": random.random(),
                    "camera_status": CameraStatus.ACTIVE,
                    "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                    "latitude": 21.68,
                    "longitude": 79.29,
                },
            )

            if decision.update_history:
                assert decision.decision == IdentityDecisionState.TRUSTED_MATCH, (
                    f"update_history=True but decision={decision.decision} "
                    f"(scores: V={vs:.2f}, Q={qs:.2f}, S={ss:.2f}, T={ts:.2f}, H={hs:.2f})"
                )


class TestCalibration:
    """Test check_calibration bucketing and ECE computation."""

    def test_calibration_accuracy_and_ece(self):
        """Check calibration buckets and Expected Calibration Error."""
        c1 = _make_candidate(0.95, 0.9, 0.9, 0.9, 0.9, image_id="img1", candidate_identity="T01")
        c2 = _make_candidate(0.20, 0.2, 0.2, 0.2, 0.2, image_id="img2", candidate_identity="T01")

        ctx = {
            "quality_score": 0.8,
            "flank_visibility": 0.7,
            "camera_status": CameraStatus.ACTIVE,
            "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
            "latitude": 21.68,
            "longitude": 79.29,
        }

        d1 = make_identity_decision([c1], context=ctx)
        d2 = make_identity_decision([c2], context=ctx)

        data = [
            {"decision": d1, "true_identity": "T01", "should_create_observation": True},
            {"decision": d2, "true_identity": None, "should_create_observation": False},
        ]

        res = check_calibration(data, n_bins=5)
        assert res["total_samples"] == 2
        assert len(res["buckets"]) == 5
        assert 0.0 <= res["expected_calibration_error"] <= 1.0
        assert 0.0 <= res["max_calibration_error"] <= 1.0

    def test_calibration_empty_input(self):
        """Empty input returns 0 samples cleanly without crashing."""
        res = check_calibration([])
        assert res["total_samples"] == 0
        assert res["expected_calibration_error"] == 0.0

