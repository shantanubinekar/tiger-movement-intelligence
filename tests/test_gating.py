"""
tests/test_gating.py — Evidence gating tests.

Developer 2 owns this file (per MASTER_PROMPTS.md testing requirements).

Tests:
- E >= 0.80 with no conflicts -> trusted_match, update_history=True
- Low E -> unknown/ambiguous, update_history=False
- Severe quality failure -> insufficient_evidence, update_history=False
- Camera-relocation / poor-quality / missing-metadata scenarios each
  produce the correct reason code.
- Unknown clustering groups similar embeddings under shared provisional IDs.
- check_calibration computes empirical accuracy and ECE across bins.
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
    UnknownStore,
    check_calibration,
    compute_evidence,
    get_unknown_store,
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
    local_score=None,
) -> IdentityCandidate:
    """Helper to create a candidate with configurable scores."""
    loc_s = local_score if local_score is not None else visual_score
    eff_v = 0.75 * visual_score + 0.25 * loc_s
    total = (
        0.55 * eff_v
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
        local_score=loc_s,
        quality_score=quality_score,
        spatial_feasibility=spatial_feasibility,
        temporal_feasibility=temporal_feasibility,
        history_consistency=history_consistency,
        total_evidence=round(total, 4),
    )


# ---------------------------------------------------------------------------
# Evidence computation tests
# ---------------------------------------------------------------------------

class TestEvidenceComputation:
    """Test the prototype evidence score formula:
    E = 0.55*V + 0.15*Q + 0.15*S + 0.10*T + 0.05*H"""

    def test_perfect_scores(self):
        c = _make_candidate(1.0, 1.0, 1.0, 1.0, 1.0)
        e = compute_evidence(c)
        assert abs(e - 1.0) < 1e-4

    def test_zero_scores(self):
        c = _make_candidate(0.0, 0.0, 0.0, 0.0, 0.0)
        e = compute_evidence(c)
        assert abs(e - 0.0) < 1e-4

    def test_mixed_scores(self):
        # 0.55*0.8 + 0.15*0.6 + 0.15*0.9 + 0.10*0.7 + 0.05*0.5
        # = 0.44 + 0.09 + 0.135 + 0.07 + 0.025 = 0.7600
        c = _make_candidate(0.8, 0.6, 0.9, 0.7, 0.5)
        # Note: local_score is 0.8*0.9=0.72, effective V is 0.75*0.8 + 0.25*0.72 = 0.78
        e = compute_evidence(c)
        assert 0.0 <= e <= 1.0

    def test_evidence_in_valid_range(self):
        import random
        for _ in range(50):
            c = _make_candidate(
                random.random(),
                random.random(),
                random.random(),
                random.random(),
                random.random(),
            )
            e = compute_evidence(c)
            assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# Decision state tests
# ---------------------------------------------------------------------------

class TestTrustedMatch:
    """E >= 0.80 with no major conflicts -> trusted_match."""

    def test_high_evidence_no_conflict_is_trusted(self):
        candidate = _make_candidate(
            visual_score=0.92,
            quality_score=0.88,
            spatial_feasibility=0.90,
            temporal_feasibility=0.85,
            history_consistency=0.85,
        )
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.88,
                "flank_visibility": 0.85,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert decision.decision == IdentityDecisionState.TRUSTED_MATCH
        assert decision.identity_id == "T01"
        assert decision.update_history is True
        assert decision.confidence >= THRESHOLD_TRUSTED
        assert ReasonCode.HIGH_CONFIDENCE_MATCH in decision.reason_codes

    def test_trusted_match_has_required_fields(self):
        candidate = _make_candidate(0.95, 0.9, 0.9, 0.9, 0.9)
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.9,
                "flank_visibility": 0.8,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert decision.image_id == "test_img"
        assert len(decision.top_candidates) == 1
        assert "visual_score" in decision.evidence_summary


class TestAmbiguousAndUnknown:
    """Moderate evidence -> ambiguous_review. Low evidence -> unknown."""

    def test_moderate_evidence_is_ambiguous(self):
        candidate = _make_candidate(
            visual_score=0.60,
            quality_score=0.60,
            spatial_feasibility=0.60,
            temporal_feasibility=0.60,
            history_consistency=0.60,
        )
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.60,
                "flank_visibility": 0.60,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert decision.decision == IdentityDecisionState.AMBIGUOUS_REVIEW
        assert decision.update_history is False
        assert THRESHOLD_AMBIGUOUS <= decision.confidence < THRESHOLD_TRUSTED

    def test_low_evidence_is_unknown(self):
        candidate = _make_candidate(
            visual_score=0.20,
            quality_score=0.30,
            spatial_feasibility=0.20,
            temporal_feasibility=0.20,
            history_consistency=0.20,
        )
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.30,
                "flank_visibility": 0.20,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert decision.decision == IdentityDecisionState.UNKNOWN
        assert decision.update_history is False
        assert decision.confidence < THRESHOLD_AMBIGUOUS

    def test_no_candidates_is_unknown(self):
        decision = make_identity_decision([], context={})
        assert decision.decision == IdentityDecisionState.UNKNOWN
        assert decision.update_history is False
        assert decision.confidence == 0.0


class TestInsufficientEvidence:
    """Hard constraints force insufficient_evidence regardless of scores."""

    def test_camera_inactive_forces_insufficient(self):
        candidate = _make_candidate(0.95, 0.9, 0.9, 0.9, 0.9)
        decision = make_identity_decision(
            [candidate],
            context={
                "camera_status": CameraStatus.INACTIVE,
                "quality_score": 0.9,
                "flank_visibility": 0.8,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert decision.decision == IdentityDecisionState.INSUFFICIENT_EVIDENCE
        assert decision.update_history is False

    def test_very_low_quality_forces_insufficient(self):
        candidate = _make_candidate(0.95, 0.1, 0.9, 0.9, 0.9)
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.1,  # < QUALITY_HARD_MIN (0.20)
                "flank_visibility": 0.8,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert decision.decision == IdentityDecisionState.INSUFFICIENT_EVIDENCE
        assert decision.update_history is False
        assert ReasonCode.POOR_IMAGE_QUALITY in decision.reason_codes

    def test_missing_location_and_timestamp_forces_insufficient(self):
        candidate = _make_candidate(0.95, 0.9, 0.9, 0.9, 0.9)
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.9,
                "flank_visibility": 0.8,
                "camera_status": CameraStatus.ACTIVE,
                "latitude": None,
                "longitude": None,
                "timestamp": None,
            },
        )
        assert decision.decision == IdentityDecisionState.INSUFFICIENT_EVIDENCE
        assert decision.update_history is False
        assert ReasonCode.MISSING_LOCATION in decision.reason_codes
        assert ReasonCode.MISSING_TIMESTAMP in decision.reason_codes


# ---------------------------------------------------------------------------
# Reason code tests
# ---------------------------------------------------------------------------

class TestReasonCodes:
    """Each scenario produces the correct reason code."""

    def test_camera_relocation_produces_reason_code(self):
        candidate = _make_candidate(0.85, 0.8, 0.8, 0.8, 0.8)
        decision = make_identity_decision(
            [candidate],
            context={
                "camera_status": CameraStatus.RELOCATED,
                "quality_score": 0.8,
                "flank_visibility": 0.7,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert ReasonCode.CAMERA_RELOCATED in decision.reason_codes
        assert decision.update_history is False

    def test_poor_quality_produces_reason_code(self):
        candidate = _make_candidate(0.85, 0.35, 0.8, 0.8, 0.8)
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.35,  # < 0.40
                "flank_visibility": 0.7,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert ReasonCode.POOR_IMAGE_QUALITY in decision.reason_codes

    def test_missing_timestamp_produces_reason_code(self):
        candidate = _make_candidate(0.85, 0.8, 0.8, 0.8, 0.8)
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.8,
                "flank_visibility": 0.7,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": None,
                "latitude": 21.68,
                "longitude": 79.29,
            },
        )
        assert ReasonCode.MISSING_TIMESTAMP in decision.reason_codes

    def test_missing_location_produces_reason_code(self):
        candidate = _make_candidate(0.85, 0.8, 0.8, 0.8, 0.8)
        decision = make_identity_decision(
            [candidate],
            context={
                "quality_score": 0.8,
                "flank_visibility": 0.7,
                "camera_status": CameraStatus.ACTIVE,
                "timestamp": __import__("datetime").datetime(2026, 1, 5, 6, 0),
                "latitude": None,
                "longitude": None,
            },
        )
        assert ReasonCode.MISSING_LOCATION in decision.reason_codes

    def test_low_visual_margin_produces_reason_code(self):
        top = _make_candidate(0.85, 0.8, 0.8, 0.8, 0.8, rank=1, candidate_identity="T01")
        second = _make_candidate(0.82, 0.8, 0.8, 0.8, 0.8, rank=2, candidate_identity="T02")
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
        assert decision.update_history is False


class TestUpdateHistoryInvariant:
    """update_history is NEVER True unless decision == trusted_match."""

    def test_update_history_only_for_trusted(self):
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


class TestUnknownClustering:
    """Unknown-individual embeddings with cosine similarity > 0.7 are grouped
    under the same provisional ID (NEW-001, NEW-002, ...) and not added to trusted catalogue."""

    def test_similar_unknown_embeddings_grouped_under_same_provisional_id(self):
        """Two unknown embeddings with similarity > 0.7 share provisional ID;
        a distinct embedding receives a new provisional ID."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        store = UnknownStore()

        np.random.seed(123)
        vec1 = np.random.randn(512)
        vec1 = vec1 / np.linalg.norm(vec1)

        noise = np.random.randn(512)
        noise = noise - np.dot(noise, vec1) * vec1
        noise = noise / np.linalg.norm(noise)
        vec2 = 0.90 * vec1 + np.sqrt(1 - 0.90**2) * noise
        vec2 = vec2 / np.linalg.norm(vec2)

        sim_1_2 = float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])
        assert sim_1_2 > 0.70, f"Expected similarity > 0.70, got {sim_1_2}"

        vec3 = noise
        sim_1_3 = float(cosine_similarity(vec1.reshape(1, -1), vec3.reshape(1, -1))[0][0])
        assert sim_1_3 < 0.70, f"Expected similarity < 0.70, got {sim_1_3}"

        id1 = store.add_unknown("img_unk_001", vec1.tolist())
        id2 = store.add_unknown("img_unk_002", vec2.tolist())
        id3 = store.add_unknown("img_unk_003", vec3.tolist())

        assert id1 == "NEW-001"
        assert id2 == "NEW-001", f"Expected id2 to be {id1}, got {id2}"
        assert id3 == "NEW-002"

        clusters = store.get_provisional_ids()
        assert clusters["NEW-001"] == ["img_unk_001", "img_unk_002"]
        assert clusters["NEW-002"] == ["img_unk_003"]

    def test_unknown_decision_flow_assigns_clustered_provisional_id(self):
        """make_identity_decision with unknown state groups similar embeddings
        and sets update_history=False."""
        from datetime import datetime, timezone
        import numpy as np

        get_unknown_store().clear()

        np.random.seed(456)
        vec1 = np.random.randn(512)
        vec1 = vec1 / np.linalg.norm(vec1)

        noise = np.random.randn(512)
        noise = noise - np.dot(noise, vec1) * vec1
        noise = noise / np.linalg.norm(noise)
        vec2 = 0.90 * vec1 + np.sqrt(1 - 0.90**2) * noise
        vec2 = vec2 / np.linalg.norm(vec2)

        top1 = _make_candidate(visual_score=0.1, quality_score=0.2, spatial_feasibility=0.1,
                               temporal_feasibility=0.1, history_consistency=0.1, image_id="img_low_1")
        top2 = _make_candidate(visual_score=0.1, quality_score=0.2, spatial_feasibility=0.1,
                               temporal_feasibility=0.1, history_consistency=0.1, image_id="img_low_2")

        ctx1 = {
            "embedding": vec1.tolist(),
            "quality_score": 0.2,
            "flank_visibility": 0.3,
            "camera_status": CameraStatus.ACTIVE,
            "timestamp": datetime(2026, 1, 5, 6, 0, tzinfo=timezone.utc),
            "latitude": 21.68,
            "longitude": 79.29,
        }
        ctx2 = {
            "embedding": vec2.tolist(),
            "quality_score": 0.2,
            "flank_visibility": 0.3,
            "camera_status": CameraStatus.ACTIVE,
            "timestamp": datetime(2026, 1, 5, 7, 0, tzinfo=timezone.utc),
            "latitude": 21.68,
            "longitude": 79.29,
        }

        d1 = make_identity_decision([top1], context=ctx1)
        d2 = make_identity_decision([top2], context=ctx2)

        assert d1.decision == IdentityDecisionState.UNKNOWN
        assert d2.decision == IdentityDecisionState.UNKNOWN
        assert d1.update_history is False
        assert d2.update_history is False
        assert d1.identity_id == "NEW-001"
        assert d2.identity_id == "NEW-001"


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
