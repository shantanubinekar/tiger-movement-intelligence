"""
tests/test_schemas.py — Schema validation tests.

Developer 2 owns this file (per MASTER_PROMPTS.md testing requirements).

Tests:
- IdentityDecision only allows the six defined states.
- Safety guard rejects update_history=True with non-trusted decision.
- All schema models can be constructed with valid data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.schemas import (
    AlertStatus,
    AlertType,
    CameraStatus,
    DataMode,
    DetectionRecord,
    EvaluationReport,
    IdentityCandidate,
    IdentityDecision,
    IdentityDecisionState,
    ImageRecord,
    MovementAlert,
    Observation,
    ObservationStatus,
    ReasonCode,
    TriageRecord,
    TriageStatus,
)


class TestIdentityDecisionStates:
    """IdentityDecision only allows the six defined states."""

    def test_all_valid_states_are_constructable(self):
        valid_states = [
            IdentityDecisionState.TRUSTED_MATCH,
            IdentityDecisionState.AMBIGUOUS_REVIEW,
            IdentityDecisionState.UNKNOWN,
            IdentityDecisionState.INSUFFICIENT_EVIDENCE,
            IdentityDecisionState.NON_TIGER,
            IdentityDecisionState.BLANK,
        ]
        for state in valid_states:
            is_trusted = state == IdentityDecisionState.TRUSTED_MATCH
            d = IdentityDecision(
                image_id="test",
                decision=state,
                confidence=0.5,
                identity_id="T01" if is_trusted else None,
                update_history=is_trusted,
            )
            assert d.decision == state

    def test_rejected_state_is_valid(self):
        """REJECTED is an additional valid state in our schema."""
        d = IdentityDecision(
            image_id="test",
            decision=IdentityDecisionState.REJECTED,
            confidence=0.1,
            update_history=False,
        )
        assert d.decision == IdentityDecisionState.REJECTED

    def test_invalid_state_raises(self):
        """An invalid string should fail Pydantic validation."""
        with pytest.raises(Exception):
            IdentityDecision(
                image_id="test",
                decision="totally_invalid_state",
                confidence=0.5,
                update_history=False,
            )


class TestSafetyGuard:
    """The safety guard on IdentityDecision rejects update_history=True
    unless decision == trusted_match."""

    def test_trusted_match_allows_update_history_true(self):
        d = IdentityDecision(
            image_id="test",
            decision=IdentityDecisionState.TRUSTED_MATCH,
            identity_id="T01",
            confidence=0.9,
            update_history=True,
        )
        assert d.update_history is True

    @pytest.mark.parametrize("state", [
        IdentityDecisionState.AMBIGUOUS_REVIEW,
        IdentityDecisionState.UNKNOWN,
        IdentityDecisionState.INSUFFICIENT_EVIDENCE,
        IdentityDecisionState.NON_TIGER,
        IdentityDecisionState.BLANK,
        IdentityDecisionState.REJECTED,
    ])
    def test_non_trusted_rejects_update_history_true(self, state):
        """THE CORE SAFETY RULE: update_history=True is forbidden for
        non-trusted decisions."""
        with pytest.raises(ValueError, match="update_history=True is only allowed"):
            IdentityDecision(
                image_id="test",
                decision=state,
                confidence=0.5,
                update_history=True,
            )

    def test_non_trusted_with_update_history_false_is_valid(self):
        """Non-trusted decisions with update_history=False are fine."""
        for state in [
            IdentityDecisionState.AMBIGUOUS_REVIEW,
            IdentityDecisionState.UNKNOWN,
            IdentityDecisionState.INSUFFICIENT_EVIDENCE,
        ]:
            d = IdentityDecision(
                image_id="test",
                decision=state,
                confidence=0.5,
                update_history=False,
            )
            assert d.update_history is False


class TestSchemaConstruction:
    """All schema models can be constructed with valid data."""

    def test_image_record(self):
        r = ImageRecord(
            image_id="img_001",
            image_path="/tmp/img.jpg",
            file_hash="abc123",
            data_mode=DataMode.DEMO,
        )
        assert r.image_id == "img_001"
        assert r.data_mode == DataMode.DEMO

    def test_triage_record(self):
        r = TriageRecord(
            image_id="img_001",
            blank_probability=0.8,
            subject_probability=0.2,
            triage_status=TriageStatus.BLANK,
        )
        assert r.triage_status == TriageStatus.BLANK

    def test_detection_record(self):
        r = DetectionRecord(
            image_id="img_001",
            species="tiger",
            detection_confidence=0.85,
            quality_score=0.7,
            flank_visibility=0.6,
        )
        assert r.species == "tiger"

    def test_identity_candidate(self):
        c = IdentityCandidate(
            image_id="img_001",
            candidate_identity="T01",
            rank=1,
            visual_score=0.9,
            quality_score=0.8,
            spatial_feasibility=0.7,
            temporal_feasibility=0.6,
            history_consistency=0.5,
            total_evidence=0.8,
        )
        assert c.rank == 1

    def test_observation(self):
        o = Observation(
            observation_id="obs_001",
            image_id="img_001",
            identity_id="T01",
            identity_confidence=0.9,
        )
        assert o.observation_status == ObservationStatus.TRUSTED

    def test_movement_alert(self):
        a = MovementAlert(
            alert_id="alert_001",
            identity_id="T01",
            alert_type=AlertType.NEW_STATION,
            confidence=0.7,
            status=AlertStatus.ACTIVE,
            explanation="Test alert",
        )
        assert a.alert_type == AlertType.NEW_STATION

    def test_evaluation_report(self):
        r = EvaluationReport(
            pipeline_name="baseline",
            not_computable=["false_confident_identity_rate"],
            notes="Test",
        )
        assert r.pipeline_name == "baseline"
