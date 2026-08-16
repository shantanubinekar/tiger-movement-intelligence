"""
tests/test_history.py — Tests for src/history.py.

Developer 3 owns this file.

Core safety rule verification:
- A non-trusted decision NEVER creates/updates an Observation.
- A trusted decision DOES create an Observation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.schemas import (
    IdentityCandidate,
    IdentityDecision,
    IdentityDecisionState,
    Observation,
    ObservationStatus,
    ReasonCode,
)
from src.history import (
    update_trusted_history,
    compute_individual_summary,
    get_history,
    reset_history,
)


@pytest.fixture(autouse=True)
def clean_history():
    """Reset the module-level history before each test."""
    reset_history()
    yield
    reset_history()


# ---------------------------------------------------------------------------
# Helper: build decisions
# ---------------------------------------------------------------------------

def _trusted_decision(image_id: str = "img_001", identity_id: str = "T01",
                      confidence: float = 0.9) -> IdentityDecision:
    """Build a trusted_match decision for testing."""
    return IdentityDecision(
        image_id=image_id,
        decision=IdentityDecisionState.TRUSTED_MATCH,
        identity_id=identity_id,
        confidence=confidence,
        top_candidates=[
            IdentityCandidate(
                image_id=image_id,
                candidate_identity=identity_id,
                rank=1,
                visual_score=confidence,
                quality_score=0.8,
                spatial_feasibility=0.8,
                temporal_feasibility=0.8,
                history_consistency=0.7,
                total_evidence=confidence,
            )
        ],
        reason_codes=[ReasonCode.HIGH_CONFIDENCE_MATCH],
        evidence_summary={"data_mode": "demo"},
        update_history=True,
    )


def _non_trusted_decision(
    state: IdentityDecisionState = IdentityDecisionState.AMBIGUOUS_REVIEW,
    image_id: str = "img_bad",
    confidence: float = 0.4,
) -> IdentityDecision:
    """Build a non-trusted decision for testing."""
    return IdentityDecision(
        image_id=image_id,
        decision=state,
        identity_id=None,
        confidence=confidence,
        top_candidates=[],
        reason_codes=[ReasonCode.LOW_VISUAL_MARGIN],
        evidence_summary={"data_mode": "demo"},
        update_history=False,
    )


def _image_metadata(station_id: str = "STATION_A1", **kwargs) -> dict:
    """Build image metadata for testing."""
    meta = {
        "station_id": station_id,
        "latitude": 21.680,
        "longitude": 79.290,
        "timestamp": datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc),
        "camera_status": "active",
        "quality_score": 0.85,
    }
    meta.update(kwargs)
    return meta


# ---------------------------------------------------------------------------
# SAFETY RULE TESTS — the most important tests in the entire project
# ---------------------------------------------------------------------------

class TestSafetyRule:
    """Verify that the critical safety rule is enforced:
    ONLY trusted_match with update_history=True may create an Observation."""

    def test_non_trusted_never_creates_observation(self):
        """A non-trusted decision must NEVER create an Observation."""
        non_trusted_states = [
            IdentityDecisionState.AMBIGUOUS_REVIEW,
            IdentityDecisionState.UNKNOWN,
            IdentityDecisionState.INSUFFICIENT_EVIDENCE,
            IdentityDecisionState.REJECTED,
            IdentityDecisionState.NON_TIGER,
            IdentityDecisionState.BLANK,
        ]
        for state in non_trusted_states:
            decision = _non_trusted_decision(state=state)
            result = update_trusted_history(decision, _image_metadata())
            assert result is None, (
                f"SAFETY VIOLATION: {state.value} decision created an "
                f"Observation — this must never happen!"
            )

    def test_trusted_decision_creates_observation(self):
        """A trusted_match with update_history=True must create an Observation."""
        decision = _trusted_decision()
        result = update_trusted_history(decision, _image_metadata())
        assert result is not None, (
            "trusted_match decision should have produced an Observation"
        )
        assert isinstance(result, Observation)
        assert result.observation_status == ObservationStatus.TRUSTED

    def test_trusted_decision_with_update_history_false(self):
        """Even if decision is trusted_match, update_history=False blocks it."""
        decision = IdentityDecision(
            image_id="img_manual",
            decision=IdentityDecisionState.TRUSTED_MATCH,
            identity_id="T01",
            confidence=0.9,
            update_history=False,  # Explicit override
        )
        result = update_trusted_history(decision, _image_metadata())
        assert result is None

    def test_history_not_contaminated_by_non_trusted(self):
        """After processing non-trusted decisions, the history store
        must remain empty."""
        for state in [
            IdentityDecisionState.AMBIGUOUS_REVIEW,
            IdentityDecisionState.UNKNOWN,
        ]:
            decision = _non_trusted_decision(state=state)
            update_trusted_history(decision, _image_metadata())

        history = get_history()
        assert len(history.get_all_identity_ids()) == 0, (
            "Trusted history was contaminated by non-trusted decisions!"
        )


# ---------------------------------------------------------------------------
# Observation creation tests
# ---------------------------------------------------------------------------

class TestObservationCreation:
    """Test that trusted observations are correctly created and stored."""

    def test_observation_fields_populated(self):
        """Observation should have all fields from decision + metadata."""
        decision = _trusted_decision(identity_id="T01")
        meta = _image_metadata(station_id="STATION_B2", latitude=21.702,
                               longitude=79.315)
        obs = update_trusted_history(decision, meta)

        assert obs is not None
        assert obs.identity_id == "T01"
        assert obs.station_id == "STATION_B2"
        assert obs.latitude == 21.702
        assert obs.longitude == 79.315
        assert obs.identity_confidence == 0.9

    def test_observation_stored_in_history(self):
        """Created observation should be retrievable from history."""
        decision = _trusted_decision(identity_id="T01")
        update_trusted_history(decision, _image_metadata())

        history = get_history()
        obs_list = history.get_observations("T01")
        assert len(obs_list) == 1

    def test_multiple_observations_sorted_by_time(self):
        """Multiple observations should be sorted by timestamp."""
        from datetime import timedelta

        base = datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc)
        for i, day_offset in enumerate([3, 1, 2]):  # Out-of-order
            decision = _trusted_decision(image_id=f"img_{i}", identity_id="T01")
            meta = _image_metadata(
                timestamp=base + timedelta(days=day_offset)
            )
            update_trusted_history(decision, meta)

        history = get_history()
        obs_list = history.get_observations("T01")
        assert len(obs_list) == 3
        # Should be sorted: day 1, 2, 3
        timestamps = [o.timestamp for o in obs_list]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Individual summary tests
# ---------------------------------------------------------------------------

class TestIndividualSummary:
    """Test compute_individual_summary() output."""

    def test_summary_for_nonexistent_identity(self):
        """Should return None if identity has no observations."""
        result = compute_individual_summary("NONEXISTENT")
        assert result is None

    def test_summary_fields(self):
        """Summary should have correct aggregate fields."""
        from datetime import timedelta
        base = datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc)

        # Add 3 observations at 2 stations
        for i, (station, lat, lon) in enumerate([
            ("STATION_A1", 21.680, 79.290),
            ("STATION_A1", 21.681, 79.291),
            ("STATION_B2", 21.702, 79.315),
        ]):
            decision = _trusted_decision(image_id=f"img_{i}", identity_id="T01")
            meta = _image_metadata(
                station_id=station, latitude=lat, longitude=lon,
                timestamp=base + timedelta(days=i),
            )
            update_trusted_history(decision, meta)

        summary = compute_individual_summary("T01")
        assert summary is not None
        assert summary.capture_count == 3
        assert summary.first_seen == base
        assert summary.last_seen == base + timedelta(days=2)
        assert len(summary.trusted_stations) == 2
        assert "STATION_A1" in summary.trusted_stations
        assert "STATION_B2" in summary.trusted_stations
        assert summary.activity_centroid is not None
        assert summary.camera_effort_history["STATION_A1"] == 2
        assert summary.camera_effort_history["STATION_B2"] == 1


class TestHistoricalCaptureArea:
    """Test compute_historical_capture_area filtering rules."""

    def test_capture_area_requires_at_least_two_obs_per_station(self):
        """Single observation stations should be filtered as transient noise."""
        from datetime import timedelta
        base = datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc)

        # 3 stations, but only 1 observation each -> hull should be None (needs >=2 obs per station)
        for i, (st, lat, lon) in enumerate([
            ("STATION_A1", 21.680, 79.290),
            ("STATION_B2", 21.702, 79.315),
            ("STATION_C3", 21.720, 79.330),
        ]):
            d = _trusted_decision(image_id=f"img_single_{i}", identity_id="T01")
            m = _image_metadata(station_id=st, latitude=lat, longitude=lon, timestamp=base + timedelta(days=i))
            update_trusted_history(d, m)

        history = get_history()
        assert history.compute_historical_capture_area("T01") is None

        # Add a 2nd observation for each of the 3 stations -> now they qualify as established stations
        for i, (st, lat, lon) in enumerate([
            ("STATION_A1", 21.680, 79.290),
            ("STATION_B2", 21.702, 79.315),
            ("STATION_C3", 21.720, 79.330),
        ]):
            d = _trusted_decision(image_id=f"img_double_{i}", identity_id="T01")
            m = _image_metadata(station_id=st, latitude=lat, longitude=lon, timestamp=base + timedelta(days=i+10))
            update_trusted_history(d, m)

        hull = history.compute_historical_capture_area("T01")
        assert hull is not None
        assert len(hull) >= 3

