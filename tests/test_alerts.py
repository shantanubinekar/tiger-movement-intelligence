"""
tests/test_alerts.py — Tests for src/alerts.py and src/movement.py.

Developer 3 owns this file.

Key tests:
- Camera relocation suppresses an alert that would otherwise fire.
- A single isolated observation does NOT trigger OUTSIDE_HISTORICAL_AREA.
- A genuine large jump with sufficient history triggers an alert with
  a non-empty explanation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.schemas import (
    AlertStatus,
    AlertType,
    CameraStatus,
    Observation,
    ObservationStatus,
)
from src.movement import (
    compute_distance,
    compute_travel_speed,
    detect_deviations,
    MovementConfig,
)
from src.alerts import (
    generate_alerts,
    AlertConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc)


def _obs(
    obs_id: str = "obs_001",
    identity_id: str = "T01",
    station_id: str = "STATION_A1",
    lat: float = 21.680,
    lon: float = 79.290,
    timestamp: datetime | None = None,
    camera_status: CameraStatus = CameraStatus.ACTIVE,
    quality: float = 0.85,
    confidence: float = 0.90,
) -> Observation:
    """Build an Observation for testing."""
    return Observation(
        observation_id=obs_id,
        image_id=f"img_{obs_id}",
        identity_id=identity_id,
        station_id=station_id,
        latitude=lat,
        longitude=lon,
        timestamp=timestamp or _BASE_TIME,
        identity_confidence=confidence,
        observation_status=ObservationStatus.TRUSTED,
        camera_status=camera_status,
        quality_score=quality,
    )


# ---------------------------------------------------------------------------
# Movement computation tests
# ---------------------------------------------------------------------------

class TestMovementComputation:

    def test_haversine_distance_same_point(self):
        a = _obs(obs_id="a", lat=21.680, lon=79.290)
        b = _obs(obs_id="b", lat=21.680, lon=79.290)
        dist = compute_distance(a, b)
        assert dist is not None
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_haversine_distance_different_points(self):
        a = _obs(obs_id="a", lat=21.680, lon=79.290)
        b = _obs(obs_id="b", lat=21.702, lon=79.315)
        dist = compute_distance(a, b)
        assert dist is not None
        assert dist > 0
        # Roughly ~3.5 km for these coordinates
        assert 2.0 < dist < 5.0

    def test_distance_with_missing_gps(self):
        a = _obs(obs_id="a", lat=21.680, lon=79.290)
        b = _obs(obs_id="b", lat=None, lon=None)
        dist = compute_distance(a, b)
        assert dist is None

    def test_travel_speed(self):
        a = _obs(obs_id="a", timestamp=_BASE_TIME)
        b = _obs(obs_id="b", lat=21.702, lon=79.315,
                 timestamp=_BASE_TIME + timedelta(hours=1))
        speed = compute_travel_speed(a, b)
        assert speed is not None
        assert speed > 0

    def test_travel_speed_with_missing_time(self):
        a = _obs(obs_id="a", timestamp=_BASE_TIME)
        b = _obs(obs_id="b", timestamp=None)
        speed = compute_travel_speed(a, b)
        assert speed is None


# ---------------------------------------------------------------------------
# Deviation detection tests
# ---------------------------------------------------------------------------

class TestDeviationDetection:

    def test_new_station_detected(self):
        """A new station should trigger NEW_STATION deviation."""
        history = [
            _obs(obs_id="h1", station_id="STATION_A1"),
            _obs(obs_id="h2", station_id="STATION_A1",
                 timestamp=_BASE_TIME + timedelta(days=1)),
        ]
        new = _obs(obs_id="new", station_id="STATION_B2",
                   lat=21.702, lon=79.315,
                   timestamp=_BASE_TIME + timedelta(days=5))

        devs = detect_deviations("T01", history, new, None)
        new_station_devs = [d for d in devs if d.deviation_type == AlertType.NEW_STATION]
        assert len(new_station_devs) == 1
        assert new_station_devs[0].details["new_station"] == "STATION_B2"

    def test_single_observation_no_outside_area(self):
        """A single isolated observation must NOT trigger
        OUTSIDE_HISTORICAL_AREA — not enough history."""
        new = _obs(obs_id="first", station_id="STATION_X",
                   lat=22.0, lon=80.0)
        devs = detect_deviations("T01", [], new, None)
        outside_devs = [d for d in devs if d.deviation_type == AlertType.OUTSIDE_HISTORICAL_AREA]
        assert len(outside_devs) == 0

    def test_unusual_travel_large_distance(self):
        """A very large jump should trigger UNUSUAL_TRAVEL."""
        config = MovementConfig(unusual_distance_km=10.0)
        history = [
            _obs(obs_id="h1", station_id="STATION_A1", lat=21.680, lon=79.290),
            _obs(obs_id="h2", station_id="STATION_A1", lat=21.681, lon=79.291,
                 timestamp=_BASE_TIME + timedelta(days=1)),
        ]
        # ~50 km away
        new = _obs(obs_id="new", station_id="STATION_FAR",
                   lat=22.1, lon=79.3,
                   timestamp=_BASE_TIME + timedelta(days=3))

        devs = detect_deviations("T01", history, new, None, config=config)
        travel_devs = [d for d in devs if d.deviation_type == AlertType.UNUSUAL_TRAVEL]
        assert len(travel_devs) >= 1
        assert travel_devs[0].magnitude > 10.0  # > threshold

    def test_prolonged_absence(self):
        """100-day gap should trigger PROLONGED_ABSENCE."""
        config = MovementConfig(prolonged_absence_days=90.0)
        history = [
            _obs(obs_id="h1", timestamp=_BASE_TIME),
        ]
        new = _obs(obs_id="new",
                   timestamp=_BASE_TIME + timedelta(days=100))

        devs = detect_deviations("T01", history, new, None, config=config)
        absence_devs = [d for d in devs if d.deviation_type == AlertType.PROLONGED_ABSENCE]
        assert len(absence_devs) == 1
        assert absence_devs[0].magnitude > 90.0

    def test_buffer_station(self):
        """Buffer-zone station should trigger BUFFER_OR_VILLAGE_ADJACENT."""
        ctx = {"buffer_stations": {"STATION_BUFFER"}}
        new = _obs(obs_id="new", station_id="STATION_BUFFER")
        devs = detect_deviations("T01", [], new, None,
                                 station_context=ctx)
        buffer_devs = [d for d in devs if d.deviation_type == AlertType.BUFFER_OR_VILLAGE_ADJACENT]
        assert len(buffer_devs) == 1


# ---------------------------------------------------------------------------
# Alert generation and suppression tests
# ---------------------------------------------------------------------------

class TestAlertSuppression:

    def test_camera_relocation_suppresses_alert(self):
        """An alert at a recently relocated camera station must be
        SUPPRESSED with a non-empty suppression_reason."""
        # Create deviation for new station
        new = _obs(obs_id="new", station_id="STATION_B2",
                   lat=21.702, lon=79.315,
                   timestamp=_BASE_TIME + timedelta(days=20))

        history = [
            _obs(obs_id="h1", station_id="STATION_A1"),
            _obs(obs_id="h2", station_id="STATION_A1",
                 timestamp=_BASE_TIME + timedelta(days=3)),
        ]

        devs = detect_deviations("T01", history, new, None)
        assert len(devs) > 0  # Should have NEW_STATION at least

        # Station B2 was relocated 14 days ago (within 30-day cooldown)
        station_context = {
            "relocated_stations": {
                "STATION_B2": _BASE_TIME + timedelta(days=6),
            }
        }

        alerts = generate_alerts(
            identity_id="T01",
            deviations=devs,
            observation=new,
            capture_count=3,
            station_context=station_context,
        )

        assert len(alerts) > 0
        for alert in alerts:
            assert alert.status == AlertStatus.SUPPRESSED, (
                f"Alert should be SUPPRESSED due to camera relocation, "
                f"but got status={alert.status}"
            )
            assert alert.suppression_reason is not None
            assert len(alert.suppression_reason) > 0
            assert "relocated" in alert.suppression_reason.lower()

    def test_insufficient_history_suppresses_alert(self):
        """With only 1 observation on record, alerts should be
        downgraded/suppressed."""
        config = AlertConfig(min_observations_for_alert=2)
        new = _obs(obs_id="first", station_id="STATION_A1")

        devs = detect_deviations("T01", [], new, None)
        if not devs:
            # Force a deviation for testing
            from src.movement import MovementDeviation
            devs = [MovementDeviation(
                deviation_type=AlertType.NEW_STATION,
                magnitude=0.0,
                details={"new_station": "STATION_A1"},
                triggering_observation=new,
            )]

        alerts = generate_alerts(
            identity_id="T01",
            deviations=devs,
            observation=new,
            capture_count=1,
            config=config,
        )

        for alert in alerts:
            assert alert.status != AlertStatus.ACTIVE, (
                "Alert should not be ACTIVE with only 1 observation"
            )
            assert alert.suppression_reason is not None

    def test_genuine_alert_has_nonempty_explanation(self):
        """A genuine alert with sufficient history should have a
        non-empty explanation describing the evidence."""
        history = [
            _obs(obs_id="h1", station_id="STATION_A1"),
            _obs(obs_id="h2", station_id="STATION_A1",
                 timestamp=_BASE_TIME + timedelta(days=1)),
        ]
        # Far-away station, no relocation or other suppression conditions
        new = _obs(obs_id="new", station_id="STATION_FAR",
                   lat=22.1, lon=79.3,
                   timestamp=_BASE_TIME + timedelta(days=3))

        config = MovementConfig(unusual_distance_km=10.0)
        devs = detect_deviations("T01", history, new, None, config=config)
        travel_devs = [d for d in devs if d.deviation_type == AlertType.UNUSUAL_TRAVEL]
        assert len(travel_devs) >= 1

        alerts = generate_alerts(
            identity_id="T01",
            deviations=travel_devs,
            observation=new,
            capture_count=3,
        )

        assert len(alerts) >= 1
        for alert in alerts:
            assert alert.status == AlertStatus.ACTIVE
            assert alert.explanation is not None
            assert len(alert.explanation) > 0
            # Explanation should mention evidence
            assert "confidence" in alert.explanation.lower() or \
                   "evidence" in alert.explanation.lower() or \
                   "identity" in alert.explanation.lower()
            assert alert.suppression_reason is None

    def test_inactive_camera_suppresses_prolonged_absence(self):
        """Prolonged absence where cameras were inactive should be
        suppressed as an observation artefact."""
        history = [
            _obs(obs_id="h1", timestamp=_BASE_TIME),
            _obs(obs_id="h2", timestamp=_BASE_TIME + timedelta(days=2)),
        ]
        # 100 days later, camera was inactive
        new = _obs(obs_id="new",
                   timestamp=_BASE_TIME + timedelta(days=102),
                   camera_status=CameraStatus.INACTIVE)

        config = MovementConfig(prolonged_absence_days=90.0)
        devs = detect_deviations("T01", history, new, None, config=config)
        absence_devs = [d for d in devs if d.deviation_type == AlertType.PROLONGED_ABSENCE]

        if absence_devs:
            # Camera at observation time is inactive — should suppress
            alerts = generate_alerts(
                identity_id="T01",
                deviations=absence_devs,
                observation=new,
                capture_count=3,
            )
            for alert in alerts:
                assert alert.status == AlertStatus.SUPPRESSED
                assert "inactive" in alert.suppression_reason.lower()
