"""
src/history.py — Trusted longitudinal history management.

Developer 3 owns this file.

CRITICAL SAFETY RULE (PROJECT_CONTRACT.md Section 5):
Only IdentityDecision.decision == "trusted_match" (equivalently
update_history == True) may create/update a trusted Observation.
All other decision states (ambiguous_review, unknown, insufficient_evidence,
rejected, provisional) may be stored/displayed but must NEVER silently
enter the trusted longitudinal history used for movement analysis.

The enforcement point is update_trusted_history() — guard clause at
the top, not buried logic.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.schemas import (
    CameraStatus,
    IdentityDecision,
    IdentityDecisionState,
    IndividualSummary,
    Observation,
    ObservationStatus,
)


# ---------------------------------------------------------------------------
# Trusted History Store — in-memory per-tiger observation store
# ---------------------------------------------------------------------------

class TrustedHistory:
    """In-memory store for trusted longitudinal observation history.

    All observations held here have passed the safety gate: only
    trusted_match decisions with update_history=True are admitted.

    Terminology note (PROJECT_CONTRACT.md Section 13): the convex hull
    of trusted station coordinates is called "historical capture area,"
    never "home range" or "validated home range."
    """

    def __init__(self) -> None:
        # identity_id -> list of trusted Observations, ordered by timestamp
        self._observations: dict[str, list[Observation]] = {}

    # -- Core access --------------------------------------------------------

    def get_observations(self, identity_id: str) -> list[Observation]:
        """Return all trusted observations for an individual, sorted by
        timestamp (oldest first)."""
        return list(self._observations.get(identity_id, []))

    def get_all_identity_ids(self) -> list[str]:
        """Return all identity IDs with at least one trusted observation."""
        return list(self._observations.keys())

    def get_latest_observation(self, identity_id: str) -> Optional[Observation]:
        """Return the most recent trusted observation for an individual."""
        obs = self._observations.get(identity_id, [])
        return obs[-1] if obs else None

    def get_trusted_stations(self, identity_id: str) -> list[str]:
        """Return list of unique station_ids where this individual has
        been observed with trusted confidence."""
        stations: list[str] = []
        seen: set[str] = set()
        for obs in self._observations.get(identity_id, []):
            if obs.station_id and obs.station_id not in seen:
                stations.append(obs.station_id)
                seen.add(obs.station_id)
        return stations

    def get_capture_count(self, identity_id: str) -> int:
        """Return number of trusted observations for an individual."""
        return len(self._observations.get(identity_id, []))

    # -- Insert (only called from update_trusted_history) -------------------

    def add_observation(self, obs: Observation) -> None:
        """Add a trusted observation. Maintains timestamp sort order."""
        if obs.identity_id not in self._observations:
            self._observations[obs.identity_id] = []
        self._observations[obs.identity_id].append(obs)
        # Keep sorted by timestamp (None timestamps sort to the end)
        self._observations[obs.identity_id].sort(
            key=lambda o: o.timestamp or datetime.max.replace(tzinfo=timezone.utc)
        )

    # -- Derived spatial fields ---------------------------------------------

    def compute_activity_centroid(
        self, identity_id: str
    ) -> Optional[tuple[float, float]]:
        """Mean (latitude, longitude) of all trusted observations with GPS."""
        obs_list = self._observations.get(identity_id, [])
        coords = [
            (o.latitude, o.longitude)
            for o in obs_list
            if o.latitude is not None and o.longitude is not None
        ]
        if not coords:
            return None
        mean_lat = sum(c[0] for c in coords) / len(coords)
        mean_lon = sum(c[1] for c in coords) / len(coords)
        return (mean_lat, mean_lon)

    def compute_historical_capture_area(
        self, identity_id: str
    ) -> Optional[list[tuple[float, float]]]:
        """Convex hull of trusted observation coordinates.

        Returns a list of (lat, lon) vertices forming the convex hull,
        or None if fewer than 3 distinct GPS points exist.

        This is called 'historical capture area' per PROJECT_CONTRACT.md
        Section 13 — never 'home range' or 'validated home range.'
        """
        obs_list = self._observations.get(identity_id, [])
        # Collect unique GPS points
        points: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for o in obs_list:
            if o.latitude is not None and o.longitude is not None:
                pt = (o.latitude, o.longitude)
                if pt not in seen:
                    points.append(pt)
                    seen.add(pt)
        if len(points) < 3:
            return None
        return _convex_hull(points)

    # -- Reset (for testing) ------------------------------------------------

    def clear(self) -> None:
        """Remove all stored observations. Used only in tests."""
        self._observations.clear()


# ---------------------------------------------------------------------------
# Module-level singleton history instance
# ---------------------------------------------------------------------------

_history = TrustedHistory()


def get_history() -> TrustedHistory:
    """Return the module-level TrustedHistory singleton."""
    return _history


def reset_history() -> None:
    """Reset the module-level history. Used only in tests and demo reset."""
    _history.clear()


# ---------------------------------------------------------------------------
# SAFETY ENFORCEMENT POINT — update_trusted_history
# ---------------------------------------------------------------------------

def update_trusted_history(
    decision: IdentityDecision,
    image_metadata: Optional[dict] = None,
) -> Optional[Observation]:
    """Create a trusted Observation from an IdentityDecision, if and only
    if the decision is a trusted_match with update_history=True.

    CRITICAL SAFETY RULE (PROJECT_CONTRACT.md Section 5):
    This function is THE enforcement point. The guard clause is at the
    top — not buried logic.

    Parameters
    ----------
    decision : IdentityDecision
        The identity decision from Developer 2's gating pipeline.
    image_metadata : dict, optional
        Additional metadata (station_id, latitude, longitude, timestamp,
        camera_status, quality_score) from the image/station context.
        If None, falls back to minimal defaults.

    Returns
    -------
    Observation or None
        The created Observation if trusted, None otherwise. None means
        the decision was correctly blocked from entering trusted history.
    """
    # ===================================================================
    # GUARD CLAUSE — SAFETY RULE — DO NOT WEAKEN, REMOVE, OR BYPASS
    # Only trusted_match with update_history=True may proceed.
    # ===================================================================
    if not decision.update_history:
        return None
    if decision.decision != IdentityDecisionState.TRUSTED_MATCH:
        return None
    # ===================================================================

    meta = image_metadata or {}

    obs = Observation(
        observation_id=f"obs_{uuid.uuid4().hex[:12]}",
        image_id=decision.image_id,
        identity_id=decision.identity_id or "UNKNOWN",
        station_id=meta.get("station_id"),
        latitude=meta.get("latitude"),
        longitude=meta.get("longitude"),
        timestamp=meta.get("timestamp"),
        identity_confidence=decision.confidence,
        observation_status=ObservationStatus.TRUSTED,
        camera_status=CameraStatus(meta.get("camera_status", "unknown")),
        quality_score=meta.get("quality_score", 0.0),
    )

    _history.add_observation(obs)
    return obs


# ---------------------------------------------------------------------------
# Individual summary
# ---------------------------------------------------------------------------

def compute_individual_summary(identity_id: str) -> Optional[IndividualSummary]:
    """Compute a summary of an individual's trusted observation history.

    Returns None if no trusted observations exist for this identity.
    """
    observations = _history.get_observations(identity_id)
    if not observations:
        return None

    first_seen = None
    last_seen = None
    for obs in observations:
        if obs.timestamp is not None:
            if first_seen is None or obs.timestamp < first_seen:
                first_seen = obs.timestamp
            if last_seen is None or obs.timestamp > last_seen:
                last_seen = obs.timestamp

    last_seen_duration_days = None
    if last_seen is not None:
        now = datetime.now(timezone.utc)
        delta = now - last_seen
        last_seen_duration_days = delta.total_seconds() / 86400.0

    # Camera effort: station_id -> count of observations at that station
    camera_effort: dict[str, int] = {}
    for obs in observations:
        sid = obs.station_id or "UNKNOWN_STATION"
        camera_effort[sid] = camera_effort.get(sid, 0) + 1

    return IndividualSummary(
        identity_id=identity_id,
        capture_count=_history.get_capture_count(identity_id),
        first_seen=first_seen,
        last_seen=last_seen,
        trusted_stations=_history.get_trusted_stations(identity_id),
        activity_centroid=_history.compute_activity_centroid(identity_id),
        historical_capture_area=_history.compute_historical_capture_area(
            identity_id
        ),
        last_seen_duration_days=last_seen_duration_days,
        camera_effort_history=camera_effort,
    )


# ---------------------------------------------------------------------------
# Convex hull — simple Graham scan, no GIS dependency
# ---------------------------------------------------------------------------

def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain convex hull algorithm.
    Returns vertices in counter-clockwise order."""
    pts = sorted(points)
    if len(pts) <= 1:
        return pts

    # Build lower hull
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove last point of each half because it's repeated
    return lower[:-1] + upper[:-1]
