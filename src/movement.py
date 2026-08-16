"""
src/movement.py — Movement deviation detection.

Developer 3 owns this file.

Computes distances, travel speeds, and deviations between successive
trusted observations for a given individual. Produces MovementDeviation
records consumed by src/alerts.py for alert generation/suppression.

All distance computations use the Haversine formula — no GIS dependency
required (per PROJECT_CONTRACT.md Section 16 / cut rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.schemas import AlertType, Observation


# ---------------------------------------------------------------------------
# Configuration — all thresholds are configurable
# ---------------------------------------------------------------------------

@dataclass
class MovementConfig:
    """Configurable thresholds for movement deviation detection.

    Thresholds conform to the SIH problem statement requirements:
    - Core zone deviation threshold: 15-20 sq km (default: 17.5 sq km)
    - Buffer zone deviation threshold: 5 km (default: 5.0 km)

    These are prototype heuristics (PROJECT_CONTRACT.md Section 12),
    NOT scientifically validated parameters.
    """

    # Core zone area deviation threshold (sq km) — SIH problem statement benchmark (15-20 sq km)
    core_zone_deviation_sqkm: float = 17.5

    # Buffer zone distance deviation threshold (km) — SIH problem statement benchmark (5 km)
    buffer_zone_deviation_km: float = 5.0

    # Distance threshold for UNUSUAL_TRAVEL (km)
    unusual_distance_km: float = 30.0

    # Speed threshold for UNUSUAL_TRAVEL (km/h).
    # Tigers can travel ~60km/day during dispersal, but normal patrol
    # territory movement is much less. 10 km/h is a conservative flag.
    unusual_speed_kmh: float = 10.0

    # Minimum days of absence to trigger PROLONGED_ABSENCE
    prolonged_absence_days: float = 90.0

    # Minimum number of trusted observations before spatial deviations
    # (OUTSIDE_HISTORICAL_AREA) are meaningful
    min_observations_for_spatial: int = 3

    # Minimum number of trusted observations before any deviation alert
    min_observations_for_alert: int = 2


# ---------------------------------------------------------------------------
# Internal deviation record
# ---------------------------------------------------------------------------

@dataclass
class MovementDeviation:
    """Internal record of a detected movement deviation.
    Consumed by src/alerts.py to decide whether to generate, suppress,
    or downgrade an alert."""

    deviation_type: AlertType
    magnitude: float  # km for distance, days for absence
    details: dict = field(default_factory=dict)
    triggering_observation: Optional[Observation] = None


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0


def compute_distance(obs_a: Observation, obs_b: Observation) -> Optional[float]:
    """Haversine distance in km between two observations.

    Returns None if either observation lacks GPS coordinates.
    """
    if (
        obs_a.latitude is None
        or obs_a.longitude is None
        or obs_b.latitude is None
        or obs_b.longitude is None
    ):
        return None

    lat1, lon1 = math.radians(obs_a.latitude), math.radians(obs_a.longitude)
    lat2, lon2 = math.radians(obs_b.latitude), math.radians(obs_b.longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_KM * c


def compute_travel_speed(
    obs_a: Observation, obs_b: Observation
) -> Optional[float]:
    """Implied travel speed in km/h between two observations.

    Returns None if distance can't be computed, timestamps are missing,
    or time delta is zero.
    """
    dist = compute_distance(obs_a, obs_b)
    if dist is None:
        return None
    if obs_a.timestamp is None or obs_b.timestamp is None:
        return None

    # Ensure both timestamps are timezone-aware for subtraction
    t_a = obs_a.timestamp
    t_b = obs_b.timestamp
    if t_a.tzinfo is None:
        t_a = t_a.replace(tzinfo=timezone.utc)
    if t_b.tzinfo is None:
        t_b = t_b.replace(tzinfo=timezone.utc)

    delta_hours = abs((t_b - t_a).total_seconds()) / 3600.0
    if delta_hours == 0:
        return None
    return dist / delta_hours


# ---------------------------------------------------------------------------
# Point-in-polygon (ray casting) — for historical capture area check
# ---------------------------------------------------------------------------

def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    """Ray-casting algorithm to test if a point is inside a polygon.
    Simple implementation — no GIS dependency."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Main deviation detection
# ---------------------------------------------------------------------------

def detect_deviations(
    identity_id: str,
    history_observations: list[Observation],
    new_obs: Observation,
    historical_capture_area: Optional[list[tuple[float, float]]],
    station_context: Optional[dict] = None,
    config: Optional[MovementConfig] = None,
) -> list[MovementDeviation]:
    """Detect movement deviations for a new trusted observation against
    the individual's history.

    Parameters
    ----------
    identity_id : str
        The tiger's identity ID.
    history_observations : list[Observation]
        All prior trusted observations for this individual (sorted by time).
    new_obs : Observation
        The new trusted observation being evaluated.
    historical_capture_area : list of (lat, lon) or None
        Convex hull of prior trusted station coordinates.
    station_context : dict, optional
        Station metadata. Keys may include:
        - "buffer_stations": set of station_ids near buffer/village
        - "relocated_stations": dict of station_id -> relocation_datetime
        - "active_stations": dict of station_id -> list of active periods
    config : MovementConfig, optional
        Thresholds. Uses defaults if None.

    Returns
    -------
    list[MovementDeviation]
        All detected deviations. May be empty if nothing unusual.
    """
    if config is None:
        config = MovementConfig()
    if station_context is None:
        station_context = {}

    deviations: list[MovementDeviation] = []

    # Collect known trusted stations
    trusted_stations: set[str] = set()
    for obs in history_observations:
        if obs.station_id:
            trusted_stations.add(obs.station_id)

    # 1. NEW_STATION: station_id not previously seen for this individual
    if (
        new_obs.station_id
        and new_obs.station_id not in trusted_stations
    ):
        deviations.append(
            MovementDeviation(
                deviation_type=AlertType.NEW_STATION,
                magnitude=0.0,
                details={
                    "new_station": new_obs.station_id,
                    "known_stations": sorted(trusted_stations),
                    "total_prior_observations": len(history_observations),
                },
                triggering_observation=new_obs,
            )
        )

    # 2. OUTSIDE_HISTORICAL_AREA: only if enough prior stations
    if (
        historical_capture_area is not None
        and len(historical_capture_area) >= 3
        and new_obs.latitude is not None
        and new_obs.longitude is not None
        and len(history_observations) >= config.min_observations_for_spatial
    ):
        point = (new_obs.latitude, new_obs.longitude)
        if not _point_in_polygon(point, historical_capture_area):
            # Compute distance to centroid for magnitude
            centroid_lat = sum(p[0] for p in historical_capture_area) / len(
                historical_capture_area
            )
            centroid_lon = sum(p[1] for p in historical_capture_area) / len(
                historical_capture_area
            )
            # Create a temporary observation for distance calc
            centroid_obs = Observation(
                observation_id="centroid",
                image_id="centroid",
                identity_id=identity_id,
                latitude=centroid_lat,
                longitude=centroid_lon,
                identity_confidence=1.0,
            )
            dist = compute_distance(new_obs, centroid_obs)

            deviations.append(
                MovementDeviation(
                    deviation_type=AlertType.OUTSIDE_HISTORICAL_AREA,
                    magnitude=dist or 0.0,
                    details={
                        "observation_lat": new_obs.latitude,
                        "observation_lon": new_obs.longitude,
                        "area_vertices": len(historical_capture_area),
                        "distance_from_centroid_km": round(dist, 2)
                        if dist
                        else None,
                    },
                    triggering_observation=new_obs,
                )
            )

    # 3. UNUSUAL_TRAVEL: distance/speed to previous observation
    if history_observations:
        prev_obs = history_observations[-1]
        dist = compute_distance(prev_obs, new_obs)
        speed = compute_travel_speed(prev_obs, new_obs)

        if dist is not None and dist > config.unusual_distance_km:
            deviations.append(
                MovementDeviation(
                    deviation_type=AlertType.UNUSUAL_TRAVEL,
                    magnitude=dist,
                    details={
                        "distance_km": round(dist, 2),
                        "speed_kmh": round(speed, 2) if speed else None,
                        "threshold_km": config.unusual_distance_km,
                        "from_station": prev_obs.station_id,
                        "to_station": new_obs.station_id,
                        "from_time": str(prev_obs.timestamp)
                        if prev_obs.timestamp
                        else None,
                        "to_time": str(new_obs.timestamp)
                        if new_obs.timestamp
                        else None,
                    },
                    triggering_observation=new_obs,
                )
            )
        elif speed is not None and speed > config.unusual_speed_kmh:
            deviations.append(
                MovementDeviation(
                    deviation_type=AlertType.UNUSUAL_TRAVEL,
                    magnitude=speed,
                    details={
                        "distance_km": round(dist, 2) if dist else None,
                        "speed_kmh": round(speed, 2),
                        "threshold_kmh": config.unusual_speed_kmh,
                        "from_station": prev_obs.station_id,
                        "to_station": new_obs.station_id,
                    },
                    triggering_observation=new_obs,
                )
            )

    # 4. PROLONGED_ABSENCE: time since last seen > threshold,
    #    but only if cameras at known stations were active
    if history_observations:
        prev_obs = history_observations[-1]
        if prev_obs.timestamp is not None and new_obs.timestamp is not None:
            t_prev = prev_obs.timestamp
            t_new = new_obs.timestamp
            if t_prev.tzinfo is None:
                t_prev = t_prev.replace(tzinfo=timezone.utc)
            if t_new.tzinfo is None:
                t_new = t_new.replace(tzinfo=timezone.utc)

            absence_days = (t_new - t_prev).total_seconds() / 86400.0

            if absence_days > config.prolonged_absence_days:
                # Check if cameras were active during absence window
                active_stations = station_context.get("active_stations", {})
                cameras_active_during_absence = _cameras_active_during_window(
                    trusted_stations, active_stations, t_prev, t_new
                )

                deviations.append(
                    MovementDeviation(
                        deviation_type=AlertType.PROLONGED_ABSENCE,
                        magnitude=absence_days,
                        details={
                            "absence_days": round(absence_days, 1),
                            "threshold_days": config.prolonged_absence_days,
                            "last_station": prev_obs.station_id,
                            "cameras_active_during_absence": cameras_active_during_absence,
                        },
                        triggering_observation=new_obs,
                    )
                )

    # 5. BUFFER_OR_VILLAGE_ADJACENT: station flagged in context
    buffer_stations = station_context.get("buffer_stations", set())
    if new_obs.station_id and new_obs.station_id in buffer_stations:
        deviations.append(
            MovementDeviation(
                deviation_type=AlertType.BUFFER_OR_VILLAGE_ADJACENT,
                magnitude=0.0,
                details={
                    "station_id": new_obs.station_id,
                    "note": "Station flagged as buffer-zone or village-adjacent",
                },
                triggering_observation=new_obs,
            )
        )

    return deviations


# ---------------------------------------------------------------------------
# Helper: were cameras active during an absence window?
# ---------------------------------------------------------------------------

def _cameras_active_during_window(
    trusted_stations: set[str],
    active_stations: dict,
    start: datetime,
    end: datetime,
) -> bool:
    """Check if any of the individual's known stations had active cameras
    during the absence window [start, end].

    active_stations: dict of station_id -> list of
        {"start": datetime, "end": datetime} periods when camera was active.

    If active_stations is empty, we assume cameras WERE active (conservative
    — we'd rather flag the absence than silently suppress).
    """
    if not active_stations:
        # No camera activity data → assume active (conservative)
        return True

    for station_id in trusted_stations:
        periods = active_stations.get(station_id, [])
        for period in periods:
            p_start = period.get("start")
            p_end = period.get("end")
            if p_start is None or p_end is None:
                continue
            # Check overlap: period overlaps [start, end]
            if p_start <= end and p_end >= start:
                return True
    return False
