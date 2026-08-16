"""
src/alerts.py — Alert generation and suppression.

Developer 3 owns this file.

Generates movement alerts ONLY when ALL of the following hold
(PROJECT_CONTRACT.md Sections 14–15):
  - identity is trusted
  - camera was active at the time
  - station was not recently relocated
  - image quality was adequate
  - there is sufficient historical data for that individual
  - the deviation exceeds a configurable threshold

Otherwise: downgrade to INSUFFICIENT_EVIDENCE, or suppress the alert
entirely with a filled suppression_reason.

Every alert's explanation field plainly states which evidence produced it.
Alerts distinguish: likely biological signal vs. likely observation/survey
artefact vs. insufficient evidence vs. human review required.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.schemas import (
    AlertStatus,
    AlertType,
    CameraStatus,
    MovementAlert,
    Observation,
    ObservationStatus,
)
from src.movement import MovementDeviation


# ---------------------------------------------------------------------------
# Alert configuration
# ---------------------------------------------------------------------------

@dataclass
class AlertConfig:
    """Configurable thresholds for alert generation and suppression.
    These are prototype heuristics, NOT scientifically validated."""

    # Minimum image quality score to trust an observation for alerting
    min_quality_score: float = 0.4

    # Minimum identity confidence to generate an alert
    min_identity_confidence: float = 0.5

    # Minimum trusted observations before generating spatial alerts
    min_observations_for_alert: int = 2

    # Days since station relocation before alerts are re-enabled
    relocation_cooldown_days: float = 30.0


# ---------------------------------------------------------------------------
# Main alert generation
# ---------------------------------------------------------------------------

def generate_alerts(
    identity_id: str,
    deviations: list[MovementDeviation],
    observation: Observation,
    capture_count: int,
    station_context: Optional[dict] = None,
    config: Optional[AlertConfig] = None,
) -> list[MovementAlert]:
    """Generate or suppress alerts for detected movement deviations.

    For each deviation, checks ALL suppression conditions. If ANY fails,
    the alert is either downgraded to INSUFFICIENT_EVIDENCE or suppressed
    entirely with a filled suppression_reason.

    Parameters
    ----------
    identity_id : str
        The tiger's identity ID.
    deviations : list[MovementDeviation]
        Deviations detected by movement.py for this observation.
    observation : Observation
        The triggering trusted observation.
    capture_count : int
        Total trusted observations for this individual (including current).
    station_context : dict, optional
        Station metadata:
        - "relocated_stations": dict of station_id -> relocation_datetime
        - "active_stations": dict of station_id -> list of active periods
        - "buffer_stations": set of station_ids
    config : AlertConfig, optional
        Thresholds. Uses defaults if None.

    Returns
    -------
    list[MovementAlert]
        Generated, suppressed, or downgraded alerts.
    """
    if config is None:
        config = AlertConfig()
    if station_context is None:
        station_context = {}

    alerts: list[MovementAlert] = []

    for deviation in deviations:
        alert = _evaluate_deviation(
            identity_id=identity_id,
            deviation=deviation,
            observation=observation,
            capture_count=capture_count,
            station_context=station_context,
            config=config,
        )
        alerts.append(alert)

    return alerts


def _evaluate_deviation(
    identity_id: str,
    deviation: MovementDeviation,
    observation: Observation,
    capture_count: int,
    station_context: dict,
    config: AlertConfig,
) -> MovementAlert:
    """Evaluate a single deviation and produce an alert, suppressed alert,
    or downgraded alert."""

    suppression_reasons: list[str] = []
    alert_id = f"alert_{uuid.uuid4().hex[:12]}"

    # --- Check all suppression conditions ---

    # 1. Identity must be trusted
    if observation.observation_status != ObservationStatus.TRUSTED:
        suppression_reasons.append(
            f"Observation status is '{observation.observation_status.value}', "
            "not trusted"
        )

    # 2. Camera must have been active at observation time
    if observation.camera_status != CameraStatus.ACTIVE:
        suppression_reasons.append(
            f"Camera status was '{observation.camera_status.value}' at time "
            "of observation"
        )

    # 3. Station must not have been recently relocated
    relocated_stations = station_context.get("relocated_stations", {})
    if observation.station_id and observation.station_id in relocated_stations:
        relocation_time = relocated_stations[observation.station_id]
        if observation.timestamp and relocation_time:
            obs_time = observation.timestamp
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
            rel_time = relocation_time
            if isinstance(rel_time, datetime) and rel_time.tzinfo is None:
                rel_time = rel_time.replace(tzinfo=timezone.utc)
            if isinstance(rel_time, datetime):
                days_since = (obs_time - rel_time).total_seconds() / 86400.0
                if days_since < config.relocation_cooldown_days:
                    suppression_reasons.append(
                        f"Camera at station '{observation.station_id}' relocated "
                        f"{days_since:.0f} days ago (cooldown: "
                        f"{config.relocation_cooldown_days:.0f} days)"
                    )

    # 4. Image quality must be adequate
    if observation.quality_score < config.min_quality_score:
        suppression_reasons.append(
            f"Image quality score ({observation.quality_score:.2f}) below "
            f"threshold ({config.min_quality_score:.2f})"
        )

    # 5. Sufficient historical data for this individual
    if capture_count < config.min_observations_for_alert:
        suppression_reasons.append(
            f"Only {capture_count} trusted observation(s) on record "
            f"(minimum: {config.min_observations_for_alert})"
        )

    # 6. Identity confidence must be adequate
    if observation.identity_confidence < config.min_identity_confidence:
        suppression_reasons.append(
            f"Identity confidence ({observation.identity_confidence:.2f}) "
            f"below threshold ({config.min_identity_confidence:.2f})"
        )

    # --- Special case for PROLONGED_ABSENCE: check camera activity ---
    if deviation.deviation_type == AlertType.PROLONGED_ABSENCE:
        cameras_active = deviation.details.get(
            "cameras_active_during_absence", True
        )
        if not cameras_active:
            suppression_reasons.append(
                "Cameras at known stations were inactive during the "
                "absence window — absence may be a survey artefact, "
                "not a biological signal"
            )

    # --- Decide alert status ---
    if suppression_reasons:
        # Determine whether to suppress or downgrade
        is_artefact = any(
            "relocated" in r.lower() or "inactive" in r.lower()
            for r in suppression_reasons
        )
        is_insufficient = any(
            "only" in r.lower() or "quality" in r.lower()
            or "confidence" in r.lower()
            for r in suppression_reasons
        )

        if is_artefact:
            status = AlertStatus.SUPPRESSED
            alert_type = AlertType.CAMERA_OR_SURVEY_ARTEFACT
        elif is_insufficient:
            status = AlertStatus.INSUFFICIENT_EVIDENCE
            alert_type = AlertType.INSUFFICIENT_EVIDENCE
        else:
            status = AlertStatus.HUMAN_REVIEW_REQUIRED
            alert_type = deviation.deviation_type

        explanation = _build_explanation(
            deviation=deviation,
            observation=observation,
            capture_count=capture_count,
            is_suppressed=True,
            suppression_reasons=suppression_reasons,
        )
        suppression_reason_text = "; ".join(suppression_reasons)

        return MovementAlert(
            alert_id=alert_id,
            identity_id=identity_id,
            alert_type=alert_type,
            confidence=observation.identity_confidence,
            status=status,
            evidence_observation_ids=[observation.observation_id],
            explanation=explanation,
            suppression_reason=suppression_reason_text,
        )

    # --- No suppression: generate active alert ---
    explanation = _build_explanation(
        deviation=deviation,
        observation=observation,
        capture_count=capture_count,
        is_suppressed=False,
        suppression_reasons=[],
    )

    return MovementAlert(
        alert_id=alert_id,
        identity_id=identity_id,
        alert_type=deviation.deviation_type,
        confidence=observation.identity_confidence,
        status=AlertStatus.ACTIVE,
        evidence_observation_ids=[observation.observation_id],
        explanation=explanation,
        suppression_reason=None,
    )


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

def _build_explanation(
    deviation: MovementDeviation,
    observation: Observation,
    capture_count: int,
    is_suppressed: bool,
    suppression_reasons: list[str],
) -> str:
    """Build a human-readable explanation for an alert.

    Plainly states which evidence produced the alert:
    - identity confidence
    - station history
    - camera status
    - deviation magnitude

    Classifies signal as: biological signal / observation artefact /
    insufficient evidence / human review required.
    """
    parts: list[str] = []

    # Signal type classification
    if is_suppressed:
        is_artefact = any(
            "relocated" in r.lower() or "inactive" in r.lower()
            for r in suppression_reasons
        )
        if is_artefact:
            parts.append(
                "CLASSIFICATION: Likely observation/survey artefact."
            )
        else:
            parts.append(
                "CLASSIFICATION: Insufficient evidence for confident alert."
            )
    else:
        parts.append("CLASSIFICATION: Likely biological signal.")

    # Deviation description
    dtype = deviation.deviation_type.value
    details = deviation.details

    if deviation.deviation_type == AlertType.NEW_STATION:
        parts.append(
            f"DEVIATION: {dtype} — Individual first observed at station "
            f"'{details.get('new_station', '?')}'. "
            f"Previously seen at {len(details.get('known_stations', []))} "
            f"station(s): {details.get('known_stations', [])}."
        )
    elif deviation.deviation_type == AlertType.OUTSIDE_HISTORICAL_AREA:
        parts.append(
            f"DEVIATION: {dtype} — Observation at "
            f"({details.get('observation_lat', '?')}, "
            f"{details.get('observation_lon', '?')}) is outside the "
            f"historical capture area ({details.get('area_vertices', '?')} "
            f"vertices). Distance from centroid: "
            f"{details.get('distance_from_centroid_km', '?')} km."
        )
    elif deviation.deviation_type == AlertType.UNUSUAL_TRAVEL:
        parts.append(
            f"DEVIATION: {dtype} — Travel of "
            f"{details.get('distance_km', '?')} km "
            f"from '{details.get('from_station', '?')}' to "
            f"'{details.get('to_station', '?')}'. "
            f"Speed: {details.get('speed_kmh', '?')} km/h."
        )
    elif deviation.deviation_type == AlertType.PROLONGED_ABSENCE:
        parts.append(
            f"DEVIATION: {dtype} — No trusted observation for "
            f"{details.get('absence_days', '?')} days "
            f"(threshold: {details.get('threshold_days', '?')} days). "
            f"Last seen at '{details.get('last_station', '?')}'."
        )
    elif deviation.deviation_type == AlertType.BUFFER_OR_VILLAGE_ADJACENT:
        parts.append(
            f"DEVIATION: {dtype} — Observed at buffer/village-adjacent "
            f"station '{details.get('station_id', '?')}'."
        )
    else:
        parts.append(f"DEVIATION: {dtype} — magnitude {deviation.magnitude}.")

    # Evidence summary
    parts.append(
        f"EVIDENCE: Identity confidence={observation.identity_confidence:.2f}, "
        f"camera_status={observation.camera_status.value}, "
        f"quality_score={observation.quality_score:.2f}, "
        f"capture_count={capture_count}."
    )

    # Suppression detail
    if is_suppressed:
        parts.append(
            f"SUPPRESSION: {'; '.join(suppression_reasons)}."
        )

    return " ".join(parts)
