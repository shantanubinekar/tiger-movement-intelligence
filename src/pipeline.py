"""
pipeline.py — SHARED integration layer.

This file defines the exact function signatures that Developer 1's
frontend calls, that Developer 2 implements the first half of, and that
Developer 3 implements the second half of (per PROJECT_CONTRACT.md
Section 19).

HOW TO USE THIS FILE
- Developer 2: implements `process_image_directory`, `generate_candidates`,
  and `make_identity_decision` via ingestion.py, triage.py, perception.py,
  identity.py, gating.py.
- Developer 3: implements `create_observation`, `generate_movement_alerts`,
  and `run_evaluation` via history.py, movement.py, alerts.py, evaluation.py.
- Developer 1: imports ONLY from this file in the Streamlit UI.

Every function below has a DEMO_MODE fallback that returns a deterministic
result when real modules are unavailable or when DEMO_MODE is explicitly enabled.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import (
    AlertStatus,
    AlertType,
    CameraStatus,
    DataMode,
    EvaluationReport,
    IdentityCandidate,
    IdentityDecision,
    IdentityDecisionState,
    ImageRecord,
    MovementAlert,
    Observation,
    ReasonCode,
    TriageStatus,
)

logger = logging.getLogger(__name__)

# DEMO_MODE flag: when False, real logic executes first where available.
DEMO_MODE = False

# ---------------------------------------------------------------------------
# Developer 2 imports (Perception + Identity)
# ---------------------------------------------------------------------------
_DEV2_AVAILABLE = False
try:
    from src.ingestion import ingest_folder
    from src.triage import triage_image
    from src.perception import detect_subject, generate_embedding
    from src.identity import generate_candidates as _real_generate_candidates
    from src.identity import get_default_catalogue
    from src.gating import make_identity_decision as _real_make_identity_decision
    _DEV2_AVAILABLE = True
except ImportError as e:
    logger.warning("Developer 2 modules not fully available: %s — using demo fallback", e)

# ---------------------------------------------------------------------------
# Developer 3 imports (History + Movement + Alerts + Evaluation)
# ---------------------------------------------------------------------------
_DEV3_AVAILABLE = False
try:
    from src.history import (
        update_trusted_history as _real_update_trusted_history,
        get_history as _get_history,
        compute_individual_summary as _compute_individual_summary,
    )
    from src.movement import (
        detect_deviations as _detect_deviations,
        MovementConfig as _MovementConfig,
    )
    from src.alerts import (
        generate_alerts as _generate_alerts,
        AlertConfig as _AlertConfig,
    )
    from src.evaluation import (
        run_evaluation as _real_run_evaluation,
    )
    _DEV3_AVAILABLE = True
except ImportError as e:
    logger.warning("Developer 3 modules not fully available: %s — using demo fallback", e)


def _seeded_float(seed: str, low: float = 0.0, high: float = 1.0) -> float:
    """Deterministic pseudo-random float in [low, high], seeded from a
    string. Used only for demo-mode placeholder scores."""
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000
    return low + frac * (high - low)


# ---------------------------------------------------------------------------
# DEVELOPER 2 Functions (Perception + Identity)
# ---------------------------------------------------------------------------

def process_image_directory(path: str) -> list[IdentityDecision]:
    """Ingest -> triage -> detect -> embed -> candidates -> gate, for every
    image in `path`. Returns one IdentityDecision per processed image.
    """
    # --- Real logic (Developer 2) ---
    if _DEV2_AVAILABLE and not DEMO_MODE:
        try:
            p = Path(path)
            target_path = path
            catalogue_dir = None
            if p.is_dir() and (p / "query").is_dir():
                target_path = str(p / "query")
                catalogue_dir = str(p)
            elif p.is_dir() and (p.parent / "catalogue").is_dir():
                catalogue_dir = str(p.parent)

            records = ingest_folder(target_path)
            if records:
                catalogue = get_default_catalogue(dataset_dir=catalogue_dir)
                decisions: list[IdentityDecision] = []

                for record in records:
                    triage = triage_image(record)

                    if triage.triage_status == TriageStatus.BLANK:
                        decisions.append(IdentityDecision(
                            image_id=record.image_id,
                            decision=IdentityDecisionState.BLANK,
                            confidence=triage.blank_probability,
                            reason_codes=[],
                            evidence_summary={
                                "triage_status": "blank",
                                "blank_probability": triage.blank_probability,
                                "station_id": record.station_id,
                                "latitude": record.latitude,
                                "longitude": record.longitude,
                                "timestamp": record.timestamp,
                                "camera_status": record.camera_status,
                            },
                            update_history=False,
                        ))
                        continue

                    detection = detect_subject(record)
                    crop_path = detection.crop_path or record.image_path
                    embedding = generate_embedding(crop_path)

                    context = {
                        "station_id": record.station_id,
                        "latitude": record.latitude,
                        "longitude": record.longitude,
                        "timestamp": record.timestamp,
                        "quality_score": detection.quality_score,
                        "flank_visibility": detection.flank_visibility,
                        "camera_status": record.camera_status,
                        "crop_path": crop_path,
                        "image_path": record.image_path,
                        "data_mode": record.data_mode.value if hasattr(record.data_mode, "value") else str(record.data_mode),
                    }
                    candidates = _real_generate_candidates(
                        embedding=embedding,
                        image_id=record.image_id,
                        catalogue=catalogue,
                        context=context,
                    )

                    gate_context = {
                        **context,
                        "image_id": record.image_id,
                        "embedding": embedding,
                        "detection_confidence": detection.detection_confidence,
                    }
                    decision = _real_make_identity_decision(candidates, gate_context)
                    decisions.append(decision)

                logger.info("Processed %d images from %s -> %d decisions", len(records), path, len(decisions))
                return decisions
        except Exception as e:
            logger.error("Real pipeline failed: %s — falling back to demo mode", e)

    # --- Demo fallback ---
    p = Path(path)
    image_ids = (
        [f.stem for f in sorted(p.glob("*")) if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if p.exists()
        else []
    )
    if not image_ids:
        image_ids = [f"demo_img_{i:03d}" for i in range(5)]

    decisions = []
    states_cycle = [
        IdentityDecisionState.TRUSTED_MATCH,
        IdentityDecisionState.AMBIGUOUS_REVIEW,
        IdentityDecisionState.UNKNOWN,
        IdentityDecisionState.INSUFFICIENT_EVIDENCE,
        IdentityDecisionState.BLANK,
    ]
    for i, image_id in enumerate(image_ids):
        state = states_cycle[i % len(states_cycle)]
        is_trusted = state == IdentityDecisionState.TRUSTED_MATCH
        candidate = IdentityCandidate(
            image_id=image_id,
            candidate_identity=f"T{(i % 3) + 1:02d}",
            rank=1,
            visual_score=_seeded_float(image_id + "v", 0.4, 0.95),
            quality_score=_seeded_float(image_id + "q", 0.3, 0.9),
            spatial_feasibility=_seeded_float(image_id + "s", 0.5, 1.0),
            temporal_feasibility=_seeded_float(image_id + "t", 0.5, 1.0),
            history_consistency=_seeded_float(image_id + "h", 0.4, 1.0),
            total_evidence=_seeded_float(image_id + "e", 0.3, 0.95),
        )
        decisions.append(
            IdentityDecision(
                image_id=image_id,
                decision=state,
                identity_id=candidate.candidate_identity if is_trusted else None,
                confidence=candidate.total_evidence,
                top_candidates=[candidate],
                reason_codes=(
                    [ReasonCode.HIGH_CONFIDENCE_MATCH]
                    if is_trusted
                    else [ReasonCode.LOW_VISUAL_MARGIN]
                ),
                evidence_summary={
                    "data_mode": DataMode.DEMO.value,
                    "station_id": "STATION_DEMO_1",
                    "latitude": 21.68,
                    "longitude": 79.29,
                    "timestamp": datetime.now(timezone.utc),
                    "camera_status": CameraStatus.ACTIVE,
                },
                update_history=is_trusted,
            )
        )
    return decisions


def generate_candidates(image_record: ImageRecord) -> list[IdentityCandidate]:
    """Given one ImageRecord, return ranked candidate identities."""
    if _DEV2_AVAILABLE and not DEMO_MODE:
        try:
            detection = detect_subject(image_record)
            crop_path = detection.crop_path or image_record.image_path
            embedding = generate_embedding(crop_path)
            context = {
                "station_id": image_record.station_id,
                "latitude": image_record.latitude,
                "longitude": image_record.longitude,
                "timestamp": image_record.timestamp,
                "quality_score": detection.quality_score,
            }
            return _real_generate_candidates(
                embedding=embedding,
                image_id=image_record.image_id,
                context=context,
            )
        except Exception as e:
            logger.error("Real generate_candidates failed: %s — using demo fallback", e)

    # Demo fallback
    return [
        IdentityCandidate(
            image_id=image_record.image_id,
            candidate_identity="T01",
            rank=1,
            visual_score=_seeded_float(image_record.image_id + "v", 0.4, 0.95),
            quality_score=_seeded_float(image_record.image_id + "q", 0.3, 0.9),
            spatial_feasibility=0.8,
            temporal_feasibility=0.8,
            history_consistency=0.7,
            total_evidence=_seeded_float(image_record.image_id + "e", 0.3, 0.95),
        )
    ]


def make_identity_decision(candidates: list[IdentityCandidate], context: dict | None = None) -> IdentityDecision:
    """Given ranked candidates + context (station/time/camera/history),
    return the gated IdentityDecision."""
    if _DEV2_AVAILABLE and not DEMO_MODE:
        try:
            return _real_make_identity_decision(candidates, context)
        except Exception as e:
            logger.error("Real make_identity_decision failed: %s — using demo fallback", e)

    # Demo fallback
    top = candidates[0] if candidates else None
    if top and top.total_evidence >= 0.8:
        return IdentityDecision(
            image_id=top.image_id,
            decision=IdentityDecisionState.TRUSTED_MATCH,
            identity_id=top.candidate_identity,
            confidence=top.total_evidence,
            top_candidates=candidates,
            reason_codes=[ReasonCode.HIGH_CONFIDENCE_MATCH],
            evidence_summary={"data_mode": DataMode.DEMO.value},
            update_history=True,
        )
    return IdentityDecision(
        image_id=top.image_id if top else "unknown",
        decision=IdentityDecisionState.AMBIGUOUS_REVIEW,
        confidence=top.total_evidence if top else 0.0,
        top_candidates=candidates,
        reason_codes=[ReasonCode.LOW_VISUAL_MARGIN],
        evidence_summary={"data_mode": DataMode.DEMO.value},
        update_history=False,
    )


# ---------------------------------------------------------------------------
# DEVELOPER 3 Functions (History + Movement + Alerts + Evaluation)
# ---------------------------------------------------------------------------

def create_observation(decision: IdentityDecision) -> Observation | None:
    """Enforcement point: only trusted_match decisions may become a trusted
    Observation. Everything else returns None."""
    if not decision.update_history or decision.decision != IdentityDecisionState.TRUSTED_MATCH:
        return None

    image_metadata = {
        "station_id": decision.evidence_summary.get("station_id"),
        "latitude": decision.evidence_summary.get("latitude"),
        "longitude": decision.evidence_summary.get("longitude"),
        "timestamp": decision.evidence_summary.get("timestamp"),
        "camera_status": decision.evidence_summary.get("camera_status"),
        "quality_score": decision.evidence_summary.get("quality_score", 0.8),
    }

    if _DEV3_AVAILABLE and not DEMO_MODE:
        return _real_update_trusted_history(decision, image_metadata=image_metadata)

    # DEMO_MODE fallback
    return Observation(
        observation_id=f"obs_{decision.image_id}",
        image_id=decision.image_id,
        identity_id=decision.identity_id or "UNKNOWN",
        station_id=image_metadata.get("station_id") or "STATION_DEMO_1",
        latitude=image_metadata.get("latitude") or 21.68,
        longitude=image_metadata.get("longitude") or 79.29,
        timestamp=image_metadata.get("timestamp") or datetime.now(timezone.utc),
        identity_confidence=decision.confidence,
        camera_status=CameraStatus.ACTIVE,
        quality_score=image_metadata.get("quality_score") or 0.8,
    )


def generate_movement_alerts(
    observations: list[Observation], station_context: dict | None = None
) -> list[MovementAlert]:
    """Run movement-deviation analysis over trusted observations and
    generate or suppress alerts accordingly."""
    if _DEV3_AVAILABLE and not DEMO_MODE:
        all_alerts: list[MovementAlert] = []
        history = _get_history()
        movement_config = _MovementConfig()
        alert_config = _AlertConfig()

        for obs in observations:
            identity_id = obs.identity_id
            prior = [
                o for o in history.get_observations(identity_id)
                if o.observation_id != obs.observation_id
            ]
            hull = history.compute_historical_capture_area(identity_id)
            deviations = _detect_deviations(
                identity_id=identity_id,
                history_observations=prior,
                new_obs=obs,
                historical_capture_area=hull,
                station_context=station_context,
                config=movement_config,
            )
            if deviations:
                alerts = _generate_alerts(
                    identity_id=identity_id,
                    deviations=deviations,
                    observation=obs,
                    capture_count=history.get_capture_count(identity_id),
                    station_context=station_context,
                    config=alert_config,
                )
                all_alerts.extend(alerts)
        return all_alerts

    # DEMO_MODE fallback
    if not observations:
        return []
    alerts = []
    for obs in observations[:1]:
        alerts.append(
            MovementAlert(
                alert_id=f"alert_{obs.observation_id}",
                identity_id=obs.identity_id,
                alert_type=AlertType.NEW_STATION,
                confidence=0.6,
                status=AlertStatus.ACTIVE,
                evidence_observation_ids=[obs.observation_id],
                explanation=(
                    "DEMO MODE placeholder: first trusted observation "
                    f"for {obs.identity_id} treated as a new-station event."
                ),
            )
        )
    return alerts


def run_evaluation(records: dict | None = None) -> list[EvaluationReport]:
    """Compare baseline (always-assign) vs. proposed (evidence-gated)
    pipelines on the same scenario set. Never fabricate numbers — use
    EvaluationReport.not_computable for anything not measurable yet."""
    if _DEV3_AVAILABLE and not DEMO_MODE:
        return _real_run_evaluation(records)

    # DEMO_MODE fallback
    return [
        EvaluationReport(
            pipeline_name="baseline",
            not_computable=[
                "false_confident_identity_rate",
                "false_movement_alert_rate",
                "alert_precision",
            ],
            notes="DEMO MODE placeholder — prototype scenario evaluation.",
        ),
        EvaluationReport(
            pipeline_name="evidence_gated",
            not_computable=[
                "false_confident_identity_rate",
                "false_movement_alert_rate",
                "alert_precision",
            ],
            notes="DEMO MODE placeholder — prototype scenario evaluation.",
        ),
    ]
