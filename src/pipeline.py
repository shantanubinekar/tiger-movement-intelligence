"""
pipeline.py — SHARED integration layer.

This file defines the exact function signatures that Developer 1's
frontend calls, that Developer 2 implements the first half of, and that
Developer 3 implements the second half of (per PROJECT_CONTRACT.md
Section 19).

HOW TO USE THIS FILE
- Developer 2: replace the bodies of `process_image_directory`,
  `generate_candidates`, and `make_identity_decision` with real logic
  from ingestion.py / triage.py / perception.py / identity.py / gating.py.
  Do not change the function names or return types.
- Developer 3: replace the bodies of `create_observation`,
  `generate_movement_alerts`, and `run_evaluation` with real logic from
  history.py / movement.py / alerts.py / evaluation.py. Do not change the
  function names or return types.
- Developer 1: import ONLY from this file (never from src/gating.py etc.
  directly) so the UI keeps working before/while Backend A and B are
  filled in.

Every function below has a DEMO_MODE fallback that returns a small,
clearly-fake, deterministic result instead of raising NotImplementedError.
This means `streamlit run app.py` and the integration smoke test both work
from hour 0, even before any real ML/logic is written. As you implement
the real logic, keep the DEMO_MODE branch working too — it is the Phase-1
demo fallback required by the project contract, not throwaway code.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
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

# Set to False once Developer 2 / Developer 3 wire in real logic for a given
# function. Keeping this per-module rather than global lets you flip pieces
# on independently as each branch lands.
DEMO_MODE = True

# Developer 2 real logic is now available — used when DEMO_MODE is False
# (or as the primary path when True, with demo fallback on import failure).
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


def _seeded_float(seed: str, low: float = 0.0, high: float = 1.0) -> float:
    """Deterministic pseudo-random float in [low, high], seeded from a
    string. Used only for demo-mode placeholder scores — never for real
    evidence computation."""
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000
    return low + frac * (high - low)


# ---------------------------------------------------------------------------
# DEVELOPER 2 owns the real implementation of these three.
# ---------------------------------------------------------------------------

def process_image_directory(path: str) -> list[IdentityDecision]:
    """Ingest -> triage -> detect -> embed -> candidates -> gate, for every
    image in `path`. Returns one IdentityDecision per processed image.

    DEMO_MODE: fabricates a handful of decisions covering each decision
    state, so downstream code and the UI have something real to render.
    """
    # --- Real logic (Developer 2) ---
    if _DEV2_AVAILABLE:
        try:
            # Step 1: Ingest folder
            records = ingest_folder(path)
            if not records:
                logger.info("No images found in %s — falling back to demo", path)
                # Fall through to demo mode below
            else:
                catalogue = get_default_catalogue()
                decisions: list[IdentityDecision] = []

                for record in records:
                    # Step 2: Triage
                    triage = triage_image(record)

                    # Skip blank images — mark as blank decision
                    if triage.triage_status == TriageStatus.BLANK:
                        decisions.append(IdentityDecision(
                            image_id=record.image_id,
                            decision=IdentityDecisionState.BLANK,
                            confidence=triage.blank_probability,
                            reason_codes=[],
                            evidence_summary={
                                "triage_status": "blank",
                                "blank_probability": triage.blank_probability,
                            },
                            update_history=False,
                        ))
                        continue

                    # Step 3: Detect subject + compute quality
                    detection = detect_subject(record)

                    # Step 4: Generate embedding
                    crop_path = detection.crop_path or record.image_path
                    embedding = generate_embedding(crop_path)

                    # Step 5: Generate candidates
                    context = {
                        "station_id": record.station_id,
                        "latitude": record.latitude,
                        "longitude": record.longitude,
                        "timestamp": record.timestamp,
                        "quality_score": detection.quality_score,
                        "flank_visibility": detection.flank_visibility,
                        "camera_status": record.camera_status,
                    }
                    candidates = _real_generate_candidates(
                        embedding=embedding,
                        image_id=record.image_id,
                        catalogue=catalogue,
                        context=context,
                    )

                    # Step 6: Gating decision
                    gate_context = {
                        **context,
                        "image_id": record.image_id,
                        "embedding": embedding,
                        "detection_confidence": detection.detection_confidence,
                    }
                    decision = _real_make_identity_decision(candidates, gate_context)
                    decisions.append(decision)

                logger.info(
                    "Processed %d images from %s: %d decisions",
                    len(records), path, len(decisions),
                )
                return decisions
        except Exception as e:
            logger.error("Real pipeline failed: %s — falling back to demo mode", e)

    # --- Demo fallback ---
    if True:  # Demo fallback always available
        p = Path(path)
        image_ids = (
            [f.stem for f in sorted(p.glob("*")) if f.is_file()]
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
                    evidence_summary={"data_mode": DataMode.DEMO.value},
                    update_history=is_trusted,
                )
            )
        return decisions


def generate_candidates(image_record: ImageRecord) -> list[IdentityCandidate]:
    """Given one ImageRecord, return ranked candidate identities."""
    # --- Real logic (Developer 2) ---
    if _DEV2_AVAILABLE:
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

    # --- Demo fallback ---
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
    # --- Real logic (Developer 2) ---
    if _DEV2_AVAILABLE:
        try:
            return _real_make_identity_decision(candidates, context)
        except Exception as e:
            logger.error("Real make_identity_decision failed: %s — using demo fallback", e)

    # --- Demo fallback ---
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
# DEVELOPER 3 owns the real implementation of these three.
# ---------------------------------------------------------------------------

def create_observation(decision: IdentityDecision) -> Observation | None:
    """Enforcement point: only trusted_match decisions may become a trusted
    Observation. Everything else returns None."""
    if not decision.update_history or decision.decision != IdentityDecisionState.TRUSTED_MATCH:
        return None

    if DEMO_MODE:
        return Observation(
            observation_id=f"obs_{decision.image_id}",
            image_id=decision.image_id,
            identity_id=decision.identity_id or "UNKNOWN",
            station_id="STATION_DEMO_1",
            latitude=21.68,
            longitude=79.29,
            timestamp=datetime.now(timezone.utc),
            identity_confidence=decision.confidence,
            camera_status=CameraStatus.ACTIVE,
            quality_score=0.8,
        )
    raise NotImplementedError("Developer 3: wire to src/history.py update_trusted_history().")


def generate_movement_alerts(
    observations: list[Observation], station_context: dict | None = None
) -> list[MovementAlert]:
    """Run movement-deviation analysis over trusted observations and
    generate or suppress alerts accordingly."""
    if DEMO_MODE:
        if not observations:
            return []
        alerts = []
        for obs in observations[:1]:  # demo: only ever alert on the first
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
    raise NotImplementedError(
        "Developer 3: wire to src/movement.py + src/alerts.py generate_alerts()."
    )


def run_evaluation(records: dict | None = None) -> list[EvaluationReport]:
    """Compare baseline (always-assign) vs. proposed (evidence-gated)
    pipelines on the same scenario set. Never fabricate numbers — use
    EvaluationReport.not_computable for anything not measurable yet."""
    if DEMO_MODE:
        return [
            EvaluationReport(
                pipeline_name="baseline",
                not_computable=[
                    "false_confident_identity_rate",
                    "false_movement_alert_rate",
                    "alert_precision",
                ],
                notes="DEMO MODE placeholder — no scenario evaluation run yet.",
            ),
            EvaluationReport(
                pipeline_name="evidence_gated",
                not_computable=[
                    "false_confident_identity_rate",
                    "false_movement_alert_rate",
                    "alert_precision",
                ],
                notes="DEMO MODE placeholder — no scenario evaluation run yet.",
            ),
        ]
    raise NotImplementedError("Developer 3: wire to src/evaluation.py run_identity_evaluation().")
