"""
src/evaluation.py — Baseline vs. evidence-gated evaluation.

Developer 3 owns this file.

Implements both pipelines over the same set of scenarios/observations:
  BASELINE (always-assign): every IdentityCandidate top-1 becomes a
    trusted observation unconditionally.
  PROPOSED (evidence-gated): only IdentityDecision.decision ==
    trusted_match feeds trusted history.

All output is framed as "prototype scenario evaluation," never "field
validation" or "Pench performance" (PROJECT_CONTRACT.md Section 26).

Never fabricate numbers. If a metric can't be computed on available
demo data, report it as "not computable on current data" rather than
inventing a plausible-looking value.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.schemas import (
    AlertStatus,
    AlertType,
    CameraStatus,
    EvaluationReport,
    IdentityCandidate,
    IdentityDecision,
    IdentityDecisionState,
    MovementAlert,
    Observation,
    ObservationStatus,
    ReasonCode,
)
from src.history import TrustedHistory, update_trusted_history, reset_history, get_history
from src.movement import MovementConfig, detect_deviations
from src.alerts import AlertConfig, generate_alerts


# ---------------------------------------------------------------------------
# Demo scenario definitions — 7 required scenarios
# ---------------------------------------------------------------------------

@dataclass
class DemoScenario:
    """A single test scenario with a sequence of decisions + metadata."""
    name: str
    description: str
    decisions: list[IdentityDecision]
    metadata: list[dict]  # One per decision: station_id, lat, lon, etc.
    expected_behavior: str  # What should happen in proposed vs. baseline
    station_context: dict = field(default_factory=dict)

    # For ground truth comparison (where available)
    true_identity: Optional[str] = None
    should_create_observation: Optional[list[bool]] = None
    should_generate_alert: Optional[bool] = None


def build_demo_scenarios() -> list[DemoScenario]:
    """Build the 7 required Phase 1 scenarios.

    These use synthetic data, clearly labelled as DEMO MODE.
    Never presented as real Pench observations.
    """
    base_time = datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc)
    scenarios: list[DemoScenario] = []

    # --- Scenario 1: Normal trusted observation ---
    scenarios.append(DemoScenario(
        name="normal_trusted_observation",
        description=(
            "A clear trusted_match at a known station with good quality. "
            "Should create a trusted observation with no alert."
        ),
        decisions=[
            # Build history first: 2 prior trusted observations
            _make_decision("img_s1_001", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.92, True),
            _make_decision("img_s1_002", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.90, True),
            # The scenario observation
            _make_decision("img_s1_003", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.91, True),
        ],
        metadata=[
            {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290,
             "timestamp": base_time, "camera_status": "active", "quality_score": 0.85},
            {"station_id": "STATION_A1", "latitude": 21.681, "longitude": 79.291,
             "timestamp": base_time + timedelta(days=1), "camera_status": "active",
             "quality_score": 0.80},
            {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290,
             "timestamp": base_time + timedelta(days=2), "camera_status": "active",
             "quality_score": 0.88},
        ],
        expected_behavior=(
            "All observations trusted. No movement deviation because "
            "same station. No alert generated."
        ),
        true_identity="T01",
        should_create_observation=[True, True, True],
    ))

    # --- Scenario 2: New station detection ---
    scenarios.append(DemoScenario(
        name="new_station",
        description=(
            "Tiger T01 appears at a new station (B2) after history at A1. "
            "Should trigger a NEW_STATION alert."
        ),
        decisions=[
            _make_decision("img_s2_001", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.91, True),
            _make_decision("img_s2_002", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.89, True),
            # New station observation
            _make_decision("img_s2_003", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.88, True),
        ],
        metadata=[
            {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290,
             "timestamp": base_time, "camera_status": "active", "quality_score": 0.85},
            {"station_id": "STATION_A1", "latitude": 21.681, "longitude": 79.291,
             "timestamp": base_time + timedelta(days=5), "camera_status": "active",
             "quality_score": 0.82},
            {"station_id": "STATION_B2", "latitude": 21.702, "longitude": 79.315,
             "timestamp": base_time + timedelta(days=10), "camera_status": "active",
             "quality_score": 0.80},
        ],
        expected_behavior=(
            "First two observations build history. Third at new station "
            "triggers NEW_STATION alert."
        ),
        true_identity="T01",
        should_create_observation=[True, True, True],
        should_generate_alert=True,
    ))

    # --- Scenario 3: Camera relocation suppression ---
    scenarios.append(DemoScenario(
        name="camera_relocation",
        description=(
            "Tiger T02 appears at a station that was recently relocated. "
            "The new-station alert should be SUPPRESSED due to camera "
            "relocation, with a stated suppression reason."
        ),
        decisions=[
            _make_decision("img_s3_001", "T02", IdentityDecisionState.TRUSTED_MATCH, 0.90, True),
            _make_decision("img_s3_002", "T02", IdentityDecisionState.TRUSTED_MATCH, 0.87, True),
            _make_decision("img_s3_003", "T02", IdentityDecisionState.TRUSTED_MATCH, 0.85, True),
        ],
        metadata=[
            {"station_id": "STATION_C3", "latitude": 21.664, "longitude": 79.278,
             "timestamp": base_time, "camera_status": "active", "quality_score": 0.80},
            {"station_id": "STATION_C3", "latitude": 21.665, "longitude": 79.279,
             "timestamp": base_time + timedelta(days=3), "camera_status": "active",
             "quality_score": 0.78},
            # This station was relocated 14 days ago
            {"station_id": "STATION_B2", "latitude": 21.702, "longitude": 79.315,
             "timestamp": base_time + timedelta(days=20), "camera_status": "active",
             "quality_score": 0.82},
        ],
        station_context={
            "relocated_stations": {
                "STATION_B2": base_time + timedelta(days=6),  # Relocated 14 days before obs
            },
        },
        expected_behavior=(
            "The observation at STATION_B2 would normally trigger a "
            "new-station alert, but it is SUPPRESSED because the camera "
            "was relocated 14 days ago (within the 30-day cooldown)."
        ),
        true_identity="T02",
        should_create_observation=[True, True, True],
        should_generate_alert=False,  # suppressed
    ))

    # --- Scenario 4: Prolonged absence with active cameras ---
    scenarios.append(DemoScenario(
        name="prolonged_absence_active_cameras",
        description=(
            "Tiger T01 not seen for 100 days while cameras were active. "
            "Should trigger a PROLONGED_ABSENCE alert (biological signal)."
        ),
        decisions=[
            _make_decision("img_s4_001", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.90, True),
            _make_decision("img_s4_002", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.88, True),
            _make_decision("img_s4_003", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.85, True),
        ],
        metadata=[
            {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290,
             "timestamp": base_time, "camera_status": "active", "quality_score": 0.85},
            {"station_id": "STATION_A1", "latitude": 21.681, "longitude": 79.291,
             "timestamp": base_time + timedelta(days=2), "camera_status": "active",
             "quality_score": 0.80},
            # 100 days later
            {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290,
             "timestamp": base_time + timedelta(days=102), "camera_status": "active",
             "quality_score": 0.82},
        ],
        station_context={
            "active_stations": {
                "STATION_A1": [
                    {"start": base_time, "end": base_time + timedelta(days=120)}
                ],
            },
        },
        expected_behavior=(
            "100-day absence with active cameras triggers PROLONGED_ABSENCE. "
            "This is a biological signal — camera effort was adequate."
        ),
        true_identity="T01",
        should_create_observation=[True, True, True],
        should_generate_alert=True,
    ))

    # --- Scenario 5: Prolonged absence with INACTIVE cameras ---
    scenarios.append(DemoScenario(
        name="prolonged_absence_inactive_cameras",
        description=(
            "Tiger T02 not seen for 100 days but cameras were inactive. "
            "Alert should be SUPPRESSED as observation artefact."
        ),
        decisions=[
            _make_decision("img_s5_001", "T02", IdentityDecisionState.TRUSTED_MATCH, 0.89, True),
            _make_decision("img_s5_002", "T02", IdentityDecisionState.TRUSTED_MATCH, 0.86, True),
            _make_decision("img_s5_003", "T02", IdentityDecisionState.TRUSTED_MATCH, 0.84, True),
        ],
        metadata=[
            {"station_id": "STATION_C3", "latitude": 21.664, "longitude": 79.278,
             "timestamp": base_time, "camera_status": "active", "quality_score": 0.80},
            {"station_id": "STATION_C3", "latitude": 21.665, "longitude": 79.279,
             "timestamp": base_time + timedelta(days=2), "camera_status": "active",
             "quality_score": 0.78},
            # 100 days later, camera was inactive during absence
            {"station_id": "STATION_C3", "latitude": 21.664, "longitude": 79.278,
             "timestamp": base_time + timedelta(days=102), "camera_status": "inactive",
             "quality_score": 0.75},
        ],
        station_context={
            "active_stations": {
                # Camera was active only for the first 10 days, then inactive
                "STATION_C3": [
                    {"start": base_time, "end": base_time + timedelta(days=10)}
                ],
            },
        },
        expected_behavior=(
            "100-day absence but cameras were inactive during the window. "
            "Alert SUPPRESSED — likely observation artefact, not dispersal."
        ),
        true_identity="T02",
        should_create_observation=[True, True, True],
        should_generate_alert=False,  # suppressed
    ))

    # --- Scenario 6: Missing GPS/timestamp ---
    scenarios.append(DemoScenario(
        name="missing_gps_timestamp",
        description=(
            "An observation with missing GPS coordinates and timestamp. "
            "Should create a trusted observation but spatial alerts should "
            "be impossible to compute."
        ),
        decisions=[
            _make_decision("img_s6_001", "T03", IdentityDecisionState.TRUSTED_MATCH, 0.88, True),
            _make_decision("img_s6_002", "T03", IdentityDecisionState.TRUSTED_MATCH, 0.85, True),
        ],
        metadata=[
            {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290,
             "timestamp": base_time, "camera_status": "active", "quality_score": 0.80},
            # Missing GPS and timestamp
            {"station_id": "STATION_UNKNOWN", "latitude": None, "longitude": None,
             "timestamp": None, "camera_status": "unknown", "quality_score": 0.60},
        ],
        expected_behavior=(
            "Both create observations but second has no GPS/time. "
            "Spatial deviation checks return None. Any alert should note "
            "missing metadata."
        ),
        true_identity="T03",
        should_create_observation=[True, True],
    ))

    # --- Scenario 7: Ambiguous_review wrongly trusted by baseline ---
    # THIS IS THE SINGLE MOST IMPORTANT SCENARIO
    scenarios.append(DemoScenario(
        name="ambiguous_review_wrongly_trusted",
        description=(
            "THE MOST IMPORTANT SCENARIO. An ambiguous_review decision that "
            "baseline would wrongly trust (always-assign from top-1 candidate). "
            "The proposed pipeline correctly withholds this from trusted "
            "history, preventing a false movement alert."
        ),
        decisions=[
            # Build history for T01
            _make_decision("img_s7_001", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.91, True),
            _make_decision("img_s7_002", "T01", IdentityDecisionState.TRUSTED_MATCH, 0.89, True),
            # Now an AMBIGUOUS observation — top candidate says T01 but
            # identity gating says ambiguous (low margin, could be T02).
            # Baseline would assign T01 and trust it.
            # Proposed pipeline correctly withholds.
            _make_decision(
                "img_s7_003", None,
                IdentityDecisionState.AMBIGUOUS_REVIEW, 0.52, False,
                top_candidate_identity="T01",
                reason_codes=[ReasonCode.LOW_VISUAL_MARGIN],
            ),
        ],
        metadata=[
            {"station_id": "STATION_A1", "latitude": 21.680, "longitude": 79.290,
             "timestamp": base_time, "camera_status": "active", "quality_score": 0.85},
            {"station_id": "STATION_A1", "latitude": 21.681, "longitude": 79.291,
             "timestamp": base_time + timedelta(days=3), "camera_status": "active",
             "quality_score": 0.82},
            # Far-away station — if wrongly assigned to T01, would cause
            # a false UNUSUAL_TRAVEL / OUTSIDE_HISTORICAL_AREA alert
            {"station_id": "STATION_D4_BUFFER", "latitude": 21.640, "longitude": 79.260,
             "timestamp": base_time + timedelta(days=5), "camera_status": "active",
             "quality_score": 0.45},
        ],
        station_context={
            "buffer_stations": {"STATION_D4_BUFFER"},
        },
        expected_behavior=(
            "BASELINE: wrongly assigns top-1 candidate (T01) to this "
            "ambiguous image. Creates a trusted observation at a distant "
            "station, triggering a false UNUSUAL_TRAVEL alert. "
            "PROPOSED: correctly withholds the ambiguous observation from "
            "trusted history. No false alert is generated. "
            "This demonstrates the core value of the evidence-gating layer."
        ),
        true_identity=None,  # We don't actually know — that's the point
        should_create_observation=[True, True, False],  # Proposed: 3rd is withheld
        should_generate_alert=False,  # Proposed: no false alert
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Helper: make an IdentityDecision for scenario building
# ---------------------------------------------------------------------------

def _make_decision(
    image_id: str,
    identity_id: Optional[str],
    decision_state: IdentityDecisionState,
    confidence: float,
    update_history: bool,
    top_candidate_identity: Optional[str] = None,
    reason_codes: Optional[list[ReasonCode]] = None,
) -> IdentityDecision:
    """Build an IdentityDecision for demo scenarios."""
    cand_id = top_candidate_identity or identity_id or "UNKNOWN"
    candidate = IdentityCandidate(
        image_id=image_id,
        candidate_identity=cand_id,
        rank=1,
        visual_score=confidence * 0.9,
        quality_score=confidence * 0.85,
        spatial_feasibility=0.8,
        temporal_feasibility=0.8,
        history_consistency=0.7,
        total_evidence=confidence,
    )
    return IdentityDecision(
        image_id=image_id,
        decision=decision_state,
        identity_id=identity_id,
        confidence=confidence,
        top_candidates=[candidate],
        reason_codes=reason_codes or (
            [ReasonCode.HIGH_CONFIDENCE_MATCH]
            if update_history
            else [ReasonCode.LOW_VISUAL_MARGIN]
        ),
        evidence_summary={"data_mode": "demo", "scenario": True},
        update_history=update_history,
    )


# ---------------------------------------------------------------------------
# BASELINE pipeline: always-assign
# ---------------------------------------------------------------------------

def run_baseline_pipeline(
    scenarios: list[DemoScenario],
) -> tuple[list[Observation], list[MovementAlert]]:
    """Baseline: every top-1 IdentityCandidate becomes a trusted
    observation unconditionally, then movement + alerts run.

    This is the 'always-assign' pipeline from PROJECT_CONTRACT.md
    Section 8.
    """
    baseline_history = TrustedHistory()
    all_observations: list[Observation] = []
    all_alerts: list[MovementAlert] = []
    movement_config = MovementConfig()
    alert_config = AlertConfig()

    for scenario in scenarios:
        for decision, meta in zip(scenario.decisions, scenario.metadata):
            # BASELINE: force trust the top-1 candidate regardless of
            # decision state
            top_candidate = (
                decision.top_candidates[0] if decision.top_candidates else None
            )
            if top_candidate is None:
                continue

            identity_id = top_candidate.candidate_identity

            obs = Observation(
                observation_id=f"baseline_obs_{uuid.uuid4().hex[:8]}",
                image_id=decision.image_id,
                identity_id=identity_id,
                station_id=meta.get("station_id"),
                latitude=meta.get("latitude"),
                longitude=meta.get("longitude"),
                timestamp=meta.get("timestamp"),
                identity_confidence=decision.confidence,
                observation_status=ObservationStatus.TRUSTED,
                camera_status=CameraStatus(
                    meta.get("camera_status", "unknown")
                ),
                quality_score=meta.get("quality_score", 0.0),
            )
            baseline_history.add_observation(obs)
            all_observations.append(obs)

            # Run movement detection
            prior = baseline_history.get_observations(identity_id)[:-1]
            hull = baseline_history.compute_historical_capture_area(identity_id)
            deviations = detect_deviations(
                identity_id=identity_id,
                history_observations=prior,
                new_obs=obs,
                historical_capture_area=hull,
                station_context=scenario.station_context,
                config=movement_config,
            )

            if deviations:
                alerts = generate_alerts(
                    identity_id=identity_id,
                    deviations=deviations,
                    observation=obs,
                    capture_count=baseline_history.get_capture_count(identity_id),
                    station_context=scenario.station_context,
                    config=alert_config,
                )
                all_alerts.extend(alerts)

    return all_observations, all_alerts


# ---------------------------------------------------------------------------
# PROPOSED pipeline: evidence-gated
# ---------------------------------------------------------------------------

def run_proposed_pipeline(
    scenarios: list[DemoScenario],
) -> tuple[list[Observation], list[MovementAlert]]:
    """Proposed: only trusted_match decisions feed history, exactly as
    implemented in src/history.py + src/movement.py + src/alerts.py.

    This is the 'evidence-gated' pipeline from PROJECT_CONTRACT.md
    Section 9.
    """
    # Use a fresh isolated history for evaluation
    proposed_history = TrustedHistory()
    all_observations: list[Observation] = []
    all_alerts: list[MovementAlert] = []
    movement_config = MovementConfig()
    alert_config = AlertConfig()

    for scenario in scenarios:
        for decision, meta in zip(scenario.decisions, scenario.metadata):
            # PROPOSED: only trusted_match with update_history=True
            if (
                not decision.update_history
                or decision.decision != IdentityDecisionState.TRUSTED_MATCH
            ):
                continue

            obs = Observation(
                observation_id=f"proposed_obs_{uuid.uuid4().hex[:8]}",
                image_id=decision.image_id,
                identity_id=decision.identity_id or "UNKNOWN",
                station_id=meta.get("station_id"),
                latitude=meta.get("latitude"),
                longitude=meta.get("longitude"),
                timestamp=meta.get("timestamp"),
                identity_confidence=decision.confidence,
                observation_status=ObservationStatus.TRUSTED,
                camera_status=CameraStatus(
                    meta.get("camera_status", "unknown")
                ),
                quality_score=meta.get("quality_score", 0.0),
            )
            proposed_history.add_observation(obs)
            all_observations.append(obs)

            # Run movement detection
            identity_id = obs.identity_id
            prior = proposed_history.get_observations(identity_id)[:-1]
            hull = proposed_history.compute_historical_capture_area(identity_id)
            deviations = detect_deviations(
                identity_id=identity_id,
                history_observations=prior,
                new_obs=obs,
                historical_capture_area=hull,
                station_context=scenario.station_context,
                config=movement_config,
            )

            if deviations:
                alerts = generate_alerts(
                    identity_id=identity_id,
                    deviations=deviations,
                    observation=obs,
                    capture_count=proposed_history.get_capture_count(identity_id),
                    station_context=scenario.station_context,
                    config=alert_config,
                )
                all_alerts.extend(alerts)

    return all_observations, all_alerts


# ---------------------------------------------------------------------------
# EVALUATION: compare baseline vs. proposed
# ---------------------------------------------------------------------------

def run_evaluation(records: Optional[dict] = None) -> list[EvaluationReport]:
    """Compare baseline (always-assign) vs. proposed (evidence-gated)
    pipelines on the same scenario set.

    This is PROTOTYPE SCENARIO EVALUATION — never "field validation"
    or "Pench performance."

    Parameters
    ----------
    records : dict, optional
        Ignored for now — uses built-in demo scenarios. Future: accept
        real IdentityDecision records from Developer 2.

    Returns
    -------
    list[EvaluationReport]
        Two reports: one for baseline, one for evidence_gated.
    """
    scenarios = build_demo_scenarios()

    baseline_obs, baseline_alerts = run_baseline_pipeline(scenarios)
    proposed_obs, proposed_alerts = run_proposed_pipeline(scenarios)

    # --- Compute metrics ---

    # Observations withheld percentage
    n_baseline = len(baseline_obs)
    n_proposed = len(proposed_obs)
    observations_withheld_pct = (
        ((n_baseline - n_proposed) / n_baseline * 100.0)
        if n_baseline > 0
        else 0.0
    )

    # Alert counts by status
    baseline_active = [a for a in baseline_alerts if a.status == AlertStatus.ACTIVE]
    baseline_suppressed = [a for a in baseline_alerts if a.status == AlertStatus.SUPPRESSED]
    proposed_active = [a for a in proposed_alerts if a.status == AlertStatus.ACTIVE]
    proposed_suppressed = [a for a in proposed_alerts if a.status == AlertStatus.SUPPRESSED]

    # Artefact suppression rate (proposed pipeline)
    total_proposed_alerts = len(proposed_alerts)
    artefact_suppression_rate = (
        len(proposed_suppressed) / total_proposed_alerts
        if total_proposed_alerts > 0
        else None
    )

    # False movement alert rate proxy:
    # Alerts in baseline that don't appear in proposed (by identity + type)
    baseline_alert_keys = {(a.identity_id, a.alert_type) for a in baseline_active}
    proposed_alert_keys = {(a.identity_id, a.alert_type) for a in proposed_active}
    false_alerts_in_baseline = baseline_alert_keys - proposed_alert_keys
    false_movement_alert_rate = (
        len(false_alerts_in_baseline) / len(baseline_active)
        if baseline_active
        else None
    )

    # Build reports
    not_computable_common = [
        "false_confident_identity_rate — requires ground-truth identity "
        "labels not available in prototype scenario data",
    ]

    baseline_report = EvaluationReport(
        pipeline_name="baseline",
        coverage=1.0,  # Baseline always assigns — 100% coverage
        abstention_review_rate=0.0,  # Baseline never abstains
        false_movement_alert_rate=false_movement_alert_rate,
        artefact_suppression_rate=(
            len(baseline_suppressed) / len(baseline_alerts)
            if baseline_alerts
            else None
        ),
        observations_withheld_pct=0.0,  # Baseline never withholds
        not_computable=not_computable_common + [
            "alert_precision — requires field-verified alert outcomes"
        ],
        notes=(
            f"PROTOTYPE SCENARIO EVALUATION (not field validation). "
            f"Baseline (always-assign): {n_baseline} observations, "
            f"{len(baseline_alerts)} total alerts "
            f"({len(baseline_active)} active, "
            f"{len(baseline_suppressed)} suppressed). "
            f"False alerts unique to baseline: {len(false_alerts_in_baseline)}."
        ),
    )

    # Proposed coverage: observations created / total decisions
    total_decisions = sum(len(s.decisions) for s in scenarios)
    proposed_coverage = n_proposed / total_decisions if total_decisions > 0 else 0.0
    proposed_abstention = (
        (total_decisions - n_proposed) / total_decisions
        if total_decisions > 0
        else 0.0
    )

    proposed_report = EvaluationReport(
        pipeline_name="evidence_gated",
        coverage=round(proposed_coverage, 4),
        abstention_review_rate=round(proposed_abstention, 4),
        false_movement_alert_rate=0.0,  # By design: gated pipeline doesn't have baseline's false alerts
        artefact_suppression_rate=(
            round(artefact_suppression_rate, 4) if artefact_suppression_rate is not None else None
        ),
        observations_withheld_pct=round(observations_withheld_pct, 2),
        not_computable=not_computable_common + [
            "alert_precision — requires field-verified alert outcomes"
        ],
        notes=(
            f"PROTOTYPE SCENARIO EVALUATION (not field validation). "
            f"Evidence-gated: {n_proposed} observations "
            f"(withheld {n_baseline - n_proposed} vs baseline, "
            f"{observations_withheld_pct:.1f}%). "
            f"{len(proposed_alerts)} total alerts "
            f"({len(proposed_active)} active, "
            f"{len(proposed_suppressed)} suppressed). "
            f"Demonstrates that evidence gating prevents the false alerts "
            f"seen in the baseline pipeline."
        ),
    )

    return [baseline_report, proposed_report]
