"""
gating.py — Evidence-aware identity gating.

Developer 2 owns this file (PROJECT_CONTRACT.md Section 18).

This is the core contribution of the project: deciding whether the
evidence is strong enough for an observation to influence a downstream
management conclusion.

Responsibilities:
- compute_evidence(candidate, context) -> float
  - E = W_V*V + W_Q*Q + W_S*S + W_T*T + W_H*H
  - All weights are configurable constants.
  - These are PROTOTYPE HEURISTICS, NOT scientifically validated parameters.

- make_identity_decision(candidates, context) -> IdentityDecision
  - Configurable thresholds for trusted/ambiguous/unknown/insufficient.
  - update_history = True ONLY when decision == trusted_match.
  - This is the one rule that must NEVER be weakened, bypassed, or
    left to another module's judgment.

- Unknown-individual handling: store unknown embeddings; if several are
  mutually similar, assign provisional id NEW-001 etc. — NOT added to
  trusted catalogue.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import (
    CameraStatus,
    IdentityCandidate,
    IdentityDecision,
    IdentityDecisionState,
    ReasonCode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable evidence weights.
# These are PROTOTYPE HEURISTICS, NOT scientifically validated parameters.
# All five weights should sum to 1.0.
# ---------------------------------------------------------------------------
W_VISUAL = 0.55       # V: visual similarity
W_QUALITY = 0.15      # Q: image quality / flank score
W_SPATIAL = 0.15      # S: spatial feasibility
W_TEMPORAL = 0.10     # T: temporal feasibility
W_HISTORY = 0.05      # H: history consistency

# ---------------------------------------------------------------------------
# Configurable decision thresholds.
# These are prototype starting points — not calibrated or validated.
# ---------------------------------------------------------------------------
THRESHOLD_TRUSTED = 0.80     # E >= this and no major conflict -> trusted_match
THRESHOLD_AMBIGUOUS = 0.55   # this <= E < THRESHOLD_TRUSTED -> ambiguous_review
# E < THRESHOLD_AMBIGUOUS -> unknown (or rejected)

# Minimum quality score below which we force insufficient_evidence
MIN_QUALITY_FOR_DECISION = 0.15

# Minimum visual margin between top-1 and top-2 candidates
# If margin is too small, the match is ambiguous even if score is high
MIN_VISUAL_MARGIN = 0.08

# Threshold for considering unknown embeddings as mutually similar
UNKNOWN_SIMILARITY_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Unknown individual storage
# ---------------------------------------------------------------------------

class UnknownStore:
    """Temporary storage for unknown-individual embeddings.
    If several are mutually similar, assigns provisional IDs (NEW-001 etc.).
    These are NOT added to the trusted catalogue."""

    def __init__(self):
        self._unknowns: list[dict] = []
        self._next_id = 1

    def add_unknown(self, image_id: str, embedding: list[float]) -> Optional[str]:
        """Store an unknown embedding and check if it clusters with
        existing unknowns. Returns a provisional ID if a cluster is found."""
        vec = np.array(embedding, dtype=np.float64).reshape(1, -1)

        # Check similarity with existing unknowns
        for entry in self._unknowns:
            existing_vec = np.array(entry["embedding"], dtype=np.float64).reshape(1, -1)
            sim = cosine_similarity(vec, existing_vec)[0][0]
            if sim >= UNKNOWN_SIMILARITY_THRESHOLD:
                # Similar to an existing unknown — assign same provisional ID
                prov_id = entry.get("provisional_id")
                if prov_id is None:
                    prov_id = f"NEW-{self._next_id:03d}"
                    self._next_id += 1
                    entry["provisional_id"] = prov_id
                self._unknowns.append({
                    "image_id": image_id,
                    "embedding": embedding,
                    "provisional_id": prov_id,
                })
                logger.info(
                    "Unknown %s clusters with existing unknown -> provisional ID %s",
                    image_id, prov_id,
                )
                return prov_id

        # No match — store as new unknown
        self._unknowns.append({
            "image_id": image_id,
            "embedding": embedding,
            "provisional_id": None,
        })
        return None

    def get_provisional_ids(self) -> dict[str, list[str]]:
        """Return mapping of provisional_id -> list of image_ids."""
        clusters: dict[str, list[str]] = {}
        for entry in self._unknowns:
            pid = entry.get("provisional_id")
            if pid:
                clusters.setdefault(pid, []).append(entry["image_id"])
        return clusters

    def clear(self) -> None:
        self._unknowns.clear()
        self._next_id = 1


# Module-level unknown store
_unknown_store = UnknownStore()


def get_unknown_store() -> UnknownStore:
    """Get the module-level unknown store."""
    return _unknown_store


# ---------------------------------------------------------------------------
# Evidence computation
# ---------------------------------------------------------------------------

def compute_evidence(candidate: IdentityCandidate, context: Optional[dict] = None) -> float:
    """Compute the composite evidence score E for a candidate.

    E = W_V*V_eff + W_Q*Q + W_S*S + W_T*T + W_H*H
    where V_eff combines global visual embedding with local stripe/flank matching proxy.

    All five component weights are configurable module-level constants.
    These are PROTOTYPE HEURISTICS, NOT scientifically validated parameters.

    Parameters
    ----------
    candidate : IdentityCandidate
        Must have visual_score, local_score, quality_score, spatial_feasibility,
        temporal_feasibility, history_consistency fields.
    context : dict, optional
        Additional context (camera_status, etc.) for conflict detection.

    Returns
    -------
    float
        Composite evidence score in [0, 1].
    """
    if candidate.local_score > 0.0:
        eff_visual = 0.75 * candidate.visual_score + 0.25 * candidate.local_score
    else:
        eff_visual = candidate.visual_score

    E = (
        W_VISUAL * eff_visual
        + W_QUALITY * candidate.quality_score
        + W_SPATIAL * candidate.spatial_feasibility
        + W_TEMPORAL * candidate.temporal_feasibility
        + W_HISTORY * candidate.history_consistency
    )
    return round(min(1.0, max(0.0, E)), 4)


def _detect_conflicts(
    candidates: list[IdentityCandidate],
    context: Optional[dict] = None,
) -> tuple[list[ReasonCode], bool]:
    """Detect conflict conditions that should downgrade a decision.

    Returns (reason_codes, has_major_conflict).
    """
    reasons: list[ReasonCode] = []
    has_major = False
    ctx = context or {}

    # Check visual margin between top-1 and top-2
    if len(candidates) >= 2:
        margin = candidates[0].visual_score - candidates[1].visual_score
        if margin < MIN_VISUAL_MARGIN:
            reasons.append(ReasonCode.LOW_VISUAL_MARGIN)
            has_major = True

    # Check image quality
    if candidates and candidates[0].quality_score < MIN_QUALITY_FOR_DECISION:
        reasons.append(ReasonCode.POOR_IMAGE_QUALITY)
        has_major = True

    # Check flank visibility (from context if available)
    flank_vis = ctx.get("flank_visibility", 0.5)
    if flank_vis < 0.15:
        reasons.append(ReasonCode.FLANK_NOT_VISIBLE)
        has_major = True

    # Check camera status
    camera_status = ctx.get("camera_status")
    if camera_status == CameraStatus.RELOCATED or camera_status == "relocated":
        reasons.append(ReasonCode.CAMERA_RELOCATED)
        # Camera relocation is a major conflict for spatial reasoning
        has_major = True

    # Check for missing metadata
    if ctx.get("timestamp") is None:
        reasons.append(ReasonCode.MISSING_TIMESTAMP)

    if ctx.get("latitude") is None or ctx.get("longitude") is None:
        reasons.append(ReasonCode.MISSING_LOCATION)

    # Check history consistency
    if candidates and candidates[0].history_consistency < 0.3:
        reasons.append(ReasonCode.INSUFFICIENT_HISTORY)

    return reasons, has_major


def _check_severe_failure(context: Optional[dict] = None) -> bool:
    """Check for severe quality/camera/metadata failures that should
    force an insufficient_evidence decision regardless of score."""
    ctx = context or {}

    # Severe: camera inactive
    camera_status = ctx.get("camera_status")
    if camera_status == CameraStatus.INACTIVE or camera_status == "inactive":
        return True

    # Severe: quality too low for any meaningful analysis
    quality = ctx.get("quality_score", 0.5)
    if quality < 0.05:
        return True

    # Severe: missing both location and timestamp
    has_location = ctx.get("latitude") is not None and ctx.get("longitude") is not None
    has_timestamp = ctx.get("timestamp") is not None
    if not has_location and not has_timestamp:
        return True

    return False


# ---------------------------------------------------------------------------
# Identity decision
# ---------------------------------------------------------------------------

def make_identity_decision(
    candidates: list[IdentityCandidate],
    context: Optional[dict] = None,
) -> IdentityDecision:
    """Given ranked candidates and context, produce an evidence-gated
    IdentityDecision.

    This is the core gating function. The decision follows this logic:
    1. Severe quality/camera/metadata failure -> insufficient_evidence
    2. E >= THRESHOLD_TRUSTED and no major conflict -> trusted_match
    3. THRESHOLD_AMBIGUOUS <= E < THRESHOLD_TRUSTED -> ambiguous_review
    4. E < THRESHOLD_AMBIGUOUS -> unknown

    CRITICAL: update_history is True ONLY when decision == trusted_match.
    This rule is NEVER weakened, bypassed, or left to another module.

    Parameters
    ----------
    candidates : list[IdentityCandidate]
        Ranked candidates from identity.py generate_candidates().
    context : dict, optional
        Must include camera_status, quality_score, flank_visibility,
        timestamp, latitude, longitude, station_id.

    Returns
    -------
    IdentityDecision
        With decision state, confidence, reason codes, evidence summary,
        and update_history flag.
    """
    ctx = context or {}

    # If no candidates at all, this is unknown
    if not candidates:
        image_id = ctx.get("image_id", "unknown")
        return IdentityDecision(
            image_id=image_id,
            decision=IdentityDecisionState.UNKNOWN,
            confidence=0.0,
            reason_codes=[ReasonCode.LOW_VISUAL_MARGIN],
            evidence_summary={
                "no_candidates": True,
                "station_id": ctx.get("station_id"),
                "latitude": ctx.get("latitude"),
                "longitude": ctx.get("longitude"),
                "timestamp": ctx.get("timestamp"),
                "camera_status": ctx.get("camera_status"),
            },
            update_history=False,
        )

    top = candidates[0]
    image_id = top.image_id

    # Step 1: Check for severe failure
    if _check_severe_failure(ctx):
        reason_codes = []
        camera_status = ctx.get("camera_status")
        if camera_status == CameraStatus.INACTIVE or camera_status == "inactive":
            reason_codes.append(ReasonCode.CAMERA_RELOCATED)
        quality = ctx.get("quality_score", 0.5)
        if quality < 0.05:
            reason_codes.append(ReasonCode.POOR_IMAGE_QUALITY)
        if not reason_codes:
            reason_codes.append(ReasonCode.MISSING_LOCATION)
            reason_codes.append(ReasonCode.MISSING_TIMESTAMP)

        return IdentityDecision(
            image_id=image_id,
            decision=IdentityDecisionState.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            top_candidates=candidates,
            reason_codes=reason_codes,
            evidence_summary={
                "severe_failure": True,
                "quality_score": ctx.get("quality_score", 0.0),
                "station_id": ctx.get("station_id"),
                "latitude": ctx.get("latitude"),
                "longitude": ctx.get("longitude"),
                "timestamp": ctx.get("timestamp"),
                "camera_status": ctx.get("camera_status"),
            },
            update_history=False,
        )

    # Step 2: Compute evidence for top candidate
    evidence_score = compute_evidence(top, ctx)

    # Step 3: Detect conflicts
    conflict_reasons, has_major_conflict = _detect_conflicts(candidates, ctx)

    # Step 4: Make decision based on thresholds and conflicts
    if evidence_score >= THRESHOLD_TRUSTED and not has_major_conflict:
        decision = IdentityDecisionState.TRUSTED_MATCH
        reason_codes = [ReasonCode.HIGH_CONFIDENCE_MATCH]
        # Also include any minor conflict reasons for transparency
        reason_codes.extend(conflict_reasons)
        identity_id = top.candidate_identity
        update_history = True

    elif evidence_score >= THRESHOLD_AMBIGUOUS:
        decision = IdentityDecisionState.AMBIGUOUS_REVIEW
        reason_codes = conflict_reasons if conflict_reasons else [ReasonCode.LOW_VISUAL_MARGIN]
        identity_id = top.candidate_identity  # best guess, but not trusted
        update_history = False

    else:
        decision = IdentityDecisionState.UNKNOWN
        reason_codes = conflict_reasons if conflict_reasons else [ReasonCode.LOW_VISUAL_MARGIN]
        identity_id = None
        update_history = False

        # Unknown handling: store embedding for clustering
        embedding = ctx.get("embedding")
        if embedding:
            prov_id = _unknown_store.add_unknown(image_id, embedding)
            if prov_id:
                identity_id = prov_id  # provisional, NOT trusted

    # Extract seasonal period if timestamp is available
    timestamp = ctx.get("timestamp")
    seasonal_period = None
    if timestamp is not None and hasattr(timestamp, "month"):
        m = timestamp.month
        if 6 <= m <= 9:
            seasonal_period = "monsoon"
        elif 10 <= m <= 12 or 1 <= m <= 2:
            seasonal_period = "winter"
        else:
            seasonal_period = "summer"

    station_id_val = ctx.get("station_id")
    buffer_stations = ctx.get("buffer_stations") or set()
    is_buffer = (
        (station_id_val in buffer_stations)
        or (isinstance(station_id_val, str) and ("BUFFER" in station_id_val.upper() or "VILLAGE" in station_id_val.upper()))
        or bool(ctx.get("is_buffer"))
    )

    # Build evidence summary
    # NOTE: station_id/latitude/longitude/timestamp/camera_status are copied
    # through so Backend B's create_observation() can extract the original
    # image metadata from the IdentityDecision alone (cross-branch contract).
    evidence_summary = {
        "evidence_score": evidence_score,
        "visual_score": top.visual_score,
        "quality_score": top.quality_score,
        "spatial_feasibility": top.spatial_feasibility,
        "temporal_feasibility": top.temporal_feasibility,
        "history_consistency": top.history_consistency,
        "top1_top2_margin": (
            round(candidates[0].visual_score - candidates[1].visual_score, 4)
            if len(candidates) >= 2 else None
        ),
        "has_major_conflict": has_major_conflict,
        "threshold_trusted": THRESHOLD_TRUSTED,
        "threshold_ambiguous": THRESHOLD_AMBIGUOUS,
        "weights": {
            "W_VISUAL": W_VISUAL,
            "W_QUALITY": W_QUALITY,
            "W_SPATIAL": W_SPATIAL,
            "W_TEMPORAL": W_TEMPORAL,
            "W_HISTORY": W_HISTORY,
        },
        "station_id": station_id_val,
        "is_buffer_zone": is_buffer,
        "seasonal_period": seasonal_period,
        "latitude": ctx.get("latitude"),
        "longitude": ctx.get("longitude"),
        "timestamp": timestamp,
        "camera_status": ctx.get("camera_status"),
    }

    return IdentityDecision(
        image_id=image_id,
        decision=decision,
        identity_id=identity_id,
        confidence=evidence_score,
        top_candidates=candidates,
        reason_codes=reason_codes,
        evidence_summary=evidence_summary,
        update_history=update_history,
    )


# ---------------------------------------------------------------------------
# Calibration check (P2 research feature)
# ---------------------------------------------------------------------------

def check_calibration(
    decisions_with_ground_truth: list[dict | tuple],
    n_bins: int = 5,
    bin_ranges: Optional[list[tuple[float, float]]] = None,
) -> dict:
    """Bucket IdentityDecisions by predicted confidence and evaluate empirical
    accuracy against ground-truth labels.

    Evaluates whether confidence scores (e.g. 0.80) reflect actual correctness
    probabilities, computing Expected Calibration Error (ECE).

    Parameters
    ----------
    decisions_with_ground_truth : list[dict | tuple]
        Entries to evaluate. Supported formats:
        - dict: {"decision": IdentityDecision, "true_identity": str | None, "should_create_observation": bool}
        - tuple: (decision, true_identity, should_create_observation)
    n_bins : int, default=5
        Number of equal-width bins between 0.0 and 1.0 if bin_ranges is None.
    bin_ranges : list[tuple[float, float]], optional
        Custom list of (min_conf, max_conf) tuples.

    Returns
    -------
    dict
        Structured calibration report with buckets, ECE, and MCE.
    """
    if not decisions_with_ground_truth:
        return {
            "total_samples": 0,
            "buckets": [],
            "expected_calibration_error": 0.0,
            "max_calibration_error": 0.0,
        }

    # Normalize entries into parsed tuples: (decision, true_identity, should_create)
    parsed_entries = []
    for item in decisions_with_ground_truth:
        if isinstance(item, dict):
            decision = item["decision"]
            true_id = item.get("true_identity")
            should_create = item.get("should_create_observation", True)
        elif isinstance(item, (tuple, list)):
            decision = item[0]
            true_id = item[1] if len(item) > 1 else None
            should_create = item[2] if len(item) > 2 else True
        else:
            continue

        # Evaluate if the decision correctly matches ground truth
        if should_create:
            is_correct = (
                decision.decision == IdentityDecisionState.TRUSTED_MATCH
                and decision.update_history is True
                and (true_id is None or decision.identity_id == true_id)
            )
        else:
            is_correct = (
                decision.decision != IdentityDecisionState.TRUSTED_MATCH
                and decision.update_history is False
            )

        parsed_entries.append({
            "confidence": float(decision.confidence),
            "is_correct": is_correct,
            "decision": decision,
        })

    total_n = len(parsed_entries)
    if total_n == 0:
        return {
            "total_samples": 0,
            "buckets": [],
            "expected_calibration_error": 0.0,
            "max_calibration_error": 0.0,
        }

    # Define bins
    if bin_ranges is not None:
        bins = bin_ranges
    else:
        step = 1.0 / n_bins
        bins = [(round(i * step, 4), round((i + 1) * step, 4)) for i in range(n_bins)]

    buckets_data = []
    total_ece_acc = 0.0
    max_ce = 0.0

    for idx, (b_low, b_high) in enumerate(bins):
        # Inclusive of upper bound for the final bin
        if idx == len(bins) - 1:
            in_bucket = [
                e for e in parsed_entries
                if b_low <= e["confidence"] <= b_high
            ]
        else:
            in_bucket = [
                e for e in parsed_entries
                if b_low <= e["confidence"] < b_high
            ]

        count = len(in_bucket)
        if count > 0:
            avg_conf = round(sum(e["confidence"] for e in in_bucket) / count, 4)
            accuracy = round(sum(1 for e in in_bucket if e["is_correct"]) / count, 4)
            cal_err = round(abs(avg_conf - accuracy), 4)
            total_ece_acc += cal_err * (count / total_n)
            max_ce = max(max_ce, cal_err)
        else:
            avg_conf = None
            accuracy = None
            cal_err = None

        buckets_data.append({
            "bin_range": (b_low, b_high),
            "bin_label": f"{b_low:.1f}-{b_high:.1f}",
            "count": count,
            "avg_confidence": avg_conf,
            "empirical_accuracy": accuracy,
            "calibration_error": cal_err,
        })

    return {
        "total_samples": total_n,
        "buckets": buckets_data,
        "expected_calibration_error": round(total_ece_acc, 4),
        "max_calibration_error": round(max_ce, 4),
    }

