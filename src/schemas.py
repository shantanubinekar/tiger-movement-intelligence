"""
schemas.py — SINGLE SOURCE OF TRUTH for shared data structures.

Evidence-Aware Tiger Camera-Trap Movement Intelligence System (SIH prototype)

DO NOT REDEFINE THESE MODELS LOCALLY IN ANY OTHER FILE.
If a field is genuinely missing for your task: STOP, do not invent it
silently, propose the addition here, and flag it to the other two
developers before depending on it.

All three developer branches (frontend/integration, backend-perception,
backend-history) must import from this exact file, unmodified, unless a
change has been agreed and committed here first.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — fixed vocabularies. Do not invent new values ad hoc; if you need
# one, add it here so every branch sees the same set.
# ---------------------------------------------------------------------------

class DataMode(str, Enum):
    """Marks whether a record is real field data or synthetic/demo data.
    Never presented as real Pench observations when demo."""
    DEMO = "demo"
    REAL = "real"


class CameraStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RELOCATED = "relocated"
    UNKNOWN = "unknown"


class TriageStatus(str, Enum):
    BLANK = "blank"
    NONBLANK = "nonblank"
    UNCERTAIN = "uncertain"


class IdentityDecisionState(str, Enum):
    """Allowed identity decision states (PROJECT_CONTRACT.md Section 11).
    Only TRUSTED_MATCH may update trusted longitudinal history."""
    TRUSTED_MATCH = "trusted_match"
    AMBIGUOUS_REVIEW = "ambiguous_review"
    UNKNOWN = "unknown"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NON_TIGER = "non_tiger"
    BLANK = "blank"
    REJECTED = "rejected"


class ObservationStatus(str, Enum):
    TRUSTED = "trusted"
    PROVISIONAL = "provisional"


class AlertType(str, Enum):
    NEW_STATION = "NEW_STATION"
    OUTSIDE_HISTORICAL_AREA = "OUTSIDE_HISTORICAL_AREA"
    UNUSUAL_TRAVEL = "UNUSUAL_TRAVEL"
    PROLONGED_ABSENCE = "PROLONGED_ABSENCE"
    BUFFER_OR_VILLAGE_ADJACENT = "BUFFER_OR_VILLAGE_ADJACENT"
    POSSIBLE_DISPERSAL = "POSSIBLE_DISPERSAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CAMERA_OR_SURVEY_ARTEFACT = "CAMERA_OR_SURVEY_ARTEFACT"


class AlertStatus(str, Enum):
    """What the alert ultimately became — this is what lets Dev1's UI and
    Dev3's evaluation distinguish signal from artefact from suppression."""
    ACTIVE = "active"                # a credible biological signal
    SUPPRESSED = "suppressed"        # likely observation/survey artefact
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class ReasonCode(str, Enum):
    """Fixed vocabulary for IdentityDecision.reason_codes. Extend here only."""
    LOW_VISUAL_MARGIN = "LOW_VISUAL_MARGIN"
    POOR_IMAGE_QUALITY = "POOR_IMAGE_QUALITY"
    FLANK_NOT_VISIBLE = "FLANK_NOT_VISIBLE"
    CAMERA_RELOCATED = "CAMERA_RELOCATED"
    MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
    MISSING_LOCATION = "MISSING_LOCATION"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    TRAVEL_SPEED_IMPLAUSIBLE = "TRAVEL_SPEED_IMPLAUSIBLE"
    NEW_STATION = "NEW_STATION"
    HIGH_CONFIDENCE_MATCH = "HIGH_CONFIDENCE_MATCH"


# ---------------------------------------------------------------------------
# Core pipeline records — Developer 2 owns creation of these (ImageRecord
# through IdentityDecision). Developer 3 consumes IdentityDecision and
# produces Observation / MovementAlert. Developer 1 only reads all of them.
# ---------------------------------------------------------------------------

class ImageRecord(BaseModel):
    image_id: str
    image_path: str
    file_hash: str
    station_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[datetime] = None
    camera_status: CameraStatus = CameraStatus.UNKNOWN
    processing_status: str = "pending"
    data_mode: DataMode = DataMode.DEMO


class TriageRecord(BaseModel):
    image_id: str
    blank_probability: float = Field(ge=0.0, le=1.0)
    subject_probability: float = Field(ge=0.0, le=1.0)
    triage_status: TriageStatus


class DetectionRecord(BaseModel):
    image_id: str
    species: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None  # x, y, w, h
    detection_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    flank_visibility: float = Field(default=0.0, ge=0.0, le=1.0)
    crop_path: Optional[str] = None


class IdentityCandidate(BaseModel):
    image_id: str
    candidate_identity: str
    rank: int = Field(ge=1)
    visual_score: float = Field(ge=0.0, le=1.0)
    local_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    spatial_feasibility: float = Field(ge=0.0, le=1.0)
    temporal_feasibility: float = Field(ge=0.0, le=1.0)
    history_consistency: float = Field(ge=0.0, le=1.0)
    total_evidence: float = Field(ge=0.0, le=1.0)


class IdentityDecision(BaseModel):
    image_id: str
    decision: IdentityDecisionState
    identity_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    top_candidates: list[IdentityCandidate] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    evidence_summary: dict = Field(default_factory=dict)
    update_history: bool = False

    def model_post_init(self, __context) -> None:
        # Enforcement checkpoint: update_history must never be True unless
        # decision is TRUSTED_MATCH. This is a safety guard, not a full
        # substitute for the guard clause Developer 3 must also implement
        # in history.py.
        if self.update_history and self.decision != IdentityDecisionState.TRUSTED_MATCH:
            raise ValueError(
                "update_history=True is only allowed when "
                "decision == trusted_match. Refusing to construct an "
                "IdentityDecision that would silently contaminate "
                "trusted history."
            )


# ---------------------------------------------------------------------------
# History / movement / alert records — Developer 3 owns creation.
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    observation_id: str
    image_id: str
    identity_id: str
    station_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[datetime] = None
    identity_confidence: float = Field(ge=0.0, le=1.0)
    observation_status: ObservationStatus = ObservationStatus.TRUSTED
    camera_status: CameraStatus = CameraStatus.UNKNOWN
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MovementAlert(BaseModel):
    alert_id: str
    identity_id: str
    alert_type: AlertType
    confidence: float = Field(ge=0.0, le=1.0)
    status: AlertStatus
    evidence_observation_ids: list[str] = Field(default_factory=list)
    explanation: str
    suppression_reason: Optional[str] = None


class IndividualSummary(BaseModel):
    identity_id: str
    capture_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    trusted_stations: list[str] = Field(default_factory=list)
    activity_centroid: Optional[tuple[float, float]] = None  # (lat, lon)
    historical_capture_area: Optional[list[tuple[float, float]]] = None
    last_seen_duration_days: Optional[float] = None
    camera_effort_history: dict = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Baseline vs. proposed comparison. Never fabricate numbers — use
    `not_computable` to flag any metric that can't be measured on current
    data instead of inventing a plausible-looking value."""
    pipeline_name: str  # "baseline" or "evidence_gated"
    false_confident_identity_rate: Optional[float] = None
    coverage: Optional[float] = None
    abstention_review_rate: Optional[float] = None
    false_movement_alert_rate: Optional[float] = None
    alert_precision: Optional[float] = None
    artefact_suppression_rate: Optional[float] = None
    observations_withheld_pct: Optional[float] = None
    not_computable: list[str] = Field(default_factory=list)
    notes: str = ""
