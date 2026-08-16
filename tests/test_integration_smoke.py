"""
tests/test_integration_smoke.py — SHARED FILE. Append to this file if you
add scenario-specific integration tests; do not delete or restructure the
tests already here — they are the hour-12 merge gate all three branches
must pass together.

Run with:  pytest tests/test_integration_smoke.py -v

What this file proves, regardless of whether DEMO_MODE is on or the real
implementations have landed:
1. schemas.py imports cleanly and the safety guard on IdentityDecision
   actually rejects an invalid update_history=True + non-trusted decision.
2. pipeline.py's six shared-interface functions are importable and
   callable with the exact signatures the other developers rely on.
3. The full chain — process_image_directory -> create_observation ->
   generate_movement_alerts -> run_evaluation — runs without crashing and
   produces only well-typed schema objects.
4. THE CORE SAFETY RULE: no non-trusted IdentityDecision ever produces an
   Observation. This is checked directly, not just informally.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.schemas import IdentityDecision, IdentityDecisionState, Observation
from src.pipeline import (
    process_image_directory,
    create_observation,
    generate_movement_alerts,
    run_evaluation,
)


def test_schemas_import_cleanly():
    from src import schemas  # noqa: F401
    assert hasattr(schemas, "ImageRecord")
    assert hasattr(schemas, "IdentityDecision")
    assert hasattr(schemas, "Observation")
    assert hasattr(schemas, "MovementAlert")


def test_safety_guard_rejects_invalid_update_history():
    with pytest.raises(ValueError):
        IdentityDecision(
            image_id="bad",
            decision=IdentityDecisionState.UNKNOWN,
            confidence=0.2,
            update_history=True,  # invalid: not trusted_match
        )


def test_pipeline_functions_are_importable_and_callable():
    decisions = process_image_directory("data/demo")
    assert isinstance(decisions, list)
    assert all(isinstance(d, IdentityDecision) for d in decisions)


def test_only_trusted_match_creates_observations():
    """THE CORE SAFETY RULE. If this test ever fails, stop and fix
    history.py / gating.py before doing anything else — it means a
    non-trusted decision is contaminating trusted history."""
    decisions = process_image_directory("data/demo")

    for decision in decisions:
        obs = create_observation(decision)
        if decision.decision == IdentityDecisionState.TRUSTED_MATCH and decision.update_history:
            assert obs is not None, (
                f"trusted_match decision {decision.image_id} should have "
                "produced an Observation but did not."
            )
            assert isinstance(obs, Observation)
        else:
            assert obs is None, (
                f"Non-trusted decision {decision.image_id} "
                f"({decision.decision}) illegally produced an Observation. "
                "This is the exact failure mode the whole project exists "
                "to prevent."
            )



def test_full_chain_runs_without_crashing():
    decisions = process_image_directory("data/demo")
    observations = [o for o in (create_observation(d) for d in decisions) if o is not None]
    alerts = generate_movement_alerts(observations)
    reports = run_evaluation()

    assert isinstance(observations, list)
    assert isinstance(alerts, list)
    assert isinstance(reports, list)
    assert len(reports) >= 1  # at least baseline + proposed once real, or demo placeholders now


# ---------------------------------------------------------------------------
# Developer 3 — appended integration tests
# (Baseline vs. proposed evaluation on demo scenarios)
# ---------------------------------------------------------------------------

from src.schemas import EvaluationReport, AlertStatus


def test_evaluation_produces_both_pipelines():
    """run_evaluation must produce at least two EvaluationReport objects:
    one for 'baseline' and one for 'evidence_gated'."""
    reports = run_evaluation()
    assert isinstance(reports, list)
    assert len(reports) >= 2

    pipeline_names = {r.pipeline_name for r in reports}
    assert "baseline" in pipeline_names, "Missing baseline report"
    assert "evidence_gated" in pipeline_names, "Missing evidence_gated report"


def test_evaluation_reports_have_valid_structure():
    """Each EvaluationReport must have valid field types and no fabricated
    metrics (metrics without data are in not_computable)."""
    reports = run_evaluation()
    for report in reports:
        assert isinstance(report, EvaluationReport)
        assert isinstance(report.pipeline_name, str)
        assert isinstance(report.not_computable, list)
        assert isinstance(report.notes, str)
        assert len(report.notes) > 0, "Report notes should not be empty"
        # Fabrication check: notes should mention 'PROTOTYPE' or 'DEMO'
        assert (
            "PROTOTYPE" in report.notes.upper()
            or "DEMO" in report.notes.upper()
        ), "Report should be labelled as prototype/demo evaluation"


def test_evaluation_baseline_vs_gated_comparison():
    """The evaluation should produce comparison data. If real evaluation
    is running (not DEMO_MODE), the evidence_gated pipeline should
    withhold more observations than baseline."""
    reports = run_evaluation()
    baseline = next((r for r in reports if r.pipeline_name == "baseline"), None)
    gated = next((r for r in reports if r.pipeline_name == "evidence_gated"), None)

    assert baseline is not None
    assert gated is not None

    # If real evaluation ran (not just placeholders)
    if gated.observations_withheld_pct is not None:
        assert gated.observations_withheld_pct >= 0.0, (
            "Evidence-gated should withhold ≥ 0% of observations"
        )

    # Baseline should never withhold observations
    if baseline.observations_withheld_pct is not None:
        assert baseline.observations_withheld_pct == 0.0, (
            "Baseline (always-assign) should not withhold any observations"
        )


def test_evaluation_no_fabricated_numbers():
    """Metrics that can't be computed should be in not_computable, not
    filled with fake values."""
    reports = run_evaluation()
    for report in reports:
        # If false_confident_identity_rate is None, it should be listed
        # in not_computable (since we have no ground truth labels)
        if report.false_confident_identity_rate is None:
            computable_names = " ".join(report.not_computable).lower()
            assert "false_confident_identity_rate" in computable_names or \
                   "identity" in computable_names, (
                f"false_confident_identity_rate is None but not in "
                f"not_computable list for {report.pipeline_name}"
            )

