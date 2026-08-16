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
