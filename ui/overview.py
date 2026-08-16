"""
ui/overview.py — Overview page: system-wide counts and status.

Shows counts of: images processed, blank, nonblank, uncertain,
trusted, ambiguous, unknown, insufficient_evidence, alerts,
suppressed alerts.
"""

import streamlit as st

from src.schemas import AlertStatus, IdentityDecisionState


def render():
    st.header("📊 System Overview")

    decisions = st.session_state.get("decisions", [])
    observations = st.session_state.get("observations", [])
    alerts = st.session_state.get("alerts", [])

    if not decisions:
        st.info(
            "No images processed yet. Go to the **Processing** page to "
            "run the pipeline on a demo or real image folder."
        )
        return

    # ------ Decision counts ------
    total = len(decisions)
    counts = {}
    for state in IdentityDecisionState:
        counts[state.value] = sum(
            1 for d in decisions if d.decision == state
        )

    st.subheader("Identity Decisions")
    cols = st.columns(4)
    cols[0].metric("Total Processed", total)
    cols[1].metric("Blank", counts.get("blank", 0))
    cols[2].metric("Non-Tiger", counts.get("non_tiger", 0))
    cols[3].metric("Trusted Match ✅", counts.get("trusted_match", 0))

    cols2 = st.columns(4)
    cols2[0].metric("Ambiguous (Review)", counts.get("ambiguous_review", 0))
    cols2[1].metric("Unknown", counts.get("unknown", 0))
    cols2[2].metric("Insufficient Evidence", counts.get("insufficient_evidence", 0))
    cols2[3].metric("Rejected", counts.get("rejected", 0))

    # ------ Observation counts ------
    st.subheader("Trusted Observations")
    st.metric("Observations Created", len(observations))

    # ------ Alert counts ------
    st.subheader("Movement Alerts")
    total_alerts = len(alerts)
    active_alerts = sum(1 for a in alerts if a.status == AlertStatus.ACTIVE)
    suppressed = sum(1 for a in alerts if a.status == AlertStatus.SUPPRESSED)
    insuff = sum(
        1 for a in alerts if a.status == AlertStatus.INSUFFICIENT_EVIDENCE
    )
    review = sum(
        1 for a in alerts if a.status == AlertStatus.HUMAN_REVIEW_REQUIRED
    )

    cols3 = st.columns(4)
    cols3[0].metric("Total Alerts", total_alerts)
    cols3[1].metric("Active 🟢", active_alerts)
    cols3[2].metric("Suppressed 🔴", suppressed)
    cols3[3].metric("Insufficient / Review", insuff + review)

    # ------ Data mode warning ------
    demo_count = sum(
        1
        for d in decisions
        if d.evidence_summary.get("data_mode") == "demo"
    )
    if demo_count > 0:
        st.caption(
            f"ℹ️ {demo_count}/{total} decisions are from **DEMO MODE** data."
        )
