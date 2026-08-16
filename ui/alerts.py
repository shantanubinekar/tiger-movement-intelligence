"""
ui/alerts.py — Alerts page: movement alert list with suppression info.

Shows: alert type, evidence used, confidence, status, and suppression
reason if suppressed. Color-coded by status.
"""

import streamlit as st

from src.schemas import AlertStatus


def render():
    st.header("🚨 Movement Alerts")

    alerts = st.session_state.get("alerts", [])

    if not alerts:
        st.info(
            "No alerts generated yet. Process images on the "
            "**Processing** page first."
        )
        return

    # ------ Summary metrics ------
    active = [a for a in alerts if a.status == AlertStatus.ACTIVE]
    suppressed = [a for a in alerts if a.status == AlertStatus.SUPPRESSED]
    insuff = [a for a in alerts if a.status == AlertStatus.INSUFFICIENT_EVIDENCE]
    review = [a for a in alerts if a.status == AlertStatus.HUMAN_REVIEW_REQUIRED]

    cols = st.columns(4)
    cols[0].metric("Active 🟢", len(active))
    cols[1].metric("Suppressed 🔴", len(suppressed))
    cols[2].metric("Insufficient Evidence ⚪", len(insuff))
    cols[3].metric("Human Review 🟡", len(review))

    # ------ Filter ------
    filter_status = st.multiselect(
        "Filter by status",
        options=[s.value for s in AlertStatus],
        default=[s.value for s in AlertStatus],
    )

    filtered = [a for a in alerts if a.status.value in filter_status]

    st.caption(f"Showing **{len(filtered)}** of {len(alerts)} alerts")

    # ------ Alert cards ------
    for a in filtered:
        # Status indicator
        status_display = {
            AlertStatus.ACTIVE: ("🟢", "Active — likely biological signal"),
            AlertStatus.SUPPRESSED: ("🔴", "Suppressed — likely artefact"),
            AlertStatus.INSUFFICIENT_EVIDENCE: ("⚪", "Insufficient evidence"),
            AlertStatus.HUMAN_REVIEW_REQUIRED: ("🟡", "Human review required"),
        }
        icon, label = status_display.get(a.status, ("⚪", a.status.value))

        with st.expander(
            f"{icon} **{a.alert_type.value}** — {a.identity_id} "
            f"(confidence: {a.confidence:.2f}) — {label}",
            expanded=(a.status == AlertStatus.ACTIVE),
        ):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Alert Details**")
                st.markdown(f"- **Alert ID:** `{a.alert_id}`")
                st.markdown(f"- **Type:** `{a.alert_type.value}`")
                st.markdown(f"- **Identity:** `{a.identity_id}`")
                st.markdown(f"- **Confidence:** `{a.confidence:.4f}`")
                st.markdown(f"- **Status:** {icon} `{a.status.value}`")

            with col2:
                st.markdown("**Evidence & Explanation**")
                st.markdown(f"- **Evidence Observations:** {', '.join(f'`{oid}`' for oid in a.evidence_observation_ids) or 'None'}")
                st.markdown(f"- **Explanation:** {a.explanation}")

                if a.suppression_reason:
                    st.error(f"**Suppression Reason:** {a.suppression_reason}")

    # ------ Interpretation guidance ------
    st.markdown("---")
    st.caption(
        "**Interpretation:** Alerts distinguish between likely biological "
        "signals, likely observation/survey artefacts, insufficient evidence, "
        "and cases requiring human review. Suppressed alerts include a stated "
        "reason (e.g. camera relocation, insufficient history)."
    )
