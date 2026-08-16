"""
ui/alerts.py — Alerts page: movement alert list with suppression info.

Shows: alert type, evidence used, confidence, status, and suppression
reason if suppressed. Color-coded by status with prominent status badges
and visible suppression explanations.
"""

from __future__ import annotations

import streamlit as st

from src.schemas import AlertStatus, AlertType


def render():
    st.header("🚨 Movement Alerts & Ecological Deviations")
    st.caption(
        "Automated deviation detection with artefact suppression. Alerts distinguish between "
        "credible biological signals, survey/sensor artefacts, and cases requiring human verification."
    )

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active (Biological) 🟢", len(active))
    col2.metric("Suppressed (Artefact) 🔴", len(suppressed))
    col3.metric("Insufficient Evidence ⚪", len(insuff))
    col4.metric("Human Review 🟡", len(review))

    # ------ Filter ------
    filter_status = st.multiselect(
        "Filter alerts by status:",
        options=[s.value for s in AlertStatus],
        default=[s.value for s in AlertStatus],
    )

    filtered = [a for a in alerts if a.status.value in filter_status]

    st.caption(f"Showing **{len(filtered)}** of {len(alerts)} alerts")

    # ------ Alert cards ------
    status_config = {
        AlertStatus.ACTIVE: {
            "icon": "🟢",
            "badge": "ACTIVE — Likely Biological Signal",
            "border": "green",
            "default_expanded": True,
        },
        AlertStatus.SUPPRESSED: {
            "icon": "🔴",
            "badge": "SUPPRESSED — Observation / Survey Artefact",
            "border": "gray",
            "default_expanded": False,
        },
        AlertStatus.INSUFFICIENT_EVIDENCE: {
            "icon": "⚪",
            "badge": "INSUFFICIENT EVIDENCE — Gated by Safety Rules",
            "border": "gray",
            "default_expanded": False,
        },
        AlertStatus.HUMAN_REVIEW_REQUIRED: {
            "icon": "🟡",
            "badge": "HUMAN REVIEW REQUIRED — Ambiguous / Boundary Case",
            "border": "orange",
            "default_expanded": True,
        },
    }

    for a in filtered:
        cfg = status_config.get(
            a.status,
            {"icon": "⚪", "badge": a.status.value, "default_expanded": False},
        )
        icon = cfg["icon"]
        badge_text = cfg["badge"]

        # If suppressed, highlight the suppression reason prominently
        suppression_summary = f" | ℹ️ Reason: {a.suppression_reason}" if a.suppression_reason else ""

        with st.expander(
            f"{icon} **{a.alert_type.value}** — Tiger `{a.identity_id}` "
            f"({badge_text}){suppression_summary}",
            expanded=cfg["default_expanded"],
        ):
            # Banner for quick context
            if a.status == AlertStatus.ACTIVE:
                st.success(f"**Credible Signal:** {a.explanation}")
            elif a.status == AlertStatus.SUPPRESSED:
                st.info(f"🛡️ **Suppressed Artefact:** {a.suppression_reason or a.explanation}")
            elif a.status == AlertStatus.INSUFFICIENT_EVIDENCE:
                st.warning(f"⚠️ **Insufficient Evidence:** {a.explanation}")
            else:
                st.warning(f"🧑‍🔬 **Review Required:** {a.explanation}")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("##### Alert Identification")
                st.markdown(f"- **Alert ID:** `{a.alert_id}`")
                st.markdown(f"- **Alert Type:** `{a.alert_type.value}`")
                st.markdown(f"- **Individual:** `{a.identity_id}`")
                st.markdown(f"- **Confidence Score:** `{a.confidence:.4f}`")
                st.markdown(f"- **Status:** {icon} `{a.status.value}`")

            with col_right:
                st.markdown("##### Supporting Evidence")
                obs_links = ", ".join(f"`{oid}`" for oid in a.evidence_observation_ids) or "None"
                st.markdown(f"- **Evidence Observation(s):** {obs_links}")
                st.markdown(f"- **Full Ecological Details:** {a.explanation}")
                if a.suppression_reason:
                    st.markdown(f"- **Root Cause of Suppression:** `{a.suppression_reason}`")

    # ------ Interpretation guidance ------
    st.markdown("---")
    st.info(
        "**Ecological Intelligence Guidance:** Movement alerts flag potential territory shifts, "
        "long-distance dispersals, and unusual absences. The artefact suppression engine prevents false alarms "
        "caused by camera relocations, seasonal survey gaps, and ambiguous captures."
    )
