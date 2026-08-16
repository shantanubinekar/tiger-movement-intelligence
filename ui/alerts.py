"""
ui/alerts.py — Movement Alerts & Ecological Deviations page.

Automated deviation detection with artefact suppression. Categorizes credible biological
signals, survey/camera relocation artefacts, and cases requiring human verification.
"""

from __future__ import annotations

import streamlit as st
from src.schemas import AlertStatus, AlertType


def render():
    st.markdown("##### Movement Alerts & Ecological Deviation Ledger")
    st.caption(
        "Automated deviation detection with artefact suppression. Distinguishes between "
        "credible biological signals, sensor/survey artefacts, and ambiguous edge cases."
    )

    alerts = st.session_state.get("alerts", [])

    if not alerts:
        st.info("No movement alerts generated. Process images on the **Processing** page first.")
        return

    # ── Summary Metrics ──────────────────────────────────────────
    active = [a for a in alerts if a.status == AlertStatus.ACTIVE]
    suppressed = [a for a in alerts if a.status == AlertStatus.SUPPRESSED]
    insuff = [a for a in alerts if a.status == AlertStatus.INSUFFICIENT_EVIDENCE]
    review = [a for a in alerts if a.status == AlertStatus.HUMAN_REVIEW_REQUIRED]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active (Biological) 🟢", len(active))
    c2.metric("Suppressed (Artefact) 🛡️", len(suppressed))
    c3.metric("Insufficient Evidence ⚪", len(insuff))
    c4.metric("Human Review 🟡", len(review))

    # ── Filter ──────────────────────────────────────────────────
    all_statuses = [s.value for s in AlertStatus]
    filter_status = st.multiselect(
        "Filter alerts by status:",
        options=all_statuses,
        default=all_statuses,
    )

    filtered = [a for a in alerts if a.status.value in filter_status]
    st.caption(f"Showing {len(filtered)} of {len(alerts)} alerts")

    if not filtered:
        st.info("No alerts match the selected filters.")
        return

    # ── Alert Cards ─────────────────────────────────────────────
    status_config = {
        AlertStatus.ACTIVE: {
            "icon": "🟢",
            "badge": "ACTIVE — Credible Biological Signal",
            "default_expanded": True,
        },
        AlertStatus.SUPPRESSED: {
            "icon": "🛡️",
            "badge": "SUPPRESSED — Sensor / Survey Artefact",
            "default_expanded": False,
        },
        AlertStatus.INSUFFICIENT_EVIDENCE: {
            "icon": "⚪",
            "badge": "INSUFFICIENT EVIDENCE — Gated by Safety Rules",
            "default_expanded": False,
        },
        AlertStatus.HUMAN_REVIEW_REQUIRED: {
            "icon": "🟡",
            "badge": "HUMAN REVIEW REQUIRED — Ambiguous Boundary Case",
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

        suppression_summary = f" · ℹ️ Suppressed: {a.suppression_reason}" if a.suppression_reason else ""

        with st.expander(
            f"{icon} **{a.alert_type.value}** — Tiger `{a.identity_id}` ({badge_text}){suppression_summary}",
            expanded=cfg["default_expanded"],
        ):
            # Context banner
            if a.status == AlertStatus.ACTIVE:
                st.success(f"**Credible Signal:** {a.explanation}")
            elif a.status == AlertStatus.SUPPRESSED:
                st.info(f"🛡️ **Suppressed Artefact:** {a.suppression_reason or a.explanation}")
            elif a.status == AlertStatus.INSUFFICIENT_EVIDENCE:
                st.warning(f"⚠️ **Insufficient Evidence:** {a.explanation}")
            else:
                st.warning(f"🧑‍🔬 **Review Required:** {a.explanation}")

            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("**Alert Details**")
                st.markdown(f"- Alert ID: `{a.alert_id}`")
                st.markdown(f"- Type: `{a.alert_type.value}`")
                st.markdown(f"- Target Individual: `{a.identity_id}`")
                st.markdown(f"- Confidence: `{a.confidence:.4f}`")
                st.markdown(f"- Status: {icon} `{a.status.value}`")

            with col_r:
                st.markdown("**Evidence Trace**")
                obs_links = ", ".join(f"`{oid}`" for oid in a.evidence_observation_ids) or "None"
                st.markdown(f"- Supporting Observations: {obs_links}")
                st.markdown(f"- Ecological Note: {a.explanation}")
                if a.suppression_reason:
                    st.markdown(f"- Suppression Cause: `{a.suppression_reason}`")

    # ── Ecological Guidance ─────────────────────────────────────
    st.markdown("---")
    st.caption(
        "💡 **Ecological Guidance:** Movement alerts identify new stations, long-distance dispersals, and territory shifts. "
        "The artefact suppression engine actively neutralizes false alarms caused by camera relocations and survey gaps."
    )
