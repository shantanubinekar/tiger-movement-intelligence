"""
ui/overview.py — Dashboard page with system-wide operational metrics.
"""

from __future__ import annotations

import streamlit as st
from src.schemas import AlertStatus, IdentityDecisionState


def render():
    is_atrw = st.session_state.get("data_source") == "Real Tiger Images (ATRW Benchmark)"

    decisions = st.session_state.get("decisions", [])
    observations = st.session_state.get("observations", [])
    alerts = st.session_state.get("alerts", [])

    # ── Welcome State ──────────────────────────────────────────
    if not decisions:
        st.markdown("### Welcome to the Movement Intelligence Portal")

        if is_atrw:
            st.info(
                "🐅 **ATRW Benchmark Mode** — Select '⚙️ Processing' from the sidebar to run the pipeline "
                "on genuine tiger images, or switch the data source above.",
            )
        else:
            st.info(
                "🌲 **Synthetic Demo Mode** — Select '⚙️ Processing' from the sidebar to run the "
                "end-to-end pipeline on bundled demo scenarios.",
            )

        st.markdown("---")
        st.markdown("##### System Architecture")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                **Ingestion Pipeline**
                1. EXIF metadata extraction & validation
                2. Visual triage — blank / non-tiger filtering
                3. Classical stripe keypoint feature extraction (SIFT/ORB)

                **Identity Matching**
                4. Candidate generation via embedding similarity
                5. Spatial-temporal feasibility scoring
                6. Multi-evidence gating (E = Σ Wᵢ · Fᵢ)
                """
            )
        with col_b:
            st.markdown(
                """
                **Ecological Intelligence**
                7. Trusted observation store (only `trusted_match`)
                8. Historical capture area computation (convex hull)
                9. Movement deviation detection & alert generation
                10. Artefact suppression (camera relocation, survey gaps)

                **Review & Audit**
                11. Human-in-the-loop verification queue
                12. Quantitative evaluation against independent ground truth
                """
            )
        return

    # ── Operational Metrics ────────────────────────────────────
    total = len(decisions)
    counts = {}
    for state in IdentityDecisionState:
        counts[state.value] = sum(1 for d in decisions if d.decision == state)

    mode_label = "ATRW Benchmark" if is_atrw else "Synthetic Demo"
    st.caption(f"Active data source: **{mode_label}** · {total} images processed")

    # Row 1: Primary counts
    st.markdown("##### Identity Decision Distribution")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Processed", total)
    c2.metric("Trusted Match", counts.get("trusted_match", 0))
    c3.metric("Pending Review", counts.get("ambiguous_review", 0))
    c4.metric("Unknown", counts.get("unknown", 0))

    # Row 2: Secondary counts
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Insufficient Evidence", counts.get("insufficient_evidence", 0))
    c6.metric("Non-Tiger", counts.get("non_tiger", 0))
    c7.metric("Blank Frame", counts.get("blank", 0))
    c8.metric("Rejected", counts.get("rejected", 0))

    st.markdown("---")

    # Row 3: Observation & Alert summary
    col_obs, col_alt = st.columns(2)

    with col_obs:
        st.markdown("##### Trusted Observation Store")
        st.metric(
            "Observations Created",
            len(observations),
            delta=f"{len(observations)} of {total} admitted",
            delta_color="normal",
        )
        withheld = total - len(observations)
        if total > 0:
            st.caption(
                f"🛡️ {withheld} capture(s) ({withheld/total*100:.0f}%) withheld by evidence gating."
            )

    with col_alt:
        st.markdown("##### Movement Alert Summary")
        active_n = sum(1 for a in alerts if a.status == AlertStatus.ACTIVE)
        suppressed_n = sum(1 for a in alerts if a.status == AlertStatus.SUPPRESSED)
        review_n = sum(
            1 for a in alerts if a.status in (AlertStatus.HUMAN_REVIEW_REQUIRED, AlertStatus.INSUFFICIENT_EVIDENCE)
        )
        st.metric(
            "Active Alerts",
            active_n,
            delta=f"{suppressed_n} artefacts suppressed",
            delta_color="inverse",
        )
        st.caption(
            f"Total candidates: {len(alerts)} · Suppressed: {suppressed_n} · Under review: {review_n}"
        )

    # Human Review Audit
    human_reviews = st.session_state.get("human_reviews", {})
    if human_reviews:
        st.markdown("---")
        st.markdown("##### Human Review Audit Trail")
        st.caption(f"{len(human_reviews)} manual decision override(s) logged this session.")
