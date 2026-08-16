"""
ui/review.py — Review Queue page: per-image identity decision details.

For each processed image: image_id, top-3 candidates with component
evidence scores, decision, confidence, reason codes, and whether
trusted history was updated.
"""

import streamlit as st

from src.schemas import IdentityDecisionState


def render():
    st.header("🔍 Review Queue")

    decisions = st.session_state.get("decisions", [])
    if not decisions:
        st.info("No decisions to review. Process images first on the **Processing** page.")
        return

    # ------ Filter controls ------
    filter_states = st.multiselect(
        "Filter by decision state",
        options=[s.value for s in IdentityDecisionState],
        default=[
            IdentityDecisionState.AMBIGUOUS_REVIEW.value,
            IdentityDecisionState.UNKNOWN.value,
            IdentityDecisionState.INSUFFICIENT_EVIDENCE.value,
        ],
        help="Select which decision states to display for review.",
    )

    filtered = [d for d in decisions if d.decision.value in filter_states] if filter_states else decisions

    st.caption(f"Showing **{len(filtered)}** of {len(decisions)} decisions")

    if not filtered:
        st.info("No decisions match the selected filters.")
        return

    # ------ Per-decision cards ------
    for d in filtered:
        # Decision state badge color
        state_colors = {
            "trusted_match": "🟢",
            "ambiguous_review": "🟡",
            "unknown": "🟠",
            "insufficient_evidence": "🔴",
            "non_tiger": "⚪",
            "blank": "⬜",
            "rejected": "🔴",
        }
        badge = state_colors.get(d.decision.value, "⚪")

        with st.expander(
            f"{badge} **{d.image_id}** — {d.decision.value} "
            f"(confidence: {d.confidence:.3f})",
            expanded=(d.decision != IdentityDecisionState.TRUSTED_MATCH),
        ):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Decision Details**")
                st.markdown(f"- **Decision:** `{d.decision.value}`")
                st.markdown(f"- **Identity:** `{d.identity_id or 'None'}`")
                st.markdown(f"- **Confidence:** `{d.confidence:.4f}`")
                st.markdown(
                    f"- **Update History:** {'✅ Yes' if d.update_history else '❌ No'}"
                )
                st.markdown(
                    f"- **Reason Codes:** {', '.join(f'`{rc.value}`' for rc in d.reason_codes) or 'None'}"
                )
                if d.evidence_summary:
                    st.markdown(f"- **Evidence Summary:** `{d.evidence_summary}`")

            with col2:
                st.markdown("**Top Candidates**")
                if not d.top_candidates:
                    st.caption("No candidates generated.")
                else:
                    for c in d.top_candidates[:3]:
                        st.markdown(f"**Rank {c.rank}: `{c.candidate_identity}`**")
                        score_cols = st.columns(3)
                        score_cols[0].metric("Visual", f"{c.visual_score:.3f}")
                        score_cols[1].metric("Quality", f"{c.quality_score:.3f}")
                        score_cols[2].metric("Total Evidence", f"{c.total_evidence:.3f}")

                        score_cols2 = st.columns(3)
                        score_cols2[0].metric("Spatial", f"{c.spatial_feasibility:.3f}")
                        score_cols2[1].metric("Temporal", f"{c.temporal_feasibility:.3f}")
                        score_cols2[2].metric("History", f"{c.history_consistency:.3f}")
                        st.markdown("---")
