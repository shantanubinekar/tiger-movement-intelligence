"""
ui/review.py — Review Queue page: interactive per-image identity decision review.

Allows human reviewers to inspect automated decisions, candidate evidence breakdown,
and actively accept, reject, mark unknown, or mark insufficient evidence.
Maintains a separate 'reviewed by human' badge/log for full traceability in demonstrations.
"""

from datetime import datetime, timezone
import streamlit as st

from src.pipeline import create_observation, generate_movement_alerts
from src.schemas import IdentityDecisionState, ReasonCode


def _sync_observations_and_alerts():
    """Re-synchronize st.session_state observations and alerts after a manual review."""
    decisions = st.session_state.get("decisions", [])
    observations = [
        obs
        for obs in (create_observation(d) for d in decisions)
        if obs is not None
    ]
    st.session_state.observations = observations
    st.session_state.alerts = generate_movement_alerts(observations)


def render():
    st.header("🔍 Review Queue")
    st.caption("Human-in-the-loop decision interface for ambiguous, unverified, or low-evidence camera-trap captures.")

    if "human_reviews" not in st.session_state:
        st.session_state.human_reviews = {}

    decisions = st.session_state.get("decisions", [])
    if not decisions:
        st.info("No decisions to review. Process images first on the **Processing** page.")
        return

    # ------ Summary & Filter controls ------
    human_reviews = st.session_state.human_reviews
    num_reviewed = len(human_reviews)

    col_stats1, col_stats2, col_stats3 = st.columns(3)
    col_stats1.metric("Total Decisions", len(decisions))
    col_stats2.metric("Manually Reviewed", num_reviewed)
    col_stats3.metric(
        "Pending Review",
        sum(1 for d in decisions if d.decision == IdentityDecisionState.AMBIGUOUS_REVIEW),
    )

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

    show_reviewed_only = st.checkbox("Show only manually reviewed decisions", value=False)

    filtered = [d for d in decisions if d.decision.value in filter_states] if filter_states else decisions
    if show_reviewed_only:
        filtered = [d for d in filtered if d.image_id in human_reviews]

    st.caption(f"Showing **{len(filtered)}** of {len(decisions)} decisions")

    if not filtered:
        st.info("No decisions match the selected filters.")
        return

    # ------ Per-decision cards ------
    state_colors = {
        "trusted_match": "🟢",
        "ambiguous_review": "🟡",
        "unknown": "🟠",
        "insufficient_evidence": "🔴",
        "non_tiger": "⚪",
        "blank": "⬜",
        "rejected": "🚫",
    }

    for d in filtered:
        review_entry = human_reviews.get(d.image_id)
        is_reviewed = review_entry is not None

        badge = state_colors.get(d.decision.value, "⚪")
        review_tag = " [🧑‍🔬 Reviewed]" if is_reviewed else ""

        with st.expander(
            f"{badge} **{d.image_id}** — `{d.decision.value}` "
            f"(Identity: {d.identity_id or 'None'}){review_tag}",
            expanded=(d.decision == IdentityDecisionState.AMBIGUOUS_REVIEW or is_reviewed),
        ):
            # Human review badge / audit log
            if is_reviewed:
                st.success(
                    f"🧑‍🔬 **Reviewed by Human**: Original automated decision `{review_entry['original_decision']}` "
                    f"→ Updated to **`{review_entry['new_decision']}`** "
                    f"(Assigned: **`{review_entry['assigned_identity'] or 'None'}`**) "
                    f"| *Logged at {review_entry['timestamp']}*"
                )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Automated Decision Details")
                st.markdown(f"- **Current State:** `{d.decision.value}`")
                st.markdown(f"- **Current Identity:** `{d.identity_id or 'None'}`")
                st.markdown(f"- **Confidence:** `{d.confidence:.4f}`")
                st.markdown(
                    f"- **Feeds Trusted History:** {'✅ Yes' if d.update_history else '❌ No'}"
                )
                st.markdown(
                    f"- **Reason Codes:** {', '.join(f'`{rc.value}`' for rc in d.reason_codes) or 'None'}"
                )
                if d.evidence_summary:
                    meta_station = d.evidence_summary.get("station_id", "—")
                    meta_cam = d.evidence_summary.get("camera_status", "—")
                    st.markdown(f"- **Station / Camera:** `{meta_station}` / `{meta_cam}`")

            with col2:
                st.markdown("##### Candidate Evidence Breakdown")
                if not d.top_candidates:
                    st.caption("No identity candidates available.")
                else:
                    for c in d.top_candidates[:2]:
                        st.markdown(f"**Rank {c.rank}: `{c.candidate_identity}`**")
                        s_col1, s_col2, s_col3 = st.columns(3)
                        s_col1.metric("Visual", f"{c.visual_score:.3f}")
                        s_col2.metric("Quality", f"{c.quality_score:.3f}")
                        s_col3.metric("Evidence", f"{c.total_evidence:.3f}")

                        s_col4, s_col5, s_col6 = st.columns(3)
                        s_col4.metric("Spatial", f"{c.spatial_feasibility:.3f}")
                        s_col5.metric("Temporal", f"{c.temporal_feasibility:.3f}")
                        s_col6.metric("History", f"{c.history_consistency:.3f}")
                        st.markdown("---")

            # ------ Interactive Review Controls ------
            st.markdown("##### 🛠️ Human Review Actions")

            # Identity selector for manual assignment
            candidate_options = [c.candidate_identity for c in d.top_candidates] if d.top_candidates else []
            if d.identity_id and d.identity_id not in candidate_options:
                candidate_options.insert(0, d.identity_id)
            if not candidate_options:
                candidate_options = ["T01", "T02", "T03", "NEW-001"]

            selected_identity = st.selectbox(
                "Confirm / Select Target Tiger Identity:",
                options=candidate_options,
                index=0,
                key=f"ident_select_{d.image_id}",
            )

            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

            # 1. Accept as Trusted Match
            if btn_col1.button("🟢 Accept as Trusted", key=f"btn_accept_{d.image_id}", type="primary"):
                orig_state = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.TRUSTED_MATCH
                d.identity_id = selected_identity
                d.update_history = True
                if ReasonCode.HIGH_CONFIDENCE_MATCH not in d.reason_codes:
                    d.reason_codes.insert(0, ReasonCode.HIGH_CONFIDENCE_MATCH)

                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig_state,
                    "new_decision": IdentityDecisionState.TRUSTED_MATCH.value,
                    "assigned_identity": selected_identity,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            # 2. Reject
            if btn_col2.button("🚫 Reject", key=f"btn_reject_{d.image_id}"):
                orig_state = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.REJECTED
                d.identity_id = None
                d.update_history = False

                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig_state,
                    "new_decision": IdentityDecisionState.REJECTED.value,
                    "assigned_identity": None,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            # 3. Mark Unknown
            if btn_col3.button("🟠 Mark Unknown", key=f"btn_unknown_{d.image_id}"):
                orig_state = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.UNKNOWN
                d.identity_id = "NEW-001" if d.identity_id and d.identity_id.startswith("NEW-") else None
                d.update_history = False

                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig_state,
                    "new_decision": IdentityDecisionState.UNKNOWN.value,
                    "assigned_identity": d.identity_id,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            # 4. Mark Insufficient Evidence
            if btn_col4.button("🔴 Insufficient", key=f"btn_insuff_{d.image_id}"):
                orig_state = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.INSUFFICIENT_EVIDENCE
                d.identity_id = None
                d.update_history = False

                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig_state,
                    "new_decision": IdentityDecisionState.INSUFFICIENT_EVIDENCE.value,
                    "assigned_identity": None,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            # 5. Reset to Automated
            if is_reviewed and btn_col5.button("↩️ Reset to Auto", key=f"btn_reset_{d.image_id}"):
                orig_state_val = review_entry["original_decision"]
                for s in IdentityDecisionState:
                    if s.value == orig_state_val:
                        d.decision = s
                        break
                d.update_history = (d.decision == IdentityDecisionState.TRUSTED_MATCH)
                d.identity_id = d.top_candidates[0].candidate_identity if d.top_candidates else None
                del st.session_state.human_reviews[d.image_id]
                _sync_observations_and_alerts()
                st.rerun()
