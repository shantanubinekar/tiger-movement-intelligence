"""
ui/review.py — Review Queue: human-in-the-loop identity decision review.

Shows real tiger photos (query + catalogue match) prominently, with evidence
metrics and interactive accept/reject/mark controls.
"""

from datetime import datetime, timezone
from pathlib import Path
import streamlit as st

from src.pipeline import create_observation, generate_movement_alerts
from src.schemas import IdentityDecisionState, ReasonCode


def _sync_observations_and_alerts():
    """Re-synchronize session state after a manual review action."""
    decisions = st.session_state.get("decisions", [])
    observations = [
        obs for obs in (create_observation(d) for d in decisions)
        if obs is not None
    ]
    st.session_state.observations = observations
    st.session_state.alerts = generate_movement_alerts(observations)


def render():
    st.markdown("##### Review Queue")
    st.caption("Inspect automated decisions. Accept, reject, or reclassify captures with full evidence visibility.")

    if "human_reviews" not in st.session_state:
        st.session_state.human_reviews = {}

    decisions = st.session_state.get("decisions", [])
    if not decisions:
        st.info("No decisions to review. Run the pipeline first.")
        return

    human_reviews = st.session_state.human_reviews

    # ── Summary Row ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Decisions", len(decisions))
    c2.metric("Reviewed", len(human_reviews))
    c3.metric(
        "Pending Review",
        sum(1 for d in decisions if d.decision == IdentityDecisionState.AMBIGUOUS_REVIEW),
    )

    # ── Filters ────────────────────────────────────────────────
    all_states = sorted({d.decision.value for d in decisions})
    filter_states = st.multiselect(
        "Filter by decision state:",
        options=all_states,
        default=all_states,
    )
    show_reviewed_only = st.checkbox("Show only manually reviewed", value=False)

    filtered = [d for d in decisions if d.decision.value in filter_states] if filter_states else decisions
    if show_reviewed_only:
        filtered = [d for d in filtered if d.image_id in human_reviews]

    st.caption(f"Showing {len(filtered)} of {len(decisions)} decisions")

    if not filtered:
        st.info("No decisions match the current filters.")
        return

    # ── Decision Cards ─────────────────────────────────────────
    state_icons = {
        "trusted_match": "🟢", "ambiguous_review": "🟡", "unknown": "🟠",
        "insufficient_evidence": "🔴", "non_tiger": "⚪", "blank": "⬜", "rejected": "🚫",
    }

    from src.identity import get_default_catalogue
    from src.perception import generate_match_visualization
    catalogue = get_default_catalogue()

    for d in filtered:
        review_entry = human_reviews.get(d.image_id)
        is_reviewed = review_entry is not None

        icon = state_icons.get(d.decision.value, "⚪")
        reviewed_tag = " · ✏️ Reviewed" if is_reviewed else ""

        with st.expander(
            f"{icon} **{d.image_id}** — {d.decision.value} · "
            f"Identity: {d.identity_id or 'None'}{reviewed_tag}",
            expanded=(d.decision == IdentityDecisionState.AMBIGUOUS_REVIEW or is_reviewed),
        ):
            # Human review log
            if is_reviewed:
                st.success(
                    f"✏️ **Manual Override:** `{review_entry['original_decision']}` → "
                    f"`{review_entry['new_decision']}` "
                    f"(Identity: `{review_entry['assigned_identity'] or 'None'}`) · "
                    f"{review_entry['timestamp']}"
                )

            # ── Photo Display ──────────────────────────────────
            query_img = (
                d.evidence_summary.get("crop_path")
                or d.evidence_summary.get("image_path")
            )
            if not query_img:
                for p_cand in [
                    Path(f"data/real_tigers/query/{d.image_id}.jpg"),
                    Path(f"data/demo/{d.image_id}.jpg"),
                ]:
                    if p_cand.exists():
                        query_img = str(p_cand)
                        break

            best_cat_id = d.identity_id
            best_cat_img = catalogue.get_image_path(best_cat_id) if best_cat_id else None
            if not best_cat_img and d.top_candidates:
                best_cat_id = d.top_candidates[0].candidate_identity
                best_cat_img = catalogue.get_image_path(best_cat_id)

            has_query = query_img and Path(query_img).exists()
            has_cat = best_cat_img and Path(best_cat_img).exists()

            if has_query or has_cat:
                img_left, img_right = st.columns(2)
                with img_left:
                    if has_query:
                        st.image(str(query_img), caption=f"Query: {d.image_id}", use_container_width=True)
                    else:
                        st.caption("📷 No query image available")
                with img_right:
                    if has_cat:
                        st.image(str(best_cat_img), caption=f"Catalogue: {best_cat_id}", use_container_width=True)
                    else:
                        st.caption("📷 No catalogue reference available")
            else:
                st.caption("📷 Synthetic scenario — no image files to display")

            # ── Decision Details + Evidence ────────────────────
            col_detail, col_evidence = st.columns(2)

            with col_detail:
                st.markdown("**Decision Details**")
                st.markdown(f"- State: `{d.decision.value}`")
                st.markdown(f"- Identity: `{d.identity_id or 'None'}`")
                st.markdown(f"- Confidence: `{d.confidence:.4f}`")
                st.markdown(f"- History: {'✅ Admitted' if d.update_history else '❌ Withheld'}")
                if d.reason_codes:
                    st.markdown(f"- Reasons: {', '.join(f'`{rc.value}`' for rc in d.reason_codes)}")
                if d.evidence_summary:
                    station = d.evidence_summary.get("station_id", "—")
                    cam = d.evidence_summary.get("camera_status", "—")
                    st.markdown(f"- Station: `{station}` · Camera: `{cam}`")

            with col_evidence:
                st.markdown("**Top Candidate Evidence**")
                if not d.top_candidates:
                    st.caption("No candidates generated.")
                else:
                    for c in d.top_candidates[:2]:
                        st.markdown(f"**Rank {c.rank}: `{c.candidate_identity}`**")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Visual", f"{c.visual_score:.3f}")
                        mc2.metric("Quality", f"{c.quality_score:.3f}")
                        mc3.metric("Evidence", f"{c.total_evidence:.3f}")

                        mc4, mc5, mc6 = st.columns(3)
                        mc4.metric("Spatial", f"{c.spatial_feasibility:.3f}")
                        mc5.metric("Temporal", f"{c.temporal_feasibility:.3f}")
                        mc6.metric("Stripe", f"{c.local_score:.3f}")

            # ── Keypoint Match Toggle ──────────────────────────
            if has_query and d.top_candidates:
                top_c = d.top_candidates[0]
                top_cat_img = catalogue.get_image_path(top_c.candidate_identity)
                if top_cat_img and Path(top_cat_img).exists():
                    if st.checkbox(
                        f"🔍 Show stripe keypoint correspondence ({top_c.candidate_identity})",
                        key=f"chk_match_{d.image_id}_{top_c.candidate_identity}",
                        value=False,
                    ):
                        vis = generate_match_visualization(str(query_img), str(top_cat_img))
                        if vis is not None:
                            st.image(vis, caption=f"SIFT/ORB keypoint matches: {d.image_id} ↔ {top_c.candidate_identity}", use_container_width=True)
                        else:
                            st.caption("Keypoint visualization unavailable for this pair.")

            st.markdown("---")

            # ── Review Controls ────────────────────────────────
            st.markdown("**Review Actions**")

            candidate_options = [c.candidate_identity for c in d.top_candidates] if d.top_candidates else []
            if d.identity_id and d.identity_id not in candidate_options:
                candidate_options.insert(0, d.identity_id)
            if not candidate_options:
                candidate_options = ["T01", "T02", "T03", "NEW-001"]

            selected_identity = st.selectbox(
                "Assign identity:",
                options=candidate_options,
                index=0,
                key=f"ident_select_{d.image_id}",
            )

            b1, b2, b3, b4, b5 = st.columns(5)

            if b1.button("✅ Accept", key=f"btn_accept_{d.image_id}", type="primary"):
                orig = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.TRUSTED_MATCH
                d.identity_id = selected_identity
                d.update_history = True
                if ReasonCode.HIGH_CONFIDENCE_MATCH not in d.reason_codes:
                    d.reason_codes.insert(0, ReasonCode.HIGH_CONFIDENCE_MATCH)
                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig,
                    "new_decision": IdentityDecisionState.TRUSTED_MATCH.value,
                    "assigned_identity": selected_identity,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            if b2.button("🚫 Reject", key=f"btn_reject_{d.image_id}"):
                orig = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.REJECTED
                d.identity_id = None
                d.update_history = False
                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig,
                    "new_decision": IdentityDecisionState.REJECTED.value,
                    "assigned_identity": None,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            if b3.button("🟠 Unknown", key=f"btn_unknown_{d.image_id}"):
                orig = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.UNKNOWN
                d.identity_id = "NEW-001" if d.identity_id and d.identity_id.startswith("NEW-") else None
                d.update_history = False
                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig,
                    "new_decision": IdentityDecisionState.UNKNOWN.value,
                    "assigned_identity": d.identity_id,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            if b4.button("⚪ Insufficient", key=f"btn_insuff_{d.image_id}"):
                orig = review_entry["original_decision"] if is_reviewed else d.decision.value
                d.decision = IdentityDecisionState.INSUFFICIENT_EVIDENCE
                d.identity_id = None
                d.update_history = False
                st.session_state.human_reviews[d.image_id] = {
                    "original_decision": orig,
                    "new_decision": IdentityDecisionState.INSUFFICIENT_EVIDENCE.value,
                    "assigned_identity": None,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                _sync_observations_and_alerts()
                st.rerun()

            if is_reviewed and b5.button("↩️ Reset", key=f"btn_reset_{d.image_id}"):
                orig_val = review_entry["original_decision"]
                for s in IdentityDecisionState:
                    if s.value == orig_val:
                        d.decision = s
                        break
                d.update_history = (d.decision == IdentityDecisionState.TRUSTED_MATCH)
                d.identity_id = d.top_candidates[0].candidate_identity if d.top_candidates else None
                del st.session_state.human_reviews[d.image_id]
                _sync_observations_and_alerts()
                st.rerun()
