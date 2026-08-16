"""
ui/processing.py — Pipeline execution page.

Handles both synthetic demo and ATRW benchmark data sources.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.pipeline import (
    create_observation,
    generate_movement_alerts,
    process_image_directory,
    run_evaluation,
)
from src.schemas import IdentityDecisionState


def render():
    default_dir = "data/real_tigers/query"

    st.markdown("##### Pipeline Execution")
    st.caption(
        "Ingestion → Triage → Stripe Keypoint Matching → Evidence Gating → "
        "Trusted History → Movement Alerts"
    )

    st.info(
        "🐅 **ATRW Benchmark Pipeline** — Ingests and processes genuine tiger camera-trap query crops against the catalogue.",
    )

    # ── Controls ───────────────────────────────────────────────
    col_path, col_btn = st.columns([3, 1])
    with col_path:
        folder_path = st.text_input(
            "Dataset folder or image directory path:",
            value="data/real_tigers",
            help="Specify any directory containing camera-trap images, or a structured dataset directory containing catalogue/ and query/ subfolders.",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_run = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if btn_run:
        p = Path(folder_path.strip())
        if not p.exists():
            st.error(f"❌ Directory not found: `{folder_path}` does not exist. Please verify the folder path.")
            return

        with st.spinner("Processing camera-trap captures and matching against catalogue…"):
            try:
                decisions = process_image_directory(folder_path.strip())
                if not decisions:
                    st.warning(
                        f"⚠️ No valid image files could be processed in `{folder_path}`.\n\n"
                        "**Expected dataset folder structure:**\n"
                        "1. A direct image folder containing `.jpg` / `.png` camera-trap files.\n"
                        "2. Or a structured dataset directory with `query/` and `catalogue/` subfolders and an optional `manifest.csv`.\n"
                        "*(Default bundled benchmark: `data/real_tigers`)*"
                    )
                    return

                st.session_state.decisions = decisions
                observations = [
                    obs for obs in (create_observation(d) for d in decisions)
                    if obs is not None
                ]
                st.session_state.observations = observations
                st.session_state.alerts = generate_movement_alerts(observations)
                st.session_state.eval_reports = run_evaluation()
                st.session_state.processed = True

                st.success(
                    f"✅ Processed **{len(decisions)}** images from `{folder_path}` → "
                    f"**{len(observations)}** trusted observations → "
                    f"**{len(st.session_state.alerts)}** alerts"
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

    # ── Results ────────────────────────────────────────────────
    decisions = st.session_state.get("decisions", [])
    if not decisions:
        st.info("Click **Run Pipeline** to process images.")
        return

    # Decision breakdown
    state_counts = {}
    for d in decisions:
        label = d.decision.value
        state_counts[label] = state_counts.get(label, 0) + 1

    st.markdown("---")
    st.markdown("##### Decision Breakdown")
    num_cols = max(1, min(len(state_counts), 4))
    cols = st.columns(num_cols)
    state_icons = {
        "trusted_match": "✅", "ambiguous_review": "🟡", "unknown": "🟠",
        "insufficient_evidence": "⚪", "non_tiger": "🦌", "blank": "🌿", "rejected": "🚫"
    }
    for i, (label, count) in enumerate(state_counts.items()):
        icon = state_icons.get(label, "·")
        cols[i % num_cols].metric(f"{icon} {label.replace('_', ' ').title()}", count)

    # Decision ledger table
    st.markdown("---")
    st.markdown("##### Decision Ledger")
    rows = []
    for d in decisions:
        rows.append({
            "Image ID": d.image_id,
            "Decision": d.decision.value,
            "Identity": d.identity_id or "—",
            "Confidence": f"{d.confidence:.4f}",
            "Reason Codes": ", ".join(rc.value for rc in d.reason_codes) or "—",
            "History": "✅" if d.update_history else "❌",
        })
    st.dataframe(rows, use_container_width=True)
