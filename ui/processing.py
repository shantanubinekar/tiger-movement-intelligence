"""
ui/processing.py — Processing page: trigger camera-trap image processing.

Allows processing of both synthetic demo scenarios and genuine ATRW benchmark
tiger images. Executes: Ingestion → Triage → Perception → Classical Stripe Keypoint
Matching → Identity Evidence Gating → Trusted Observation Store → Movement Alerts.
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
    st.header("🔄 Field Ingestion & Processing Pipeline")

    is_atrw = st.session_state.get("data_source") == "Real Tiger Images (ATRW Benchmark)"
    default_dir = "data/real_tigers/query" if is_atrw else "data/demo"

    st.caption(
        "Execute the end-to-end intelligence pipeline: Ingestion → Triage → "
        "Perception & Embedding → Classical Stripe Keypoint Matching → Evidence Gating → Trusted History."
    )

    if is_atrw:
        st.info(
            "🐅 **Benchmark Mode Active:** Processing genuine ATRW tiger query crops against ATRW catalogue entries. "
            "Validated on genuine tiger stripe patterns — illustrative reserve station layout.",
            icon="🐅",
        )

    # ------ Folder selection ------
    col_input, col_info = st.columns([3, 2])
    with col_input:
        folder_path = st.text_input(
            "Camera-trap image folder directory:",
            value=default_dir,
            help="Path to a directory containing camera-trap images (or the bundled dataset).",
        )
    with col_info:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_run = st.button("▶ Run Processing Pipeline", type="primary", use_container_width=True)

    if btn_run:
        with st.spinner("Executing pipeline layers (Triage → Stripe Keypoint Matching → Evidence Gating)…"):
            try:
                decisions = process_image_directory(folder_path)
                st.session_state.decisions = decisions

                # Route through create_observation → movement alerts → evaluation
                observations = [
                    obs
                    for obs in (create_observation(d) for d in decisions)
                    if obs is not None
                ]
                st.session_state.observations = observations

                alerts = generate_movement_alerts(observations)
                st.session_state.alerts = alerts

                reports = run_evaluation()
                st.session_state.eval_reports = reports

                st.session_state.processed = True
                st.success(
                    f"✅ **Processing Complete:** Processed **{len(decisions)}** image(s) → "
                    f"Created **{len(observations)}** trusted observation(s) → "
                    f"Generated **{len(alerts)}** movement alert(s)."
                )
            except Exception as e:
                st.error(f"Processing failed: {e}")
                return

    # ------ Show results if available ------
    decisions = st.session_state.get("decisions", [])
    if not decisions:
        st.info("💡 Click **Run Processing Pipeline** above to process images.")
        return

    # Decision distribution summary bar
    state_counts = {}
    for d in decisions:
        label = d.decision.value
        state_counts[label] = state_counts.get(label, 0) + 1

    st.markdown("---")
    st.subheader("Decision Breakdown")
    num_cols = max(1, min(len(state_counts), 4))
    cols = st.columns(num_cols)
    for i, (label, count) in enumerate(state_counts.items()):
        emoji = "✅" if label == "trusted_match" else ("🟡" if label == "ambiguous_review" else "🟠")
        cols[i % num_cols].metric(f"{emoji} {label}", count)

    st.markdown("---")
    st.subheader("Processed Image Decision Ledger")

    rows = []
    for d in decisions:
        rows.append(
            {
                "Image ID": d.image_id,
                "Decision State": d.decision.value,
                "Assigned Identity": d.identity_id or "—",
                "Evidence Confidence": f"{d.confidence:.4f}",
                "Reason Codes": ", ".join(rc.value for rc in d.reason_codes) or "—",
                "Feeds Trusted History": "✅ Yes" if d.update_history else "❌ No (Withheld)",
                "Data Mode": d.evidence_summary.get("data_mode", "demo"),
            }
        )

    st.dataframe(rows, use_container_width=True)
