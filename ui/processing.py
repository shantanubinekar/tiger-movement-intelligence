"""
ui/processing.py — Processing page: trigger image directory processing.

Lets the user pick a folder (defaults to data/demo), calls
process_image_directory(path), then routes results through
create_observation and generate_movement_alerts so all downstream
pages have data to render.
"""

import streamlit as st

from src.pipeline import (
    create_observation,
    generate_movement_alerts,
    process_image_directory,
    run_evaluation,
)
from src.schemas import IdentityDecisionState


def render():
    st.header("🔄 Image Processing")

    # ------ Folder selection ------
    folder_path = st.text_input(
        "Image folder path",
        value="data/demo",
        help="Path to a directory of camera-trap images (or the bundled demo folder).",
    )

    if st.button("▶ Process Image Directory", type="primary"):
        with st.spinner("Running ingestion → triage → perception → candidates → gating…"):
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
                    f"✅ Processed **{len(decisions)}** images → "
                    f"**{len(observations)}** trusted observations → "
                    f"**{len(alerts)}** alerts generated."
                )
            except Exception as e:
                st.error(f"Processing failed: {e}")
                return

    # ------ Show results if available ------
    decisions = st.session_state.get("decisions", [])
    if not decisions:
        st.info("Click **Process Image Directory** above to start.")
        return

    st.subheader("Per-Image Results")

    # Build a table
    rows = []
    for d in decisions:
        rows.append(
            {
                "Image ID": d.image_id,
                "Decision": d.decision.value,
                "Identity": d.identity_id or "—",
                "Confidence": f"{d.confidence:.3f}",
                "Reason Codes": ", ".join(rc.value for rc in d.reason_codes),
                "Update History": "✅" if d.update_history else "❌",
                "Data Mode": d.evidence_summary.get("data_mode", "unknown"),
            }
        )

    st.dataframe(rows, use_container_width=True)

    # Summary bar
    state_counts = {}
    for d in decisions:
        label = d.decision.value
        state_counts[label] = state_counts.get(label, 0) + 1

    st.subheader("Decision Distribution")
    cols = st.columns(len(state_counts))
    for col, (label, count) in zip(cols, state_counts.items()):
        emoji = "✅" if label == "trusted_match" else "⚠️"
        col.metric(f"{emoji} {label}", count)
