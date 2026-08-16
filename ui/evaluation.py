"""
ui/evaluation.py — Evaluation page: baseline vs. evidence-gated comparison.

Side-by-side comparison of metrics: false confident-identity rate,
coverage, abstention rate, false alert rate, artefact-suppression rate.
Labelled "prototype scenario evaluation" — NEVER "field validation."
"""

import streamlit as st


def render():
    st.header("📈 Evaluation — Baseline vs. Evidence-Gated")
    st.caption(
        "**Prototype scenario evaluation** — These results compare the "
        "always-assign baseline against the evidence-gated system on the "
        "same set of scenarios. This is NOT field validation."
    )

    reports = st.session_state.get("eval_reports", [])

    if not reports:
        st.info(
            "No evaluation data yet. Process images on the **Processing** "
            "page first to generate a comparison."
        )
        return

    # ------ Find baseline and gated reports ------
    baseline = None
    gated = None
    for r in reports:
        if r.pipeline_name == "baseline":
            baseline = r
        elif r.pipeline_name == "evidence_gated":
            gated = r

    if not baseline or not gated:
        st.warning("Expected both a 'baseline' and 'evidence_gated' report.")
        return

    # ------ Side-by-side comparison ------
    st.subheader("Side-by-Side Comparison")

    metrics = [
        ("False Confident-Identity Rate", "false_confident_identity_rate"),
        ("Coverage", "coverage"),
        ("Abstention / Review Rate", "abstention_review_rate"),
        ("False Movement-Alert Rate", "false_movement_alert_rate"),
        ("Alert Precision", "alert_precision"),
        ("Artefact Suppression Rate", "artefact_suppression_rate"),
        ("Observations Withheld (%)", "observations_withheld_pct"),
    ]

    col_header = st.columns([2, 1, 1])
    col_header[0].markdown("**Metric**")
    col_header[1].markdown("**Baseline (Always-Assign)**")
    col_header[2].markdown("**Evidence-Gated (Proposed)**")

    st.markdown("---")

    for display_name, field_name in metrics:
        cols = st.columns([2, 1, 1])
        cols[0].markdown(f"**{display_name}**")

        b_val = getattr(baseline, field_name, None)
        g_val = getattr(gated, field_name, None)

        b_not_computable = field_name in baseline.not_computable
        g_not_computable = field_name in gated.not_computable

        if b_not_computable or b_val is None:
            cols[1].markdown("*Not computable on current data*")
        else:
            cols[1].metric(label="", value=f"{b_val:.4f}", label_visibility="collapsed")

        if g_not_computable or g_val is None:
            cols[2].markdown("*Not computable on current data*")
        else:
            cols[2].metric(label="", value=f"{g_val:.4f}", label_visibility="collapsed")

    # ------ Notes ------
    st.markdown("---")
    st.subheader("Report Notes")

    col_notes = st.columns(2)
    with col_notes[0]:
        st.markdown("**Baseline**")
        st.markdown(baseline.notes or "No notes.")
        if baseline.not_computable:
            st.caption(
                f"Not computable: {', '.join(baseline.not_computable)}"
            )

    with col_notes[1]:
        st.markdown("**Evidence-Gated**")
        st.markdown(gated.notes or "No notes.")
        if gated.not_computable:
            st.caption(
                f"Not computable: {', '.join(gated.not_computable)}"
            )

    # ------ Interpretation ------
    st.markdown("---")
    st.info(
        "**Interpretation Guidance:** The central experimental question is "
        "whether the identity-gating layer reduces downstream movement-alert "
        "errors compared with the always-assign baseline. Metrics marked "
        "'not computable' require labelled scenario data that is not yet "
        "available. All numbers shown are **prototype scenario evaluation** "
        "results — never claim these as field validation or Pench performance."
    )
