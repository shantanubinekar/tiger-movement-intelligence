"""
app.py — Streamlit entrypoint for the Evidence-Aware Tiger Camera-Trap
Movement Intelligence & Ecological Security Portal (Pench Tiger Reserve).

Government-portal design theme for Smart India Hackathon prototype.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pench Tiger Reserve — Movement Intelligence Portal",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Government Portal Custom Theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Government portal color variables and container styling */
    :root {
        --gov-navy: #0f2942;
        --gov-saffron: #d97706;
        --gov-saffron-light: #fef3c7;
        --gov-green: #047857;
        --gov-border: #cbd5e1;
    }

    .gov-header-bar {
        background: linear-gradient(90deg, #0f2942 0%, #1e3a5f 70%, #d97706 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 6px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .gov-header-bar h1 {
        color: white !important;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .gov-header-bar p {
        color: #f1f5f9;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }

    .gov-footer {
        margin-top: 50px;
        padding: 18px 24px;
        background-color: #f8fafc;
        border-top: 3px solid #d97706;
        border-radius: 4px;
        text-align: center;
        color: #475569;
        font-size: 0.82rem;
    }

    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .stMetric label { color: #475569 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0f2942 !important; }
    .stMetric [data-testid="stMetricDelta"] { opacity: 1 !important; }

    .sidebar-badge {
        background-color: #f1f5f9;
        border-left: 3px solid #0f2942;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #334155;
        margin-top: 10px;
    }
    </style>
    <div class="gov-header-bar">
        <h1>🐯 Pench Tiger Reserve — Movement Intelligence & Ecological Security Portal</h1>
        <p>Evidence-gated identity matching using classical stripe-pattern keypoint matching & spatial-temporal gating</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults — shared across pages
# ---------------------------------------------------------------------------
if "decisions" not in st.session_state:
    st.session_state.decisions = []
if "observations" not in st.session_state:
    st.session_state.observations = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "eval_reports" not in st.session_state:
    st.session_state.eval_reports = []
if "human_reviews" not in st.session_state:
    st.session_state.human_reviews = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "data_source" not in st.session_state:
    st.session_state.data_source = "Synthetic Demo Scenarios (Pench Layout)"

# ---------------------------------------------------------------------------
# Sidebar navigation & Data Source Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🏛️ Ministry & Reserve Portal")
st.sidebar.caption("Pench Tiger Reserve (MP / Maharashtra)")

# Data source selector
data_source = st.sidebar.selectbox(
    "Active Data Source:",
    [
        "Synthetic Demo Scenarios (Pench Layout)",
        "Real Tiger Images (ATRW Benchmark)",
    ],
    index=0 if st.session_state.data_source == "Synthetic Demo Scenarios (Pench Layout)" else 1,
    help="Select between synthetic camera-trap scenarios or genuine ATRW benchmark tiger photos.",
)

if data_source != st.session_state.data_source:
    st.session_state.data_source = data_source
    # Reset processed state on switch
    st.session_state.processed = False
    st.session_state.decisions = []
    st.session_state.observations = []
    st.session_state.alerts = []
    st.session_state.eval_reports = []
    st.session_state._auto_process_pending = True

# Auto-process on data source switch (runs once after the rerun triggered by selectbox change)
if st.session_state.get("_auto_process_pending", False) and not st.session_state.processed:
    from src.pipeline import (
        create_observation,
        generate_movement_alerts,
        process_image_directory,
        run_evaluation,
    )

    folder = (
        "data/real_tigers/query"
        if st.session_state.data_source == "Real Tiger Images (ATRW Benchmark)"
        else "data/demo"
    )
    source_label = (
        "ATRW benchmark tiger images"
        if st.session_state.data_source == "Real Tiger Images (ATRW Benchmark)"
        else "synthetic demo scenarios"
    )

    with st.spinner(f"Loading {source_label} from `{folder}`…"):
        try:
            decisions = process_image_directory(folder)
            st.session_state.decisions = decisions
            observations = [
                obs for obs in (create_observation(d) for d in decisions) if obs is not None
            ]
            st.session_state.observations = observations
            st.session_state.alerts = generate_movement_alerts(observations)
            st.session_state.eval_reports = run_evaluation()
            st.session_state.processed = True
        except Exception as e:
            st.error(f"Auto-processing failed: {e}")
    st.session_state._auto_process_pending = False

st.sidebar.markdown("---")

PAGES = {
    "Overview": "overview",
    "Processing": "processing",
    "Review Queue": "review",
    "Movement / Catalogue": "movement",
    "Alerts": "alerts",
    "Evaluation": "evaluation",
}

selected = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-badge">
        <b>System Integrity Notice</b><br>
        Identity matching via classical stripe keypoints (SIFT/ORB) with strict evidence gating.
        Safety invariant: only <code>trusted_match</code> updates longitudinal history.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dynamic Banner based on Data Mode
# ---------------------------------------------------------------------------
if st.session_state.data_source == "Real Tiger Images (ATRW Benchmark)":
    st.info(
        "🐅 **REAL DATASET BENCHMARK ACTIVE** — Running on ATRW (Amur Tiger Re-ID Benchmark) image data with illustrative Pench reserve station grid. "
        "**Note:** Validated on genuine tiger photographs — not Pench-specific field observations.",
        icon="🐅",
    )
else:
    st.warning(
        "⚠️ **DEMO MODE** — All results shown are synthetic/prototype camera-trap data. "
        "They must **never** be presented as unverified field observations.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Route to the selected page
# ---------------------------------------------------------------------------
if selected == "Overview":
    from ui.overview import render
    render()
elif selected == "Processing":
    from ui.processing import render
    render()
elif selected == "Review Queue":
    from ui.review import render
    render()
elif selected == "Movement / Catalogue":
    from ui.movement import render
    render()
elif selected == "Alerts":
    from ui.alerts import render
    render()
elif selected == "Evaluation":
    from ui.evaluation import render
    render()

# ---------------------------------------------------------------------------
# Universal Government Portal Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="gov-footer">
        🏛️ <b>Prototype — Evidence-Gated Identity & Movement Intelligence System</b><br>
        Developed for Smart India Hackathon | Not an official government release |
        Evidence formula: <i>E = W<sub>V</sub>·V<sub>eff</sub> + W<sub>Q</sub>·Q + W<sub>S</sub>·S + W<sub>T</sub>·T + W<sub>H</sub>·H</i> |
        Scientific proof of concept
    </div>
    """,
    unsafe_allow_html=True,
)
