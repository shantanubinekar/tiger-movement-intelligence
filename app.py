"""
app.py — Streamlit entrypoint for the Pench Tiger Reserve
Evidence-Gated Movement Intelligence Portal.

Matches the official Government Intelligence Portal UI specification.
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
# Comprehensive UI Theme CSS (Exact match to reference design)
# ---------------------------------------------------------------------------
PORTAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Serif:wght@600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #f8fafc;
    color: #0f172a;
}

/* ── Top Institutional Header Bar ──────────────────────────── */
.top-gov-header {
    background: #0f2942;
    padding: 24px 36px 20px 36px;
    margin: -1rem -1rem 24px -1rem;
    box-shadow: 0 4px 12px rgba(15, 41, 66, 0.15);
}
.top-gov-header h1 {
    color: #ffffff !important;
    font-family: 'Noto Serif', Georgia, serif;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: 0.2px;
}
.top-gov-header .breadcrumbs {
    color: #94a3b8;
    font-size: 0.84rem;
    font-weight: 400;
    margin: 0;
}
.top-gov-header .breadcrumbs span {
    color: #cbd5e1;
}

/* ── Sidebar Styling ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
.sidebar-header {
    text-align: center;
    padding: 10px 0 16px 0;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 16px;
}
.sidebar-emblem {
    width: 60px;
    height: auto;
    margin-bottom: 8px;
}
.sidebar-title {
    font-family: 'Noto Serif', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f2942;
    margin: 0;
}

/* Sidebar Radio Navigation */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px;
    border-radius: 6px;
    font-weight: 500;
    color: #334155;
    transition: all 0.15s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background-color: #f8fafc;
    color: #0f172a;
}
[data-testid="stSidebar"] [data-checked="true"] {
    background-color: #fff7ed !important;
    border-left: 4px solid #ea580c !important;
    color: #9a3412 !important;
    font-weight: 600 !important;
}

/* ── Metric Stat Cards ─────────────────────────────────────── */
.stat-card-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    position: relative;
    overflow: hidden;
}
.stat-card.accent-orange {
    border-left: 4px solid #ea580c;
}
.stat-card.accent-green {
    border-left: 4px solid #059669;
}
.stat-card .val {
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 4px;
}
.stat-card .lbl {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
}

/* ── White Container Cards ─────────────────────────────────── */
.portal-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 22px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}
.portal-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 16px 0;
}

/* ── Government Portal Data Table ──────────────────────────── */
.gov-table-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}
.gov-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    text-align: left;
}
.gov-table th {
    background-color: #0f2942;
    color: #ffffff;
    padding: 12px 18px;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.gov-table td {
    padding: 12px 18px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
}
.gov-table tr:last-child td {
    border-bottom: none;
}
.gov-table tr:hover {
    background-color: #f8fafc;
}

/* Status Badges */
.badge-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 14px;
    font-size: 0.76rem;
    font-weight: 600;
    text-align: center;
}
.badge-active {
    background-color: #047857;
    color: #ffffff;
}
.badge-pending {
    background-color: #fed7aa;
    color: #9a3412;
}
.badge-suppressed {
    background-color: #cbd5e1;
    color: #334155;
}

/* ── Photo Comparison Card ─────────────────────────────────── */
.photo-comp-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    padding: 12px 0;
}
.photo-comp-circle {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid #e2e8f0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.photo-comp-circle img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.photo-comp-rect {
    width: 130px;
    height: 100px;
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid #e2e8f0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.photo-comp-rect img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.photo-connector {
    display: flex;
    align-items: center;
    color: #ea580c;
    font-weight: 700;
}

/* ── Universal Footer ──────────────────────────────────────── */
.portal-footer {
    margin-top: 40px;
    padding: 16px 20px;
    background: #ffffff;
    border-top: 2px solid #e2e8f0;
    border-radius: 6px;
    text-align: center;
    font-size: 0.8rem;
    color: #64748b;
}
</style>
"""

st.markdown(PORTAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top Institutional Header Bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-gov-header">
        <h1>Pench Tiger Reserve — Movement Intelligence Portal</h1>
        <div class="breadcrumbs">
            Home &nbsp;›&nbsp; Dashboard &nbsp;›&nbsp; <span>Movement Intelligence Portal</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "decisions": [],
    "observations": [],
    "alerts": [],
    "eval_reports": [],
    "human_reviews": {},
    "processed": False,
    "data_source": "Synthetic Demo Scenarios (Pench Layout)",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar Setup (Emblem + Title + Navigation)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <div style="font-size: 2.2rem; margin-bottom: 4px;">🇮🇳</div>
        <div class="sidebar-title">Pench Tiger Reserve</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">NTCA · Forest Department</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Data source selector
data_source = st.sidebar.selectbox(
    "Data Source:",
    [
        "Synthetic Demo Scenarios (Pench Layout)",
        "Real Tiger Images (ATRW Benchmark)",
    ],
    index=0 if st.session_state.data_source == "Synthetic Demo Scenarios (Pench Layout)" else 1,
    help="Select between synthetic camera-trap scenarios or genuine ATRW benchmark tiger photos.",
)

if data_source != st.session_state.data_source:
    st.session_state.data_source = data_source
    st.session_state.processed = False
    st.session_state.decisions = []
    st.session_state.observations = []
    st.session_state.alerts = []
    st.session_state.eval_reports = []
    st.session_state._auto_process_pending = True

# Auto-process on data source switch
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

    with st.spinner(f"Loading {source_label}…"):
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

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Navigation Menu matching the image icons
PAGES = {
    "🏠 Overview": "overview",
    "📄 Processing": "processing",
    "📑 Review Queue": "review",
    "🗺️ Movement Map": "movement",
    "🔔 Alerts": "alerts",
    "📋 Evaluation": "evaluation",
}

selected = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0)

# Banner notification for active data source
if st.session_state.data_source == "Real Tiger Images (ATRW Benchmark)":
    st.info(
        "🐅 **Real ATRW Benchmark Active** — Validated on genuine tiger photographs with illustrative Pench reserve station grid.",
        icon="🐅",
    )
elif not st.session_state.processed:
    st.warning(
        "⚠️ **Demo Mode Active** — Prototype scenario data for system demonstration.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Page Routing
# ---------------------------------------------------------------------------
if selected == "🏠 Overview":
    from ui.overview import render
    render()
elif selected == "📄 Processing":
    from ui.processing import render
    render()
elif selected == "📑 Review Queue":
    from ui.review import render
    render()
elif selected == "🗺️ Movement Map":
    from ui.movement import render
    render()
elif selected == "🔔 Alerts":
    from ui.alerts import render
    render()
elif selected == "📋 Evaluation":
    from ui.evaluation import render
    render()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="portal-footer">
        🏛️ <b>Pench Tiger Reserve Movement Intelligence System</b> · Smart India Hackathon Prototype ·
        Evidence Formula: <i>E = W<sub>V</sub>·V + W<sub>Q</sub>·Q + W<sub>S</sub>·S + W<sub>T</sub>·T + W<sub>H</sub>·H</i>
    </div>
    """,
    unsafe_allow_html=True,
)
