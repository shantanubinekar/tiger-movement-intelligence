"""
app.py — Streamlit entrypoint for the Pench Tiger Reserve
Evidence-Gated Movement Intelligence Portal.

Government-portal-grade UI with clean formal styling.
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
# Comprehensive Government Portal CSS
# ---------------------------------------------------------------------------
PORTAL_CSS = """
<style>
/* ── Base Reset ────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Serif:wght@700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Top Institutional Banner ──────────────────────────────── */
.portal-banner {
    background: linear-gradient(135deg, #0c1f36 0%, #162d4a 60%, #1e3a5f 100%);
    padding: 20px 28px 18px;
    border-radius: 0 0 10px 10px;
    margin: -1rem -1rem 24px -1rem;
    border-bottom: 4px solid #d97706;
    position: relative;
    overflow: hidden;
}
.portal-banner::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 100%;
    background: linear-gradient(135deg, transparent 40%, rgba(217,119,6,0.15) 100%);
    pointer-events: none;
}
.portal-banner h1 {
    color: #ffffff !important;
    font-family: 'Noto Serif', Georgia, serif;
    font-size: 1.55rem;
    margin: 0 0 4px 0;
    font-weight: 700;
    letter-spacing: 0.3px;
    line-height: 1.3;
}
.portal-banner .subtitle {
    color: #cbd5e1;
    font-size: 0.82rem;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.2px;
}
.portal-banner .accent {
    color: #fbbf24;
    font-weight: 600;
}

/* ── Section Cards ─────────────────────────────────────────── */
.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.section-card-accent {
    border-left: 4px solid #d97706;
}

/* ── Metric Cards ──────────────────────────────────────────── */
.stMetric {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.stMetric:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.stMetric label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stMetric [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}
.stMetric [data-testid="stMetricDelta"] {
    opacity: 1 !important;
    font-size: 0.75rem !important;
}

/* ── Expanders ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #1e293b;
}

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background-color: #b45309;
    border: none;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover {
    background-color: #92400e;
}

/* ── Dataframes ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .stRadio label {
    font-weight: 500;
    color: #334155;
}
.sidebar-brand {
    text-align: center;
    padding: 8px 0 16px 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 16px;
}
.sidebar-brand .icon {
    font-size: 2.2rem;
    line-height: 1;
}
.sidebar-brand .title {
    font-family: 'Noto Serif', Georgia, serif;
    font-size: 1rem;
    font-weight: 700;
    color: #0f2942;
    margin: 6px 0 2px 0;
    line-height: 1.2;
}
.sidebar-brand .sub {
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.3px;
}
.sidebar-notice {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.75rem;
    color: #78350f;
    line-height: 1.5;
    margin-top: 12px;
}

/* ── Portal Footer ─────────────────────────────────────────── */
.portal-footer {
    margin-top: 48px;
    padding: 16px 24px;
    background: #f8fafc;
    border-top: 3px solid #d97706;
    border-radius: 6px;
    text-align: center;
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.6;
}
.portal-footer strong { color: #334155; }
.portal-footer .formula { color: #94a3b8; font-style: italic; }

/* ── Status Badges ─────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-trusted { background: #dcfce7; color: #166534; }
.badge-ambiguous { background: #fef3c7; color: #92400e; }
.badge-unknown { background: #ffedd5; color: #9a3412; }
.badge-rejected { background: #fee2e2; color: #991b1b; }

/* ── Info/Warning/Success Box Refinements ──────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px;
    font-size: 0.88rem;
}

/* ── Image captions ────────────────────────────────────────── */
[data-testid="stImage"] + div {
    font-size: 0.8rem;
    color: #64748b;
}
</style>
"""

st.markdown(PORTAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Institutional Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="portal-banner">
        <h1>🐯 Pench Tiger Reserve — Movement Intelligence Portal</h1>
        <p class="subtitle">
            <span class="accent">Evidence-Gated Identity Matching</span> ·
            Classical Stripe-Pattern Keypoint Analysis (SIFT/ORB) · Spatial-Temporal Gating ·
            Ecological Deviation Detection
        </p>
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
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="icon">🐯</div>
        <div class="title">Pench Tiger Reserve</div>
        <div class="sub">MP / Maharashtra · Movement Intelligence Portal</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Data source selector
data_source = st.sidebar.selectbox(
    "📂 Active Data Source",
    [
        "Synthetic Demo Scenarios (Pench Layout)",
        "Real Tiger Images (ATRW Benchmark)",
    ],
    index=0 if st.session_state.data_source == "Synthetic Demo Scenarios (Pench Layout)" else 1,
    help="Toggle between synthetic camera-trap scenarios or genuine ATRW benchmark tiger images.",
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

st.sidebar.markdown("---")

PAGES = {
    "📊 Dashboard": "overview",
    "⚙️ Processing": "processing",
    "🔍 Review Queue": "review",
    "🗺️ Movement & Catalogue": "movement",
    "🚨 Alerts": "alerts",
    "📈 Evaluation": "evaluation",
}

selected = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0)

st.sidebar.markdown(
    """
    <div class="sidebar-notice">
        <strong>⚖️ Safety Invariant</strong><br>
        Only <code>trusted_match</code> decisions with sufficient multi-factor
        evidence update longitudinal history. All other states are withheld.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data Mode Banner
# ---------------------------------------------------------------------------
if st.session_state.data_source == "Real Tiger Images (ATRW Benchmark)":
    st.info(
        "🐅 **Real Dataset Active** — Processing ATRW (Amur Tiger Re-ID Benchmark) images. "
        "Validated on genuine tiger photographs with illustrative Pench station grid — not field observations.",
        icon="🐅",
    )
elif not st.session_state.processed:
    st.warning(
        "⚠️ **Demo Mode** — Synthetic prototype data. "
        "Results must never be presented as field observations.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Page Routing
# ---------------------------------------------------------------------------
if selected == "📊 Dashboard":
    from ui.overview import render
    render()
elif selected == "⚙️ Processing":
    from ui.processing import render
    render()
elif selected == "🔍 Review Queue":
    from ui.review import render
    render()
elif selected == "🗺️ Movement & Catalogue":
    from ui.movement import render
    render()
elif selected == "🚨 Alerts":
    from ui.alerts import render
    render()
elif selected == "📈 Evaluation":
    from ui.evaluation import render
    render()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="portal-footer">
        <strong>Prototype — Evidence-Gated Identity & Movement Intelligence</strong><br>
        Developed for Smart India Hackathon · Not an official government system ·
        <span class="formula">E = W<sub>V</sub>·V + W<sub>Q</sub>·Q + W<sub>S</sub>·S + W<sub>T</sub>·T + W<sub>H</sub>·H</span>
    </div>
    """,
    unsafe_allow_html=True,
)
