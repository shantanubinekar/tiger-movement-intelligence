"""
app.py — Streamlit entrypoint for the Pench Tiger Reserve
Evidence-Gated Movement Intelligence Portal.

Operates directly on the ATRW Real Tiger Benchmark dataset.
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

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #ffffff;
    color: #0f172a;
}

/* ── Top Institutional Banner ──────────────────────────────── */
.portal-banner {
    background: linear-gradient(135deg, #0c1f36 0%, #162d4a 60%, #1e3a5f 100%);
    padding: 20px 28px 18px;
    border-radius: 0 0 8px 8px;
    margin: -1rem -1rem 22px -1rem;
    border-bottom: 4px solid #d97706;
    position: relative;
    overflow: hidden;
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
.sidebar-brand {
    text-align: center;
    padding: 8px 0 14px 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 14px;
}
.sidebar-brand .icon {
    font-size: 2rem;
    line-height: 1;
}
.sidebar-brand .title {
    font-family: 'Noto Serif', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f2942;
    margin: 4px 0 2px 0;
}
.sidebar-brand .sub {
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.3px;
}

/* ── Labeled Nav Menu ──────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex;
    align-items: center;
    padding: 10px 14px;
    border-radius: 6px;
    background-color: transparent;
    border-left: 3px solid transparent;
    color: #334155;
    font-size: 0.92rem;
    font-weight: 500;
    transition: all 0.15s ease;
    cursor: pointer;
    width: 100%;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #f1f5f9;
    color: #0f2942;
}
/* Hide default radio circle */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
/* Active/Selected item */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    background-color: #fef3c7 !important;
    border-left: 3px solid #0f2942 !important;
    color: #0f2942 !important;
    font-weight: 600 !important;
}

/* Badges */
.sidebar-badge {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #0f2942;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 0.78rem;
    color: #334155;
    line-height: 1.45;
}
.badge-datasource {
    background-color: #fef3c7;
    border: 1px solid #fde68a;
    border-left: 3px solid #d97706;
    color: #92400e;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.78rem;
    margin-bottom: 14px;
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
# Session state defaults & Auto-Ingestion
# ---------------------------------------------------------------------------
for key, default in {
    "decisions": [],
    "observations": [],
    "alerts": [],
    "eval_reports": [],
    "human_reviews": {},
    "processed": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Auto-ingest real ATRW data directly on startup
if not st.session_state.processed:
    from src.pipeline import (
        create_observation,
        generate_movement_alerts,
        process_image_directory,
        run_evaluation,
    )
    try:
        decisions = process_image_directory("data/real_tigers/query")
        st.session_state.decisions = decisions
        observations = [
            obs for obs in (create_observation(d) for d in decisions) if obs is not None
        ]
        st.session_state.observations = observations
        st.session_state.alerts = generate_movement_alerts(observations)
        st.session_state.eval_reports = run_evaluation()
        st.session_state.processed = True
    except Exception as e:
        st.error(f"Initial ingestion failed: {e}")

# ---------------------------------------------------------------------------
# Sidebar (Brand + Static Data Source Badge + Labeled Navigation)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="icon">🐯</div>
        <div class="title">Pench Tiger Reserve</div>
        <div class="sub">MP / Maharashtra · Movement Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Static Data Source Pill/Badge (Fix 1 & Fix 3)
st.sidebar.markdown(
    """
    <div class="badge-datasource">
        <span style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700; display: block; color: #78350f; margin-bottom: 2px;">Active Data Source</span>
        <b>ATRW Real Tiger Benchmark</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# Labeled Navigation Menu (Fix 2)
PAGES = {
    "📊  Overview": "overview",
    "⚙️  Processing": "processing",
    "🔍  Review Queue": "review",
    "🗺️  Movement Map": "movement",
    "🚨  Alerts": "alerts",
    "📈  Evaluation": "evaluation",
}

selected = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-badge">
        <b>⚖️ Safety Invariant Active</b><br>
        Only <code>trusted_match</code> decisions with sufficient multi-factor
        evidence update longitudinal history.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Notification Banner
# ---------------------------------------------------------------------------
st.info(
    "🐅 **ATRW Real Tiger Benchmark Active** — Validated on genuine Amur tiger stripe photographs with illustrative Pench station layout.",
    icon="🐅",
)

# ---------------------------------------------------------------------------
# Page Routing
# ---------------------------------------------------------------------------
if selected == "📊  Overview":
    from ui.overview import render
    render()
elif selected == "⚙️  Processing":
    from ui.processing import render
    render()
elif selected == "🔍  Review Queue":
    from ui.review import render
    render()
elif selected == "🗺️  Movement Map":
    from ui.movement import render
    render()
elif selected == "🚨  Alerts":
    from ui.alerts import render
    render()
elif selected == "📈  Evaluation":
    from ui.evaluation import render
    render()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="portal-footer">
        <strong>Prototype — Evidence-Gated Identity & Movement Intelligence</strong><br>
        Developed for Smart India Hackathon · Not an official government release ·
        <span class="formula">E = W<sub>V</sub>·V + W<sub>Q</sub>·Q + W<sub>S</sub>·S + W<sub>T</sub>·T + W<sub>H</sub>·H</span>
    </div>
    """,
    unsafe_allow_html=True,
)
