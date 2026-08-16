"""
app.py — Streamlit entrypoint for the Evidence-Aware Tiger Camera-Trap
Movement Intelligence System (SIH Phase 1 prototype).

Developer 1 owns this file. It provides page navigation and a shared
session-state container for pipeline results. Each page is implemented
in ui/<page>.py and calls ONLY src.pipeline functions — never Backend
A/B internals directly.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tiger Movement Intelligence",
    page_icon="🐯",
    layout="wide",
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
if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------------------------------------------------------------------
# DEMO MODE banner — always visible when demo data is in use
# ---------------------------------------------------------------------------
_is_demo = True  # Flip when real Backend A/B are wired in
if _is_demo or not st.session_state.processed:
    st.warning(
        "⚠️ **DEMO MODE** — All results shown are synthetic/demo data. "
        "They must **never** be presented as real Pench observations.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🐯 Tiger Movement Intelligence")
st.sidebar.caption("SIH Phase 1 Prototype")

PAGES = {
    "Overview": "overview",
    "Processing": "processing",
    "Review Queue": "review",
    "Movement / Catalogue": "movement",
    "Alerts": "alerts",
    "Evaluation": "evaluation",
}

selected = st.sidebar.radio("Navigate", list(PAGES.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Developer 1** — Frontend + Integration\n\n"
    "This UI consumes outputs from Backend A (Dev2) "
    "and Backend B (Dev3) via `src/pipeline.py`."
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
