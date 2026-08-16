"""
ui/movement.py — Movement & Individual Tiger Catalogue.

Migration trajectory maps, historical capture area polygons,
flagged deviation highlights, and catalogue reference photos.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import streamlit as st

from src.history import get_history
from src.schemas import AlertType


def render():
    observations = st.session_state.get("observations", [])
    alerts = st.session_state.get("alerts", [])

    st.markdown("##### Individual Tiger Catalogue & Spatial Intelligence")
    st.caption(
        "🐅 ATRW benchmark dataset — illustrative Pench reserve station grid for spatial trajectory demonstration."
    )

    if not observations:
        st.info("No trusted observations yet. Process images or accept pending reviews to populate this view.")
        return

    # ── Build per-tiger data ───────────────────────────────────
    tiger_data: dict[str, dict] = {}
    history = get_history()

    for obs in observations:
        tid = obs.identity_id
        if tid not in tiger_data:
            tiger_data[tid] = {
                "identity_id": tid,
                "capture_count": 0,
                "first_seen": None,
                "last_seen": None,
                "trusted_stations": set(),
                "station_details": {},
                "observations": [],
            }
        entry = tiger_data[tid]
        entry["capture_count"] += 1
        entry["observations"].append(obs)

        if obs.timestamp:
            if entry["first_seen"] is None or obs.timestamp < entry["first_seen"]:
                entry["first_seen"] = obs.timestamp
            if entry["last_seen"] is None or obs.timestamp > entry["last_seen"]:
                entry["last_seen"] = obs.timestamp

        st_id = obs.station_id or "UNKNOWN"
        if obs.station_id:
            entry["trusted_stations"].add(obs.station_id)

        if obs.latitude is not None and obs.longitude is not None:
            if st_id not in entry["station_details"]:
                entry["station_details"][st_id] = {
                    "station_id": st_id, "latitude": obs.latitude,
                    "longitude": obs.longitude, "count": 0,
                }
            entry["station_details"][st_id]["count"] += 1

    # ── Catalogue Table ────────────────────────────────────────
    rows = []
    for tid, data in tiger_data.items():
        rows.append({
            "Identity": data["identity_id"],
            "Captures": data["capture_count"],
            "First Seen": data["first_seen"].strftime("%Y-%m-%d %H:%M") if data["first_seen"] else "—",
            "Last Seen": data["last_seen"].strftime("%Y-%m-%d %H:%M") if data["last_seen"] else "—",
            "Stations": ", ".join(sorted(data["trusted_stations"])) or "—",
        })
    st.dataframe(rows, use_container_width=True)

    # ── Per-Tiger Detail ───────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Migration Trajectories & Capture Areas")
    st.caption(
        "Shaded amber boundary = **historical capture area** (convex hull) — not a validated home range."
    )

    for tid, data in tiger_data.items():
        with st.expander(
            f"🐅 {tid} — {data['capture_count']} trusted captures",
            expanded=True,
        ):
            # Metrics row
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Captures", data["capture_count"])
            mc2.metric("First Seen", data["first_seen"].strftime("%Y-%m-%d") if data["first_seen"] else "—")
            mc3.metric("Last Seen", data["last_seen"].strftime("%Y-%m-%d") if data["last_seen"] else "—")

            st.caption(f"Stations: `{', '.join(sorted(data['trusted_stations'])) or 'None'}`")

            # Catalogue reference photo
            from src.identity import get_default_catalogue
            cat_img = get_default_catalogue().get_image_path(tid)
            if cat_img and Path(cat_img).exists():
                photo_col, desc_col = st.columns([1, 3])
                with photo_col:
                    st.image(str(cat_img), caption=f"Catalogue: {tid}", use_container_width=True)
                with desc_col:
                    st.caption(
                        f"Registered reference photograph for **{tid}** — used as the template for "
                        f"classical stripe-pattern keypoint matching (SIFT/ORB)."
                    )

            # ── Plotly Migration Map ───────────────────────────
            if data["station_details"]:
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    sorted_obs = sorted(
                        [o for o in data["observations"] if o.latitude is not None and o.longitude is not None],
                        key=lambda x: x.timestamp or datetime.min,
                    )
                    path_lats = [o.latitude for o in sorted_obs]
                    path_lons = [o.longitude for o in sorted_obs]

                    # Historical capture area polygon
                    hull_points = history.compute_historical_capture_area(tid)
                    if hull_points and len(hull_points) >= 3:
                        hull_lats = [p[0] for p in hull_points] + [hull_points[0][0]]
                        hull_lons = [p[1] for p in hull_points] + [hull_points[0][1]]
                        fig.add_trace(go.Scattermapbox(
                            lat=hull_lats, lon=hull_lons, mode="lines",
                            fill="toself", fillcolor="rgba(245, 158, 11, 0.15)",
                            line=dict(color="#d97706", width=2),
                            name="Historical Capture Area", hoverinfo="name",
                        ))

                    # Movement trajectory
                    if len(path_lats) >= 2:
                        fig.add_trace(go.Scattermapbox(
                            lat=path_lats, lon=path_lons, mode="lines",
                            line=dict(color="#1e40af", width=3),
                            name="Movement Path", hoverinfo="name",
                        ))

                    # Flagged deviations
                    flagged_ids = set()
                    for alert in alerts:
                        if (alert.identity_id == tid and
                            alert.alert_type in (AlertType.OUTSIDE_HISTORICAL_AREA, AlertType.UNUSUAL_TRAVEL) and
                            alert.triggering_observation):
                            flagged_ids.add(alert.triggering_observation.observation_id)

                    flagged_obs = [o for o in sorted_obs if o.observation_id in flagged_ids]
                    if flagged_obs:
                        fig.add_trace(go.Scattermapbox(
                            lat=[o.latitude for o in flagged_obs],
                            lon=[o.longitude for o in flagged_obs],
                            mode="markers", marker=dict(size=16, color="#dc2626"),
                            name="⚠️ Flagged Deviation", hoverinfo="name",
                        ))

                    # Station markers
                    stations = list(data["station_details"].values())
                    st_lats = [s["latitude"] for s in stations]
                    st_lons = [s["longitude"] for s in stations]
                    st_counts = [s["count"] for s in stations]
                    st_names = [s["station_id"] for s in stations]

                    fig.add_trace(go.Scattermapbox(
                        lat=st_lats, lon=st_lons, mode="markers+text",
                        marker=dict(
                            size=[max(12, min(32, 10 + c * 4)) for c in st_counts],
                            color="#0f2942", opacity=0.85,
                        ),
                        text=st_names, textposition="top right",
                        textfont=dict(size=10, color="#1e293b"),
                        hovertext=[f"{n}: {c} captures" for n, c in zip(st_names, st_counts)],
                        hoverinfo="text", name="Stations",
                    ))

                    center_lat = sum(st_lats) / len(st_lats)
                    center_lon = sum(st_lons) / len(st_lons)

                    fig.update_layout(
                        mapbox=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=11.5),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=400, showlegend=True,
                        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.9)"),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Map render failed ({e}). Showing tabular summary.")
                    st.dataframe(data["station_details"], use_container_width=True)
            else:
                st.caption("No GPS coordinate data available.")
