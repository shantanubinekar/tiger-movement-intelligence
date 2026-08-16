"""
ui/movement.py — Movement / Catalogue page.

Per-tiger: capture count, first/last seen, trusted stations, and an
interactive Plotly map of the "historical capture area" (NEVER "home range").
If coordinates are available, show station markers sized by capture effort
and the shaded polygon outline of the historical capture area.
"""

from __future__ import annotations

import streamlit as st

from src.history import get_history
from src.schemas import IdentityDecisionState


def render():
    st.header("🗺️ Movement / Catalogue")

    observations = st.session_state.get("observations", [])
    decisions = st.session_state.get("decisions", [])

    if not observations:
        st.info(
            "No trusted observations yet. Process images on the "
            "**Processing** page first."
        )
        return

    # ------ Build per-tiger summary from observations ------
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
                "station_details": {},  # station_id -> {lat, lon, count}
                "latitudes": [],
                "longitudes": [],
            }
        entry = tiger_data[tid]
        entry["capture_count"] += 1

        if obs.timestamp:
            if entry["first_seen"] is None or obs.timestamp < entry["first_seen"]:
                entry["first_seen"] = obs.timestamp
            if entry["last_seen"] is None or obs.timestamp > entry["last_seen"]:
                entry["last_seen"] = obs.timestamp

        st_id = obs.station_id or "UNKNOWN_STATION"
        if obs.station_id:
            entry["trusted_stations"].add(obs.station_id)

        if obs.latitude is not None and obs.longitude is not None:
            entry["latitudes"].append(obs.latitude)
            entry["longitudes"].append(obs.longitude)

            if st_id not in entry["station_details"]:
                entry["station_details"][st_id] = {
                    "station_id": st_id,
                    "latitude": obs.latitude,
                    "longitude": obs.longitude,
                    "count": 0,
                }
            entry["station_details"][st_id]["count"] += 1

    # ------ Catalogue table ------
    st.subheader("Individual Tiger Catalogue")

    rows = []
    for tid, data in tiger_data.items():
        rows.append(
            {
                "Identity ID": data["identity_id"],
                "Capture Count": data["capture_count"],
                "First Seen": str(data["first_seen"]) if data["first_seen"] else "—",
                "Last Seen": str(data["last_seen"]) if data["last_seen"] else "—",
                "Trusted Stations": ", ".join(sorted(data["trusted_stations"])) or "—",
                "Observation Status": "✅ Trusted",
            }
        )
    st.dataframe(rows, use_container_width=True)

    # ------ Per-tiger detail with Plotly Mapbox ------
    st.subheader("Spatial Intelligence & Capture Area Analysis")

    for tid, data in tiger_data.items():
        with st.expander(f"🐯 **{tid}** — {data['capture_count']} trusted capture(s)", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Captures", data["capture_count"])
            col2.metric(
                "First Seen",
                data["first_seen"].strftime("%Y-%m-%d") if data["first_seen"] else "—",
            )
            col3.metric(
                "Last Seen",
                data["last_seen"].strftime("%Y-%m-%d") if data["last_seen"] else "—",
            )

            st.markdown(
                f"**Trusted Stations:** {', '.join(sorted(data['trusted_stations'])) or 'None'}"
            )

            # Historical capture area (NOT "home range")
            if data["station_details"]:
                st.markdown("#### Historical Capture Area Map")
                st.caption(
                    "Interactive spatial map showing trusted camera stations sized by capture frequency. "
                    "Shaded boundary represents the convex hull **historical capture area** "
                    "(requires ≥2 observations per station) — NOT a validated home range."
                )

                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    station_list = list(data["station_details"].values())
                    lats = [s["latitude"] for s in station_list]
                    lons = [s["longitude"] for s in station_list]
                    counts = [s["count"] for s in station_list]
                    st_names = [s["station_id"] for s in station_list]

                    # 1. Historical Capture Area (Convex Hull polygon)
                    hull_points = history.compute_historical_capture_area(tid)
                    if hull_points and len(hull_points) >= 3:
                        # Close the polygon for plotting
                        hull_lats = [p[0] for p in hull_points] + [hull_points[0][0]]
                        hull_lons = [p[1] for p in hull_points] + [hull_points[0][1]]

                        fig.add_trace(
                            go.Scattermapbox(
                                lat=hull_lats,
                                lon=hull_lons,
                                mode="lines",
                                fill="toself",
                                fillcolor="rgba(245, 158, 11, 0.20)",
                                line=dict(color="#D97706", width=2.5),
                                name="Historical Capture Area",
                                hoverinfo="name",
                            )
                        )

                    # 2. Camera Stations (sized by capture count)
                    marker_sizes = [max(12, min(36, 12 + c * 5)) for c in counts]
                    hover_texts = [
                        f"<b>Station:</b> {name}<br>"
                        f"<b>Trusted Captures:</b> {c}<br>"
                        f"<b>Lat:</b> {lat:.4f}, <b>Lon:</b> {lon:.4f}"
                        for name, c, lat, lon in zip(st_names, counts, lats, lons)
                    ]

                    fig.add_trace(
                        go.Scattermapbox(
                            lat=lats,
                            lon=lons,
                            mode="markers+text",
                            marker=dict(
                                size=marker_sizes,
                                color="#EA580C",
                                opacity=0.9,
                                symbol="circle",
                            ),
                            text=st_names,
                            textposition="top right",
                            textfont=dict(size=12, color="#1F2937"),
                            hovertext=hover_texts,
                            hoverinfo="text",
                            name="Trusted Stations",
                        )
                    )

                    center_lat = sum(lats) / len(lats)
                    center_lon = sum(lons) / len(lons)

                    fig.update_layout(
                        mapbox=dict(
                            style="carto-positron",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=11,
                        ),
                        margin=dict(l=0, r=0, t=20, b=0),
                        height=380,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.98,
                            xanchor="left",
                            x=0.02,
                            bgcolor="rgba(255, 255, 255, 0.85)",
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    # Graceful fallback to table
                    st.warning(f"Could not render interactive map ({e}). Showing table summary.")
                    st.dataframe(data["station_details"], use_container_width=True)

            else:
                st.caption("No GPS coordinate data available for spatial mapping.")

    # ------ Trusted history status ------
    st.subheader("Trusted History Status")
    st.markdown(
        f"**{len(observations)}** observation(s) stored in trusted history. "
        f"Only `trusted_match` decisions with `update_history=True` "
        f"are admitted."
    )

    if decisions:
        non_trusted = sum(
            1 for d in decisions if not d.update_history
        )
        st.caption(
            f"ℹ️ {non_trusted} decision(s) were correctly withheld "
            f"from trusted history by identity evidence gating."
        )
