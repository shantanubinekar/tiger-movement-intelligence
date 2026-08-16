"""
ui/movement.py — Movement & Individual Tiger Catalogue page.

Visualizes:
1. Individual tiger catalogue with capture metrics and observation status.
2. Interactive migration maps (Plotly Mapbox) showing connected movement trajectories
   between capture stations in chronological order.
3. Shaded "historical capture area" polygon (convex hull, requires ≥2 captures/station).
4. Highlighted anomalous movement segments for flagged deviations (OUTSIDE_HISTORICAL_AREA, UNUSUAL_TRAVEL).
5. Visual stripe pattern keypoint correspondence evidence.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import streamlit as st

from src.history import get_history
from src.schemas import AlertType, IdentityDecisionState


def render():
    st.header("🗺️ Individual Tiger Movement & Spatial Intelligence")

    is_atrw = st.session_state.get("data_source") == "Real Tiger Images (ATRW Benchmark)"
    observations = st.session_state.get("observations", [])
    decisions = st.session_state.get("decisions", [])
    alerts = st.session_state.get("alerts", [])

    if is_atrw:
        st.info(
            "🐅 **Benchmark Data Notice:** Displaying illustrative reserve station layout for ATRW benchmark tigers. "
            "Station coordinates demonstrate spatial-temporal tracking mechanics — not actual Pench GPS readings.",
            icon="ℹ️",
        )

    if not observations:
        st.info(
            "👋 **No trusted observations in the active session store.** "
            "Navigate to the **Processing** page and run the pipeline to populate trusted movements."
        )
        return

    # Build per-tiger summary from observations
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

        st_id = obs.station_id or "UNKNOWN_STATION"
        if obs.station_id:
            entry["trusted_stations"].add(obs.station_id)

        if obs.latitude is not None and obs.longitude is not None:
            if st_id not in entry["station_details"]:
                entry["station_details"][st_id] = {
                    "station_id": st_id,
                    "latitude": obs.latitude,
                    "longitude": obs.longitude,
                    "count": 0,
                }
            entry["station_details"][st_id]["count"] += 1

    # Catalogue summary table
    st.subheader("1. Registered Tiger Catalogue")
    rows = []
    for tid, data in tiger_data.items():
        rows.append(
            {
                "Identity ID": data["identity_id"],
                "Capture Count": data["capture_count"],
                "First Observed": data["first_seen"].strftime("%Y-%m-%d %H:%M") if data["first_seen"] else "—",
                "Last Observed": data["last_seen"].strftime("%Y-%m-%d %H:%M") if data["last_seen"] else "—",
                "Trusted Stations": ", ".join(sorted(data["trusted_stations"])) or "—",
                "Longitudinal Store": "✅ Admitted into History",
            }
        )
    st.dataframe(rows, use_container_width=True)

    # Spatial Intelligence & Migration Mapping
    st.markdown("---")
    st.subheader("2. Spatial Migration Trajectory & Historical Capture Area")
    st.caption(
        "Chronological movement paths between capture stations. "
        "The shaded amber boundary represents the **historical capture area** (convex hull) — strictly NOT a home range."
    )

    for tid, data in tiger_data.items():
        with st.expander(f"🐅 **{tid}** — Migration Path & Territory Analysis ({data['capture_count']} captures)", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trusted Captures", data["capture_count"])
            col2.metric(
                "First Seen",
                data["first_seen"].strftime("%Y-%m-%d") if data["first_seen"] else "—",
            )
            col3.metric(
                "Last Seen",
                data["last_seen"].strftime("%Y-%m-%d") if data["last_seen"] else "—",
            )

            st.markdown(f"**Associated Stations:** `{', '.join(sorted(data['trusted_stations'])) or 'None'}`")

            # Show catalogue reference photo if available
            from src.identity import get_default_catalogue
            cat_img = get_default_catalogue().get_image_path(tid)
            if cat_img and Path(cat_img).exists():
                ref_col, info_col = st.columns([1, 2])
                with ref_col:
                    st.image(
                        str(cat_img),
                        caption=f"Catalogue Reference: {tid}",
                        use_container_width=True,
                    )
                with info_col:
                    st.caption(
                        f"This is the registered catalogue photograph for **{tid}** used as the "
                        f"reference template for classical stripe-pattern keypoint matching (SIFT/ORB)."
                    )

            if data["station_details"]:
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    # Sort observations chronologically for migration path
                    sorted_obs = sorted(
                        [o for o in data["observations"] if o.latitude is not None and o.longitude is not None],
                        key=lambda x: x.timestamp or datetime.min,
                    )

                    path_lats = [o.latitude for o in sorted_obs]
                    path_lons = [o.longitude for o in sorted_obs]
                    path_hover = [
                        f"<b>Obs ID:</b> {o.observation_id}<br>"
                        f"<b>Station:</b> {o.station_id}<br>"
                        f"<b>Time:</b> {o.timestamp.strftime('%Y-%m-%d %H:%M') if o.timestamp else 'Unknown'}<br>"
                        f"<b>Identity Conf:</b> {o.identity_confidence:.3f}"
                        for o in sorted_obs
                    ]

                    # 1. Historical Capture Area (Convex Hull polygon)
                    hull_points = history.compute_historical_capture_area(tid)
                    if hull_points and len(hull_points) >= 3:
                        hull_lats = [p[0] for p in hull_points] + [hull_points[0][0]]
                        hull_lons = [p[1] for p in hull_points] + [hull_points[0][1]]

                        fig.add_trace(
                            go.Scattermapbox(
                                lat=hull_lats,
                                lon=hull_lons,
                                mode="lines",
                                fill="toself",
                                fillcolor="rgba(245, 158, 11, 0.18)",
                                line=dict(color="#D97706", width=2.5),
                                name="Historical Capture Area",
                                hoverinfo="name",
                            )
                        )

                    # 2. Sequential Migration Trajectory (Connected line path)
                    if len(path_lats) >= 2:
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=path_lats,
                                lon=path_lons,
                                mode="lines",
                                line=dict(color="#1D4ED8", width=3),
                                name="Movement Trajectory",
                                hoverinfo="name",
                            )
                        )

                    # 3. Check for flagged movement alerts
                    flagged_obs_ids = set()
                    for alert in alerts:
                        if (
                            alert.identity_id == tid
                            and alert.alert_type in (AlertType.OUTSIDE_HISTORICAL_AREA, AlertType.UNUSUAL_TRAVEL)
                        ):
                            if alert.triggering_observation:
                                flagged_obs_ids.add(alert.triggering_observation.observation_id)

                    # 4. Station Markers (sized by capture effort)
                    station_list = list(data["station_details"].values())
                    st_lats = [s["latitude"] for s in station_list]
                    st_lons = [s["longitude"] for s in station_list]
                    st_counts = [s["count"] for s in station_list]
                    st_names = [s["station_id"] for s in station_list]

                    st_hover = [
                        f"<b>Station:</b> {name}<br>"
                        f"<b>Trusted Captures:</b> {c}<br>"
                        f"<b>Coordinates:</b> ({lat:.4f}, {lon:.4f})"
                        for name, c, lat, lon in zip(st_names, st_counts, st_lats, st_lons)
                    ]

                    fig.add_trace(
                        go.Scattermapbox(
                            lat=st_lats,
                            lon=st_lons,
                            mode="markers+text",
                            marker=dict(
                                size=[max(14, min(36, 12 + c * 5)) for c in st_counts],
                                color="#0F2942",
                                opacity=0.9,
                            ),
                            text=st_names,
                            textposition="top right",
                            textfont=dict(size=11, color="#0F2942"),
                            hovertext=st_hover,
                            hoverinfo="text",
                            name="Camera Stations",
                        )
                    )

                    # 5. Highlight Flagged Movement Points if any
                    flagged_points = [o for o in sorted_obs if o.observation_id in flagged_obs_ids]
                    if flagged_points:
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=[o.latitude for o in flagged_points],
                                lon=[o.longitude for o in flagged_points],
                                mode="markers",
                                marker=dict(
                                    size=18,
                                    color="#DC2626",
                                    symbol="circle",
                                ),
                                name="⚠️ Flagged Movement Deviation",
                                hovertext=[f"⚠️ Flagged Deviation: {o.station_id}" for o in flagged_points],
                                hoverinfo="text",
                            )
                        )

                    center_lat = sum(st_lats) / len(st_lats)
                    center_lon = sum(st_lons) / len(st_lons)

                    fig.update_layout(
                        mapbox=dict(
                            style="carto-positron",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=11.5,
                        ),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=420,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.98,
                            xanchor="left",
                            x=0.02,
                            bgcolor="rgba(255, 255, 255, 0.90)",
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Map rendering fallback ({e}). Tabular capture records shown below.")
                    st.dataframe(data["station_details"], use_container_width=True)
            else:
                st.caption("No GPS coordinate data available for spatial trajectory mapping.")
