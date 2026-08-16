"""
ui/movement.py — Movement / Catalogue page.

Per-tiger: capture count, first/last seen, trusted stations, and a
simple table of the "historical capture area" (NEVER "home range").
If coordinates are available, show a simple map.
"""

import streamlit as st

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
    tiger_data = {}
    for obs in observations:
        tid = obs.identity_id
        if tid not in tiger_data:
            tiger_data[tid] = {
                "identity_id": tid,
                "capture_count": 0,
                "first_seen": None,
                "last_seen": None,
                "trusted_stations": set(),
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

        if obs.station_id:
            entry["trusted_stations"].add(obs.station_id)

        if obs.latitude is not None and obs.longitude is not None:
            entry["latitudes"].append(obs.latitude)
            entry["longitudes"].append(obs.longitude)

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

    # ------ Per-tiger detail ------
    for tid, data in tiger_data.items():
        with st.expander(f"🐯 **{tid}** — {data['capture_count']} capture(s)"):
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
            if data["latitudes"] and data["longitudes"]:
                st.markdown("**Historical Capture Area**")
                st.caption(
                    "This is the set of coordinates where this individual has "
                    "been observed with trusted identity — NOT a validated home range."
                )

                import pandas as pd

                map_df = pd.DataFrame(
                    {
                        "lat": data["latitudes"],
                        "lon": data["longitudes"],
                    }
                )
                st.map(map_df, use_container_width=True)
            else:
                st.caption("No coordinate data available for mapping.")

    # ------ Trusted history status ------
    st.subheader("Trusted History Status")
    st.markdown(
        f"**{len(observations)}** observation(s) in trusted history. "
        f"Only `trusted_match` decisions with `update_history=True` "
        f"are included."
    )

    if decisions:
        non_trusted = sum(
            1 for d in decisions if not d.update_history
        )
        st.caption(
            f"ℹ️ {non_trusted} decision(s) were correctly withheld "
            f"from trusted history."
        )
