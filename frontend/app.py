"""
Streamlit User Interface for PathSnap (Runs on Laptop 2).
Connects to FastAPI running on Laptop 1.
Displays the 10 reserved evaluation trips, submits requests, and visualizes
accuracy metrics and interactive Folium trajectory maps.
"""

import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="PathSnap: Distributed Map-Matching",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.markdown(
    "**Distributed Evaluation Dashboard** | Connects to Laptop 1 (FastAPI Model Backend) "
    "to reconstruct vehicle routes from noisy GPS tracks."
)

# Sidebar configuration
st.sidebar.header("⚙️ Laptop 1 Connection Settings")
api_url = st.sidebar.text_input(
    "Laptop 1 API Base URL",
    value="http://10.8.120.105:8000",
    help="Enter http://<Laptop-1-IP>:8000 when running across two machines.",
)

# Connection check
api_online = False
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2.5)
    if health_resp.status_code == 200:
        api_online = True
        st.sidebar.success("🟢 Laptop 1 API Online")
    else:
        st.sidebar.error("🔴 Laptop 1 API Error")
except Exception:
    st.sidebar.error("🔴 Cannot reach Laptop 1 API. Ensure Laptop 1 has api/app.py running.")

# Fetch available trips (prioritizing the 10 reserved evaluator trips)
trips_list = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094",
    "00000000", "00000007", "00000009", "00000040", "00000052"
]
if api_online:
    try:
        trips_resp = requests.get(f"{api_url}/trips", timeout=2.0)
        if trips_resp.status_code == 200:
            server_trips = trips_resp.json().get("available_trips", [])
            if server_trips:
                trips_list = server_trips
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.subheader("🚗 Select Reserved Evaluation Trip")
selected_trip = st.sidebar.selectbox(
    "Choose one of the 10 Reserved Trips:",
    trips_list,
    index=0
)

use_hmm = st.sidebar.checkbox("Enable HMM Viterbi Decoding", value=True)
run_btn = st.sidebar.button("🚀 Run Map-Matching Evaluation", type="primary", use_container_width=True)

# Handle evaluation submission
if run_btn:
    if not api_online:
        st.error(f"Cannot run evaluation: Laptop 1 API at `{api_url}` is offline.")
    else:
        with st.spinner(f"Sending Trip ID `{selected_trip}` to Laptop 1 API for end-to-end inference..."):
            try:
                payload = {"trip_id": selected_trip, "use_hmm": use_hmm}
                resp = requests.post(f"{api_url}/evaluate", json=payload, timeout=60.0)
                if resp.status_code == 200:
                    st.session_state["eval_data"] = resp.json()
                    st.session_state["evaluated_trip"] = selected_trip
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Failed to communicate with Laptop 1: {e}")

# Render evaluation results if available in session state
if "eval_data" in st.session_state:
    data = st.session_state["eval_data"]
    evaluated_trip = st.session_state.get("evaluated_trip", selected_trip)
    metrics = data["metrics"]

    st.success(f"✅ Successfully evaluated Trip **{evaluated_trip}** across complete pipeline!")

    # Metric cards row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Top-1 Accuracy (HMM)",
        f"{metrics['top1_accuracy']}%",
        delta=f"{metrics['top1_accuracy'] - metrics['classifier_only_accuracy']:+.1f}% vs Clf",
    )
    col2.metric(
        "Route Continuity",
        f"{metrics['route_continuity']}%",
        delta=f"{metrics['route_continuity'] - metrics['classifier_only_continuity']:+.1f}% vs Clf",
    )
    col3.metric("Classifier-Only Acc", f"{metrics['classifier_only_accuracy']}%")
    col4.metric("Sequence Recovery", f"{metrics['sequence_recovery']}%")
    col5.metric("Latency", f"{metrics['inference_time_ms']} ms")

    st.markdown("---")
    st.subheader(f"📍 Interactive Folium Map: Trip {evaluated_trip}")

    gps_sample = data.get("gps_sample", [])
    ref_coords = data.get("reference_route_coords", [])
    pred_coords = data.get("predicted_route_coords", [])

    if gps_sample:
        center_lat = sum(p["lat"] for p in gps_sample) / len(gps_sample)
        center_lon = sum(p["lon"] for p in gps_sample) / len(gps_sample)
    else:
        center_lat, center_lon = 49.85, 8.60

    # Create folium map using OpenStreetMap (free, no API key required)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    # Layer 1: GPS Points (Blue)
    gps_group = folium.FeatureGroup(name="Raw GPS Trajectory (Blue Dots)", show=True)
    for pt in gps_sample:
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=3.5,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            popup=f"Time: {pt['ts']:.1f}s",
        ).add_to(gps_group)
    gps_group.add_to(m)

    # Layer 2: Reference Route (Green)
    if ref_coords:
        ref_group = folium.FeatureGroup(name="Ground-Truth Reference Route (Green)", show=True)
        for seg in ref_coords:
            folium.PolyLine(locations=seg, color="#2ca02c", weight=4, opacity=0.85).add_to(ref_group)
        ref_group.add_to(m)

    # Layer 3: Predicted Route (Red)
    if pred_coords:
        pred_group = folium.FeatureGroup(name="Predicted HMM Map-Matched Route (Red)", show=True)
        for seg in pred_coords:
            folium.PolyLine(locations=seg, color="#d62728", weight=3.5, dash_array="6, 6", opacity=0.9).add_to(pred_group)
        pred_group.add_to(m)

    folium.LayerControl().add_to(m)

    # Render Folium map in Streamlit
    st_folium(m, width=1100, height=550, returned_objects=[])

else:
    st.info("👈 Select a trip from the sidebar and click **'Run Map-Matching Evaluation'** to begin.")
    st.markdown("""
    ### 📋 System Architecture
    - **Laptop 1 (Backend)**: Hosts FastAPI server, trained ML classifier weights, road network graphs, and computes server-side evaluation.
    - **Laptop 2 (Frontend)**: Runs this Streamlit UI, sends requests containing only the Trip ID, and renders real-time map-matched trajectories.
    """)
