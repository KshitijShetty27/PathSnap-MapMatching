"""
Visualization Module for PathSnap.
Generates interactive Folium maps and high-resolution matplotlib figures
for trajectory analysis, route comparison, and report embedding.
"""

import folium
import matplotlib.pyplot as plt
import numpy as np


def create_folium_map(trip_data, predicted_arcs=None, out_html_path=None):
    """
    Creates an interactive Folium map showing:
    - Raw GPS trajectory observations (blue markers)
    - Ground-truth reference route (green polyline)
    - Predicted route from map-matching (red dashed polyline)
    - Layer controls for toggling layers
    """
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data.get("route")

    # Center map on mean trajectory coordinates
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    # 1. Raw GPS Observations Layer
    gps_group = folium.FeatureGroup(name="Raw GPS Trajectory (Observations)", show=True)
    # Downsample points slightly if track is very long for snappy map rendering
    step = max(1, len(track) // 300)
    for idx in range(0, len(track), step):
        lon, lat, ts = track[idx]
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            popup=f"Obs #{idx}<br>Time: {ts:.1f}s<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
        ).add_to(gps_group)
    gps_group.add_to(m)

    # 2. Ground-Truth Reference Route Layer (Green)
    if route is not None and len(route) > 0:
        ref_group = folium.FeatureGroup(name="Ground-Truth Reference Route (Green)", show=True)
        ref_coords = []
        for aid in route:
            u, v = arcs[aid]
            ref_coords.append([nodes[u][1], nodes[u][0]])
            ref_coords.append([nodes[v][1], nodes[v][0]])

        folium.PolyLine(
            locations=ref_coords,
            color="#2ca02c",
            weight=4,
            opacity=0.85,
            tooltip="Ground-Truth Route",
        ).add_to(ref_group)
        ref_group.add_to(m)

    # 3. Predicted Route Layer (Red / Crimson)
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        pred_group = folium.FeatureGroup(name="Predicted HMM Map-Matched Route (Red)", show=True)
        pred_coords = []
        # Condense consecutive duplicates
        condensed = []
        for aid in predicted_arcs:
            if aid != -1 and (not condensed or condensed[-1] != aid):
                condensed.append(aid)

        for aid in condensed:
            u, v = arcs[aid]
            pred_coords.append([nodes[u][1], nodes[u][0]])
            pred_coords.append([nodes[v][1], nodes[v][0]])

        folium.PolyLine(
            locations=pred_coords,
            color="#d62728",
            weight=3.5,
            dash_array="6, 6",
            opacity=0.9,
            tooltip="Predicted Route",
        ).add_to(pred_group)
        pred_group.add_to(m)

    folium.LayerControl().add_to(m)

    if out_html_path:
        m.save(out_html_path)

    return m


def plot_static_route_comparison(trip_data, predicted_arcs, out_png_path):
    """
    Renders a clean static matplotlib plot comparing GPS track, true route, and predicted route.
    Used for embedding into the final PDF technical report.
    """
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data.get("route")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Reference route
    if route is not None:
        for aid in route:
            u, v = arcs[aid]
            ax.plot(
                [nodes[u][0], nodes[v][0]],
                [nodes[u][1], nodes[v][1]],
                color="#2ca02c",
                linewidth=2.5,
                alpha=0.7,
                label="Ground Truth Route" if aid == route[0] else "",
            )

    # Predicted route
    if predicted_arcs is not None:
        condensed = []
        for aid in predicted_arcs:
            if aid != -1 and (not condensed or condensed[-1] != aid):
                condensed.append(aid)
        for aid in condensed:
            u, v = arcs[aid]
            ax.plot(
                [nodes[u][0], nodes[v][0]],
                [nodes[u][1], nodes[v][1]],
                color="#d62728",
                linestyle="--",
                linewidth=1.8,
                alpha=0.85,
                label="HMM Predicted Route" if aid == condensed[0] else "",
            )

    # GPS points
    ax.scatter(
        track[:, 0],
        track[:, 1],
        s=12,
        color="#1f77b4",
        alpha=0.6,
        label="Noisy GPS Observations",
        zorder=5,
    )

    ax.set_title(f"PathSnap Map-Matching Reconstruction: Trip {trip_data['trip_id']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches="tight")
    plt.close()
