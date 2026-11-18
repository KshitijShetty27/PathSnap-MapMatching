"""
Data Loader Module for PathSnap.
Handles loading .track, .nodes, .arcs, and .route files,
metric coordinate projection, and ground-truth arc alignment.
"""

import os
import math
import numpy as np


def project_coordinates(coords, ref_lat):
    """
    Projects (lon, lat) angular coordinates into local Cartesian coordinates in meters
    using an equirectangular projection centered at ref_lat.
    """
    cos_lat = math.cos(math.radians(ref_lat))
    scale = np.array([111320.0 * cos_lat, 110540.0])
    return coords * scale, scale


def point_to_segment_distance_batch(p, a, b):
    """
    Calculates perpendicular distance and projection ratio t for point p (2,)
    against arrays of segment endpoints a (N, 2) and b (N, 2) in meters.
    Returns:
        dists: array of distances in meters (N,)
        t: projection parameter clamped to [0, 1] (N,)
    """
    ab = b - a
    ap = p - a
    ab_len_sq = np.sum(ab**2, axis=1)
    mask = ab_len_sq > 0
    t = np.zeros(len(a), dtype=np.float64)
    t[mask] = np.clip(np.sum(ap[mask] * ab[mask], axis=1) / ab_len_sq[mask], 0.0, 1.0)
    proj = a + t[:, None] * ab
    dists = np.linalg.norm(p - proj, axis=1)
    return dists, t


def load_trip(trip_dir, trip_id=None):
    """
    Loads all data for a single trip.
    trip_dir: path to directory containing trip files.
    trip_id: optional trip ID string; if None, inferred from directory name or filenames.
    Returns dict with keys:
        'trip_id': string
        'track': ndarray (N, 3) -> [lon, lat, timestamp]
        'nodes': ndarray (M, 2) -> [lon, lat]
        'arcs': ndarray (K, 2) -> [from_node, to_node]
        'route': ndarray (R,) -> ground truth arc IDs (or None if not available)
        'nodes_m': ndarray (M, 2) -> nodes in local metric coordinates
        'track_m': ndarray (N, 2) -> track in local metric coordinates
        'scale': projection scale factors
        'ref_lat': reference latitude
    """
    if trip_id is None:
        trip_id = os.path.basename(os.path.normpath(trip_dir))

    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")

    if not os.path.exists(track_file):
        # try find matching files in directory
        for f in os.listdir(trip_dir):
            if f.endswith(".track"):
                trip_id = f.replace(".track", "")
                track_file = os.path.join(trip_dir, f"{trip_id}.track")
                nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
                arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
                route_file = os.path.join(trip_dir, f"{trip_id}.route")
                break

    track = np.loadtxt(track_file)
    nodes = np.loadtxt(nodes_file)
    arcs = np.loadtxt(arcs_file, dtype=int)
    route = np.loadtxt(route_file, dtype=int) if os.path.exists(route_file) else None

    # Reference latitude from track median
    ref_lat = float(np.median(track[:, 1]))
    nodes_m, scale = project_coordinates(nodes, ref_lat)
    track_m, _ = project_coordinates(track[:, :2], ref_lat)

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "nodes_m": nodes_m,
        "track_m": track_m,
        "scale": scale,
        "ref_lat": ref_lat,
    }


def align_ground_truth_route(trip_data):
    """
    Aligns each GPS observation in track_m to its corresponding ground-truth arc
    from the continuous sequence of arcs in route.
    Uses sequential windowed search to preserve monotonic trajectory progress.
    Returns:
        true_arcs: ndarray (N,) of ground-truth arc IDs for each GPS point.
    """
    route = trip_data["route"]
    if route is None:
        return None

    nodes_m = trip_data["nodes_m"]
    arcs = trip_data["arcs"]
    track_m = trip_data["track_m"]

    route_u = nodes_m[arcs[route, 0]]
    route_v = nodes_m[arcs[route, 1]]

    true_arcs = []
    cur_idx = 0
    n_route = len(route)

    for pt in track_m:
        # Window around current route progress
        search_start = max(0, cur_idx - 4)
        search_end = min(n_route, cur_idx + 14)
        dists, _ = point_to_segment_distance_batch(
            pt, route_u[search_start:search_end], route_v[search_start:search_end]
        )
        best_local = np.argmin(dists)
        best_route_idx = search_start + best_local
        true_arcs.append(route[best_route_idx])
        cur_idx = best_route_idx

    return np.array(true_arcs, dtype=int)
