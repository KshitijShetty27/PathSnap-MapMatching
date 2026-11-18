"""
Preprocessing and Feature Engineering Module for PathSnap.
Builds spatial index, generates candidate road segments, and extracts discriminative features.
"""

import math
import numpy as np
from scipy.spatial import cKDTree
from src.data_loader import point_to_segment_distance_batch


def build_spatial_arc_index(nodes_m, arcs):
    """
    Builds a cKDTree spatial index by densely sampling points along each arc.
    Returns:
        tree: cKDTree instance
        sample_arc_ids: ndarray mapping each tree point to its corresponding arc_id
    """
    u_m = nodes_m[arcs[:, 0]]
    v_m = nodes_m[arcs[:, 1]]

    # Sample at 20%, 50%, and 80% along each segment
    sample_pts = []
    sample_arc_ids = []
    n_arcs = len(arcs)
    arc_indices = np.arange(n_arcs, dtype=int)

    for alpha in [0.2, 0.5, 0.8]:
        pts = u_m + alpha * (v_m - u_m)
        sample_pts.append(pts)
        sample_arc_ids.append(arc_indices)

    all_sample_pts = np.vstack(sample_pts)
    all_sample_arc_ids = np.concatenate(sample_arc_ids)

    tree = cKDTree(all_sample_pts)
    return tree, all_sample_arc_ids


def generate_candidates_for_track(tree, all_sample_arc_ids, track_m, max_candidates=10, k_query=35):
    """
    Retrieves top candidate arcs for each GPS observation in track_m.
    Returns:
        point_candidates: list of candidate arc ID lists, one per GPS observation.
    """
    _, idxs = tree.query(track_m, k=k_query)
    point_candidates = []
    for i in range(len(track_m)):
        # Preserve order of closeness while removing duplicates
        seen = set()
        cands = []
        for sample_idx in idxs[i]:
            aid = all_sample_arc_ids[sample_idx]
            if aid not in seen:
                seen.add(aid)
                cands.append(aid)
                if len(cands) >= max_candidates:
                    break
        point_candidates.append(cands)
    return point_candidates


def extract_features_for_trip(trip_data, point_candidates, true_arcs=None):
    """
    Extracts discriminative features for each candidate arc associated with each GPS observation.
    Features extracted:
        0. dist_to_segment: perpendicular distance from GPS point to arc (meters)
        1. proj_ratio: projection ratio t along segment [0, 1]
        2. dist_to_u: distance to arc start node (meters)
        3. dist_to_v: distance to arc end node (meters)
        4. arc_length: physical length of the road segment (meters)
        5. heading_diff: angular difference between GPS travel direction and arc direction [0, pi]
        6. gps_speed: estimated speed between consecutive GPS points (m/s)
        7. dist_rank: rank order among candidates sorted by distance (0 = closest)
        8. time_delta: time interval from previous observation (seconds)

    Returns:
        X: ndarray (N_candidates_total, n_features)
        y: ndarray (N_candidates_total,) of binary labels (if true_arcs provided, else None)
        point_slices: list of tuples (start_idx, end_idx, candidate_list)
    """
    nodes_m = trip_data["nodes_m"]
    arcs = trip_data["arcs"]
    track = trip_data["track"]
    track_m = trip_data["track_m"]

    X = []
    y = [] if true_arcs is not None else None
    point_slices = []

    cur_idx = 0
    n_pts = len(track_m)

    for i in range(n_pts):
        pt = track_m[i]
        c_list = point_candidates[i]
        n_cands = len(c_list)
        if n_cands == 0:
            point_slices.append((cur_idx, cur_idx, []))
            continue

        # GPS motion features
        if i > 0:
            dt = max(0.1, track[i, 2] - track[i - 1, 2])
            dx_gps = pt[0] - track_m[i - 1, 0]
            dy_gps = pt[1] - track_m[i - 1, 1]
            gps_dist = math.hypot(dx_gps, dy_gps)
            gps_speed = gps_dist / dt
            gps_heading = math.atan2(dx_gps, dy_gps)
        else:
            dt = 1.0
            gps_speed = 0.0
            gps_heading = 0.0

        cand_u = nodes_m[arcs[c_list, 0]]
        cand_v = nodes_m[arcs[c_list, 1]]

        d_seg, proj_t = point_to_segment_distance_batch(pt, cand_u, cand_v)

        # Distance ranking among candidates
        sorted_order = np.argsort(d_seg)
        ranks = np.empty_like(sorted_order)
        ranks[sorted_order] = np.arange(n_cands)

        # Segment geometries
        dx_arc = cand_v[:, 0] - cand_u[:, 0]
        dy_arc = cand_v[:, 1] - cand_u[:, 1]
        arc_lengths = np.hypot(dx_arc, dy_arc)
        arc_headings = np.arctan2(dx_arc, dy_arc)

        # Node endpoint distances
        du = np.hypot(pt[0] - cand_u[:, 0], pt[1] - cand_u[:, 1])
        dv = np.hypot(pt[0] - cand_v[:, 0], pt[1] - cand_v[:, 1])

        start_idx = cur_idx
        for c_idx in range(n_cands):
            cid = c_list[c_idx]

            if i > 0:
                h_diff = abs(gps_heading - arc_headings[c_idx])
                while h_diff > math.pi:
                    h_diff = abs(h_diff - 2 * math.pi)
            else:
                h_diff = 0.0

            feats = [
                d_seg[c_idx],
                proj_t[c_idx],
                du[c_idx],
                dv[c_idx],
                arc_lengths[c_idx],
                h_diff,
                gps_speed,
                float(ranks[c_idx]),
                dt,
            ]
            X.append(feats)

            if true_arcs is not None:
                label = 1 if cid == true_arcs[i] else 0
                y.append(label)

            cur_idx += 1

        point_slices.append((start_idx, cur_idx, c_list))

    X_arr = np.array(X, dtype=np.float32) if len(X) > 0 else np.empty((0, 9), dtype=np.float32)
    y_arr = np.array(y, dtype=int) if y is not None else None

    return X_arr, y_arr, point_slices
