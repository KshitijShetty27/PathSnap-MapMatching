"""
Common experimental utilities for independent team member model training.
Caches preprocessed features so each teammate can train their model in seconds.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_trip, align_ground_truth_route
from src.preprocessing import build_spatial_arc_index, generate_candidates_for_track, extract_features_for_trip

DATA_ROOT = os.path.join(os.path.dirname(BASE_DIR), "Student folder", "Student folder")
CACHE_DIR = os.path.join(BASE_DIR, "artifacts", "cache")


def get_cached_dataset():
    """
    Returns (X_train, y_train, test_trips_data) for model experimentation.
    Caches features to disk for instant loading across parallel model training runs.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    x_tr_path = os.path.join(CACHE_DIR, "X_train.npy")
    y_tr_path = os.path.join(CACHE_DIR, "y_train.npy")

    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir = os.path.join(DATA_ROOT, "student_test")

    # Load or extract training data
    if os.path.exists(x_tr_path) and os.path.exists(y_tr_path):
        X_train = np.load(x_tr_path)
        y_train = np.load(y_tr_path)
    else:
        train_trips = sorted(os.listdir(train_dir))[:15]
        X_list, y_list = [], []
        for tid in train_trips:
            t_path = os.path.join(train_dir, tid)
            data = load_trip(t_path, tid)
            true_arcs = align_ground_truth_route(data)
            tree, arc_ids = build_spatial_arc_index(data["nodes_m"], data["arcs"])
            cands = generate_candidates_for_track(tree, arc_ids, data["track_m"], max_candidates=10)
            X, y, _ = extract_features_for_trip(data, cands, true_arcs=true_arcs)
            X_list.append(X)
            y_list.append(y)
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        np.save(x_tr_path, X_train)
        np.save(y_tr_path, y_train)

    # Prepare student-test data
    test_trips = sorted(os.listdir(test_dir))
    test_trips_data = []
    for tid in test_trips:
        t_path = os.path.join(test_dir, tid)
        data = load_trip(t_path, tid)
        true_arcs = align_ground_truth_route(data)
        tree, arc_ids = build_spatial_arc_index(data["nodes_m"], data["arcs"])
        cands = generate_candidates_for_track(tree, arc_ids, data["track_m"], max_candidates=10)
        X, y, slices = extract_features_for_trip(data, cands, true_arcs=true_arcs)
        test_trips_data.append({
            "data": data, "true_arcs": true_arcs, "slices": slices, "X": X, "y": y
        })

    return X_train, y_train, test_trips_data
