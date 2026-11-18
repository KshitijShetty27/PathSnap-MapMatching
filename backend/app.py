"""
FastAPI Server for PathSnap (Runs on Laptop 1).
Exposes REST endpoints for model inference and server-side evaluation.
Keeps ground-truth reference routes strictly server-side.
Outputs clear, verifiable request logs to the terminal.
"""

import os
import sys
import time
import logging
import joblib
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_trip, align_ground_truth_route
from src.preprocessing import build_spatial_arc_index, generate_candidates_for_track, extract_features_for_trip
from src.hmm import HMMRouteDecoder, calculate_route_continuity
from src.evaluation import evaluate_trip_observation_level, evaluate_sequence_recovery

# Configure terminal logging for evaluator inspection
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed Trajectory Reconstruction & Evaluation Backend",
    version="1.0.0",
)

# Enable CORS for cross-laptop network communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths for evaluation trips (evaluator_test and student_test)
EVALUATOR_TEST_DIR = os.path.join(
    os.path.dirname(BASE_DIR), "Student folder", "Student folder", "evaluator_test"
)
STUDENT_TEST_DIR = os.path.join(
    os.path.dirname(BASE_DIR), "Student folder", "Student folder", "student_test"
)
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.joblib")

# Global state for loaded model and decoder
MODEL = None
DECODER = HMMRouteDecoder(p_self=1.0, p_connected=0.85, penalty_disconnected=1e-5, beta=15.0)


@app.on_event("startup")
def load_model_on_startup():
    global MODEL
    if os.path.exists(MODEL_ARTIFACT_PATH):
        logger.info(f"Loading trained model artifact from: {MODEL_ARTIFACT_PATH}")
        MODEL = joblib.load(MODEL_ARTIFACT_PATH)
        logger.info("Trained model successfully loaded into memory.")
    else:
        logger.warning(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Run training pipeline first.")


class EvaluationRequest(BaseModel):
    trip_id: str
    use_hmm: Optional[bool] = True


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap Map-Matching API",
        "model_loaded": MODEL is not None,
        "test_data_dir_exists": os.path.exists(STUDENT_TEST_DIR),
        "timestamp": time.time(),
    }


@app.get("/trips")
def list_available_trips():
    """Lists evaluation trips available on the server (evaluator trips prioritized)."""
    trips = []
    if os.path.exists(EVALUATOR_TEST_DIR):
        trips.extend(sorted([d for d in os.listdir(EVALUATOR_TEST_DIR) if os.path.isdir(os.path.join(EVALUATOR_TEST_DIR, d))]))
    if os.path.exists(STUDENT_TEST_DIR):
        for d in sorted([d for d in os.listdir(STUDENT_TEST_DIR) if os.path.isdir(os.path.join(STUDENT_TEST_DIR, d))]):
            if d not in trips:
                trips.append(d)
    return {"available_trips": trips, "count": len(trips)}


@app.post("/evaluate")
def evaluate_trip_endpoint(req: EvaluationRequest):
    """
    Executes complete map-matching prediction and calculates accuracy against server-side reference data.
    """
    t_start = time.time()
    trip_id = req.trip_id.strip()
    logger.info(f"=== [REQUEST RECEIVED] Evaluation for Trip ID: '{trip_id}' ===")

    if MODEL is None:
        logger.error("Inference rejected: Model artifact not loaded.")
        raise HTTPException(status_code=500, detail="Model artifact is not loaded on server.")

    trip_path = os.path.join(EVALUATOR_TEST_DIR, trip_id)
    if not os.path.exists(trip_path):
        trip_path = os.path.join(STUDENT_TEST_DIR, trip_id)
    if not os.path.exists(trip_path):
        logger.error(f"Trip path not found for: {trip_id}")
        raise HTTPException(status_code=404, detail=f"Trip ID '{trip_id}' not found in reserved evaluation dataset.")

    # 1. Load trip data
    logger.info(f"Loading raw files for trip {trip_id} (.track, .nodes, .arcs, .route)...")
    trip_data = load_trip(trip_path, trip_id)
    n_obs = len(trip_data["track"])
    logger.info(f"Loaded {n_obs} GPS observations, {len(trip_data['nodes'])} nodes, {len(trip_data['arcs'])} arcs.")

    # 2. Align server-side ground truth route
    true_arcs = align_ground_truth_route(trip_data)

    # 3. Spatial indexing and candidate generation
    t_cands = time.time()
    tree, all_sample_arc_ids = build_spatial_arc_index(trip_data["nodes_m"], trip_data["arcs"])
    point_cands = generate_candidates_for_track(tree, all_sample_arc_ids, trip_data["track_m"], max_candidates=10)
    cand_time = time.time() - t_cands
    logger.info(f"Spatial candidate indexing completed in {cand_time*1000:.1f}ms.")

    # 4. Feature engineering
    X, y, slices = extract_features_for_trip(trip_data, point_cands, true_arcs=true_arcs)
    logger.info(f"Extracted {X.shape[1]} features across {len(X)} candidate road segments.")

    # 5. Classifier Inference
    t_inf = time.time()
    candidate_probs = MODEL.predict_proba(X)[:, 1]
    cls_infer_time = time.time() - t_inf
    logger.info(f"ML Model inference completed in {cls_infer_time*1000:.1f}ms.")

    # 6. Classifier-Only evaluation
    cls_acc, cls_arcs = evaluate_trip_observation_level(slices, candidate_probs, true_arcs)
    cls_continuity = calculate_route_continuity(cls_arcs, trip_data["arcs"])

    # 7. HMM / Viterbi Sequential Reconstruction
    t_hmm = time.time()
    predicted_arcs, condensed_route = DECODER.decode(slices, candidate_probs, trip_data)
    hmm_time = time.time() - t_hmm
    logger.info(f"HMM / Viterbi decoding completed in {hmm_time*1000:.1f}ms.")

    # 8. Server-side accuracy calculation
    hmm_acc = (float(np.mean([predicted_arcs[i] == true_arcs[i] for i in range(len(true_arcs))]))) * 100.0
    hmm_continuity = calculate_route_continuity(predicted_arcs, trip_data["arcs"])
    seq_recovery = evaluate_sequence_recovery(condensed_route, trip_data.get("route"))

    total_time_ms = (time.time() - t_start) * 1000.0
    logger.info(
        f"=== [RESULTS] Trip {trip_id} | Top-1 Accuracy: {hmm_acc:.2f}% | Continuity: {hmm_continuity:.2f}% | Total Time: {total_time_ms:.1f}ms ==="
    )

    # 9. Format coordinates for visualization in UI
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    track = trip_data["track"]

    # Sample GPS points (e.g. up to 250 points) for fast UI rendering
    subsample_step = max(1, len(track) // 250)
    gps_sample = [
        {"lat": float(track[i, 1]), "lon": float(track[i, 0]), "ts": float(track[i, 2])}
        for i in range(0, len(track), subsample_step)
    ]

    # Reference route coordinate segments
    ref_route = trip_data.get("route")
    ref_coords = []
    if ref_route is not None:
        for aid in ref_route:
            u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
            ref_coords.append([[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]])

    # Predicted route coordinate segments
    pred_coords = []
    for aid in condensed_route:
        u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
        pred_coords.append([[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]])

    return {
        "trip_id": trip_id,
        "observations_count": int(n_obs),
        "metrics": {
            "top1_accuracy": round(float(hmm_acc), 2),
            "route_continuity": round(float(hmm_continuity), 2),
            "sequence_recovery": round(float(seq_recovery), 2),
            "classifier_only_accuracy": round(float(cls_acc), 2),
            "classifier_only_continuity": round(float(cls_continuity), 2),
            "inference_time_ms": round(float(total_time_ms), 2),
        },
        "gps_sample": gps_sample,
        "reference_route_coords": ref_coords,
        "predicted_route_coords": pred_coords,
        "predicted_arcs_sample": [int(x) for x in predicted_arcs[:20]],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
