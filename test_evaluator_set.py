import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_trip, align_ground_truth_route
from src.preprocessing import build_spatial_arc_index, generate_candidates_for_track, extract_features_for_trip
from src.hmm import HMMRouteDecoder, calculate_route_continuity
from src.evaluation import evaluate_trip_observation_level, evaluate_sequence_recovery

DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Student folder", "Student folder")
eval_dir = os.path.join(DATA_DIR, "evaluator_test")
model_path = os.path.join(BASE_DIR, "artifacts", "best_model.joblib")

def run_evaluator_benchmark():
    print(f"Loading trained model: {model_path}")
    model = joblib.load(model_path)
    decoder = HMMRouteDecoder(p_self=1.0, p_connected=0.85, penalty_disconnected=1e-5, beta=15.0)

    trips = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
    print(f"\n======================================================================")
    print(f"       PATHSNAP: BENCHMARK ON {len(trips)} RESERVED EVALUATOR TRIPS           ")
    print(f"======================================================================\n")

    results = []
    for trip_id in trips:
        t0 = time.time()
        trip_path = os.path.join(eval_dir, trip_id)
        trip_data = load_trip(trip_path, trip_id)
        true_arcs = align_ground_truth_route(trip_data)

        tree, all_sample_arc_ids = build_spatial_arc_index(trip_data["nodes_m"], trip_data["arcs"])
        point_cands = generate_candidates_for_track(tree, all_sample_arc_ids, trip_data["track_m"], max_candidates=10)
        X, y, slices = extract_features_for_trip(trip_data, point_cands, true_arcs=true_arcs)

        probs = model.predict_proba(X)[:, 1]

        # Classifier only
        cls_acc, cls_arcs = evaluate_trip_observation_level(slices, probs, true_arcs)
        cls_cont = calculate_route_continuity(cls_arcs, trip_data["arcs"])

        # HMM Viterbi decoding
        hmm_arcs, condensed_route = decoder.decode(slices, probs, trip_data)
        hmm_acc = (np.mean([hmm_arcs[i] == true_arcs[i] for i in range(len(true_arcs))])) * 100.0
        hmm_cont = calculate_route_continuity(hmm_arcs, trip_data["arcs"])
        seq_rec = evaluate_sequence_recovery(condensed_route, trip_data["route"])

        elapsed = time.time() - t0

        results.append({
            "trip_id": trip_id,
            "observations": len(trip_data["track"]),
            "classifier_acc": cls_acc,
            "hmm_acc": hmm_acc,
            "classifier_continuity": cls_cont,
            "hmm_continuity": hmm_cont,
            "sequence_recovery": seq_rec,
            "time_sec": elapsed
        })
        print(f"Trip {trip_id} ({len(trip_data['track'])} pts): Cls Acc={cls_acc:.2f}%, HMM Acc={hmm_acc:.2f}%, HMM Cont={hmm_cont:.2f}%, Seq Rec={seq_rec:.2f}% [{elapsed:.2f}s]")

    df = pd.DataFrame(results)
    print("\n======================================================================")
    print("                    EVALUATOR TEST SUMMARY METRICS                    ")
    print("======================================================================")
    print(f"Mean Classifier Top-1 Accuracy : {df['classifier_acc'].mean():.2f}%")
    print(f"Mean HMM Top-1 Accuracy        : {df['hmm_acc'].mean():.2f}%")
    print(f"Mean Classifier Continuity     : {df['classifier_continuity'].mean():.2f}%")
    print(f"Mean HMM Route Continuity      : {df['hmm_continuity'].mean():.2f}%")
    print(f"Mean Sequence Recovery Jaccard : {df['sequence_recovery'].mean():.2f}%")
    print("======================================================================\n")

    # Save to artifacts
    out_csv = os.path.join(BASE_DIR, "artifacts", "evaluator_test_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved full evaluator results to: {out_csv}")

if __name__ == "__main__":
    run_evaluator_benchmark()
