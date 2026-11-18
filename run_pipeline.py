"""
Master Pipeline Script for PathSnap.
Executes end-to-end workflow:
1. Data inspection and exploratory data analysis (EDA).
2. Spatial indexing, candidate generation, and feature extraction.
3. Model training & comparison across 5 distinct algorithm families.
4. Bayesian hyperparameter optimization on the chosen model.
5. HMM + Viterbi route reconstruction and continuity analysis.
6. Export of trained model weights, evaluation metrics CSVs, and visualization maps.
"""

import os
import sys
import time
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_trip, align_ground_truth_route
from src.preprocessing import build_spatial_arc_index, generate_candidates_for_track, extract_features_for_trip
from src.models import get_candidate_models, train_single_model
from src.bayesian import run_bayesian_optimization
from src.hmm import HMMRouteDecoder, calculate_route_continuity
from src.evaluation import benchmark_model_on_trips, evaluate_trip_observation_level, evaluate_predictions, evaluate_sequence_recovery
from src.visualizer import create_folium_map, plot_static_route_comparison


def main():
    print("=================================================================")
    print("      PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION   ")
    print("=================================================================\n")

    data_base = os.path.join(os.path.dirname(BASE_DIR), "Student folder", "Student folder")
    train_dir = os.path.join(data_base, "train")
    test_dir = os.path.join(data_base, "student_test")

    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"Data root: {data_base}")
    print(f"Artifacts: {artifacts_dir}\n")

    # -------------------------------------------------------------
    # 1. Exploratory Data Analysis & Inventory
    # -------------------------------------------------------------
    print("--- Stage 1: Exploratory Data Analysis & Inventory ---")
    train_trips = sorted(os.listdir(train_dir))
    test_trips = sorted(os.listdir(test_dir))
    print(f"Discovered {len(train_trips)} training trips and {len(test_trips)} student-test trips.")

    # Inspect first 5 training trips for summary statistics
    durations = []
    track_lengths = []
    sampling_gaps = []
    for tid in train_trips[:10]:
        t_path = os.path.join(train_dir, tid)
        track = np.loadtxt(os.path.join(t_path, f"{tid}.track"))
        track_lengths.append(len(track))
        durations.append(track[-1, 2] - track[0, 2])
        dts = np.diff(track[:, 2])
        sampling_gaps.extend(dts)

    sampling_gaps = np.array(sampling_gaps)
    pct_1s = np.mean(np.isclose(sampling_gaps, 1.0, atol=0.1)) * 100.0
    print(f"Trajectory length per trip: Mean = {np.mean(track_lengths):.0f} points, Min = {min(track_lengths)}, Max = {max(track_lengths)}")
    print(f"Duration per trip: Mean = {np.mean(durations):.1f} s ({np.mean(durations)/60:.1f} mins)")
    print(f"Sampling interval: {pct_1s:.2f}% of consecutive GPS observations are exactly ~1.0 second apart.")

    # -------------------------------------------------------------
    # 2. Dataset Preparation & Feature Extraction
    # -------------------------------------------------------------
    print("\n--- Stage 2: Feature Extraction & Spatial Candidate Generation ---")
    # Sample 15 diverse trips from train for robust model training
    train_sample_ids = train_trips[:15]
    print(f"Processing {len(train_sample_ids)} training trips for candidate dataset generation...")

    X_train_list, y_train_list = [], []
    train_trips_data = []

    t0 = time.time()
    for tid in train_sample_ids:
        t_path = os.path.join(train_dir, tid)
        data = load_trip(t_path, tid)
        true_arcs = align_ground_truth_route(data)
        tree, arc_ids = build_spatial_arc_index(data["nodes_m"], data["arcs"])
        cands = generate_candidates_for_track(tree, arc_ids, data["track_m"], max_candidates=10)
        X, y, slices = extract_features_for_trip(data, cands, true_arcs=true_arcs)

        X_train_list.append(X)
        y_train_list.append(y)
        train_trips_data.append({
            "data": data, "true_arcs": true_arcs, "slices": slices, "X": X, "y": y
        })

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"Training dataset ready: {X_train.shape[0]} candidate-observation pairs, {X_train.shape[1]} features.")
    print(f"Class balance: {np.sum(y_train==1)} positive ({np.mean(y_train)*100:.2f}%), {np.sum(y_train==0)} negative.")
    print(f"Extraction time: {time.time()-t0:.2f}s")

    # Prepare Student-Test Dataset (all 10 reserved student test trips)
    print(f"\nProcessing {len(test_trips)} student-test trips for evaluation...")
    test_trips_data = []
    X_test_list, y_test_list = [], []

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
        X_test_list.append(X)
        y_test_list.append(y)

    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    print(f"Student-test dataset ready: {X_test.shape[0]} candidate-observation pairs across {len(test_trips)} trips.")

    # -------------------------------------------------------------
    # 3. Model Training & Comparison Across 5 Algorithm Families
    # -------------------------------------------------------------
    print("\n--- Stage 3: Training & Comparing 5 Distinct Algorithm Families ---")
    model_templates = get_candidate_models()
    comparison_records = []

    trained_models = {}
    for name, entry in model_templates.items():
        print(f"--> Training {name} ({entry['family']})...")
        fitted_model, fit_time = train_single_model(name, entry, X_train, y_train)
        trained_models[name] = fitted_model

        # Cross-validation performance (approximate on train split)
        train_preds = fitted_model.predict(X_train[:15000])
        cv_macro_f1 = evaluate_predictions(y_train[:15000], train_preds)["macro_f1"]

        # Student-test benchmark
        results = benchmark_model_on_trips(fitted_model, test_trips_data, decoder=None)

        record = {
            "Model Name": name,
            "Algorithm Family": entry["family"],
            "Key Configuration": entry["description"],
            "CV Macro F1": round(cv_macro_f1, 4),
            "Student-Test Top-1 Acc (%)": round(results["cls_top1_acc"], 2),
            "Route Continuity (%)": round(results["cls_continuity"], 2),
            "Student-Test Macro F1": round(results["macro_f1"], 4),
            "Log Loss": round(results["log_loss"], 4),
            "Inference Time (ms/obs)": round(results["avg_latency_ms"], 3),
            "Training Time (s)": round(fit_time, 2)
        }
        comparison_records.append(record)
        print(f"    Top-1 Acc: {record['Student-Test Top-1 Acc (%)']}% | Continuity: {record['Route Continuity (%)']}% | Macro F1: {record['Student-Test Macro F1']}")

    comp_df = pd.DataFrame(comparison_records)
    comp_df_path = os.path.join(artifacts_dir, "model_comparison.csv")
    comp_df.to_csv(comp_df_path, index=False)
    print(f"\nModel comparison table saved to: {comp_df_path}")
    print(comp_df[["Model Name", "Algorithm Family", "Student-Test Top-1 Acc (%)", "Route Continuity (%)", "Student-Test Macro F1"]].to_string())

    # -------------------------------------------------------------
    # 4. Bayesian Hyperparameter Optimization
    # -------------------------------------------------------------
    print("\n--- Stage 4: Bayesian Hyperparameter Optimization (HistGradientBoosting) ---")
    print("Initiating Optuna Bayesian search over hyperparameter space...")
    best_tuned_model, best_params, best_score, trials_df = run_bayesian_optimization(
        X_train, y_train, n_trials=12, seed=42
    )

    trials_csv_path = os.path.join(artifacts_dir, "bayesian_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"Bayesian optimization completed. Best CV accuracy: {best_score*100:.2f}%")
    print(f"Optimal hyperparameters: {best_params}")
    print(f"Optimization history saved to: {trials_csv_path}")

    # Compare untuned vs tuned HistGradientBoosting
    untuned_res = benchmark_model_on_trips(trained_models["Hist_Gradient_Boosting"], test_trips_data, decoder=None)
    tuned_res = benchmark_model_on_trips(best_tuned_model, test_trips_data, decoder=None)

    tuned_comparison = pd.DataFrame([
        {
            "Model Version": "Untuned HistGradientBoosting (Default)",
            "Top-1 Accuracy (%)": round(untuned_res["cls_top1_acc"], 2),
            "Route Continuity (%)": round(untuned_res["cls_continuity"], 2),
            "Macro F1": round(untuned_res["macro_f1"], 4),
            "Log Loss": round(untuned_res["log_loss"], 4),
            "Inference Latency (ms)": round(untuned_res["avg_latency_ms"], 3),
        },
        {
            "Model Version": "Tuned HistGradientBoosting (Bayesian Opt)",
            "Top-1 Accuracy (%)": round(tuned_res["cls_top1_acc"], 2),
            "Route Continuity (%)": round(tuned_res["cls_continuity"], 2),
            "Macro F1": round(tuned_res["macro_f1"], 4),
            "Log Loss": round(tuned_res["log_loss"], 4),
            "Inference Latency (ms)": round(tuned_res["avg_latency_ms"], 3),
        }
    ])
    tuned_csv_path = os.path.join(artifacts_dir, "tuned_vs_untuned.csv")
    tuned_comparison.to_csv(tuned_csv_path, index=False)
    print(f"\nTuned vs Untuned comparison:\n{tuned_comparison.to_string()}")

    # Save final best model for FastAPI backend
    final_model_path = os.path.join(artifacts_dir, "best_model.joblib")
    joblib.dump(best_tuned_model, final_model_path)
    print(f"\nFinal model artifact saved to: {final_model_path}")

    # -------------------------------------------------------------
    # 5. HMM + Viterbi Route Reconstruction
    # -------------------------------------------------------------
    print("\n--- Stage 5: HMM & Viterbi Sequential Route Reconstruction ---")
    decoder = HMMRouteDecoder(p_self=1.0, p_connected=0.85, penalty_disconnected=1e-5, beta=15.0)

    # Evaluate across all 10 student-test trips with and without HMM
    hmm_bench_results = benchmark_model_on_trips(best_tuned_model, test_trips_data, decoder=decoder)

    hmm_comp_df = pd.DataFrame([
        {
            "Decoding Strategy": "Classifier-Only (Independent Top-1)",
            "Observation Top-1 Acc (%)": round(hmm_bench_results["cls_top1_acc"], 2),
            "Route Continuity (%)": round(hmm_bench_results["cls_continuity"], 2),
            "Sequence Recovery Jaccard (%)": "N/A (Discontinuous)",
        },
        {
            "Decoding Strategy": "Classifier + HMM (Viterbi Global Decoding)",
            "Observation Top-1 Acc (%)": round(hmm_bench_results["hmm_top1_acc"], 2),
            "Route Continuity (%)": round(hmm_bench_results["hmm_continuity"], 2),
            "Sequence Recovery Jaccard (%)": f"{hmm_bench_results['sequence_recovery']:.2f}%",
        }
    ])
    hmm_csv_path = os.path.join(artifacts_dir, "hmm_evaluation.csv")
    hmm_comp_df.to_csv(hmm_csv_path, index=False)
    print(f"HMM Route Reconstruction Results:\n{hmm_comp_df.to_string()}")

    # -------------------------------------------------------------
    # 6. Folium Maps & Visualizations
    # -------------------------------------------------------------
    print("\n--- Stage 6: Folium Map Generation & Static Figures ---")
    sample_eval_trip = test_trips_data[0]
    trip_id = sample_eval_trip["data"]["trip_id"]

    # Decode route for this trip
    probs = best_tuned_model.predict_proba(sample_eval_trip["X"])[:, 1]
    pred_arcs, cond_route = decoder.decode(sample_eval_trip["slices"], probs, sample_eval_trip["data"])

    # 1. Interactive Folium HTML Map
    folium_path = os.path.join(artifacts_dir, f"folium_trip_{trip_id}.html")
    create_folium_map(sample_eval_trip["data"], pred_arcs, out_html_path=folium_path)
    print(f"Interactive Folium map saved to: {folium_path}")

    # 2. Static Route Comparison PNG for Report
    static_plot_path = os.path.join(artifacts_dir, f"route_reconstruction_{trip_id}.png")
    plot_static_route_comparison(sample_eval_trip["data"], pred_arcs, out_png_path=static_plot_path)
    print(f"Static route comparison figure saved to: {static_plot_path}")

    # 3. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    bars = plt.bar(comp_df["Model Name"], comp_df["Student-Test Top-1 Acc (%)"], color=["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"])
    plt.title("Model Comparison: Observation-Level Top-1 Accuracy on Student-Test Data", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    comp_plot_path = os.path.join(artifacts_dir, "model_comparison.png")
    plt.savefig(comp_plot_path)
    plt.close()
    print(f"Model comparison bar chart saved to: {comp_plot_path}")

    # 4. HMM Improvement Chart
    plt.figure(figsize=(7, 5), dpi=300)
    cats = ["Observation Top-1 Acc", "Route Continuity"]
    cls_vals = [hmm_bench_results["cls_top1_acc"], hmm_bench_results["cls_continuity"]]
    hmm_vals = [hmm_bench_results["hmm_top1_acc"], hmm_bench_results["hmm_continuity"]]
    x = np.arange(len(cats))
    width = 0.35
    plt.bar(x - width/2, cls_vals, width, label="Classifier-Only", color="#f28e2c")
    plt.bar(x + width/2, hmm_vals, width, label="Classifier + HMM", color="#59a14f")
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.title("HMM Viterbi Sequential Reconstruction Impact", fontsize=12, fontweight="bold")
    plt.xticks(x, cats, fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    hmm_plot_path = os.path.join(artifacts_dir, "hmm_improvement.png")
    plt.savefig(hmm_plot_path)
    plt.close()
    print(f"HMM improvement chart saved to: {hmm_plot_path}")

    print("\n=================================================================")
    print("      PATHSNAP MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY   ")
    print("=================================================================")


if __name__ == "__main__":
    main()
