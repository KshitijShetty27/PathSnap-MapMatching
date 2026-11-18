"""
Experiment 4: Random Forest (Ensemble / Bagging Model)
Author: Kshama (PES1UG24AM142)
Algorithm Family: Ensemble (Bagging)
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.ensemble import RandomForestClassifier
from experiments.common import get_cached_dataset
from src.evaluation import benchmark_model_on_trips


def main():
    print("==================================================================")
    print("  EXPERIMENT 4: RANDOM FOREST (ENSEMBLE / BAGGING)                ")
    print("==================================================================")

    print("Loading preprocessed dataset...")
    X_train, y_train, test_trips_data = get_cached_dataset()
    print(f"Train samples: {len(X_train)}, Test trips: {len(test_trips_data)}")

    # Model definition with 80 trees
    model = RandomForestClassifier(
        n_estimators=80, max_depth=14, min_samples_split=8, n_jobs=-1, random_state=42
    )

    print("Training Random Forest ensemble model...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f} seconds.")

    print("Evaluating on Student-Test Trips...")
    results = benchmark_model_on_trips(model, test_trips_data, decoder=None)

    print("\n--- Model 4 Results ---")
    print(f"Observation Top-1 Accuracy: {results['cls_top1_acc']:.2f}%")
    print(f"Route Continuity:          {results['cls_continuity']:.2f}%")
    print(f"Candidate Macro F1:        {results['macro_f1']:.4f}")
    print(f"Log Loss:                  {results['log_loss']:.4f}")
    print(f"Avg Inference Latency:     {results['avg_latency_ms']:.3f} ms/obs")
    print("==================================================================")


if __name__ == "__main__":
    main()
