"""
Evaluation Metrics Module for PathSnap.
Calculates Observation-Level Top-1 Accuracy, Route Continuity, Macro F1,
Precision, Recall, Log Loss, and Sequence Recovery.
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from src.hmm import calculate_route_continuity


def evaluate_predictions(y_true, y_probs, threshold=0.5):
    """
    Candidate-level evaluation metrics:
    Accuracy, Precision, Recall, Macro F1, and Log Loss.
    """
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Clipped probabilities for robust log loss
    probs_clipped = np.clip(y_probs, 1e-7, 1.0 - 1e-7)
    try:
        loss = log_loss(y_true, probs_clipped)
    except Exception:
        loss = 0.0

    return {
        "candidate_accuracy": acc,
        "candidate_precision": prec,
        "candidate_recall": rec,
        "macro_f1": macro_f1,
        "log_loss": loss,
    }


def evaluate_trip_observation_level(point_slices, candidate_probs, true_arcs):
    """
    Observation-level Top-1 evaluation (Classifier-Only without HMM):
    Selects candidate with maximum predicted probability for each GPS observation.
    Returns:
        top1_accuracy: percentage of GPS observations where top predicted candidate == true_arc
        pred_arcs: list of top-1 predicted arcs
    """
    n_pts = len(point_slices)
    correct = 0
    pred_arcs = []

    for i in range(n_pts):
        s, e, c_list = point_slices[i]
        if len(c_list) == 0:
            pred_arcs.append(-1)
            continue

        probs = candidate_probs[s:e]
        best_cand = c_list[np.argmax(probs)]
        pred_arcs.append(best_cand)

        if best_cand == true_arcs[i]:
            correct += 1

    top1_acc = (correct / n_pts) * 100.0 if n_pts > 0 else 0.0
    return top1_acc, pred_arcs


def evaluate_sequence_recovery(condensed_pred, reference_route):
    """
    Calculates sequence-level route recovery metric:
    Jaccard overlap between the predicted condensed route arcs and true route arcs.
    """
    if reference_route is None or len(reference_route) == 0:
        return 0.0
    set_pred = set(condensed_pred)
    set_ref = set(reference_route)
    intersection = len(set_pred.intersection(set_ref))
    union = len(set_pred.union(set_ref))
    return (intersection / union) * 100.0 if union > 0 else 0.0


def benchmark_model_on_trips(model, trips_data, decoder=None):
    """
    Evaluates a trained model across a list of loaded trip dictionaries.
    Computes:
        - Observation-level Top-1 Accuracy (Classifier-Only)
        - Route Continuity (Classifier-Only)
        - Observation-level Top-1 Accuracy (Classifier + HMM, if decoder provided)
        - Route Continuity (Classifier + HMM, if decoder provided)
        - Sequence Recovery Jaccard %
        - Candidate-level Macro F1
        - Log Loss
        - Average Inference Latency (ms per observation)
    """
    all_y_true = []
    all_y_probs = []

    cls_top1_list = []
    cls_cont_list = []
    hmm_top1_list = []
    hmm_cont_list = []
    seq_rec_list = []
    latencies = []

    for trip in trips_data:
        X = trip["X"]
        y = trip["y"]
        slices = trip["slices"]
        true_arcs = trip["true_arcs"]
        arcs = trip["data"]["arcs"]
        ref_route = trip["data"]["route"]

        t0 = time.time()
        probs = model.predict_proba(X)[:, 1]
        lat_ms = ((time.time() - t0) / len(slices)) * 1000.0
        latencies.append(lat_ms)

        all_y_true.extend(y)
        all_y_probs.extend(probs)

        # Classifier only
        cls_acc, cls_arcs = evaluate_trip_observation_level(slices, probs, true_arcs)
        cls_cont = calculate_route_continuity(cls_arcs, arcs)
        cls_top1_list.append(cls_acc)
        cls_cont_list.append(cls_cont)

        # HMM decoding
        if decoder is not None:
            hmm_arcs, condensed_route = decoder.decode(slices, probs, trip["data"])
            hmm_acc = (np.mean([hmm_arcs[i] == true_arcs[i] for i in range(len(true_arcs))])) * 100.0
            hmm_cont = calculate_route_continuity(hmm_arcs, arcs)
            hmm_top1_list.append(hmm_acc)
            hmm_cont_list.append(hmm_cont)
            seq_rec = evaluate_sequence_recovery(condensed_route, ref_route)
            seq_rec_list.append(seq_rec)

    cand_metrics = evaluate_predictions(np.array(all_y_true), np.array(all_y_probs))

    results = {
        "cls_top1_acc": float(np.mean(cls_top1_list)),
        "cls_continuity": float(np.mean(cls_cont_list)),
        "macro_f1": float(cand_metrics["macro_f1"]),
        "log_loss": float(cand_metrics["log_loss"]),
        "candidate_acc": float(cand_metrics["candidate_accuracy"]),
        "avg_latency_ms": float(np.mean(latencies)),
    }

    if decoder is not None:
        results["hmm_top1_acc"] = float(np.mean(hmm_top1_list))
        results["hmm_continuity"] = float(np.mean(hmm_cont_list))
        results["sequence_recovery"] = float(np.mean(seq_rec_list))

    return results
