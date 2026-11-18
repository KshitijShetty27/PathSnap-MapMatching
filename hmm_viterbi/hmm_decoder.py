"""
Hidden Markov Model (HMM) and Viterbi Decoding Module for PathSnap.
Performs sequential route reconstruction and enforces topological road network continuity.
"""

import math
import numpy as np


class HMMRouteDecoder:
    """
    HMM Decoder using Viterbi algorithm for map-matching.
    Hidden states at step t: candidate road segments C_t.
    Emission: normalized classifier probabilities.
    Transition: topological graph connectivity and spatial distance differentials.
    """

    def __init__(self, p_self=1.0, p_connected=0.85, penalty_disconnected=1e-5, beta=15.0):
        self.p_self = p_self
        self.p_connected = p_connected
        self.penalty_disconnected = penalty_disconnected
        self.beta = beta

        # Precompute log probabilities
        self.log_p_self = math.log(self.p_self)
        self.log_p_connected = math.log(self.p_connected)
        self.log_penalty_disconnected = math.log(self.penalty_disconnected)

    def decode(self, point_slices, candidate_probs, trip_data):
        """
        Runs Viterbi decoding over the candidate road segments for the trip.
        point_slices: list of (start_idx, end_idx, candidate_list)
        candidate_probs: ndarray of shape (total_candidates,) from ML model predict_proba
        trip_data: dictionary from load_trip containing 'arcs' and 'nodes_m', 'track_m'
        Returns:
            predicted_arcs: list of arc IDs (one per GPS observation)
            condensed_route: list of unique sequential arcs traversed
        """
        arcs = trip_data["arcs"]
        arc_u = arcs[:, 0]
        arc_v = arcs[:, 1]
        track_m = trip_data["track_m"]

        n_steps = len(point_slices)
        if n_steps == 0:
            return [], []

        V = []
        backptr = []

        # Step 0: Initialize with emissions
        s0, e0, c0 = point_slices[0]
        if len(c0) == 0:
            # Fallback for empty candidate
            V.append(np.array([0.0]))
        else:
            probs_0 = candidate_probs[s0:e0]
            # Softmax / normalize emissions
            p_sum = np.sum(probs_0)
            if p_sum > 0:
                p_norm = probs_0 / p_sum
            else:
                p_norm = np.ones(len(c0)) / len(c0)
            V.append(np.log(np.maximum(p_norm, 1e-9)))

        # Recursion steps 1 to n_steps - 1
        for t in range(1, n_steps):
            st, et, ct = point_slices[t]
            prev_slice = point_slices[t - 1]
            prev_c = prev_slice[2]
            prev_v = V[t - 1]

            n_curr = len(ct)
            n_prev = len(prev_c)

            if n_curr == 0:
                V.append(np.array([0.0]))
                backptr.append(np.zeros(1, dtype=int))
                continue

            if n_prev == 0:
                # No previous candidates, just take emissions
                probs_t = candidate_probs[st:et]
                p_sum = np.sum(probs_t)
                p_norm = probs_t / p_sum if p_sum > 0 else np.ones(n_curr) / n_curr
                V.append(np.log(np.maximum(p_norm, 1e-9)))
                backptr.append(np.zeros(n_curr, dtype=int))
                continue

            # Candidate emissions
            probs_t = candidate_probs[st:et]
            p_sum = np.sum(probs_t)
            p_norm = probs_t / p_sum if p_sum > 0 else np.ones(n_curr) / n_curr
            log_emissions = np.log(np.maximum(p_norm, 1e-9))

            # GPS observation displacement
            d_gps = math.hypot(track_m[t, 0] - track_m[t - 1, 0], track_m[t, 1] - track_m[t - 1, 1])

            v_curr = np.empty(n_curr, dtype=np.float64)
            bp_curr = np.empty(n_curr, dtype=int)

            for j, c_next in enumerate(ct):
                best_val = -1e12
                best_i = 0
                next_u = arc_u[c_next]

                for i, c_prev in enumerate(prev_c):
                    if c_prev == c_next:
                        # Vehicle stayed on same road segment
                        log_t = self.log_p_self
                    elif arc_v[c_prev] == next_u:
                        # Direct graph transition: connected arcs
                        log_t = self.log_p_connected
                    else:
                        # Disconnected jump
                        log_t = self.log_penalty_disconnected - (d_gps / self.beta)

                    score = prev_v[i] + log_t
                    if score > best_val:
                        best_val = score
                        best_i = i

                v_curr[j] = best_val + log_emissions[j]
                bp_curr[j] = best_i

            V.append(v_curr)
            backptr.append(bp_curr)

        # Backtrack to find most likely path
        best_end = int(np.argmax(V[-1]))
        opt_path = [best_end]

        for t in range(n_steps - 1, 0, -1):
            if len(backptr[t - 1]) == 0:
                opt_path.append(0)
            else:
                prev_idx = backptr[t - 1][opt_path[-1]]
                opt_path.append(prev_idx)

        opt_path.reverse()

        predicted_arcs = []
        for t in range(n_steps):
            c_list = point_slices[t][2]
            if len(c_list) > 0 and opt_path[t] < len(c_list):
                predicted_arcs.append(c_list[opt_path[t]])
            elif len(c_list) > 0:
                predicted_arcs.append(c_list[0])
            else:
                predicted_arcs.append(-1)

        # Condense consecutive duplicates into sequential route arcs
        condensed_route = []
        for aid in predicted_arcs:
            if aid != -1 and (not condensed_route or condensed_route[-1] != aid):
                condensed_route.append(aid)

        return predicted_arcs, condensed_route


def calculate_route_continuity(arc_sequence, arcs):
    """
    Calculates route continuity score as percentage of consecutive arc pairs
    that are identical or directly connected in the road network graph.
    """
    if len(arc_sequence) <= 1:
        return 100.0

    arc_u = arcs[:, 0]
    arc_v = arcs[:, 1]

    valid_transitions = 0
    total_transitions = len(arc_sequence) - 1

    for i in range(total_transitions):
        a1, a2 = arc_sequence[i], arc_sequence[i + 1]
        if a1 == -1 or a2 == -1:
            continue
        if a1 == a2 or arc_v[a1] == arc_u[a2]:
            valid_transitions += 1

    return (valid_transitions / total_transitions) * 100.0
