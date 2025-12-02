"""
ws_link_prediction_manual.py

Full pipeline:
 - generate Watts–Strogatz small-world graph (custom generator)
 - split edges into train/test (shuffled temporal split)
 - sample negative (non-)edges for testing
 - compute heuristic link-prediction scores (CN, Jaccard, AA, RA, PA)
 - compute Katz scores via dense NumPy matrix power-series (no scipy)
 - evaluate with manually implemented ROC-AUC (no sklearn)
"""

import random
from typing import Dict, Set, Optional, List, Tuple
import numpy as np
import time

# ---------------------------
# 1) WS Generator (Your Version)
# ---------------------------

Adjacency = Dict[int, Set[int]]

def generate_watts_strogatz(n: int, k: int, beta: float, seed: Optional[int] = None) -> Adjacency:

    if k % 2 != 0:
        raise ValueError("k must be even")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be between 0 and 1")
    if n <= k:
        raise ValueError("n must be greater than k")
    if n <= 0:
        raise ValueError("n must be positive")

    if seed is not None:
        random.seed(seed)

    adjacency: Adjacency = {i: set() for i in range(n)}
    half_k = k // 2

    # 1. Ring lattice
    for i in range(n):
        for j in range(1, half_k + 1):
            neighbor = (i + j) % n
            adjacency[i].add(neighbor)
            adjacency[neighbor].add(i)

    # 2. Rewiring
    for i in range(n):
        targets = [(i + j) % n for j in range(1, half_k + 1)]

        for j in targets:
            if j not in adjacency[i]:
                continue

            if random.random() < beta:
                adjacency[i].remove(j)
                adjacency[j].remove(i)

                new_node = _find_valid_target(i, adjacency, n)
                if new_node is not None:
                    adjacency[i].add(new_node)
                    adjacency[new_node].add(i)

    return adjacency


def _find_valid_target(i: int, adjacency: Adjacency, n: int) -> Optional[int]:
    for _ in range(n):
        k = random.randrange(n)
        if k != i and k not in adjacency[i]:
            return k
    return None


# ---------------------------
# 2) Helpers
# ---------------------------

def adjacency_to_edge_list(adj: Adjacency) -> List[Tuple[int, int]]:
    edges = set()
    for u, neigh in adj.items():
        for v in neigh:
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return list(edges)


# ---------------------------
# 3) Temporal Split
# ---------------------------

def generate_temporal_ws_graph_manual(n=200, k=6, beta=0.3, train_frac=0.7, seed=int(time.time())):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adj_full = generate_watts_strogatz(n, k, beta, seed)
    edges = adjacency_to_edge_list(adj_full)
    random.shuffle(edges)

    split_idx = int(len(edges) * train_frac)
    Eold = edges[:split_idx]
    Enew = edges[split_idx:]

    adj_train: Adjacency = {i: set() for i in range(n)}
    for u, v in Eold:
        adj_train[u].add(v)
        adj_train[v].add(u)

    return adj_train, adj_full, Eold, Enew


# ---------------------------
# 4) Negative Sampling
# ---------------------------

def generate_test_pairs_manual(adj_full: Adjacency, Enew: List[Tuple[int, int]], seed: Optional[int] = None):

    if seed is not None:
        random.seed(seed)

    n = len(adj_full)
    existing = set(adjacency_to_edge_list(adj_full))

    neg_set = set()
    needed = len(Enew)

    while len(neg_set) < needed:
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:
            continue

        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing or (a, b) in neg_set:
            continue

        neg_set.add((a, b))

    neg = list(neg_set)
    y_true = np.array([1] * len(Enew) + [0] * len(neg))
    pairs = Enew + neg

    return pairs, y_true


# ---------------------------
# 5) Heuristic Scores
# ---------------------------

def common_neighbors_score(adj, u, v):
    return float(len(adj[u] & adj[v]))

def jaccard_score(adj, u, v):
    union = adj[u] | adj[v]
    if not union:
        return 0.0
    return float(len(adj[u] & adj[v]) / len(union))

def adamic_adar_score(adj, u, v):
    score = 0.0
    for w in adj[u] & adj[v]:
        deg = len(adj[w])
        if deg > 1:
            score += 1.0 / np.log(deg)
    return float(score)

def resource_allocation_score(adj, u, v):
    score = 0.0
    for w in adj[u] & adj[v]:
        deg = len(adj[w])
        if deg > 0:
            score += 1.0 / deg
    return float(score)

def preferential_attachment_score(adj, u, v):
    return float(len(adj[u]) * len(adj[v]))


# ---------------------------
# 6) Katz (Dense NumPy)
# ---------------------------

def katz_scores_matrix_dense(adj: Adjacency, beta=0.005, max_iter=5):

    n = len(adj)
    A = np.zeros((n, n), dtype=float)

    for u, neigh in adj.items():
        for v in neigh:
            A[u, v] = 1.0
            A[v, u] = 1.0

    K = beta * A
    Ak = A.copy()

    for k in range(2, max_iter + 1):
        Ak = Ak @ A
        K += (beta ** k) * Ak

    return K

def katz_score(katz_mat, u, v):
    return float(katz_mat[u, v])


# ---------------------------
# 7) Manual ROC-AUC
# ---------------------------

def manual_roc_auc(y_true, scores):

    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    wins = 0
    ties = 0

    for p in pos_scores:
        cmp = p - neg_scores
        wins += np.count_nonzero(cmp > 0)
        ties += np.count_nonzero(cmp == 0)

    auc = (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))
    return float(auc)


# ---------------------------
# 8) Evaluation
# ---------------------------

def evaluate_model(adj_train, pairs, y_true, method, katz_mat=None):

    scores = []

    for u, v in pairs:
        if method == "CN":
            scores.append(common_neighbors_score(adj_train, u, v))
        elif method == "Jaccard":
            scores.append(jaccard_score(adj_train, u, v))
        elif method == "AA":
            scores.append(adamic_adar_score(adj_train, u, v))
        elif method == "RA":
            scores.append(resource_allocation_score(adj_train, u, v))
        elif method == "PA":
            scores.append(preferential_attachment_score(adj_train, u, v))
        elif method == "Katz":
            scores.append(katz_score(katz_mat, u, v))

    scores = np.array(scores)
    return manual_roc_auc(y_true, scores)


# ---------------------------
# 9) Main Experiment
# ---------------------------

if __name__ == "__main__":

    N = 200
    K = 6
    BETA_WS = 0.3
    TRAIN_FRAC = 0.7
    SEED = int(time.time())

    BETA_KATZ = 0.005
    KATZ_MAX_ITER = 5

    print("\n=== GENERATING TEMPORAL WS GRAPH (MANUAL) ===")

    adj_train, adj_full, Eold, Enew = generate_temporal_ws_graph_manual(
        n=N, k=K, beta=BETA_WS, train_frac=TRAIN_FRAC, seed=SEED
    )

    print("Training Edges:", len(Eold))
    print("Test Edges:", len(Enew))

    print("\n=== GENERATING TEST PAIRS ===")
    pairs, y_true = generate_test_pairs_manual(adj_full, Enew, seed=SEED)
    print("Total Test Pairs:", len(pairs))

    print("\n=== COMPUTING KATZ MATRIX (DENSE NUMPY) ===")
    katz_mat = katz_scores_matrix_dense(adj_train, beta=BETA_KATZ, max_iter=KATZ_MAX_ITER)

    print("\n=== EVALUATING MODELS ===")

    methods = ["CN", "Jaccard", "AA", "RA", "PA", "Katz"]
    results = {}

    for method in methods:
        auc = evaluate_model(adj_train, pairs, y_true, method, katz_mat=katz_mat)
        results[method] = auc
        print(f"{method:8s} AUC: {auc:.4f}")

    best_model = max(results, key=results.get)
    print("\n✅ BEST RECOMMENDER ON WS GRAPH:", best_model)
