"""
ba_link_prediction_manual.py

Full pipeline:
 - generate Barabási–Albert graph (custom generator)
 - split edges into train/test (temporal split via shuffled edge list)
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
# 1) BA generator (your code)
# ---------------------------

Adjacency = Dict[int, Set[int]]


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = None
) -> Adjacency:
    """
    Generate a Barabási–Albert scale-free network.

    Returns adjacency: dict[node] -> set(neighbors)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if m <= 0:
        raise ValueError("m must be positive")
    if n <= m:
        raise ValueError("n must be greater than m")

    if seed is not None:
        random.seed(seed)

    adjacency: Adjacency = {i: set() for i in range(n)}

    # 1. Start with a small fully connected core of size m
    for i in range(m):
        for j in range(i + 1, m):
            adjacency[i].add(j)
            adjacency[j].add(i)

    # 2. Preferential attachment "bag":
    bag: List[int] = []
    for node in range(m):
        degree = len(adjacency[node])
        bag.extend([node] * degree)

    # 3. Add new nodes one by one, each connecting to m existing nodes chosen
    for new_node in range(m, n):
        adjacency[new_node] = set()

        targets = set()
        while len(targets) < m:
            if not bag:
                cand = random.randrange(new_node)
            else:
                cand = random.choice(bag)
            targets.add(cand)

        for t in targets:
            adjacency[new_node].add(t)
            adjacency[t].add(new_node)

        # Update bag: add each target once and new_node m times
        bag.extend(list(targets))
        bag.extend([new_node] * m)

    return adjacency


# ---------------------------
# 2) Helpers (edges, conversions)
# ---------------------------

def adjacency_to_edge_list(adj: Adjacency) -> List[Tuple[int, int]]:
    """Return sorted unique undirected edges as (u, v) with u < v."""
    edges = set()
    for u, neigh in adj.items():
        for v in neigh:
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return list(edges)


# ---------------------------
# 3) Temporal split (manual)
# ---------------------------

def generate_temporal_ba_graph_manual(n=200, m=3, train_frac=0.7, seed=int(time.time())):
    """Generate full BA graph with custom generator and split edges into train/test."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adj_full = generate_barabasi_albert(n, m, seed)
    edges = adjacency_to_edge_list(adj_full)
    random.shuffle(edges)

    split_idx = int(len(edges) * train_frac)
    Eold = edges[:split_idx]
    Enew = edges[split_idx:]

    # Build training adjacency dict
    adj_train: Adjacency = {i: set() for i in range(n)}
    for u, v in Eold:
        adj_train[u].add(v)
        adj_train[v].add(u)

    return adj_train, adj_full, Eold, Enew


# ---------------------------
# 4) Negative sampling
# ---------------------------

def generate_test_pairs_manual(adj_full: Adjacency, Enew: List[Tuple[int, int]], seed: Optional[int] = None):
    """
    Create positive (Enew) and equal number of negative (non-edge) pairs.
    Avoids using networkx; samples random pairs until enough negatives are found.
    """
    if seed is not None:
        random.seed(seed)

    n = len(adj_full)
    existing = set(adjacency_to_edge_list(adj_full))

    neg_set = set()
    attempts = 0
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
        attempts += 1
        # defensive cutoff (shouldn't be hit in reasonable graphs)
        if attempts > needed * 10000:
            raise RuntimeError("Too many attempts to sample negative edges; graph may be dense.")

    neg = list(neg_set)
    y_true = np.array([1] * len(Enew) + [0] * len(neg))
    pairs = Enew + neg
    return pairs, y_true


# ---------------------------
# 5) Link-prediction heuristics (dict-based)
# ---------------------------

def common_neighbors_score(adj: Adjacency, u: int, v: int) -> float:
    return float(len(adj[u] & adj[v]))


def jaccard_score(adj: Adjacency, u: int, v: int) -> float:
    union = adj[u] | adj[v]
    if not union:
        return 0.0
    return float(len(adj[u] & adj[v]) / len(union))


def adamic_adar_score(adj: Adjacency, u: int, v: int) -> float:
    score = 0.0
    for w in adj[u] & adj[v]:
        deg = len(adj[w])
        if deg > 1:
            score += 1.0 / np.log(deg)
    return float(score)


def resource_allocation_score(adj: Adjacency, u: int, v: int) -> float:
    score = 0.0
    for w in adj[u] & adj[v]:
        deg = len(adj[w])
        if deg > 0:
            score += 1.0 / deg
    return float(score)


def preferential_attachment_score(adj: Adjacency, u: int, v: int) -> float:
    return float(len(adj[u]) * len(adj[v]))


# ---------------------------
# 6) Katz (dense NumPy, no scipy)
# ---------------------------

def katz_scores_matrix_dense(adj: Adjacency, beta: float = 0.005, max_iter: int = 5) -> np.ndarray:
    """
    Compute Katz index K = sum_{k>=1} beta^k A^k truncated at max_iter.
    Uses dense NumPy arrays (no scipy). Returns an (n,n) array.
    """
    n = len(adj)
    A = np.zeros((n, n), dtype=float)

    # adjacency -> dense matrix
    for u, neigh in adj.items():
        for v in neigh:
            A[u, v] = 1.0
            A[v, u] = 1.0  # undirected

    K = beta * A
    Ak = A.copy()
    # start from power 2 .. max_iter
    for k in range(2, max_iter + 1):
        Ak = Ak @ A
        K += (beta ** k) * Ak
    return K


def katz_score(katz_mat: np.ndarray, u: int, v: int) -> float:
    return float(katz_mat[u, v])


# ---------------------------
# 7) Manual ROC-AUC (vectorized fast fallback)
# ---------------------------

def manual_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC AUC using rank comparison:
      AUC = (wins + 0.5 * ties) / (n_pos * n_neg)
    This implementation tries to be efficient by vectorized comparison;
    if arrays are huge, it falls back to pairwise loop.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    n_pos = pos_scores.size
    n_neg = neg_scores.size

    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined when only one class is present.")

    # Try a vectorized approach if sizes allow
    # Comparison matrix size = n_pos * n_neg
    max_allowed = 10_000_000  # limit on comparisons to avoid memory blowups
    if n_pos * n_neg <= max_allowed:
        # broadcast differences
        # shape (n_pos, n_neg)
        diffs = pos_scores[:, None] - neg_scores[None, :]
        wins = np.count_nonzero(diffs > 0)
        ties = np.count_nonzero(diffs == 0)
    else:
        # fallback to loops (memory safe)
        wins = 0
        ties = 0
        for p in pos_scores:
            # vectorized per-positive helps a bit
            cmp = p - neg_scores
            wins += np.count_nonzero(cmp > 0)
            ties += np.count_nonzero(cmp == 0)

    auc = (wins + 0.5 * ties) / (n_pos * n_neg)
    return float(auc)


# ---------------------------
# 8) Evaluation orchestration
# ---------------------------

def evaluate_model(adj_train: Adjacency, pairs: List[Tuple[int, int]], y_true: np.ndarray, method: str, katz_mat: np.ndarray = None) -> float:
    scores = []
    for (u, v) in pairs:
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
            if katz_mat is None:
                raise ValueError("katz_mat required for Katz method")
            scores.append(katz_score(katz_mat, u, v))
        else:
            raise ValueError(f"Unknown method {method}")

    scores = np.array(scores, dtype=float)
    return manual_roc_auc(y_true, scores)


# ---------------------------
# 9) Main experiment
# ---------------------------

if __name__ == "__main__":
    # Parameters (you can change these)
    N = 200
    M = 3
    TRAIN_FRAC = 0.7
    SEED = int(time.time())
    BETA = 0.005
    KATZ_MAX_ITER = 5

    print("\n=== GENERATING TEMPORAL BA GRAPH (MANUAL) ===")
    adj_train, adj_full, Eold, Enew = generate_temporal_ba_graph_manual(n=N, m=M, train_frac=TRAIN_FRAC, seed=SEED)

    print("Training Edges:", len(Eold))
    print("Test Edges:", len(Enew))

    print("\n=== GENERATING TEST PAIRS ===")
    pairs, y_true = generate_test_pairs_manual(adj_full, Enew, seed=SEED)
    print("Total Test Pairs:", len(pairs))

    print("\n=== COMPUTING KATZ MATRIX (DENSE NUMPY) ===")
    katz_mat = katz_scores_matrix_dense(adj_train, beta=BETA, max_iter=KATZ_MAX_ITER)

    print("\n=== EVALUATING MODELS ===")
    methods = ["CN", "Jaccard", "AA", "RA", "PA", "Katz"]
    results = {}
    for method in methods:
        auc = evaluate_model(adj_train, pairs, y_true, method, katz_mat=katz_mat)
        results[method] = auc
        print(f"{method:8s} AUC: {auc:.4f}")

    best_model = max(results, key=results.get)
    print("\n✅ BEST RECOMMENDER ON BA GRAPH:", best_model)
