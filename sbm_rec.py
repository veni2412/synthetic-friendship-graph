# ===============================================================
# SBM (Stochastic Block Model) + Recommender System (Self-contained)
# ===============================================================

import random
import math
from typing import Dict, Set, List, Optional, Tuple

Adjacency = Dict[int, Set[int]]

# ===============================================================
# 1. STOCHASTIC BLOCK MODEL (SBM)
# ===============================================================

def generate_sbm(
    block_sizes: List[int],
    p_matrix: List[List[float]],
    seed: Optional[int] = None
) -> Adjacency:
    """
    GENERAL SBM:
        block_sizes = [n1, n2, ..., nk]
        p_matrix[a][b] = P(edge between block a and block b)
    """
    if seed is not None:
        random.seed(seed)

    if not block_sizes:
        raise ValueError("block_sizes must be non-empty")

    k = len(block_sizes)
    # Validate probability matrix size
    if len(p_matrix) != k or any(len(row) != k for row in p_matrix):
        raise ValueError("p_matrix must be a k x k matrix")

    # Validate probabilities
    for a in range(k):
        for b in range(k):
            if not (0 <= p_matrix[a][b] <= 1):
                raise ValueError(f"p_matrix[{a}][{b}] out of range")

    # Assign each node to a block
    block_of = []
    for b, size in enumerate(block_sizes):
        if size <= 0:
            raise ValueError("block size must be positive")
        block_of.extend([b] * size)

    n = len(block_of)
    adjacency: Adjacency = {i: set() for i in range(n)}

    # Generate edges
    for i in range(n):
        bi = block_of[i]
        for j in range(i+1, n):
            bj = block_of[j]
            if random.random() < p_matrix[bi][bj]:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def generate_sbm_symmetric(
    block_sizes: List[int],
    p_intra: float,
    p_inter: float,
    seed: Optional[int] = None
) -> Adjacency:
    """
    Symmetric SBM wrapper.
    block_sizes = [n1, n2, ..., nk]
    p_intra = P(edge within same block)
    p_inter = P(edge across blocks)
    """
    k = len(block_sizes)
    p_matrix = [
        [
            (p_intra if a == b else p_inter)
            for b in range(k)
        ]
        for a in range(k)
    ]
    return generate_sbm(block_sizes, p_matrix, seed)


# ===============================================================
# 2. LINK-PREDICTION ALGORITHMS (PURE PYTHON)
# ===============================================================

def common_neighbors(adj: Adjacency, u: int, v: int) -> float:
    return len(adj[u].intersection(adj[v]))

def jaccard(adj: Adjacency, u: int, v: int) -> float:
    Au, Av = adj[u], adj[v]
    if not (Au or Av):
        return 0.0
    return len(Au & Av) / len(Au | Av)

def adamic_adar(adj: Adjacency, u: int, v: int) -> float:
    score = 0.0
    for w in adj[u].intersection(adj[v]):
        deg = len(adj[w])
        if deg > 1:
            score += 1.0 / math.log(deg)
    return score

def resource_allocation(adj: Adjacency, u: int, v: int) -> float:
    score = 0.0
    for w in adj[u].intersection(adj[v]):
        deg = len(adj[w])
        if deg > 0:
            score += 1.0 / deg
    return score

def preferential_attachment(adj: Adjacency, u: int, v: int) -> float:
    return len(adj[u]) * len(adj[v])

# -------------------------------
# Katz index (power series approx)
# -------------------------------

def katz_score(adj: Adjacency, u: int, v: int, beta=0.005, max_iter=4) -> float:
    n = len(adj)
    # Build adjacency matrix
    A = [[0]*n for _ in range(n)]
    for i in adj:
        for j in adj[i]:
            A[i][j] = 1
            A[j][i] = 1

    # Start with K = beta*A
    K = [[beta * A[i][j] for j in range(n)] for i in range(n)]
    Ak = A

    for k in range(2, max_iter+1):
        # Compute Ak = Ak @ A
        Ak_new = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for t in range(n):
                    s += Ak[i][t] * A[t][j]
                Ak_new[i][j] = s
        Ak = Ak_new

        # Add beta^k * Ak
        for i in range(n):
            for j in range(n):
                K[i][j] += (beta ** k) * Ak[i][j]

    return K[u][v]


# ===============================================================
# 3. HYBRID RECOMMENDER SYSTEM
# ===============================================================

def hybrid_recommender(
    adj: Adjacency,
    user: int,
    alpha=0.6
) -> List[Tuple[int, float]]:
    """
    Higher-level recommender combining structural + popularity scores.
    alpha controls structural weighting.
    """

    nodes = set(adj.keys())
    existing = adj[user]
    candidates = nodes - existing - {user}

    scores = []

    for v in candidates:
        structural = (
            0.25 * common_neighbors(adj, user, v) +
            0.25 * jaccard(adj, user, v) +
            0.20 * adamic_adar(adj, user, v) +
            0.15 * resource_allocation(adj, user, v) +
            0.15 * katz_score(adj, user, v)
        )

        pa = preferential_attachment(adj, user, v)
        popularity = pa / (1 + pa)

        final_score = alpha * structural + (1 - alpha) * popularity
        scores.append((v, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:10]


# ===============================================================
# 4. DEMO
# ===============================================================

if __name__ == "__main__":
    print("\n=== Generating SBM Graph ===")

    sizes = [30, 30, 40]
    adj = generate_sbm_symmetric(
        sizes,
        p_intra=0.3,
        p_inter=0.05,
        seed=42
    )

    num_edges = sum(len(adj[u]) for u in adj) // 2
    print("Nodes:", len(adj))
    print("Edges:", num_edges)

    user = 12
    recs = hybrid_recommender(adj, user)

    print(f"\n=== Recommendations for user {user} ===")
    for target, score in recs:
        print(f"  {target:2d}  score={score:.4f}")
