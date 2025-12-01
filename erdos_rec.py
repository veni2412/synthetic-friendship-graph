# ===============================================================
# Erdős–Rényi G(n,p) Graph + Recommender System (Self-contained)
# ===============================================================

import random
from typing import Dict, Set, Optional, List, Tuple
import math

Adjacency = Dict[int, Set[int]]

# ===============================================================
# 1. ERDŐS–RÉNYI GRAPH GENERATOR (G(n,p))
# ===============================================================

def generate_erdos_renyi(
    n: int,
    p: float,
    seed: Optional[int] = None
) -> Adjacency:
    """
    Pure Python implementation of ER G(n,p).
    Returns adjacency: dict[node] -> set(neighbors)
    """

    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")

    if seed is not None:
        random.seed(seed)

    adjacency: Adjacency = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


# ===============================================================
# 2. LINK-PREDICTION ALGORITHMS (PURE PYTHON)
# ===============================================================

def common_neighbors(adj: Adjacency, u: int, v: int) -> float:
    return len(adj[u].intersection(adj[v]))

def jaccard(adj: Adjacency, u: int, v: int) -> float:
    A = adj[u]
    B = adj[v]
    if not (A or B):
        return 0.0
    return len(A & B) / len(A | B)

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
    """
    Pure Python Katz implementation (dense adjacency).
    Avoid for large n > 200.
    """

    n = len(adj)

    # Build adjacency matrix
    A = [[0]*n for _ in range(n)]
    for i in adj:
        for j in adj[i]:
            A[i][j] = 1
            A[j][i] = 1

    # K = beta*A
    K = [[beta * A[i][j] for j in range(n)] for i in range(n)]

    # Ak = A
    Ak = A

    for k in range(2, max_iter + 1):

        # Compute A^k = Ak @ A
        Ak_new = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for t in range(n):
                    s += Ak[i][t] * A[t][j]
                Ak_new[i][j] = s
        Ak = Ak_new

        # Add beta^k * A^k to K
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
    Combine multiple link-prediction algorithms into one score.
    alpha = weight for structure-based similarity
    (1-alpha) = weight for degree-based popularity
    """

    nodes = set(adj.keys())
    existing = adj[user]
    candidates = nodes - existing - {user}

    results = []

    for v in candidates:

        # Structural LP score
        lp = (
            0.25 * common_neighbors(adj, user, v) +
            0.25 * jaccard(adj, user, v) +
            0.20 * adamic_adar(adj, user, v) +
            0.15 * resource_allocation(adj, user, v) +
            0.15 * katz_score(adj, user, v)
        )

        # Normalized popularity component
        pa = preferential_attachment(adj, user, v)
        popularity = pa / (1 + pa)

        hybrid_score = alpha * lp + (1 - alpha) * popularity

        results.append((v, hybrid_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:10]


# ===============================================================
# 4. DEMO
# ===============================================================

if __name__ == "__main__":
    print("\n=== Generating ER Graph ===")
    adj = generate_erdos_renyi(n=40, p=0.15, seed=42)

    num_edges = sum(len(adj[u]) for u in adj) // 2
    print("Nodes:", len(adj))
    print("Edges:", num_edges)

    user = 7
    recs = hybrid_recommender(adj, user)

    print(f"\n=== Recommendations for user {user} ===")
    for r, score in recs:
        print(f"  {r:2d}  score={score:.4f}")
