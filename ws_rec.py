# ===============================================================
# Watts–Strogatz (WS) Graph + Recommender System (Self-contained)
# ===============================================================

import random
import math
from typing import Dict, Set, Optional, List, Tuple

Adjacency = Dict[int, Set[int]]

# ===============================================================
# 1. WATTS–STROGATZ GRAPH GENERATOR
# ===============================================================

def generate_watts_strogatz(
    n: int,
    k: int,
    beta: float,
    seed: Optional[int] = None
) -> Adjacency:
    """
    Pure Python Watts–Strogatz small-world generator.
    Returns adjacency[node] = set(neighbors)
    """
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

    # -------------------------
    # 1. Create ring lattice
    # -------------------------
    for i in range(n):
        for j in range(1, half_k + 1):
            neighbor = (i + j) % n
            adjacency[i].add(neighbor)
            adjacency[neighbor].add(i)

    # -------------------------
    # 2. Rewire edges with prob β
    # -------------------------
    for i in range(n):
        right_side = [(i + j) % n for j in range(1, half_k + 1)]

        for j in right_side:
            if j not in adjacency[i]:
                continue

            if random.random() < beta:
                # Remove original ring edge
                adjacency[i].remove(j)
                adjacency[j].remove(i)

                # Add new random edge
                new_node = _find_valid_target(i, adjacency, n)
                if new_node is not None:
                    adjacency[i].add(new_node)
                    adjacency[new_node].add(i)

    return adjacency


def _find_valid_target(i: int, adjacency: Adjacency, n: int) -> Optional[int]:
    """Pick a node != i that is not already connected."""
    for _ in range(n):
        k = random.randrange(n)
        if k != i and k not in adjacency[i]:
            return k
    return None


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

# ------------------------------
# Katz score (power series approx)
# ------------------------------

def katz_score(adj: Adjacency, u: int, v: int, beta=0.005, max_iter=4) -> float:
    n = len(adj)
    # adjacency matrix
    A = [[0]*n for _ in range(n)]
    for i in adj:
        for j in adj[i]:
            A[i][j] = 1
            A[j][i] = 1

    # K = beta*A
    K = [[beta * A[i][j] for j in range(n)] for i in range(n)]
    Ak = A

    for k in range(2, max_iter+1):
        # Ak = Ak @ A
        Ak_new = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for t in range(n):
                    s += Ak[i][t] * A[t][j]
                Ak_new[i][j] = s
        Ak = Ak_new

        # add beta^k * Ak
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
    alpha: float = 0.6
) -> List[Tuple[int, float]]:
    """
    Weighted combination of:
        CN, Jaccard, AA, RA, Katz, and PA
    """
    nodes = set(adj.keys())
    existing = adj[user]
    candidates = nodes - existing - {user}

    results = []

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
        results.append((v, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:10]


# ===============================================================
# 4. DEMO
# ===============================================================

if __name__ == "__main__":
    print("\n=== Generating WS Graph ===")
    adj = generate_watts_strogatz(
        n=40, k=6, beta=0.2, seed=42
    )

    num_edges = sum(len(adj[u]) for u in adj) // 2
    print("Nodes:", len(adj))
    print("Edges:", num_edges)

    user = 12
    recs = hybrid_recommender(adj, user)

    print(f"\n=== Recommendations for user {user} ===")
    for tgt, score in recs:
        print(f"  {tgt:2d}  score={score:.4f}")
