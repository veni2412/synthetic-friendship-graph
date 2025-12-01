# ===============================================================
# Barabási–Albert Graph + Recommender System (Self-contained)
# ===============================================================

import random
from typing import Dict, Set, Optional, List, Tuple
import math

Adjacency = Dict[int, Set[int]]

# ===============================================================
# 1. BARABÁSI–ALBERT SCALE-FREE GRAPH GENERATOR
# ===============================================================

def generate_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = None
) -> Adjacency:
    """
    Pure Python implementation of the BA model.
    Returns adjacency: dict[node] -> set(neighbors)
    """

    if n <= m:
        raise ValueError("n must be greater than m ≥ 1")

    if seed is not None:
        random.seed(seed)

    adjacency: Adjacency = {i: set() for i in range(n)}

    # Start with fully connected core of size m
    for i in range(m):
        for j in range(i + 1, m):
            adjacency[i].add(j)
            adjacency[j].add(i)

    # Build preferential attachment bag
    bag = []
    for node in range(m):
        deg = len(adjacency[node])
        bag.extend([node] * deg)

    # Add new nodes 1 by 1
    for new in range(m, n):
        targets = set()

        while len(targets) < m:
            if not bag:
                cand = random.randrange(new)
            else:
                cand = random.choice(bag)
            targets.add(cand)

        # Add edges
        for t in targets:
            adjacency[new].add(t)
            adjacency[t].add(new)

        # Update bag
        for t in targets:
            bag.append(t)
        bag.extend([new] * m)

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
    Pure Python Katz implementation, no NumPy.
    Very small-scale version for demonstration.
    """

    n = len(adj)
    # Build adjacency matrix (dense list)
    A = [[0]*n for _ in range(n)]
    for i in adj:
        for j in adj[i]:
            A[i][j] = 1
            A[j][i] = 1

    # Starting term beta*A
    K = [[beta * A[i][j] for j in range(n)] for i in range(n)]

    # Power series sum: beta^k A^k
    Ak = A
    for k in range(2, max_iter+1):
        # Matrix multiply Ak = Ak @ A
        Ak_new = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for t in range(n):
                    s += Ak[i][t] * A[t][j]
                Ak_new[i][j] = s
        Ak = Ak_new

        # Add beta^k * (A^k)
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
    alpha = weight for structural (CN/Jaccard/etc)
    (1-alpha) = weight for popularity (PA)
    """

    all_nodes = set(adj.keys())
    existing_friends = adj[user]
    candidates = all_nodes - existing_friends - {user}

    scores = []

    for v in candidates:
        lp = (
            0.25 * common_neighbors(adj, user, v) +
            0.25 * jaccard(adj, user, v) +
            0.20 * adamic_adar(adj, user, v) +
            0.15 * resource_allocation(adj, user, v) +
            0.15 * katz_score(adj, user, v)
        )

        pa = preferential_attachment(adj, user, v)

        hybrid_score = alpha * lp + (1 - alpha) * (pa / (1 + pa))

        scores.append((v, hybrid_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:10]


# ===============================================================
# 4. DEMO
# ===============================================================

if __name__ == "__main__":
    print("\n=== Generating BA Graph ===")
    adj = generate_barabasi_albert(n=40, m=3, seed=42)

    num_edges = sum(len(adj[u]) for u in adj) // 2
    print("Nodes:", len(adj))
    print("Edges:", num_edges)

    user = 5
    recs = hybrid_recommender(adj, user)

    print(f"\n=== Recommendations for user {user} ===")
    for r, score in recs:
        print(f"  {r:2d}  score={score:.4f}")
