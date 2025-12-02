"""
Barabási–Albert scale-free graph generator (preferential attachment).

Output format:
    - Returns a dict[int, set[int]] mapping each node to a set of neighbors.
    - Nodes are labeled 0, 1, ..., n-1.
"""

import random
from typing import Dict, Set, Optional, List


Adjacency = Dict[int, Set[int]]


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = None
) -> Adjacency:
    """
    Generate a Barabási–Albert scale-free network.

    Args:
        n: Total number of nodes (>= m + 1).
        m: Number of edges to attach from each new node to existing nodes (>= 1).
        seed: Optional random seed for reproducibility.

    Returns:
        adjacency: dict[node] -> set(neighbors)
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
    #    list of nodes where each node appears as many times as its degree.
    bag: List[int] = []
    for node in range(m):
        degree = len(adjacency[node])
        bag.extend([node] * degree)

    # 3. Add new nodes one by one, each connecting to m existing nodes chosen
    #    with probability proportional to their degree (via 'bag').
    for new_node in range(m, n):
        adjacency[new_node] = set()

        targets = set()
        while len(targets) < m:
            # If bag is empty (extremely unlikely after the core), fallback to uniform
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


if __name__ == "__main__":
    G = generate_barabasi_albert(20, 2, seed=42)
    num_edges = sum(len(neigh) for neigh in G.values()) // 2
    print("Generated BA graph with", len(G), "nodes and", num_edges, "edges.")
