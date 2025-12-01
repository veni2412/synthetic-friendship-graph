# erdos.py
"""
Erdős–Rényi G(n, p) random graph generator.

Output format (uniform across all generators):
    - Returns a dict[int, set[int]] mapping each node to a set of neighbors.
    - Nodes are labeled 0, 1, ..., n-1.
"""

import random
from typing import Dict, Set, Optional


Adjacency = Dict[int, Set[int]]


def generate_erdos_renyi(n: int, p: float, seed: Optional[int] = None) -> Adjacency:
    """
    Generate an undirected Erdős–Rényi G(n, p) graph.

    Args:
        n: Number of nodes (>= 1), nodes will be 0..n-1.
        p: Edge probability in [0, 1].
        seed: Optional random seed for reproducibility.

    Returns:
        adjacency: dict[node] -> set(neighbors)
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")

    if n <= 0:
        raise ValueError("n must be positive")

    if seed is not None:
        random.seed(seed)

    adjacency: Adjacency = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


if __name__ == "__main__":
    # Tiny smoke test (not analysis: just sanity-check)
    G = generate_erdos_renyi(10, 0.3, seed=42)
    num_edges = sum(len(neigh) for neigh in G.values()) // 2
    print("Generated ER graph with", len(G), "nodes and", num_edges, "edges.")
