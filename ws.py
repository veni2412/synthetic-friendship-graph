"""
Watts–Strogatz small-world graph generator.

Output format:
    - Returns a dict[int, set[int]] mapping each node to a set of neighbors.
    - Nodes are labeled 0, 1, ..., N-1.
"""

import random
from typing import Dict, Set, Optional


Adjacency = Dict[int, Set[int]]


def generate_watts_strogatz(
    n: int,
    k: int,
    beta: float,
    seed: Optional[int] = None
) -> Adjacency:
    """
    Generate a Watts–Strogatz small-world network.

    Args:
        n: Number of nodes (>= 3), nodes will be 0..n-1.
        k: Each node is initially connected to k/2 neighbors on each side in a ring.
           Must be even and < n.
        beta: Rewiring probability in [0, 1].
        seed: Optional random seed for reproducibility.

    Returns:
        adjacency: dict[node] -> set(neighbors)
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

    # 1. Ring lattice: each node connected to k/2 neighbors on each side
    for i in range(n):
        for j in range(1, half_k + 1):
            neighbor = (i + j) % n
            adjacency[i].add(neighbor)
            adjacency[neighbor].add(i)

    # 2. Rewire edges (only those that were added "forward" to avoid double handling)
    for i in range(n):
        # Original ring neighbors on one side
        targets = [(i + j) % n for j in range(1, half_k + 1)]
        for j in targets:
            # Ensure edge still exists (it might have been rewired before)
            if j not in adjacency[i]:
                continue

            if random.random() < beta:
                # Remove (i, j)
                adjacency[i].remove(j)
                adjacency[j].remove(i)

                # Find a new node k to connect to
                new_node = _find_valid_target(i, adjacency, n)
                if new_node is not None:
                    adjacency[i].add(new_node)
                    adjacency[new_node].add(i)

    return adjacency


def _find_valid_target(i: int, adjacency: Adjacency, n: int) -> Optional[int]:
    """Pick a node != i that is not already connected to i."""
    # Try at most n random attempts before giving up (very unlikely to fail)
    for _ in range(n):
        k = random.randrange(n)
        if k != i and k not in adjacency[i]:
            return k
    return None


if __name__ == "__main__":
    G = generate_watts_strogatz(20, 4, 0.3, seed=42)
    num_edges = sum(len(neigh) for neigh in G.values()) // 2
    print("Generated WS graph with", len(G), "nodes and", num_edges, "edges.")
