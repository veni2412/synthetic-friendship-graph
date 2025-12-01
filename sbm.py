# sbm.py
"""
Stochastic Block Model (SBM) graph generator.

Output format (uniform with ER / WS / BA):
    - Returns a dict[int, set[int]] mapping each node to a set of neighbors.
    - Nodes are labeled 0, 1, ..., N-1.

Two interfaces:
    1) generate_sbm(block_sizes, p_matrix, seed=None)
       - Fully general k x k probability matrix.

    2) generate_sbm_symmetric(block_sizes, p_intra, p_inter, seed=None)
       - Convenience wrapper with p_intra for same-block edges and
         p_inter for different-block edges.
"""

import random
from typing import Dict, Set, List, Optional


Adjacency = Dict[int, Set[int]]


def generate_sbm(
    block_sizes: List[int],
    p_matrix: List[List[float]],
    seed: Optional[int] = None
) -> Adjacency:
    """
    Generate an undirected Stochastic Block Model (SBM) graph.

    Args:
        block_sizes: List of sizes for each block, e.g. [50, 50, 100]
                     for k=3 communities.
        p_matrix:   k x k matrix of edge probabilities between blocks.
                    p_matrix[a][b] is the probability of an edge between
                    a node in block a and a node in block b.
                    Should be symmetric for an undirected graph.
        seed:       Optional random seed for reproducibility.

    Returns:
        adjacency: dict[node] -> set(neighbors)
    """
    if seed is not None:
        random.seed(seed)

    if not block_sizes:
        raise ValueError("block_sizes must be a non-empty list")

    k = len(block_sizes)
    if len(p_matrix) != k or any(len(row) != k for row in p_matrix):
        raise ValueError("p_matrix must be a k x k matrix where k = len(block_sizes)")

    # Check probabilities are in [0, 1]
    for a in range(k):
        for b in range(k):
            p = p_matrix[a][b]
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"p_matrix[{a}][{b}] = {p} is not in [0, 1]")

    # Assign nodes to blocks
    # Example:
    #   block 0: nodes 0 .. size0-1
    #   block 1: nodes size0 .. size0+size1-1
    #   ...
    block_of: List[int] = []   # block_of[node] = block index
    start = 0
    for block_index, size in enumerate(block_sizes):
        if size <= 0:
            raise ValueError("Block sizes must be positive")
        for _ in range(size):
            block_of.append(block_index)
        start += size

    n = len(block_of)
    adjacency: Adjacency = {i: set() for i in range(n)}

    # Generate edges according to p_matrix[block_of[i]][block_of[j]]
    for i in range(n):
        bi = block_of[i]
        for j in range(i + 1, n):
            bj = block_of[j]
            p = p_matrix[bi][bj]
            if random.random() < p:
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
    Convenience wrapper for a symmetric SBM with:
        - p_intra: edge probability within the same block
        - p_inter: edge probability between different blocks

    Args:
        block_sizes: List of sizes for each block, e.g. [50, 50, 100]
        p_intra:     Probability for edges within same block.
        p_inter:     Probability for edges between different blocks.
        seed:        Optional random seed.

    Returns:
        adjacency: dict[node] -> set(neighbors)
    """
    k = len(block_sizes)
    if k == 0:
        raise ValueError("block_sizes must be non-empty")

    # Build k x k matrix with p_intra on diagonal, p_inter off-diagonal
    p_matrix = [
        [
            (p_intra if a == b else p_inter)
            for b in range(k)
        ]
        for a in range(k)
    ]

    return generate_sbm(block_sizes, p_matrix, seed=seed)


if __name__ == "__main__":
    # Tiny smoke test (not analysis; just to be sure it runs)
    sizes = [30, 30, 40]
    G = generate_sbm_symmetric(sizes, p_intra=0.3, p_inter=0.05, seed=42)
    num_edges = sum(len(neigh) for neigh in G.values()) // 2
    print("Generated SBM graph with", len(G), "nodes and", num_edges, "edges.")
