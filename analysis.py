"""
Analysis pipeline for synthetic graphs.

All algorithms are implemented from scratch and operate on a uniform
graph representation:

    adjacency: dict[int, set[int]]

The analysis functions do NOT care about how the graph was generated
(ER, BA, WS, SBM, etc.).
"""

import argparse
from collections import deque, defaultdict, Counter
from typing import Dict, Set, List, Tuple, Optional
import csv
import math
import random

from erdos import generate_erdos_renyi
from ba import generate_barabasi_albert
from ws import generate_watts_strogatz
from sbm import generate_sbm_symmetric


Adjacency = Dict[int, Set[int]]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def num_nodes(adj: Adjacency) -> int:
    return len(adj)


def num_edges(adj: Adjacency) -> int:
    return sum(len(neigh) for neigh in adj.values()) // 2


# ---------------------------------------------------------------------
# Connectivity (BFS-based) + Connected Components (DFS-like)
# ---------------------------------------------------------------------

def bfs_component(adj: Adjacency, start: int, visited: Set[int]) -> List[int]:
    q = deque([start])
    visited.add(start)
    comp = []

    while q:
        u = q.popleft()
        comp.append(u)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)

    return comp


def compute_connectivity(adj: Adjacency) -> Dict[str, object]:
    """
    Compute connected components and basic connectivity stats.

    Returns:
        {
            "num_components": int,
            "component_sizes": List[int] (sorted desc),
            "giant_component_size": int,
            "isolated_nodes": int,
        }
    """
    visited: Set[int] = set()
    components: List[List[int]] = []

    for node in adj.keys():
        if node not in visited:
            comp = bfs_component(adj, node, visited)
            components.append(comp)

    sizes = sorted((len(c) for c in components), reverse=True)
    giant = sizes[0] if sizes else 0
    isolated = sum(1 for c in components if len(c) == 1)

    return {
        "num_components": len(components),
        "component_sizes": sizes,
        "giant_component_size": giant,
        "isolated_nodes": isolated,
    }


def connected_components(adj: Adjacency) -> List[List[int]]:
    """
    DFS-style connected components (functionality matching traversal.py).

    Returns:
        List of components, each = list of node ids.
    """
    visited: Set[int] = set()
    components: List[List[int]] = []

    for start in adj.keys():
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        comp: List[int] = []

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)

        components.append(comp)

    return components


# ---------------------------------------------------------------------
# Centrality measures
# ---------------------------------------------------------------------

def degree_centrality(adj: Adjacency) -> Dict[int, float]:
    n = num_nodes(adj)
    if n <= 1:
        return {u: 0.0 for u in adj}

    return {u: len(adj[u]) / (n - 1) for u in adj}


def bfs_shortest_paths(adj: Adjacency, src: int) -> Dict[int, int]:
    dist = {src: 0}
    q = deque([src])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist


def closeness_centrality(adj: Adjacency) -> Dict[int, float]:
    closeness: Dict[int, float] = {}
    n = num_nodes(adj)

    for u in adj.keys():
        dist = bfs_shortest_paths(adj, u)
        if len(dist) <= 1:
            closeness[u] = 0.0
        else:
            total = sum(dist.values())
            # Standard definition for possibly disconnected graphs:
            # (reachable_nodes - 1) / sum(distances)
            closeness[u] = (len(dist) - 1) / total if total > 0 else 0.0

    return closeness


def betweenness_centrality(adj: Adjacency) -> Dict[int, float]:
    """
    Brandes' algorithm for betweenness centrality on an undirected graph.
    Implemented from scratch, using only the adjacency dict.
    """
    nodes = list(adj.keys())
    bc = {v: 0.0 for v in nodes}

    for s in nodes:
        stack: List[int] = []
        pred: Dict[int, List[int]] = {v: [] for v in nodes}
        sigma: Dict[int, float] = {v: 0.0 for v in nodes}
        dist: Dict[int, int] = {v: -1 for v in nodes}

        sigma[s] = 1.0
        dist[s] = 0

        q = deque([s])

        # Single-source shortest paths
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:  # w found for the first time
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Accumulation
        delta: Dict[int, float] = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0.0:
                    delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    delta[v] += delta_v
            if w != s:
                bc[w] += delta[w]

    # Normalize for undirected graphs (divide by 2)
    for v in bc:
        bc[v] *= 0.5

    return bc


def clustering_coefficient(adj: Adjacency) -> Dict[int, float]:
    """
    Local clustering coefficient for each node:
        C(u) = (number of edges among neighbors of u) / (k_u * (k_u - 1) / 2)
    """
    coeff: Dict[int, float] = {}
    for u, neighbors in adj.items():
        k = len(neighbors)
        if k < 2:
            coeff[u] = 0.0
            continue

        # Count edges between neighbors of u
        edges_between = 0
        neigh_list = list(neighbors)
        for i in range(len(neigh_list)):
            v = neigh_list[i]
            for j in range(i + 1, len(neigh_list)):
                w = neigh_list[j]
                if w in adj[v]:
                    edges_between += 1

        possible = k * (k - 1) / 2
        coeff[u] = edges_between / possible if possible > 0 else 0.0

    return coeff


def _minmax_normalize(values: Dict[int, float]) -> Dict[int, float]:
    """
    Min–max normalize a dict of node -> value into [0, 1].
    If all values are equal, everyone gets 0.5.
    """
    if not values:
        return {}

    vs = list(values.values())
    vmin, vmax = min(vs), max(vs)
    if vmax == vmin:
        return {u: 0.5 for u in values}

    return {u: (v - vmin) / (vmax - vmin) for u, v in values.items()}


def _zscore_normalize(values: Dict[int, float]) -> Dict[int, float]:
    """
    Z-score normalize a dict of node -> value and squash to [0, 1]
    via logistic sigmoid: norm = 1 / (1 + exp(-z)). If std == 0,
    return 0.5 for all.
    """
    if not values:
        return {}
    vs = list(values.values())
    mean = sum(vs) / len(vs)
    var = sum((v - mean) ** 2 for v in vs) / len(vs)
    std = var ** 0.5
    if std == 0:
        return {u: 0.5 for u in values}
    import math
    return {u: 1.0 / (1.0 + math.exp(-(v - mean) / std)) for u, v in values.items()}


# ---------------------------------------------------------------------
# PageRank (from scratch)
# ---------------------------------------------------------------------

def pagerank(
    adj: Adjacency,
    d: float = 0.85,
    iterations: int = 40
) -> Dict[int, float]:
    """
    Simple PageRank implementation from scratch.

    Args:
        d: Damping factor.
        iterations: Number of power iterations.
    """
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}

    pr = {u: 1.0 / n for u in nodes}
    degrees = {u: len(adj[u]) for u in nodes}

    for _ in range(iterations):
        new_pr = {u: (1.0 - d) / n for u in nodes}

        # Basic version: ignores dangling-node redistribution
        for u in nodes:
            deg_u = degrees[u]
            if deg_u == 0:
                continue
            share = d * pr[u] / deg_u
            for v in adj[u]:
                new_pr[v] += share

        pr = new_pr

    return pr


# ---------------------------------------------------------------------
# Personality mapping (Big Five-style)
# ---------------------------------------------------------------------

def compute_big_five_personality(
    adj: Adjacency,
    deg_cent: Dict[int, float],
    close_cent: Dict[int, float],
    bet_cent: Dict[int, float],
    noise_sigma: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    use_zscore: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Compute Big Five-like personality traits for each node based on graph structure.

    Mapping (all traits normalized to [0, 1]):

        E (Extraversion)      ~ degree centrality
        O (Openness)          ~ betweenness centrality
        A (Agreeableness)     ~ clustering coefficient
        C (Conscientiousness) ~ closeness centrality
        N (Neuroticism)       ~ inverse of (degree + clustering)
    """
    # additional local structure: clustering
    clust = clustering_coefficient(adj)

    # normalize all metrics into [0, 1]
    if use_zscore:
        deg_n   = _zscore_normalize(deg_cent)
        bet_n   = _zscore_normalize(bet_cent)
        close_n = _zscore_normalize(close_cent)
        clust_n = _zscore_normalize(clust)
    else:
        deg_n   = _minmax_normalize(deg_cent)
        bet_n   = _minmax_normalize(bet_cent)
        close_n = _minmax_normalize(close_cent)
        clust_n = _minmax_normalize(clust)

    # default weights for N: balanced contribution from degree, clustering, closeness
    # users can override by passing weights={"deg": ..., "clust": ..., "close": ...}
    w = {"deg": 0.33, "clust": 0.33, "close": 0.34}
    if weights:
        w.update({k: v for k, v in weights.items() if k in w})

    personalities: Dict[int, Dict[str, float]] = {}

    # fixed seed for reproducibility (optional)
    random.seed(0)

    for u in adj.keys():
        # base (deterministic) structural scores
        E = deg_n[u]             # extraversion
        O = bet_n[u]             # openness
        A = clust_n[u]           # agreeableness
        C = close_n[u]           # conscientiousness
        # Reweighted Neuroticism: inverse of weighted combination of degree, clustering, closeness
        inv_sum = w["deg"] * deg_n[u] + w["clust"] * clust_n[u] + w["close"] * close_n[u]
        N = 1.0 - inv_sum

        def noisy(x: float) -> float:
            if noise_sigma <= 0.0:
                return max(0.0, min(1.0, x))
            return max(0.0, min(1.0, x + random.gauss(0.0, noise_sigma)))

        personalities[u] = {
            "E": noisy(E),
            "O": noisy(O),
            "A": noisy(A),
            "C": noisy(C),
            "N": noisy(N),
        }

    return personalities


# ---------------------------------------------------------------------
# Community detection (greedy modularity)
# ---------------------------------------------------------------------

def compute_modularity(adj: Adjacency, communities: Dict[int, int]) -> float:
    """
    Compute modularity Q = Σ_c (e_cc - a_c^2)

    e_cc: fraction of edges inside community c
    a_c : fraction of edge endpoints incident to nodes in community c

    `communities` is a dict: node -> community_id
    """
    m = num_edges(adj)
    if m == 0:
        return 0.0

    degrees = {u: len(adj[u]) for u in adj}
    e = defaultdict(int)   # edges inside each community
    a = defaultdict(float) # sum of degrees in each community / (2m)

    for u in adj:
        cu = communities[u]
        a[cu] += degrees[u] / (2.0 * m)
        for v in adj[u]:
            if u < v:  # count each edge once
                cv = communities[v]
                if cu == cv:
                    e[cu] += 1

    Q = 0.0
    for c in e:
        Q += (e[c] / m) - (a[c] ** 2)
    return Q


def community_detection(adj: Adjacency) -> List[List[int]]:
    """
    Very simple greedy modularity optimization:
        - Start with each node in its own community.
        - Repeatedly move nodes to neighboring communities if modularity improves.
    """
    nodes = list(adj.keys())
    # Initially, each node is its own community
    communities = {u: u for u in nodes}

    improved = True
    while improved:
        improved = False
        current_mod = compute_modularity(adj, communities)

        for u in nodes:
            best_comm = communities[u]
            best_mod = current_mod

            # Candidate communities: communities of neighbors
            neighbor_comms = {communities[v] for v in adj[u]}
            for c in neighbor_comms:
                if c == communities[u]:
                    continue
                old_comm = communities[u]
                communities[u] = c
                new_mod = compute_modularity(adj, communities)
                if new_mod > best_mod:
                    best_comm = c
                    best_mod = new_mod
                communities[u] = old_comm

            if best_comm != communities[u]:
                communities[u] = best_comm
                current_mod = best_mod
                improved = True

    # Group nodes by community id
    groups = defaultdict(list)
    for node, c in communities.items():
        groups[c].append(node)

    return list(groups.values())


# ---------------------------------------------------------------------
# Union-Find (Disjoint Set) + Louvain-style one-pass
# ---------------------------------------------------------------------

class UnionFind:
    """
    Simple Union-Find / Disjoint Set structure (functionality from community.py).

    Works with arbitrary hashable elements (not just 0..n-1).
    """
    def __init__(self, elements: List[int]):
        self.parent = {x: x for x in elements}
        self.rank   = {x: 0 for x in elements}

    def find(self, x: int) -> int:
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1


def louvain_one_pass(adj: Adjacency) -> List[List[int]]:
    """
    A simple Louvain-like local moving algorithm (one iteration)
    adapted from community.py to work with generic node labels.

    Steps:
        - Start with each node in its own community.
        - For each node (in random order), consider moving it to
          the communities of its neighbors if that locally improves
          modularity (approximate ΔQ rule).
        - Return resulting communities as list[list[node]].
    """
    nodes = list(adj.keys())
    if not nodes:
        return []

    community: Dict[int, int] = {u: u for u in nodes}
    degrees: Dict[int, int]   = {u: len(adj[u]) for u in nodes}
    m = sum(degrees.values()) / 2.0

    # Precompute neighbor lists
    neighbors = {u: list(adj[u]) for u in nodes}

    random_nodes = nodes[:]
    random.shuffle(random_nodes)

    for u in random_nodes:
        current_comm = community[u]
        best_comm = current_comm
        best_deltaQ = 0.0

        # Candidate: communities of neighbors
        neighbor_comms = {community[v] for v in neighbors[u]}

        for target_c in neighbor_comms:
            if target_c == current_comm:
                continue

            # k_i,in = number of edges u has inside target community
            k_i_in = sum(1 for w in neighbors[u] if community[w] == target_c)

            # sum_tot = total degree of target community
            sum_tot = sum(degrees[w] for w in nodes if community[w] == target_c)

            k_u = degrees[u]

            # approximate modularity gain (Louvain local rule)
            # ΔQ ∝ k_i,in - (k_u * sum_tot) / (2m)
            deltaQ = k_i_in - (k_u * sum_tot) / (2.0 * m)

            if deltaQ > best_deltaQ:
                best_deltaQ = deltaQ
                best_comm = target_c

        community[u] = best_comm

    # rebuild communities list
    groups = defaultdict(list)
    for node, c in community.items():
        groups[c].append(node)

    return list(groups.values())


# ---------------------------------------------------------------------
# Degree distribution / basic stats
# ---------------------------------------------------------------------

def compute_degree_distribution(adj: Adjacency) -> Dict[str, object]:
    degrees = [len(adj[u]) for u in adj]
    if not degrees:
        return {
            "degree_list": [],
            "degree_histogram": {},
            "avg_degree": 0.0,
            "min_degree": 0,
            "max_degree": 0,
            "variance": 0.0,
        }

    freq = defaultdict(int)
    for d in degrees:
        freq[d] += 1

    n = len(degrees)
    avg = sum(degrees) / n
    min_d = min(degrees)
    max_d = max(degrees)
    mean = avg
    variance = sum((d - mean) ** 2 for d in degrees) / n

    return {
        "degree_list": degrees,
        "degree_histogram": dict(sorted(freq.items())),
        "avg_degree": avg,
        "min_degree": min_d,
        "max_degree": max_d,
        "variance": variance,
    }


# ---------------------------------------------------------------------
# Homophily utilities (from homophily.py)
# ---------------------------------------------------------------------

def compute_homophily(adj: Adjacency, labels: Dict[int, object]) -> Tuple[float, float]:
    """
    labels: dict node → category (e.g., personality tag)
    Computes:
        - edge homophily score
        - baseline expected homophily under random mixing
    """
    same = 0
    total = 0

    for u in adj:
        for v in adj[u]:
            if u < v:
                total += 1
                if labels[u] == labels[v]:
                    same += 1

    score = same / total if total > 0 else 0.0

    # Baseline expected homophily (random mixing)
    counts = Counter(labels.values())
    n = len(labels)
    baseline = sum((c / n) ** 2 for c in counts.values())

    return score, baseline


def label_majority_in_communities(
    communities: List[List[int]],
    labels: Dict[int, object]
) -> List[Tuple[object, int]]:
    """
    For each community, return (most_common_label, count).
    """
    result: List[Tuple[object, int]] = []
    for comm in communities:
        counter = Counter(labels[v] for v in comm)
        label, count = counter.most_common(1)[0]
        result.append((label, count))
    return result


# ---------------------------------------------------------------------
# CSV export + plotting
# ---------------------------------------------------------------------

def export_personalities_to_csv(
    path: str,
    adj: Adjacency,
    deg_cent: Dict[int, float],
    close_cent: Dict[int, float],
    bet_cent: Dict[int, float],
    pr_cent: Dict[int, float],
    personalities: Dict[int, Dict[str, float]],
    node_to_comm: Dict[int, int],
) -> None:
    """
    Write node-level metrics + Big Five personalities to a CSV.
    Each row: node, degree, centralities, E, O, A, C, N, community.
    """
    nodes = sorted(adj.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node",
            "degree",
            "degree_centrality",
            "closeness_centrality",
            "betweenness_centrality",
            "pagerank",
            "E",
            "O",
            "A",
            "C",
            "N",
            "community",
        ])
        for u in nodes:
            pers = personalities[u]
            writer.writerow([
                u,
                len(adj[u]),
                deg_cent[u],
                close_cent[u],
                bet_cent[u],
                pr_cent[u],
                pers["E"],
                pers["O"],
                pers["A"],
                pers["C"],
                pers["N"],
                node_to_comm.get(u, -1),
            ])
    print(f"Saved personalities + metrics to {path}")


def plot_personality_scatter(
    personalities: Dict[int, Dict[str, float]],
    node_to_comm: Dict[int, int],
    out_path: str,
) -> None:
    """
    Simple 2D scatterplot in personality space:
        x-axis: Extraversion (E)
        y-axis: Agreeableness (A)
    Colored by community id.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping personality visualization.")
        return

    xs = []
    ys = []
    colors = []

    for u, pers in personalities.items():
        xs.append(pers["E"])
        ys.append(pers["A"])
        c_id = node_to_comm.get(u, -1)
        # map community id to a matplotlib category color
        colors.append(f"C{c_id % 10}" if c_id >= 0 else "C0")

    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=10, alpha=0.7, c=colors)
    plt.xlabel("Extraversion (E)")
    plt.ylabel("Agreeableness (A)")
    plt.title("Personality scatter (colored by community)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved personality scatter plot to {out_path}")


# ---------------------------------------------------------------------
# CLI entry point: generate + analyze (model-agnostic analysis)
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic graph analysis pipeline")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ER", "BA", "WS", "SBM"],
        help="Graph model to generate",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")

    # ER
    parser.add_argument("--p", type=float, help="Edge probability for ER / WS")

    # BA
    parser.add_argument("--m", type=int, help="Number of edges per new node for BA")

    # WS
    parser.add_argument("--k", type=int, help="Each node's ring lattice degree for WS")
    parser.add_argument("--beta", type=float, help="Rewiring probability for WS")

    # SBM
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--blocks", type=int, help="Number of blocks for SBM")
    parser.add_argument("--block-size", type=int, help="Size of each SBM block")
    parser.add_argument("--p-intra", type=float, help="Intra-block probability for SBM")
    parser.add_argument("--p-inter", type=float, help="Inter-block probability for SBM")
    parser.add_argument(
        "--out-csv",
        type=str,
        help="Path to CSV for exporting node metrics + Big Five personalities",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        help="Path to PNG for personality scatter plot (E vs A, colored by community)",
    )

    args = parser.parse_args()

    # --- Generate graph (uniform adjacency format) --------------------
    if args.model == "ER":
        if args.p is None:
            raise ValueError("ER model requires --p")
        adj = generate_erdos_renyi(args.n, args.p, seed=args.seed)

    elif args.model == "BA":
        if args.m is None:
            raise ValueError("BA model requires --m")
        adj = generate_barabasi_albert(args.n, args.m, seed=args.seed)

    elif args.model == "WS":
        if args.k is None or args.p is None or args.beta is None:
            raise ValueError("WS model requires --k, --p, and --beta")
        # We interpret p as unused here; only k and beta matter for WS.
        adj = generate_watts_strogatz(args.n, args.k, args.beta, seed=args.seed)

    elif args.model == "SBM":
        if (
            args.blocks is None
            or args.block_size is None
            or args.p_intra is None
            or args.p_inter is None
        ):
            raise ValueError("SBM model requires --blocks, --block-size, --p-intra, and --p-inter")
        block_sizes = [args.block_size] * args.blocks
        adj = generate_sbm_symmetric(block_sizes, args.p_intra, args.p_inter, seed=args.seed)

    # --- Run analysis (model-agnostic) --------------------------------
    print("=== BASIC INFO ===")
    print("Nodes:", num_nodes(adj))
    print("Edges:", num_edges(adj))

    connectivity = compute_connectivity(adj)
    print("\n=== CONNECTIVITY ===")
    print(connectivity)

    # You can also directly access connected_components(adj) if needed:
    # comps = connected_components(adj)

    deg_cent = degree_centrality(adj)
    close_cent = closeness_centrality(adj)
    bet_cent = betweenness_centrality(adj)
    pr_cent = pagerank(adj)

    # Personality computation (Big Five)
    personalities = compute_big_five_personality(
        adj,
        deg_cent=deg_cent,
        close_cent=close_cent,
        bet_cent=bet_cent,
        noise_sigma=0.05,  # set to 0.0 if you want no randomness
    )

    print("\n=== CENTRALITY (samples) ===")
    sample_nodes = list(adj.keys())[:5]
    print("Degree:", {u: deg_cent[u] for u in sample_nodes})
    print("Closeness:", {u: close_cent[u] for u in sample_nodes})
    print("Betweenness:", {u: bet_cent[u] for u in sample_nodes})
    print("PageRank:", {u: pr_cent[u] for u in sample_nodes})

    print("\n=== PERSONALITY (Big Five, samples) ===")
    print({u: personalities[u] for u in sample_nodes})

    # Greedy modularity communities
    communities = community_detection(adj)
    comm_sizes = [len(c) for c in communities]

    # build node -> community id mapping
    node_to_comm: Dict[int, int] = {}
    for cid, group in enumerate(communities):
        for u in group:
            node_to_comm[u] = cid

    print("\n=== COMMUNITIES (greedy modularity) ===")
    print("Num communities:", len(communities))
    print("Community sizes:", comm_sizes)

    degree_stats = compute_degree_distribution(adj)
    print("\n=== DEGREE DISTRIBUTION ===")
    print("Average degree:", degree_stats["avg_degree"])
    print("Min degree:", degree_stats["min_degree"])
    print("Max degree:", degree_stats["max_degree"])
    print("Variance:", degree_stats["variance"])
    print("Histogram (first 10):", list(degree_stats["degree_histogram"].items())[:10])

    # --- Export personalities / visualization if requested ------------
    if args.out_csv:
        export_personalities_to_csv(
            args.out_csv,
            adj,
            deg_cent,
            close_cent,
            bet_cent,
            pr_cent,
            personalities,
            node_to_comm,
        )

    if args.out_plot:
        plot_personality_scatter(
            personalities,
            node_to_comm,
            args.out_plot,
        )


if __name__ == "__main__":
    main()
