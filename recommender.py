"""
Friend recommendation system on synthetic graphs.

Graph models supported:
    - ER (Erdős–Rényi)
    - BA (Barabási–Albert)
    - WS (Watts–Strogatz)
    - SBM (Stochastic Block Model, symmetric)

Recommendation / link prediction scores implemented:
    - CN: Common Neighbors
    - Jaccard: Jaccard similarity
    - PA: Preferential Attachment
    - AA: Adamic–Adar
    - RA: Resource Allocation
    - Katz: Katz index (truncated walks from the source node)

We also combine these scores into a single weighted score:
    combined(v) = w_cn * CN(v) + w_jac * Jaccard(v) + ... + w_katz * Katz(v)
(using per-method min–max normalization so scales are comparable).
"""

import argparse
from typing import Dict, Set, List, Tuple
import math

from erdos import generate_erdos_renyi
from ba import generate_barabasi_albert
from ws import generate_watts_strogatz
from sbm import generate_sbm_symmetric

from analysis import (
    degree_centrality,
    closeness_centrality,
    betweenness_centrality,
    compute_big_five_personality,
)


Adjacency = Dict[int, Set[int]]


# ------------------------------------------------------------
# Helper: candidate set and safe access
# ------------------------------------------------------------

def compute_personality_vectors(adj: Adjacency) -> Dict[int, List[float]]:
    """
    Compute Big Five personality traits for each node and return
    node -> [E, O, A, C, N] vectors.
    """
    deg_cent = degree_centrality(adj)
    close_cent = closeness_centrality(adj)
    bet_cent = betweenness_centrality(adj)

    personalities = compute_big_five_personality(
        adj,
        deg_cent=deg_cent,
        close_cent=close_cent,
        bet_cent=bet_cent,
        noise_sigma=0.05,  # or 0.0 if you want purely structural
    )

    vectors: Dict[int, List[float]] = {}
    for u, p in personalities.items():
        vectors[u] = [p["E"], p["O"], p["A"], p["C"], p["N"]]
    return vectors


def cosine_similarity(x: List[float], y: List[float]) -> float:
    num = sum(a * b for a, b in zip(x, y))
    den_x = math.sqrt(sum(a * a for a in x))
    den_y = math.sqrt(sum(b * b for b in y))
    den = den_x * den_y
    if den == 0.0:
        return 0.0
    return num / den


def score_personality_similarity(
    adj: Adjacency,
    u: int,
    pers_vectors: Dict[int, List[float]],
) -> Dict[int, float]:
    """
    For a given user u, compute cosine similarity between u's personality
    vector and every candidate v (non-neighbor).
    """
    if u not in pers_vectors:
        raise ValueError(f"No personality vector for user {u}")

    scores: Dict[int, float] = {}
    candidates = get_candidates(adj, u)
    p_u = pers_vectors[u]

    for v in candidates:
        p_v = pers_vectors[v]
        scores[v] = cosine_similarity(p_u, p_v)

    return scores


def get_candidates(adj: Adjacency, u: int) -> List[int]:
    """
    All nodes that are NOT u and NOT already friends with u.
    These are potential "new friends" for link prediction.
    """
    if u not in adj:
        raise ValueError(f"User {u} not in graph (valid range: 0..{len(adj)-1})")

    neighbors = adj[u]
    return [v for v in adj.keys() if v != u and v not in neighbors]


def degree(adj: Adjacency, v: int) -> int:
    return len(adj[v])


# ------------------------------------------------------------
# Link prediction scores
# ------------------------------------------------------------

def score_common_neighbors(adj: Adjacency, u: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    neigh_u = adj[u]
    candidates = get_candidates(adj, u)
    for v in candidates:
        cn = len(neigh_u.intersection(adj[v]))
        scores[v] = float(cn)
    return scores


def score_jaccard(adj: Adjacency, u: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    neigh_u = adj[u]
    candidates = get_candidates(adj, u)
    for v in candidates:
        neigh_v = adj[v]
        inter = len(neigh_u.intersection(neigh_v))
        union = len(neigh_u.union(neigh_v))
        scores[v] = (inter / union) if union > 0 else 0.0
    return scores


def score_preferential_attachment(adj: Adjacency, u: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    deg_u = degree(adj, u)
    candidates = get_candidates(adj, u)
    for v in candidates:
        scores[v] = float(deg_u * degree(adj, v))
    return scores


def score_adamic_adar(adj: Adjacency, u: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    neigh_u = adj[u]
    candidates = get_candidates(adj, u)
    for v in candidates:
        neigh_v = adj[v]
        common = neigh_u.intersection(neigh_v)
        s = 0.0
        for w in common:
            deg_w = degree(adj, w)
            if deg_w > 1:
                s += 1.0 / math.log(deg_w)
        scores[v] = s
    return scores


def score_resource_allocation(adj: Adjacency, u: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    neigh_u = adj[u]
    candidates = get_candidates(adj, u)
    for v in candidates:
        neigh_v = adj[v]
        common = neigh_u.intersection(neigh_v)
        s = 0.0
        for w in common:
            deg_w = degree(adj, w)
            if deg_w > 0:
                s += 1.0 / deg_w
        scores[v] = s
    return scores


def score_katz(
    adj: Adjacency,
    u: int,
    beta: float = 0.05,
    max_length: int = 4
) -> Dict[int, float]:
    """
    Katz index from source node u:
        Katz(u, v) = sum_{l=1..L} beta^l * (# walks of length l from u to v)

    We compute walks in a dynamic-programming style:
        paths_prev[v] = # walks of current length ending at v.

    This is truncated at max_length and counts walks (nodes can repeat).
    """
    if u not in adj:
        raise ValueError(f"User {u} not in graph")

    katz_scores: Dict[int, float] = {v: 0.0 for v in adj.keys()}

    # length-1 walks: direct neighbors of u
    paths_prev: Dict[int, float] = {}
    for w in adj[u]:
        paths_prev[w] = paths_prev.get(w, 0.0) + 1.0

    for path_len in range(1, max_length + 1):
        factor = beta ** path_len
        for v, count in paths_prev.items():
            katz_scores[v] += factor * count

        paths_next: Dict[int, float] = {}
        for x, count_x in paths_prev.items():
            for nbr in adj[x]:
                paths_next[nbr] = paths_next.get(nbr, 0.0) + count_x

        paths_prev = paths_next

    candidates = get_candidates(adj, u)
    return {v: katz_scores[v] for v in candidates}


# ------------------------------------------------------------
# Utility: normalize, sort, display
# ------------------------------------------------------------

def minmax_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Min–max normalize scores to [0, 1] per method.
    If all scores are equal or dict is empty, return zeros.
    """
    if not scores:
        return {}

    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        # All equal → no discrimination, just return 0 for everyone
        return {v: 0.0 for v in scores}

    return {v: (s - vmin) / (vmax - vmin) for v, s in scores.items()}


def top_k_from_scores(scores: Dict[int, float], k: int) -> List[Tuple[int, float]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def print_recommendations(
    method_name: str,
    scores: Dict[int, float],
    k: int
) -> None:
    topk = top_k_from_scores(scores, k)
    print(f"\n=== {method_name} recommendations (top {k}) ===")
    if not topk:
        print("No candidates found (user may already be connected to almost everyone).")
        return
    for v, s in topk:
        print(f"User {v}: score = {s:.6f}")


# ------------------------------------------------------------
# Graph generation wrapper
# ------------------------------------------------------------

def generate_graph_from_args(args) -> Adjacency:
    """
    Generate a graph based on CLI arguments and return adjacency dict.
    """
    if args.model == "ER":
        if args.p is None:
            raise ValueError("ER model requires --p")
        return generate_erdos_renyi(args.n, args.p, seed=args.seed)

    if args.model == "BA":
        if args.m is None:
            raise ValueError("BA model requires --m")
        return generate_barabasi_albert(args.n, args.m, seed=args.seed)

    if args.model == "WS":
        if args.k is None or args.beta is None:
            raise ValueError("WS model requires --k and --beta")
        return generate_watts_strogatz(args.n, args.k, args.beta, seed=args.seed)

    if args.model == "SBM":
        if args.blocks is None or args.block_size is None \
           or args.p_intra is None or args.p_inter is None:
            raise ValueError("SBM model requires --blocks, --block-size, --p-intra, and --p-inter")
        block_sizes = [args.block_size] * args.blocks
        return generate_sbm_symmetric(block_sizes, args.p_intra, args.p_inter, seed=args.seed)

    raise ValueError(f"Unknown model: {args.model}")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Friend recommender on synthetic graphs")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ER", "BA", "WS", "SBM"],
        help="Graph model to generate"
    )
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")

    # ER
    parser.add_argument("--p", type=float, help="Edge probability for ER")

    # BA
    parser.add_argument("--m", type=int, help="Number of edges per new node for BA")

    # WS
    parser.add_argument("--k", type=int, help="Ring lattice degree for WS (must be even)")
    parser.add_argument("--beta", type=float, help="Rewiring probability for WS")

    # SBM
    parser.add_argument("--blocks", type=int, help="Number of blocks for SBM")
    parser.add_argument("--block-size", type=int, help="Size of each SBM block")
    parser.add_argument("--p-intra", type=float, help="Intra-block probability for SBM")
    parser.add_argument("--p-inter", type=float, help="Inter-block probability for SBM")
    parser.add_argument("--w-pers", type=float, default=1.0,
                    help="Weight for personality similarity")

    # Shared
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Recommendation params
    parser.add_argument(
        "--user",
        type=int,
        required=True,
        help="User id for which to recommend new friends (0..n-1)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations per method"
    )
    parser.add_argument(
        "--katz-beta",
        type=float,
        default=0.05,
        help="Decay factor beta for Katz index"
    )
    parser.add_argument(
        "--katz-max-length",
        type=int,
        default=4,
        help="Maximum walk length for Katz index"
    )

    # Weights for combined score
    parser.add_argument("--w-cn", type=float, default=1.0, help="Weight for Common Neighbors")
    parser.add_argument("--w-jac", type=float, default=1.0, help="Weight for Jaccard")
    parser.add_argument("--w-pa", type=float, default=1.0, help="Weight for Preferential Attachment")
    parser.add_argument("--w-aa", type=float, default=1.0, help="Weight for Adamic–Adar")
    parser.add_argument("--w-ra", type=float, default=1.0, help="Weight for Resource Allocation")
    parser.add_argument("--w-katz", type=float, default=1.0, help="Weight for Katz index")

    args = parser.parse_args()

    # Generate graph
    adj = generate_graph_from_args(args)
    num_edges = sum(len(neigh) for neigh in adj.values()) // 2
    print("Generated graph:")
    print("  Model:", args.model)
    print("  Nodes:", len(adj))
    print("  Edges:", num_edges)

    if args.user < 0 or args.user >= len(adj):
        raise ValueError(f"User id {args.user} is out of range (0..{len(adj)-1})")

    # --- Per-method scores -------------------------------------------
    cn_scores = score_common_neighbors(adj, args.user)
    jaccard_scores = score_jaccard(adj, args.user)
    pa_scores = score_preferential_attachment(adj, args.user)
    aa_scores = score_adamic_adar(adj, args.user)
    ra_scores = score_resource_allocation(adj, args.user)
    katz_scores = score_katz(
        adj,
        args.user,
        beta=args.katz_beta,
        max_length=args.katz_max_length
    )

    # Print individual method recommendations
    print_recommendations("Common Neighbors (CN)", cn_scores, args.top_k)
    print_recommendations("Jaccard", jaccard_scores, args.top_k)
    print_recommendations("Preferential Attachment (PA)", pa_scores, args.top_k)
    print_recommendations("Adamic–Adar (AA)", aa_scores, args.top_k)
    print_recommendations("Resource Allocation (RA)", ra_scores, args.top_k)
    print_recommendations("Katz", katz_scores, args.top_k)

        # --- Personality vectors and scores ------------------------------
    pers_vectors = compute_personality_vectors(adj)
    pers_scores = score_personality_similarity(adj, args.user, pers_vectors)

    print_recommendations("Personality Similarity", pers_scores, args.top_k)


    # --- Combined weighted score -------------------------------------
    # Normalize each method to [0,1] so weights are comparable
    cn_norm = minmax_normalize(cn_scores)
    jac_norm = minmax_normalize(jaccard_scores)
    pa_norm = minmax_normalize(pa_scores)
    aa_norm = minmax_normalize(aa_scores)
    ra_norm = minmax_normalize(ra_scores)
    katz_norm = minmax_normalize(katz_scores)
    pers_norm = minmax_normalize(pers_scores)


    all_candidates = set(cn_scores.keys())  # all methods use the same candidate set
    combined_scores: Dict[int, float] = {}

    for v in all_candidates:
        combined_scores[v] = (
            args.w_cn   * cn_norm.get(v, 0.0) +
            args.w_jac  * jac_norm.get(v, 0.0) +
            args.w_pa   * pa_norm.get(v, 0.0) +
            args.w_aa   * aa_norm.get(v, 0.0) +
            args.w_ra   * ra_norm.get(v, 0.0) +
            args.w_katz * katz_norm.get(v, 0.0) +
            args.w_pers * pers_norm.get(v, 0.0)
        )

    print_recommendations("Combined (weighted)", combined_scores, args.top_k)


if __name__ == "__main__":
    main()
