#!/usr/bin/env python3
"""
compare.py (NO CLI VERSION)

Automatically:
    - Generates 4 synthetic graphs (ER, BA, WS, SBM)
    - Computes metrics using analysis.py
    - Prints comparison
    - Saves comparison plots automatically

Run with:
    python compare.py
"""

import csv
from statistics import mean
from math import inf
import matplotlib.pyplot as plt

from analysis import (
    num_nodes,
    num_edges,
    compute_connectivity,
    clustering_coefficient,
    community_detection,
    bfs_shortest_paths,
)

from erdos import generate_erdos_renyi
from ba import generate_barabasi_albert
from ws import generate_watts_strogatz
from sbm import generate_sbm_symmetric

Adjacency = dict[int, set[int]]


# ------------------------------------------------------------------
# PARAMETERS — EDIT IF YOU WANT DIFFERENT SETTINGS
# ------------------------------------------------------------------

N = 500          # nodes
SEED = 42        # reproducibility

# ER
P_ER = 0.03

# BA
M_BA = 3

# WS
K_WS = 4
BETA_WS = 0.2

# SBM
BLOCKS = 4
BLOCK_SIZE = 125
P_INTRA = 0.12
P_INTER = 0.02

PLOT_PREFIX = "cmp_"   # all PNGs saved as cmp_xxx.png


# ------------------------------------------------------------------
# Approx avg path length
# ------------------------------------------------------------------

def approx_avg_path_length(adj: Adjacency, samples: int = 20) -> float:
    nodes = list(adj.keys())
    if not nodes:
        return 0.0
    n = len(nodes)
    samples = min(samples, n)

    import random
    random_nodes = random.sample(nodes, samples)
    dvals: list[int] = []

    for u in random_nodes:
        dist = bfs_shortest_paths(adj, u)
        reachable = [d for d in dist.values() if d > 0]
        if reachable:
            dvals.extend(reachable)

    if not dvals:
        return inf
    return sum(dvals) / len(dvals)


# ------------------------------------------------------------------
# Compute metrics for a single graph
# ------------------------------------------------------------------

def compute_metrics(adj: Adjacency) -> dict:
    cinfo = compute_connectivity(adj)
    comms = community_detection(adj)
    clust = clustering_coefficient(adj)
    avg_clust = mean(clust.values()) if clust else 0.0
    avg_path = approx_avg_path_length(adj)

    n = num_nodes(adj)
    e = num_edges(adj)

    return {
        "nodes": n,
        "edges": e,
        "avg_degree": (2 * e) / max(n, 1),
        "num_components": cinfo["num_components"],
        "giant_component": cinfo["giant_component_size"],
        "avg_clustering": avg_clust,
        "approx_avg_path": avg_path,
        "num_communities": len(comms),
    }


# ------------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------------

def plot_metric_bar(results, metric_key, ylabel, out_path):
    labels = [name for name, _ in results]
    values = [m[metric_key] for _, m in results]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(f"{metric_key} comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    # --- Generate graphs -------------------------------------------------

    print("Generating graphs...")

    adj_er = generate_erdos_renyi(N, P_ER, seed=SEED)
    adj_ba = generate_barabasi_albert(N, M_BA, seed=SEED)
    adj_ws = generate_watts_strogatz(N, K_WS, BETA_WS, seed=SEED)
    block_sizes = [BLOCK_SIZE] * BLOCKS
    adj_sbm = generate_sbm_symmetric(block_sizes, P_INTRA, P_INTER, seed=SEED)

    graphs = {
        "ER": adj_er,
        "BA": adj_ba,
        "WS": adj_ws,
        "SBM": adj_sbm,
    }

    # --- Compute metrics -------------------------------------------------

    results = []
    print("\n=== Graph Comparison ===")

    for name, adj in graphs.items():
        print(f"\n--- {name} ---")
        metrics = compute_metrics(adj)
        results.append((name, metrics))
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # --- Make plots automatically ---------------------------------------

    print("\nGenerating plots...")

    plot_metric_bar(results, "avg_degree", "Average Degree",
                    f"{PLOT_PREFIX}avg_degree.png")

    plot_metric_bar(results, "avg_clustering", "Average Clustering Coefficient",
                    f"{PLOT_PREFIX}avg_clustering.png")

    plot_metric_bar(results, "approx_avg_path", "Approx. Avg Path Length",
                    f"{PLOT_PREFIX}avg_path.png")

    plot_metric_bar(results, "num_communities", "Number of Communities",
                    f"{PLOT_PREFIX}num_communities.png")

    plot_metric_bar(results, "giant_component", "Giant Component Size",
                    f"{PLOT_PREFIX}giant_component.png")

    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
