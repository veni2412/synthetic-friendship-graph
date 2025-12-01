# compare_models.py
"""
Compare synthetic graph models (ER, BA, WS, SBM) using the analysis
functions defined in analysis.py.

For each model, this script:
  - Generates a graph with roughly similar average degree.
  - Runs connectivity, centralities, clustering, communities, etc.
  - Stores a summary of metrics in a CSV.
  - Produces comparison plots of degree distributions and clustering.

Usage example:

    python compare_models.py \
        --n 500 \
        --avg-degree 8 \
        --ws-beta 0.1 \
        --sbm-blocks 4 \
        --sbm-p-intra 0.12 \
        --sbm-p-inter 0.02 \
        --out-dir results_models

"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from erdos import generate_erdos_renyi
from ba import generate_barabasi_albert
from ws import generate_watts_strogatz
from sbm import generate_sbm_symmetric

from analysis import (
    Adjacency,
    num_nodes,
    num_edges,
    compute_connectivity,
    degree_centrality,
    closeness_centrality,
    betweenness_centrality,
    pagerank,
    clustering_coefficient,
    compute_degree_distribution,
    community_detection,
    compute_modularity,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def summary_row_from_graph(
    model_name: str,
    adj: Adjacency,
) -> dict:
    """Run analysis.py-style metrics and return a summary row for one graph."""

    n = num_nodes(adj)
    m = num_edges(adj)

    # Connectivity
    conn = compute_connectivity(adj)
    num_components = conn["num_components"]
    giant_size = conn["giant_component_size"]
    isolated = conn["isolated_nodes"]

    # Degree stats
    deg_stats = compute_degree_distribution(adj)
    avg_degree = deg_stats["avg_degree"]
    min_degree = deg_stats["min_degree"]
    max_degree = deg_stats["max_degree"]
    degree_var = deg_stats["variance"]

    # Centralities
    deg_cent = degree_centrality(adj)
    close_cent = closeness_centrality(adj)
    bet_cent = betweenness_centrality(adj)
    pr_cent = pagerank(adj)

    # Clustering
    clust = clustering_coefficient(adj)
    mean_clust = sum(clust.values()) / n if n > 0 else 0.0

    # Communities (greedy modularity)
    communities = community_detection(adj)
    node_to_comm = {}
    for cid, group in enumerate(communities):
        for u in group:
            node_to_comm[u] = cid
    mod = compute_modularity(adj, node_to_comm)

    # A crude global closeness summary (average node closeness)
    mean_closeness = sum(close_cent.values()) / n if n > 0 else 0.0

    return {
        "model": model_name,
        "n": n,
        "m": m,
        "avg_degree": avg_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "degree_variance": degree_var,
        "num_components": num_components,
        "giant_component_size": giant_size,
        "isolated_nodes": isolated,
        "mean_clustering": mean_clust,
        "modularity": mod,
        "mean_closeness": mean_closeness,
        # You can add more aggregates here if you want.
    }, deg_stats, clust


def write_summary_csv(path: str, rows: list[dict]) -> None:
    """Write list of dicts to CSV with a shared header."""
    if not rows:
        return

    import csv

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_degree_distributions(
    out_path: str,
    model_to_deg_hist: dict[str, dict[int, int]],
):
    """
    Plot degree distributions for each model on the same figure.
    x: degree
    y: fraction of nodes with that degree
    """
    plt.figure(figsize=(7, 5))

    for model, hist in model_to_deg_hist.items():
        if not hist:
            continue
        total = sum(hist.values())
        xs = sorted(hist.keys())
        ys = [hist[d] / total for d in xs]  # normalize to probabilities
        plt.plot(xs, ys, marker="o", linestyle="-", label=model)

    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.title("Degree Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_clustering_boxplot(
    out_path: str,
    model_to_clustering: dict[str, dict[int, float]],
):
    """
    Boxplot of node-level clustering coefficients for each model.
    """
    labels = []
    data = []

    for model, clust_dict in model_to_clustering.items():
        if not clust_dict:
            continue
        labels.append(model)
        data.append(list(clust_dict.values()))

    if not data:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Clustering coefficient")
    plt.title("Local Clustering Coefficient Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare ER / BA / WS / SBM models using analysis.py"
    )

    parser.add_argument("--n", type=int, required=True, help="Number of nodes")

    parser.add_argument(
        "--avg-degree",
        type=float,
        default=8.0,
        help="Target average degree (used to choose model parameters)",
    )

    parser.add_argument(
        "--ws-beta",
        type=float,
        default=0.1,
        help="Rewiring probability for Watts–Strogatz",
    )

    # SBM parameters
    parser.add_argument(
        "--sbm-blocks",
        type=int,
        default=4,
        help="Number of blocks for SBM (blocks * block-size should ≈ n)",
    )
    parser.add_argument(
        "--sbm-block-size",
        type=int,
        default=None,
        help="Block size for SBM; if not given, n // sbm-blocks",
    )
    parser.add_argument(
        "--sbm-p-intra",
        type=float,
        default=0.12,
        help="Intra-block edge probability for SBM",
    )
    parser.add_argument(
        "--sbm-p-inter",
        type=float,
        default=0.02,
        help="Inter-block edge probability for SBM",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (per-model offsets are used)",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for CSVs and plots",
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    n = args.n
    target_k = args.avg_degree

    # -----------------------------
    # Choose model-specific params
    # -----------------------------
    # ER: p = desired_avg_degree / (n - 1)
    if n > 1:
        er_p = target_k / (n - 1)
    else:
        er_p = 0.0

    # BA: m ≈ target_k / 2 (since average degree ~ 2m)
    ba_m = max(1, int(round(target_k / 2.0)))

    # WS: k should be even and around target_k
    ws_k = int(round(target_k))
    if ws_k % 2 == 1:
        ws_k += 1
    ws_beta = args.ws_beta

    # SBM: block sizes
    sbm_blocks = args.sbm_blocks
    if args.sbm_block_size is not None:
        sbm_block_size = args.sbm_block_size
    else:
        sbm_block_size = max(1, n // sbm_blocks)

    # Adjust total nodes if needed (we stick to n in generation where possible)
    total_sbm_nodes = sbm_blocks * sbm_block_size
    if total_sbm_nodes != n:
        print(
            f"[SBM] Note: blocks*block_size = {total_sbm_nodes} != n={n}. "
            f"SBM will use {total_sbm_nodes} nodes."
        )

    sbm_p_intra = args.sbm_p_intra
    sbm_p_inter = args.sbm_p_inter

    print("=== PARAMETER SUMMARY ===")
    print(f"n = {n}, target avg degree ≈ {target_k}")
    print(f"ER: p = {er_p:.4f}")
    print(f"BA: m = {ba_m}")
    print(f"WS: k = {ws_k}, beta = {ws_beta}")
    print(
        f"SBM: blocks = {sbm_blocks}, block_size = {sbm_block_size}, "
        f"p_intra = {sbm_p_intra}, p_inter = {sbm_p_inter}"
    )

    # -----------------------------
    # Generate and analyze graphs
    # -----------------------------
    summary_rows: list[dict] = []
    model_to_deg_hist: dict[str, dict[int, int]] = {}
    model_to_clust: dict[str, dict[int, float]] = {}

    # ER
    er_adj = generate_erdos_renyi(n, er_p, seed=args.seed)
    er_row, er_degstats, er_clust = summary_row_from_graph("ER", er_adj)
    summary_rows.append(er_row)
    model_to_deg_hist["ER"] = er_degstats["degree_histogram"]
    model_to_clust["ER"] = er_clust

    # BA
    ba_adj = generate_barabasi_albert(n, ba_m, seed=args.seed + 1)
    ba_row, ba_degstats, ba_clust = summary_row_from_graph("BA", ba_adj)
    summary_rows.append(ba_row)
    model_to_deg_hist["BA"] = ba_degstats["degree_histogram"]
    model_to_clust["BA"] = ba_clust

    # WS
    ws_adj = generate_watts_strogatz(n, ws_k, ws_beta, seed=args.seed + 2)
    ws_row, ws_degstats, ws_clust = summary_row_from_graph("WS", ws_adj)
    summary_rows.append(ws_row)
    model_to_deg_hist["WS"] = ws_degstats["degree_histogram"]
    model_to_clust["WS"] = ws_clust

    # SBM
    sbm_block_sizes = [sbm_block_size] * sbm_blocks
    sbm_adj = generate_sbm_symmetric(
        sbm_block_sizes,
        sbm_p_intra,
        sbm_p_inter,
        seed=args.seed + 3,
    )
    sbm_row, sbm_degstats, sbm_clust = summary_row_from_graph("SBM", sbm_adj)
    summary_rows.append(sbm_row)
    model_to_deg_hist["SBM"] = sbm_degstats["degree_histogram"]
    model_to_clust["SBM"] = sbm_clust

    # -----------------------------
    # Store summary CSV
    # -----------------------------
    summary_csv_path = os.path.join(args.out_dir, "model_summary.csv")
    write_summary_csv(summary_csv_path, summary_rows)
    print(f"Saved summary metrics to {summary_csv_path}")

    # -----------------------------
    # Plots
    # -----------------------------
    deg_plot_path = os.path.join(args.out_dir, "degree_distribution_comparison.png")
    plot_degree_distributions(deg_plot_path, model_to_deg_hist)
    print(f"Saved degree distribution comparison plot to {deg_plot_path}")

    clust_plot_path = os.path.join(args.out_dir, "clustering_comparison_boxplot.png")
    plot_clustering_boxplot(clust_plot_path, model_to_clust)
    print(f"Saved clustering comparison boxplot to {clust_plot_path}")


if __name__ == "__main__":
    main()
