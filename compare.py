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
import os
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
    degree_centrality,
    closeness_centrality,
    betweenness_centrality,
    pagerank,
    compute_big_five_personality,
    compute_homophily,
    compute_modularity,
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
PLOTS_DIR = "plots"
CSV_DIR = "csv"


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


# Collect sample of distances for histogram plots
def sample_distances(adj: Adjacency, samples: int = 20) -> list[int]:
    nodes = list(adj.keys())
    if not nodes:
        return []
    import random
    random_nodes = random.sample(nodes, min(samples, len(nodes)))
    dvals: list[int] = []
    for u in random_nodes:
        dist = bfs_shortest_paths(adj, u)
        reachable = [d for d in dist.values() if d > 0]
        dvals.extend(reachable)
    return dvals


# Degree assortativity (Pearson correlation of endpoint degrees)
def degree_assortativity(adj: Adjacency) -> float:
    xs = []
    ys = []
    for u in adj:
        du = len(adj[u])
        for v in adj[u]:
            if u < v:
                dv = len(adj[v])
                xs.append(du)
                ys.append(dv)
    if not xs:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (den_x * den_y) ** 0.5
    return (num / denom) if denom > 0 else 0.0


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

    # --- Centralities & Big Five personality tags ---
    deg_cent = degree_centrality(adj)
    close_cent = closeness_centrality(adj)
    bet_cent = betweenness_centrality(adj)
    pr = pagerank(adj)
    personalities = compute_big_five_personality(adj, deg_cent, close_cent, bet_cent, noise_sigma=0.0)

    # Sample distances for histogram
    dist_sample = sample_distances(adj, samples=20)

    # Degree assortativity
    assort_r = degree_assortativity(adj)

    return {
        "nodes": n,
        "edges": e,
        "avg_degree": (2 * e) / max(n, 1),
        "num_components": cinfo["num_components"],
        "giant_component": cinfo["giant_component_size"],
        "avg_clustering": avg_clust,
        "approx_avg_path": avg_path,
        "num_communities": len(comms),
        "adj": adj,
        "personalities": personalities,
        "degrees": {u: len(adj[u]) for u in adj},
        "clustering_dict": clust,
        "centralities": {
            "degree": deg_cent,
            "closeness": close_cent,
            "betweenness": bet_cent,
            "pagerank": pr,
        },
        "communities": comms,
        "distances_sample": dist_sample,
        "assortativity_degree": assort_r,
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
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # Save summary metrics to CSV
    with open(f"{CSV_DIR}/{PLOT_PREFIX}summary_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "nodes", "edges", "avg_degree", "num_components", "giant_component", "avg_clustering", "approx_avg_path", "num_communities"]) 
        for name, m in results:
            writer.writerow([name, m["nodes"], m["edges"], m["avg_degree"], m["num_components"], m["giant_component"], m["avg_clustering"], m["approx_avg_path"], m["num_communities"]])

    plot_metric_bar(results, "avg_degree", "Average Degree",
                    f"{PLOTS_DIR}/{PLOT_PREFIX}avg_degree.png")

    plot_metric_bar(results, "avg_clustering", "Average Clustering Coefficient",
                    f"{PLOTS_DIR}/{PLOT_PREFIX}avg_clustering.png")

    plot_metric_bar(results, "approx_avg_path", "Approx. Avg Path Length",
                    f"{PLOTS_DIR}/{PLOT_PREFIX}avg_path.png")

    plot_metric_bar(results, "num_communities", "Number of Communities",
                    f"{PLOTS_DIR}/{PLOT_PREFIX}num_communities.png")

    plot_metric_bar(results, "giant_component", "Giant Component Size",
                    f"{PLOTS_DIR}/{PLOT_PREFIX}giant_component.png")

    # --- Big Five Homophily Analysis ---
    def get_dominant_trait(pers):
        return max(pers, key=pers.get)

    homophily_scores = {}
    homophily_baselines = {}
    for name, metrics in results:
        pers = metrics["personalities"]
        labels = {u: get_dominant_trait(pers[u]) for u in pers}
        score, baseline = compute_homophily(metrics["adj"], labels)
        homophily_scores[name] = score
        homophily_baselines[name] = baseline

    # Plot homophily vs. baseline
    plt.figure(figsize=(7, 5))
    x = list(range(len(homophily_scores)))
    width = 0.35
    labels_order = list(homophily_scores.keys())
    obs_vals = [homophily_scores[m] for m in labels_order]
    base_vals = [homophily_baselines[m] for m in labels_order]
    plt.bar([xi - width/2 for xi in x], obs_vals, width, label="Observed")
    plt.bar([xi + width/2 for xi in x], base_vals, width, label="Random Baseline")
    plt.xticks(x, labels_order)
    plt.ylabel("Homophily (fraction of same label edges)")
    plt.title("Personality Homophily by Model (Dominant Big Five Trait)")
    plt.legend()
    plt.tight_layout()
    # Save homophily CSV
    with open(f"{CSV_DIR}/{PLOT_PREFIX}personality_homophily.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "observed", "baseline"])
        for name in labels_order:
            writer.writerow([name, homophily_scores[name], homophily_baselines[name]])

    plt.savefig(f"{PLOTS_DIR}/{PLOT_PREFIX}personality_homophily.png", dpi=200)
    plt.close()
    print(f"Saved plot: {PLOTS_DIR}/{PLOT_PREFIX}personality_homophily.png")

    # --- Cluster Distinction: Community Personality Dominance ---
    for name, metrics in results:
        pers = metrics["personalities"]
        comms = metrics["communities"]
        comm_labels = []
        for comm in comms:
            if not comm:
                continue
            trait_counts = {trait: 0 for trait in ["E", "O", "A", "C", "N"]}
            for u in comm:
                trait = get_dominant_trait(pers[u])
                trait_counts[trait] += 1
            dominant_trait = max(trait_counts, key=trait_counts.get)
            comm_labels.append(dominant_trait)
        trait_order = ["E", "O", "A", "C", "N"]
        trait_hist = [comm_labels.count(tr) for tr in trait_order]
        plt.figure(figsize=(6, 4))
        plt.bar(trait_order, trait_hist)
        plt.xlabel("Dominant Personality Trait")
        plt.ylabel("#Communities (dominant trait)")
        plt.title(f"Community Personality Dominance: {name}")
        plt.tight_layout()
        # Save dominant trait counts to CSV
        with open(f"{CSV_DIR}/{PLOT_PREFIX}{name}_community_personality_dominance.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trait", "count"])
            for tr, ct in zip(trait_order, trait_hist):
                writer.writerow([tr, ct])

        out_path = f"{PLOTS_DIR}/{PLOT_PREFIX}{name}_community_personality_dominance.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    # --- Degree CCDF (log-log) ---
    plt.figure(figsize=(7, 5))
    for name, metrics in results:
        degs = list(metrics["degrees"].values())
        if not degs:
            continue
        # Compute CCDF: P(K>=k)
        from collections import Counter
        c = Counter(degs)
        ks = sorted(c.keys())
        total = len(degs)
        tail_counts = []
        for i, k in enumerate(ks):
            tail = sum(c[j] for j in ks[i:])
            tail_counts.append(tail / total)
        plt.plot(ks, tail_counts, marker='o', linestyle='-', label=name)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree k')
    plt.ylabel('P(K ≥ k)')
    plt.title('Degree CCDF (log–log)')
    plt.legend()
    plt.tight_layout()
    # Save CCDF points to CSV
    with open(f"{CSV_DIR}/{PLOT_PREFIX}degree_ccdf.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "k", "P(K>=k)"])
        # Recompute per-model for CSV consistency
        for name, metrics in results:
            degs = list(metrics["degrees"].values())
            if not degs:
                continue
            from collections import Counter
            c = Counter(degs)
            ks = sorted(c.keys())
            total = len(degs)
            for i, k in enumerate(ks):
                tail = sum(c[j] for j in ks[i:])
                writer.writerow([name, k, tail / total])

    out_ccdf = f"{PLOTS_DIR}/{PLOT_PREFIX}degree_ccdf.png"
    plt.savefig(out_ccdf, dpi=200)
    plt.close()
    print(f"Saved plot: {out_ccdf}")

    # --- Clustering vs Degree scatter (sampled) ---
    for name, metrics in results:
        degs = metrics["degrees"]
        clust = metrics["clustering_dict"]
        xs = []
        ys = []
        for u in degs:
            xs.append(degs[u])
            ys.append(clust.get(u, 0.0))
        # Sample to limit points
        sample_size = min(1000, len(xs))
        import random
        idx = list(range(len(xs)))
        random.shuffle(idx)
        idx = idx[:sample_size]
        xs_s = [xs[i] for i in idx]
        ys_s = [ys[i] for i in idx]
        plt.figure(figsize=(6, 4))
        plt.scatter(xs_s, ys_s, s=10, alpha=0.5)
        plt.xlabel('Degree')
        plt.ylabel('Clustering Coefficient')
        plt.title(f'Clustering vs Degree: {name}')
        plt.tight_layout()
        out_path = f"{PLOTS_DIR}/{PLOT_PREFIX}{name}_clustering_vs_degree.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    # --- Path length distribution (sampled BFS) ---
    plt.figure(figsize=(7, 5))
    for name, metrics in results:
        dvals = metrics["distances_sample"]
        if not dvals:
            continue
        plt.hist(dvals, bins=20, alpha=0.5, label=name)
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Frequency (sampled)')
    plt.title('Path Length Distribution (sampled BFS)')
    plt.legend()
    plt.tight_layout()
    # Save sampled path length distributions to CSV
    with open(f"{CSV_DIR}/{PLOT_PREFIX}path_length_distribution.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "path_length"])
        for name, metrics in results:
            for d in metrics["distances_sample"]:
                writer.writerow([name, d])

    out_paths = f"{PLOTS_DIR}/{PLOT_PREFIX}path_length_distribution.png"
    plt.savefig(out_paths, dpi=200)
    plt.close()
    print(f"Saved plot: {out_paths}")

    # --- Degree assortativity bar ---
    labels = [name for name, _ in results]
    vals = [m["assortativity_degree"] for _, m in results]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.ylabel('Degree Assortativity (r)')
    plt.title('Degree Assortativity by Model')
    plt.tight_layout()
    # Save assortativity to CSV
    with open(f"{CSV_DIR}/{PLOT_PREFIX}degree_assortativity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "assortativity_r"])
        for name, m in results:
            writer.writerow([name, m["assortativity_degree"]])

    out_assort = f"{PLOTS_DIR}/{PLOT_PREFIX}degree_assortativity.png"
    plt.savefig(out_assort, dpi=200)
    plt.close()
    print(f"Saved plot: {out_assort}")

    # --- Centrality distributions (histograms) ---
    def plot_centrality_hist(key: str, title: str, filename: str):
        plt.figure(figsize=(7, 5))
        for name, metrics in results:
            vals = list(metrics["centralities"][key].values())
            if not vals:
                continue
            plt.hist(vals, bins=40, alpha=0.5, label=name)
        plt.xlabel(key.capitalize())
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        # Save centrality values to CSV
        with open(f"{CSV_DIR}/{PLOT_PREFIX}{key}_centrality.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "node", key])
            for name, metrics in results:
                vals = metrics["centralities"][key]
                for u, val in vals.items():
                    writer.writerow([name, u, val])
        out = f"{PLOTS_DIR}/{PLOT_PREFIX}{filename}"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved plot: {out}")

    plot_centrality_hist('degree', 'Degree Centrality Distribution', 'degree_centrality_hist.png')
    plot_centrality_hist('closeness', 'Closeness Centrality Distribution', 'closeness_centrality_hist.png')
    plot_centrality_hist('betweenness', 'Betweenness Centrality Distribution', 'betweenness_centrality_hist.png')
    plot_centrality_hist('pagerank', 'PageRank Distribution', 'pagerank_hist.png')

    # --- PageRank vs Degree scatter (per model) ---
    for name, metrics in results:
        degs = metrics["degrees"]
        pr = metrics["centralities"]["pagerank"]
        xs = []
        ys = []
        for u in degs:
            xs.append(degs[u])
            ys.append(pr.get(u, 0.0))
        # Sample
        sample_size = min(1000, len(xs))
        import random
        idx = list(range(len(xs)))
        random.shuffle(idx)
        idx = idx[:sample_size]
        xs_s = [xs[i] for i in idx]
        ys_s = [ys[i] for i in idx]
        plt.figure(figsize=(6, 4))
        plt.scatter(xs_s, ys_s, s=10, alpha=0.5)
        plt.xlabel('Degree')
        plt.ylabel('PageRank')
        plt.title(f'PageRank vs Degree: {name}')
        plt.tight_layout()
        # Save PR vs degree points to CSV
        with open(f"{CSV_DIR}/{PLOT_PREFIX}{name}_pagerank_vs_degree.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node", "degree", "pagerank"])
            for u in degs:
                writer.writerow([u, degs[u], pr.get(u, 0.0)])
        out_prdeg = f"{PLOTS_DIR}/{PLOT_PREFIX}{name}_pagerank_vs_degree.png"
        plt.savefig(out_prdeg, dpi=200)
        plt.close()
        print(f"Saved plot: {out_prdeg}")

    # --- Community size histograms (per model) ---
    for name, metrics in results:
        sizes = [len(c) for c in metrics["communities"]]
        plt.figure(figsize=(6, 4))
        plt.hist(sizes, bins=20, alpha=0.8)
        plt.xlabel('Community Size')
        plt.ylabel('Frequency')
        plt.title(f'Community Sizes: {name}')
        plt.tight_layout()
        # Save community sizes CSV
        with open(f"{CSV_DIR}/{PLOT_PREFIX}{name}_community_sizes.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["community_id", "size"])
            for cid, c in enumerate(metrics["communities"]):
                writer.writerow([cid, len(c)])
        out_comm = f"{PLOTS_DIR}/{PLOT_PREFIX}{name}_community_sizes.png"
        plt.savefig(out_comm, dpi=200)
        plt.close()
        print(f"Saved plot: {out_comm}")

    # --- Modularity Q comparison ---
    labels = []
    qvals = []
    for name, metrics in results:
        # Build node -> community id mapping
        node_to_comm = {}
        for cid, comm in enumerate(metrics["communities"]):
            for u in comm:
                node_to_comm[u] = cid
        Q = compute_modularity(metrics["adj"], node_to_comm)
        labels.append(name)
        qvals.append(Q)
    plt.figure(figsize=(6, 4))
    plt.bar(labels, qvals)
    plt.ylabel('Modularity Q')
    plt.title('Modularity by Model')
    plt.tight_layout()
    # Save modularity CSV
    with open(f"{CSV_DIR}/{PLOT_PREFIX}modularity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "modularity_Q"])
        for name, Q in zip(labels, qvals):
            writer.writerow([name, Q])
    out_mod = f"{PLOTS_DIR}/{PLOT_PREFIX}modularity.png"
    plt.savefig(out_mod, dpi=200)
    plt.close()
    print(f"Saved plot: {out_mod}")

    # --- Personality: average trait radar (per model) ---
    import math
    traits = ["E", "O", "A", "C", "N"]
    for name, metrics in results:
        pers = metrics["personalities"]
        means = []
        for t in traits:
            vals = [pers[u][t] for u in pers]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        # Radar setup
        angles = [i * 2 * math.pi / len(traits) for i in range(len(traits))]
        angles += [angles[0]]
        vals_plot = means + [means[0]]
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, vals_plot, 'o-', linewidth=2)
        ax.fill(angles, vals_plot, alpha=0.25)
        ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], traits)
        ax.set_title(f'Average Big Five Traits: {name}')
        ax.set_ylim(0, 1)
        # Save trait means CSV
        with open(f"{CSV_DIR}/{PLOT_PREFIX}{name}_trait_means.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trait", "mean"])
            for t, mval in zip(traits, means):
                writer.writerow([t, mval])
        out_radar = f"{PLOTS_DIR}/{PLOT_PREFIX}{name}_trait_radar.png"
        plt.tight_layout()
        plt.savefig(out_radar, dpi=200)
        plt.close()
        print(f"Saved plot: {out_radar}")

    # --- Personality mixing matrix (dominant trait heatmap) ---
    trait_index = {"E": 0, "O": 1, "A": 2, "C": 3, "N": 4}
    for name, metrics in results:
        pers = metrics["personalities"]
        # assign dominant trait per node
        dom = {u: max(pers[u], key=pers[u].get) for u in pers}
        # build 5x5 matrix of edge fractions
        size = 5
        mat = [[0 for _ in range(size)] for _ in range(size)]
        total_edges = 0
        adj = metrics["adj"]
        for u in adj:
            for v in adj[u]:
                if u < v:
                    i = trait_index[dom[u]]
                    j = trait_index[dom[v]]
                    mat[i][j] += 1
                    mat[j][i] += 1
                    total_edges += 2
        # normalize by total edges (or by row sums)
        if total_edges > 0:
            mat = [[val / total_edges for val in row] for row in mat]
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, cmap='Blues')
        plt.colorbar(label='Edge fraction')
        plt.xticks(list(range(5)), traits)
        plt.yticks(list(range(5)), traits)
        plt.title(f'Trait Mixing Heatmap: {name}')
        plt.tight_layout()
        # Save mixing matrix CSV
        with open(f"{CSV_DIR}/{PLOT_PREFIX}{name}_trait_mixing.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_trait", "col_trait", "edge_fraction"])
            for i, ri in enumerate(traits):
                for j, cj in enumerate(traits):
                    writer.writerow([ri, cj, mat[i][j]])
        out_mix = f"{PLOTS_DIR}/{PLOT_PREFIX}{name}_trait_mixing.png"
        plt.savefig(out_mix, dpi=200)
        plt.close()
        print(f"Saved plot: {out_mix}")

    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
