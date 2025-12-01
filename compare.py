#!/usr/bin/env python3

import csv
import os
import matplotlib.pyplot as plt
import networkx as nx

from er import generate_er_graph
from ws import generate_ws_graph
from ba import generate_ba_graph
from sbm import generate_sbm_graph

import analysis as A


# ---------------------------------------------------------
# Helper: ensure output directories exist
# ---------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


OUTPUT_DIR = "comparison_output"
ensure_dir(OUTPUT_DIR)
ensure_dir(f"{OUTPUT_DIR}/csv")
ensure_dir(f"{OUTPUT_DIR}/plots")


# ---------------------------------------------------------
# Run models
# ---------------------------------------------------------
def run_models(num_runs=20):
    results = {
        "ER": [],
        "WS": [],
        "BA": [],
        "SBM": []
    }

    for run in range(num_runs):
        print(f"[RUN {run+1}/{num_runs}] Generating graphs...")

        # --- ER ---
        G_er = generate_er_graph(n=500, p=0.01)
        results["ER"].append(A.compute_all_metrics(G_er))

        # --- WS ---
        G_ws = generate_ws_graph(n=500, k=6, beta=0.1)
        results["WS"].append(A.compute_all_metrics(G_ws))

        # --- BA ---
        G_ba = generate_ba_graph(n=500, m=3)
        results["BA"].append(A.compute_all_metrics(G_ba))

        # --- SBM ---
        sizes = [150, 150, 200]
        P = [
            [0.02, 0.005, 0.003],
            [0.005, 0.02, 0.004],
            [0.003, 0.004, 0.015],
        ]
        G_sbm = generate_sbm_graph(500, sizes, P)
        results["SBM"].append(A.compute_all_metrics(G_sbm))

    return results


# ---------------------------------------------------------
# Save CSV datasets for each metric
# ---------------------------------------------------------
def export_csv(results):
    for model, runs in results.items():

        # Degree distribution
        with open(f"{OUTPUT_DIR}/csv/{model}_degree.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "degree"])
            for i, r in enumerate(runs):
                for d in r["degrees"]:
                    writer.writerow([i, d])

        # Clustering
        with open(f"{OUTPUT_DIR}/csv/{model}_clustering.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "clustering"])
            for i, r in enumerate(runs):
                for c in r["clustering"]:
                    writer.writerow([i, c])

        # Path lengths
        with open(f"{OUTPUT_DIR}/csv/{model}_paths.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "path_length"])
            for i, r in enumerate(runs):
                for d in r["distances"]:
                    writer.writerow([i, d])

        # Component sizes
        with open(f"{OUTPUT_DIR}/csv/{model}_components.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "component_size"])
            for i, r in enumerate(runs):
                for s in r["components"]:
                    writer.writerow([i, s])


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_compare(results):

    # ------------------------
    # DEGREE DISTRIBUTION
    # ------------------------
    plt.figure(figsize=(8, 6))
    for model, runs in results.items():
        all_degrees = []
        for r in runs:
            all_degrees.extend(r["degrees"])
        plt.hist(all_degrees, bins=40, alpha=0.5, label=model)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution Comparison")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/plots/degree_distribution.png")
    plt.close()

    # ------------------------
    # CLUSTERING DISTRIBUTION
    # ------------------------
    plt.figure(figsize=(8, 6))
    for model, runs in results.items():
        all_c = []
        for r in runs:
            all_c.extend(r["clustering"])
        plt.hist(all_c, bins=40, alpha=0.5, label=model)
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.title("Clustering Coefficient Distribution Comparison")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/plots/clustering_distribution.png")
    plt.close()

    # ------------------------
    # PATH LENGTH DISTRIBUTION
    # ------------------------
    plt.figure(figsize=(8, 6))
    for model, runs in results.items():
        all_d = []
        for r in runs:
            all_d.extend(r["distances"])
        plt.hist(all_d, bins=40, alpha=0.5, label=model)
    plt.xlabel("Shortest Path Length")
    plt.ylabel("Frequency")
    plt.title("Shortest Path Length Distribution Comparison")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/plots/path_distribution.png")
    plt.close()

    # ------------------------
    # COMPONENT SIZE DISTRIBUTION
    # ------------------------
    plt.figure(figsize=(8, 6))
    for model, runs in results.items():
        all_c = []
        for r in runs:
            all_c.extend(r["components"])
        plt.hist(all_c, bins=40, alpha=0.5, label=model)
    plt.xlabel("Component Size")
    plt.ylabel("Frequency")
    plt.title("Component Size Distribution Comparison")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/plots/component_distribution.png")
    plt.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    results = run_models(num_runs=20)
    export_csv(results)
    plot_compare(results)

    print("\nAll tasks complete!")
    print(f"CSV files saved in: {OUTPUT_DIR}/csv/")
    print(f"Plots saved in: {OUTPUT_DIR}/plots/")



