# ===============================================================
# Unified CLI Graph Recommender System
# Supports BA / ER / WS / SBM
# ===============================================================

import sys

# Import your actual files (renamed with _rec suffix)
from ba_rec import generate_barabasi_albert, hybrid_recommender as ba_rec
from erdos_rec import generate_erdos_renyi, hybrid_recommender as er_rec
from ws_rec import generate_watts_strogatz, hybrid_recommender as ws_rec
from sbm_rec import generate_sbm_symmetric, hybrid_recommender as sbm_rec


# ===============================================================
# Utility: Count edges from adjacency dict
# ===============================================================

def count_edges(adj):
    return sum(len(adj[u]) for u in adj) // 2


# ===============================================================
# Menu display utility
# ===============================================================

def menu(options):
    print("\nSelect an option:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    choice = int(input("\nEnter choice: "))
    return choice


# ===============================================================
# MAIN PROGRAM
# ===============================================================

def main():

    print("\n==============================")
    print("   NETWORK FRIEND RECOMMENDER")
    print("==============================")

    # ----------------------------------------------------
    # 1. Choose graph model
    # ----------------------------------------------------
    graph_choice = menu([
        "Barabási–Albert (BA)",
        "Erdős–Rényi (ER)",
        "Watts–Strogatz (WS)",
        "Stochastic Block Model (SBM)",
        "Exit"
    ])

    if graph_choice == 5:
        print("Exiting.")
        sys.exit(0)

    # ----------------------------------------------------
    # 2. Input parameters based on chosen model
    # ----------------------------------------------------

    # -------- BA --------
    if graph_choice == 1:
        print("\n--- Barabási–Albert (BA) Model ---")
        n = int(input("Number of nodes (e.g., 100): "))
        m = int(input("m (edges per new node, e.g., 3): "))
        adj = generate_barabasi_albert(n, m, seed=42)
        rec_func = ba_rec

    # -------- ER --------
    elif graph_choice == 2:
        print("\n--- Erdős–Rényi (ER) Model ---")
        n = int(input("Number of nodes (e.g., 100): "))
        p = float(input("Edge probability p (0 to 1): "))
        adj = generate_erdos_renyi(n, p, seed=42)
        rec_func = er_rec

    # -------- WS --------
    elif graph_choice == 3:
        print("\n--- Watts–Strogatz (WS) Model ---")
        n = int(input("Number of nodes (e.g., 100): "))
        k = int(input("k (even number, e.g., 6): "))
        beta = float(input("Rewiring β (0 to 1): "))
        adj = generate_watts_strogatz(n, k, beta, seed=42)
        rec_func = ws_rec

    # -------- SBM --------
    elif graph_choice == 4:
        print("\n--- Stochastic Block Model (SBM) ---")
        num_blocks = int(input("Number of blocks (e.g., 3): "))

        block_sizes = []
        for i in range(num_blocks):
            size = int(input(f"Size of block {i}: "))
            block_sizes.append(size)

        p_intra = float(input("Intra-community probability p_intra: "))
        p_inter = float(input("Inter-community probability p_inter: "))

        adj = generate_sbm_symmetric(block_sizes, p_intra, p_inter, seed=42)
        rec_func = sbm_rec

    # ----------------------------------------------------
    # 3. Graph summary
    # ----------------------------------------------------
    num_nodes = len(adj)
    num_edges = count_edges(adj)

    print("\n===================================")
    print("          GRAPH GENERATED")
    print("===================================")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")

    # ----------------------------------------------------
    # 4. Choose user for recommendations
    # ----------------------------------------------------
    user = int(input(f"\nEnter user ID (0 to {num_nodes - 1}): "))

    # ----------------------------------------------------
    # 5. Generate recommendations
    # ----------------------------------------------------
    recs = rec_func(adj, user)

    print(f"\n=== Friend Recommendations for User {user} ===")
    if not recs:
        print("No recommendations (this user is already connected to all others).")
    else:
        for v, score in recs:
            print(f"  -> {v:2d}   score={score:.4f}")

    print("\nDone.\n")


# ===============================================================
# Run program
# ===============================================================
if __name__ == "__main__":
    main()
