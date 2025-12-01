import csv
import os
import statistics
import networkx as nx

import analysis  # your existing analysis.py


DATA_DIR = "fb_friends.csv"   # directory containing edges.csv + nodes.csv


def parse_pos(pos_str):
    """
    Parse strings like: "array([ 0.06259288, -1.12057376])" -> (x, y)
    Returns (x, y) or None if parsing fails.
    """
    if pos_str is None:
        return None

    pos_str = pos_str.strip().strip('"')
    if not pos_str.startswith("array([") or not pos_str.endswith("])"):
        return None

    inner = pos_str[len("array(["):-2]
    parts = inner.split(",")
    if len(parts) != 2:
        return None

    try:
        x = float(parts[0])
        y = float(parts[1])
        return (x, y)
    except ValueError:
        return None


def load_fb_graph(directory: str = DATA_DIR):
    """
    Load the fb_friends graph from a directory containing:
        edges.csv
        nodes.csv

    Returns:
        G: networkx.Graph with node attributes
        node_meta: dict[node_index] -> metadata dict
    """
    edges_path = os.path.join(directory, "edges.csv")
    nodes_path = os.path.join(directory, "nodes.csv")

    if not os.path.exists(edges_path) or not os.path.exists(nodes_path):
        raise FileNotFoundError(
            f"Expected files edges.csv and nodes.csv inside directory '{directory}'"
        )

    # ---- Load edges ----
    edges = []
    with open(edges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src, tgt = line.split(",")
            edges.append((int(src), int(tgt)))

    # ---- Load nodes ----
    node_meta = {}
    with open(nodes_path, "r", encoding="utf-8") as f:
        # Skip header line (e.g., "# index, id, female, _pos")
        header = f.readline()

        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # Expect: [index, id, female, _pos]
            if len(row) < 4:
                continue

            idx_str, id_str, female_str, pos_str = row[0], row[1], row[2], row[3]
            idx = int(idx_str)
            node_id = int(id_str)
            female = int(female_str)
            pos = parse_pos(pos_str)

            node_meta[idx] = {
                "id": node_id,
                "female": female,
                "pos": pos,
            }

    # ---- Build Graph ----
    G = nx.Graph()
    G.add_nodes_from(node_meta.keys())
    G.add_edges_from(edges)

    # Attach attributes
    for n, meta in node_meta.items():
        for k, v in meta.items():
            G.nodes[n][k] = v

    return G, node_meta


def summarize_by_gender(values, node_meta):
    """
    values: dict[node] -> numeric (e.g., centrality score)
    node_meta: dict[node] -> {'female': 0/1, ...}

    Returns:
        {0: {"n": ..., "mean": ..., "median": ...},
         1: {...}}
    """
    groups = {0: [], 1: []}
    for node, val in values.items():
        if node not in node_meta:
            continue
        gender = node_meta[node]["female"]
        if gender in groups:
            groups[gender].append(val)

    out = {}
    for g, vals in groups.items():
        if vals:
            out[g] = {
                "n": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
            }
        else:
            out[g] = {"n": 0, "mean": None, "median": None}

    return out


def main():
    print("Loading fb_friends graph from directory:", DATA_DIR)
    G, node_meta = load_fb_graph(DATA_DIR)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # ------------------------------------------------------------------
    # Convert NetworkX Graph -> adjacency dict expected by analysis.py
    # ------------------------------------------------------------------
    adj = {u: set(G.neighbors(u)) for u in G.nodes()}

    # ------------------------------------------------------------------
    # Run your analysis.py functions (connectivity, centralities, etc.)
    # ------------------------------------------------------------------
    print("\n=== CONNECTIVITY ===")
    connectivity = analysis.compute_connectivity(adj)
    print(connectivity)

    print("\n=== CENTRALITY ===")
    deg_cent = analysis.degree_centrality(adj)
    clo_cent = analysis.closeness_centrality(adj)
    bet_cent = analysis.betweenness_centrality(adj)
    pr_cent = analysis.pagerank(adj)

    sample_nodes = list(adj.keys())[:5]
    print("Sample degree centrality (first 5 nodes):",
          {u: deg_cent[u] for u in sample_nodes})
    print("Sample betweenness (first 5 nodes):",
          {u: bet_cent[u] for u in sample_nodes})

    print("\n=== COMMUNITIES ===")
    communities = analysis.community_detection(adj)
    comm_sizes = [len(c) for c in communities]
    print(f"Found {len(communities)} communities.")
    print("Largest 10 community sizes:", sorted(comm_sizes, reverse=True)[:10])

    print("\n=== DEGREE DISTRIBUTION ===")
    deg_stats = analysis.compute_degree_distribution(adj)
    print("Average degree:", deg_stats["avg_degree"])
    print("Min degree:", deg_stats["min_degree"])
    print("Max degree:", deg_stats["max_degree"])

    # ------------------------------------------------------------------
    # Compare centrality distributions by gender
    # ------------------------------------------------------------------
    print("\n=== CENTRALITY VS GENDER (0 = male, 1 = female) ===")

    deg_by_gender = summarize_by_gender(deg_cent, node_meta)
    clo_by_gender = summarize_by_gender(clo_cent, node_meta)
    bet_by_gender = summarize_by_gender(bet_cent, node_meta)
    pr_by_gender = summarize_by_gender(pr_cent, node_meta)

    def pretty(label, stats_dict):
        print(f"\n{label}:")
        for g in sorted(stats_dict.keys()):
            stats_g = stats_dict[g]
            if stats_g["mean"] is None:
                print(f"  gender={g}: n=0")
            else:
                print(
                    f"  gender={g}: n={stats_g['n']}, "
                    f"mean={stats_g['mean']:.6f}, median={stats_g['median']:.6f}"
                )

    pretty("Degree centrality", deg_by_gender)
    pretty("Closeness centrality", clo_by_gender)
    pretty("Betweenness centrality", bet_by_gender)
    pretty("PageRank", pr_by_gender)

    # ------------------------------------------------------------------
    # Gender composition of largest communities
    # ------------------------------------------------------------------
    print("\n=== GENDER MIX IN TOP 10 COMMUNITIES ===")
    largest_10 = sorted(
        [(i, len(c)) for i, c in enumerate(communities)],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for cid, size in largest_10:
        nodes = communities[cid]
        g_counts = {0: 0, 1: 0}
        for u in nodes:
            if u in node_meta:
                g = node_meta[u]["female"]
                if g in g_counts:
                    g_counts[g] += 1

        print(
            f"Community {cid}: size={size}, "
            f"gender 0: {g_counts[0]}, gender 1: {g_counts[1]}"
        )


if __name__ == "__main__":
    main()
