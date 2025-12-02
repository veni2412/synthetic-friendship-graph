from manim import *
import networkx as nx
import random
import argparse
import sys
import numpy as np

# =========================================================
# BIG FIVE PERSONALITY (DOMINANT TRAIT ONLY)
# =========================================================

def compute_big_five_personality(adj):
    deg = {u: len(neigh) for u, neigh in adj.items()}
    max_d = max(deg.values()) if deg else 1
    personalities = {}
    for u in adj:
        E = deg[u] / max_d
        O = random.random()
        A = random.random()
        C = random.random()
        N = random.random()
        traits = {"E": E, "O": O, "A": A, "C": C, "N": N}
        personalities[u] = max(traits, key=traits.get)
    return personalities

# =========================================================
# TRAIT COLORS
# =========================================================

TRAIT_COLORS = {
    "E": RED,
    "O": BLUE,
    "A": GREEN,
    "C": YELLOW,
    "N": PURPLE,
}

# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

def parse_custom_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BA", choices=["ER","BA","WS","SBM"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--p_intra", type=float, default=0.5)
    parser.add_argument("--p_inter", type=float, default=0.05)

    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])
    return args

# =========================================================
# GRAPH GENERATORS
# =========================================================

def build_graph(args):
    if args.model == "ER":
        G = nx.erdos_renyi_graph(args.n, args.p)
    elif args.model == "BA":
        G = nx.barabasi_albert_graph(args.n, args.m)
    elif args.model == "WS":
        G = nx.watts_strogatz_graph(args.n, args.k, args.beta)
    elif args.model == "SBM":
        sizes = [args.block_size]*args.blocks
        p_matrix = [[args.p_intra if i==j else args.p_inter for j in range(args.blocks)] for i in range(args.blocks)]
        G = nx.stochastic_block_model(sizes, p_matrix)
    else:
        raise ValueError(f"Unknown model {args.model}")
    return G

# =========================================================
# COMMUNITY DETECTION
# =========================================================

def get_communities(G):
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    return [list(c) for c in communities]

# =========================================================
# MANIM SCENE
# =========================================================

class FriendshipClusters(Scene):
    def construct(self):
        args = parse_custom_args()
        G = build_graph(args)
        adj = {u: list(G.neighbors(u)) for u in G.nodes()}

        personalities = compute_big_five_personality(adj)
        communities = get_communities(G)

        # Cluster layout
        k = len(communities)
        radius = max(2.5, 6 - k*0.9)
        angles = np.linspace(0, 2*np.pi, k, endpoint=False)
        cluster_centers = [np.array([radius*np.cos(a), radius*np.sin(a),0]) for a in angles]

        positions = {}
        for ci, group in enumerate(communities):
            center = cluster_centers[ci]
            for u in group:
                positions[u] = center + np.random.normal(scale=0.5, size=3)

        # Community highlight colors (pastels)
        community_colors = [
            "#ff9999",  # light red
            "#9999ff",  # light blue
            "#99ff99",  # light green
            "#ffff99",  # light yellow
            "#cc99ff",  # light purple
        ]

        # Highlight communities with transparent circles
        for ci, group in enumerate(communities):
            center = cluster_centers[ci]
            if group:
                nodes_positions = np.array([positions[u] for u in group])
                max_dist = np.max(np.linalg.norm(nodes_positions - center, axis=1))
                bubble = Circle(
                    radius=max_dist + 0.6,
                    color=community_colors[ci % len(community_colors)],
                    fill_opacity=0.15
                )
                bubble.move_to(center)
                self.add(bubble)

        # Draw nodes individually
        for u in G.nodes():
            color = TRAIT_COLORS[personalities[u]]
            dot = Dot(positions[u], radius=0.1, color=color)
            self.add(dot)
            self.wait(0.05)

        # Draw edges
        for u, v in G.edges():
            line = Line(positions[u], positions[v], stroke_width=1)
            self.add(line)
            self.wait(0.10)

        # Legend
        legend = VGroup(
            Text("E = Extraversion", font_size=24, color=RED),
            Text("O = Openness", font_size=24, color=BLUE),
            Text("A = Agreeableness", font_size=24, color=GREEN),
            Text("C = Conscientiousness", font_size=24, color=YELLOW),
            Text("N = Neuroticism", font_size=24, color=PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT)
        legend.to_corner(UR)
        self.play(FadeIn(legend))
        self.wait(4)
