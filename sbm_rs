import networkx as nx
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix

# ================================
# 1. TEMPORAL BA GRAPH GENERATOR
# ================================

import networkx as nx
import random
import numpy as np

def generate_temporal_sbm_graph(n=200, sizes=[50, 50, 100], p_in=0.5, p_out=0.05, train_frac=0.7, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Generate SBM graph
    # sizes is a list of block sizes (i.e., how many nodes in each block)
    # p_in is the probability of an edge within the same block
    # p_out is the probability of an edge between different blocks
    p_matrix = np.full((len(sizes), len(sizes)), p_out)  # initialize with p_out
    np.fill_diagonal(p_matrix, p_in)  # fill diagonal with p_in for intra-block edges

    # Generate the SBM graph
    G_full = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    edges = list(G_full.edges())
    random.shuffle(edges)

    # Step 2: Split into train / test edges
    split_idx = int(len(edges) * train_frac)
    Eold = edges[:split_idx]   # Training edges
    Enew = edges[split_idx:]   # Test edges

    # Step 3: Build training graph
    G_train = nx.Graph()
    G_train.add_nodes_from(G_full.nodes())
    G_train.add_edges_from(Eold)

    return G_train, G_full, Eold, Enew



# =========================================
# 2. POSITIVE & NEGATIVE TEST PAIRS
# =========================================

def generate_test_pairs(G_full, Enew, num_neg=None):
    if num_neg is None:
        num_neg = len(Enew)

    non_edges = list(nx.non_edges(G_full))
    random.shuffle(non_edges)

    pos = Enew
    neg = non_edges[:num_neg]

    y_true = np.array([1]*len(pos) + [0]*len(neg))
    pairs = pos + neg

    return pairs, y_true


# =========================================
# 3. LINK PREDICTION SCORES
# =========================================

def common_neighbors_score(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))

def jaccard_score(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    if len(nu | nv) == 0:
        return 0
    return len(nu & nv) / len(nu | nv)

def adamic_adar_score(G, u, v):
    score = 0
    for w in nx.common_neighbors(G, u, v):
        deg = G.degree(w)
        if deg > 1:
            score += 1 / np.log(deg)
    return score

def resource_allocation_score(G, u, v):
    score = 0
    for w in nx.common_neighbors(G, u, v):
        deg = G.degree(w)
        if deg > 0:
            score += 1 / deg
    return score

def preferential_attachment_score(G, u, v):
    return G.degree(u) * G.degree(v)


# =========================================
# 4. ✅ SAFE KATZ (POWER SERIES VERSION)
# =========================================

def katz_scores_matrix_safe(G, beta=0.005, max_iter=5):
    A = nx.to_scipy_sparse_array(G, format="csr")
    n = A.shape[0]

    K = beta * A
    Ak = A.copy()

    for _ in range(2, max_iter + 1):
        Ak = Ak @ A
        K += (beta ** _) * Ak

    return K.toarray()

def katz_score(katz_mat, u, v):
    return katz_mat[u, v]


# =========================================
# 5. MODEL EVALUATION
# =========================================

def evaluate_model(G_train, pairs, y_true, method, katz_mat=None):
    scores = []

    for (u, v) in pairs:
        if method == "CN":
            scores.append(common_neighbors_score(G_train, u, v))
        elif method == "Jaccard":
            scores.append(jaccard_score(G_train, u, v))
        elif method == "AA":
            scores.append(adamic_adar_score(G_train, u, v))
        elif method == "RA":
            scores.append(resource_allocation_score(G_train, u, v))
        elif method == "PA":
            scores.append(preferential_attachment_score(G_train, u, v))
        elif method == "Katz":
            scores.append(katz_score(katz_mat, u, v))

    return roc_auc_score(y_true, scores)


# =========================================
# 6. RUN FULL EXPERIMENT
# =========================================

if __name__ == "__main__":

    print("\n=== GENERATING TEMPORAL BA GRAPH ===")
    G_train, G_full, Eold, Enew = generate_temporal_sbm_graph(n=200, sizes=[50, 50, 100], p_in=0.5, p_out=0.05, train_frac=0.7)

    print("Training Edges:", len(Eold))
    print("Test Edges:", len(Enew))

    print("\n=== GENERATING TEST PAIRS ===")
    pairs, y_true = generate_test_pairs(G_full, Enew)
    print("Total Test Pairs:", len(pairs))

    print("\n=== COMPUTING KATZ MATRIX (SAFE MODE) ===")
    katz_mat = katz_scores_matrix_safe(G_train)

    print("\n=== EVALUATING MODELS ===")
    methods = ["CN", "Jaccard", "AA", "RA", "PA", "Katz"]
    results = {}

    for method in methods:
        auc = evaluate_model(G_train, pairs, y_true, method, katz_mat)
        results[method] = auc
        print(f"{method} AUC: {auc:.4f}")

    best_model = max(results, key=results.get)
    print("\n✅ BEST RECOMMENDER ON BA GRAPH:", best_model)
