Synthetic social network toolkit for generation, analysis, comparison, and recommendations using classic random graph models and graph algorithms.

**Highlights**
- Synthetic graph generation (ER, WS, BA, SBM)
- Connectivity, centrality, clustering, communities
- Big Five personality tagging from structure
- Homophily inspection
- Cross-model comparison with auto plots
- Personality-aware friend recommender

**Project Structure**
- `erdos.py`: Erdős–Rényi `G(n, p)`
- `ws.py`: Watts–Strogatz (small-world)
- `ba.py`: Barabási–Albert (scale-free)
- `sbm.py`: Stochastic Block Model (symmetric)
- `analysis.py`: Core analysis, personalities, CSV export, visualization
- `compare.py`: Multi-model generation + comparison plots
- `recommender.py`: Link prediction + personality recommendations

**Analysis (`analysis.py`)**
- **Connectivity**: components, giant component, isolated nodes
- **Centrality**: degree, closeness, betweenness (Brandes), PageRank
- **Clustering**: local coefficient per node
- **Communities**: greedy modularity optimization
- **Personality (Big Five)**:
  - `E`: degree centrality
  - `O`: betweenness
  - `A`: clustering
  - `C`: closeness
  - `N`: inverse of (degree + clustering)
- **Extras**: homophily utilities, degree stats, CSV export, personality scatter (E vs A)

**Model Comparison (`compare.py`)**
- Generates ER, BA, WS, SBM with comparable parameters
- Computes per-run summaries:
  - nodes, edges, avg degree
  - components, giant component size
  - avg clustering, approx avg shortest path (giant component)
  - communities (count)
- Saves plots:
  - `avg_degree_by_model.png`
  - `avg_clustering_by_model.png`
  - `avg_path_by_model.png`
  - `giant_component_by_model.png`
  - histograms: degree, clustering, path, component size

**Friend Recommender (`recommender.py`)**
- Structural scores: CN, Jaccard, PA, Adamic–Adar, Resource Allocation, Katz
- Personality similarity: cosine of Big Five vectors
- Min–max normalization + weighted sum:
  - `final = w_cn*CN + w_jac*Jacc + ... + w_katz*Katz + w_pers*Personality`

**Quick Start**

Analyze a single graph (SBM example):

```bash
python analysis.py \
  --model SBM \
  --n 500 \
  --blocks 4 \
  --block-size 125 \
  --p-intra 0.12 \
  --p-inter 0.02 \
  --seed 42 \
  --out-csv sbm_personalities.csv \
  --out-plot sbm_personality_scatter.png
```

Compare ER, BA, WS, SBM:

```bash
python compare.py
```

Generate friend recommendations (BA example):

```bash
python recommender.py \
  --model BA --n 500 --m 3 --seed 42 \
  --user 10 --top-k 10 \
  --w-cn 0.5 --w-jac 0.5 \
  --w-aa 2.0 --w-pers 3.0
```

**Outputs**
- Plots: personality scatter, model comparisons, histograms
- CSVs: node metrics + Big Five traits, summary tables

**Goals**
- Understand differences among ER/BA/WS/SBM
- Visualize centrality, connectivity, clustering, communities
- Assign personality traits and study homophily
- Build practical friend recommender combining local/global/personality signals

**Notes**
- Synthetic graphs with fully controlled parameters
- Modular and extensible to weighted, temporal, ML evaluation, and real datasets