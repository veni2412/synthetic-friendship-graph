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