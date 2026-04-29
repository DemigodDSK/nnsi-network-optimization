# NNSI — Network Node Significance Index

A composite centrality metric for identifying critical nodes and optimizing topology in Software-Defined Networks (SDN). Evaluated on **260 of 261 real-world networks** from the Internet Topology Zoo.

📄 **Reference paper:** *A Framework for Optimizing Network Topology Based on Graph Theory in Software-Defined Networking* — Springer ICOMP'25 (citation below)
📊 **Full 261-network run log:** [`experiments/full_run_log.txt`](experiments/full_run_log.txt)

## Quick demo (2 minutes)

```bash
git clone https://github.com/DemigodDSK/nnsi-network-optimization.git
cd nnsi-network-optimization
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m nnsi --demo
```

The demo runs on 5 sample networks (Forthnet, UsSignal, Geant2012, Kdl, Esnet) and prints a comparison table.

## Headline results (260 networks)

| Metric | Value |
|---|---|
| Networks successfully analyzed | **260** of 261 |
| NNSI outperforms IVI baseline on | **52 networks (20.0%)** |
| Average NNSI advantage (impact difference) | **0.0811** |
| Top-node overlap with IVI (avg) | 90.2% |
| **Avg latency reduction post-optimization** | **5.86%** |
| **Avg resilience improvement** | **10.64%** |
| **Avg link-utilization improvement** | **14.59%** |
| Full-run wallclock | 4151s (~69 min) |

## NNSI formulation

```
NNSI(v) = CS(v) × FCS(v) × IPS(v)
```

- **CS** — Centrality Score (composite of degree, betweenness, closeness, eigenvector, PageRank)
- **FCS** — Failure Cascade Score (collective influence + clusterrank + local h-index)
- **IPS** — Information Propagation Score (neighborhood connectivity + ego-network density)

The product structure penalizes nodes that score high on only one dimension, surfacing nodes critical along multiple network axes. Compared against the IVI (Integrated Value of Influence) baseline.

## Reproduce the full 261-network run

```bash
# Topology Zoo: http://www.topology-zoo.org/dataset.html (CC BY 4.0)
python -m nnsi --data-dir /path/to/your/topology_zoo
```

Expect ~70 min on a single CPU (the bundled `experiments/full_run_log.txt` records the original run).

## Project layout

```
nnsi-network-optimization/
├── nnsi/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry — python -m nnsi --demo
│   └── analyzer.py              # NetworkAnalyzer class (centrality, NNSI, IVI, optimization)
├── data/topology_zoo_sample/    # 5 GraphML files for the demo
├── experiments/full_run_log.txt # Original 261-network run output
├── pyproject.toml
└── requirements.txt
```

## Stack

Python 3.11+ · NetworkX · NumPy · SciPy · pandas · matplotlib · seaborn · tqdm

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{naidu2025nnsi,
  title     = {A Framework for Optimizing Network Topology Based on Graph Theory in Software-Defined Networking},
  author    = {Naidu, Datta Sai Krishna},
  booktitle = {Proceedings of the International Conference on Computational Methods (ICOMP'25)},
  publisher = {Springer},
  year      = {2025}
}
```

## Data attribution

Topology data is from the **Internet Topology Zoo** project (Knight et al.), licensed CC BY 4.0. See http://www.topology-zoo.org/

---

MIT License · © 2026 Datta Sai Krishna Naidu
