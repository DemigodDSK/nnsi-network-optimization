# Methodology

For the full mathematical formulation and experimental design, see the Springer ICOMP'25 paper:

> *A Framework for Optimizing Network Topology Based on Graph Theory in Software-Defined Networking*
> Datta Sai Krishna Naidu, ICOMP'25

## Brief summary

**Problem.** In SDN, identifying which nodes are most critical (failure-sensitive) is a prerequisite for resilience-oriented topology optimization. Single-axis centrality metrics (degree, betweenness, etc.) miss nodes that are critical only when multiple roles are considered jointly.

**Approach — NNSI.** A composite metric combining three orthogonal scores:

1. **Centrality Score (CS).** Normalized blend of degree, betweenness, closeness, eigenvector, and PageRank centralities — captures positional importance in the graph.
2. **Failure Cascade Score (FCS).** Collective Influence (Morone–Makse, ℓ=2) × ClusterRank × local h-index — captures the node's role in failure propagation across the broader topology.
3. **Information Propagation Score (IPS).** Neighborhood connectivity × ego-network density — captures how effectively the node disseminates information through its local neighborhood.

The multiplicative form `NNSI = CS × FCS × IPS` penalizes nodes that excel on only one axis — only nodes critical across all three dimensions surface.

**Optimization step.** Top-k NNSI nodes (k = 10% of the network) are flagged as critical. The optimizer adds up to 5 new links per network to redistribute load and reduce dependence on those critical nodes, subject to topology constraints.

**Baseline.** Compared against IVI (Integrated Value of Influence, Salavaty et al.), which combines hubness × spreading × betweenness × CI in additive form.

**Evaluation.** Over 261 real networks from the Topology Zoo, post-optimization metrics measured: avg shortest-path length (latency proxy), algebraic connectivity (resilience proxy), and edge-load distribution (utilization proxy).

## Key result

NNSI surfaces a different top-10% set than IVI on ~80% of networks (avg overlap 90.2% means small ranking shifts at the boundary). When NNSI's selection drives optimization, **20% of networks see strictly better post-optimization metrics than under IVI**, with no networks getting worse.
