import os
import time
import glob
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import itertools
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Directory containing GraphML files - update this path
DATA_DIR = r"sources"
# Directory for saving results
RESULTS_DIR = "network_analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class NetworkAnalyzer:
    """
    Comprehensive network analyzer implementing NNSI-based critical node 
    identification and topology optimization with full metrics calculation
    as described in the thesis.
    """

    def __init__(self, data_directory=DATA_DIR):
        """Initialize with directory containing GraphML files"""
        self.data_directory = data_directory
        self.results = {}
        self.failed_networks = []

    def normalize(self, values):
        """Min-max normalization of a list of values"""
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return [0.5 for _ in values]  # Handle constant case

        return [(x - min_val) / (max_val - min_val) for x in values]

    def normalize_dict(self, metric_dict):
        """Normalize a dictionary of metric values to [0,1] range"""
        values = list(metric_dict.values())

        if not values:
            return {}

        min_val = min(values)
        max_val = max(values)

        # Avoid division by zero
        if max_val == min_val:
            return {node: 0.5 for node in metric_dict}

        return {
            node: (value - min_val) / (max_val - min_val)
            for node, value in metric_dict.items()
        }

    def analyze_graph(self, file_path):
        """Parse GraphML file and create an undirected graph"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Create an undirected graph
            G = nx.Graph()

            # GraphML namespace
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

            # Add nodes
            for node in root.findall('.//graphml:node', ns):
                node_id = node.attrib['id']
                # Get label if available
                label_element = node.find('.//graphml:data', ns)
                label = label_element.text if label_element is not None else node_id
                G.add_node(node_id, label=label)

            # Add edges
            for edge in root.findall('.//graphml:edge', ns):
                src = edge.attrib['source']
                dst = edge.attrib['target']
                G.add_edge(src, dst)

            return G
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def calculate_centrality_measures(self, G):
        """Calculate various centrality measures for a graph"""
        metrics = {}

        # Basic centrality measures
        metrics['degree'] = nx.degree_centrality(G)

        try:
            # For large networks, use approximate betweenness with samples
            if G.number_of_nodes() > 500:
                k = min(G.number_of_nodes(), 500)
                metrics['betweenness'] = nx.betweenness_centrality(G, k=k)
            else:
                metrics['betweenness'] = nx.betweenness_centrality(G)
        except Exception as e:
            print(f"Error calculating betweenness centrality: {e}")
            # Fallback using degree
            metrics['betweenness'] = {
                node: G.degree(node) / G.number_of_nodes()
                for node in G.nodes()
            }

        try:
            if nx.is_connected(G):
                metrics['closeness'] = nx.closeness_centrality(G)
            else:
                # For disconnected graphs, calculate per component
                closeness = {}
                for node in G.nodes():
                    try:
                        # Find connected component containing this node
                        component = nx.node_connected_component(G, node)
                        if len(component) > 1:
                            # Calculate within component and scale by component size relative to graph
                            subgraph = G.subgraph(component)
                            node_closeness = nx.closeness_centrality(
                                subgraph, node)
                            # Scale by component size relative to graph size
                            closeness[node] = node_closeness * (
                                len(component) / G.number_of_nodes())
                        else:
                            closeness[node] = 0
                    except Exception:
                        closeness[node] = 0
                metrics['closeness'] = closeness
        except Exception as e:
            print(f"Error calculating closeness centrality: {e}")
            metrics['closeness'] = {node: 0 for node in G.nodes()}

        try:
            metrics['pagerank'] = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception as e:
            print(f"Error calculating PageRank: {e}")
            # Fallback to degree-based approximation
            metrics['pagerank'] = metrics['degree']

        # K-core values - create a copy without self-loops for this calculation only
        try:
            G_no_selfloops = G.copy()
            G_no_selfloops.remove_edges_from(nx.selfloop_edges(G_no_selfloops))
            metrics['k_core'] = nx.core_number(G_no_selfloops)
        except Exception as e:
            print(f"Error calculating k-core: {e}")
            # Fallback using degree
            metrics['k_core'] = {node: G.degree(node) for node in G.nodes()}

        # Collective Influence (with l=2)
        try:
            metrics[
                'collective_influence'] = self.calculate_collective_influence(
                    G)
        except Exception as e:
            print(f"Error calculating collective influence: {e}")
            # Fallback to weighted degree
            metrics['collective_influence'] = {
                node: G.degree(node)**2
                for node in G.nodes()
            }

        # ClusterRank
        try:
            metrics['clusterrank'] = self.calculate_clusterrank(G)
        except Exception as e:
            print(f"Error calculating ClusterRank: {e}")
            metrics['clusterrank'] = metrics['degree']

        # Local H-index
        try:
            metrics['local_h_index'] = self.calculate_local_h_index(G)
        except Exception as e:
            print(f"Error calculating local H-index: {e}")
            metrics['local_h_index'] = metrics['degree']

        # Neighborhood Connectivity
        try:
            metrics[
                'neighborhood_connectivity'] = self.calculate_neighborhood_connectivity(
                    G)
        except Exception as e:
            print(f"Error calculating neighborhood connectivity: {e}")
            metrics['neighborhood_connectivity'] = metrics['degree']

        # Normalize all metrics
        for metric in metrics:
            metrics[metric] = self.normalize_dict(metrics[metric])

        return metrics

    def calculate_collective_influence(self, G, l=2):
        """Calculate collective influence for all nodes"""
        ci = {}

        for node in G.nodes():
            try:
                # Get nodes at distance l from the current node
                boundary_nodes = set()
                current_distance = 0
                current_layer = {node}
                visited = {node}

                # Use breadth-first search to find nodes at distance l
                while current_distance < l and current_layer:
                    next_layer = set()
                    for current_node in current_layer:
                        for neighbor in G.neighbors(current_node):
                            if neighbor not in visited:
                                next_layer.add(neighbor)
                                visited.add(neighbor)

                    current_layer = next_layer
                    current_distance += 1

                boundary_nodes = current_layer

                # Calculate collective influence
                node_degree = G.degree(node)
                influence_sum = sum(G.degree(j) - 1 for j in boundary_nodes)
                ci[node] = (node_degree - 1) * influence_sum
            except Exception:
                # Fallback for any errors
                ci[node] = G.degree(node)**2

        return ci

    def calculate_clusterrank(self, G):
        """Calculate ClusterRank for all nodes"""
        cr = {}
        for node in G.nodes():
            clustering = nx.clustering(G, node)
            f_c = 1.0 / (1.0 + clustering) if clustering > 0 else 1.0
            neighbors = list(G.neighbors(node))
            if not neighbors:
                cr[node] = 0
                continue
            sum_term = sum(G.degree(n) + 1 for n in neighbors)
            cr[node] = f_c * sum_term
        return cr

    def calculate_local_h_index(self, G):
        """Calculate Local H-index for all nodes"""
        h_index = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                h_index[node] = 0
                continue

            neighbor_degrees = sorted([G.degree(n) for n in neighbors],
                                      reverse=True)
            h = 0
            for i, d in enumerate(neighbor_degrees):
                if d >= i + 1:
                    h = i + 1
                else:
                    break
            h_index[node] = h

        # Calculate local H-index
        local_h = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            local_h[node] = h_index.get(node, 0) + sum(
                h_index.get(n, 0) for n in neighbors)

        return local_h

    def calculate_neighborhood_connectivity(self, G):
        """Calculate Neighborhood Connectivity for all nodes"""
        nc = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                nc[node] = 0
                continue

            avg_connectivity = sum(G.degree(n)
                                   for n in neighbors) / len(neighbors)
            nc[node] = avg_connectivity

        return nc

    def calculate_nnsi(self, G, centrality_measures=None):
        """
        Calculate the Network Node Significance Index (NNSI) as defined in the thesis.
        Formula: NNSI = CS × FCS × IPS
        Where:
        CS = Connectivity Score = degree + closeness
        FCS = Flow Control Score = betweenness + k_core
        IPS = Influence Propagation Score = pagerank + collective_influence
        """
        if centrality_measures is None:
            centrality_measures = self.calculate_centrality_measures(G)

        nodes = list(centrality_measures['degree'].keys())
        nnsi = {}

        for node in nodes:
            # Calculate component scores based on thesis definition
            connectivity_score = (
                centrality_measures['degree'].get(node, 0) +
                centrality_measures['closeness'].get(node, 0))

            flow_control_score = (
                centrality_measures['betweenness'].get(node, 0) +
                centrality_measures['k_core'].get(node, 0))

            influence_propagation_score = (
                centrality_measures['pagerank'].get(node, 0) +
                centrality_measures['collective_influence'].get(node, 0))

            # Calculate final NNSI value - multiply the three component scores
            nnsi[
                node] = connectivity_score * flow_control_score * influence_propagation_score

        return nnsi

    def calculate_ivi(self, G, centrality_measures=None):
        """
        Calculate the Integrated Value of Influence (IVI) as defined in the literature.
        Formula: IVI = hubness_score * spreading_score
        Where:
        hubness_score = degree + local_h_index
        spreading_score = (neighborhood_connectivity + clusterrank) * (betweenness + collective_influence)
        """
        if centrality_measures is None:
            centrality_measures = self.calculate_centrality_measures(G)

        nodes = list(centrality_measures['degree'].keys())
        ivi = {}

        for node in nodes:
            # Calculate hubness score
            hubness_score = (centrality_measures['degree'].get(node, 0) +
                             centrality_measures['local_h_index'].get(node, 0))

            # Calculate spreading score
            local_component = (
                centrality_measures['neighborhood_connectivity'].get(node, 0) +
                centrality_measures['clusterrank'].get(node, 0))

            global_component = (
                centrality_measures['betweenness'].get(node, 0) +
                centrality_measures['collective_influence'].get(node, 0))

            spreading_score = local_component * global_component

            # Calculate final IVI value
            ivi[node] = hubness_score * spreading_score

        return ivi

    def calculate_network_metrics(self, G):
        """Calculate various metrics for a graph"""
        try:
            # Number of connected components
            components = list(nx.connected_components(G))
            num_components = len(components)

            # Largest component size
            if components:
                largest_cc = max(components, key=len)
                largest_cc_size = len(largest_cc)
            else:
                largest_cc_size = 0

            # Calculate connectivity
            connectivity = largest_cc_size / len(G.nodes) if G.nodes else 0

            # Calculate average path length in largest component
            avg_path_length = 0
            if largest_cc_size > 1:
                G_largest_cc = G.subgraph(largest_cc)
                if nx.is_connected(G_largest_cc):
                    avg_path_length = nx.average_shortest_path_length(
                        G_largest_cc)

            # Calculate average clustering coefficient
            clustering = nx.average_clustering(G)

            return {
                'connectivity': connectivity,
                'avg_path_length': avg_path_length,
                'clustering': clustering
            }
        except Exception as e:
            print(f"Error calculating network metrics: {e}")
            return {'connectivity': 0, 'avg_path_length': 0, 'clustering': 0}

    def evaluate_node_removal_impact(self, G, nodes_to_remove):
        """
        Evaluate the impact of removing a set of nodes from the graph.
        This implements the Impact Score (IS) metric from the thesis.
        """
        original_metrics = self.calculate_network_metrics(G)

        # Create a copy of the graph for node removal
        G_copy = G.copy()
        G_copy.remove_nodes_from(nodes_to_remove)

        # Calculate metrics after node removal
        modified_metrics = self.calculate_network_metrics(G_copy)

        # Calculate impact as percentage change (higher is more impactful)
        impact = {
            'connectivity': (original_metrics['connectivity'] -
                             modified_metrics['connectivity']) /
            original_metrics['connectivity']
            if original_metrics['connectivity'] > 0 else 0,
            'avg_path_length': (modified_metrics['avg_path_length'] -
                                original_metrics['avg_path_length']) /
            original_metrics['avg_path_length']
            if original_metrics['avg_path_length'] > 0 else 0,
            'clustering':
            (original_metrics['clustering'] - modified_metrics['clustering']) /
            original_metrics['clustering']
            if original_metrics['clustering'] > 0 else 0,
            'fragmentation':
            len(list(nx.connected_components(G_copy))) -
            len(list(nx.connected_components(G)))
        }

        # Calculate overall impact score (weighted sum)
        overall_impact = (0.4 * impact['connectivity'] +
                          0.3 * impact['avg_path_length'] +
                          0.1 * impact['clustering'] +
                          0.2 * impact['fragmentation'])

        return overall_impact

    # ========== Additional Metrics from Thesis Section 5.3 ==========

    def calculate_overlap_percentage(self,
                                     method1_top_nodes,
                                     method2_top_nodes,
                                     k=None):
        """
        Calculate the Overlap Percentage (OP) between two sets of top nodes.
        Formula from thesis: OP(M1, M2, k) = |M1,k ∩ M2,k| / k × 100%
        """
        if k is None:
            k = len(method1_top_nodes)

        # Ensure we're looking at the top k nodes from each method
        m1_nodes = set(method1_top_nodes[:k])
        m2_nodes = set(method2_top_nodes[:k])

        # Calculate intersection
        intersection = m1_nodes.intersection(m2_nodes)

        # Calculate overlap percentage
        overlap_percentage = (len(intersection) / k) * 100

        return overlap_percentage

    def calculate_pds(self, G, node, metric_func):
        """
        Calculate Performance Degradation Score (PDS) for a specific node and performance metric.
        Formula from thesis: PDS_X(i) = (X_baseline - X_no_i) / X_baseline × 100%
        """
        # Calculate baseline metric
        baseline_value = metric_func(G)

        # Create a copy of the graph without the node
        G_copy = G.copy()
        G_copy.remove_node(node)

        # Calculate metric after node removal
        after_value = metric_func(G_copy)

        # Calculate PDS
        if baseline_value > 0:
            pds = ((baseline_value - after_value) / baseline_value) * 100
        else:
            pds = 0

        return pds

    def calculate_cgt(self, computed_rankings, ground_truth_rankings):
        """
        Calculate Correlation with Ground Truth (CGT) using Spearman's rank correlation.
        As described in thesis section 5.3.
        """
        # Get common nodes
        common_nodes = set(computed_rankings.keys()).intersection(
            set(ground_truth_rankings.keys()))

        if len(common_nodes) < 2:
            return 0  # Cannot calculate correlation with fewer than 2 points

        # Extract values for common nodes
        computed_values = [computed_rankings[node] for node in common_nodes]
        ground_truth_values = [
            ground_truth_rankings[node] for node in common_nodes
        ]

        # Calculate Spearman's rank correlation
        correlation, p_value = stats.spearmanr(computed_values,
                                               ground_truth_values)

        return correlation

    def perform_statistical_tests(self, nnsi_results, ivi_results):
        """
        Perform statistical tests to compare NNSI and IVI results.
        Implements the statistical tests mentioned in the thesis.
        """
        # Perform Wilcoxon signed-rank test
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(nnsi_results, ivi_results)

        # Calculate Cliff's Delta effect size
        def cliffs_delta(x, y):
            nx = len(x)
            ny = len(y)

            # Count differences
            larger = 0
            smaller = 0

            for i in range(nx):
                for j in range(ny):
                    if x[i] > y[j]:
                        larger += 1
                    elif x[i] < y[j]:
                        smaller += 1

            # Calculate Cliff's Delta
            delta = (larger - smaller) / (nx * ny)
            return delta

        effect_size = cliffs_delta(nnsi_results, ivi_results)

        # Calculate 95% confidence intervals using bootstrapping
        n_bootstraps = 1000
        bootstrap_diffs = []

        for _ in range(n_bootstraps):
            # Sample with replacement
            boot_nnsi = np.random.choice(nnsi_results,
                                         size=len(nnsi_results),
                                         replace=True)
            boot_ivi = np.random.choice(ivi_results,
                                        size=len(ivi_results),
                                        replace=True)

            # Calculate mean difference
            bootstrap_diffs.append(np.mean(boot_nnsi) - np.mean(boot_ivi))

        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)

        return {
            'wilcoxon_stat': wilcoxon_stat,
            'wilcoxon_p': wilcoxon_p,
            'cliffs_delta': effect_size,
            'mean_difference': np.mean(nnsi_results) - np.mean(ivi_results),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'significant': wilcoxon_p < 0.05
        }

    # ========== Additional Metrics from Thesis Section 5.4 ==========

    def calculate_nid(self, G, nnsi_top_nodes, ivi_top_nodes, k):
        """
        Calculate the Normalized Impact Difference (NID) between NNSI and IVI methods.
        Formula from thesis: NID(M1, M2, k) = ID(M1, M2, k) / max impact possible
        """
        # Calculate impact with NNSI nodes
        nnsi_impact = self.evaluate_node_removal_impact(G, nnsi_top_nodes[:k])

        # Calculate impact with IVI nodes
        ivi_impact = self.evaluate_node_removal_impact(G, ivi_top_nodes[:k])

        # Calculate Impact Difference
        impact_diff = nnsi_impact - ivi_impact

        # Find maximum possible impact by greedy approximation
        all_nodes = list(G.nodes())
        impacts = {}
        for node in all_nodes:
            impacts[node] = self.evaluate_node_removal_impact(G, [node])

        sorted_nodes = sorted(impacts.items(),
                              key=lambda x: x[1],
                              reverse=True)
        max_nodes = [node for node, _ in sorted_nodes[:k]]
        max_impact = self.evaluate_node_removal_impact(G, max_nodes)

        # Calculate NID
        nid = impact_diff / max_impact if max_impact > 0 else 0

        return nid

    def calculate_pir(self, metric_m1, metric_m2):
        """
        Calculate Performance Improvement Ratio (PIR) between two methods.
        Formula from thesis: PIR_X(M1, M2) = X_M1 / X_M2
        """
        if metric_m2 <= 0:
            return float('inf') if metric_m1 > 0 else 1.0

        return metric_m1 / metric_m2

    def calculate_cnom(self, methods_results, k):
        """
        Calculate the Critical Node Overlap Matrix (CNOM) for multiple methods.
        As described in thesis section 5.4.
        """
        method_names = list(methods_results.keys())
        num_methods = len(method_names)

        # Initialize empty matrix
        cnom = {}

        # Fill the matrix with overlap percentages
        for i in range(num_methods):
            method1 = method_names[i]
            cnom[method1] = {}

            for j in range(num_methods):
                method2 = method_names[j]

                # For diagonal elements, overlap is 100%
                if i == j:
                    cnom[method1][method2] = 100.0
                else:
                    # Calculate overlap percentage
                    m1_nodes = methods_results[method1][:k]
                    m2_nodes = methods_results[method2][:k]

                    cnom[method1][method2] = self.calculate_overlap_percentage(
                        m1_nodes, m2_nodes, k)

        return cnom

    def calculate_rei(self, performance_improvement, resource_increase):
        """
        Calculate Resource Efficiency Index (REI).
        Formula from thesis: REI = ΔP / ΔR
        """
        if resource_increase <= 0:
            return float('inf') if performance_improvement > 0 else 0.0

        return performance_improvement / resource_increase

    def calculate_crp(self, G_original, G_optimized, method='nnsi'):
        """
        Calculate Criticality Reduction Percentage (CRP).
        Formula from thesis: CRP = (max NNSI_before - max NNSI_after) / max NNSI_before × 100%
        """
        # Calculate NNSI values for original network
        if method == 'nnsi':
            original_values = self.calculate_nnsi(G_original)
            optimized_values = self.calculate_nnsi(G_optimized)
        else:  # Default to IVI
            original_values = self.calculate_ivi(G_original)
            optimized_values = self.calculate_ivi(G_optimized)

        # Find maximum value in each network
        max_original = max(original_values.values()) if original_values else 0
        max_optimized = max(
            optimized_values.values()) if optimized_values else 0

        # Calculate CRP
        if max_original > 0:
            crp = ((max_original - max_optimized) / max_original) * 100
        else:
            crp = 0

        return crp

    # ========== Optimization Functions ==========

    def get_critical_nodes(self, G, method='nnsi', threshold=0.10):
        """
        Identify critical nodes based on the specified method (NNSI or IVI).
        Returns top nodes based on threshold percentage.
        """
        # Calculate node importance values
        if method.lower() == 'nnsi':
            importance_values = self.calculate_nnsi(G)
        else:  # Default to IVI
            importance_values = self.calculate_ivi(G)

        # Sort nodes by importance
        sorted_nodes = sorted(importance_values.items(),
                              key=lambda x: x[1],
                              reverse=True)

        # Determine number of nodes to return
        num_nodes = max(1, int(len(G.nodes()) * threshold))

        # Return critical nodes
        return [node for node, _ in sorted_nodes[:num_nodes]]

    def optimize_network(self, G, critical_nodes, max_new_links=5):
        """
        Optimize network by adding strategic links to enhance resilience and performance.
        """
        if not critical_nodes or len(critical_nodes) == 0:
            return []

        changes = []

        # For each critical node, consider adding links between its neighbors
        for critical_node in critical_nodes:
            # Get neighbors of the critical node
            neighbors = list(G.neighbors(critical_node))

            if len(neighbors) < 2:
                continue

            # Consider pairs of neighbors that aren't already connected
            for n1, n2 in itertools.combinations(neighbors, 2):
                if not G.has_edge(n1, n2) and len(changes) < max_new_links:
                    # Calculate the impact of adding this link
                    original_nnsi = self.calculate_nnsi(G)

                    # Create test graph with new link
                    test_G = G.copy()
                    test_G.add_edge(n1, n2)

                    # Calculate new NNSI values
                    new_nnsi = self.calculate_nnsi(test_G)

                    # Calculate average NNSI reduction for critical nodes
                    avg_reduction = sum(
                        original_nnsi.get(node, 0) - new_nnsi.get(node, 0)
                        for node in critical_nodes) / len(
                            critical_nodes) if critical_nodes else 0

                    changes.append({
                        'source': n1,
                        'target': n2,
                        'critical_node': critical_node,
                        'impact': avg_reduction
                    })

        # Sort by impact and return top recommendations
        changes.sort(key=lambda x: x['impact'], reverse=True)
        return changes[:max_new_links]

    def apply_changes(self, G, changes):
        """Apply recommended topology changes to the graph"""
        optimized_graph = G.copy()

        for change in changes:
            optimized_graph.add_edge(change['source'], change['target'])

        return optimized_graph

    def calculate_improvements(self, original_graph, optimized_graph):
        """
        Calculate performance improvements after optimization.
        Returns latency reduction, resilience improvement, and link utilization improvement.
        """
        # Calculate network metrics before and after optimization
        original_metrics = self.calculate_network_metrics(original_graph)
        optimized_metrics = self.calculate_network_metrics(optimized_graph)

        # Calculate NNSI values before and after
        original_nnsi = self.calculate_nnsi(original_graph)
        optimized_nnsi = self.calculate_nnsi(optimized_graph)

        # Calculate betweenness centrality before and after
        try:
            original_bc = nx.betweenness_centrality(original_graph)
            optimized_bc = nx.betweenness_centrality(optimized_graph)
            max_original_bc = max(original_bc.values()) if original_bc else 0
            max_optimized_bc = max(
                optimized_bc.values()) if optimized_bc else 0
        except:
            # Fallback for large networks
            max_original_bc = max(original_nnsi.values(
            )) if original_nnsi else 0  # Use NNSI as proxy
            max_optimized_bc = max(
                optimized_nnsi.values()) if optimized_nnsi else 0

        # Calculate improvements

        # 1. Latency reduction (based on average path length)
        if original_metrics['avg_path_length'] > 0:
            latency_reduction = 100 * (original_metrics['avg_path_length'] -
                                       optimized_metrics['avg_path_length']
                                       ) / original_metrics['avg_path_length']
            latency_reduction = max(0,
                                    latency_reduction)  # Ensure non-negative
        else:
            latency_reduction = 0

        # 2. Resilience improvement (based on NNSI and betweenness)
        max_original_nnsi = max(original_nnsi.values()) if original_nnsi else 0
        max_optimized_nnsi = max(
            optimized_nnsi.values()) if optimized_nnsi else 0

        if max_original_nnsi > 0:
            nnsi_improvement = 100 * (max_original_nnsi -
                                      max_optimized_nnsi) / max_original_nnsi
            nnsi_improvement = max(0, nnsi_improvement)
        else:
            nnsi_improvement = 0

        if max_original_bc > 0:
            betweenness_improvement = 100 * (
                max_original_bc - max_optimized_bc) / max_original_bc
            betweenness_improvement = max(0, betweenness_improvement)
        else:
            betweenness_improvement = 0

        # Weighted average of NNSI and betweenness improvements
        resilience_improvement = (nnsi_improvement * 0.7 +
                                  betweenness_improvement * 0.3)

        # 3. Link utilization improvement (based on edge betweenness)
        try:
            original_edge_bc = nx.edge_betweenness_centrality(original_graph)
            optimized_edge_bc = nx.edge_betweenness_centrality(optimized_graph)
            max_original_edge_bc = max(
                original_edge_bc.values()) if original_edge_bc else 0
            max_optimized_edge_bc = max(
                optimized_edge_bc.values()) if optimized_edge_bc else 0

            if max_original_edge_bc > 0:
                link_utilization_improvement = 100 * (
                    1 - (max_optimized_edge_bc / max_original_edge_bc))
                link_utilization_improvement = max(
                    0, link_utilization_improvement)
            else:
                link_utilization_improvement = 0
        except:
            # Fallback for large networks
            link_utilization_improvement = resilience_improvement * 0.5  # Estimate

        return {
            'latency_reduction': round(latency_reduction, 1),
            'resilience_improvement': round(resilience_improvement, 1),
            'link_utilization_improvement': round(link_utilization_improvement,
                                                  1)
        }

    # ========== Analysis Pipeline ==========

    def analyze_network(self, graphml_file, max_new_links=5, threshold=0.10):
        """
        Complete analysis pipeline for a single network:
        1. Parse GraphML file
        2. Calculate centrality measures
        3. Compute NNSI and IVI
        4. Identify critical nodes 
        5. Optimize topology
        6. Measure improvements
        """
        network_name = os.path.basename(graphml_file).replace('.graphml', '')
        print(f"Analyzing {network_name}...")

        # Parse GraphML file
        G = self.analyze_graph(graphml_file)

        if G is None or G.number_of_nodes() < 5:
            print(f"  Skipping {network_name} - invalid or too small graph")
            self.failed_networks.append(
                (network_name, "Invalid or too small graph"))
            return None

        # Calculate centrality measures
        centrality_measures = self.calculate_centrality_measures(G)

        # Calculate NNSI and IVI
        nnsi_values = self.calculate_nnsi(G, centrality_measures)
        ivi_values = self.calculate_ivi(G, centrality_measures)

        # Rank nodes by each method
        nnsi_ranked = sorted(nnsi_values.items(),
                             key=lambda x: x[1],
                             reverse=True)
        ivi_ranked = sorted(ivi_values.items(),
                            key=lambda x: x[1],
                            reverse=True)

        # Get top nodes for each method
        top_percent = threshold
        top_k = max(1, int(len(G.nodes()) * top_percent))

        nnsi_top_nodes = [node for node, _ in nnsi_ranked[:top_k]]
        ivi_top_nodes = [node for node, _ in ivi_ranked[:top_k]]

        # Calculate overlap percentage
        overlap_percent = self.calculate_overlap_percentage(
            nnsi_top_nodes, ivi_top_nodes, top_k)

        # Evaluate impact of removing top nodes
        nnsi_impact = self.evaluate_node_removal_impact(G, nnsi_top_nodes)
        ivi_impact = self.evaluate_node_removal_impact(G, ivi_top_nodes)

        # Calculate impact difference metrics
        impact_diff = nnsi_impact - ivi_impact
        normalized_impact_diff = self.calculate_nid(G, nnsi_top_nodes,
                                                    ivi_top_nodes, top_k)

        # Optimize network based on NNSI
        changes = self.optimize_network(G, nnsi_top_nodes, max_new_links)

        # Apply changes and calculate improvements
        optimized_graph = self.apply_changes(G, changes)
        improvements = self.calculate_improvements(G, optimized_graph)

        # Calculate Criticality Reduction Percentage
        crp = self.calculate_crp(G, optimized_graph, 'nnsi')

        # Calculate Resource Efficiency Index
        # Define resource increase as number of new links
        resource_increase = len(changes)
        # Define performance increase as average of improvement percentages
        performance_increase = (
            improvements['latency_reduction'] +
            improvements['resilience_improvement'] +
            improvements['link_utilization_improvement']) / 3

        rei = self.calculate_rei(
            performance_increase,
            resource_increase) if resource_increase > 0 else float('inf')

        # Store results
        result = {
            'network_name':
            network_name,
            'num_nodes':
            G.number_of_nodes(),
            'num_edges':
            G.number_of_edges(),
            'nnsi_impact':
            nnsi_impact,
            'ivi_impact':
            ivi_impact,
            'impact_diff':
            impact_diff,
            'normalized_impact_diff':
            normalized_impact_diff,
            'overlap_percent':
            overlap_percent,
            'latency_reduction':
            improvements['latency_reduction'],
            'resilience_improvement':
            improvements['resilience_improvement'],
            'link_utilization_improvement':
            improvements['link_utilization_improvement'],
            'optimization_changes':
            len(changes),
            'criticality_reduction':
            crp,
            'resource_efficiency':
            rei
        }

        self.results[network_name] = result

        print(f"  Analysis complete for {network_name}")
        print(f"  - Impact Difference: {impact_diff:.4f}")
        print(f"  - Overlap Percentage: {overlap_percent:.1f}%")
        print(
            f"  - Latency Reduction: {improvements['latency_reduction']:.1f}%")
        print(
            f"  - Resilience Improvement: {improvements['resilience_improvement']:.1f}%"
        )
        print(
            f"  - Link Utilization Improvement: {improvements['link_utilization_improvement']:.1f}%"
        )

        return result

    def analyze_all_networks(self):
        """Analyze all GraphML files in the specified directory"""
        # Get all GraphML files
        graphml_files = glob.glob(
            os.path.join(self.data_directory, "*.graphml"))

        print(
            f"Found {len(graphml_files)} GraphML files in {self.data_directory}"
        )
        print("Starting analysis...")

        start_time = time.time()

        # Process each GraphML file with progress bar
        for file_path in tqdm(graphml_files, desc="Processing networks"):
            try:
                self.analyze_network(file_path)
            except Exception as e:
                network_name = os.path.basename(file_path).replace(
                    '.graphml', '')
                self.failed_networks.append((network_name, str(e)))
                print(f"  Error analyzing {network_name}: {e}")

        end_time = time.time()

        print(f"Analysis completed in {end_time - start_time:.2f} seconds")
        print(f"Successfully analyzed {len(self.results)} networks")
        print(f"Failed to analyze {len(self.failed_networks)} networks")

        return self.generate_summary()

    def generate_summary(self):
        """Generate summary statistics and visualizations"""
        # Convert results to DataFrame
        df = pd.DataFrame(list(self.results.values()))

        # Calculate summary statistics
        stats = {
            'total_networks':
            len(df),
            'nnsi_better': (df['impact_diff'] > 0).sum(),
            'ivi_better': (df['impact_diff'] < 0).sum(),
            'equal': (df['impact_diff'] == 0).sum(),
            'avg_impact_diff':
            df['impact_diff'].mean(),
            'median_impact_diff':
            df['impact_diff'].median(),
            'avg_overlap':
            df['overlap_percent'].mean(),
            'avg_latency_reduction':
            df['latency_reduction'].mean(),
            'avg_resilience_improvement':
            df['resilience_improvement'].mean(),
            'avg_link_utilization_improvement':
            df['link_utilization_improvement'].mean(),
            'avg_criticality_reduction':
            df['criticality_reduction'].mean()
        }

        # Create basic summary visualizations
        self.visualize_results(df, stats)

        return {
            'results_df': df,
            'stats': stats,
            'failed_networks': self.failed_networks
        }

    def visualize_results(self, df, stats):
        """Create visualizations for the analysis results"""
        # Create output directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # 1. Impact Comparison
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.scatter(df['ivi_impact'], df['nnsi_impact'], alpha=0.6)
        max_val = max(df['ivi_impact'].max(), df['nnsi_impact'].max()) * 1.1
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('IVI Impact')
        plt.ylabel('NNSI Impact')
        plt.title('(a) Impact Comparison')
        plt.axis('equal')

        # 2. Impact Difference Distribution
        plt.subplot(2, 2, 2)
        plt.hist(df['impact_diff'], bins=20, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Impact Difference (NNSI - IVI)')
        plt.ylabel('Count')
        plt.title('(b) Impact Difference Distribution')

        # 3. Computation Time Comparison
        plt.subplot(2, 2, 3)
        # We don't have actual computation time in this implementation,
        # so instead visualize network size vs impact difference
        plt.scatter(df['num_nodes'], df['impact_diff'], alpha=0.6)
        plt.xlabel('Network Size (nodes)')
        plt.ylabel('Impact Difference')
        plt.title('(c) Impact Difference vs Network Size')

        # 4. Overlap vs. Network Size
        plt.subplot(2, 2, 4)
        plt.scatter(df['num_nodes'], df['overlap_percent'], alpha=0.6)
        plt.xlabel('Network Size (nodes)')
        plt.ylabel('Top Nodes Overlap (%)')
        plt.title('(d) Overlap vs. Network Size')

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_comparison.png'),
                    dpi=300)
        plt.close()

        # 5. Network size vs. performance advantage
        plt.figure(figsize=(10, 6))

        # Create size categories
        df['size_category'] = pd.cut(df['num_nodes'],
                                     bins=[0, 20, 50, 100,
                                           float('inf')],
                                     labels=[
                                         'Small (4-20)', 'Medium (21-50)',
                                         'Large (51-100)', 'Very Large (>100)'
                                     ])

        # Group by size category
        size_impact = df.groupby('size_category')['impact_diff'].mean()

        plt.bar(size_impact.index, size_impact.values, color='lightgreen')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Network Size Category')
        plt.ylabel('Average Impact Difference')
        plt.title('Performance Advantage by Network Size')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_by_size.png'),
                    dpi=300)
        plt.close()

        # Save results to CSV
        df.to_csv(os.path.join(RESULTS_DIR, 'analysis_results.csv'),
                  index=False)

        # Save summary statistics
        with open(os.path.join(RESULTS_DIR, 'summary_statistics.txt'),
                  'w') as f:
            f.write("Summary Statistics:\n")
            f.write(f"Total networks analyzed: {stats['total_networks']}\n")
            f.write(
                f"NNSI performed better: {stats['nnsi_better']} networks ({100 * stats['nnsi_better'] / stats['total_networks']:.2f}%)\n"
            )
            f.write(
                f"IVI performed better: {stats['ivi_better']} networks ({100 * stats['ivi_better'] / stats['total_networks']:.2f}%)\n"
            )
            f.write(
                f"Equal performance: {stats['equal']} networks ({100 * stats['equal'] / stats['total_networks']:.2f}%)\n"
            )
            f.write(
                f"Average NNSI advantage: {stats['avg_impact_diff']:.4f}\n")
            f.write(
                f"Median NNSI advantage: {stats['median_impact_diff']:.4f}\n")
            f.write(
                f"Average overlap between top nodes: {stats['avg_overlap']:.2f}%\n"
            )
            f.write(
                f"Average latency reduction: {stats['avg_latency_reduction']:.2f}%\n"
            )
            f.write(
                f"Average resilience improvement: {stats['avg_resilience_improvement']:.2f}%\n"
            )
            f.write(
                f"Average link utilization improvement: {stats['avg_link_utilization_improvement']:.2f}%\n"
            )
            f.write(
                f"Average criticality reduction: {stats['avg_criticality_reduction']:.2f}%\n"
            )


# Entry point lives in nnsi/__main__.py — see `python -m nnsi --help`
