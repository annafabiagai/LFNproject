import markov_clustering as mc
import networkx as nx
import random
import math
import numpy as np

from cdlib.benchmark import LFR
from cdlib import evaluation
from cdlib.classes import NodeClustering


def generate_original_graph(N, mu):
    while True:
        try:
            G, ground_truth_comms = LFR(
                n=N,
                mu=mu,
                tau1=3,
                tau2=2,
                average_degree=max(5, int(N/50)),
                min_community=max(5, int(N/20)),
                seed=random.randint(1, 10000)
            )

            G.remove_edges_from(nx.selfloop_edges(G))
            print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            return G, ground_truth_comms

        except nx.ExceededMaxIterations:
            print("Generation failed (ExceededMaxIterations). Retrying...")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e

        
def MCL_clustering(network, ground_truth):
    A = nx.adjacency_matrix(network)
    matrix = A.toarray()  

    best_inflation = 2.1
    best_modularity = -1

    inflations = [1.8, 2.0, 2.2]

    for inflation in inflations:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)

        communities = [set(c) for c in clusters]

        # Skip degenerate partitions
        if len(communities) <= 1:
            continue

        Q = nx.algorithms.community.modularity(network, communities)

        if Q > best_modularity:
            best_modularity = Q
            best_inflation = inflation


    # Final clustering with best inflation
    print(f"Best inflation parameter: {best_inflation} with modularity: {best_modularity}")
    final_result = mc.run_mcl(matrix, inflation=best_inflation)
    final_clusters = mc.get_clusters(final_result)

    return Evaluate_MCL(network, final_clusters, ground_truth)


def Evaluate_MCL(network, clusters, ground_truth):
    mcl_communities = NodeClustering(
        clusters,
        graph=network,
        method_name="MCL"
    )

    nmi = evaluation.normalized_mutual_information(
        ground_truth, mcl_communities
    ).score

    fscore = evaluation.f1(
        ground_truth, mcl_communities
    ).score

    ari = evaluation.adjusted_rand_index(
        ground_truth, mcl_communities
    ).score
    print(f"NMI: {nmi}, F-score: {fscore}, ARI: {ari}")
    return nmi, fscore, ari
    
if __name__ == "__main__":
    runs = 20

    graph_configs = [
        ("Graph 1", 500, 0.1),
        ("Graph 2", 1000, 0.1),
        ("Graph 3", 1000, 0.3),
    ]

    final_results = []

    for graph_name, N, mu in graph_configs:
        nmi_vals = []
        fscore_vals = []
        ari_vals = []

        print(f"Running test: N={N}, mu={mu}")
        for _ in range(runs):
            network, ground_truth = generate_original_graph(N, mu)
            nmi, fscore, ari = MCL_clustering(network, ground_truth)

            nmi_vals.append(nmi)
            fscore_vals.append(fscore)
            ari_vals.append(ari)

        final_results.append({
            "Graph": graph_name,
            "Avg-NMI": np.mean(nmi_vals),
            "Avg-Fscore": np.mean(fscore_vals),
            "Avg-ARI": np.mean(ari_vals)
        })

    # --- Print results in requested format ---
    print("\nGraph\tAvg-NMI\t\tAvg-Fscore\tAvg-ARI")
    for r in final_results:
        print(
            f"{r['Graph']}\t"
            f"{r['Avg-NMI']:.5f}\t"
            f"{r['Avg-Fscore']:.5f}\t"
            f"{r['Avg-ARI']:.5f}"
        )