import networkx as nx
import random
import math
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from cdlib.benchmark import LFR
from node2vec import Node2Vec


def generate_original_graph(N, mu):
    while True:
        try:  # needed because sometimes LFR is not able to assign communities
            # generate LFR graph
            # TODO: VALUTARE UN ATTIMO COME SETTARE average_degree E min_community
            G, ground_truth_comms = LFR(n=N, mu=mu,
                                        tau1=3, tau2=2, min_community=N / 20,
                                        average_degree=random.randint(math.floor(N / 100),
                                                                      math.ceil(N / 50)),
                                        seed=random.randint(1, 10000)
                                        )
            # if successful remove self loops and return the graph + communities
            G.remove_edges_from(nx.selfloop_edges(G))
            return G, ground_truth_comms

        # if errors occur retry / give warning
        except nx.ExceededMaxIterations:
            print("Generation failed (ExceededMaxIterations). Retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise e

def get_reconstructed_graph(G, reconstruction, metric):

    # if input is not correct stop the execution
    if (reconstruction != "knn" and reconstruction != "threshold") or (metric != "cosine" and metric != "dot"):
        print("Incorrect reconstruction method or metric")
        return None

    # create node2vec model and get the embeddings
    # P E Q MOTIVATI DAL PAPER DI NODE2VEC (STESSI NUMERI DEL CASE STUDY), N=64 PERCHÈ USIAMO GRAFI PICCOLI (evita curse of dim)
    node2vec_model = Node2Vec(G, dimensions=64,
                              q=0.5, p=1,
                              seed=123)
    embeddings = node2vec_model.fit() # output works like a dictionary, key = node_id

    # convert embeddings into a numpy array
    model_wv = embeddings.wv  # this gets all keyed vectors in the attribute .wv
    nodes = sorted(list(G.nodes()))  # get nodes list from original graph
    X = np.array([model_wv[str(n)] for n in nodes])  # do the lookup in model_wv node by node,
                                                     # convert 'n' to str() because node2vec stores keys as strings
    # initialize empty graph and edge list
    reconstructed_graph = nx.Graph()
    reconstructed_graph.add_nodes_from(nodes)
    edge_list = []

    # based on the selected reconstruction method and metric build the graph
    if reconstruction == "knn":
        # get k for k-NN as the average node degree
        k = int(np.round((2 * G.number_of_edges()) / G.number_of_nodes()))

        if metric == "cosine":
            # get knn model
            knn = NearestNeighbors(n_neighbors=k + 1, # use k+1 because the first neighbor is always the node itself (distance=0)
                                   metric=metric)
            knn.fit(X)
            distances, indices = knn.kneighbors(X)

            # get the list of edges and weights to add to the reconstructed graph
            for i, (neighbor_indices, neighbor_dists) in enumerate(zip(indices, distances)):
                u = nodes[i]
                for neighbor_idx, dist in zip(neighbor_indices[1:], neighbor_dists[1:]):
                    v = nodes[neighbor_idx]
                    weight = 1 - dist # convert distance to similarity
                    edge_list.append((u, v, {'weight': weight}))

        elif metric == "dot":
            # compute similarity matrix and fill diagonal with -inf to avoid self loops
            sim_matrix = np.dot(X, X.T)
            np.fill_diagonal(sim_matrix, -np.inf)

            # for every node find the k highest values
            # argpartition puts the top k indices at the end of the array
            # take the last k indices
            top_k_indices = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]

            # get the list of edges and weights to add to the reconstructed graph
            for i, neighbor_indices in enumerate(top_k_indices):
                u = nodes[i]
                for neighbor_idx in neighbor_indices:
                    v = nodes[neighbor_idx]
                    weight = sim_matrix[i, neighbor_idx] # get the dot product as weight
                    edge_list.append((u, v, {'weight': float(weight)}))

    else: # if reconstruction == "thresholding"
        t = 0.5  # set threshold
        if metric == "cosine":
            # compute cosine similarity matrix and remove self loops
            sim_matrix = cosine_similarity(X)
            np.fill_diagonal(sim_matrix, 0)

            # gets indices of node couples that overcome the threshold that are
            # located in the upper triangle of the matrix (avoids duplicates)
            rows, cols = np.where(np.triu(sim_matrix, k=1) >= t)

            # get the list of edges and weights to add to the reconstructed graph
            for u_idx, v_idx in zip(rows, cols):
                u, v = nodes[u_idx], nodes[v_idx]
                weight = sim_matrix[u_idx, v_idx]
                edge_list.append((u, v, {'weight': float(weight)}))

        elif metric == "dot":
            # compute similarity matrix and fill diagonal with -inf to avoid self loops
            sim_matrix = np.dot(X, X.T)
            np.fill_diagonal(sim_matrix, -np.inf)

            # do min-max normalization to bring range to [0, 1]
            valid_values = sim_matrix[sim_matrix != -np.inf] # ignore -inf values
            min_val = valid_values.min()
            max_val = valid_values.max()
            sim_matrix_norm = (sim_matrix - min_val) / (max_val - min_val)

            # apply threshold (same line of code of cosine)
            rows, cols = np.where(np.triu(sim_matrix_norm, k=1) >= t)

            # get the list of edges and weights to add to the reconstructed graph
            for u_idx, v_idx in zip(rows, cols):
                u, v = nodes[u_idx], nodes[v_idx]
                weight = sim_matrix_norm[u_idx, v_idx]
                edge_list.append((u, v, {'weight': float(weight)}))

    # add edges to the graph
    reconstructed_graph.add_edges_from(edge_list)

    return reconstructed_graph