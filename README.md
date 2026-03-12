# Evaluating Leiden and MCL on Graphs Reconstructed from Node Embeddings

This project investigates whether reconstructing graphs from node embeddings can improve the performance of community detection algorithms when the original graph structure is weak or noisy.

The study compares two well-known community detection algorithms:

- **Leiden Algorithm**
- **Markov Cluster Algorithm (MCL)**

The goal is to understand whether embedding-based graph reconstruction can help these algorithms recover communities more accurately compared to operating directly on the original graph.

---

# Motivation

Traditional community detection methods rely only on the topology of the graph. However, when the graph structure is sparse or noisy, these algorithms may fail to correctly identify communities.

Node embeddings capture **higher-order connectivity patterns** in the network and by reconstructing a graph from these embeddings, we aim to create a representation that better reflects latent relationships between nodes.

This project evaluates whether clustering on these reconstructed graphs improves community detection performance.

---

# Dataset

The experiments use **synthetic graphs generated with the LFR benchmark**, a standard method for evaluating community detection algorithms which allows precise control of graph properties and provides **ground-truth communities** for evaluation.

Key parameters include:

- **N** – number of nodes in the graph
- **μ (mixing parameter)** – fraction of edges connecting nodes to other communities

Interpretation of μ:

- **Low μ → well separated communities**
- **High μ → overlapping or weak communities**

The experiments focus on configurations where community detection becomes progressively harder.

---

# Experimental Setup

Experiments compare community detection in two scenarios:

1. **Direct clustering on the original graph**
2. **Clustering on graphs reconstructed from node embeddings**

The workflow is:

1. Generate synthetic networks using the **LFR benchmark**
2. Compute **node embeddings**
3. Reconstruct graphs based on embedding similarity
4. Apply community detection algorithms
5. Compare detected communities with ground truth

---

# Algorithms

## Leiden Algorithm
A modularity optimization method that improves upon the Louvain algorithm which guarantees better connected communities and typically produces high-quality partitions.

## Markov Cluster Algorithm (MCL)
A flow-based clustering algorithm that simulates random walks in the graph which works well when community structure is strong but may degrade when communities overlap.

---

# Evaluation

Performance is evaluated by comparing detected communities with the **ground-truth communities** provided by the LFR benchmark.

Different graph configurations are tested to analyze how the algorithms behave under varying levels of community overlap and structural noise.

---

# Technologies Used

- **Python**
- **NetworkX** – graph manipulation
- **NumPy / SciPy** – numerical computation
- **Node Embedding methods** (Node2Vec)
- **Leiden algorithm implementation**
- **Markov Cluster Algorithm (MCL)**
- **LFR benchmark generator**



