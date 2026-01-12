#LFNproject - MCL vs Leiden Community Detection
This repository provides a controlled experimental framework to study how graph embeddings affect community detection, and how MCL compares to Leiden under varying structural conditions.

Graphs are generated using LFR-benchmark and the performance is measured against ground-truth communities using NMI, F-score, and ARI.

Steps presented in final_notebook.ipynb:

1. Graph reconstruction from embeddings using multiple similarity strategies
2. Implementation of MCL with automatic inflation tuning
3. Run of Leiden algorithm
4. Systematic comparison between:
  4.1 Raw vs embedded graphs
  4.2 MCL vs Leiden
5. Repeated experiments for statistical reliability

