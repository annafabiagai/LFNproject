#LFNproject - MCL vs Leiden Community Detection
This repository provides a controlled experimental framework to study how graph embeddings affect community detection, and how MCL compares to Leiden under varying structural conditions.

The Performance is measured against ground-truth communities using NMI, F-score, and ARI.

• The MCL code is modified version of: https://github.com/GuyAllard/markov_clustering.git

Pipeline:

1. Implementation of MCL with automatic inflation tuning
2. Graph reconstruction from embeddings using multiple similarity strategies
3. Robust Leiden baseline on LFR benchmark graphs
4. Systematic comparison between:
  4.1 Raw vs embedded graphs
  4.2 MCL vs Leiden
5. Repeated experiments for statistical reliability

