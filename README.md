# Similarity Degree Analysis of Anthropometric and Body Composition Variables

## Overview

This repository contains scripts for analyzing the similarity degree of anthropometric and body composition variables of Brazilian and Mozambican schoolchildren. The approach involves using the Smoothed Jaccard Index Surface. This project is part of the paper titled "Similarity degree of the anthropometric and body composition variables of Brazilian and Mozambican schoolchildren: a new approach by using Smoothed Jaccard Index Surface" by André Luiz de Góes Pacheco.

## Prerequisites

To use the scripts in this repository, you need to have Python installed along with the following packages:

- `pandas`: Data analysis and manipulation library.
- `numpy`: Fundamental package for scientific computing with Python.
- `networkx`: Library for the creation, manipulation, and study of complex networks.
- `matplotlib`: Comprehensive library for creating static, animated, and interactive visualizations in Python.
- `seaborn`: Statistical data visualization library based on matplotlib.

You can install these packages using pip:

```bash
pip install pandas numpy networkx matplotlib seaborn


Scripts
1. Data Loading and Preprocessing
File: data_loading.py

Description:
Loads data from an Excel file, processes it by filtering based on provinces, and calculates correlation matrices for each province.

Functions:

file_path: Specifies the path to the Excel file.
dataset: A DataFrame loaded from the Excel file.
data_dict: A dictionary containing correlation matrices for each province.
2. Graph Creation and Density Calculation
File: graph_creation.py

Description:
Creates a NetworkX graph from a correlation matrix with a specified density and removes edges to achieve the desired density.

Functions:

create_graph(matrix, density, verbose): Creates a graph from a correlation matrix with a specified density.
3. Jaccard Index Calculation and Visualization
File: jaccard_index.py

Description:
Calculates the Jaccard index for pairs of nodes in a graph and plots a 3D visualization.

Functions:

plot_jaccard_index(G, node_names, peak_threshold, valley_threshold): Plots the Jaccard index for a given graph.
4. Centrality Calculations
File: centrality_calculations.py

Description:
Calculates various centralities for a given graph including betweenness, degree, and eigenvector centralities.

Functions:

calculate_centralities(graph): Calculates centralities for a graph.
5. Clustermap Visualization
File: clustermap.py

Description:
Creates a cluster map visualization using Seaborn.

Functions:

plot_cluster_map(matrix): Plots a cluster map from a data matrix.
6. DataFrame Processing and Centrality Correlation
File: df_processing.py

Description:
Processes a DataFrame and calculates correlations between centrality measures and other columns.

Functions:

calculate_centralities2(df, G): Calculates centralities and correlations for the largest connected component of a graph.


Citation
If you use this code, please cite the following article once it is published:

Pacheco, A. L. G. (2024). Similarity degree of the anthropometric and body composition variables of Brazilian and Mozambican schoolchildren: a new approach by using Smoothed Jaccard Index Surface.

Contact
For any questions or issues, please contact André Luiz de Góes Pacheco at algp@cin.ufpe.br .
