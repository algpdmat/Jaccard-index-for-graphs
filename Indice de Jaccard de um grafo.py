#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


# In[37]:


# Caminhos para os arquivos

caminho_arquivo = 'C:/Users/algp/Pictures/img artigo L1/DATASETS.xlsx'


# Importar arquivo Excel
dataset = pd.read_excel(caminho_arquivo)

# Definição de métricas
metricas = ['PNASC', 'PESO', 'ALTURA', 'Circ. Cintura', '%GC', 'Massa Gorda', 'Massa Magra']


Inhambane_data = dataset[dataset['Província']  == 'Inhambane']
inhambane = Inhambane_data.T[1:]
inhambane.dropna()
inhambane = inhambane.apply(pd.to_numeric, errors='coerce')
matriz_inhambane = inhambane.T.corr()
m_inhambane = matriz_inhambane.values

matola_data = dataset[dataset['Província']  == 'Matola']
matola = matola_data.T[1:]
matola.dropna()
matola = matola.apply(pd.to_numeric, errors='coerce')
matriz_matola = matola.T.corr()

maputo_data = dataset[dataset['Província']  == 'Maputo']
maputo = maputo_data.T[1:]
maputo.dropna()
maputo = maputo.apply(pd.to_numeric, errors='coerce')
matriz_maputo = maputo.T.corr()

vitoria2019_data = dataset[dataset['Província']  == 'Vitoria2009']
vitoria2019 = vitoria2019_data.T[1:]
vitoria2019.dropna()
vitoria2019 = vitoria2019.apply(pd.to_numeric, errors='coerce')
matriz_vitoria2019 = vitoria2019.T.corr()


# In[38]:


def G_den(matrix, d, verbose=False):
    """Returns a networkx Graph from a adjacency matrix, with a given density d

    Parameters
    ----------
    matrix: matrix
        A matrix of values - connectivity matrix

    d: float
        Density value for matrix binaziring

    Returns
    -------
        NetworkX graph

    """

    #matrix i, density d. i is a matrix - ravel flatten the matrix
    np.fill_diagonal(matrix,0)
    temp = sorted(matrix.ravel(), reverse=True) # will flatten it and rank corr values
    size = len(matrix)
    cutoff = np.ceil(d * (size * (size-1))) # number of links with a given density
    tre = temp[int(cutoff)]
    G0 = nx.from_numpy_array(matrix)

    G0.remove_edges_from(list(nx.selfloop_edges(G0)))
    G1 = nx.from_numpy_array(matrix)
    for u,v,a in G0.edges(data=True):
        if (a.get('weight')) <= tre:
            G1.remove_edge(u, v)
    finaldensity = nx.density(G1)
    if verbose == True:
        print(finaldensity)

    return G1

def plot_jaccard_index(G, nodes_name, peak_threshold=0.8, valley_threshold=0.2):
    """
    Plots the Jaccard index for a given graph.

    Parameters:
    G (networkx.Graph): The graph for which Jaccard index is calculated.
    nodes_name (list): List of node names.
    peak_threshold (float): Threshold for highlighting peaks.
    valley_threshold (float): Threshold for highlighting valleys.
    """

    n = len(G.nodes())
    matrix = np.zeros((n, n))

    # Calculate Jaccard index for each pair of nodes
    for i, u in enumerate(G.nodes()):
        for j, v in enumerate(G.nodes()):
            if u == v:
                continue
            common_neighbors = len(list(nx.common_neighbors(G, u, v)))
            union_size = len(set(G[u]) | set(G[v]))
            jaccard_index = common_neighbors / union_size if union_size else 0
            matrix[i, j] = jaccard_index

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = np.meshgrid(range(n), range(n))

    # Use a color map to indicate the Jaccard index values
    surf = ax.plot_surface(x, y, matrix, cmap=cm.coolwarm)

    # Highlight peaks and label them
    for i in range(n):
        for j in range(n):
            if matrix[i, j] >= peak_threshold:
                ax.scatter(x[i, j], y[i, j], matrix[i, j], color='green', s=100)
                ax.text(x[i, j], y[i, j], matrix[i, j], f'({nodes_name[i]},{nodes_name[j]})', fontsize=12)
            elif matrix[i, j] <= valley_threshold:
                ax.scatter(x[i, j], y[i, j], matrix[i, j], color='red', s=100)

    # Add a color bar with padding
    fig.colorbar(surf, ax=ax, pad=0.2)

    ax.set_zlabel('Jaccard Index')

    # Adjust the view angle for better visibility of the z-axis
    ax.view_init(elev=20, azim=-45)

    # Set custom tick labels for x and y axes
    ax.set_xticks(range(n))
    ax.set_xticklabels(nodes_name, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(nodes_name, rotation=90)

    plt.title('Jaccard Index Visualization')

    plt.show()


def calculate_centralities(graph):
    # Calculate Betweenness Centrality
    betweenness_centrality = list(nx.betweenness_centrality(graph).values())

    # Calculate Degree Centrality
    degree_centrality = list(nx.degree_centrality(graph).values())

    # Calculate Eigenvector Centrality with max_iter set to 1000
    eigenvector_centrality = list(nx.eigenvector_centrality(graph, max_iter=1000).values())

    return betweenness_centrality, degree_centrality, eigenvector_centrality

def plot_cluster_map(matrix):
    """
    Plots a cluster map using Seaborn.

    Parameters:
    matrix (2D array-like): A matrix of data to create the cluster map.
    """
    sns.set_theme()  # Set the default theme for Seaborn
    clustermap = sns.clustermap(matrix)
    return clustermap


# In[17]:


# Separar o DataFrame em grupos com base na coluna "Província"
grupos = dataset.groupby('Província')

# Criar uma lista para armazenar os grupos de DataFrames
lista_grupos = []

# Iterar sobre os grupos e adicionar cada grupo à lista
for nome_provincia, grupo in grupos:
    lista_grupos.append(grupo)


Inhambane = lista_grupos[0]
Maputo = lista_grupos[1]
Matola = lista_grupos[2]
Vitoria2009 = lista_grupos[3]
Vitoria2019 = lista_grupos[4]

df_Maputo = Maputo.drop(columns=['Província'])
df_Matola = Matola.drop(columns=['Província'])
df_Vitoria2009 = Vitoria2009.drop(columns=['Província'])
df_Vitoria2019 = Vitoria2019.drop(columns=['Província'])
df_Inhambane = Inhambane.drop(columns=['Província'])


# In[7]:


import pandas as pd
import networkx as nx

def calculate_centralities2(df, G):
    # Get the largest connected component of the graph
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc).copy()

    # Calculating centralities for the largest connected component
    betw_centrality = nx.betweenness_centrality(subgraph)
    deg_centrality = nx.degree_centrality(subgraph)
    eig_centrality = nx.eigenvector_centrality(subgraph, max_iter=1000)

    # Adicionando centralidades ao DataFrame
    df['BC'] = pd.Series(betw_centrality)
    df['DC'] = pd.Series(deg_centrality)
    df['EC'] = pd.Series(eig_centrality)

    # Calculando as correlações
    bc_correlations = df.corr()['BC'].drop(['BC', 'DC', 'EC'])
    dc_correlations = df.corr()['DC'].drop(['BC', 'DC', 'EC'])
    ec_correlations = df.corr()['EC'].drop(['BC', 'DC', 'EC'])

    # Criando um DataFrame para armazenar as correlações
    correlations_df = pd.DataFrame({
        'BC': bc_correlations,
        'DC': dc_correlations,
        'EC': ec_correlations
    })

    return correlations_df

