"""Module algorithm"""

# PR(A) = (1-d) + d (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define the edge list
edge_list = [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D'), ('C', 'B'), ('C', 'A'), ('D', 'C')]

# Create a directed graph and add edges
graph = nx.DiGraph()
graph.add_edges_from(edge_list)

# Initialize parameters
ITERATIONS = len(graph.nodes)
START_RANK = 1 / ITERATIONS
EPSILON = 0.01

# Initialize PageRank scores
page_rank = {key: START_RANK for key in graph.nodes}
page_prev = {key: 0 for key in graph.nodes}

# Compute PageRank iteratively
while True:
    for node in page_rank:
        current_rank = 0
        for el in graph.edges:
            if el[1] == node:
                out_links = graph.out_degree(el[0])
                current_rank += page_rank[el[0]] / out_links
        page_prev[node] = current_rank

    diff = sum(abs(page_prev[node] - page_rank[node]) for node in graph.nodes)
    if diff < EPSILON:
        break
    page_rank = dict(page_prev.items())

# Sort PageRank for visualization purposes
page_rank = dict(sorted(page_rank.items(), key=lambda x: -x[1]))

# Normalize PageRank values for color mapping
rank_values = np.array(list(page_rank.values()))
norm_rank_values = (rank_values - rank_values.min()) / (rank_values.max() - rank_values.min())

# Create a custom colormap from light blue to dark blue
light_to_dark_blue = LinearSegmentedColormap.from_list("LightToDarkBlue", ["#add8e6", "#00008b"])

# Create the figure and axis explicitly
fig, ax = plt.subplots(figsize=(8, 6))

# Draw the graph with circular layout
pos = nx.circular_layout(graph)
nodes = nx.draw_networkx_nodes(
    graph, pos, node_color=norm_rank_values, cmap=light_to_dark_blue, node_size=800, ax=ax
)
edges = nx.draw_networkx_edges(graph, pos, edge_color='gray', ax=ax)
nx.draw_networkx_labels(graph, pos, font_size=10, font_color='white', ax=ax)

# Add a colorbar to indicate PageRank values
sm = plt.cm.ScalarMappable(cmap=light_to_dark_blue, norm=plt.Normalize(vmin=rank_values.min(), vmax=rank_values.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='PageRank Score')

# Display the graph
plt.show()
