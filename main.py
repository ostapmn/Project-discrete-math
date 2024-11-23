"""Module algorithm"""

# PR(A) = (1-d) + d (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import networkx as nx
import matplotlib.pyplot as plt

edge_list = [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D'), ('C', 'B'), ('C', 'A'), ('D', 'C')]

graph = nx.DiGraph()
graph.add_edges_from(edge_list)

ITERATIONS = len(graph.nodes)
START_RANK = 1 / ITERATIONS
EPSILON = 0.01

page_rank = {key: START_RANK for key in graph.nodes}
page_prev = {key: 0 for key in graph.nodes}

while True:
    for node in page_rank:
        current_rank = 0
        for el in graph.edges:
            if el[1] == node:
                out_links = graph.out_degree(el[0])
                current_rank += page_rank[el[0]]/out_links
        page_prev[node] = current_rank

    diff = sum(abs(page_prev[node] - page_rank[node]) for node in graph.nodes)
    if diff < EPSILON:
        break

    page_rank = dict(page_prev.items())

page_rank = dict(sorted(page_rank.items(), key = lambda x: -x[1]))

nx.draw_circular(graph, with_labels = True)
plt.show()
