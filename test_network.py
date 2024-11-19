"""Module for testing lib functions"""
import networkx as nx
import matplotlib.pyplot as plt

edge_list = [(1,3), (3,2), (1, 5)]

graph = nx.DiGraph()
graph.add_edges_from(edge_list)
graph.add_edge(1,2)


nx.draw_spring(graph)
plt.show()
