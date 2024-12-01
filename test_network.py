"""Module algorithm"""

# PR(A) = (1-d) + d * (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

edge_list = [('acharacterizationoffibrantsegalcategories', 'threemodelsforthehomotopytheoryofhomotopytheories'), ('higherandderivedstacksaglobaloverview', 'acharacterizationoffibrantsegalcategories'), ('higherandderivedstacksaglobaloverview', 'threemodelsforthehomotopytheoryofhomotopytheories'), ('threemodelsforthehomotopytheoryofhomotopytheories', 'acharacterizationoffibrantsegalcategories'), ('threemodelsforthehomotopytheoryofhomotopytheories', 'threemodelsforthehomotopytheoryofhomotopytheories')]

graph = nx.DiGraph()
graph.add_edges_from(edge_list)

def bfs(graph: dict, start: str) -> list:
    """
    BFS algorithm.

    :param graph: dict, A directed graph.
    :param start: str, A start node.
    :return: list, An ordered sequence of visited elements.
    """
    visited = []
    queue = deque([start])

    while queue:
        node = queue.popleft()

        if node not in visited:
            visited.append(node)
            queue.extend(neighbor for neighbor in graph.neighbors(node) if neighbor not in visited)

    return visited

ITERATIONS = len(graph.nodes)
START_RANK = 1 / ITERATIONS
EPSILON = 0.01
DAMPING_FACTOR = 0.85

page_rank = {key: START_RANK for key in graph.nodes}
page_prev = {key: 0 for key in graph.nodes}

while True:
    for node in page_rank:
        current_rank = 0
        reachable_nodes = bfs(graph, node)

        for el in reachable_nodes:
            if graph.has_edge(el, node):
                out_links = graph.out_degree(el)

                if out_links > 0:
                    current_rank += page_rank[el] / out_links

        page_prev[node] = (1 - DAMPING_FACTOR) + DAMPING_FACTOR * current_rank

    diff = sum(abs(page_prev[node] - page_rank[node]) for node in graph.nodes)

    if diff < EPSILON:
        break

    page_rank = dict(page_prev.items())

page_rank = dict(sorted(page_rank.items(), key = lambda x: -x[1]))
print(page_rank)
nx.draw_circular(graph, with_labels = True)
plt.show()