"""Module algorithm"""

# PR(A) = (1-d) + d * (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import unicodedata
import re
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from pdfminer.high_level import extract_text # lib to work with pdf files

PATTERN = re.compile(r'\[(.+?)\]\s+([\w.,\s&]+?),\s+(.*?),\s+(.*?)(?:\.|\n)') # pattern to identify paper name

edge_list = [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D'), ('C', 'B'), ('C', 'A'), ('D', 'C')]

def normalize_text(text):
    """Function to convert 'unique' elements to normal one"""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized

def read_pdf(file_name: str, graph = None) -> list:
    """
    Function reads pdf file and creates graph from it
    Args:
        file_name (str): path to local files

    Returns:
        list[tuple[str]]: _description_
    """
    if graph is None:
        graph = {}
    folder, paper_name = file_name.split('/') # we assume that all our files in one directory
    if file_name[:-4] in graph.keys():
        return graph
    
    text = extract_text(file_name) #extracting text from pdf
    start_point = text.find('References\n') # starting point from wear to parse the text
    matches = re.findall(PATTERN, normalize_text(text[start_point:])) # finding matches
    if not matches: # if ther is no matches
        return graph
# changing el name deliting special symbols and checking if we have such element in our folder
# so not to add unnececary ones
    graph[paper_name[:-4]] = [el[2].lower().replace(' ', '').replace(':', '') for el in matches
    if f'{el[2].lower().replace(' ', '').replace(':', '')}.pdf' in os.listdir(folder)]
    for paper in graph[paper_name[:-4]]:
        if paper not in graph.keys():
            new_path = f'{folder}/{paper}.pdf'
            read_pdf(new_path, graph)
    return graph

# print(read_pdf('test_papers/higherandderivedstacksaglobaloverview.pdf'))

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

graph_rank = nx.DiGraph()
graph_rank.add_edges_from(edge_list)


page_rank = {key: START_RANK for key in graph_rank.nodes}
page_prev = {key: 0 for key in graph_rank.nodes}


# diff = [abs(page_prev[node] - page_rank[node]) < EPSILON for node in graph_rank.nodes]

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


page_rank = dict(sorted(page_rank.items(), key = lambda x: -x[1]))
print(page_rank)

nx.draw_circular(graph_rank, with_labels = True)
plt.show()
