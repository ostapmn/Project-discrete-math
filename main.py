"""Module algorithm"""

# PR(A) = (1-d) + d * (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import unicodedata
import re
import os
from collections import deque
from functools import lru_cache
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text # lib to work with pdf files

PATTERN = re.compile(r'\[(.+?)\]\s+([\w.,\s&]+?),\s+(.*?),\s+(.*?)(?:\.|\n)') # pattern to identify paper name

# edge_list = [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D'), ('C', 'B'), ('C', 'A'), ('D', 'C')]

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

#===Pages directions===#

def pages_directions(pages: dict) -> list[tuple[str]]:
    """
    Converts inout dict into a list of list with a links directions.
    """
    pages_list = []
    for key, values in pages.items():
        if len(values) == 1:
            values = str(values).lstrip("['").rstrip("']")
            pages_list.append((key, values))
        elif len(values) > 1:
            for one_val in values:
                pages_list.append((key, one_val))
    pages_list = sorted(pages_list)
    return pages_list

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

# print(pages_directions(
#     read_pdf('test_papers/higherandderivedstacksaglobaloverview.pdf')))

graph_rank = nx.DiGraph()
# graph_rank.add_edges_from(pages_directions(
#     read_pdf('test_papers/higherandderivedstacksaglobaloverview.pdf')))

graph_rank.add_edges_from([('acharacterizationoffibrantsegalcategories', 'threemodelsforthehomotopytheoryofhomotopytheories'), ('higherandderivedstacksaglobaloverview', 'acharacterizationoffibrantsegalcategories'), ('higherandderivedstacksaglobaloverview', 'threemodelsforthehomotopytheoryofhomotopytheories'), ('threemodelsforthehomotopytheoryofhomotopytheories', 'acharacterizationoffibrantsegalcategories')])

ITERATIONS = len(graph_rank.nodes)
START_RANK = 1 / ITERATIONS
EPSILON = 0.01
DAMPING_FACTOR = 0.85

page_rank = {key: START_RANK for key in graph_rank.nodes}
page_prev = {key: 0 for key in graph_rank.nodes}

while True:
    for node in page_rank:
        current_rank = 0
        reachable_nodes = bfs(graph_rank, node)

        for el in reachable_nodes:
            if graph_rank.has_edge(el, node):
                out_links = graph_rank.out_degree(el)

                if out_links > 0:
                    current_rank += page_rank[el] / out_links

        page_prev[node] = (1 - DAMPING_FACTOR) + DAMPING_FACTOR * current_rank

    diff = [abs(page_prev[node] - page_rank[node])<EPSILON for node in graph_rank.nodes]

    if all(diff):
        break

    page_rank = dict(page_prev.items())


page_rank = dict(sorted(page_rank.items(), key = lambda x: -x[1]))

# Normalize PageRank values for color mapping
rank_values = np.array(list(page_rank.values()))
norm_rank_values = (rank_values - rank_values.min()) / (rank_values.max() - rank_values.min())

# Create a custom colormap from light blue to dark blue
light_to_dark_blue = LinearSegmentedColormap.from_list("LightToDarkBlue", ["#add8e6", "#00008b"])

# Create the figure and axis explicitly
fig, ax = plt.subplots(figsize=(8, 6))

# Draw the graph with circular layout
pos = nx.circular_layout(graph_rank)
nodes = nx.draw_networkx_nodes(
    graph_rank, pos, node_color=norm_rank_values, cmap=light_to_dark_blue, node_size=800, ax=ax
)
edges = nx.draw_networkx_edges(graph_rank, pos, edge_color='gray', ax=ax)
nx.draw_networkx_labels(graph_rank, pos, font_size=10, font_color='white', ax=ax)

# Add a colorbar to indicate PageRank values
sm = plt.cm.ScalarMappable(cmap=light_to_dark_blue, norm=plt.Normalize(vmin=rank_values.min(), vmax=rank_values.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='PageRank Score')

# Display the graph
print(page_rank)
nx.draw_circular(graph_rank, with_labels = True)

plt.show()
