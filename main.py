"""Module algorithm"""

# PR(A) = (1-d) + d * (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import unicodedata
import re
import os
from collections import deque
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text # lib to work with pdf files

# Pattern to identify paper name
PATTERN = re.compile(r'\[(.+?)\]\s+([\w.,\s&]+?),\s+(.*?),\s+(.*?)(?:\.|\n)')

def name_changer(name:str) -> dict:
    """
    Function to normalize file names
    so they align by file naming rules
    Args:
        name (str): old name

    Returns:
        dict: new name
    """
    new_name = name.lower().replace(':', '')
    return new_name

def normalize_text(text):
    """Function to convert 'unique' elements to normal one
    ﬁ → a single ligature character: U+FB01 ('unique')
    fi → two separate characters: f + i
    """
    normalized = unicodedata.normalize("NFKD", text)
    return normalized

def read_pdf(file_name: str, graph = None) -> dict:
    """
    Function reads pdf file and creates graph from it
    Args:
        file_name (str): path to local files

    Returns:
        dict[str: list]: files and their links 
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
    graph[paper_name[:-4]] = [name_changer(el[2]) for el in matches
    if f'{name_changer(el[2])}.pdf' in os.listdir(folder) and
    name_changer(el[2]) != paper_name[:-4]]
    for paper in graph[paper_name[:-4]]:
        if paper not in graph.keys():
            new_path = f'{folder}/{paper}.pdf'
            read_pdf(new_path, graph)
    return graph

#===Pages directions===#

def pages_directions(pages: dict) -> tuple[list[tuple], dict]:
    """
    Converts inout dict into a list of list with a links directions.
    """
    pages_list = [(key, val) for key in pages for val in pages[key]]
    f_name = set(i for el in pages_list for i in el)
    characters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    new_name = dict(zip(f_name, characters))
    return ([tuple(new_name[c] for c in el) for el in pages_list], new_name)


def bfs(graph: nx.DiGraph, start: str) -> list:
    """
    BFS algorithm.

    :param graph: dict, A directed graph.
    :param start: str, A start node.
    :return: list, An ordered sequence of visited elements.
    """
    visited = []
    queue = deque([start])

    while queue:
        node_graph = queue.popleft()

        if node_graph not in visited:
            visited.append(node_graph)
            queue.extend(neighbor for neighbor in graph.neighbors(node_graph)
                         if neighbor not in visited)

    return visited

def main(path:str):
    graph_rank = nx.DiGraph()
    graph_rank.add_edges_from(pages_directions(
        read_pdf(path))[0])

    start_rank = 1 / len(graph_rank.nodes)
    epsilon = 0.01
    damping_factor = 0.85

    page_rank = {key: start_rank for key in graph_rank.nodes}
    page_current = {key: 0 for key in graph_rank.nodes}

    while True:
        for node in page_rank:
            current_rank = 0
            reachable_nodes = bfs(graph_rank, node)

            for el in reachable_nodes:
                if graph_rank.has_edge(el, node):
                    out_links = graph_rank.out_degree(el)

                    if out_links > 0:
                        current_rank += page_rank[el] / out_links

            page_current[node] = round((1 - damping_factor) + damping_factor * current_rank, 5)

        diff = [abs(page_current[node] - page_rank[node])<epsilon for node in graph_rank.nodes]

        if all(diff):
            break

        page_rank = dict(page_current.items())


    norm_rank_values = list(page_rank.values())

    page_rank = dict(sorted(page_rank.items(), key = lambda x: -x[1]))

    # Create a custom colormap from light blue to dark blue
    light_to_dark_blue = LinearSegmentedColormap.from_list("LightToDarkBlue",
                                                           ["#add8e6", "#00008b"])

    # Create the figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the graph with circular layout
    pos = nx.circular_layout(graph_rank)
    nx.draw_networkx_nodes(
        graph_rank, pos, node_color=norm_rank_values, cmap=light_to_dark_blue,
        node_size=400, ax=ax)

    nx.draw_networkx_edges(graph_rank, pos, edge_color='gray', ax=ax, arrowsize=20)
    nx.draw_networkx_labels(graph_rank, pos, font_size=10, font_color='black', ax=ax)

    # Add a colorbar to indicate PageRank values
    sm = plt.cm.ScalarMappable(cmap=light_to_dark_blue,
                        norm=plt.Normalize(vmin=min(norm_rank_values), vmax=max(norm_rank_values)))
    fig.colorbar(sm, ax=ax, label='PageRank Score')

    # Display the graph
    info = pages_directions(read_pdf(path))[1]

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=10, label=category)
        for category in info.items()
    ]
    fig.legend(handles=legend_handles, title="Names", loc="upper left")

    plt.show()
    print(page_rank)


if __name__ == "__main__":  
    main('test_papers/higher and derived stacks a global overview.pdf')
