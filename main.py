"""Module algorithm"""

# PR(A) = (1-d) + d * (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import argparse
import unicodedata # used in the normalize_text function to display ligatures correctly
import re # pattern for reading papers
import os # checking the presence of files in the folder
from collections import deque # creating a queue for bfs
import networkx as nx # to work with graphs
from matplotlib.colors import LinearSegmentedColormap #for graph visualization by color gradation
import matplotlib.pyplot as plt # visualization
from pdfminer.high_level import extract_text # lib to work with pdf files

# Pattern to identify paper name
PATTERN = re.compile(r'\[(.+?)\]\s+([\w.,\s&]+?),\s+(.*?),\s+(.*?)(?:\.|\n)')

parser = argparse.ArgumentParser(description="Path to the paper of database")
parser.add_argument('filepath', type=str, help = "Path to the file")
args = parser.parse_args()

def name_changer(name:str) -> str:
    """
    Function to normalize file names
    so they align by file naming rules
    Args:
        name (str): old name

    Returns:
        str: new name
    >>> name_changer('higher and derived stacks: a "global" overview')
    'higher and derived stacks a global overview'
    """
    new_name = name.lower().replace(':', '').replace('"', '')
    return new_name

def normalize_text(text):
    """Function to convert 'unique' elements to normal one
    ﬁ → a single ligature character: U+FB01 ('unique')
    fi → two separate characters: f + i
    >>> normalize_text('ﬁ')
    'fi'
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
    >>> list(read_pdf('test_papers/higher and derived stacks a global overview.pdf').keys())
    ['higher and derived stacks a global overview', 'three models for the homotopy theory \
of homotopy theories', 'a characterization of fibrant segal categories', 'a model category \
structure on the category of simplicial categories', 'simplicial monoids and segal categories']
    >>> read_pdf('test_papers/higher and derived stacks a global \
overview.pdf')['higher and derived stacks a global overview']
    ['three models for the homotopy theory of homotopy theories', 'a characterization \
of fibrant segal categories']
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
    >>> import networkx as nx
    >>> graph = nx.DiGraph()
    >>> graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    >>> bfs(graph, 1)
    [1, 2, 3, 4]
    >>> bfs(graph, 2)
    [2, 4]
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

def page_rank_calc(graph: nx.DiGraph, page_rank:dict,
                   page_current:dict, damping_factor:float) -> dict:
    """
    Function calculates page rank

    Args:
        graph (nx.DiGraph): directed graph
        page_rank (dict): page rank
        page_current (dict): page rank for current iteration
        damping_factor (float): damping factor

    Returns:
        dict: calculated page rank
    """
    while True:
        for node in page_rank:
            current_rank = 0
            reachable_nodes = bfs(graph, node)

            for el in reachable_nodes:
                if graph.has_edge(el, node):
                    out_links = graph.out_degree(el)

                    if out_links > 0:
                        current_rank += page_rank[el] / out_links

            page_current[node] = round((1 - damping_factor) + damping_factor * current_rank, 5)

        diff = [abs(page_current[node] - page_rank[node])<0.01 for node in graph.nodes]

        if all(diff):
            break

        page_rank = dict(page_current.items())
    return page_rank


def main(path: str) -> dict:
    """
    Function to compile project
    Args:
        path (str): path to the file
    """
    # path = input()
    graph_rank = nx.DiGraph()
    graph_rank.add_edges_from(pages_directions(
        read_pdf(path))[0])

    start_rank = 1 / len(graph_rank.nodes)
    damping_factor = 0.85

    page_rank = {key: start_rank for key in graph_rank.nodes}
    page_current = {key: 0 for key in graph_rank.nodes}

    page_rank= page_rank_calc(graph_rank, page_rank, page_current, damping_factor)

    norm_rank_values = list(page_rank.values())

    page_rank = dict(sorted(page_rank.items(), key = lambda x: -x[1]))

    # Create a custom colormap from light blue to dark blue
    light_to_dark_blue = LinearSegmentedColormap.from_list("LightToDarkBlue",
                                                           ["#add8e6", "#00008b"])

    # Create the figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 10))

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
    return page_rank




if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
    main(args.filepath)
