"""Module algorithm"""

# PR(A) = (1-d) + d (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

import unicodedata
import re
import os
import networkx as nx
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text # lib to work with pdf files

PATTERN = re.compile(r'\[(.+?)\]\s+([\w.,\s&]+?),\s+(.*?),\s+(.*?)(?:\.|\n)') # pattern to identify paper name
DAMPING_FACT = 0.85

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
    graph[paper_name[:-4]] = [el[2].lower() for el in matches
    if f'{el[2].lower().replace(':', '')}.pdf' in os.listdir(folder)]
    for paper in graph[paper_name[:-4]]:
        if paper not in graph.keys():
            new_path = f'{folder}/{paper}.pdf'
            read_pdf(new_path, graph)
    return graph

# print(read_pdf('test_papers/higher and derived stacks a global overview.pdf').keys())

graph_rank = nx.DiGraph()
graph_rank.add_edges_from(edge_list)

ITERATIONS = len(graph_rank.nodes)
START_RANK = 1 / ITERATIONS
EPSILON = 0.01

page_rank = {key: START_RANK for key in graph_rank.nodes}
page_prev = {key: 0 for key in graph_rank.nodes}


while True:
    for node in page_rank:
        current_rank = 0
        for el in graph_rank.edges:
            if el[1] == node:
                out_links = graph_rank.out_degree(el[0])
                current_rank += page_rank[el[0]]/out_links
        current_rank = (1 - DAMPING_FACT) + (DAMPING_FACT *current_rank)
        page_prev[node] = current_rank

    diff = [abs(page_prev[node] - page_rank[node]) < EPSILON for node in graph_rank.nodes]
    if all(diff):
        break
    page_rank = dict(page_prev.items())

page_rank = dict(sorted(page_rank.items(), key = lambda x: -x[1]))
print(page_rank)

nx.draw_circular(graph_rank, with_labels = True)
plt.show()
