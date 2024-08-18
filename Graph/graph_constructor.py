import json
import networkx as nx


def add_nodes(nodes: list) -> nx.classes.digraph.DiGraph:

    G = nx.DiGraph()
    for node in nodes:
        node.pop('position', None) # removes position item from node
        node['description'] = node['description'].replace("\n", " ") # remove newline characters from description
        node_id = node.pop('id')
        G.add_node(node_id, **node)
    return G

def add_edges(transitions: list, G: nx.classes.digraph.DiGraph):

    for transition in transitions:
        if isinstance(transition['description'],list):
            # extracts just the name
            transition['description'] = transition['description'][0]
        transition.pop('frames', None)  # removes frames item from transition
        start = transition['from']['node']
        end = transition['to']['node']
        G.add_edge(start,end,**transition)
    return G
def load_json(fpath):
    with open(fpath, 'r') as file:
        return json.load(file)
def construct_graph(nodes_path = 'files/nodes.json',transitions_path = 'files/transitions.json') -> nx.classes.digraph.DiGraph:

    nodes = load_json(nodes_path)
    transitions = load_json(transitions_path)
    #tags = load_json('files/tags.json')

    G = add_nodes(nodes)
    G = add_edges(transitions,G)
    return G

G = construct_graph()
