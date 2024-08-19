import json
import networkx as nx
from reward import add_rewards_to_graph

def add_nodes(node_list: list) -> nx.classes.digraph.DiGraph:

    G = nx.DiGraph()
    for node in node_list:
        node.pop('position', None) # removes unnecessary 3D information from node
        node['description'] = node['description'].replace("\n", " ") # remove newline characters from description
        G.add_node(node['id'], **node)
    return G

def add_edges(transitions: list, G: nx.classes.digraph.DiGraph):

    for transition in transitions:
        if isinstance(transition['description'],list):
            # extracts just the name
            transition['description'] = transition['description'][0]
        transition.pop('frames', None)  # removes unnecessary 3D data
        start = transition['from']['node']
        end = transition['to']['node']
        G.add_edge(start,end,**transition)
        # adds edge in opposite direction if the transition is bidirectional
        if 'bidirectional' in transition['properties']:
            G.add_edge(end, start, **transition)

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

    #add rewards signal to GrappleMap data
    G = add_rewards_to_graph(G)

    return G

G = construct_graph()
