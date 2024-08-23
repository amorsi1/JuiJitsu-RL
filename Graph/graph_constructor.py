import json
import networkx as nx
from reward import add_rewards_to_graph
from typing import List, Tuple, Dict
import copy

def add_nodes(node_list: list) -> nx.classes.digraph.DiGraph:

    G = nx.DiGraph()
    for node in node_list:
        node.pop('position', None) # removes unnecessary 3D information from node
        node['description'] = node['description'].replace("\n", " ") # remove newline characters from description
        G.add_node(node['id'], **node)
    return G

def add_edges(transitions: list, G: nx.DiGraph):
    def clean_up_dict(trans_dict) -> Dict:
        """
        processes some of the data so that the relevant information is explicitly stated in each edge
        instead of calculating it during the game
        """
        if isinstance(trans_dict['description'],list):
            # extracts just the name
            trans_dict['description'] = trans_dict['description'][0]
        trans_dict.pop('frames', None)  # removes unnecessary 3D data
        # extracts just the swap players bool from the 'reo' key and deletes all the irrelevant information
        for key in ['from', 'to']:
            start_or_end_node = trans_dict[key]
            start_or_end_node['players_swapped'] = start_or_end_node['reo']['swap_players']
            start_or_end_node.pop('reo', None)
        # adds whether the transition swaps player's relative positions (top/bottom)
        trans_dict['swaps_players'] = False
        if trans_dict['from']['players_swapped'] != trans_dict['to']['players_swapped']:
            trans_dict['swaps_players'] = True
        # adds default value of whether the transition is bidirectional. Will be reassigned in the following step
        trans_dict['reversible'] = False
        # add explicit tag of whether move is for top, bottom, or any player
        properties = trans_dict['properties']
        trans_dict['bottom'] = False
        trans_dict['top'] = False
        if any(['top' in prop for prop in properties]):
            trans_dict['top'] = True
        elif any(['bottom' in prop for prop in properties]):
            trans_dict['bottom'] = True

        return trans_dict

    for transition in transitions:
        transition = clean_up_dict(transition)
        edge_id = transition['id']
        start = transition['from']['node']
        end = transition['to']['node']
        G.add_edge(start, end, **transition)
        # adds edge in opposite direction if the transition is bidirectional
        if any(['bidirectional' in prop for prop in transition['properties']]):
            transition_copy = copy.deepcopy(transition)
            # note: this is a different id than in the imported .js object. We will fix this in the next step
            transition['id'] = -edge_id
            transition['reversible'] = True
            G.add_edge(end, start, **transition_copy)

    return G

def refactor_incoming_and_outgoing(G: nx.DiGraph) -> nx.DiGraph:
    """
    Rewrites the values of the 'incoming' and 'outgoing' attributes of each node to reflect the actual edges in this
    graph, not the edge names imported from GrappleMap. Otherwise, they would miss some edges

    This will also add some information on the edge that the agent can observe easily, rather than calculate at
    each timestep
    """
    def create_edge_dict(edge_data: List[Tuple[int, int, Dict]]) -> list[dict[str, bool]]:
        transitions_in_or_out = []
        for edge in edge_data:
            assert len(edge) == 3, "3 items expected per edge. Check that data=True is passed to G.out_edges or G.in_edges"
            edge_data = edge[2]
            edge_id = edge_data['id']
            transitions_in_or_out.append({'transition': edge_id,
                                          'swaps_players': edge_data['swaps_players'],
                                          'top': edge_data['top'],
                                          'bottom': edge_data['bottom']})
        return transitions_in_or_out
    for node in G.nodes:
        # get edges in and out
        out_edges = list(G.out_edges(node, data=True))
        in_edges = list(G.in_edges(node, data=True))
        # create list for incoming and outgoing edge
        out_refactor = create_edge_dict(out_edges)
        in_refactor = create_edge_dict(in_edges)
        # Update only the 'incoming' and 'outgoing' attributes
        G.nodes[node]['incoming'] = out_refactor
        G.nodes[node]['outgoing'] = in_refactor

    return G



def load_json(fpath):
    with open(fpath, 'r') as file:
        return json.load(file)
def construct_graph(nodes_path = 'files/nodes.json',transitions_path = 'files/transitions.json') -> nx.classes.digraph.DiGraph:

    nodes = load_json(nodes_path)
    transitions = load_json(transitions_path)
    #tags = load_json('files/tags.json')

    G = add_nodes(nodes)
    G = add_edges(transitions, G)
    G = refactor_incoming_and_outgoing(G)

    # add rewards signal to GrappleMap data
    G = add_rewards_to_graph(G)

    return G

G = construct_graph()
