import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from position import positions_are_equivalent, Position
grapplemap = pd.read_csv('files/grapplemap_df.csv', dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
positions = grapplemap[grapplemap['is_position'] == 1]
G_transitions = pd.read_csv('files/graph_transitions.csv')

def load_graph_from_csv(nodes_file='files/nodes.csv', edges_file='files/edges.csv'):

    # def str_to_bool(string: str):
    #     if string == 'True': return True
    #     else: return False

    # Create a new directed graph
    G = nx.DiGraph()

    # Read nodes CSV
    nodes_df = pd.read_csv(nodes_file)
    for _, row in nodes_df.iterrows():
        node = row['node']
        attrs = {
            'description': row['description'],
            'tags': row['tags'],
            'properties': row['properties'],
            'is_explicit_position': row['is_explicit_position'],
            # 'is_explicit_position': str_to_bool(row['is_explicit_position']),
            'from_transition': row['from_transition']
        }
        # Remove NaN values
        # attrs = {k: v for k, v in attrs.items() if pd.notna(v)}
        G.add_node(node, **attrs)

    # Read edges CSV
    edges_df = pd.read_csv(edges_file)
    for _, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        attrs = {
            'description': row['description'],
            'tags': row['tags'],
            'properties': row['properties']
        }
        # Remove NaN values
        # attrs = {k: v for k, v in attrs.items() if pd.notna(v)}
        G.add_edge(source, target, **attrs)

    return G

def graphs_are_identical(G1, G2):
    # Check if they have the same nodes
    if set(G1.nodes()) != set(G2.nodes()):
        print('diff nodes')
        return False
    # Check if they have the same edges
    if set(G1.edges()) != set(G2.edges()):
        print('diff edges')
        return False
    # Check node attributes
    for node in G1.nodes():
        if G1.nodes[node] != G2.nodes[node]:
            print('diff node attributes')
            print(G1.nodes[node])
            print(G2.nodes[node])
            return False
    # Check edge attributes
    for edge in G1.edges():
        if G1.edges[edge] != G2.edges[edge]:
            print('diff edge attributes')
            print(G1.edges[edge])
            print(G2.edges[edge])
            return False
    return True

def main():
    G = load_graph_from_csv()

    # Find terminal nodes (nodes with out-degree of 0)
    terminal_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    terminal_df = pd.DataFrame({'terminal_nodes': [G.nodes[node].get('description') for node in terminal_nodes],
                                'non_end_position': np.zeros(len(terminal_nodes))
                                }
                               )
    # terminal_df.to_csv('files/terminal_nodes_annotate.csv')
    # Print the list of terminal nodes
    print("Terminal nodes:")

    for node in terminal_nodes:
        if G.nodes[node].get('is_explicit_position') == True:
            print(G.nodes[node].get('description'))

    #     print(type(node))
    #     print(node['description'])

    # Print the count of terminal nodes
    print(f"\nTotal number of terminal nodes: {len(terminal_nodes)}")
    print('complete')