import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from position import positions_are_equivalent, Position
grapplemap = pd.read_csv('files/grapplemap_df.csv', dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
positions = grapplemap[grapplemap['is_position'] == 1]
G_transitions = pd.read_csv('files/graph_transitions.csv')

def create_graph_from_csv(grapplemap_path='grapplemap_df.csv',G_transitions_path='graph_transitions.csv'):
    grapplemap = pd.read_csv(grapplemap_path, dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
    positions = grapplemap[grapplemap['is_position'] == 1]
    G_transitions = pd.read_csv(G_transitions_path)
    G = nx.DiGraph()
    # Add nodes (positions)
    for _, row in positions.iterrows():
        G.add_node(row['code'], description=row['description'], tags=row['tags'], properties=row['properties'],is_explicit_position=True)

    for _, row in G_transitions.iterrows():
        # Add the edge (transition)
        G.add_edge(row['trans_start_node'], row['trans_end_node'], description=row['description'], tags=row['tags'], properties=row['properties'])

        # add edge in opposite direction if the move is bidirectional
        if isinstance(row['properties'], str) and 'bidirectional' in row['properties']:
                G.add_edge(row['trans_end_node'],row['trans_start_node'], description=row['description'], tags=row['tags'], properties=row['properties'])
    return G

G = create_graph_from_csv()

# Find terminal nodes (nodes with out-degree of 0)
terminal_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
terminal_df = pd.DataFrame({'terminal_nodes': [G.nodes[node].get('description') for node in terminal_nodes],
                            'non_end_position': np.zeros(len(terminal_nodes))
                            }
                           )
terminal_df.to_csv('terminal_nodes_annotate.csv')
# Print the list of terminal nodes
print("Terminal nodes:")

for node in terminal_nodes:
    if G.nodes[node].get('is_explicit_position') == True:
        print(G.nodes[node].get('description'))

#     print(type(node))
#     print(node['description'])

# Print the count of terminal nodes
print(f"\nTotal number of terminal nodes: {len(terminal_nodes)}")
print(complete)