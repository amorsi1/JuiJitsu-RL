import pandas as pd
import networkx as nx
from tqdm import tqdm
from position import positions_are_equivalent, Position

grapplemap = pd.read_csv('files/grapplemap_df.csv', dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
positions = grapplemap[grapplemap['is_position'] == 1]
transitions = grapplemap[grapplemap['is_transition'] == 1]

# Create a directed graph
G = nx.DiGraph()
def find_or_add_node(pos: Position,G):
    # note: doesn't need row
    #check if node exists
    for node in G.nodes():
        if positions_are_equivalent(pos,Position(node)):
            return node
    #makes new node if it doesn't exist
    G.add_node(pos.codeblock,description='Unknown', tags='Unknown', is_explicit_position=False, from_transition=row['description'])
    return pos.codeblock

# Add nodes (positions)
for _, row in positions.iterrows():
    G.add_node(row['code'], description=row['description'], tags=row['tags'], properties=row['properties'],is_explicit_position=True)

for idx, row in tqdm(transitions.iterrows(), total=len(transitions), desc="Processing transitions"):
    start_pos = Position(row['start_position'])
    end_pos = Position(row['end_position'])

    # find or add start node then update df with result
    start_node = find_or_add_node(start_pos, G)
    transitions.loc[idx, 'trans_start_node'] = start_node
    # find or add end node then update df with result
    end_node = find_or_add_node(end_pos, G)
    transitions.loc[idx, 'trans_end_node'] = end_node

    # Add the edge (transition)
    G.add_edge(start_node, end_node, description=row['description'], tags=row['tags'], properties=row['properties'])

    # add edge in opposite direction if the move is bidirectional
    if isinstance(row['properties'], str) and 'bidirectional' in row['properties']:
        G.add_edge(end_node, start_node, description=row['description'], tags=row['tags'], properties=row['properties'])

transitions.to_csv('graph_transitions.csv')
# Basic graph information
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

print('complete')
