import pandas as pd
import networkx as nx
from tqdm import tqdm
from Graph.depracated.position import positions_are_equivalent, Position

def find_or_add_node(pos: Position,row,G):
    # note: doesn't need row
    #check if node exists
    for node in G.nodes():
        if positions_are_equivalent(pos,Position(node)):
            return node
    #makes new node if it doesn't exist
    G.add_node(pos.codeblock,description='Unknown', tags='Unknown', is_explicit_position=False, from_transition=row['description'])
    return pos.codeblock


def save_graph_to_csv(G: nx.classes.digraph.DiGraph, nodes_file='files/nodes.csv', edges_file='files/edges.csv'):
    # Save nodes and their attributes
    node_data = []
    for node, attrs in G.nodes(data=True):
        node_info = {'node': node}
        node_info.update(attrs)
        node_data.append(node_info)

    nodes_df = pd.DataFrame(node_data)
    nodes_df.to_csv(nodes_file, index=False)
    print(f"Nodes saved to {nodes_file}")

    # Save edges and their attributes
    edge_data = []
    for u, v, attrs in G.edges(data=True):
        edge_info = {'source': u, 'target': v}
        edge_info.update(attrs)
        edge_data.append(edge_info)

    edges_df = pd.DataFrame(edge_data)
    edges_df.to_csv(edges_file, index=False)
    print(f"Edges saved to {edges_file}")

def main():
    grapplemap = pd.read_csv('files/grapplemap_df.csv',
                             dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
    positions = grapplemap[grapplemap['is_position'] == 1]
    transitions = grapplemap[grapplemap['is_position'] == 0]
    # Create Directed Graph
    G = nx.DiGraph()
    # Add nodes (positions)
    for _, row in positions.iterrows():
        G.add_node(row['code'], description=row['description'], tags=row['tags'], properties=row['properties'], is_explicit_position=True)

    for idx, row in tqdm(transitions.iterrows(), total=len(transitions), desc="Processing transitions"):
        start_pos = Position(row['start_position'])
        end_pos = Position(row['end_position'])

        # find or add start node then update df with result
        start_node = find_or_add_node(start_pos, row, G)
        transitions.loc[idx, 'trans_start_node'] = start_node
        # find or add end node then update df with result
        end_node = find_or_add_node(end_pos, row, G)
        transitions.loc[idx, 'trans_end_node'] = end_node

        # Add the edge (transition)
        G.add_edge(start_node, end_node, description=row['description'], tags=row['tags'], properties=row['properties'])

        # add edge in opposite direction if the move is bidirectional
        if isinstance(row['properties'], str) and 'bidirectional' in row['properties']:
            G.add_edge(end_node, start_node, description=row['description'], tags=row['tags'], properties=row['properties'])
    save_graph_to_csv(G)
    transitions.to_csv('files/graph_transitions.csv') #Depracated

    # Basic graph information
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    print('complete')

if __name__ == "__main__":
    main()