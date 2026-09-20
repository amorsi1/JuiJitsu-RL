from pathlib import Path

import networkx as nx

from Game.play_game import Board, Game, GameState, Player
from Graph.graph_constructor import load_json


def test_swap_players_positions_exchanges_top_bottom_flags():
    game = Game.__new__(Game)
    game.player1 = Player("player_1")
    game.player2 = Player("player_2")

    game.player1.is_top = True
    game.player1.is_bottom = False
    game.player2.is_top = False
    game.player2.is_bottom = True

    game._swap_players_positions()

    assert game.player1.is_top is False
    assert game.player1.is_bottom is True
    assert game.player2.is_top is True
    assert game.player2.is_bottom is False


def test_load_json_nodes_file_successfully():
    nodes_path = Path(__file__).resolve().parents[1] / "GrappleMap_files" / "nodes.json"

    nodes = load_json(nodes_path)

    assert isinstance(nodes, list)
    assert nodes
    assert isinstance(nodes[0], dict)


def test_initialize_samples_only_valid_nodes_without_node_94_bias() -> None:
    """initialize() must be uniform over nodes with outgoing edges (no 50% node-94 bias)."""
    graph = nx.DiGraph()
    graph.add_nodes_from([(n, {"outgoing": [{}]}) for n in (1, 2, 3)])
    graph.add_node(94, outgoing=[])  # dead end, so never a valid start

    state = GameState(Board(graph))
    starts = set()
    for _ in range(200):
        state.initialize()
        starts.add(state.current_node)

    assert starts == {1, 2, 3}
