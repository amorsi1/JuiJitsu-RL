from pathlib import Path

from src.Game.play_game import Game, Player
from src.Graph.graph_constructor import load_json


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
