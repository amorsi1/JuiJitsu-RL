import numpy as np
import networkx as nx

from Game.gym_env import build_action_edge_maps, build_obs
from Game.play_game import Board, GameState, Player


def test_build_action_edge_maps_returns_consistent_mappings() -> None:
    graph = nx.DiGraph()
    graph.add_edge(10, 20, id=101)
    graph.add_edge(20, 30, id=202)

    edge_ids, id_to_index, index_to_id, edge_id_to_nodes = build_action_edge_maps(graph)

    assert edge_ids == [101, 202]
    assert id_to_index == {101: 0, 202: 1}
    assert index_to_id == {0: 101, 1: 202}
    assert edge_id_to_nodes == {101: (10, 20), 202: (20, 30)}


def test_build_obs_matches_expected_vector() -> None:
    graph = nx.DiGraph()
    graph.add_node(77, outgoing=[], description="test")
    game_state = GameState(Board(graph))
    game_state.current_node = 77

    player = Player("P1")
    player.points = 9
    player.is_top = True
    player.is_bottom = False

    other_player = Player("P2")
    other_player.points = 4

    obs = build_obs(game_state, player, turns_left=13, other_player=other_player)

    np.testing.assert_array_equal(
        obs,
        np.array([77, 5, 1, 0, 13], dtype=np.float32),
    )
