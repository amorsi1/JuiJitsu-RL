import numpy as np
import networkx as nx
import pytest

from Game.play_game import Board, GameState, Player
from Game.sb3_strategy import make_sb3_strategy


class StubModel:
    def __init__(self, action_index: int):
        self.action_index = action_index
        self.last_obs = None
        self.last_mask = None

    def predict(self, obs, action_masks=None, deterministic=True):
        self.last_obs = obs
        self.last_mask = action_masks
        return np.array(self.action_index), None


class StubGame:
    def __init__(self, graph: nx.DiGraph) -> None:
        self.board = Board(graph)
        self.game_state = GameState(self.board)
        self.game_state.current_node = 10
        self.max_turns = 30
        self.turn_count = 7
        self.player1 = Player("P1")
        self.player2 = Player("P2")
        self.player1.points = 11
        self.player2.points = 4
        self.player1.is_top = True
        self.player1.is_bottom = False
        self.player2.is_top = False
        self.player2.is_bottom = True
        self.current_player = self.player1

    def choose_other_player(self, player):
        return self.player2 if player is self.player1 else self.player1


def _build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge(10, 11, id=1001, description="move A")
    graph.add_edge(10, 12, id=1002, description="move B")
    return graph


def test_make_sb3_strategy_returns_selected_legal_move(monkeypatch) -> None:
    graph = _build_graph()
    game = StubGame(graph)
    model = StubModel(action_index=1)
    monkeypatch.setattr("Game.sb3_strategy._load_maskable_ppo", lambda _: model)

    strategy = make_sb3_strategy("fake-model.zip", game)
    possible_moves = [
        (11, {"id": 1001, "description": "move A"}),
        (12, {"id": 1002, "description": "move B"}),
    ]
    selected = strategy(possible_moves)

    assert selected == possible_moves[1]
    np.testing.assert_array_equal(
        model.last_obs,
        np.array([10, 7, 1, 0, 23], dtype=np.float32),
    )
    assert model.last_mask.dtype == bool
    assert model.last_mask.shape == (2,)
    np.testing.assert_array_equal(model.last_mask, np.array([True, True]))


def test_make_sb3_strategy_rejects_masked_actions(monkeypatch) -> None:
    graph = _build_graph()
    game = StubGame(graph)
    model = StubModel(action_index=1)
    monkeypatch.setattr("Game.sb3_strategy._load_maskable_ppo", lambda _: model)

    strategy = make_sb3_strategy("fake-model.zip", game)
    possible_moves = [(11, {"id": 1001, "description": "move A"})]

    with pytest.raises(ValueError, match="masked action"):
        strategy(possible_moves)
