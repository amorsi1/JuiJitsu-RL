import networkx as nx
import pytest

from Game.human_player import make_human_strategy
from Game.play_game import Board, Game, GameState


class StubPositionServer:
    def __init__(self) -> None:
        self.on_move_selected = None
        self.sent_messages = []

    def send_legal_moves(self, current_node: int, moves: list[dict]) -> None:
        self.sent_messages.append(
            {"current_node": current_node, "moves": moves}
        )


def test_make_human_strategy_returns_selected_legal_move() -> None:
    graph = nx.DiGraph()
    graph.add_node(10, outgoing=[])
    game_state = GameState(Board(graph))
    game_state.current_node = 10
    server = StubPositionServer()

    strategy = make_human_strategy(server, game_state)
    possible_moves = [
        (11, {"id": 101, "description": "sweep"}),
        (12, {"id": 202, "description": "pass"}),
    ]

    server.on_move_selected(12)
    selected_move = strategy(possible_moves)

    assert selected_move == possible_moves[1]
    assert server.sent_messages[-1] == {
        "current_node": 10,
        "moves": [
            {"to_node": 11, "transition_id": 101, "description": "sweep"},
            {"to_node": 12, "transition_id": 202, "description": "pass"},
        ],
    }


def test_make_human_strategy_raises_for_illegal_selection() -> None:
    graph = nx.DiGraph()
    graph.add_node(10, outgoing=[])
    game_state = GameState(Board(graph))
    game_state.current_node = 10
    server = StubPositionServer()

    strategy = make_human_strategy(server, game_state)
    possible_moves = [(11, {"id": 101, "description": "sweep"})]

    server.on_move_selected(999)
    with pytest.raises(ValueError, match="not among legal moves"):
        strategy(possible_moves)


def test_game_play_turn_works_with_human_strategy() -> None:
    game = Game("Human integration smoke")
    game.initialize_game("Human", "Agent")
    game.ensure_playable_state()
    server = StubPositionServer()

    human_player = game.current_player
    human_player.strategy = make_human_strategy(server, game.game_state)

    possible_moves = game.game_state.get_possible_moves(
        human_player.is_top,
        human_player.is_bottom,
    )
    assert possible_moves
    expected_node = possible_moves[0][0]

    server.on_move_selected(expected_node)
    game.play_turn()

    assert game.game_state.current_node == expected_node
    assert server.sent_messages, "human strategy should publish legal moves before blocking"
