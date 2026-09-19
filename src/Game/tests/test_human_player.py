import networkx as nx
import pytest

from Game.human_player import make_human_strategy
from Game.play_game import Board, Game, GameState


class StubPositionServer:
    """Stub server that can answer a legal_moves prompt the way the browser would.

    ``reply_with`` matters for ordering: the strategy discards anything queued before
    it sends the prompt, so a selection must arrive *in response to* the prompt, not
    ahead of it. That is also the only ordering the real viewer can produce -- nodes
    are not clickable until a legal_moves message renders them.
    """

    def __init__(self, reply_with: int | None = None) -> None:
        self.on_move_selected = None
        self.sent_messages = []
        self.reply_with = reply_with

    def send_legal_moves(self, current_node: int, moves: list[dict]) -> None:
        self.sent_messages.append(
            {"current_node": current_node, "moves": moves}
        )
        if self.reply_with is not None and self.on_move_selected is not None:
            self.on_move_selected(self.reply_with)


def test_make_human_strategy_returns_selected_legal_move() -> None:
    graph = nx.DiGraph()
    graph.add_node(10, outgoing=[])
    game_state = GameState(Board(graph))
    game_state.current_node = 10
    server = StubPositionServer(reply_with=12)

    strategy = make_human_strategy(server, game_state)
    possible_moves = [
        (11, {"id": 101, "description": "sweep"}),
        (12, {"id": 202, "description": "pass"}),
    ]

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
    server = StubPositionServer(reply_with=999)

    strategy = make_human_strategy(server, game_state)
    # Need 2+ moves so the validation path is reached (single-move is auto-selected)
    possible_moves = [
        (11, {"id": 101, "description": "sweep"}),
        (12, {"id": 202, "description": "pass"}),
    ]

    with pytest.raises(ValueError, match="not among legal moves"):
        strategy(possible_moves)


def test_make_human_strategy_auto_selects_sole_move() -> None:
    graph = nx.DiGraph()
    graph.add_node(10, outgoing=[])
    game_state = GameState(Board(graph))
    game_state.current_node = 10
    server = StubPositionServer()

    strategy = make_human_strategy(server, game_state)
    only_move = (11, {"id": 101, "description": "sweep"})

    result = strategy([only_move])

    assert result == only_move
    # The move is forced, but the browser still needs the legal_moves message so the
    # graph overlay can animate the transition instead of jumping to it.
    assert server.sent_messages, "forced move should still publish legal moves"
    assert server.sent_messages[-1]["moves"] == [
        {"to_node": 11, "transition_id": 101, "description": "sweep"}
    ]


def test_forced_move_click_does_not_desync_the_next_turn() -> None:
    """A click on a forced move must not be consumed as the next turn's choice.

    Forced moves still publish legal_moves, and the viewer renders that lone node as
    clickable -- but the forced path returns without reading the queue, so the click
    is left behind. Without the stale-selection drain, the next turn would consume it
    and every turn after would be shifted.
    """
    graph = nx.DiGraph()
    graph.add_node(10, outgoing=[])
    game_state = GameState(Board(graph))
    game_state.current_node = 10
    server = StubPositionServer()

    strategy = make_human_strategy(server, game_state)

    # Turn 1: forced move, and the player clicks the lone node anyway.
    only_move = (11, {"id": 101, "description": "sweep"})
    server.reply_with = 11
    assert strategy([only_move]) == only_move

    # Turn 2: the real choice must win over the orphaned click from turn 1.
    server.reply_with = 13
    possible_moves = [
        (11, {"id": 101, "description": "sweep"}),
        (13, {"id": 303, "description": "mount"}),
    ]
    assert strategy(possible_moves) == possible_moves[1]


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

    server.reply_with = expected_node
    game.play_turn()

    assert game.game_state.current_node == expected_node
    assert server.sent_messages, "human strategy should publish legal moves every turn"
