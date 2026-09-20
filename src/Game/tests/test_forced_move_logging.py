import networkx as nx

from Game.play_game import Board, Game, GameState, Player


class RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def _make_game(move_count: int) -> tuple[Game, list[tuple[int, dict]], RecordingLogger]:
    graph = nx.DiGraph()
    outgoing_edges = []
    moves = []
    for index in range(move_count):
        target = index + 2
        outgoing_edges.append(
            {"from": 1, "to": target, "top": True, "bottom": False}
        )
        graph.add_node(target, outgoing=[], description=f"target {index + 1}")
        graph.add_edge(
            1,
            target,
            id=100 + index,
            description=f"move {index + 1}",
        )
        moves.append((target, graph.edges[1, target]))
    graph.add_node(1, outgoing=outgoing_edges, description="start")

    logger = RecordingLogger()
    game = Game.__new__(Game)
    game.name = "forced move logging"
    game.logger = logger
    game.board = Board(graph)
    game.game_state = GameState(game.board, logger=logger)
    game.game_state.current_node = 1
    game.max_turns = 100
    game.turn_count = 0
    game.player1 = Player("Player 1")
    game.player1.is_top = True
    game.player1.is_bottom = False
    game.player2 = Player("Player 2")
    game.player2.is_top = False
    game.player2.is_bottom = True
    game.current_player = game.player1
    game.winner = None
    game.win_reason = None
    game.visualizer = None
    game.ensure_playable_state = lambda: None
    return game, moves, logger


def test_single_legal_move_logs_forced_to_perform() -> None:
    game, _, logger = _make_game(move_count=1)

    game.play_turn()

    assert "Player 1 forced to perform 'move 1'" in logger.messages
    assert "Player 1 performed 'move 1'" not in logger.messages


def test_multiple_legal_moves_logs_standard_performed() -> None:
    game, _, logger = _make_game(move_count=2)
    game.current_player.strategy = lambda possible_moves: possible_moves[0]

    game.play_turn()

    assert "Player 1 performed 'move 1'" in logger.messages
    assert "Player 1 forced to perform 'move 1'" not in logger.messages


def test_chosen_single_legal_move_logs_forced_to_perform() -> None:
    game, moves, logger = _make_game(move_count=1)
    target, edge_data = moves[0]

    game.play_turn((target, dict(edge_data)))

    assert "Player 1 forced to perform 'move 1'" in logger.messages
