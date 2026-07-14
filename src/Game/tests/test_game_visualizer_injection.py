from typing import Any, List, Optional, Tuple

from Game.play_game import Game


class StubVisualizer:
    """Typed stand-in for Visualizer3D that records calls instead of opening a browser/server."""

    def __init__(self) -> None:
        self.game_ref: Optional[Game] = None
        self.turn_delay: float = 0.0
        self.update_calls: List[Tuple[Tuple[Any, ...], dict]] = []
        self.close_count: int = 0

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.update_calls.append((args, kwargs))

    def close(self) -> None:
        self.close_count += 1


def test_injected_visualizer_is_used_and_linked_back_to_game() -> None:
    stub = StubVisualizer()

    game = Game("injected-visualizer-test", visualizer=stub)

    assert game.visualizer is stub
    assert stub.game_ref is game


def test_injected_visualizer_is_not_closed_after_play_game() -> None:
    stub = StubVisualizer()
    game = Game("injected-visualizer-not-closed", visualizer=stub, max_turns=3)

    game.initialize_game("A", "B")
    game.play_game()

    assert stub.close_count == 0


def test_game_without_visualizer_arg_has_none_visualizer() -> None:
    game = Game("no-visualizer-test")

    assert game.visualizer is None
