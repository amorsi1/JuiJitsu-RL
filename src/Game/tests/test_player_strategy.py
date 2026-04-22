from Game.play_game import Player


def test_choose_move_uses_callable_strategy() -> None:
    moves = [(1, {"id": 101}), (2, {"id": 202})]

    def pick_last(possible_moves):
        return possible_moves[-1]

    player = Player("Human", strategy=pick_last)
    assert player.choose_move(moves) == moves[-1]
