import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest

import Game.policies as policies
from Game.play_game import Game
from Game.policies import (
    PolicySpec,
    StrategyContext,
    available_policies,
    build_config_options,
    discover_sb3_policies,
    get_policy,
    register_policy,
    resolve_strategies,
)


class StubPositionServer:
    """Minimal stand-in for PositionServer, matching the shape used in test_human_player.py."""

    def __init__(self) -> None:
        self.on_move_selected: Optional[Callable[[int], None]] = None
        self.sent_messages: List[Dict[str, Any]] = []

    def send_legal_moves(self, current_node: int, moves: List[dict]) -> None:
        self.sent_messages.append({"current_node": current_node, "moves": moves})


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot and restore the module-level policy registry around each test."""
    snapshot = dict(policies._REGISTRY)
    try:
        yield
    finally:
        policies._REGISTRY.clear()
        policies._REGISTRY.update(snapshot)


def _make_spec(policy_id: str = "dummy", kind: str = "computer") -> PolicySpec:
    return PolicySpec(
        id=policy_id,
        label="Dummy",
        kind=kind,
        factory=lambda ctx: "random",
        description="a dummy policy",
    )


# --- register/get roundtrip ---

def test_register_and_get_policy_roundtrip() -> None:
    spec = _make_spec("roundtrip-policy")
    register_policy(spec)

    fetched = get_policy("roundtrip-policy")

    assert fetched is spec


def test_register_duplicate_id_raises_value_error() -> None:
    spec = _make_spec("dup-policy")
    register_policy(spec)

    with pytest.raises(ValueError):
        register_policy(_make_spec("dup-policy"))


def test_get_unknown_policy_id_lists_available_ids() -> None:
    register_policy(_make_spec("known-policy"))

    with pytest.raises(ValueError) as exc_info:
        get_policy("totally-unknown-policy")

    message = str(exc_info.value)
    assert "known-policy" in message
    assert "random" in message  # built-in should also be listed


# --- built-ins ---

def test_builtin_random_policy_registered() -> None:
    spec = get_policy("random")

    assert spec.kind == "computer"
    ctx = StrategyContext(game=Game("builtin-random-test"))
    assert spec.factory(ctx) == "random"


def test_builtin_human_policy_registered() -> None:
    spec = get_policy("human")

    assert spec.kind == "human"


def test_builtin_human_factory_raises_without_server() -> None:
    spec = get_policy("human")
    game = Game("builtin-human-test")
    game.initialize_game("P1", "P2")
    ctx = StrategyContext(game=game, server=None)

    with pytest.raises(ValueError):
        spec.factory(ctx)


def test_available_policies_includes_builtins() -> None:
    ids = {spec.id for spec in available_policies()}

    assert "random" in ids
    assert "human" in ids


# --- discover_sb3_policies ---

def test_discover_sb3_policies_finds_nested_and_top_level_zips(tmp_path: Path) -> None:
    (tmp_path / "a.zip").touch()
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "b.zip").touch()

    discovered = discover_sb3_policies(tmp_path)

    ids = {spec.id for spec in discovered}
    assert ids == {"sb3:a", "sb3:sub/b"}
    for spec in discovered:
        assert spec.kind == "computer"


def test_discover_sb3_policies_does_not_import_sb3(tmp_path: Path) -> None:
    (tmp_path / "c.zip").touch()

    # If discovery imported sb3_contrib/torch just to list models, this would
    # either raise (if not installed) or be needlessly slow. Neither factory
    # invocation nor import should happen during discovery.
    discovered = discover_sb3_policies(tmp_path)

    assert len(discovered) == 1
    # Factories are not invoked during discovery — only registered.
    spec = discovered[0]
    assert callable(spec.factory)


def test_discover_sb3_policies_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "d.zip").touch()

    first = discover_sb3_policies(tmp_path)
    # Calling again with the same models_dir should not raise on duplicate ids.
    second = discover_sb3_policies(tmp_path)

    assert len(first) == 1
    assert len(second) == 1


# --- build_config_options ---

def test_build_config_options_is_json_serializable_and_contains_defaults() -> None:
    register_policy(_make_spec("config-options-policy"))

    options = build_config_options(max_turns_default=42, turn_delay_default=1.5)

    # Should not raise.
    serialized = json.dumps(options)
    assert serialized

    assert options["settings"]["max_turns"]["default"] == 42
    assert options["settings"]["turn_delay"]["default"] == 1.5
    assert "type" not in options


def test_build_config_options_one_entry_per_registered_policy_with_expected_keys() -> None:
    register_policy(_make_spec("policy-a"))
    register_policy(_make_spec("policy-b", kind="human"))

    options = build_config_options()

    policy_ids = {entry["id"] for entry in options["policies"]}
    assert "policy-a" in policy_ids
    assert "policy-b" in policy_ids
    assert len(options["policies"]) == len(available_policies())

    for entry in options["policies"]:
        assert set(entry.keys()) == {"id", "label", "kind", "description"}


# --- resolve_strategies ---

def test_resolve_strategies_human_vs_human_shares_single_strategy() -> None:
    game = Game("resolve-human-vs-human")
    game.initialize_game("P1", "P2")
    server = StubPositionServer()
    ctx = StrategyContext(game=game, server=server)
    config_players = {"p1": {"type": "human"}, "p2": {"type": "human"}}

    strategies = resolve_strategies(config_players, ctx)

    assert strategies["p1"] is strategies["p2"]
    assert callable(strategies["p1"])


def test_resolve_strategies_computer_random() -> None:
    game = Game("resolve-computer-random")
    game.initialize_game("P1", "P2")
    ctx = StrategyContext(game=game)
    config_players = {
        "p1": {"type": "computer", "policy_id": "random"},
        "p2": {"type": "computer", "policy_id": "random"},
    }

    strategies = resolve_strategies(config_players, ctx)

    assert strategies["p1"] == "random"
    assert strategies["p2"] == "random"


def test_resolve_strategies_rejects_human_policy_id_under_computer_type() -> None:
    game = Game("resolve-human-under-computer")
    game.initialize_game("P1", "P2")
    server = StubPositionServer()
    ctx = StrategyContext(game=game, server=server)
    config_players = {
        "p1": {"type": "computer", "policy_id": "human"},
        "p2": {"type": "computer", "policy_id": "random"},
    }

    with pytest.raises(ValueError):
        resolve_strategies(config_players, ctx)


def test_resolve_strategies_unknown_policy_id_raises() -> None:
    game = Game("resolve-unknown-policy")
    game.initialize_game("P1", "P2")
    ctx = StrategyContext(game=game)
    config_players = {
        "p1": {"type": "computer", "policy_id": "does-not-exist"},
        "p2": {"type": "computer", "policy_id": "random"},
    }

    with pytest.raises(ValueError):
        resolve_strategies(config_players, ctx)
