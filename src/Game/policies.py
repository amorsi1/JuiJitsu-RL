"""Extensible registry of player policies (human and computer strategies).

A `PolicySpec` describes how to build a `Player.strategy`-compatible callable
(or the `"random"` string strategy key) from a `StrategyContext`. Callers
register policies (built-ins are registered at import time) and later resolve
a config dict — as produced by e.g. a game-setup UI — into concrete
strategies via `resolve_strategies`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Optional

Strategy = str | Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class StrategyContext:
    game: Any  # Game — typed as Any to avoid circular import
    server: Optional[Any] = None  # PositionServer


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    id: str
    label: str
    kind: str  # 'human' | 'computer'
    factory: Callable[[StrategyContext], Strategy]
    description: str = ""
    metadata: dict = dataclasses.field(default_factory=dict)


_REGISTRY: dict[str, PolicySpec] = {}


def register_policy(spec: PolicySpec) -> None:
    if spec.id in _REGISTRY:
        raise ValueError(f"Policy id already registered: {spec.id!r}")
    _REGISTRY[spec.id] = spec


def get_policy(policy_id: str) -> PolicySpec:
    try:
        return _REGISTRY[policy_id]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown policy id: {policy_id!r}. Available policy ids: {available}"
        ) from None


def available_policies() -> list[PolicySpec]:
    return [spec for _, spec in sorted(_REGISTRY.items())]


def _human_policy_factory(ctx: StrategyContext) -> Strategy:
    if ctx.server is None:
        raise ValueError("The 'human' policy requires a StrategyContext.server")
    from Game.human_player import make_human_strategy

    return make_human_strategy(ctx.server, ctx.game.game_state)


def build_config_options(
    max_turns_default: int = 30, turn_delay_default: float = 0.0
) -> dict:
    """Build a JSON-serializable payload describing configurable game options."""
    return {
        "settings": {
            "max_turns": {
                "default": max_turns_default,
                "min": 1,
                "max": 500,
            },
            "turn_delay": {
                "default": turn_delay_default,
                "min": 0.0,
                "max": 10.0,
            },
        },
        "policies": [
            {
                "id": spec.id,
                "label": spec.label,
                "kind": spec.kind,
                "description": spec.description,
            }
            for spec in available_policies()
        ],
    }


def discover_sb3_policies(models_dir: Path) -> list[PolicySpec]:
    """Discover MaskablePPO checkpoints under `models_dir` without importing sb3/torch.

    Returns fresh `PolicySpec` objects on every call; callers are responsible
    for registering them (this function never touches `_REGISTRY`).
    """
    specs = []
    for zip_path in sorted(Path(models_dir).glob("**/*.zip")):
        relative_path = zip_path.relative_to(models_dir).with_suffix("")
        relative_id = relative_path.as_posix()
        policy_id = f"sb3:{relative_id}"

        def factory(ctx: StrategyContext, _zip_path: Path = zip_path) -> Strategy:
            from Game.sb3_strategy import make_sb3_strategy

            return make_sb3_strategy(_zip_path, ctx.game)

        specs.append(
            PolicySpec(
                id=policy_id,
                label=relative_path.name,
                kind="computer",
                factory=factory,
                description=f"MaskablePPO checkpoint at {relative_id}.zip",
            )
        )
    return specs


def resolve_strategies(
    config_players: dict, ctx: StrategyContext
) -> dict[str, Strategy]:
    """Resolve a `{player_key: {"type": ..., "policy_id": ...}}` config into strategies.

    Factory results are memoized per policy id within a single call, so e.g.
    two human players share the same underlying strategy callable.
    """
    resolved_cache: dict[str, Strategy] = {}

    def build(policy_id: str) -> Strategy:
        if policy_id not in resolved_cache:
            resolved_cache[policy_id] = get_policy(policy_id).factory(ctx)
        return resolved_cache[policy_id]

    strategies: dict[str, Strategy] = {}
    for player_key, player_config in config_players.items():
        player_type = player_config["type"]
        if player_type == "human":
            strategies[player_key] = build("human")
        elif player_type == "computer":
            if "policy_id" not in player_config:
                raise ValueError(
                    f"Player {player_key!r} config of type 'computer' requires 'policy_id'"
                )
            policy_id = player_config["policy_id"]
            spec = get_policy(policy_id)
            if spec.kind != player_type:
                raise ValueError(
                    f"Player {player_key!r} declared type {player_type!r} but "
                    f"policy {policy_id!r} has kind {spec.kind!r}"
                )
            strategies[player_key] = build(policy_id)
        else:
            raise ValueError(
                f"Player {player_key!r} has unknown config type: {player_type!r}"
            )
    return strategies


register_policy(
    PolicySpec(
        id="random",
        label="Random",
        kind="computer",
        factory=lambda ctx: "random",
        description="Selects a uniformly random legal move.",
    )
)

register_policy(
    PolicySpec(
        id="human",
        label="Human",
        kind="human",
        factory=_human_policy_factory,
        description="Prompts a human player via the browser overlay.",
    )
)
