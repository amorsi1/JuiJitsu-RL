from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AnnouncementKind = Literal["win", "teleport"]
WinType = Literal["submission", "points", "position"]
Placement = Literal["center", "upper_third"]
PanelStyle = Literal["celebration", "text_only"]

WIN_FLASH_DURATION: float = 1.5
TELEPORT_FLASH_DURATION: float = 1.0
# Keep announcement text aligned with HUD player-point text size.
WIN_FLASH_FONT_SIZE: int = 36
TELEPORT_FLASH_FONT_SIZE: int = 36

TELEPORT_TEXT_COLOR: tuple[int, int, int] = (210, 210, 210)
WIN_PANEL_FILL: tuple[int, int, int] = (245, 196, 66)
WIN_PANEL_BORDER: tuple[int, int, int] = (214, 118, 32)
WIN_PANEL_SHADOW: tuple[int, int, int] = (40, 24, 8)

_WIN_TEXT_COLORS: dict[int, tuple[int, int, int]] = {
    0: (220, 60, 60),
    1: (60, 100, 220),
}


@dataclass(frozen=True)
class AnnouncementEvent:
    kind: AnnouncementKind
    winner_index: int | None = None
    win_type: WinType | None = None


@dataclass(frozen=True)
class AnnouncementSpec:
    text: str
    kind: AnnouncementKind
    text_color: tuple[int, int, int]
    font_size: int
    placement: Placement
    hold_seconds: float
    panel_style: PanelStyle


def winner_color_name(winner_index: int) -> str:
    if winner_index == 0:
        return "Red"
    if winner_index == 1:
        return "Blue"
    raise ValueError(f"Unsupported winner_index={winner_index}")


def build_announcement(event: AnnouncementEvent) -> AnnouncementSpec:
    if event.kind == "win":
        if event.winner_index is None:
            raise ValueError("winner_index is required for win announcements")
        if event.win_type is None:
            raise ValueError("win_type is required for win announcements")
        return _build_win_announcement(event.winner_index, event.win_type)
    return _build_teleport_announcement()


def _build_win_announcement(winner_index: int, win_type: WinType) -> AnnouncementSpec:
    color = winner_color_name(winner_index)
    return AnnouncementSpec(
        text=f"{color} Player won by {win_type}",
        kind="win",
        text_color=_WIN_TEXT_COLORS[winner_index],
        font_size=WIN_FLASH_FONT_SIZE,
        placement="center",
        hold_seconds=WIN_FLASH_DURATION,
        panel_style="celebration",
    )


def _build_teleport_announcement() -> AnnouncementSpec:
    return AnnouncementSpec(
        text="Timeout... resetting position",
        kind="teleport",
        text_color=TELEPORT_TEXT_COLOR,
        font_size=TELEPORT_FLASH_FONT_SIZE,
        placement="upper_third",
        hold_seconds=TELEPORT_FLASH_DURATION,
        panel_style="text_only",
    )
