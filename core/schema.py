"""Shared, sport-agnostic data shapes.

Every sport's normalizer must produce these; the rest of the pipeline only
ever consumes these. Field names avoid sport-specific vocabulary (``period``
rather than ``inning``) so the same shapes describe MLB, NBA, and beyond.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Normalized game states. The raw feeds use many strings ("In Progress",
# "Warmup", "Game Over", ...); we collapse them to these three plus "other".
SCHEDULED = "scheduled"
LIVE = "live"
FINAL = "final"
OTHER = "other"


@dataclass
class Team:
    """One side of a game."""

    id: str
    name: str


@dataclass
class Game:
    """A single game, normalized across sports."""

    sport: str
    game_id: str
    state: str  # one of SCHEDULED / LIVE / FINAL / OTHER
    status_detail: str  # raw provider status, kept for display/debugging
    start_time: str | None  # ISO-8601 UTC
    home_team: Team
    away_team: Team
    home_score: int | None
    away_score: int | None
    venue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Play:
    """A single scoring/at-bat-level event within a game."""

    sport: str
    game_id: str
    sequence: int  # order of the play within the game
    period: int  # inning (MLB), quarter (NBA), ...
    event: str  # short label, e.g. "Home Run", "Strikeout"
    description: str  # full human-readable description
    players: list[str] = field(default_factory=list)  # names involved
    period_half: str | None = None  # "top"/"bottom" for MLB; None otherwise
    home_score: int | None = None  # score after the play
    away_score: int | None = None
    is_scoring: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerStat:
    """A player's stat line for one group over one scope.

    ``stats`` is an open mapping of stat name -> value. Its keys are
    sport-specific (homeRuns, points, ...) by nature; the container is uniform.
    """

    sport: str
    player_id: str
    player_name: str
    team: str | None
    group: str  # "hitting", "pitching", ...
    scope: str  # "season" or "career"
    season: int | None  # None for career
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
