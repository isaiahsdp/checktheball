"""Fantasy point scoring for MLB hitters.

Fantasy points are not returned by the stats API; they are a deterministic
formula applied to a player's real counting stats. Computing them here (rather
than letting the model do the arithmetic) keeps the number verifiable against
the underlying line.

Uses DraftKings classic hitter scoring. Hit-by-pitch (+2 on DraftKings) is
omitted because the live box score feed does not expose it; the same formula is
used for game and season lines so the two stay consistent.
"""

from __future__ import annotations

from typing import Any

SCORING_SYSTEM = "DraftKings classic"

# DraftKings MLB hitter point values.
_SINGLE = 3
_DOUBLE = 5
_TRIPLE = 8
_HOME_RUN = 10
_RBI = 2
_RUN = 2
_WALK = 2
_STOLEN_BASE = 5


def _num(stats: dict[str, Any], key: str) -> float:
    value = stats.get(key, 0)
    return value if isinstance(value, (int, float)) else 0


def hitter_points(stats: dict[str, Any]) -> float:
    """Fantasy points for a hitter from a counting-stat line.

    Expects keys ``h, doubles, triples, hr, rbi, r, bb, sb``. Singles are
    derived from hits minus extra-base hits.
    """
    doubles = _num(stats, "doubles")
    triples = _num(stats, "triples")
    home_runs = _num(stats, "hr")
    singles = max(0, _num(stats, "h") - doubles - triples - home_runs)
    return round(
        singles * _SINGLE
        + doubles * _DOUBLE
        + triples * _TRIPLE
        + home_runs * _HOME_RUN
        + _num(stats, "rbi") * _RBI
        + _num(stats, "r") * _RUN
        + _num(stats, "bb") * _WALK
        + _num(stats, "sb") * _STOLEN_BASE,
        2,
    )
