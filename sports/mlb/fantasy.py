"""Fantasy point scoring for MLB hitters and pitchers.

Fantasy points are not returned by the stats API; they are a deterministic
formula applied to a player's real counting stats. Computing them here (rather
than letting the model do the arithmetic) keeps the number verifiable against
the underlying line.

Uses DraftKings classic scoring. For hitters, hit-by-pitch (+2 on DraftKings) is
omitted because the live box score feed does not expose it. For pitchers, the
no-hitter bonus is in the formula but no feed exposes a no-hitter flag, so it is
always 0 from real data. The same formula is used for game and season lines so
the two stay consistent.
"""

from __future__ import annotations

from typing import Any

SCORING_SYSTEM = "DraftKings classic"
SCORING_SYSTEM_PITCHING = "DraftKings classic (pitching)"

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


# DraftKings MLB classic pitcher point values, verified against DraftKings'
# published rules (not assumed). Complete-game shutout and no-hitter each stack
# on top of the complete-game bonus, matching DraftKings.
_IP = 2.25            # per inning pitched (0.75 per out)
_STRIKEOUT = 2
_WIN = 4
_EARNED_RUN = -2
_HIT_AGAINST = -0.6
_WALK_AGAINST = -0.6
_HIT_BATSMAN = -0.6
_COMPLETE_GAME = 2.5
_COMPLETE_GAME_SHUTOUT = 2.5
_NO_HITTER = 5


def _innings(value: Any) -> float:
    """MLB innings notation to real innings: '5.1' is 5 and 1/3, '5.2' is 5 and 2/3."""
    whole, _, frac = str(value).partition(".")
    try:
        outs = int(frac[0]) if frac else 0
        return int(whole or 0) + outs / 3
    except ValueError:
        return 0.0


def pitcher_points(stats: dict[str, Any]) -> float:
    """Fantasy points for a pitcher from a counting-stat line.

    Expects keys ``ip, k, w, er, h, bb, hbp, cg, cgso, nh``. Innings use MLB
    notation (``5.1`` = 5 and 1/3). ``nh`` (no-hitter) is in the formula for a
    faithful DraftKings score, but no stats feed exposes a no-hitter flag, so it
    is always 0 from real data (the same data-availability reason hitter_points
    omits hit-by-pitch).
    """
    return round(
        _innings(stats.get("ip", 0)) * _IP
        + _num(stats, "k") * _STRIKEOUT
        + _num(stats, "w") * _WIN
        + _num(stats, "er") * _EARNED_RUN
        + _num(stats, "h") * _HIT_AGAINST
        + _num(stats, "bb") * _WALK_AGAINST
        + _num(stats, "hbp") * _HIT_BATSMAN
        + _num(stats, "cg") * _COMPLETE_GAME
        + _num(stats, "cgso") * _COMPLETE_GAME_SHUTOUT
        + _num(stats, "nh") * _NO_HITTER,
        2,
    )
