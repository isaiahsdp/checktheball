"""Converts raw MLB Stats API responses into the shared schema (core.schema).

This is the only module aware of MLB's raw JSON layout. It isolates those
quirks so the rest of the pipeline sees uniform, sport-agnostic data.
"""

from __future__ import annotations

from typing import Any

from core import schema
from core.schema import Game, Play, PlayerStat, Team

SPORT = "mlb"

# Raw MLB status strings -> normalized game state.
_LIVE_STATUSES = {"In Progress", "Manager challenge"}
_FINAL_STATUSES = {"Final", "Game Over", "Completed Early"}
_SCHEDULED_STATUSES = {"Scheduled", "Pre-Game", "Warmup", "Delayed Start"}


def _state_from_status(status: str) -> str:
    if status in _LIVE_STATUSES:
        return schema.LIVE
    if status in _FINAL_STATUSES:
        return schema.FINAL
    if status in _SCHEDULED_STATUSES:
        return schema.SCHEDULED
    return schema.OTHER


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_game(raw: dict[str, Any]) -> Game:
    """One schedule entry -> Game."""
    status = raw.get("status", "")
    return Game(
        sport=SPORT,
        game_id=str(raw["game_id"]),
        state=_state_from_status(status),
        status_detail=status,
        start_time=raw.get("game_datetime"),
        home_team=Team(id=str(raw["home_id"]), name=raw["home_name"]),
        away_team=Team(id=str(raw["away_id"]), name=raw["away_name"]),
        home_score=_int_or_none(raw.get("home_score")),
        away_score=_int_or_none(raw.get("away_score")),
        venue=raw.get("venue_name"),
    )


def normalize_play(raw: dict[str, Any], game_id: str, sequence: int) -> Play:
    """One entry from ``allPlays`` -> Play."""
    result = raw.get("result", {})
    about = raw.get("about", {})
    matchup = raw.get("matchup", {})

    players = [
        p["fullName"]
        for p in (matchup.get("batter"), matchup.get("pitcher"))
        if p and p.get("fullName")
    ]

    return Play(
        sport=SPORT,
        game_id=game_id,
        sequence=raw.get("atBatIndex", sequence),
        period=about.get("inning", 0),
        period_half=about.get("halfInning"),
        event=result.get("event", ""),
        description=result.get("description", ""),
        players=players,
        home_score=_int_or_none(result.get("homeScore")),
        away_score=_int_or_none(result.get("awayScore")),
        is_scoring=bool(about.get("isScoringPlay", False)),
    )


def normalize_playbyplay(raw: dict[str, Any], game_id: str) -> list[Play]:
    """Full play-by-play payload -> list of Play, in order."""
    return [
        normalize_play(p, game_id, i)
        for i, p in enumerate(raw.get("allPlays", []))
    ]


def normalize_player_stat(
    raw: dict[str, Any], group: str | None = None
) -> PlayerStat:
    """``player_stat_data`` output -> PlayerStat.

    Picks the stat group matching ``group`` (or the first one returned).
    """
    groups = raw.get("stats", [])
    chosen = None
    if group is not None:
        chosen = next((g for g in groups if g.get("group") == group), None)
    if chosen is None:
        chosen = groups[0] if groups else {}

    scope = chosen.get("type", "season")
    season = _int_or_none(chosen.get("season")) if scope == "season" else None

    name = f"{raw.get('first_name', '')} {raw.get('last_name', '')}".strip()
    return PlayerStat(
        sport=SPORT,
        player_id=str(raw.get("id", "")),
        player_name=name,
        team=raw.get("current_team"),
        group=chosen.get("group", group or ""),
        scope=scope,
        season=season,
        stats=chosen.get("stats", {}),
    )


# Box-score batting fields that describe today's game (counting stats). The
# avg/ops in a box score are season rates, not the game, so they are left out.
_BOX_HITTING_STATS = ("ab", "r", "h", "doubles", "triples", "hr", "rbi", "sb", "bb", "k")

# Our batting-line keys mapped to the MLB byDateRange stat names.
_DATE_RANGE_HITTING = {
    "ab": "atBats",
    "r": "runs",
    "h": "hits",
    "doubles": "doubles",
    "triples": "triples",
    "hr": "homeRuns",
    "rbi": "rbi",
    "sb": "stolenBases",
    "bb": "baseOnBalls",
    "k": "strikeOuts",
}


def normalize_date_range_hitters(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """byDateRange payload -> one batting line per player, same shape as a box line."""
    groups = raw.get("stats", [])
    splits = groups[0].get("splits", []) if groups else []
    lines: list[dict[str, Any]] = []
    for split in splits:
        stat = split.get("stat", {})
        lines.append(
            {
                "player": split.get("player", {}).get("fullName"),
                "team": split.get("team", {}).get("name"),
                "stats": {k: to_number(stat.get(src, 0)) for k, src in _DATE_RANGE_HITTING.items()},
            }
        )
    return lines


# Pitching counting stats from the byDateRange stat type. Keyed by the MLB field
# name (not short keys like the hitting map) so the rankable "strikeOuts" stat
# lines up directly. inningsPitched arrives as a string ("5.1" = 5 and 1/3). The
# win/HBP/complete-game/shutout fields feed pitcher fantasy scoring on today's
# games; the leaderboard itself only ranks by strikeouts.
_DATE_RANGE_PITCHING = {
    "strikeOuts": "strikeOuts",
    "inningsPitched": "inningsPitched",
    "earnedRuns": "earnedRuns",
    "baseOnBalls": "baseOnBalls",
    "hits": "hits",
    "wins": "wins",
    "hitBatsmen": "hitBatsmen",
    "completeGames": "completeGames",
    "shutouts": "shutouts",
}


def normalize_date_range_pitchers(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """byDateRange pitching payload -> one pitching line per player."""
    groups = raw.get("stats", [])
    splits = groups[0].get("splits", []) if groups else []
    lines: list[dict[str, Any]] = []
    for split in splits:
        stat = split.get("stat", {})
        lines.append(
            {
                "player": split.get("player", {}).get("fullName"),
                "team": split.get("team", {}).get("name"),
                "stats": {k: to_number(stat.get(src, 0)) for k, src in _DATE_RANGE_PITCHING.items()},
            }
        )
    return lines


def normalize_boxscore_batters(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Box-score payload -> one batting line per player across both teams.

    Header and total rows (personId 0) are dropped; stats are this game's
    counting totals coerced to numbers.
    """
    team_info = raw.get("teamInfo", {})
    player_info = raw.get("playerInfo", {})
    lines: list[dict[str, Any]] = []
    for side in ("away", "home"):
        info = team_info.get(side, {})
        team = f"{info.get('shortName', '')} {info.get('teamName', '')}".strip()
        for batter in raw.get(f"{side}Batters", []):
            person_id = batter.get("personId")
            if not person_id:  # skip header / total rows
                continue
            # Prefer the full name so answers don't guess ("Machado, M" -> full).
            full_name = player_info.get(f"ID{person_id}", {}).get("fullName")
            lines.append(
                {
                    "player": full_name or batter.get("name"),
                    "team": team,
                    "position": batter.get("position"),
                    "stats": {k: to_number(batter.get(k)) for k in _BOX_HITTING_STATS},
                }
            )
    return lines


def to_number(value: Any) -> Any:
    """Coerce a stat value to int/float when possible; leave it unchanged if not.

    The API returns some stats as strings ("58", ".322"); numeric comparison
    and projection need real numbers, but non-numeric values (e.g. "-.--")
    pass through untouched.
    """
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return value
    return value


def normalize_leaders(rows: list[list[Any]]) -> list[dict[str, Any]]:
    """Leaderboard rows [rank, name, team, value] -> list of dicts."""
    leaders = []
    for row in rows:
        if len(row) < 4:
            continue
        leaders.append(
            {
                "rank": to_number(row[0]),
                "player": row[1],
                "team": row[2],
                "value": to_number(row[3]),
            }
        )
    return leaders


def normalize_total_stat(raw: dict[str, Any]) -> dict[str, Any]:
    """Typed-stats person payload (byDateRange, lastXGames, vsTeam) -> total line.

    These types return a split per team plus an "All" grand total (sport id 0),
    often duplicated. Pick the grand total; never sum, or duplicates double it.
    """
    people = raw.get("people", [])
    stats_groups = people[0].get("stats", []) if people else []
    splits = stats_groups[0].get("splits", []) if stats_groups else []
    if not splits:
        return {}
    total = next((s for s in splits if s.get("sport", {}).get("id") == 0), None)
    if total is None:
        total = max(splits, key=lambda s: s.get("stat", {}).get("gamesPlayed", 0) or 0)
    return total.get("stat", {})


def normalize_splits(
    raw: dict[str, Any], group: str | None = None
) -> list[dict[str, Any]]:
    """Split-hydrated person payload -> list of {code, description, stats}."""
    people = raw.get("people", [])
    if not people:
        return []

    splits = []
    for stat_group in people[0].get("stats", []):
        for entry in stat_group.get("splits", []):
            meta = entry.get("split", {})
            splits.append(
                {
                    "code": meta.get("code"),
                    "description": meta.get("description"),
                    "stats": entry.get("stat", {}),
                }
            )
    return splits
