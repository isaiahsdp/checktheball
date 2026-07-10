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
