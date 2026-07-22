"""Fetches raw data from the MLB Stats API. No transformation happens here.

Every function returns the provider's data structure as-is. Turning it into
the shared schema is the normalizer's job, kept separate so this layer can be
cached or swapped without touching downstream code.
"""

from __future__ import annotations

from typing import Any

import statsapi


def get_schedule(date: str | None = None) -> list[dict[str, Any]]:
    """Games for a date (YYYY-MM-DD). Defaults to today when ``date`` is None."""
    return statsapi.schedule(date=date)


def get_game_playbyplay(game_id: int | str) -> dict[str, Any]:
    """Full play-by-play payload for one game."""
    return statsapi.get("game_playByPlay", {"gamePk": game_id})


def get_game_boxscore(game_id: int | str) -> dict[str, Any]:
    """Box score payload (per-player batting and pitching lines) for one game."""
    return statsapi.boxscore_data(game_id)


def get_player_season_stats(
    player_id: int | str, season: int | None = None, group: str = "hitting"
) -> dict[str, Any]:
    """A player's season stat line for one stat group."""
    return statsapi.player_stat_data(
        player_id, group=group, type="season", season=season
    )


def get_player_career_stats(
    player_id: int | str, group: str = "hitting"
) -> dict[str, Any]:
    """A player's career stat line for one stat group."""
    return statsapi.player_stat_data(player_id, group=group, type="career")


def find_players(name: str) -> list[dict[str, Any]]:
    """Look up players by (partial) name; used to resolve names to IDs."""
    return statsapi.lookup_player(name)


def get_league_leaders(
    category: str,
    season: int | None = None,
    limit: int = 10,
    group: str | None = None,
) -> list[list[Any]]:
    """Leaderboard rows [rank, name, team, value] for a stat category."""
    return statsapi.league_leader_data(
        category, season=season, limit=limit, statGroup=group
    )


def get_player_splits(
    player_id: int | str,
    sit_codes: list[str],
    season: int | None = None,
    group: str = "hitting",
) -> dict[str, Any]:
    """A player's situational splits for the given sitCodes (h, a, vl, vr, ...)."""
    codes = ",".join(sit_codes)
    hydrate = f"stats(group=[{group}],type=[statSplits],sitCodes=[{codes}]"
    if season is not None:
        hydrate += f",season={season}"
    hydrate += ")"
    return statsapi.get("person", {"personId": player_id, "hydrate": hydrate})
