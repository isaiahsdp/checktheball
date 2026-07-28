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


def get_stats_by_date(
    date: str, group: str = "hitting", limit: int = 500, sort_stat: str = "homeRuns"
) -> dict[str, Any]:
    """Every player's stat line for a single date, league-wide, in one request."""
    return statsapi.get(
        "stats",
        {
            "stats": "byDateRange",
            "group": group,
            "sportId": 1,
            "gameType": "R",
            "startDate": date,
            "endDate": date,
            "limit": limit,
            "sortStat": sort_stat,
        },
    )


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
    try:
        return statsapi.league_leader_data(
            category, season=season, limit=limit, statGroup=group
        )
    except IndexError:  # wrapper indexes leagueLeaders[0]; an empty category is no rows
        return []


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


def get_player_stats_by_date_range(
    player_id: int | str,
    start_date: str,
    end_date: str,
    group: str = "hitting",
    season: int | None = None,
) -> dict[str, Any]:
    """A player's stats aggregated over a date range (MLB-computed, not by us)."""
    hydrate = (
        f"stats(group=[{group}],type=[byDateRange],"
        f"startDate={start_date},endDate={end_date}"
    )
    if season is not None:
        hydrate += f",season={season}"
    hydrate += ")"
    return statsapi.get("person", {"personId": player_id, "hydrate": hydrate})


def get_player_last_x_games(
    player_id: int | str, count: int, group: str = "hitting", season: int | None = None
) -> dict[str, Any]:
    """A player's stats over their last ``count`` games (MLB-computed)."""
    hydrate = f"stats(group=[{group}],type=[lastXGames],limit={count}"
    if season is not None:
        hydrate += f",season={season}"
    hydrate += ")"
    return statsapi.get("person", {"personId": player_id, "hydrate": hydrate})


def get_player_vs_team(
    player_id: int | str, opponent_team_id: int | str, season: int | None = None, group: str = "hitting"
) -> dict[str, Any]:
    """A player's stats against one opponent team (MLB-computed head-to-head).

    Uses ``vsTeamTotal`` (the aggregate line), not ``vsTeam``: the latter returns
    per-matchup sub-splits with no grand total, which cannot be collapsed safely.
    """
    hydrate = f"stats(group=[{group}],type=[vsTeamTotal],opposingTeamId={opponent_team_id}"
    if season is not None:
        hydrate += f",season={season}"
    hydrate += ")"
    return statsapi.get("person", {"personId": player_id, "hydrate": hydrate})


def get_season_info(season: int) -> dict[str, Any]:
    """Season metadata, including the All-Star break dates."""
    return statsapi.get("season", {"seasonId": season, "sportId": 1})


def find_teams(name: str) -> list[dict[str, Any]]:
    """Look up teams by (partial) name; used to resolve an opponent to an id."""
    return statsapi.lookup_team(name)
