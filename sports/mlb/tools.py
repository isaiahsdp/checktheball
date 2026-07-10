"""Claude-callable tool functions for MLB, with their tool-use JSON schemas.

Each function returns a JSON-serializable dict built from real data. Expected
failures (unknown player, unavailable stat) return an ``{"error": ...}`` dict
rather than raising, so the orchestrator can hand the model a message it can
explain to the user instead of crashing the request.

``TOOL_SCHEMAS`` advertises these functions to Claude; ``TOOL_FUNCTIONS`` maps
tool names back to the callables so the orchestrator can dispatch a tool call.
Every sport module exposes this same pair, keeping the core loop sport-agnostic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sports.mlb import client, normalizer

SPORT = "mlb"
FULL_MLB_SEASON_GAMES = 162

# Friendly split name -> MLB sitCode.
_SPLIT_CODES = {
    "home": "h",
    "away": "a",
    "vs_left": "vl",
    "vs_right": "vr",
}


def _current_season() -> int:
    return datetime.now().year


def _resolve_player(name: str) -> tuple[str, str] | None:
    """Resolve a name to (player_id, full_name), or None if not found."""
    matches = client.find_players(name)
    if not matches:
        return None
    top = matches[0]
    return str(top["id"]), top["fullName"]


def get_player_stat(
    player: str, stat: str, season: int | None = None, group: str = "hitting"
) -> dict[str, Any]:
    """One stat for one player, for a season (or career if season is None)."""
    resolved = _resolve_player(player)
    if resolved is None:
        return {"error": f"No player found matching '{player}'."}
    player_id, full_name = resolved

    if season is not None:
        raw = client.get_player_season_stats(player_id, season=season, group=group)
    else:
        raw = client.get_player_career_stats(player_id, group=group)
    line = normalizer.normalize_player_stat(raw, group=group)

    if stat not in line.stats:
        return {
            "error": f"Stat '{stat}' not available for {full_name} ({group}).",
            "available_stats": sorted(line.stats.keys()),
        }

    return {
        "player": line.player_name or full_name,
        "player_id": player_id,
        "team": line.team,
        "stat": stat,
        "value": line.stats[stat],
        "scope": line.scope,
        "season": line.season,
        "group": group,
    }


def compare_players(
    player_a: str,
    player_b: str,
    stat: str,
    season: int | None = None,
    group: str = "hitting",
) -> dict[str, Any]:
    """Compare two players on one stat and report who leads and by how much."""
    a = get_player_stat(player_a, stat, season, group)
    if "error" in a:
        return a
    b = get_player_stat(player_b, stat, season, group)
    if "error" in b:
        return b

    va = normalizer.to_number(a["value"])
    vb = normalizer.to_number(b["value"])

    leader: str | None = None
    difference: float | None = None
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        if va > vb:
            leader = a["player"]
        elif vb > va:
            leader = b["player"]
        else:
            leader = "tie"
        difference = round(abs(va - vb), 3)

    return {
        "stat": stat,
        "scope": a["scope"],
        "season": a["season"],
        "group": group,
        "players": [a, b],
        "leader": leader,
        "difference": difference,
    }


def get_top_performers(
    stat: str, season: int | None = None, limit: int = 5, group: str = "hitting"
) -> dict[str, Any]:
    """Leaderboard of the top players in a stat category for a season."""
    if season is None:
        season = _current_season()
    rows = client.get_league_leaders(stat, season=season, limit=limit, group=group)
    leaders = normalizer.normalize_leaders(rows)
    if not leaders:
        return {"error": f"No leaderboard data for '{stat}' in {season}."}
    return {"stat": stat, "season": season, "limit": limit, "leaders": leaders}


def compute_pace_projection(
    player: str, stat: str, season: int | None = None
) -> dict[str, Any]:
    """Project a player's current-season counting stat over a full 162 games."""
    if season is None:
        season = _current_season()
    resolved = _resolve_player(player)
    if resolved is None:
        return {"error": f"No player found matching '{player}'."}
    player_id, full_name = resolved

    raw = client.get_player_season_stats(player_id, season=season, group="hitting")
    line = normalizer.normalize_player_stat(raw, group="hitting")

    if stat not in line.stats or "gamesPlayed" not in line.stats:
        return {
            "error": f"Can't project '{stat}' for {full_name} in {season}.",
            "available_stats": sorted(line.stats.keys()),
        }

    value = normalizer.to_number(line.stats[stat])
    games_played = normalizer.to_number(line.stats["gamesPlayed"])
    if not isinstance(value, (int, float)) or not isinstance(games_played, (int, float)):
        return {"error": f"'{stat}' is not a countable stat to project."}
    if games_played <= 0:
        return {"error": f"{full_name} has no games played in {season} to project from."}

    projected = round(value / games_played * FULL_MLB_SEASON_GAMES, 1)
    return {
        "player": line.player_name or full_name,
        "stat": stat,
        "season": season,
        "current_value": value,
        "games_played": games_played,
        "full_season_games": FULL_MLB_SEASON_GAMES,
        "projected_value": projected,
        "method": "current_value / games_played * 162",
    }


def get_situational_split(
    player: str, split: str, season: int | None = None, group: str = "hitting"
) -> dict[str, Any]:
    """A player's stat line in a situation (home, away, vs_left, vs_right)."""
    code = _SPLIT_CODES.get(split)
    if code is None:
        return {
            "error": f"Unknown split '{split}'.",
            "available_splits": sorted(_SPLIT_CODES),
        }
    resolved = _resolve_player(player)
    if resolved is None:
        return {"error": f"No player found matching '{player}'."}
    player_id, full_name = resolved

    if season is None:
        season = _current_season()
    raw = client.get_player_splits(player_id, [code], season=season, group=group)
    splits = normalizer.normalize_splits(raw, group=group)
    match = next((s for s in splits if s["code"] == code), None)
    if match is None:
        return {"error": f"No '{split}' split data for {full_name} in {season}."}

    return {
        "player": full_name,
        "player_id": player_id,
        "split": split,
        "split_description": match["description"],
        "season": season,
        "group": group,
        "stats": match["stats"],
    }


TOOL_FUNCTIONS = {
    "get_player_stat": get_player_stat,
    "compare_players": compare_players,
    "get_top_performers": get_top_performers,
    "compute_pace_projection": compute_pace_projection,
    "get_situational_split": get_situational_split,
}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name. Used by the orchestrator in Stage 3."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    return func(**arguments)


_STAT_HINT = (
    "Stat key as used by the MLB API, e.g. 'homeRuns', 'hits', 'avg', 'ops', "
    "'rbi', 'stolenBases' (hitting) or 'era', 'wins', 'strikeOuts' (pitching)."
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_player_stat",
        "description": (
            "Look up a single statistic for one player. Use for questions about "
            "one player's number in a stat. Returns career totals if no season "
            "is given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name, e.g. 'Aaron Judge'."},
                "stat": {"type": "string", "description": _STAT_HINT},
                "season": {"type": "integer", "description": "Four-digit year. Omit for career totals."},
                "group": {
                    "type": "string",
                    "enum": ["hitting", "pitching", "fielding"],
                    "description": "Stat group. Defaults to 'hitting'.",
                },
            },
            "required": ["player", "stat"],
        },
    },
    {
        "name": "compare_players",
        "description": (
            "Compare two players on the same stat and report who leads and by "
            "how much. Use for 'who has more X, A or B' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_a": {"type": "string", "description": "First player's full name."},
                "player_b": {"type": "string", "description": "Second player's full name."},
                "stat": {"type": "string", "description": _STAT_HINT},
                "season": {"type": "integer", "description": "Four-digit year. Omit to compare career totals."},
                "group": {
                    "type": "string",
                    "enum": ["hitting", "pitching", "fielding"],
                    "description": "Stat group. Defaults to 'hitting'.",
                },
            },
            "required": ["player_a", "player_b", "stat"],
        },
    },
    {
        "name": "get_top_performers",
        "description": (
            "Get the league leaderboard for a stat category in a season. Use for "
            "'who leads the league in X' or 'top 5 in X' questions. The category "
            "uses MLB leader names, e.g. 'homeRuns', 'battingAverage', "
            "'runsBattedIn', 'stolenBases', 'earnedRunAverage', 'strikeouts'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stat": {"type": "string", "description": "Leader category name."},
                "season": {"type": "integer", "description": "Four-digit year. Defaults to the current season."},
                "limit": {"type": "integer", "description": "How many players to return. Defaults to 5."},
                "group": {
                    "type": "string",
                    "enum": ["hitting", "pitching", "fielding"],
                    "description": "Stat group. Defaults to 'hitting'.",
                },
            },
            "required": ["stat"],
        },
    },
    {
        "name": "compute_pace_projection",
        "description": (
            "Project a player's current-season counting stat (home runs, hits, "
            "RBIs, etc.) across a full 162-game season, based on games played so "
            "far. Use for 'on pace for' questions. Not for rate stats like avg."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name."},
                "stat": {"type": "string", "description": "A counting stat, e.g. 'homeRuns', 'hits', 'rbi'."},
                "season": {"type": "integer", "description": "Four-digit year. Defaults to the current season."},
            },
            "required": ["player", "stat"],
        },
    },
    {
        "name": "get_situational_split",
        "description": (
            "Get a player's stat line in a specific situation for a season. Use "
            "for 'how does X hit at home / on the road / vs lefties / vs righties'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name."},
                "split": {
                    "type": "string",
                    "enum": ["home", "away", "vs_left", "vs_right"],
                    "description": "Which situational split to return.",
                },
                "season": {"type": "integer", "description": "Four-digit year. Defaults to the current season."},
                "group": {
                    "type": "string",
                    "enum": ["hitting", "pitching"],
                    "description": "Stat group. Defaults to 'hitting'.",
                },
            },
            "required": ["player", "split"],
        },
    },
]
