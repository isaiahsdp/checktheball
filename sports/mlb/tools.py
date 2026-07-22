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

from core import db
from sports.mlb import client, fantasy, normalizer

SPORT = "mlb"
FULL_MLB_SEASON_GAMES = 162

# Live box scores and the schedule move during the day, so cache them briefly.
_BOXSCORE_MAX_AGE_SECONDS = 60
_SCHEDULE_MAX_AGE_SECONDS = 60

# Schedule statuses whose games have a readable box score.
_STARTED_STATUSES = {"In Progress", "Manager challenge", "Final", "Game Over", "Completed Early"}

# Today's-game counting stats that make sense to rank live performers by, plus
# "fantasy" for a computed fantasy-point ranking.
_GAME_HITTING_STATS = ("ab", "r", "h", "doubles", "triples", "hr", "rbi", "sb", "bb", "k")
_RANKABLE_STATS = _GAME_HITTING_STATS + ("fantasy",)

# Season stat keys (MLB naming) mapped to the keys the fantasy formula expects.
_SEASON_TO_FANTASY = {
    "h": "hits",
    "doubles": "doubles",
    "triples": "triples",
    "hr": "homeRuns",
    "rbi": "rbi",
    "r": "runs",
    "bb": "baseOnBalls",
    "sb": "stolenBases",
}

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


def _cached_schedule(date: str | None) -> list[dict[str, Any]]:
    key = f"raw_schedule:{date or 'today'}"
    return db.cached_fetch(
        "games", key, SPORT, lambda: client.get_schedule(date=date), _SCHEDULE_MAX_AGE_SECONDS
    )


def _cached_boxscore(game_id: int | str) -> dict[str, Any]:
    return db.cached_fetch(
        "player_stats_cache",
        f"boxscore:{game_id}",
        SPORT,
        lambda: client.get_game_boxscore(game_id),
        _BOXSCORE_MAX_AGE_SECONDS,
    )


def _rank_value(stats: dict[str, Any], stat: str) -> float:
    if stat == "fantasy":
        return fantasy.hitter_points(stats)
    value = stats.get(stat)
    return value if isinstance(value, (int, float)) else 0


def _name_matches(query: str, full_name: str) -> bool:
    q, full = query.lower().strip(), (full_name or "").lower()
    return bool(q) and (q in full or all(word in full for word in q.split()))


def _team_matches(query: str, full_name: str) -> bool:
    q, full = query.lower().strip(), full_name.lower()
    return q in full or any(word in full for word in q.split())


def get_todays_top_performers(
    stat: str = "h", limit: int = 5, date: str | None = None
) -> dict[str, Any]:
    """Rank the top batters across today's games by a single-game counting stat.

    ``stat="fantasy"`` ranks by computed DraftKings fantasy points.
    """
    if stat not in _RANKABLE_STATS:
        return {
            "error": f"Can't rank live games by '{stat}'.",
            "available_stats": list(_RANKABLE_STATS),
        }

    games = [g for g in _cached_schedule(date) if g.get("status") in _STARTED_STATUSES]
    if not games:
        return {"error": "No games have started yet for that date."}

    batters: list[dict[str, Any]] = []
    for game in games:
        box = _cached_boxscore(game["game_id"])
        batters.extend(normalizer.normalize_boxscore_batters(box))

    batters.sort(key=lambda b: _rank_value(b["stats"], stat), reverse=True)
    leaders = [
        {
            "player": b["player"],
            "team": b["team"],
            "value": _rank_value(b["stats"], stat),
            "fantasy_points": fantasy.hitter_points(b["stats"]),
            "line": b["stats"],
        }
        for b in batters[: max(1, limit)]
    ]
    result = {"date": date or "today", "stat": stat, "games_counted": len(games), "leaders": leaders}
    if stat == "fantasy":
        result["scoring"] = fantasy.SCORING_SYSTEM
    return result


def get_fantasy_points(
    player: str, season: int | None = None, date: str | None = None
) -> dict[str, Any]:
    """Compute a hitter's fantasy points for today's game, or a full season.

    With ``season`` set, scores the season totals; otherwise scores the player's
    line in today's game. Uses DraftKings classic hitter scoring.
    """
    if season is not None:
        resolved = _resolve_player(player)
        if resolved is None:
            return {"error": f"No player found matching '{player}'."}
        player_id, full_name = resolved
        raw = client.get_player_season_stats(player_id, season=season, group="hitting")
        line = normalizer.normalize_player_stat(raw, group="hitting")
        stats = {
            key: normalizer.to_number(line.stats.get(src, 0))
            for key, src in _SEASON_TO_FANTASY.items()
        }
        return {
            "player": line.player_name or full_name,
            "scope": "season",
            "season": line.season,
            "fantasy_points": fantasy.hitter_points(stats),
            "scoring": fantasy.SCORING_SYSTEM,
            "line": stats,
        }

    # No season given: score the player's line in today's game.
    for game in _cached_schedule(date):
        if game.get("status") not in _STARTED_STATUSES:
            continue
        for batter in normalizer.normalize_boxscore_batters(_cached_boxscore(game["game_id"])):
            if _name_matches(player, batter["player"]):
                return {
                    "player": batter["player"],
                    "team": batter["team"],
                    "scope": "game",
                    "date": date or "today",
                    "fantasy_points": fantasy.hitter_points(batter["stats"]),
                    "scoring": fantasy.SCORING_SYSTEM,
                    "line": batter["stats"],
                }
    return {"error": f"No player matching '{player}' found in today's games. Specify a season for season totals."}


def get_game_boxscore(team_a: str, team_b: str, date: str | None = None) -> dict[str, Any]:
    """Batting lines for a single game today, identified by its two teams."""
    game = next(
        (
            g
            for g in _cached_schedule(date)
            if (_team_matches(team_a, g.get("away_name", "")) and _team_matches(team_b, g.get("home_name", "")))
            or (_team_matches(team_a, g.get("home_name", "")) and _team_matches(team_b, g.get("away_name", "")))
        ),
        None,
    )
    if game is None:
        return {"error": f"No game found between '{team_a}' and '{team_b}' for that date."}
    if game.get("status") not in _STARTED_STATUSES:
        return {
            "error": f"The {game.get('away_name')} at {game.get('home_name')} game hasn't started yet.",
            "status": game.get("status"),
        }

    batters = normalizer.normalize_boxscore_batters(_cached_boxscore(game["game_id"]))
    for batter in batters:
        batter["fantasy_points"] = fantasy.hitter_points(batter["stats"])
    batters.sort(key=lambda b: b["fantasy_points"], reverse=True)
    return {
        "date": date or "today",
        "away_team": game.get("away_name"),
        "home_team": game.get("home_name"),
        "status": game.get("status"),
        "away_score": game.get("away_score"),
        "home_score": game.get("home_score"),
        "scoring": fantasy.SCORING_SYSTEM,
        "batters": batters,
    }


TOOL_FUNCTIONS = {
    "get_player_stat": get_player_stat,
    "compare_players": compare_players,
    "get_top_performers": get_top_performers,
    "compute_pace_projection": compute_pace_projection,
    "get_situational_split": get_situational_split,
    "get_todays_top_performers": get_todays_top_performers,
    "get_game_boxscore": get_game_boxscore,
    "get_fantasy_points": get_fantasy_points,
}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name. Used by the orchestrator."""
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
    {
        "name": "get_todays_top_performers",
        "description": (
            "Rank the best batters across today's live and finished games by a "
            "single-game counting stat, or by 'fantasy' for DraftKings fantasy "
            "points. Use for 'who is the best/top performer today', 'who has the "
            "most hits today', 'who has the most fantasy points today'. Reflects "
            "today's games only, not season totals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stat": {
                    "type": "string",
                    "enum": list(_RANKABLE_STATS),
                    "description": "Stat to rank by. 'fantasy' ranks by fantasy points. Defaults to 'h' (hits).",
                },
                "limit": {"type": "integer", "description": "How many players to return. Defaults to 5."},
                "date": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
            },
            "required": [],
        },
    },
    {
        "name": "get_fantasy_points",
        "description": (
            "Compute a hitter's DraftKings fantasy points, either for today's "
            "game (default) or for a full season (pass 'season'). Use for 'how "
            "many fantasy points does X have today' or 'X's fantasy points in "
            "2024'. Hitters only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name, e.g. 'Aaron Judge'."},
                "season": {"type": "integer", "description": "Four-digit year for season totals. Omit for today's game."},
                "date": {"type": "string", "description": "YYYY-MM-DD for the game. Defaults to today."},
            },
            "required": ["player"],
        },
    },
    {
        "name": "get_game_boxscore",
        "description": (
            "Get the per-batter lines for one specific game today, identified by "
            "its two teams. Use for 'who is performing in the Giants vs Royals "
            "game' or 'how is X doing in today's game'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string", "description": "One team's name, e.g. 'Giants' or 'San Francisco'."},
                "team_b": {"type": "string", "description": "The other team's name."},
                "date": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
            },
            "required": ["team_a", "team_b"],
        },
    },
]
