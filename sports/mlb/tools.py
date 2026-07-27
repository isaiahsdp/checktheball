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

from datetime import datetime, timedelta
from typing import Any

from core import db
from sports.mlb import client, fantasy, normalizer

SPORT = "mlb"
FULL_MLB_SEASON_GAMES = 162

# Live box scores and the schedule move during the day, so cache them briefly.
_BOXSCORE_MAX_AGE_SECONDS = 60
_SCHEDULE_MAX_AGE_SECONDS = 60
# Season splits move at most once a game, and a slash-line question asks for
# several stats from the same split, so cache the fetch to dedupe those.
_SEASON_SPLIT_MAX_AGE_SECONDS = 900

# Schedule statuses whose games have a readable box score.
_STARTED_STATUSES = {"In Progress", "Manager challenge", "Final", "Game Over", "Completed Early"}

# Today's-game counting stats that make sense to rank live performers by, plus
# "fantasy" for a computed fantasy-point ranking.
_GAME_HITTING_STATS = ("ab", "r", "h", "doubles", "triples", "hr", "rbi", "sb", "bb", "k")
_RANKABLE_STATS = _GAME_HITTING_STATS + ("fantasy",)

# Today's pitchers are ranked by strikeouts only for now. A composite game-score
# metric (K, ER, IP, ...) is deferred future work.
_PITCHING_RANKABLE_STATS = ("strikeOuts",)

# Fantasy formula keys mapped to their MLB stat-field names. The hitter and
# pitcher formulas expect different keys, so each group has its own mapping.
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
_SEASON_TO_FANTASY_PITCHING = {
    "ip": "inningsPitched",
    "k": "strikeOuts",
    "w": "wins",
    "er": "earnedRuns",
    "h": "hits",
    "bb": "baseOnBalls",
    "hbp": "hitBatsmen",
    "cg": "completeGames",
    "cgso": "shutouts",
    # no-hitter has no field in the season or game feeds, so "nh" stays 0.
}

# Rate stats are already normalized, so scaling one by games played or by the
# elapsed season projects nothing. Rejected by name rather than by value: a rate
# that lands on a whole number (an era of 0.00, an avg of .000) is
# indistinguishable from a counting stat once it is a float.
_RATE_STATS = frozenset({
    "avg", "obp", "slg", "ops", "babip", "atBatsPerHomeRun",
    "era", "whip", "strikeoutWalkRatio", "pitchesPerInning",
})
# Families of rate stat, for the names not spelled out above.
_RATE_STAT_MARKERS = ("percentage", "average", "per9", "pergame", "ratio")

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


def _cached_player_splits(
    player_id: str, code: str, season: int, group: str
) -> dict[str, Any]:
    key = f"splits:{player_id}:{group}:{season}:{code}"
    return db.cached_fetch(
        "player_stats_cache",
        key,
        SPORT,
        lambda: client.get_player_splits(player_id, [code], season=season, group=group),
        _SEASON_SPLIT_MAX_AGE_SECONDS,
    )


def _is_rate_stat(stat: str) -> bool:
    lowered = stat.lower()
    return stat in _RATE_STATS or any(m in lowered for m in _RATE_STAT_MARKERS)


def _resolve_team(name: str) -> tuple[str, str] | None:
    """Resolve a team name to (team_id, full_name), or None if not found."""
    teams = client.find_teams(name)
    if not teams:
        return None
    return str(teams[0]["id"]), teams[0]["name"]


def _second_half_range(season: int) -> tuple[str, str]:
    """(start, end) dates for a season's post-All-Star period, from the API."""
    info = client.get_season_info(season)["seasons"][0]
    today = datetime.now().strftime("%Y-%m-%d")
    return info["firstDate2ndHalf"], min(today, info["regularSeasonEndDate"])


def _date_range_stats(
    player_id: str,
    group: str,
    date_range: str | None,
    start_date: str | None,
    end_date: str | None,
    season: int,
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    """Fetch a player's stats over a preset or custom range, MLB-aggregated.

    Returns (stats, scope_label, window_meta), or None for an unknown preset.
    ``window_meta`` documents the queried window and, crucially, exposes its
    numbers (span size and year) so an answer that cites "last 10 games",
    "30 days", or the year of a date range ("...in 2024") stays grounded. The
    year is a real, queried fact; without it as a number the deterministic check
    can't verify a year the model correctly states.
    """
    if start_date and end_date:
        raw = client.get_player_stats_by_date_range(player_id, start_date, end_date, group, season=int(start_date[:4]))
        return normalizer.normalize_total_stat(raw), "date_range", {"start_date": start_date, "end_date": end_date, "year": int(start_date[:4])}
    if date_range == "last_10_games":
        raw = client.get_player_last_x_games(player_id, 10, group, season=season)
        return normalizer.normalize_total_stat(raw), "last_10_games", {"games": 10, "year": season}
    if date_range == "last_30_days":
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        raw = client.get_player_stats_by_date_range(player_id, start, end, group, season=datetime.now().year)
        return normalizer.normalize_total_stat(raw), "last_30_days", {"days": 30, "start_date": start, "end_date": end, "year": int(start[:4])}
    if date_range == "since_allstar":
        start, end = _second_half_range(season)
        raw = client.get_player_stats_by_date_range(player_id, start, end, group, season=season)
        return normalizer.normalize_total_stat(raw), "since_allstar", {"start_date": start, "end_date": end, "year": season}
    return None


def get_player_stat(
    player: str,
    stat: str,
    season: int | None = None,
    group: str = "hitting",
    split: str | None = None,
    date_range: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    opponent: str | None = None,
) -> dict[str, Any]:
    """One stat for one player, optionally filtered by a single dimension.

    Filters (use one at a time): ``split`` (home/away/vs_left/vs_right),
    ``date_range`` (last_10_games, last_30_days, since_allstar) or a custom
    ``start_date``/``end_date`` window, or ``opponent`` (a team name). With no
    filter, returns the season line, or career totals when ``season`` is omitted.
    """
    # Both ends or neither: one alone would fall through to the season line and
    # answer a different question than the one asked, with no error.
    if bool(start_date) != bool(end_date):
        return {"error": "A custom window needs both start_date and end_date."}

    resolved = _resolve_player(player)
    if resolved is None:
        return {"error": f"No player found matching '{player}'."}
    player_id, full_name = resolved

    # Each filter is a distinct native MLB stat type; they are not combinable.
    has_range = bool(date_range or start_date)  # end_date paired by the guard above
    if (split is not None) + has_range + (opponent is not None) > 1:
        return {"error": "Use only one of split, date_range/start_date, or opponent at a time."}

    team = None
    extra: dict[str, Any] = {}

    if opponent is not None:
        matched_team = _resolve_team(opponent)
        if matched_team is None:
            return {"error": f"No team found matching '{opponent}'."}
        team_id, team_name = matched_team
        stats = normalizer.normalize_total_stat(
            client.get_player_vs_team(player_id, team_id, season, group)
        )
        scope, extra = "vs_team", {"opponent": team_name}
        if not stats:
            where = f" in {season}" if season else ""
            return {"error": f"No data for {full_name} vs {team_name}{where}."}
    elif has_range:
        ranged = _date_range_stats(player_id, group, date_range, start_date, end_date, season or _current_season())
        if ranged is None:
            return {
                "error": f"Unknown date_range '{date_range}'.",
                "available_ranges": ["last_10_games", "last_30_days", "since_allstar"],
            }
        stats, scope, window = ranged
        extra = {"range": date_range or "custom", **window}
        if not stats:
            return {"error": f"No data for {full_name} over {date_range or 'that window'}."}
    elif split is not None:
        code = _SPLIT_CODES.get(split)
        if code is None:
            return {"error": f"Unknown split '{split}'.", "available_splits": sorted(_SPLIT_CODES)}
        if season is None:
            season = _current_season()
        matched = next(
            (s for s in normalizer.normalize_splits(_cached_player_splits(player_id, code, season, group), group=group) if s["code"] == code),
            None,
        )
        if matched is None:
            return {"error": f"No '{split}' split data for {full_name} in {season}."}
        stats, scope = matched["stats"], "season"
        extra = {"split": split, "split_description": matched["description"]}
    else:
        if season is not None:
            raw = client.get_player_season_stats(player_id, season=season, group=group)
        else:
            raw = client.get_player_career_stats(player_id, group=group)
        line = normalizer.normalize_player_stat(raw, group=group)
        stats, scope, season = line.stats, line.scope, line.season
        full_name, team = line.player_name or full_name, line.team

    if stat not in stats:
        return {
            "error": f"Stat '{stat}' not available for {full_name} ({group}).",
            "available_stats": sorted(stats.keys()),
        }

    return {
        "player": full_name,
        "player_id": player_id,
        "team": team,
        "stat": stat,
        "value": stats[stat],
        "scope": scope,
        "season": season,
        "group": group,
        **extra,
    }


def compare_players(
    player_a: str,
    player_b: str,
    stat: str,
    season: int | None = None,
    group: str = "hitting",
    date_range: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Compare two players on one stat and report who leads and by how much.

    ``date_range`` / ``start_date`` / ``end_date`` compare over a window (e.g.
    the last 30 days) instead of the full season.
    """
    kwargs = {"date_range": date_range, "start_date": start_date, "end_date": end_date}
    a = get_player_stat(player_a, stat, season, group, **kwargs)
    if "error" in a:
        return a
    b = get_player_stat(player_b, stat, season, group, **kwargs)
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


def _season_top_performers(
    stat: str, season: int | None, limit: int, group: str
) -> dict[str, Any]:
    """Leaderboard of the top players in a stat category for a season."""
    if season is None:
        season = _current_season()
    rows = client.get_league_leaders(stat, season=season, limit=limit, group=group)
    leaders = normalizer.normalize_leaders(rows)
    if not leaders:
        return {"error": f"No leaderboard data for '{stat}' in {season}."}
    _add_gap_from_leader(leaders)
    return {"scope": "season", "stat": stat, "season": season, "limit": limit, "leaders": leaders}


def _season_fraction_elapsed(season: int) -> float:
    """Fraction of the regular season elapsed as of today, clamped to [0, 1].

    A completed (past) season returns 1.0, so a projection over it equals the
    actual total. Used for pitchers, whose counting stats accrue across the
    calendar rather than in all 162 team games.
    """
    info = client.get_season_info(season)["seasons"][0]
    start = datetime.strptime(info["regularSeasonStartDate"], "%Y-%m-%d")
    end = datetime.strptime(info["regularSeasonEndDate"], "%Y-%m-%d")
    total_days = (end - start).days
    if total_days <= 0:
        return 1.0
    elapsed = (datetime.now() - start).days
    return max(0.0, min(1.0, elapsed / total_days))


def compute_pace_projection(
    player: str, stat: str, season: int | None = None, group: str = "hitting"
) -> dict[str, Any]:
    """Project a player's counting stat over a full season from current pace.

    Hitters project a per-game-played rate across a full 162-game season.
    Pitchers appear in far fewer than 162 games, so a per-game x162 projection
    would overcount several-fold; instead their totals scale by the fraction of
    the season elapsed (a completed season projects to its actual total). Not for
    rate stats (avg, era), and not for projecting a fixed number of extra games.
    """
    if season is None:
        season = _current_season()
    if _is_rate_stat(stat):
        return {
            "error": f"'{stat}' is a rate, not a counting stat, so it has no pace to project.",
            "hint": "Project a counting stat instead, e.g. homeRuns, hits, rbi, strikeOuts, wins.",
        }
    resolved = _resolve_player(player)
    if resolved is None:
        return {"error": f"No player found matching '{player}'."}
    player_id, full_name = resolved

    raw = client.get_player_season_stats(player_id, season=season, group=group)
    line = normalizer.normalize_player_stat(raw, group=group)

    if stat not in line.stats:
        return {
            "error": f"Can't project '{stat}' for {full_name} in {season} ({group}).",
            "available_stats": sorted(line.stats.keys()),
        }
    value = normalizer.to_number(line.stats[stat])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {"error": f"'{stat}' is not a countable stat to project."}

    if group == "pitching":
        fraction = _season_fraction_elapsed(season)
        if fraction <= 0:
            return {"error": f"Season {season} has not started yet to project from."}
        return {
            "player": line.player_name or full_name,
            "stat": stat,
            "season": season,
            "group": group,
            "current_value": value,
            "season_fraction_elapsed": round(fraction, 3),
            "projected_value": round(value / fraction, 1),
            "method": "current_value / fraction_of_season_elapsed",
        }

    if "gamesPlayed" not in line.stats:
        return {
            "error": f"Can't project '{stat}' for {full_name} in {season} ({group}).",
            "available_stats": sorted(line.stats.keys()),
        }
    games_played = normalizer.to_number(line.stats["gamesPlayed"])
    if not isinstance(games_played, (int, float)):
        return {"error": f"'{stat}' is not a countable stat to project."}
    if games_played <= 0:
        return {"error": f"{full_name} has no games played in {season} to project from."}

    return {
        "player": line.player_name or full_name,
        "stat": stat,
        "season": season,
        "group": group,
        "current_value": value,
        "games_played": games_played,
        "full_season_games": FULL_MLB_SEASON_GAMES,
        "projected_value": round(value / games_played * FULL_MLB_SEASON_GAMES, 1),
        "method": "current_value / games_played * 162",
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


def _todays_hitting_lines(date: str | None) -> list[dict[str, Any]]:
    """Every hitter's line for the day in one cached call, newest-first by fetch.

    Backs the "today" questions with a single league-wide request instead of one
    box score per game.
    """
    day = date or datetime.now().strftime("%Y-%m-%d")
    return db.cached_fetch(
        "player_stats_cache",
        f"day_hitting:{day}",
        SPORT,
        lambda: normalizer.normalize_date_range_hitters(client.get_stats_by_date(day)),
        _BOXSCORE_MAX_AGE_SECONDS,
    )


def _todays_pitching_lines(date: str | None) -> list[dict[str, Any]]:
    """Every pitcher's line for the day in one cached call. Mirrors the hitting path."""
    day = date or datetime.now().strftime("%Y-%m-%d")
    return db.cached_fetch(
        "player_stats_cache",
        f"day_pitching:{day}",
        SPORT,
        lambda: normalizer.normalize_date_range_pitchers(client.get_stats_by_date(day, group="pitching")),
        _BOXSCORE_MAX_AGE_SECONDS,
    )


def _rank_value(stats: dict[str, Any], stat: str) -> float:
    if stat == "fantasy":
        return fantasy.hitter_points(stats)
    value = stats.get(stat)
    return value if isinstance(value, (int, float)) else 0


def _add_gap_from_leader(leaders: list[dict[str, Any]]) -> None:
    """Annotate each leader entry with its gap behind the top value, in place.

    Precomputed so an answer citing "a 14-strikeout lead over second" grounds
    against a real tool value instead of the model's own subtraction. The gap is
    a magnitude, so a lower-is-better category (era, whip), where the leader
    holds the smallest value, reads the same way as a higher-is-better one
    instead of going negative. The leader gets 0. Skipped entirely if any value
    is non-numeric (same guard style as _rank_value), so a leaderboard with odd
    values is left untouched.
    """
    values = [entry.get("value") for entry in leaders]
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return
    top = values[0]
    for entry in leaders:
        entry["gap_from_leader"] = round(abs(top - entry["value"]), 2)


def _name_matches(query: str, full_name: str) -> bool:
    q, full = query.lower().strip(), (full_name or "").lower()
    return bool(q) and (q in full or all(word in full for word in q.split()))


def _team_matches(query: str, full_name: str) -> bool:
    q, full = query.lower().strip(), full_name.lower()
    return q in full or any(word in full for word in q.split())


def _todays_top_performers(stat: str, limit: int, date: str | None) -> dict[str, Any]:
    """Rank the top batters across today's games by a single-game counting stat.

    ``stat="fantasy"`` ranks by computed DraftKings fantasy points.
    """
    if stat not in _RANKABLE_STATS:
        return {
            "error": f"Can't rank today's games by '{stat}'.",
            "available_stats": list(_RANKABLE_STATS),
        }

    batters = _todays_hitting_lines(date)
    if not batters:
        return {"error": "No games have started yet for that date."}

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
    _add_gap_from_leader(leaders)
    result = {"scope": "today", "date": date or "today", "stat": stat, "hitters_counted": len(batters), "leaders": leaders}
    if stat == "fantasy":
        result["scoring"] = fantasy.SCORING_SYSTEM
    return result


def _todays_pitching_performers(stat: str, limit: int, date: str | None) -> dict[str, Any]:
    """Rank today's pitchers by a single counting stat (strikeouts only for now).

    This is a most-strikeouts ranking, not a composite "best pitching
    performance" score. Same leaderboard shape as the hitting path.
    """
    if stat not in _PITCHING_RANKABLE_STATS:
        return {
            "error": f"Can't rank today's pitchers by '{stat}'. Only strikeouts is supported.",
            "available_stats": list(_PITCHING_RANKABLE_STATS),
        }

    pitchers = _todays_pitching_lines(date)
    if not pitchers:
        return {"error": "No games have started yet for that date."}

    pitchers.sort(key=lambda p: _rank_value(p["stats"], stat), reverse=True)
    leaders = [
        {
            "player": p["player"],
            "team": p["team"],
            "value": _rank_value(p["stats"], stat),
            "line": p["stats"],
        }
        for p in pitchers[: max(1, limit)]
    ]
    _add_gap_from_leader(leaders)
    return {"scope": "today", "date": date or "today", "stat": stat, "group": "pitching", "pitchers_counted": len(pitchers), "leaders": leaders}


def get_top_performers(
    stat: str,
    scope: str = "season",
    season: int | None = None,
    limit: int = 5,
    group: str = "hitting",
    date: str | None = None,
) -> dict[str, Any]:
    """Leaderboard of top players in a stat, for a full season or today's games.

    ``scope="season"`` ranks a season's league leaders; ``scope="today"`` ranks
    batters across today's games (and supports ``stat="fantasy"``), or today's
    pitchers by strikeouts when ``group="pitching"``.
    """
    if scope == "today":
        if group == "pitching":
            return _todays_pitching_performers(stat, limit, date)
        return _todays_top_performers(stat, limit, date)
    if scope == "season":
        return _season_top_performers(stat, season, limit, group)
    return {"error": f"Unknown scope '{scope}'. Use 'season' or 'today'."}


def get_fantasy_points(
    player: str,
    season: int | None = None,
    date: str | None = None,
    opponent: str | None = None,
    group: str = "hitting",
) -> dict[str, Any]:
    """Compute a player's DraftKings fantasy points, for hitters or pitchers.

    ``group="pitching"`` scores the pitcher formula (innings, K, win, ER, etc.);
    otherwise the hitter formula. With ``opponent`` set, scores head-to-head
    totals vs a team (combine with ``season`` for one year, or alone for the
    career matchup). With ``season`` set, scores season totals. Otherwise scores
    the player's line in today's game.
    """
    if group not in ("hitting", "pitching"):
        return {"error": f"Fantasy points are available for 'hitting' or 'pitching', not '{group}'."}
    pitching = group == "pitching"
    # Pick the group's mapping, scoring formula, and label once; the three data
    # paths below differ only in their source, not in how they score.
    mapping = _SEASON_TO_FANTASY_PITCHING if pitching else _SEASON_TO_FANTASY
    score = fantasy.pitcher_points if pitching else fantasy.hitter_points
    label = fantasy.SCORING_SYSTEM_PITCHING if pitching else fantasy.SCORING_SYSTEM

    if opponent is not None:
        # Head-to-head totals vs a team, mirroring get_player_stat's opponent
        # branch. Combinable with season (that year) or alone (career matchup),
        # but not with the today's-game path.
        if date is not None:
            return {"error": "Can't combine opponent with a game date. Use opponent with an optional season for head-to-head totals."}
        resolved = _resolve_player(player)
        if resolved is None:
            return {"error": f"No player found matching '{player}'."}
        player_id, full_name = resolved
        matched_team = _resolve_team(opponent)
        if matched_team is None:
            return {"error": f"No team found matching '{opponent}'."}
        team_id, team_name = matched_team
        totals = normalizer.normalize_total_stat(
            client.get_player_vs_team(player_id, team_id, season, group=group)
        )
        if not totals:
            where = f" in {season}" if season else ""
            return {"error": f"No data for {full_name} vs {team_name}{where}."}
        stats = {key: normalizer.to_number(totals.get(src, 0)) for key, src in mapping.items()}
        return {
            "player": full_name,
            "scope": "vs_team",
            "season": season,
            "opponent": team_name,
            "group": group,
            "fantasy_points": score(stats),
            "scoring": label,
            "line": stats,
        }

    if season is not None:
        resolved = _resolve_player(player)
        if resolved is None:
            return {"error": f"No player found matching '{player}'."}
        player_id, full_name = resolved
        raw = client.get_player_season_stats(player_id, season=season, group=group)
        line = normalizer.normalize_player_stat(raw, group=group)
        stats = {key: normalizer.to_number(line.stats.get(src, 0)) for key, src in mapping.items()}
        return {
            "player": line.player_name or full_name,
            "scope": "season",
            "season": line.season,
            "group": group,
            "fantasy_points": score(stats),
            "scoring": label,
            "line": stats,
        }

    # No season given: score the player's line in today's game.
    lines = _todays_pitching_lines(date) if pitching else _todays_hitting_lines(date)
    for entry in lines:
        if _name_matches(player, entry["player"]):
            # Pitching lines carry MLB field names; map them to the formula's keys.
            # Hitting lines already use the hitter formula's keys directly.
            scored = {key: entry["stats"].get(src, 0) for key, src in mapping.items()} if pitching else entry["stats"]
            return {
                "player": entry["player"],
                "team": entry["team"],
                "scope": "game",
                "date": date or "today",
                "group": group,
                "fantasy_points": score(scored),
                "scoring": label,
                "line": scored,
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
            "Look up a single statistic for one player. Returns career totals if "
            "no season is given. Optionally narrow by ONE filter: 'split' (at "
            "home, on the road, vs left/right pitching), a recent window via "
            "'date_range' (last_10_games, last_30_days, since_allstar) or a "
            "custom 'start_date'/'end_date', or 'opponent' (a team name, e.g. "
            "'how does he hit against the Dodgers'). Only one filter at a time."
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
                "split": {
                    "type": "string",
                    "enum": ["home", "away", "vs_left", "vs_right"],
                    "description": "Situational split. Defaults to the overall line.",
                },
                "date_range": {
                    "type": "string",
                    "enum": ["last_10_games", "last_30_days", "since_allstar"],
                    "description": "Preset recent window. For a custom window use start_date/end_date instead.",
                },
                "start_date": {"type": "string", "description": "Custom window start, YYYY-MM-DD (needs end_date)."},
                "end_date": {"type": "string", "description": "Custom window end, YYYY-MM-DD (needs start_date)."},
                "opponent": {"type": "string", "description": "Opponent team name for a head-to-head split, e.g. 'Dodgers'."},
            },
            "required": ["player", "stat"],
        },
    },
    {
        "name": "compare_players",
        "description": (
            "Compare two players on the same stat and report who leads and by "
            "how much. Use for 'who has more X, A or B' questions. Pass "
            "'date_range' (last_10_games, last_30_days, since_allstar) or a "
            "custom 'start_date'/'end_date' to compare over a recent window."
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
                "date_range": {
                    "type": "string",
                    "enum": ["last_10_games", "last_30_days", "since_allstar"],
                    "description": "Preset recent window to compare over. Custom window: use start_date/end_date.",
                },
                "start_date": {"type": "string", "description": "Custom window start, YYYY-MM-DD (needs end_date)."},
                "end_date": {"type": "string", "description": "Custom window end, YYYY-MM-DD (needs start_date)."},
            },
            "required": ["player_a", "player_b", "stat"],
        },
    },
    {
        "name": "get_top_performers",
        "description": (
            "Leaderboard of top players in a stat. scope='season' ranks a "
            "season's league leaders ('who led MLB in X in 2024'); the stat uses "
            "MLB leader names, e.g. 'homeRuns', 'battingAverage', 'runsBattedIn', "
            "'stolenBases', 'earnedRunAverage'. scope='today' ranks batters "
            "across today's games ('who is the top performer today', 'most "
            "fantasy points today'); the stat uses single-game keys like 'h', "
            "'hr', 'rbi', or 'fantasy'. With scope='today' and group='pitching' it "
            "ranks today's pitchers by strikeouts (stat='strikeOuts'); this is a "
            "most-strikeouts ranking only, not a composite best-performance score. "
            "Each leader entry includes 'gap_from_leader', the precomputed size of "
            "its gap behind the top player, always a positive number (0 for the "
            "leader, and still positive for lower-is-better categories like "
            "earnedRunAverage), so cite that rather than subtracting values yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stat": {"type": "string", "description": "Leader category (season) or single-game key / 'fantasy' (today)."},
                "scope": {
                    "type": "string",
                    "enum": ["season", "today"],
                    "description": "'season' for a season leaderboard, 'today' for today's games. Defaults to 'season'.",
                },
                "season": {"type": "integer", "description": "Four-digit year for scope='season'. Defaults to current."},
                "limit": {"type": "integer", "description": "How many players to return. Defaults to 5."},
                "group": {
                    "type": "string",
                    "enum": ["hitting", "pitching", "fielding"],
                    "description": "Stat group. With scope='today', 'pitching' ranks today's pitchers by strikeouts. Defaults to 'hitting'.",
                },
                "date": {"type": "string", "description": "YYYY-MM-DD for scope='today'. Defaults to today."},
            },
            "required": ["stat"],
        },
    },
    {
        "name": "compute_pace_projection",
        "description": (
            "Project a counting stat over a FULL SEASON from a player's current "
            "pace. Hitters project across a full 162-game season (from games "
            "played so far); pitchers scale by the fraction of the season "
            "elapsed. Pass group='pitching' for pitching stats like 'strikeOuts' "
            "or 'wins'. Use for 'on pace for [a full season / a milestone]' "
            "questions. Do NOT use for rate stats (avg, era), and NOT for "
            "projecting a specific number of ADDITIONAL games (e.g. 'over his "
            "next 10 starts') - this tool only projects a full season."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name."},
                "stat": {"type": "string", "description": "A counting stat, e.g. 'homeRuns', 'hits', 'rbi', 'strikeOuts', 'wins'."},
                "season": {"type": "integer", "description": "Four-digit year. Defaults to the current season."},
                "group": {
                    "type": "string",
                    "enum": ["hitting", "pitching"],
                    "description": "Stat group. Use 'pitching' for pitching stats. Defaults to 'hitting'.",
                },
            },
            "required": ["player", "stat"],
        },
    },
    {
        "name": "get_fantasy_points",
        "description": (
            "Compute a player's DraftKings fantasy points for today's game "
            "(default), a full season (pass 'season'), or against one opponent "
            "team (pass 'opponent', optionally with 'season' for a single year). "
            "Set group='pitching' for pitcher scoring (innings, strikeouts, win, "
            "earned runs, etc.); defaults to hitter scoring. Use for 'how many "
            "fantasy points does X have today', 'X's fantasy points in 2024', "
            "'X's fantasy points against the Angels in 2024', or 'how many fantasy "
            "points did pitcher Y score in 2024'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name, e.g. 'Aaron Judge'."},
                "season": {"type": "integer", "description": "Four-digit year for season or vs-opponent totals. Omit for today's game (or the career matchup with opponent)."},
                "date": {"type": "string", "description": "YYYY-MM-DD for the game. Defaults to today. Not combinable with opponent."},
                "opponent": {"type": "string", "description": "Opponent team name for head-to-head fantasy totals, e.g. 'Angels'."},
                "group": {
                    "type": "string",
                    "enum": ["hitting", "pitching"],
                    "description": "Scoring group. 'pitching' uses the pitcher formula. Defaults to 'hitting'.",
                },
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
