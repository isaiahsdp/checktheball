"""Verify the MLB data layer and tools end to end.

Kinds of checks:

* Offline: feed hand-built payloads through the normalizer to confirm the
  defensive guards (missing fields, empty values) behave. No network.
* Live: drive client -> normalizer -> schema against the real MLB Stats API,
  and exercise the SQLite cache. Anchored on historical facts (a 2024 game,
  Aaron Judge's 2024 season) that don't change, so results stay stable.
* Tools: call each Claude-facing tool and check its output against known 2024
  numbers, plus the error paths and schema shape.

Run from the repo root:

    python scripts/verify_mlb_data.py

Exit code is 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

# Make the project importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db, schema
from sports.mlb import client, fantasy, normalizer, tools

_passed = 0
_failed = 0


def check(name: str, condition: bool) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}")


def offline_checks() -> None:
    print("Offline checks (defensive normalization, no network)")

    # Unknown status -> OTHER; missing/empty optionals -> None; ids coerced.
    game = normalizer.normalize_game(
        {
            "game_id": 1,
            "home_id": 2,
            "home_name": "Home",
            "away_id": 3,
            "away_name": "Away",
            "status": "Rain Delay",  # not in any known set
            "home_score": "",        # empty -> None
            "away_score": "9",       # string -> int
        }
    )
    check("unknown status maps to OTHER", game.state == schema.OTHER)
    check("empty score becomes None", game.home_score is None)
    check("string score becomes int", game.away_score == 9)
    check("missing venue becomes None", game.venue is None)
    check("game_id coerced to str", game.game_id == "1")

    # A play with no matchup block should yield an empty players list, not crash.
    play = normalizer.normalize_play(
        {"result": {"event": "Single"}, "about": {"inning": 3, "halfInning": "bottom"}},
        game_id="g1",
        sequence=0,
    )
    check("missing matchup -> empty players", play.players == [])
    check("play period parsed", play.period == 3)

    # Empty stats list should not raise.
    ps = normalizer.normalize_player_stat(
        {"id": 99, "first_name": "Test", "last_name": "Player", "stats": []}
    )
    check("empty stats -> empty dict", ps.stats == {})
    check("player name assembled", ps.player_name == "Test Player")

    # Fantasy scoring: pure formula on a known line.
    line = {"h": 2, "doubles": 1, "triples": 0, "hr": 1, "rbi": 2, "r": 1, "bb": 1, "sb": 1}
    # 0 singles*3 + 1 double*5 + 1 hr*10 + 2 rbi*2 + 1 run*2 + 1 bb*2 + 1 sb*5 = 28
    check("fantasy points on known line == 28", fantasy.hitter_points(line) == 28)
    judge = {"h": 180, "doubles": 36, "triples": 1, "hr": 58, "rbi": 144, "r": 122, "bb": 133, "sb": 10}
    check("fantasy points on Judge 2024 line == 1871", fantasy.hitter_points(judge) == 1871)
    check("fantasy points on empty line == 0", fantasy.hitter_points({}) == 0)

    # Box-score normalizer: skip header rows, use full names, drop season rates.
    box = {
        "teamInfo": {
            "away": {"shortName": "Minnesota", "teamName": "Twins"},
            "home": {"shortName": "Cleveland", "teamName": "Guardians"},
        },
        "playerInfo": {"ID123": {"fullName": "Full Name"}},
        "awayBatters": [
            {"personId": 0, "name": "Twins Batters", "ab": "AB"},  # header row
            {"personId": 123, "name": "Name", "ab": "4", "h": "2", "doubles": "1",
             "triples": "0", "hr": "1", "rbi": "3", "r": "1", "bb": "1", "sb": "0",
             "k": "2", "avg": ".300", "ops": ".900"},
        ],
        "homeBatters": [],
    }
    batters = normalizer.normalize_boxscore_batters(box)
    check("box normalizer: header row skipped", len(batters) == 1)
    check("box normalizer: full name from playerInfo", batters[0]["player"] == "Full Name")
    check("box normalizer: team assembled", batters[0]["team"] == "Minnesota Twins")
    check("box normalizer: stats coerced to numbers", batters[0]["stats"]["hr"] == 1 and batters[0]["stats"]["h"] == 2)
    check("box normalizer: season rates excluded", "avg" not in batters[0]["stats"] and "ops" not in batters[0]["stats"])

    # Date-range normalizer: maps MLB stat names to our keys.
    date_range = {"stats": [{"splits": [
        {"player": {"fullName": "CJ Abrams"}, "team": {"name": "Washington Nationals"},
         "stat": {"atBats": 3, "runs": 2, "hits": 2, "doubles": 0, "triples": 0,
                  "homeRuns": 2, "rbi": 4, "stolenBases": 0, "baseOnBalls": 1, "strikeOuts": 0}},
    ]}]}
    hitters = normalizer.normalize_date_range_hitters(date_range)
    check("date-range normalizer: one hitter", len(hitters) == 1)
    check("date-range normalizer: name and team", hitters[0]["player"] == "CJ Abrams" and hitters[0]["team"] == "Washington Nationals")
    check("date-range normalizer: keys mapped (atBats->ab, homeRuns->hr)", hitters[0]["stats"]["ab"] == 3 and hitters[0]["stats"]["hr"] == 2 and hitters[0]["stats"]["bb"] == 1)
    check("date-range normalizer: empty payload -> []", normalizer.normalize_date_range_hitters({}) == [])


def live_checks() -> None:
    print("Live checks (real MLB Stats API)")

    # 1. Schedule -> Game (Cubs @ Orioles, 2024-07-09, final 9-2).
    games = [normalizer.normalize_game(g) for g in client.get_schedule(date="2024-07-09")]
    game = next((g for g in games if g.game_id == "747014"), None)
    check("known game found and normalized", game is not None)
    if game:
        check("state normalized to 'final'", game.state == "final")
        check("scores correct (away 9, home 2)", game.away_score == 9 and game.home_score == 2)

    # 2. Play-by-play -> [Play].
    plays = normalizer.normalize_playbyplay(client.get_game_playbyplay(747014), "747014")
    check("play-by-play produced plays", len(plays) > 0)
    check("first play in inning 1", plays[0].period == 1)
    check("scoring plays flagged", any(p.is_scoring for p in plays))

    # 3. Season stats -> PlayerStat (anchor: Judge hit 58 HR in 2024).
    pid = client.find_players("Aaron Judge")[0]["id"]
    season = normalizer.normalize_player_stat(
        client.get_player_season_stats(pid, season=2024, group="hitting"), group="hitting"
    )
    check("Judge 2024 home runs == 58", season.stats.get("homeRuns") == 58)
    check("season scope and year set", season.scope == "season" and season.season == 2024)

    # 4. Career stats -> PlayerStat.
    career = normalizer.normalize_player_stat(
        client.get_player_career_stats(pid, group="hitting"), group="hitting"
    )
    check("career scope, season is None", career.scope == "career" and career.season is None)


def cache_checks() -> None:
    print("Cache checks (SQLite freshness + dependency injection)")

    tmp_dir = tempfile.mkdtemp(prefix="checktheball_verify_")
    db_path = os.path.join(tmp_dir, "verify.sqlite")
    try:
        db.init_db(db_path)
        calls = {"n": 0}

        def fetch():
            calls["n"] += 1
            return {"homeRuns": 58}

        key = "mlb:test:2024:hitting"
        db.cached_fetch("player_stats_cache", key, "mlb", fetch, 3600, db_path)
        result = db.cached_fetch("player_stats_cache", key, "mlb", fetch, 3600, db_path)
        check("second call served from cache (fetch ran once)", calls["n"] == 1)
        check("cached payload intact", result["homeRuns"] == 58)

        # max_age=0 forces the row to count as stale -> refetch.
        db.cached_fetch("player_stats_cache", key, "mlb", fetch, 0, db_path)
        check("stale row triggers refetch", calls["n"] == 2)

        # Whitelist guards against unknown table names.
        try:
            db.read_cache("not_a_table", key, db_path=db_path)
            check("unknown table rejected", False)
        except ValueError:
            check("unknown table rejected", True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def tool_checks() -> None:
    print("Tool checks (Claude-facing tools vs known 2024 numbers)")

    stat = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024)
    check("get_player_stat: Judge 2024 HR == 58", stat.get("value") == 58)

    comp = tools.compare_players("Aaron Judge", "Shohei Ohtani", "homeRuns", season=2024)
    check("compare_players: Judge leads by 4", comp.get("leader") == "Aaron Judge" and comp.get("difference") == 4)

    top = tools.get_top_performers("homeRuns", season=2024, limit=5)
    check("get_top_performers: Judge is #1 with 58", top["leaders"][0]["value"] == 58)

    pace = tools.compute_pace_projection("Aaron Judge", "homeRuns", season=2024)
    check("compute_pace_projection: 58 in 158 G -> 59.5", pace.get("projected_value") == round(58 / 158 * 162, 1))

    split = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, split="home")
    check("get_player_stat split: Judge home HR == 31", split.get("value") == 31 and split.get("split") == "home")

    # Error paths return an {"error": ...} dict rather than raising.
    check("unknown player -> error", "error" in tools.get_player_stat("Zzz Notreal", "homeRuns", season=2024))
    bad_stat = tools.get_player_stat("Aaron Judge", "notAStat", season=2024)
    check("unavailable stat -> error + hint", "error" in bad_stat and "available_stats" in bad_stat)
    check("unknown split -> error", "error" in tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, split="in_the_rain"))
    check("call_tool dispatches unknown -> error", tools.call_tool("nope", {}).get("error", "").startswith("Unknown tool"))

    # Every advertised schema has a matching function and the required shape.
    names_match = {s["name"] for s in tools.TOOL_SCHEMAS} == set(tools.TOOL_FUNCTIONS)
    well_formed = all(
        {"name", "description", "input_schema"} <= set(s)
        and s["input_schema"]["type"] == "object"
        for s in tools.TOOL_SCHEMAS
    )
    check("tool schemas match functions and are well-formed", names_match and well_formed)


def filter_checks() -> None:
    print("Filter checks (date_range, opponent, split vs stable 2024 facts)")

    # date_range: all three presets + a custom window, MLB-aggregated.
    sa = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, date_range="since_allstar")
    check("date_range since_allstar: Judge 2024 HR == 24", sa.get("value") == 24 and sa.get("scope") == "since_allstar")
    custom = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, start_date="2024-06-01", end_date="2024-06-30")
    check("date_range custom Jun 2024: Judge HR == 11", custom.get("value") == 11)
    lastx = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, date_range="last_10_games")
    check("date_range last_10_games (2024): Judge HR == 5", lastx.get("value") == 5)
    check("date_range window meta embedded (games == 10)", lastx.get("games") == 10)

    # opponent: native head-to-head, with team-name resolution.
    vs = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, opponent="Dodgers")
    check("opponent vs Dodgers 2024: Judge HR == 3", vs.get("value") == 3)
    check("opponent resolves 'Dodgers' -> Los Angeles Dodgers", vs.get("opponent") == "Los Angeles Dodgers")

    # split: consolidated into get_player_stat.
    vl = tools.get_player_stat("Aaron Judge", "avg", season=2024, split="vs_left")
    check("split vs_left 2024: Judge avg == .311", vl.get("value") == ".311")

    # fantasy: season path.
    fp = tools.get_fantasy_points("Aaron Judge", season=2024)
    check("season fantasy 2024: Judge == 1871", fp.get("fantasy_points") == 1871 and fp.get("scope") == "season")

    # merged leaderboard, season scope.
    lb = tools.get_top_performers("homeRuns", scope="season", season=2024, limit=3)
    check(
        "leaderboard scope=season: Judge #1 with 58",
        lb["leaders"][0]["player"] == "Aaron Judge" and lb["leaders"][0]["value"] == 58 and lb["scope"] == "season",
    )

    # compare over a window keeps the window scope.
    cmp = tools.compare_players("Aaron Judge", "Shohei Ohtani", "homeRuns", season=2024, date_range="since_allstar")
    check("compare over date_range: scope propagates", cmp.get("scope") == "since_allstar")

    # Guards: filters are not combinable, and unknown values error cleanly.
    check("combine split+opponent -> error", "error" in tools.get_player_stat("Aaron Judge", "homeRuns", split="home", opponent="Dodgers"))
    check("unknown opponent -> error", "error" in tools.get_player_stat("Aaron Judge", "homeRuns", opponent="Nonexistent FC"))
    check("unknown date_range -> error", "error" in tools.get_player_stat("Aaron Judge", "homeRuns", date_range="last_5_years"))
    check("unknown scope -> error", "error" in tools.get_top_performers("homeRuns", scope="lifetime"))


def live_feature_checks() -> None:
    print("Live-feature checks (today's games; structural, tolerant of off-days)")

    today = tools.get_top_performers("h", scope="today", limit=3)
    if "error" in today:
        check("today leaderboard: clean no-games message", "no games" in today["error"].lower())
    else:
        check("today leaderboard: scope today with leaders", today.get("scope") == "today" and len(today["leaders"]) > 0)
        check("today leaderboard: leaders carry fantasy_points", all("fantasy_points" in leader for leader in today["leaders"]))

    fantasy = tools.get_top_performers("fantasy", scope="today", limit=3)
    if "error" not in fantasy:
        check("today fantasy: scoring system labeled", fantasy.get("scoring") == "DraftKings classic")

    started = [g for g in tools._cached_schedule(None) if g.get("status") in tools._STARTED_STATUSES]
    if started:
        game = started[0]
        box = tools.get_game_boxscore(game["away_name"].split()[-1], game["home_name"].split()[-1])
        check(
            "game boxscore: batters carry fantasy points",
            "batters" in box and (not box["batters"] or "fantasy_points" in box["batters"][0]),
        )
    check("game boxscore unknown matchup -> error", "error" in tools.get_game_boxscore("Lakers", "Celtics"))


def main() -> int:
    offline_checks()
    live_checks()
    cache_checks()
    tool_checks()
    filter_checks()
    live_feature_checks()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
