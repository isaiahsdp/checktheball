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

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone

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

    # A today's pitching line must be scored with the PITCHER formula. Scoring it
    # with the hitter formula matches none of these keys and silently returns 0,
    # which is why _rank_value takes the group instead of guessing it.
    # 7.0 IP, 9 K, 3 ER, 3 H, 0 BB, 1 HBP, no win:
    #   7*2.25 + 9*2 - 3*2 - 3*0.6 - 1*0.6 = 15.75 + 18 - 6 - 1.8 - 0.6 = 25.35
    day_line = {"strikeOuts": 9, "inningsPitched": 7.0, "earnedRuns": 3, "baseOnBalls": 0,
                "hits": 3, "wins": 0, "hitBatsmen": 1, "completeGames": 0, "shutouts": 0}
    check("rank_value: pitching fantasy uses the pitcher formula == 25.35", tools._rank_value(day_line, "fantasy", "pitching") == 25.35)
    check("rank_value: the same line under the hitter formula scores 0 (the trap)", tools._rank_value(day_line, "fantasy") == 0)
    check("rank_value: a plain counting stat is unaffected by group", tools._rank_value(day_line, "strikeOuts", "pitching") == 9)
    # The win bonus is available here but not from a box score, so it must apply.
    check("rank_value: pitching fantasy counts the win (+4)", tools._rank_value({**day_line, "wins": 1}, "fantasy", "pitching") == 29.35)

    # Pitcher fantasy: hand-computed DraftKings classic lines.
    # 6.2 IP (= 6 2/3): 6.667*2.25 + 7*2 - 3*2 - 6*0.6 - 2*0.6 - 1*0.6
    #                 = 15.0 + 14 - 6 - 3.6 - 1.2 - 0.6 = 17.6
    pline = {"ip": "6.2", "k": 7, "w": 0, "er": 3, "h": 6, "bb": 2, "hbp": 1}
    check("pitcher fantasy on 6.2 IP line == 17.6", fantasy.pitcher_points(pline) == 17.6)
    # No-hit complete-game shutout with a win: 9*2.25 + 12*2 + 4 - 1*0.6
    #                 + 2.5 (CG) + 2.5 (CG shutout) + 5 (no-hitter) = 57.65
    gem = {"ip": "9.0", "k": 12, "w": 1, "er": 0, "h": 0, "bb": 1, "cg": 1, "cgso": 1, "nh": 1}
    check("pitcher fantasy on no-hit CG shutout == 57.65", fantasy.pitcher_points(gem) == 57.65)
    check("pitcher fantasy on empty line == 0", fantasy.pitcher_points({}) == 0)

    # Box-score normalizer: skip header rows, use full names, drop season rates.
    box = {
        "teamInfo": {
            "away": {"shortName": "Minnesota", "teamName": "Twins"},
            "home": {"shortName": "Cleveland", "teamName": "Guardians"},
        },
        "playerInfo": {"ID123": {"fullName": "Full Name"}},
        "awayBatters": [
            {"personId": 0, "name": "Twins Batters", "ab": "AB"},  # header row
            {"personId": 123, "name": "Name", "position": "CF", "battingOrder": "400",
             "substitution": False, "ab": "4", "h": "2", "doubles": "1",
             "triples": "0", "hr": "1", "rbi": "3", "r": "1", "bb": "1", "sb": "0",
             "k": "2", "avg": ".300", "ops": ".900"},
            {"personId": 124, "name": "Sub", "position": "CF", "battingOrder": "401",
             "substitution": True, "ab": "1", "h": "0", "doubles": "0", "triples": "0",
             "hr": "0", "rbi": "0", "r": "0", "bb": "0", "sb": "0", "k": "1"},
        ],
        "homeBatters": [],
    }
    batters = normalizer.normalize_boxscore_batters(box)
    check("box normalizer: header row skipped", len(batters) == 2)
    check("box normalizer: full name from playerInfo", batters[0]["player"] == "Full Name")
    # No team label here on purpose: the box score's own team fields are display
    # abbreviations that double up, so the caller labels rows from the side.
    check("box normalizer: carries side, not a concatenated team label",
          batters[0].get("side") == "away" and "team" not in batters[0])
    check("box normalizer: batting order keeps slot and substitution suffix",
          batters[0].get("batting_order") == 400 and batters[1].get("batting_order") == 401)
    check("box normalizer: substitution flag passed through",
          batters[0].get("substitution") is False and batters[1].get("substitution") is True)
    check("box normalizer: blank batting order -> None",
          normalizer.normalize_boxscore_batters({"awayBatters": [{"personId": 5, "battingOrder": ""}], "homeBatters": []})[0]["batting_order"] is None)
    check("box normalizer: stats coerced to numbers", batters[0]["stats"]["hr"] == 1 and batters[0]["stats"]["h"] == 2)
    check("box normalizer: season rates excluded", "avg" not in batters[0]["stats"] and "ops" not in batters[0]["stats"])

    # Box-score pitching normalizer: header row skipped, full name preferred,
    # season era excluded (same reason avg/ops are), decision parsed from note.
    pitch_box = {
        "teamInfo": {
            "away": {"shortName": "Chi", "teamName": "Cubs"},
            "home": {"shortName": "Baltimore", "teamName": "Orioles"},
        },
        "playerInfo": {"ID77": {"fullName": "Full Pitcher"}},
        "awayPitchers": [
            {"personId": 0, "name": "Cubs Pitchers", "ip": "IP", "era": "ERA"},  # header row
            {"personId": 77, "name": "Pitcher", "note": "(W, 5-2)", "ip": "6.0", "h": "4",
             "r": "2", "er": "2", "bb": "1", "k": "7", "hr": "1", "p": "88", "s": "64", "era": "3.21"},
            {"personId": 78, "name": "Reliever", "note": "", "ip": "0.2", "h": "1",
             "r": "0", "er": "0", "bb": "0", "k": "1", "hr": "0", "p": "12", "s": "8", "era": "2.00"},
        ],
        "homePitchers": [],
    }
    arms = normalizer.normalize_boxscore_pitchers(pitch_box)
    check("box pitching normalizer: header row skipped", len(arms) == 2)
    check("box pitching normalizer: full name from playerInfo", arms[0]["player"] == "Full Pitcher")
    check("box pitching normalizer: carries side, not a concatenated team label",
          arms[0].get("side") == "away" and "team" not in arms[0])
    check("box pitching normalizer: season era excluded", "era" not in arms[0]["stats"])
    check("box pitching normalizer: p/s renamed, values coerced", arms[0]["stats"].get("pitches") == 88 and arms[0]["stats"].get("strikes") == 64)
    check("box pitching normalizer: decision parsed from note", arms[0]["decision"] == "W" and arms[1]["decision"] is None)
    check("box pitching normalizer: MLB innings notation preserved", arms[1]["stats"].get("ip") == 0.2)
    check("box pitching normalizer: empty payload -> []", normalizer.normalize_boxscore_pitchers({}) == [])

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

    # Date-range pitching normalizer: maps MLB pitching stat names, coerces values
    # (inningsPitched "7.0" -> 7.0 via to_number).
    date_range_pitch = {"stats": [{"splits": [
        {"player": {"fullName": "Tarik Skubal"}, "team": {"name": "Detroit Tigers"},
         "stat": {"strikeOuts": 9, "inningsPitched": "7.0", "earnedRuns": 1,
                  "baseOnBalls": 2, "hits": 4}},
    ]}]}
    pitchers = normalizer.normalize_date_range_pitchers(date_range_pitch)
    check("date-range pitching normalizer: one pitcher", len(pitchers) == 1)
    check("date-range pitching normalizer: name and team", pitchers[0]["player"] == "Tarik Skubal" and pitchers[0]["team"] == "Detroit Tigers")
    check("date-range pitching normalizer: fields mapped and coerced", pitchers[0]["stats"]["strikeOuts"] == 9 and pitchers[0]["stats"]["earnedRuns"] == 1 and pitchers[0]["stats"]["inningsPitched"] == 7.0)
    check("date-range pitching normalizer: empty payload -> []", normalizer.normalize_date_range_pitchers({}) == [])

    # team_from_splits reads the player's own team off any split-bearing payload,
    # so a filtered lookup reports a team the way a season line already does. A
    # live 0.00 came from this being absent: the answer said "Grisham (Yankees)"
    # and the team, true but unretrieved, could not ground.
    split_payload = {"people": [{"stats": [{"splits": [
        {"team": {"id": 147, "name": "New York Yankees"}, "stat": {"strikeOuts": 41}},
    ]}]}]}
    check("team_from_splits: reads the team off a split entry",
          normalizer.team_from_splits(split_payload) == "New York Yankees")
    # On vsTeam the entry carries both; the opponent is a different team and
    # must not be mistaken for the player's own.
    vs_payload = {"people": [{"stats": [{"splits": [
        {"team": {"name": "New York Yankees"}, "opponent": {"name": "Los Angeles Dodgers"},
         "stat": {"homeRuns": 2}},
    ]}]}]}
    check("team_from_splits: returns the player's team, not the opponent",
          normalizer.team_from_splits(vs_payload) == "New York Yankees")
    # The grand-total split names no team, so the first entry that does wins.
    totals_first = {"people": [{"stats": [{"splits": [
        {"sport": {"id": 0}, "stat": {"homeRuns": 12}},
        {"team": {"name": "Detroit Tigers"}, "stat": {"homeRuns": 12}},
    ]}]}]}
    check("team_from_splits: skips the teamless grand total",
          normalizer.team_from_splits(totals_first) == "Detroit Tigers")
    check("team_from_splits: empty payload -> None", normalizer.team_from_splits({}) is None)
    check("team_from_splits: splits without a team -> None",
          normalizer.team_from_splits({"people": [{"stats": [{"splits": [{"stat": {}}]}]}]}) is None)

    # normalize_total_stat picks the grand total, never sums the duplicate splits.
    # With no "All" split (sport id 0), it falls back to the most-games split.
    no_all = {"people": [{"stats": [{"splits": [
        {"sport": {"id": 1}, "stat": {"gamesPlayed": 10, "homeRuns": 3}},
        {"sport": {"id": 11}, "stat": {"gamesPlayed": 40, "homeRuns": 9}},
    ]}]}]}
    check("total_stat: no sport-0 split -> falls back to most-games split", normalizer.normalize_total_stat(no_all).get("homeRuns") == 9)
    # When the "All" grand total (sport id 0) exists, it wins even with fewer games.
    with_all = {"people": [{"stats": [{"splits": [
        {"sport": {"id": 1}, "stat": {"gamesPlayed": 40, "homeRuns": 9}},
        {"sport": {"id": 0}, "stat": {"gamesPlayed": 25, "homeRuns": 11}},
    ]}]}]}
    check("total_stat: sport-0 grand total preferred over more-games split", normalizer.normalize_total_stat(with_all).get("homeRuns") == 11)
    check("total_stat: empty payload -> {}", normalizer.normalize_total_stat({}) == {})

    # gap_from_leader is a magnitude. In a lower-is-better category the leader
    # holds the smallest value, so a signed subtraction would hand the model a
    # negative "gap" to cite.
    era_board = [{"value": 1.90}, {"value": 2.40}, {"value": 2.75}]
    tools._add_gap_from_leader(era_board)
    check("gap_from_leader: lower-is-better gaps stay positive", [e["gap_from_leader"] for e in era_board] == [0.0, 0.5, 0.85])
    hr_board = [{"value": 58}, {"value": 54}, {"value": 48}]
    tools._add_gap_from_leader(hr_board)
    check("gap_from_leader: higher-is-better gaps unchanged", [e["gap_from_leader"] for e in hr_board] == [0, 4, 10])
    odd_board = [{"value": 5}, {"value": "-.--"}]
    tools._add_gap_from_leader(odd_board)
    check("gap_from_leader: non-numeric leaderboard left untouched", all("gap_from_leader" not in e for e in odd_board))

    # One source of truth for the box-score batting keys: the normalizer produces
    # them, tools ranks by them. Identity rather than equality, so reintroducing a
    # second literal is caught even while its values still happen to match.
    check(
        "batting keys: tools reuses the normalizer's tuple, not a second copy",
        tools._GAME_HITTING_STATS is getattr(normalizer, "BOX_HITTING_STATS", None),
    )

    # The rate-stat classifier must catch both the named stats and the families
    # (percentage / average / per9 / ...) without swallowing counting stats.
    rates = ("avg", "obp", "ops", "era", "whip", "battingAverage", "stolenBasePercentage", "strikeoutsPer9Inn")
    counting = ("homeRuns", "hits", "rbi", "strikeOuts", "wins", "stolenBases", "gamesPlayed", "inningsPitched")
    check("rate-stat classifier: named rates and rate families detected", all(tools._is_rate_stat(s) for s in rates))
    check("rate-stat classifier: counting stats not caught", not any(tools._is_rate_stat(s) for s in counting))


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

    # 3b. Box-score pitching lines, anchored on the same completed 2024 game.
    # Taillon went 6 IP, 4 H, 2 ER, 1 BB, 7 K and took the win, so his line and
    # its DraftKings score are fixed: 13.5 (IP) + 14 (K) + 4 (W) - 4 (ER)
    # - 2.4 (H) - 0.6 (BB) = 24.5. Home runs carry no separate pitcher penalty.
    arms_2024 = normalizer.normalize_boxscore_pitchers(client.get_game_boxscore(747014))
    winner = next((p for p in arms_2024 if p["decision"] == "W"), None)
    check("2024 box score: winning pitcher identified from the note", winner is not None and winner["player"] == "Jameson Taillon")
    if winner:
        check("2024 box score: pitching line matches", winner["stats"]["ip"] == 6.0 and winner["stats"]["k"] == 7 and winner["stats"]["er"] == 2)
        check("2024 box score: win bonus applied to the pitcher's score", tools._pitcher_line_points(winner) == 24.5)

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

        # log_query writes the query log row verbatim.
        db.log_query("who won?", "The Yankees.", [{"name": "get_player_stat"}], 0.9, db_path)
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute("SELECT * FROM queries ORDER BY id DESC LIMIT 1").fetchone()
        finally:
            con.close()
        check(
            "log_query: row written with question/answer/score",
            row["question"] == "who won?" and row["answer"] == "The Yankees." and row["grounding_score"] == 0.9,
        )
        check("log_query: tool_calls stored as JSON", json.loads(row["tool_calls"]) == [{"name": "get_player_stat"}])

        # write_cache upserts: the same key is replaced in place, not duplicated.
        db.write_cache("games", "upsert_key", "mlb", {"v": 1}, db_path)
        db.write_cache("games", "upsert_key", "mlb", {"v": 2}, db_path)
        check("write_cache: same key replaced with newest payload", db.read_cache("games", "upsert_key", db_path=db_path) == {"v": 2})
        con = sqlite3.connect(db_path)
        try:
            count = con.execute("SELECT COUNT(*) FROM games WHERE key = 'upsert_key'").fetchone()[0]
        finally:
            con.close()
        check("write_cache: upsert leaves a single row (no duplicate)", count == 1)

        # The sweep drops aged cache rows and leaves the query log alone. Rows
        # are backdated directly since write_cache always stamps "now".
        db.write_cache("games", "aged_row", "mlb", {"v": 1}, db_path)
        db.write_cache("games", "recent_row", "mlb", {"v": 2}, db_path)
        con = sqlite3.connect(db_path)
        try:
            aged = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            con.execute("UPDATE games SET updated_at = ? WHERE key = 'aged_row'", (aged,))
            con.commit()
            queries_before = con.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        finally:
            con.close()

        removed = db.sweep_cache(7, db_path)
        check("sweep: aged row deleted", db.read_cache("games", "aged_row", db_path=db_path) is None)
        check("sweep: recent row kept", db.read_cache("games", "recent_row", db_path=db_path) == {"v": 2})
        check("sweep: reports what it removed", removed >= 1)
        con = sqlite3.connect(db_path)
        try:
            queries_after = con.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        finally:
            con.close()
        check("sweep: query log untouched", queries_after == queries_before)

        # _cached_schedule must key on the real date. Under a literal "today"
        # key, a row written just before midnight is still fresh after the
        # rollover and resolves a game against the previous day's slate.
        # cached_fetch is stubbed so this stays offline and off the real DB.
        seen = {}
        original_cached_fetch = db.cached_fetch

        def spy(table, key, sport, fetch_fn, max_age, *args, **kwargs):
            seen["key"] = key
            return []

        db.cached_fetch = spy
        try:
            tools._cached_schedule(None)
            implicit = seen["key"]
            tools._cached_schedule("2024-06-01")
            explicit = seen["key"]
        finally:
            db.cached_fetch = original_cached_fetch
        today = datetime.now().strftime("%Y-%m-%d")
        check("schedule key: implicit today carries the real date", implicit == f"raw_schedule:{today}")
        check("schedule key: explicit date preserved", explicit == "raw_schedule:2024-06-01")
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
    # gap_from_leader is precomputed so the model cites it instead of subtracting.
    check("get_top_performers: leader gap_from_leader == 0", top["leaders"][0]["gap_from_leader"] == 0)
    check("get_top_performers: #2 gap_from_leader == 4 (58 - 54)", top["leaders"][1]["value"] == 54 and top["leaders"][1]["gap_from_leader"] == 4)

    pace = tools.compute_pace_projection("Aaron Judge", "homeRuns", season=2024)
    check("compute_pace_projection: 58 in 158 G -> 59.5", pace.get("projected_value") == round(58 / 158 * 162, 1))
    # Pitching pace uses season-fraction (not games x 162, which would overcount
    # ~5x). A completed season is fully elapsed, so it projects to the real total.
    ppace = tools.compute_pace_projection("Tarik Skubal", "strikeOuts", season=2024, group="pitching")
    check(
        "compute_pace_projection pitching: Skubal 2024 K projects to 228 (completed season)",
        ppace.get("projected_value") == 228 and ppace.get("group") == "pitching",
    )

    # Rate stats are documented as unprojectable in both the docstring and the
    # tool schema; the tool must reject them rather than return a scaled,
    # meaningless number (avg .322 would otherwise "project" to 0.3).
    # The error must be the rate-stat rejection specifically: a stat that simply
    # isn't in the season line also errors, which would make this pass for the
    # wrong reason.
    for rate in ("avg", "ops", "battingAverage"):
        rejected = tools.compute_pace_projection("Aaron Judge", rate, season=2024)
        check(f"compute_pace_projection rejects rate stat '{rate}'", "rate" in rejected.get("error", "") and "projected_value" not in rejected)
    era_rejected = tools.compute_pace_projection("Tarik Skubal", "era", season=2024, group="pitching")
    check("compute_pace_projection rejects rate stat 'era' (pitching)", "rate" in era_rejected.get("error", "") and "projected_value" not in era_rejected)

    # The completed-season anchor above only exercises fraction == 1.0, where
    # value / fraction is the identity. Swap in a partial season to cover the
    # scaling arithmetic and the not-started guard deterministically, year-round.
    real_fraction = tools._season_fraction_elapsed
    try:
        tools._season_fraction_elapsed = lambda season: 0.5
        half = tools.compute_pace_projection("Tarik Skubal", "strikeOuts", season=2024, group="pitching")
        check(
            "compute_pace_projection pitching: 228 K at half a season projects to 456",
            half.get("projected_value") == 456.0 and half.get("season_fraction_elapsed") == 0.5,
        )
        tools._season_fraction_elapsed = lambda season: 0.0
        unstarted = tools.compute_pace_projection("Tarik Skubal", "strikeOuts", season=2024, group="pitching")
        check("compute_pace_projection pitching: an unstarted season is rejected", "error" in unstarted)
    finally:
        tools._season_fraction_elapsed = real_fraction
    check("season fraction of a completed season is 1.0", tools._season_fraction_elapsed(2024) == 1.0)

    # group is validated the same way in every tool that takes one: an unknown
    # value errors immediately instead of silently defaulting to hitting (today's
    # leaderboard), raising out of the wrapper (season leaderboard), or failing
    # downstream with a stat list for the wrong group (pace projection).
    bad_today = tools.get_top_performers("h", scope="today", group="fielding", limit=2)
    check("group guard: today's leaderboard rejects an unsupported group", "error" in bad_today and "leaders" not in bad_today)
    bad_typo = tools.get_top_performers("h", scope="today", group="hittting", limit=2)
    check("group guard: today's leaderboard rejects a misspelled group", "error" in bad_typo and "leaders" not in bad_typo)
    bad_season = tools.get_top_performers("homeRuns", scope="season", season=2024, group="nonsense", limit=2)
    check("group guard: season leaderboard rejects an unknown group instead of raising", "error" in bad_season and "leaders" not in bad_season)
    bad_pace = tools.compute_pace_projection("Aaron Judge", "homeRuns", season=2024, group="fielding")
    check("group guard: pace projection names the bad group, not a stat list", "not 'fielding'" in bad_pace.get("error", "") and "available_stats" not in bad_pace)
    bad_fantasy = tools.get_fantasy_points("Aaron Judge", season=2024, group="fielding")
    check("group guard: fantasy points rejects an unsupported group", "not 'fielding'" in bad_fantasy.get("error", "") and "fantasy_points" not in bad_fantasy)
    # A category with no leaders must come back as a clean error, not raise out of
    # the statsapi wrapper, which indexes leagueLeaders[0] unguarded.
    empty_lb = tools.get_top_performers("notAStat", scope="season", season=2024, limit=3)
    check("leaderboard: an empty stat category errors cleanly instead of raising", "error" in empty_lb and "leaders" not in empty_lb)

    # fielding is a real season leaderboard group, so the whitelist must keep it.
    fielding_lb = tools.get_top_performers("assists", scope="season", season=2024, limit=3, group="fielding")
    check("group guard: fielding season leaderboard still works", fielding_lb.get("leaders", [{}])[0].get("player") == "Ezequiel Tovar")

    split = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, split="home")
    check("get_player_stat split: Judge home HR == 31", split.get("value") == 31 and split.get("split") == "home")

    # Pitching anchor (every other anchor is a hitter): Skubal's 2024 line.
    pit_k = tools.get_player_stat("Tarik Skubal", "strikeOuts", season=2024, group="pitching")
    check("get_player_stat pitching: Skubal 2024 K == 228", normalizer.to_number(pit_k.get("value")) == 228 and pit_k.get("group") == "pitching")
    pit_w = tools.get_player_stat("Tarik Skubal", "wins", season=2024, group="pitching")
    check("get_player_stat pitching: Skubal 2024 wins == 18", normalizer.to_number(pit_w.get("value")) == 18)

    # Pitcher fantasy, season path: Skubal's completed-2024 line scores 746.4 DK
    # points (192 IP, 228 K, 18 W, 51 ER, 142 H, 35 BB, 9 HBP), an immutable anchor.
    skubal_fp = tools.get_fantasy_points("Tarik Skubal", season=2024, group="pitching")
    check(
        "get_fantasy_points pitching: Skubal 2024 == 746.4",
        skubal_fp.get("fantasy_points") == 746.4 and skubal_fp.get("scope") == "season" and skubal_fp.get("group") == "pitching",
    )

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
    check("date_range since_allstar: real tool exposes year == 2024", sa.get("year") == 2024)
    custom = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, start_date="2024-06-01", end_date="2024-06-30")
    check("date_range custom Jun 2024: Judge HR == 11", custom.get("value") == 11)
    # The real tool (not a fixture) must expose the queried year so a claim that
    # states it stays grounded. Guards the tools.py -> grounding.py contract.
    check("date_range custom: real tool exposes year == 2024", custom.get("year") == 2024)
    lastx = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, date_range="last_10_games")
    check("date_range last_10_games (2024): Judge HR == 5", lastx.get("value") == 5)
    check("date_range window meta embedded (games == 10)", lastx.get("games") == 10)
    check("date_range last_10_games: real tool exposes year == 2024", lastx.get("year") == 2024)
    # last_30_days is the one preset that can't be anchored: its window is always
    # the real last 30 days, and a player who hasn't played in it returns a clean
    # error. Assert the metadata contract instead of a number, and check the year
    # against the window's own start date so the check can't flake in January.
    l30 = tools.get_player_stat("Shohei Ohtani", "homeRuns", date_range="last_30_days")
    if "error" in l30:
        check("date_range last_30_days: clean no-data message", "no data" in l30["error"].lower())
    else:
        check(
            "date_range last_30_days: window meta embedded (days == 30, year matches start_date)",
            l30.get("scope") == "last_30_days" and l30.get("days") == 30 and l30.get("year") == int(l30["start_date"][:4]),
        )

    # Every filtered path reports the player's team, the same as the season line.
    # Anchored on 2024, when Judge was a Yankee, so the fact cannot drift. Runs
    # against the real tool because the point is the tools.py -> grounding.py
    # contract: a team the answer states has to be retrieved data, not recall.
    season_line = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024)
    check("team: season line reports the team (unchanged)", season_line.get("team") == "New York Yankees")
    check("team: date_range path reports the team", sa.get("team") == "New York Yankees")
    check("team: custom window reports the team", custom.get("team") == "New York Yankees")
    vs_dodgers = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, opponent="Dodgers")
    check("team: opponent path reports Judge's team, not the opponent",
          vs_dodgers.get("team") == "New York Yankees" and vs_dodgers.get("opponent") == "Los Angeles Dodgers")
    vs_left = tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, split="vs_left")
    check("team: split path reports the team", vs_left.get("team") == "New York Yankees")

    # A custom window needs both ends. Supplying one alone used to skip the range
    # branch entirely and return the full-season line, answering a different
    # question with no error and a grounding score of 1.0.
    check("custom window: start_date without end_date -> error", "error" in tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, start_date="2024-06-01"))
    check("custom window: end_date without start_date -> error", "error" in tools.get_player_stat("Aaron Judge", "homeRuns", season=2024, end_date="2024-06-30"))
    check(
        "custom window: half a window rejected through compare_players too",
        "error" in tools.compare_players("Aaron Judge", "Shohei Ohtani", "homeRuns", season=2024, start_date="2024-06-01"),
    )

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

    # fantasy: opponent path, anchored to the same Judge-vs-Dodgers-2024 line as
    # the opponent hitting check above (hr == 3). That line scores 70 DK points.
    fvs = tools.get_fantasy_points("Aaron Judge", season=2024, opponent="Dodgers")
    check(
        "opponent fantasy 2024: Judge vs Dodgers == 70 (line hr == 3)",
        fvs.get("fantasy_points") == 70 and fvs.get("line", {}).get("hr") == 3
        and fvs.get("scope") == "vs_team" and fvs.get("opponent") == "Los Angeles Dodgers",
    )
    check("opponent fantasy: opponent + date rejected", "error" in tools.get_fantasy_points("Aaron Judge", opponent="Dodgers", date="2024-06-01"))
    check("opponent fantasy: unknown team -> error", "error" in tools.get_fantasy_points("Aaron Judge", opponent="Nonexistent FC", season=2024))

    # merged leaderboard, season scope.
    lb = tools.get_top_performers("homeRuns", scope="season", season=2024, limit=3)
    check(
        "leaderboard scope=season: Judge #1 with 58",
        lb["leaders"][0]["player"] == "Aaron Judge" and lb["leaders"][0]["value"] == 58 and lb["scope"] == "season",
    )

    # Lower-is-better leaderboard against the real tool: the 2024 ERA leaders are
    # Sale 2.38, Skubal 2.39, Wheeler 2.57, so the gaps must read as positive
    # magnitudes rather than the negatives a signed subtraction would produce.
    era_lb = tools.get_top_performers("earnedRunAverage", scope="season", season=2024, limit=3, group="pitching")
    check(
        "leaderboard ERA 2024: gap_from_leader positive for a lower-is-better category",
        [leader["gap_from_leader"] for leader in era_lb["leaders"]] == [0.0, 0.01, 0.19],
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
        # The today paths must actually emit gap_from_leader, not just be capable
        # of being grounded against it in a fixture.
        check(
            "today leaderboard: leaders carry gap_from_leader, leader at 0",
            all("gap_from_leader" in leader for leader in today["leaders"]) and today["leaders"][0]["gap_from_leader"] == 0,
        )

    fantasy = tools.get_top_performers("fantasy", scope="today", limit=3)
    if "error" not in fantasy:
        check("today fantasy: scoring system labeled", fantasy.get("scoring") == "DraftKings classic")

    # Today's pitching leaderboard (strikeouts): same off-day-tolerant pattern.
    pitch_today = tools.get_top_performers("strikeOuts", scope="today", group="pitching", limit=3)
    if "error" in pitch_today:
        check("today pitching leaderboard: clean no-games message", "no games" in pitch_today["error"].lower())
    else:
        check(
            "today pitching leaderboard: scope today, group pitching, with leaders",
            pitch_today.get("scope") == "today" and pitch_today.get("group") == "pitching" and len(pitch_today["leaders"]) > 0,
        )
        check(
            "today pitching leaderboard: leaders carry player/team/value/line",
            all({"player", "team", "value", "line"} <= set(l) and "strikeOuts" in l["line"] for l in pitch_today["leaders"]),
        )
        check(
            "today pitching leaderboard: leaders carry gap_from_leader, leader at 0",
            all("gap_from_leader" in l for l in pitch_today["leaders"]) and pitch_today["leaders"][0]["gap_from_leader"] == 0,
        )
    # Deterministic (games or not): an unsupported pitching stat is rejected, so
    # the tool doesn't imply a composite best-performance ranking exists.
    bad_pitch = tools.get_top_performers("era", scope="today", group="pitching")
    check("today pitching leaderboard: unsupported stat rejected cleanly", "error" in bad_pitch and "available_stats" in bad_pitch)

    # Today's pitchers can be ranked by computed fantasy points, the counterpart
    # of the hitting path's stat="fantasy".
    fantasy_pitch = tools.get_top_performers("fantasy", scope="today", group="pitching", limit=3)
    if "error" in fantasy_pitch:
        check("today pitching fantasy: clean no-games message", "no games" in fantasy_pitch["error"].lower())
    else:
        check("today pitching fantasy: labels the pitcher scoring system", fantasy_pitch.get("scoring") == "DraftKings classic (pitching)")
        values = [l["value"] for l in fantasy_pitch["leaders"]]
        check("today pitching fantasy: ranked by score, descending", values == sorted(values, reverse=True))
        check(
            "today pitching fantasy: value is the pitcher score, not a zeroed hitter score",
            any(v != 0 for v in values) and all(l["value"] == l["fantasy_points"] for l in fantasy_pitch["leaders"]),
        )

    started = [g for g in tools._cached_schedule(None) if g.get("status") in tools._STARTED_STATUSES]
    if started:
        game = started[0]
        box = tools.get_game_boxscore(game["away_name"].split()[-1], game["home_name"].split()[-1])
        check(
            "game boxscore: batters carry fantasy points",
            "batters" in box and (not box["batters"] or "fantasy_points" in box["batters"][0]),
        )
    check("game boxscore unknown matchup -> error", "error" in tools.get_game_boxscore("Lakers", "Celtics"))

    # The by-id lookup backs the API's box-score endpoint. Same shape as the
    # team-name path, and it can address a game the team-name path cannot.
    schedule = tools._cached_schedule(None)
    started_games = [g for g in schedule if g.get("status") in tools._STARTED_STATUSES]
    if started_games:
        # Prefer a club whose raw box-score label disagrees with its schedule name
        # ("NY Mets Mets", "Arizona D-backs"). A game where the two happen to
        # coincide cannot catch the labelling bug, so it is the wrong test subject.
        mislabelled = ("Mets", "Yankees", "Cubs", "White Sox", "Diamondbacks", "Angels", "Athletics")
        game = next(
            (g for g in started_games if any(c in f"{g['away_name']} {g['home_name']}" for c in mislabelled)),
            started_games[0],
        )
        by_id = tools.get_game_boxscore_by_id(game["game_id"])
        check(
            "game boxscore by id: resolves to that game with batting lines",
            by_id.get("away_team") == game.get("away_name")
            and by_id.get("home_team") == game.get("home_name")
            and (not by_id["batters"] or "fantasy_points" in by_id["batters"][0]),
        )
        points = [b["fantasy_points"] for b in by_id["batters"]]
        check("game boxscore by id: batters sorted best fantasy line first", points == sorted(points, reverse=True))
        check(
            "game boxscore by id: pitching lines alongside the batting lines",
            "pitchers" in by_id
            and (not by_id["pitchers"] or {"player", "team", "side", "decision", "stats", "fantasy_points"} <= set(by_id["pitchers"][0])),
        )
        # Every row's team must equal one of the two top-level names exactly, so a
        # client can split the box score by side with an equality check. The raw
        # feed's own labels don't ("NY Mets Mets", "Arizona D-backs").
        labels = {row["team"] for row in by_id["batters"] + by_id["pitchers"]}
        check(
            "game boxscore by id: every row's team matches away_team or home_team exactly",
            labels and labels <= {by_id["away_team"], by_id["home_team"]},
        )
        check(
            "game boxscore by id: batters carry batting order and the substitution flag",
            all(isinstance(b.get("substitution"), bool) and (b.get("batting_order") is None or isinstance(b["batting_order"], int)) for b in by_id["batters"]),
        )
        starters = [b for b in by_id["batters"] if not b["substitution"] and b["batting_order"] is not None]
        check(
            "game boxscore by id: starters occupy whole-hundred slots, subs do not",
            all(b["batting_order"] % 100 == 0 for b in starters),
        )
    check("game boxscore by id: unknown id -> error", "error" in tools.get_game_boxscore_by_id(0))
    # A doubleheader is exactly what team names cannot disambiguate: only the
    # first match is reachable that way, so each id must resolve to its own game.
    pairs = [(g.get("away_name"), g.get("home_name")) for g in schedule]
    doubleheader = [g for g in schedule if pairs.count((g.get("away_name"), g.get("home_name"))) > 1]
    started_dh = [g for g in doubleheader if g.get("status") in tools._STARTED_STATUSES]
    if len(started_dh) > 1:
        check(
            "game boxscore by id: each half of a doubleheader resolves to its own game",
            all(
                tools.get_game_boxscore_by_id(g["game_id"]).get("home_score") == g.get("home_score")
                for g in started_dh
            ),
        )


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
