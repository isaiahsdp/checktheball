"""Verify the FastAPI layer offline with a TestClient and injected fakes.

The orchestrator and grounding calls are replaced with fakes, and the database
points at a temporary file, so every check runs with no network, no API key, and
no cost. This mirrors the fake-client pattern in verify_orchestrator.py.

Run from the repo root:

    python scripts/verify_api.py

Exit code is 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the DB at a temp file BEFORE anything imports core.db, so the default
# path (bound at import time) is the throwaway database.
_TMP_DIR = tempfile.mkdtemp(prefix="checktheball_api_")
_DB_PATH = os.path.join(_TMP_DIR, "api_test.sqlite")
os.environ["CHECKTHEBALL_DB"] = _DB_PATH

from fastapi.testclient import TestClient

import api.main as main
from core import db, grounding, orchestrator
from sports.mlb import client
from sports.mlb import tools as mlb_tools

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


# Originals restored at the end so the process leaves modules as it found them.
_ORIGINALS = {
    "answer_question": orchestrator.answer_question,
    "ground_answer": grounding.ground_answer,
    "get_schedule": client.get_schedule,
}


# --- Fakes -----------------------------------------------------------------

_ANSWER = {
    "answer": "Aaron Judge hit 58 home runs.",
    "tool_calls_made": [{"name": "get_player_stat", "input": {"player": "Aaron Judge", "stat": "homeRuns"}}],
    "tool_results": [{"name": "get_player_stat", "input": {}, "result": {"player": "Aaron Judge", "value": 58}}],
}


def _answer_ok(question, tools, **kwargs):
    return _ANSWER


def _answer_boom(question, tools, **kwargs):
    raise RuntimeError("orchestrator down")


def _grounding_ok(answer, tool_results, **kwargs):
    return {
        "grounding_score": 1.0,
        "supported_claims": 1,
        "total_claims": 1,
        "claims": [{"text": "Aaron Judge hit 58 home runs", "values": ["58"], "supported": True, "missing": []}],
    }


def _grounding_boom(answer, tool_results, **kwargs):
    raise RuntimeError("grounding down")


def _schedule_ok(date=None):
    # 4 games, 2 of them live (In Progress + Manager challenge).
    return [
        {"game_id": 1, "home_id": 10, "home_name": "Home A", "away_id": 11, "away_name": "Away B",
         "status": "In Progress", "home_score": 3, "away_score": 1, "game_datetime": "2024-06-01T18:00:00Z", "venue_name": "Park"},
        {"game_id": 2, "home_id": 12, "home_name": "Home C", "away_id": 13, "away_name": "Away D",
         "status": "Manager challenge", "home_score": 2, "away_score": 2, "game_datetime": "2024-06-01T18:00:00Z", "venue_name": "Park"},
        {"game_id": 3, "home_id": 14, "home_name": "Home E", "away_id": 15, "away_name": "Away F",
         "status": "Final", "home_score": 5, "away_score": 2, "game_datetime": "2024-06-01T18:00:00Z", "venue_name": "Park"},
        {"game_id": 4, "home_id": 16, "home_name": "Home G", "away_id": 17, "away_name": "Away H",
         "status": "Scheduled", "home_score": "", "away_score": "", "game_datetime": "2024-06-01T22:00:00Z", "venue_name": "Park"},
    ]


def _schedule_boom(date=None):
    raise RuntimeError("schedule down")


def _schedule_empty(date=None):
    return []


def _schedule_only_on(target: str):
    """Schedule that has games on exactly one date, for the fallback walk."""

    def fetch(date=None):
        return _schedule_ok() if date == target else []

    return fetch


_BOXSCORE = {
    "date": "today",
    "away_team": "Texas Rangers",
    "home_team": "Tampa Bay Rays",
    "status": "In Progress",
    "away_score": 0,
    "home_score": 3,
    "scoring": "DraftKings classic",
    "batters": [
        {"player": "Ryan Vilade", "team": "Tampa Bay Rays", "side": "home", "position": "RF",
         "batting_order": 900, "substitution": False,
         "stats": {"ab": 2, "r": 1, "h": 1, "doubles": 0, "triples": 0, "hr": 1, "rbi": 2, "sb": 0, "bb": 0, "k": 0},
         "fantasy_points": 16},
    ],
    "pitchers": [
        {"player": "Logan Webb", "team": "San Francisco Giants", "side": "away", "decision": "W",
         "stats": {"ip": 6.0, "h": 5, "r": 2, "er": 2, "bb": 1, "k": 4, "hr": 0, "pitches": 96, "strikes": 61},
         "fantasy_points": 17.9},
    ],
}


def _boxscore_ok(game_id, date=None):
    return _BOXSCORE


def _boxscore_missing(game_id, date=None):
    return {"error": f"No game found with id '{game_id}' for that date."}


def _boxscore_not_started(game_id, date=None):
    return {"error": "The Boston Red Sox at Athletics game hasn't started yet.", "status": "Pre-Game"}


def _boxscore_boom(game_id, date=None):
    raise RuntimeError("boxscore down")


# --- Temp-DB helpers -------------------------------------------------------

def _queries_count() -> int:
    with sqlite3.connect(_DB_PATH) as con:
        return con.execute("SELECT COUNT(*) FROM queries").fetchone()[0]


def _last_query() -> sqlite3.Row:
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        return con.execute("SELECT * FROM queries ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        con.close()


# --- Checks ----------------------------------------------------------------

def ask_happy_path() -> None:
    orchestrator.answer_question = _answer_ok
    grounding.ground_answer = _grounding_ok
    before = _queries_count()
    with TestClient(main.app) as tc:
        r = tc.post("/ask", json={"question": "How many home runs did Judge hit?"})
    check("ask happy: 200", r.status_code == 200)
    body = r.json()
    check("ask happy: answer returned", body.get("answer") == "Aaron Judge hit 58 home runs.")
    check("ask happy: grounding_score present", body.get("grounding_score") == 1.0)
    check("ask happy: grounding.claims present", len(body["grounding"]["claims"]) == 1)
    check("ask happy: tool_calls present", body["tool_calls"][0]["name"] == "get_player_stat")
    # The client needs the retrieved values, not just which tool ran. The fake's
    # tool_calls_made carries no result, so sourcing from it again fails here.
    call = body["tool_calls"][0]
    check(
        "ask happy: tool_calls carry their result data",
        "result" in call and call["result"] and call["result"]["value"] == 58,
    )
    check("ask happy: tool_calls still carry name and input", "input" in call and call["name"] == "get_player_stat")
    check("ask happy: one row logged", _queries_count() == before + 1)
    row = _last_query()
    check(
        "ask happy: logged row has question/answer/score",
        row["question"].startswith("How many") and row["answer"] == "Aaron Judge hit 58 home runs." and row["grounding_score"] == 1.0,
    )


def ask_empty_and_whitespace() -> None:
    # Empty string is rejected by the Pydantic min_length=1 (422) before the
    # handler runs; whitespace-only passes validation then hits the strip -> 400.
    with TestClient(main.app) as tc:
        empty = tc.post("/ask", json={"question": ""})
        blank = tc.post("/ask", json={"question": "   "})
    check("ask empty: 422 (schema validation)", empty.status_code == 422)
    check("ask whitespace-only: 400 (handler guard)", blank.status_code == 400)


def ask_too_long() -> None:
    # A question over max_length is rejected by schema validation (422) before the
    # handler runs, so the orchestrator is never reached (no wasted paid call).
    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        return _ANSWER

    orchestrator.answer_question = _spy
    long_question = "a" * (main.MAX_QUESTION_LENGTH + 100)
    with TestClient(main.app) as tc:
        r = tc.post("/ask", json={"question": long_question})
    check("ask too-long: 422 (schema validation)", r.status_code == 422)
    check("ask too-long: orchestrator never invoked", called["n"] == 0)


def ask_orchestrator_raises() -> None:
    orchestrator.answer_question = _answer_boom
    grounding.ground_answer = _grounding_ok  # must not be reached
    before = _queries_count()
    with TestClient(main.app) as tc:
        r = tc.post("/ask", json={"question": "boom"})
    check("orchestrator raises: 502", r.status_code == 502)
    check("orchestrator raises: nothing logged", _queries_count() == before)
    # The exception detail ("orchestrator down") must not leak to the client.
    check("orchestrator raises: response does not leak the exception", "orchestrator down" not in r.text)


def ask_grounding_raises() -> None:
    orchestrator.answer_question = _answer_ok
    grounding.ground_answer = _grounding_boom
    before = _queries_count()
    with TestClient(main.app) as tc:
        r = tc.post("/ask", json={"question": "grounding fails but answer is fine"})
    check("grounding raises: still 200", r.status_code == 200)
    body = r.json()
    check("grounding raises: grounding_score falls back to None", body.get("grounding_score") is None)
    check("grounding raises: answer still returned", body.get("answer") == "Aaron Judge hit 58 home runs.")
    check("grounding raises: still logged", _queries_count() == before + 1)
    check("grounding raises: logged score is NULL", _last_query()["grounding_score"] is None)


def games_live_happy_path() -> None:
    client.get_schedule = _schedule_ok
    with TestClient(main.app) as tc:
        r = tc.get("/games/live", params={"date": "2024-06-01"})
    check("games happy: 200", r.status_code == 200)
    body = r.json()
    check("games happy: count == 4", body["count"] == 4)
    live_in_list = sum(1 for g in body["games"] if g["state"] == "live")
    check("games happy: live_count filters to state==live (2)", body["live_count"] == 2 and live_in_list == 2)


def games_live_today_key_is_dated() -> None:
    # A request with no date must cache under the real date. Under a literal
    # "today" key, a row written just before midnight is still fresh after the
    # rollover, so the next day's first callers get the previous day's games.
    client.get_schedule = _schedule_ok
    with TestClient(main.app) as tc:
        r = tc.get("/games/live")
    check("today key: 200", r.status_code == 200)
    with sqlite3.connect(_DB_PATH) as con:
        keys = {row[0] for row in con.execute("SELECT key FROM games")}
    today = datetime.now().strftime("%Y-%m-%d")
    check("today key: row is keyed by the real date", f"schedule:{today}" in keys)
    check("today key: no literal 'today' key written", "schedule:today" not in keys)


def games_live_fetch_raises() -> None:
    client.get_schedule = _schedule_boom
    with TestClient(main.app) as tc:
        # Fresh date key so the cache can't serve a prior result.
        r = tc.get("/games/live", params={"date": "2024-06-02"})
    check("games fetch raises: 502", r.status_code == 502)
    # The exception detail ("schedule down") must not leak to the client.
    check("games fetch raises: response does not leak the exception", "schedule down" not in r.text)


def games_live_fallback() -> None:
    # An empty day returns nothing by default, and fell_back_to is null so the
    # client can tell a real day from a substituted one.
    client.get_schedule = _schedule_empty
    with TestClient(main.app) as tc:
        plain = tc.get("/games/live", params={"date": "2026-01-15"}).json()
    check("fallback absent: empty day stays empty", plain["count"] == 0 and plain["games"] == [])
    check("fallback absent: fell_back_to is null", plain["fell_back_to"] is None)

    # With the fallback, the walk finds the one day that has games and names it.
    client.get_schedule = _schedule_only_on("2026-01-12")
    with TestClient(main.app) as tc:
        fell = tc.get("/games/live", params={"date": "2026-01-15", "fallback": "last_played"}).json()
    check("fallback: walks back to the most recent day with games", fell["count"] == 4)
    check("fallback: fell_back_to names the day used", fell["fell_back_to"] == "2026-01-12")
    check("fallback: response keeps its existing shape", {"date", "count", "live_count", "games"} <= set(fell))

    # A non-empty day must not walk at all, even with the fallback set.
    client.get_schedule = _schedule_ok
    with TestClient(main.app) as tc:
        direct = tc.get("/games/live", params={"date": "2026-02-01", "fallback": "last_played"}).json()
    check("fallback: a day that has games does not fall back", direct["count"] == 4 and direct["fell_back_to"] is None)

    # Beyond the cap the walk gives up rather than running away. Uses a date
    # window no earlier check touched: the schedule cache is shared across checks,
    # so a day another fake already populated would be served from cache here.
    client.get_schedule = _schedule_empty
    with TestClient(main.app) as tc:
        gave_up = tc.get("/games/live", params={"date": "2026-05-20", "fallback": "last_played"}).json()
    check("fallback: gives up past the day cap", gave_up["count"] == 0 and gave_up["fell_back_to"] is None)

    # An unknown value is rejected rather than silently ignored.
    with TestClient(main.app) as tc:
        bad = tc.get("/games/live", params={"fallback": "yesterday"})
    check("fallback: unknown value -> 400", bad.status_code == 400)


def boxscore_endpoint() -> None:
    main_boxscore = mlb_tools.get_game_boxscore_by_id
    try:
        mlb_tools.get_game_boxscore_by_id = _boxscore_ok
        with TestClient(main.app) as tc:
            r = tc.get("/games/822947/boxscore")
        check("boxscore: 200", r.status_code == 200)
        body = r.json()
        check("boxscore: teams, status, and scores present", body["away_team"] == "Texas Rangers" and body["status"] == "In Progress" and body["home_score"] == 3)
        check("boxscore: batters carry team, side, position, order, stats, fantasy_points",
              {"player", "team", "side", "position", "batting_order", "substitution", "stats", "fantasy_points"} <= set(body["batters"][0]))
        check("boxscore: short box-score stat keys, not MLB camelCase",
              {"ab", "r", "h", "doubles", "triples", "hr", "rbi", "sb", "bb", "k"} == set(body["batters"][0]["stats"]))
        arm = body["pitchers"][0]
        check("boxscore: pitchers array present alongside batters",
              {"player", "team", "decision", "stats", "fantasy_points"} <= set(arm))
        check("boxscore: pitching stats carry innings and the W/L/S decision",
              arm["stats"]["ip"] == 6.0 and arm["decision"] == "W" and "era" not in arm["stats"])

        mlb_tools.get_game_boxscore_by_id = _boxscore_missing
        with TestClient(main.app) as tc:
            missing = tc.get("/games/999999/boxscore")
        check("boxscore: unknown game_id -> 404", missing.status_code == 404)

        mlb_tools.get_game_boxscore_by_id = _boxscore_not_started
        with TestClient(main.app) as tc:
            early = tc.get("/games/824973/boxscore")
        check("boxscore: not-yet-started -> 409", early.status_code == 409)
        check("boxscore: 409 detail carries the schedule status", "Pre-Game" in early.json()["detail"])

        mlb_tools.get_game_boxscore_by_id = _boxscore_boom
        with TestClient(main.app) as tc:
            broken = tc.get("/games/822947/boxscore")
        check("boxscore: upstream failure -> 502", broken.status_code == 502)
        check("boxscore: 502 does not leak the exception", "boxscore down" not in broken.text)

        # Not on the /ask limiter: a cached read must not burn the model budget.
        main.limiter.reset()
        mlb_tools.get_game_boxscore_by_id = _boxscore_ok
        with TestClient(main.app) as tc:
            codes = [tc.get("/games/822947/boxscore").status_code for _ in range(main.ASK_RATE_LIMIT_PER_MINUTE + 3)]
        check("boxscore: not rate-limited like /ask", all(c == 200 for c in codes))
        main.limiter.reset()
    finally:
        mlb_tools.get_game_boxscore_by_id = main_boxscore


def ask_rate_limit() -> None:
    # The per-IP /ask limit shares in-memory state across requests, so reset it
    # to isolate this test from the other /ask checks, then fire more than the
    # limit's worth in quick succession.
    main.limiter.reset()
    orchestrator.answer_question = _answer_ok
    grounding.ground_answer = _grounding_ok
    limit = main.ASK_RATE_LIMIT_PER_MINUTE
    with TestClient(main.app) as tc:
        responses = [tc.post("/ask", json={"question": "spam"}) for _ in range(limit + 2)]
    codes = [r.status_code for r in responses]
    check("rate limit: requests up to the limit succeed", codes[:limit] == [200] * limit)
    check("rate limit: requests over the limit return 429", all(c == 429 for c in codes[limit:]))
    over_body = responses[-1].json()
    check("rate limit: 429 body is clean JSON (not a stack trace)", isinstance(over_body, dict) and "error" in over_body)
    main.limiter.reset()  # leave the limiter clean for any later tests


def ask_daily_limit_registered() -> None:
    # The daily limb can't be driven behaviourally: the per-minute limb trips at
    # 6 requests, so the 50th is unreachable inside one minute. Assert instead
    # that both limbs are registered on the route, which is what would break if
    # the ";50/day" half of the composed limit string were ever dropped.
    registered = [str(limit.limit) for limit in main.limiter._route_limits["api.main.ask"]]
    check(
        "rate limit: per-minute limb registered on /ask",
        f"{main.ASK_RATE_LIMIT_PER_MINUTE} per 1 minute" in registered,
    )
    check(
        "rate limit: per-day limb registered on /ask (bounds spend per IP)",
        f"{main.ASK_RATE_LIMIT_PER_DAY} per 1 day" in registered,
    )


def main_() -> int:
    db.init_db()  # ensure the queries table exists before the first count
    print("API checks (TestClient, injected fakes, temp DB, no network)")
    try:
        ask_happy_path()
        ask_empty_and_whitespace()
        ask_too_long()
        ask_orchestrator_raises()
        ask_grounding_raises()
        ask_rate_limit()
        ask_daily_limit_registered()
        games_live_happy_path()
        games_live_today_key_is_dated()
        games_live_fetch_raises()
        games_live_fallback()
        boxscore_endpoint()
    finally:
        orchestrator.answer_question = _ORIGINALS["answer_question"]
        grounding.ground_answer = _ORIGINALS["ground_answer"]
        client.get_schedule = _ORIGINALS["get_schedule"]
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main_())
