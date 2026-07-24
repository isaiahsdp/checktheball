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


def games_live_fetch_raises() -> None:
    client.get_schedule = _schedule_boom
    with TestClient(main.app) as tc:
        # Fresh date key so the cache can't serve a prior result.
        r = tc.get("/games/live", params={"date": "2024-06-02"})
    check("games fetch raises: 502", r.status_code == 502)
    # The exception detail ("schedule down") must not leak to the client.
    check("games fetch raises: response does not leak the exception", "schedule down" not in r.text)


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
        games_live_happy_path()
        games_live_fetch_raises()
    finally:
        orchestrator.answer_question = _ORIGINALS["answer_question"]
        grounding.ground_answer = _ORIGINALS["ground_answer"]
        client.get_schedule = _ORIGINALS["get_schedule"]
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main_())
