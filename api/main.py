"""FastAPI app exposing the CheckTheBall pipeline over HTTP.

Two endpoints back the product: ``GET /games/live`` lists today's games from
the MLB feed, and ``POST /ask`` runs a question through the orchestrator, scores
the answer with the grounding layer, logs it, and returns all three together so
a client can show the answer alongside how well it traces back to the data.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core import db, grounding, orchestrator
from sports.mlb import client, normalizer
from sports.mlb import tools as mlb_tools

logger = logging.getLogger(__name__)

# The schedule changes as games start and scores move, so keep the cache short.
LIVE_GAMES_MAX_AGE_SECONDS = 30

# Opt-in fallback for an empty day, and how far back it will look before giving
# up. Bounded so an off-season request can't walk backwards indefinitely.
_FALLBACK_LAST_PLAYED = "last_played"
MAX_FALLBACK_DAYS = 10

# Cap question length: long enough for any real sports question, short enough to
# reject a huge pasted block meant to burn tokens on a call that would fail anyway.
MAX_QUESTION_LENGTH = 500

# /ask triggers paid Anthropic calls, so cap it per client IP: a burst limit to
# stop a bot loop or spam-click, and a daily limit to bound total spend per IP.
ASK_RATE_LIMIT_PER_MINUTE = 5
ASK_RATE_LIMIT_PER_DAY = 50
ASK_RATE_LIMIT = f"{ASK_RATE_LIMIT_PER_MINUTE}/minute;{ASK_RATE_LIMIT_PER_DAY}/day"
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()  # ensure the cache tables and query log exist before serving
    yield


app = FastAPI(title="CheckTheBall", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "checktheball"}


def _load_games(date: str | None) -> list[dict[str, Any]]:
    """Normalized schedule for one day, served from the short-lived cache."""

    def fetch() -> list[dict[str, Any]]:
        raw = client.get_schedule(date=date)
        return [normalizer.normalize_game(g).to_dict() for g in raw]

    return db.cached_fetch(
        "games", f"schedule:{date or 'today'}", normalizer.SPORT, fetch, LIVE_GAMES_MAX_AGE_SECONDS
    )


def _last_played(date: str | None) -> tuple[list[dict[str, Any]], str | None]:
    """Walk back day by day to the most recent day with games, bounded.

    Returns (games, day_used) or ([], None) if nothing is found within the cap,
    so an off-season request terminates instead of walking indefinitely.
    """
    start = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
    for back in range(1, MAX_FALLBACK_DAYS + 1):
        day = (start - timedelta(days=back)).strftime("%Y-%m-%d")
        games = _load_games(day)
        if games:
            return games, day
    return [], None


@app.get("/games/live")
def games_live(date: str | None = None, fallback: str | None = None) -> dict:
    """Today's MLB games (or a given ``date=YYYY-MM-DD``), normalized.

    Results are cached briefly so repeated polling doesn't hammer the feed. With
    ``fallback=last_played``, an empty day returns the most recent day that had
    games instead, and ``fell_back_to`` names the day actually returned.
    """
    if fallback is not None and fallback != _FALLBACK_LAST_PLAYED:
        raise HTTPException(
            status_code=400, detail=f"Unknown fallback '{fallback}'. Use '{_FALLBACK_LAST_PLAYED}'."
        )

    try:
        games = _load_games(date)
        fell_back_to = None
        if not games and fallback == _FALLBACK_LAST_PLAYED:
            games, fell_back_to = _last_played(date)
    except Exception:
        # Log the real detail server-side; don't leak internals to the client.
        logger.exception("Schedule lookup failed")
        raise HTTPException(status_code=502, detail="Could not load games right now.")

    live = [g for g in games if g["state"] == "live"]
    return {
        "date": date,
        "fell_back_to": fell_back_to,
        "count": len(games),
        "live_count": len(live),
        "games": games,
    }


@app.get("/games/{game_id}/boxscore")
def game_boxscore(game_id: str, date: str | None = None) -> dict:
    """Batting lines for one game, keyed by the game_id ``/games/live`` returns.

    Batters only, both teams in one array, best fantasy line first. Keyed by id
    rather than team names so each half of a doubleheader is reachable. Not
    rate-limited: it is a cached read and makes no model call.
    """
    try:
        box = mlb_tools.get_game_boxscore_by_id(game_id, date=date)
    except Exception:
        # Log the real detail server-side; don't leak internals to the client.
        logger.exception("Box score lookup failed")
        raise HTTPException(status_code=502, detail="Could not load that box score right now.")

    if "error" in box:
        # Only the not-yet-started case carries the schedule status; a missing id
        # has nothing to report but the id itself.
        if "status" in box:
            raise HTTPException(status_code=409, detail=f"{box['error']} Status: {box['status']}.")
        raise HTTPException(status_code=404, detail=box["error"])
    return box


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=MAX_QUESTION_LENGTH, description="A natural-language sports question."
    )


@app.post("/ask")
@limiter.limit(ASK_RATE_LIMIT)
def ask(request: Request, req: AskRequest) -> dict:
    """Answer a question from real data and report how grounded the answer is.

    Rate-limited per client IP because each call makes paid model requests.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        answered = orchestrator.answer_question(question, mlb_tools)
    except Exception:
        # Log the real detail server-side; don't leak internals to the client.
        logger.exception("Answering failed")
        raise HTTPException(status_code=502, detail="Could not answer the question right now.")

    # Grounding is best-effort: a scoring failure shouldn't drop a good answer.
    try:
        graded = grounding.ground_answer(answered["answer"], answered["tool_results"])
    except Exception:
        graded = {"grounding_score": None, "supported_claims": 0, "total_claims": 0, "claims": []}

    db.log_query(
        question, answered["answer"], answered["tool_calls_made"], graded["grounding_score"]
    )

    return {
        "question": question,
        "answer": answered["answer"],
        "grounding_score": graded["grounding_score"],
        "grounding": {
            "supported_claims": graded["supported_claims"],
            "total_claims": graded["total_claims"],
            "claims": graded["claims"],
        },
        # tool_results carries the same calls as tool_calls_made plus each one's
        # returned data, which a client needs to render the underlying numbers.
        "tool_calls": answered["tool_results"],
    }
