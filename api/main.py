"""FastAPI app exposing the CheckTheBall pipeline over HTTP.

Two endpoints back the product: ``GET /games/live`` lists today's games from
the MLB feed, and ``POST /ask`` runs a question through the orchestrator, scores
the answer with the grounding layer, logs it, and returns all three together so
a client can show the answer alongside how well it traces back to the data.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import db, grounding, orchestrator
from sports.mlb import client, normalizer
from sports.mlb import tools as mlb_tools

# The schedule changes as games start and scores move, so keep the cache short.
LIVE_GAMES_MAX_AGE_SECONDS = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()  # ensure the cache tables and query log exist before serving
    yield


app = FastAPI(title="CheckTheBall", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "checktheball"}


@app.get("/games/live")
def games_live(date: str | None = None) -> dict:
    """Today's MLB games (or a given ``date=YYYY-MM-DD``), normalized.

    Results are cached briefly so repeated polling doesn't hammer the feed.
    """
    key = f"schedule:{date or 'today'}"

    def fetch() -> list[dict[str, Any]]:
        raw = client.get_schedule(date=date)
        return [normalizer.normalize_game(g).to_dict() for g in raw]

    try:
        games = db.cached_fetch(
            "games", key, normalizer.SPORT, fetch, LIVE_GAMES_MAX_AGE_SECONDS
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Schedule lookup failed: {exc}")

    live = [g for g in games if g["state"] == "live"]
    return {
        "date": date,
        "count": len(games),
        "live_count": len(live),
        "games": games,
    }


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A natural-language sports question.")


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer a question from real data and report how grounded the answer is."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        answered = orchestrator.answer_question(question, mlb_tools)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Answering failed: {exc}")

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
        "tool_calls": answered["tool_calls_made"],
    }
