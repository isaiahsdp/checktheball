# CheckTheBall

Natural-language sports Q&A that answers from real stats data instead of the
model's memory. Ask something like _"who's had the better season, Ohtani or
Judge?"_ and each claim in the answer is checked against the underlying numbers
before it's shown.

Backend/API only. MLB first, built so other sports can be added later.

## Why

Asking an LLM directly for a stat is unreliable, since it can answer from
memory and be wrong with no way to tell. Stat lookup tools are accurate but
rigid, so a compound question ("who's better") usually means pulling several
numbers yourself and comparing them by hand. This project tries to get both:
a model that can handle the flexible part of the question, with every number
it states checked against real data before it's shown.

## Demo

## Approach

The model picks which data to look up, but never supplies the numbers
itself. Two separate, automatic checks then run on its answer:

1. **Grounding.** Every factual claim in the answer is checked against the
   data that was actually retrieved. Nothing gets a pass just for sounding
   right.
2. **Tool faithfulness.** A check on whether the model looked up the right
   thing in the first place, not just whether its answer is internally
   consistent. This matters because a claim-free answer can pass grounding
   perfectly even if no real lookup happened at all.

The model is also told today's real date on every request, so questions like
"this season" resolve correctly instead of guessing from training data.

Every fix and feature here came from asking it real questions and noticing
something subtly wrong.

## What it can answer

Player stats (season, career, or filtered by home/away, a date range, or an
opponent), player comparisons, leaderboards, season pace projections, box
scores, who's performing best in today's games, and DraftKings-style fantasy
scoring for hitters and pitchers.

## Layout

- `sports/<sport>/`: per-sport data and tools (MLB first)
- `core/`: shared logic, sport-agnostic
- `api/`: FastAPI service (`POST /ask`, `GET /games/live`)

## API

**`POST /ask`**: ask a question, get an answer with its grounding score and
the data it was checked against. Rate-limited per IP, since each call makes
a real, paid model request.

**`GET /games/live`**: today's games and scores. Pass
`?fallback=last_played` and an empty day returns the most recent day that had
games instead, so a scoreboard doesn't have to hide itself in the off-season.
No model call, no key needed.

**`GET /games/{game_id}/boxscore`**: every batter's and pitcher's line for one
game. Keyed by game id rather than team names, so both halves of a
doubleheader are reachable. No model call, no key needed.

## Stack

Python, FastAPI, SQLite, MLB-StatsAPI, and the Anthropic API.

## Testing

Four fast offline test suites run before every commit. A separate live
benchmark runs real questions through the actual model and tracks results
over time, since live model output varies run to run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

Run the API:

```bash
uvicorn api.main:app --port 8000 --reload
```

Serves on `http://localhost:8000`. `GET /games/live` works without a key;
`POST /ask` needs `ANTHROPIC_API_KEY` set in `.env`.
