# CheckTheBall

Natural-language sports Q&A that answers from real stats data instead of the
model's memory. Ask something like *"who's had the better season, Ohtani or
Judge?"* and each claim in the answer is checked against the underlying numbers
before it's shown.

Work in progress. MLB first, structured so other sports can be added without
changing the core pipeline.

## Approach

An LLM chooses which data lookup to run but never invents the numbers — the
values come from the stats API through a typed data layer. A separate pass then
extracts each factual claim from the answer and verifies it against the data
that was actually retrieved.

## Layout

- `sports/<sport>/` — per-sport client, normalizer, and tools (MLB first)
- `core/` — shared schema, orchestrator, grounding, and storage; no sport-specific imports
- `api/` — FastAPI service
- `frontend/` — React + Tailwind UI

## Stack

Python, FastAPI, SQLite, MLB-StatsAPI, the Anthropic API, and React + Tailwind.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```
