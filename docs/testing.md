# Testing

How CheckTheBall is tested, what each suite covers, and the rules that keep the
tests trustworthy.

## Philosophy

- Offline and hermetic where possible. Most checks run with no network, no
  API key, and no cost, by injecting fakes for the model and, for the API, the
  database. Fakes follow one pattern across the suites: `FakeClient` /
  `FakeTools` (orchestrator), `FakeExtractor` (grounding), and a `TestClient`
  with monkeypatched pipeline functions (API).
- Deterministic where it matters. Live checks anchor on completed-season
  facts (a 2024 game, a 2024 stat line) that cannot change, so a green run means
  the pipeline is correct, not that today's data happened to line up.
- Run it, don't assume it. Suites are meant to be executed, not eyeballed.
  This caught a real bug: the opponent filter used `vsTeam`, which returns
  per-matchup sub-splits, and a stale anchor turned red until it was switched to
  `vsTeamTotal`. An assumed-passing test would have shipped wrong numbers.

## Two kinds of tests

- Offline / hermetic (`verify_orchestrator`, `verify_grounding`,
  `verify_api`, most of `verify_mlb_data`): fast, free, deterministic. Run these
  before every commit.
- Live / anchored (parts of `verify_mlb_data`, and `run_eval`): hit the real
  MLB API and, for `run_eval`, the real models. Anchored on stable facts so they
  stay comparable over time.

## The suites

| Suite | Checks | Network | What it covers |
|---|---|---|---|
| `scripts/verify_orchestrator.py` | 24 | none | the tool-use loop |
| `scripts/verify_grounding.py` | 19 | none | the deterministic grounding check |
| `scripts/verify_api.py` | 27 | none | the FastAPI endpoints |
| `scripts/verify_mlb_data.py` | 77 | live (anchored) | the MLB data layer, tools, filters, cache |
| `scripts/run_eval.py` | 35 questions | live (models) | full-pipeline benchmark |

### verify_orchestrator.py (24, offline)

A scripted fake Claude client and fake tool provider exercise the loop's
mechanics: a single tool call, a multi-round chain, an out-of-scope answer with
no tools, a tool that returns an `{"error": ...}` result, a tool that raises,
the round cap forcing a final answer, and parallel tool use (two `tool_use`
blocks in one response, both executed with both results returned in one turn).

### verify_grounding.py (19, offline)

A scripted extractor stands in for Haiku so the deterministic verification is
tested without a model. Covers fully grounded and partially grounded claim sets,
string vs number value forms, value-less claims being skipped, and several
edges: a value present only in a tool's `input` (the query-faithfulness gap made
explicit), a multi-value claim where only some values are backed, negative
numbers, whether text inside an `{"error": ...}` string can spuriously ground a
claim, and a date-range claim that states the queried year.

### verify_api.py (27, offline)

`TestClient` drives the app with the orchestrator and grounding calls faked and
the database pointed at a temporary file. Covers `POST /ask` happy path
(response shape plus a row landing in `queries`), empty question (422 from schema
validation) vs whitespace-only (400 from the handler), an over-length question
(422, orchestrator never invoked), the orchestrator raising (502, nothing logged,
and the exception detail not leaked to the client), grounding raising (still 200,
score falls back to None, answer still returned and logged), the per-IP rate
limit (requests over the limit get a clean 429), `GET /games/live` happy path
(live_count filters to `state == "live"`), and the schedule fetch raising (502,
no leak).

### verify_mlb_data.py (77, mixed)

A few `live_feature_checks` are conditional on there being games today, so the
runtime count can be a couple lower than the 77 `check()` calls in the file.

The main regression suite, in five groups:

- offline_checks: defensive normalization (unknown status, empty fields),
  the fantasy scoring formula on known lines, the box-score and date-range
  normalizers, and the `normalize_total_stat` grand-total selection (including
  the no-`sport.id==0` fallback and the never-sum-the-duplicates rule).
- live_checks: schedule, play-by-play, season, and career normalization
  anchored on a known 2024 game and Aaron Judge's 2024 line.
- cache_checks: freshness (a stale row refetches), the whitelist guard, the
  upsert (same key replaced in place, no duplicate row), and `log_query` writing
  a row verbatim.
- tool_checks: each Claude-facing tool against known 2024 numbers, error
  paths returning `{"error": ...}`, a pitching anchor (Skubal 2024), and a check
  that every schema has a matching function.
- filter_checks: date-range presets and a custom window, opponent, and split
  against stable 2024 facts, plus the guards (filters not combinable, unknown
  values error cleanly).
- live_feature_checks: today's-games tools, structural and tolerant of
  off-days (assert shape and presence, not fixed numbers that change daily).

## The eval and its two metrics

`run_eval.py` runs the full pipeline over 35 questions across 15 categories and
reports two separate rates. It is a benchmark, not a pass/fail gate: scores vary
between runs because claim extraction and tool selection are model calls. Each
run appends a summary row to `docs/eval_history.md`.

- Grounding rate answers: does the answer's content trace back to the data
  that was retrieved? (faithful to the data)
- Tool-faithfulness rate answers: did the model retrieve the right data, the
  correct tool with the key arguments, for the question? (faithful to the
  question). Annotated per question with an expected tool and only the arguments
  that disambiguate the right lookup; deterministic, no model judge.

The two are complementary, shown by one case in a run: an ambiguous surname
question scored grounding 1.0 (the answer asserted nothing false) while
tool-faithfulness flagged it a miss (the model made no lookup at all). Grounding
alone could not see that the question went unanswered by data.

## The anchoring rule

Anchor every new live assertion on completed 2024 (or earlier) data, never on
current-season or today's numbers. A finished season cannot change, so the
assertion stays valid on every future run. This is why `run_eval` questions and
the `verify_mlb_data` anchors use 2024 facts.

## What is not covered

- Live and today features can only be tested structurally (shape, field
  presence), because the underlying data changes daily. `live_feature_checks`
  and the `live` eval category are tolerant by design.
- `run_eval` is a benchmark, not a gate. Treat its scores as a tracked range
  in `eval_history.md`, not a threshold to pass.
- Tool faithfulness is scoped to eval questions whose correct call is
  unambiguous. Scoring arbitrary live queries against inferred intent is future
  work (see `roadmap.md`).

## How to run

Offline suites (fast, free, run before committing):

```bash
python scripts/verify_orchestrator.py
python scripts/verify_grounding.py
python scripts/verify_api.py
python scripts/verify_mlb_data.py   # includes a few live, anchored checks
```

Full live benchmark (makes real MLB and model calls; needs `ANTHROPIC_API_KEY`):

```bash
python scripts/run_eval.py
```

Each script exits non-zero if any check fails and prints a per-check pass/fail
line.
