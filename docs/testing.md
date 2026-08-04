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
| `scripts/verify_orchestrator.py` | 56 | none | the tool-use loop |
| `scripts/verify_grounding.py` | 37 | none | the deterministic grounding check |
| `scripts/verify_api.py` | 54 | none | the FastAPI endpoints |
| `scripts/verify_mlb_data.py` | 156 | live (anchored) | the MLB data layer, tools, filters, cache |
| `scripts/run_eval.py` | 40 questions | live (models) | full-pipeline benchmark |

### verify_orchestrator.py (56, offline)

A scripted fake Claude client and fake tool provider exercise the loop's
mechanics: a single tool call, a multi-round chain, an out-of-scope answer with
no tools, a tool that returns an `{"error": ...}` result, a tool that raises,
the round cap forcing a final answer, and parallel tool use (two `tool_use`
blocks in one response, both executed with both results returned in one turn).
Also asserts what each request's system prompt carries: the current date, so the
model resolves "this season" against reality instead of guessing from its
training, and the rule that a number the model works out itself is not verified
data (the rule `gap_from_leader` and the `compute` tool exist to serve).

Also covers the `compute` tool: the operand check that refuses numbers no tool
returned, each operation and its guards (divide by zero, unknown operation, a
single operand, a non-numeric operand), and its wiring into the loop: handled
by the orchestrator rather than dispatched to the sport provider, offered
alongside the provider's schemas without mutating them, recorded in
`tool_calls_made` like any other call, and a refused operand fed back as an
error the model can correct.

### verify_grounding.py (37, offline)

A scripted extractor stands in for Haiku so the deterministic verification is
tested without a model. Covers fully grounded and partially grounded claim sets,
string vs number value forms, value-less claims being skipped, and several
edges: a value present only in a tool's `input` (the query-faithfulness gap made
explicit), a multi-value claim where only some values are backed, negative
numbers, whether text inside an `{"error": ...}` string can spuriously ground a
claim, and a date-range claim that states the queried year.

Two paired sections carry the arithmetic design. One asserts that a number the
model computed itself does not ground, covering a correct rate in decimal,
percent, and rounded-percent form, and a correct difference, all unsupported on
their own. The other runs the real `compute` tool and shows the same rate
grounding once it arrives as a tool result. Re-adding an after-the-fact check
would flip the first set, which is why they are guarded. A leaderboard gap is the same
story landed at the source: it grounds via the precomputed `gap_from_leader`,
and does not ground when that field is stripped.

Two sections cover the conditions a stat was measured under, which are context
rather than claims. An ISO date is recognized by shape and waved through, since
a refusal citing today's date put one in `values` and scored 0.00; a bare year
is still checked, and a partial date is not treated as one. An uppercase split
label ("RHP") is checked as a proper noun and fails against data that says
`vs_right`, so the extraction prompt excludes handedness and home/away labels
the way it already excludes dates. The lowercase spelled-out form was never
affected, which is why only abbreviated answers failed in production.

### verify_api.py (54, offline)

`TestClient` drives the app with the orchestrator and grounding calls faked and
the database pointed at a temporary file. Covers `POST /ask` happy path
(response shape plus a row landing in `queries`), empty question (422 from schema
validation) vs whitespace-only (400 from the handler), an over-length question
(422, orchestrator never invoked), the orchestrator raising (502, nothing logged,
and the exception detail not leaked to the client), grounding raising (still 200,
score falls back to None, answer still returned and logged), the per-IP rate
limit (requests over the limit get a clean 429), `GET /games/live` happy path
(live_count filters to `state == "live"`), and the schedule fetch raising (502,
no leak). The daily limb of the rate limit can't be driven behaviourally (the
per-minute limb trips at 6 requests, so the 50th is unreachable inside a
minute), so it is asserted to be registered on the route instead.

Also covers the two `/games/*` additions: `GET /games/{game_id}/boxscore` (200
shape, 404 unknown id, 409 not yet started with the status in the detail, 502 with
no leak, and that it is *not* on the `/ask` limiter), and
`?fallback=last_played` (an empty day walks back to the most recent day with
games and names it in `fell_back_to`, a non-empty day doesn't walk, the walk gives
up past the day cap, and an unknown value is rejected rather than ignored). One
trap worth knowing before adding schedule tests: the schedule cache is shared
across checks in this file, so a day another fake already populated is served from
cache. The cap check deliberately uses a date window no earlier check touches.

A request with no `date` must cache under the real date, so the row written by
`/games/live` is asserted to be keyed `schedule:<today>` with no literal
`schedule:today` key present. Keyed on the string, a row written just before
midnight is still inside its freshness window after the rollover and serves the
previous day's slate.

### verify_mlb_data.py (156, mixed)

A few `live_feature_checks` are conditional on there being games today, so the
runtime count can be several lower than the 167 `check()` calls in the file.

The main regression suite, in six groups:

- offline_checks: defensive normalization (unknown status, empty fields),
  the fantasy scoring formula on known lines, the box-score batting and pitching
  normalizers (header rows skipped, season rates excluded, the W/L/S decision
  parsed out of the note, batting order and substitution passed through, and
  `side` carried instead of the box score's own doubled-up team label) and the
  date-range normalizers, the `normalize_total_stat` grand-total selection
  (including the no-`sport.id==0` fallback and the never-sum-the-duplicates
  rule), `team_from_splits` (reads the player's own team, never the vsTeam
  `opponent`, and skips the teamless grand total), the `gap_from_leader`
  magnitude on both a higher- and a lower-is-better leaderboard, the rate-stat
  classifier, and that `_rank_value` scores a pitching line with the pitcher
  formula — the same line under the hitter formula returns 0 for every pitcher,
  silently, which is why the group is passed in rather than inferred.
- live_checks: schedule, play-by-play, season, and career normalization
  anchored on a known 2024 game and Aaron Judge's 2024 line.
- cache_checks: freshness (a stale row refetches), the whitelist guard, the
  upsert (same key replaced in place, no duplicate row), `log_query` writing
  a row verbatim, the eviction sweep (an aged row deleted, a recent one kept,
  the count reported, and `queries` left alone — the sweep is cache-only), and
  that `_cached_schedule` keys on the real date rather than a literal "today".
  The sweep check backdates `updated_at` directly, since `write_cache` always
  stamps now; the key check stubs `cached_fetch` to capture the key, which keeps
  it offline and off the real database.
- tool_checks: each Claude-facing tool against known 2024 numbers, error
  paths returning `{"error": ...}`, a pitching anchor (Skubal 2024), the
  rate-stat rejection in `compute_pace_projection`, the pitcher pace scaling at
  a partial season (injected, since the completed-season anchor only exercises
  the identity case), and a check that every schema has a matching function.
- filter_checks: date-range presets and a custom window, opponent, and split
  against stable 2024 facts, a lower-is-better leaderboard (2024 ERA leaders),
  every filtered path reporting the player's team the way the season line does
  (so a stated team is retrieved data rather than recall), plus the guards
  (filters not combinable, a custom window needing both ends, unknown values
  erroring cleanly).
- live_feature_checks: today's-games tools, structural and tolerant of
  off-days (assert shape and presence, not fixed numbers that change daily).

## The eval and its two metrics

`run_eval.py` runs the full pipeline over 40 questions across 16 categories and
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
  and the `live` eval category are tolerant by design. `last_30_days` is the one
  date-range preset in the same position: its window is always the real last 30
  days, and a player who hasn't played in it returns a clean error, so the check
  asserts the window metadata rather than a number.
- `run_eval` is a benchmark, not a gate. Treat its scores as a tracked range
  in `eval_history.md`, not a threshold to pass.
- Tool faithfulness is scoped to eval questions whose correct call is
  unambiguous. Scoring arbitrary live queries against inferred intent is future
  work.

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
