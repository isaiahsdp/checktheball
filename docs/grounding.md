# Grounding evaluation

Every answer is verified before it reaches the user. A small model (Haiku)
decomposes the answer into atomic factual claims and the specific data points
each asserts; then a deterministic check confirms each data point appears in the
tool results retrieved that turn. The grounded/not-grounded decision is
mechanical, so we are not using one model to grade another. The small model
only parses language; the verification is a check against the source data.

A claim is **supported** when every value it asserts is found in the retrieved
data: numbers must match numerically, and named entities (players, teams,
awards) must appear in the data. Lowercase descriptive labels ("home runs",
"road games") are not treated as checkable facts. Neither are the conditions a
stat was measured under: calendar dates and spans, and the split a number came
from ("RHP", "at home"). Both name which slice of data produced a number rather
than asserting one, so the extractor leaves them out of a claim's values.
`grounding_score` is the fraction of a claim set that is supported.

## Measured rate

Run `python scripts/run_eval.py` over the question set, which spans simple,
comparison, leaderboard, split, projection, compound, opinion, date_range,
opponent, live, date_awareness, fantasy, pitching, ambiguous, edge_case, and
out-of-scope categories (40 questions). Claim extraction is a model call, so
scores vary between runs; each run appends its summary to `docs/eval_history.md`,
so the rate is tracked as a range rather than a single number.

A recent run (over the question set as it stood at 35 questions, before the two
date_awareness questions were added; see `docs/eval_history.md` for the full
history):

| Metric | Value |
|---|---|
| Questions answered | 35 / 35 |
| Claims checked | 98 |
| Claims grounded | 96 |
| Claim-level grounding | 0.980 |
| Mean per-answer score | 0.994 |
| Tool faithfulness | 27 / 28 |
| Questions fully grounded | 33 / 35 |

The sub-1.0 answers are typically opinion or evaluative questions where the model
adds editorial framing from its own knowledge ("joined the 50-50 club", "no one
had done it in MLB history") that no tool returned. The numbers are still
verified; the unverified editorial context is flagged rather than silently
trusted.

Across the runs logged in `docs/eval_history.md`, claim-level grounding has
ranged from 0.980 to 0.989. Date-range
answers were a source of false negatives: a claim that states the queried year
("...in 2024") failed because the year lived only inside the tool result's date
strings, not as a matchable number. Two changes fixed it. The extractor treats
specific calendar dates and spans as context, not checkable values; and the
date-range tools now put the queried year in their result, so a year the model
correctly states is verifiable instead of missing.

## What the score does and doesn't capture

- It verifies that the **numbers and named entities** in an answer trace back to
  retrieved data. This is the core promise: stats are not invented.
- It does **not** judge whether the answer is well-reasoned or complete, and the
  numeric check can in principle accept a number that is correct but attached to
  the wrong label. Verification is intentionally strict on numbers and lenient
  on descriptive phrasing.
- It does **not** accept arithmetic the model did itself. A number is grounded
  only if a tool returned it. A correct rate ("18.47% of his at-bats") whose
  inputs were both retrieved still fails on its own, because the rate itself was
  never retrieved.

## Why arithmetic goes through a tool

Grounding once had a fallback that accepted a number equal to the division of
two retrieved numbers, on the theory that a rate the model computed from real
inputs was safe. It was not. The check tried every ordered pair, so the set of
accepted values grew with the square of the retrieved data while the space of
plausible claims stayed fixed. Measured against a five-row leaderboard, it
accepted **48% of all percentages between 0 and 100**, so an invented number was
about as likely to pass as a real one.

The two failure modes are not symmetric. A correct answer marked ungrounded is
visible and cheap: it shows up in `scripts/review_queries.py` and points at a
real gap. A fabricated number marked grounded is invisible and defeats the
point of the layer. Verification should lean toward the first.

So the fallback is gone, and `core/compute.py` replaces it. The model calls
`compute` for a rate, difference, or total; the arithmetic runs in Python, and
every operand is checked against the numbers already retrieved this turn, so
the model can combine verified numbers but never introduce one. The result
lands in the tool output and grounds by direct match like any other lookup.
Division also returns the percentage at three precisions (`as_percent`,
`as_percent_1dp`, `as_percent_whole`), because which one an answer states varies
with the number. Returning all three means the rounded form actually written is
a number a tool really returned, so the verifier needs no rounding tolerance of
its own.

This is the same move as `gap_from_leader` in `get_top_performers` and as
`sports/mlb/fantasy.py`: compute deterministically in code, then verify the
model reported it. Where a derived value is predictable for a given tool,
precomputing it there is still preferred over a `compute` round trip.

## Limitations and future work

Grounding checks **answer-to-data faithfulness**, not **query correctness**. It
confirms the answer only states what the retrieved data supports; it does not
confirm the right data was retrieved. If the model calls a tool with the wrong
arguments (wrong season, wrong player), the tool returns correct-but-irrelevant
data, and an answer built from that data still grounds. The tool calls are
returned alongside each answer for transparency, so a reader can see what ran.

A scoped, deterministic **query-faithfulness** check now runs in
`scripts/run_eval.py` (`check_tool_faithfulness`). Eval questions whose correct
tool call is unambiguous carry an expected tool and key arguments; the run
records whether the model used them and reports a tool-faithfulness rate
separate from grounding. It is deterministic (no model judge), for the same
reason the grounding cross-check is. It already surfaced a real case (an
ambiguous surname) where grounding scored 1.0 but the model made no lookup.

This is a first version tied to annotated eval questions. A general-purpose
version that scores arbitrary live queries against inferred intent, rather than
pre-annotated ones, is still future work.
