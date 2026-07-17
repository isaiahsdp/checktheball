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
"road games") are not treated as checkable facts. `grounding_score` is the
fraction of a claim set that is supported.

## Measured rate

Run: `python scripts/run_eval.py` over an 18-question set (simple, comparison,
leaderboard, split, projection, compound, opinion, and out-of-scope).

| Metric | Value |
|---|---|
| Questions answered | 18 / 18 |
| Claims checked | 58 |
| Claims grounded | 55 |
| Claim-level grounding | **0.948** |
| Mean per-answer score | 0.987 |
| Questions fully grounded | 17 / 18 |

The single sub-1.0 answer was an opinion question ("who had the better season").
All ten of its statistical claims were grounded; the flagged claims were
editorial framing the model added from its own knowledge ("joined the 50-50
club", "no one had done it in MLB history") that no tool returned. That is the
grounding layer working as intended: the numbers are verified, and unverified
context is surfaced rather than trusted.

Claim extraction is a model call, so exact counts vary slightly between runs;
observed claim-level grounding sits around 0.92–0.95.

## What the score does and doesn't capture

- It verifies that the **numbers and named entities** in an answer trace back to
  retrieved data. This is the core promise: stats are not invented.
- It does **not** judge whether the answer is well-reasoned or complete, and the
  numeric check can in principle accept a number that is correct but attached to
  the wrong label. Verification is intentionally strict on numbers and lenient
  on descriptive phrasing.

## Limitations and future work

Grounding checks **answer-to-data faithfulness**, not **query correctness**. It
confirms the answer only states what the retrieved data supports; it does not
confirm the right data was retrieved. If the model calls a tool with the wrong
arguments (wrong season, wrong player), the tool returns correct-but-irrelevant
data, and an answer built from that data still grounds. The tool calls are
returned alongside each answer for transparency, so a reader can see what ran.

A natural extension is a separate **query-faithfulness** check that compares the
structured tool arguments (season, player, stat) against the question's intent.
The deterministic version of that check is preferable to an LLM judge, for the
same reason the grounding cross-check is deterministic: it avoids using one
model to grade another.
