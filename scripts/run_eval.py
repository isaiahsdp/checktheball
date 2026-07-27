"""Run the evaluation question set through the full pipeline and report the
measured grounding rate.

For each question: the orchestrator answers it (live Sonnet + tools), the
grounding layer scores the answer against the retrieved data (live Haiku), and
the result is logged to the ``queries`` table. Makes live API calls.

Run from the repo root:

    python scripts/run_eval.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db, grounding, orchestrator
from sports.mlb import tools

# Run-level summary log (one row per run), separate from the per-question
# db.log_query rows. Tracks the grounding rate as a range across runs.
_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "eval_history.md"
)
_HISTORY_HEADER = (
    "# Eval run history\n\n"
    "One row per `scripts/run_eval.py` run. Claim extraction is a model call, so\n"
    "scores vary between runs; read this as a range, not a single number.\n\n"
    "| Timestamp (UTC) | Questions | Claims | Grounded | Claim-level | Mean/answer | Tool faithful | By category (mean/answer) |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def check_tool_faithfulness(expected: dict | None, tool_calls_made: list[dict]) -> bool | None:
    """Did the model use the right tool with the right key args?

    Deterministic, no model judge. Returns None when ``expected`` is None (not
    checked). Otherwise True if any call matches the expected tool name and every
    listed arg (exact match on just those args), False otherwise.
    """
    if expected is None:
        return None
    want_args = expected.get("args", {})
    for call in tool_calls_made:
        if call.get("name") != expected["tool"]:
            continue
        call_input = call.get("input", {})
        if all(call_input.get(k) == v for k, v in want_args.items()):
            return True
    return False


def append_history(
    results: list[dict],
    answered: int,
    total: int,
    claims_checked: int,
    claims_grounded: int,
    claim_rate: float,
    mean_score: float,
    faith_correct: int,
    faith_checked: int,
) -> str:
    """Append one summary row for this run to docs/eval_history.md; return the row."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r["score"] is not None:
            by_cat[r["category"]].append(r["score"])
    cat_breakdown = "; ".join(f"{c} {sum(v) / len(v):.2f}" for c, v in sorted(by_cat.items()))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    row = (
        f"| {ts} | {answered}/{total} | {claims_checked} | {claims_grounded} "
        f"| {claim_rate:.3f} | {mean_score:.3f} | {faith_correct}/{faith_checked} | {cat_breakdown} |\n"
    )

    write_header = not os.path.exists(_HISTORY_PATH)
    with open(_HISTORY_PATH, "a", encoding="utf-8") as fh:
        if write_header:
            fh.write(_HISTORY_HEADER)
        fh.write(row)
    return row

# (category, question, expected). ``expected`` is None (tool call not checked)
# or {"tool": name, "args": {...}} naming the correct tool and only the 1-2
# arguments that actually disambiguate the right lookup, not a full match. Drives
# the deterministic tool-faithfulness check, which is separate from grounding.
# None where there is no single unambiguous correct call (compound, opinion,
# live, out_of_scope).
QUESTIONS = [
    ("simple", "How many home runs did Aaron Judge hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("simple", "What was Shohei Ohtani's batting average in 2024?", {"tool": "get_player_stat", "args": {"stat": "avg", "season": 2024}}),
    ("simple", "How many RBIs did Jose Ramirez have in 2024?", {"tool": "get_player_stat", "args": {"stat": "rbi", "season": 2024}}),
    ("simple", "How many stolen bases did Elly De La Cruz have in 2024?", {"tool": "get_player_stat", "args": {"stat": "stolenBases", "season": 2024}}),
    ("comparison", "Who hit more home runs in 2024, Aaron Judge or Shohei Ohtani?", {"tool": "compare_players", "args": {"stat": "homeRuns", "season": 2024}}),
    ("comparison", "Who had a higher OPS in 2024, Juan Soto or Bryce Harper?", {"tool": "compare_players", "args": {"stat": "ops", "season": 2024}}),
    ("comparison", "Who had more RBIs in 2024, Vladimir Guerrero Jr. or Jose Ramirez?", {"tool": "compare_players", "args": {"stat": "rbi", "season": 2024}}),
    ("leaderboard", "Who led MLB in home runs in 2024?", {"tool": "get_top_performers", "args": {"stat": "homeRuns", "season": 2024}}),
    ("leaderboard", "Who were the top 5 in stolen bases in 2024?", {"tool": "get_top_performers", "args": {"stat": "stolenBases", "season": 2024}}),
    # RBI leaderboard stat key is ambiguous (rbi vs runsBattedIn); check tool + season only.
    ("leaderboard", "Who led the majors in RBIs in 2024?", {"tool": "get_top_performers", "args": {"season": 2024}}),
    ("split", "How did Aaron Judge hit against left-handed pitching in 2024?", {"tool": "get_player_stat", "args": {"split": "vs_left", "season": 2024}}),
    ("split", "How many home runs did Shohei Ohtani hit on the road in 2024?", {"tool": "get_player_stat", "args": {"split": "away", "stat": "homeRuns"}}),
    ("projection", "At his 2024 pace, how many home runs would Aaron Judge hit over a full 162-game season?", {"tool": "compute_pace_projection", "args": {"stat": "homeRuns", "season": 2024}}),
    ("compound", "Who led MLB in home runs in 2024, and how many RBIs did that player have?", None),
    ("compound", "Between the 2024 MLB home run leader and Shohei Ohtani, who had more stolen bases?", None),
    ("opinion", "Who had the better season in 2024, Aaron Judge or Shohei Ohtani?", None),
    ("date_range", "How many home runs did Aaron Judge hit after the All-Star break in 2024?", {"tool": "get_player_stat", "args": {"date_range": "since_allstar", "stat": "homeRuns"}}),
    ("date_range", "How many home runs did Aaron Judge hit from June 1 to June 30, 2024?", {"tool": "get_player_stat", "args": {"start_date": "2024-06-01", "end_date": "2024-06-30"}}),
    ("date_range", "How many home runs has Manny Machado hit in his last 10 games?", {"tool": "get_player_stat", "args": {"date_range": "last_10_games", "stat": "homeRuns"}}),
    ("date_range", "Who has more home runs over the last 30 days, Manny Machado or Jackson Merrill?", {"tool": "compare_players", "args": {"date_range": "last_30_days", "stat": "homeRuns"}}),
    ("opponent", "How does Aaron Judge hit against the Dodgers?", {"tool": "get_player_stat", "args": {"opponent": "Dodgers"}}),
    ("opponent", "How many home runs did Aaron Judge hit against the Dodgers in 2024?", {"tool": "get_player_stat", "args": {"opponent": "Dodgers", "stat": "homeRuns", "season": 2024}}),
    ("live", "Who is the top performer in today's games?", None),
    ("live", "Who has the most fantasy points today?", None),
    ("live", "Who has the most strikeouts in today's games?", None),
    # Date awareness: no explicit year, so the correct season is whatever "this
    # season"/"current" resolves to today. expected stays None on purpose:
    # check_tool_faithfulness only matches fixed arg values, and the right season
    # here changes every year, so any hardcoded season would silently rot. What
    # these probe is live behavior: does the model resolve to the real current
    # season and query it, rather than defaulting to a training-era year or
    # claiming the data doesn't exist yet.
    ("date_awareness", "How many home runs does Aaron Judge have this season?", None),
    ("date_awareness", "Who leads MLB in home runs this current season?", None),
    ("fantasy", "How many DraftKings fantasy points did Aaron Judge score in 2024?", {"tool": "get_fantasy_points", "args": {"season": 2024}}),
    ("fantasy", "How many fantasy points has Rafael Devers scored against the Angels in 2024?", {"tool": "get_fantasy_points", "args": {"opponent": "Angels", "season": 2024}}),
    ("out_of_scope", "Who won the 2024 NBA championship?", None),
    ("out_of_scope", "What's the weather in New York today?", None),
    # Pitching (the rest of the set is hitting-only).
    ("pitching", "How many strikeouts did Tarik Skubal have in 2024?", {"tool": "get_player_stat", "args": {"stat": "strikeOuts", "group": "pitching","season": 2024}}),
    ("pitching", "Who had more strikeouts in 2024, Tarik Skubal or Chris Sale?", {"tool": "compare_players", "args": {"stat": "strikeOuts", "group": "pitching"}}),
    # Pitching leaderboard stat key is ambiguous; check tool + group + season.
    ("pitching", "Who led MLB in strikeouts in 2024?", {"tool": "get_top_performers", "args": {"group": "pitching", "season": 2024}}),
    ("pitching", "How many DraftKings fantasy points did Tarik Skubal score in 2024?", {"tool": "get_fantasy_points", "args": {"group": "pitching", "season": 2024}}),
    # Ambiguous: annotated on purpose. A faithfulness miss here is the signal.
    ("ambiguous", "How many home runs did Hernandez hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("ambiguous", "How many home runs does Aaron Judge have?", {"tool": "get_player_stat", "args": {"stat": "homeRuns"}}),
    ("ambiguous", "What is Shohei Ohtani's average?", {"tool": "get_player_stat", "args": {"stat": "avg"}}),
    # Edge cases: correct tool is still clear despite the messy scenario.
    ("edge_case", "How many home runs did Jazz Chisholm Jr. hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("edge_case", "What was Julio Rodríguez's batting average in 2024?", {"tool": "get_player_stat", "args": {"stat": "avg", "season": 2024}}),
]


def main() -> int:
    db.init_db()
    results = []

    for category, question, expected in QUESTIONS:
        try:
            answered = orchestrator.answer_question(question, tools)
            graded = grounding.ground_answer(answered["answer"], answered["tool_results"])
            score = graded["grounding_score"]
            calls = answered["tool_calls_made"]
            faithful = check_tool_faithfulness(expected, calls)
            db.log_query(question, answered["answer"], calls, score)
            results.append(
                {
                    "category": category,
                    "score": score,
                    "supported": graded["supported_claims"],
                    "total": graded["total_claims"],
                    "tools": len(calls),
                    "faithful": faithful,
                }
            )
            faith_tag = {True: "faith:ok", False: "faith:MISS", None: "faith:n/a"}[faithful]
            print(
                f"[{score:.2f}] ({category}) {question}\n"
                f"        {graded['supported_claims']}/{graded['total_claims']} claims grounded, "
                f"{len(calls)} tool call(s), {faith_tag}"
            )
            if faithful is False:
                actual = [{"name": c["name"], "input": c["input"]} for c in calls] or "no tool calls"
                print(f"        expected: {expected['tool']} {expected.get('args', {})}")
                print(f"        actual:   {actual}")
        except Exception as exc:
            print(f"[ERR ] ({category}) {question}\n        {exc}")
            results.append({"category": category, "score": None, "supported": 0, "total": 0, "tools": 0, "faithful": None})

    scored = [r for r in results if r["score"] is not None]
    mean_score = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
    all_supported = sum(r["supported"] for r in scored)
    all_claims = sum(r["total"] for r in scored)
    claim_rate = all_supported / all_claims if all_claims else 0.0

    faith_checked_rows = [r for r in results if r["faithful"] is not None]
    faith_correct = sum(1 for r in faith_checked_rows if r["faithful"])
    faith_checked = len(faith_checked_rows)
    faith_na = len(results) - faith_checked

    print("\n" + "=" * 60)
    print(f"Questions answered:      {len(scored)}/{len(QUESTIONS)}")
    print(f"Total claims checked:    {all_claims}")
    print(f"Claims grounded:         {all_supported}")
    print(f"Claim-level grounding:   {claim_rate:.3f}")
    print(f"Mean per-answer score:   {mean_score:.3f}")
    print(f"Tool faithfulness:       {faith_correct}/{faith_checked} correct ({faith_checked} checked, {faith_na} not applicable)")

    row = append_history(
        results, len(scored), len(QUESTIONS), all_claims, all_supported, claim_rate, mean_score, faith_correct, faith_checked
    )
    print(f"\nLogged to {os.path.relpath(_HISTORY_PATH)}:\n{row.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
