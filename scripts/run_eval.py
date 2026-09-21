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
# live, out_of_scope), or where the right answer is to decline.
#
# 100 questions across 16 categories: 6 each, and 7 for the four categories the
# production query log shows failing (split, opponent, fantasy, pitching).
# Six is the floor for a category mean to mean anything -- below that a single
# miss swings the score by half or more, which is what made the old n=1
# projection and opinion columns swing between 1.00 and 0.00 run to run.
#
# Everything anchors on 2024, a finished season whose numbers cannot change, so
# scores stay comparable across runs. The two exceptions are deliberate: `live`
# and `date_awareness` have to be relative to today, which is the whole point of
# those categories. Do not write "this season" anywhere else -- it rots.
QUESTIONS = [
    # --- simple (6): one player, one stat, one lookup ---
    ("simple", "How many home runs did Aaron Judge hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("simple", "What was Shohei Ohtani's batting average in 2024?", {"tool": "get_player_stat", "args": {"stat": "avg", "season": 2024}}),
    ("simple", "How many RBIs did Jose Ramirez have in 2024?", {"tool": "get_player_stat", "args": {"stat": "rbi", "season": 2024}}),
    ("simple", "How many stolen bases did Elly De La Cruz have in 2024?", {"tool": "get_player_stat", "args": {"stat": "stolenBases", "season": 2024}}),
    ("simple", "How many hits did Bobby Witt Jr. have in 2024?", {"tool": "get_player_stat", "args": {"stat": "hits", "season": 2024}}),
    # Doubles key is not in the tool's stat hint; check tool + season only.
    ("simple", "How many doubles did Marcell Ozuna hit in 2024?", {"tool": "get_player_stat", "args": {"season": 2024}}),

    # --- comparison (6): two players, one stat ---
    ("comparison", "Who hit more home runs in 2024, Aaron Judge or Shohei Ohtani?", {"tool": "compare_players", "args": {"stat": "homeRuns", "season": 2024}}),
    ("comparison", "Who had a higher OPS in 2024, Juan Soto or Bryce Harper?", {"tool": "compare_players", "args": {"stat": "ops", "season": 2024}}),
    ("comparison", "Who had more RBIs in 2024, Vladimir Guerrero Jr. or Jose Ramirez?", {"tool": "compare_players", "args": {"stat": "rbi", "season": 2024}}),
    ("comparison", "Who had more stolen bases in 2024, Elly De La Cruz or Bobby Witt Jr.?", {"tool": "compare_players", "args": {"stat": "stolenBases", "season": 2024}}),
    ("comparison", "Who hit for a higher average in 2024, Luis Arraez or Bobby Witt Jr.?", {"tool": "compare_players", "args": {"stat": "avg", "season": 2024}}),
    ("comparison", "Between Yordan Alvarez and Kyle Tucker, who had more RBIs in 2024?", {"tool": "compare_players", "args": {"stat": "rbi", "season": 2024}}),

    # --- leaderboard (6): league leaders for a season ---
    ("leaderboard", "Who led MLB in home runs in 2024?", {"tool": "get_top_performers", "args": {"stat": "homeRuns", "season": 2024}}),
    ("leaderboard", "Who were the top 5 in stolen bases in 2024?", {"tool": "get_top_performers", "args": {"stat": "stolenBases", "season": 2024}}),
    # Leaderboard stat keys use MLB leader names (runsBattedIn, battingAverage),
    # which differ from the per-player keys; check tool + season only.
    ("leaderboard", "Who led the majors in RBIs in 2024?", {"tool": "get_top_performers", "args": {"season": 2024}}),
    ("leaderboard", "Who led MLB in batting average in 2024?", {"tool": "get_top_performers", "args": {"season": 2024}}),
    ("leaderboard", "Who were the top 3 in OPS in 2024?", {"tool": "get_top_performers", "args": {"season": 2024, "limit": 3}}),
    ("leaderboard", "Who led the majors in doubles in 2024?", {"tool": "get_top_performers", "args": {"season": 2024}}),

    # --- split (7): home/away and platoon splits. Weighted up because the live
    # query log shows this shape failing more than any other: seven of sixteen
    # real "does X strike out against righties" questions scored below 1.0.
    ("split", "How did Aaron Judge hit against left-handed pitching in 2024?", {"tool": "get_player_stat", "args": {"split": "vs_left", "season": 2024}}),
    ("split", "How many home runs did Shohei Ohtani hit on the road in 2024?", {"tool": "get_player_stat", "args": {"split": "away", "stat": "homeRuns"}}),
    ("split", "Does Anthony Volpe strike out a lot against right-handed pitching?", {"tool": "get_player_stat", "args": {"split": "vs_right", "stat": "strikeOuts"}}),
    ("split", "How did Rafael Devers hit at home in 2024?", {"tool": "get_player_stat", "args": {"split": "home", "season": 2024}}),
    ("split", "What was Bobby Witt Jr.'s batting average on the road in 2024?", {"tool": "get_player_stat", "args": {"split": "away", "stat": "avg", "season": 2024}}),
    ("split", "How many home runs did Kyle Schwarber hit against right-handed pitching in 2024?", {"tool": "get_player_stat", "args": {"split": "vs_right", "stat": "homeRuns", "season": 2024}}),
    # Two lookups to answer, so no single call is correct.
    ("split", "Does Jose Ramirez hit better against lefties or righties?", None),

    # --- projection (6): full-season pace ---
    ("projection", "At his 2024 pace, how many home runs would Aaron Judge hit over a full 162-game season?", {"tool": "compute_pace_projection", "args": {"stat": "homeRuns", "season": 2024}}),
    ("projection", "At his 2024 pace, how many hits would Bobby Witt Jr. finish a full season with?", {"tool": "compute_pace_projection", "args": {"stat": "hits", "season": 2024}}),
    ("projection", "At his 2024 pace, how many RBIs would Jose Ramirez have over 162 games?", {"tool": "compute_pace_projection", "args": {"stat": "rbi", "season": 2024}}),
    ("projection", "Is Aaron Judge on pace for 60 home runs this season?", {"tool": "compute_pace_projection", "args": {"stat": "homeRuns"}}),
    ("projection", "How many strikeouts is Tarik Skubal on pace for over a full season?", {"tool": "compute_pace_projection", "args": {"stat": "strikeOuts", "group": "pitching"}}),
    # Guard: the tool rejects rate-stat projections, so the right behavior is to
    # decline rather than project an average. A confident number here is a bug.
    ("projection", "What batting average is Luis Arraez on pace for this season?", None),

    # --- compound (6): several lookups combined into one answer ---
    ("compound", "Who led MLB in home runs in 2024, and how many RBIs did that player have?", None),
    ("compound", "Between the 2024 MLB home run leader and Shohei Ohtani, who had more stolen bases?", None),
    ("compound", "Who led MLB in home runs in 2024, and what was that player's batting average?", None),
    ("compound", "Compare Aaron Judge and Shohei Ohtani in 2024 on home runs, RBIs, and OPS.", None),
    ("compound", "Who had more strikeouts in 2024, the AL Cy Young winner or Chris Sale?", None),
    # Exercises the compute tool: the difference must come back as a tool result.
    ("compound", "How many more home runs did Aaron Judge hit than Juan Soto in 2024?", None),

    # --- opinion (6): judgment grounded in retrieved numbers ---
    ("opinion", "Who had the better season in 2024, Aaron Judge or Shohei Ohtani?", None),
    ("opinion", "Who was the better hitter in 2024, Bobby Witt Jr. or Gunnar Henderson?", None),
    ("opinion", "Was Tarik Skubal's 2024 season better than Chris Sale's?", None),
    ("opinion", "Who had the more valuable 2024 season, Aaron Judge or Bobby Witt Jr.?", None),
    ("opinion", "Is Shohei Ohtani the best player in baseball?", None),
    ("opinion", "Which 2024 rookie had the best season?", None),

    # --- date_range (6): preset windows and custom start/end ---
    ("date_range", "How many home runs did Aaron Judge hit after the All-Star break in 2024?", {"tool": "get_player_stat", "args": {"date_range": "since_allstar", "stat": "homeRuns"}}),
    ("date_range", "How many home runs did Aaron Judge hit from June 1 to June 30, 2024?", {"tool": "get_player_stat", "args": {"start_date": "2024-06-01", "end_date": "2024-06-30"}}),
    ("date_range", "How many home runs has Manny Machado hit in his last 10 games?", {"tool": "get_player_stat", "args": {"date_range": "last_10_games", "stat": "homeRuns"}}),
    ("date_range", "Who has more home runs over the last 30 days, Manny Machado or Jackson Merrill?", {"tool": "compare_players", "args": {"date_range": "last_30_days", "stat": "homeRuns"}}),
    ("date_range", "How many RBIs did Jose Ramirez have from July 1 to July 31, 2024?", {"tool": "get_player_stat", "args": {"start_date": "2024-07-01", "end_date": "2024-07-31"}}),
    ("date_range", "How has Aaron Judge hit over his last 10 games?", {"tool": "get_player_stat", "args": {"date_range": "last_10_games"}}),

    # --- opponent (7): head-to-head splits. Weighted up because this is where
    # the vsTeam/vsTeamTotal bug lived -- a stale anchor caught it once already.
    ("opponent", "How does Aaron Judge hit against the Dodgers?", {"tool": "get_player_stat", "args": {"opponent": "Dodgers"}}),
    ("opponent", "How many home runs did Aaron Judge hit against the Dodgers in 2024?", {"tool": "get_player_stat", "args": {"opponent": "Dodgers", "stat": "homeRuns", "season": 2024}}),
    ("opponent", "How many home runs has Shohei Ohtani hit against the Giants?", {"tool": "get_player_stat", "args": {"opponent": "Giants", "stat": "homeRuns"}}),
    ("opponent", "How does Mookie Betts hit against the Padres?", {"tool": "get_player_stat", "args": {"opponent": "Padres"}}),
    ("opponent", "What was Juan Soto's batting average against the Orioles in 2024?", {"tool": "get_player_stat", "args": {"opponent": "Orioles", "stat": "avg", "season": 2024}}),
    ("opponent", "How many strikeouts did Tarik Skubal have against the Guardians in 2024?", {"tool": "get_player_stat", "args": {"opponent": "Guardians", "stat": "strikeOuts", "group": "pitching", "season": 2024}}),
    ("opponent", "Does Rafael Devers hit well against the Yankees?", {"tool": "get_player_stat", "args": {"opponent": "Yankees"}}),

    # --- live (6): today's games. Non-deterministic by nature, so expected is
    # always None -- what these probe is that the model reaches for today's data
    # at all rather than answering from a season line.
    ("live", "Who is the top performer in today's games?", None),
    ("live", "Who has the most fantasy points today?", None),
    ("live", "Who has the most strikeouts in today's games?", None),
    ("live", "Who has the most hits in today's games?", None),
    ("live", "Which pitcher has the most strikeouts today?", None),
    ("live", "How is the Yankees game going today?", None),

    # --- date_awareness (6) ---
    # No explicit year, so the correct season is whatever "this season"/"current"
    # resolves to today. expected stays None on purpose: check_tool_faithfulness
    # only matches fixed arg values, and the right season here changes every
    # year, so any hardcoded season would silently rot. What these probe is live
    # behavior: does the model resolve to the real current season and query it,
    # rather than defaulting to a training-era year or claiming the data doesn't
    # exist yet.
    ("date_awareness", "How many home runs does Aaron Judge have this season?", None),
    ("date_awareness", "Who leads MLB in home runs this current season?", None),
    ("date_awareness", "How many RBIs does Jose Ramirez have this season?", None),
    ("date_awareness", "Who is leading the majors in stolen bases right now?", None),
    ("date_awareness", "What is Tarik Skubal's ERA this year?", None),
    ("date_awareness", "How many home runs does Shohei Ohtani have so far in the current season?", None),

    # --- fantasy (7): DraftKings scoring. Weighted up because it is the weakest
    # tool in production -- 0.333 mean across real questions, including two 0.00s
    # on "fantasy points today" for a player not in that day's games.
    ("fantasy", "How many DraftKings fantasy points did Aaron Judge score in 2024?", {"tool": "get_fantasy_points", "args": {"season": 2024}}),
    ("fantasy", "How many fantasy points has Rafael Devers scored against the Angels in 2024?", {"tool": "get_fantasy_points", "args": {"opponent": "Angels", "season": 2024}}),
    ("fantasy", "How many DraftKings fantasy points did Bobby Witt Jr. score in 2024?", {"tool": "get_fantasy_points", "args": {"season": 2024}}),
    ("fantasy", "How many fantasy points did Shohei Ohtani score against the Padres in 2024?", {"tool": "get_fantasy_points", "args": {"opponent": "Padres", "season": 2024}}),
    ("fantasy", "How many fantasy points did Chris Sale score as a pitcher in 2024?", {"tool": "get_fantasy_points", "args": {"group": "pitching", "season": 2024}}),
    ("fantasy", "What were Juan Soto's fantasy points on June 15, 2024?", {"tool": "get_fantasy_points", "args": {"date": "2024-06-15"}}),
    # Taken from the live query log, where it scored 0.00 twice: the player was
    # not in that day's games, and the refusal cited a date it never retrieved.
    ("fantasy", "How many fantasy points did Jung Hoo Lee score today?", {"tool": "get_fantasy_points", "args": {"player": "Jung Hoo Lee"}}),

    # --- out_of_scope (6): no tool covers these; saying so is the right answer.
    # The last three are real questions from the production log that returned an
    # answer with no tool call at all.
    ("out_of_scope", "Who won the 2024 NBA championship?", None),
    ("out_of_scope", "What's the weather in New York today?", None),
    ("out_of_scope", "Who won the Super Bowl in 2024?", None),
    ("out_of_scope", "What are the most hitter-friendly ballparks in August?", None),
    ("out_of_scope", "What is the longest streak of consecutive strikeouts to start a game?", None),
    ("out_of_scope", "Should I start Aaron Judge or Juan Soto in my fantasy lineup tonight?", None),

    # --- pitching (7): the rest of the set is hitting-only, and group='pitching'
    # is a separate code path through every tool.
    ("pitching", "How many strikeouts did Tarik Skubal have in 2024?", {"tool": "get_player_stat", "args": {"stat": "strikeOuts", "group": "pitching","season": 2024}}),
    ("pitching", "Who had more strikeouts in 2024, Tarik Skubal or Chris Sale?", {"tool": "compare_players", "args": {"stat": "strikeOuts", "group": "pitching"}}),
    # Pitching leaderboard stat key is ambiguous; check tool + group + season.
    ("pitching", "Who led MLB in strikeouts in 2024?", {"tool": "get_top_performers", "args": {"group": "pitching", "season": 2024}}),
    ("pitching", "How many DraftKings fantasy points did Tarik Skubal score in 2024?", {"tool": "get_fantasy_points", "args": {"group": "pitching", "season": 2024}}),
    ("pitching", "What was Chris Sale's ERA in 2024?", {"tool": "get_player_stat", "args": {"stat": "era", "group": "pitching", "season": 2024}}),
    ("pitching", "How many wins did Tarik Skubal have in 2024?", {"tool": "get_player_stat", "args": {"stat": "wins", "group": "pitching", "season": 2024}}),
    ("pitching", "Who had the lowest ERA in MLB in 2024?", {"tool": "get_top_performers", "args": {"group": "pitching", "season": 2024}}),

    # --- ambiguous (6): annotated on purpose. A faithfulness miss here is the
    # signal -- these are underspecified, and guessing is worse than asking.
    ("ambiguous", "How many home runs did Hernandez hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("ambiguous", "How many home runs does Aaron Judge have?", {"tool": "get_player_stat", "args": {"stat": "homeRuns"}}),
    ("ambiguous", "What is Shohei Ohtani's average?", {"tool": "get_player_stat", "args": {"stat": "avg"}}),
    ("ambiguous", "How many home runs did Rodriguez hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("ambiguous", "What are Ohtani's numbers?", None),
    # A bare name with no question, straight from the production log.
    ("ambiguous", "Spencer Jones", None),

    # --- edge_case (6): messy scenarios where the correct tool is still clear.
    ("edge_case", "How many home runs did Jazz Chisholm Jr. hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("edge_case", "What was Julio Rodríguez's batting average in 2024?", {"tool": "get_player_stat", "args": {"stat": "avg", "season": 2024}}),
    ("edge_case", "How many home runs did Ronald Acuña Jr. hit in 2024?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2024}}),
    ("edge_case", "What was Luis Robert Jr.'s batting average in 2024?", {"tool": "get_player_stat", "args": {"stat": "avg", "season": 2024}}),
    # Seasons with no possible data. The lookup should run and come back empty;
    # inventing a number for either is the failure being watched for.
    ("edge_case", "How many home runs did Aaron Judge hit in 1985?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 1985}}),
    ("edge_case", "How many home runs did Mike Trout hit in 2035?", {"tool": "get_player_stat", "args": {"stat": "homeRuns", "season": 2035}}),
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
