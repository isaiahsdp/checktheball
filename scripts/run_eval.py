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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db, grounding, orchestrator
from sports.mlb import tools

# (category, question): a mix of simple, compound, and opinion questions.
QUESTIONS = [
    ("simple", "How many home runs did Aaron Judge hit in 2024?"),
    ("simple", "What was Shohei Ohtani's batting average in 2024?"),
    ("simple", "How many RBIs did Jose Ramirez have in 2024?"),
    ("simple", "How many stolen bases did Elly De La Cruz have in 2024?"),
    ("comparison", "Who hit more home runs in 2024, Aaron Judge or Shohei Ohtani?"),
    ("comparison", "Who had a higher OPS in 2024, Juan Soto or Bryce Harper?"),
    ("comparison", "Who had more RBIs in 2024, Vladimir Guerrero Jr. or Jose Ramirez?"),
    ("leaderboard", "Who led MLB in home runs in 2024?"),
    ("leaderboard", "Who were the top 5 in stolen bases in 2024?"),
    ("leaderboard", "Who led the majors in RBIs in 2024?"),
    ("split", "How did Aaron Judge hit against left-handed pitching in 2024?"),
    ("split", "How many home runs did Shohei Ohtani hit on the road in 2024?"),
    ("projection", "At his 2024 pace, how many home runs would Aaron Judge hit over a full 162-game season?"),
    ("compound", "Who led MLB in home runs in 2024, and how many RBIs did that player have?"),
    ("compound", "Between the 2024 MLB home run leader and Shohei Ohtani, who had more stolen bases?"),
    ("opinion", "Who had the better season in 2024, Aaron Judge or Shohei Ohtani?"),
    ("date_range", "How many home runs did Aaron Judge hit after the All-Star break in 2024?"),
    ("date_range", "How many home runs did Aaron Judge hit from June 1 to June 30, 2024?"),
    ("date_range", "How many home runs has Manny Machado hit in his last 10 games?"),
    ("date_range", "Who has more home runs over the last 30 days, Manny Machado or Jackson Merrill?"),
    ("opponent", "How does Aaron Judge hit against the Dodgers?"),
    ("opponent", "How many home runs did Aaron Judge hit against the Dodgers in 2024?"),
    ("live", "Who is the top performer in today's games?"),
    ("live", "Who has the most fantasy points today?"),
    ("fantasy", "How many DraftKings fantasy points did Aaron Judge score in 2024?"),
    ("out_of_scope", "Who won the 2024 NBA championship?"),
    ("out_of_scope", "What's the weather in New York today?"),
]


def main() -> int:
    db.init_db()
    results = []

    for category, question in QUESTIONS:
        try:
            answered = orchestrator.answer_question(question, tools)
            graded = grounding.ground_answer(answered["answer"], answered["tool_results"])
            score = graded["grounding_score"]
            db.log_query(question, answered["answer"], answered["tool_calls_made"], score)
            results.append(
                {
                    "category": category,
                    "score": score,
                    "supported": graded["supported_claims"],
                    "total": graded["total_claims"],
                    "tools": len(answered["tool_calls_made"]),
                }
            )
            print(
                f"[{score:.2f}] ({category}) {question}\n"
                f"        {graded['supported_claims']}/{graded['total_claims']} claims grounded, "
                f"{len(answered['tool_calls_made'])} tool call(s)"
            )
        except Exception as exc:
            print(f"[ERR ] ({category}) {question}\n        {exc}")
            results.append({"category": category, "score": None, "supported": 0, "total": 0, "tools": 0})

    scored = [r for r in results if r["score"] is not None]
    mean_score = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
    all_supported = sum(r["supported"] for r in scored)
    all_claims = sum(r["total"] for r in scored)
    claim_rate = all_supported / all_claims if all_claims else 0.0

    print("\n" + "=" * 60)
    print(f"Questions answered:      {len(scored)}/{len(QUESTIONS)}")
    print(f"Total claims checked:    {all_claims}")
    print(f"Claims grounded:         {all_supported}")
    print(f"Claim-level grounding:   {claim_rate:.3f}")
    print(f"Mean per-answer score:   {mean_score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
