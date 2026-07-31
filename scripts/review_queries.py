"""Read the query log and report what it says about answer quality.

Read-only: this never writes to the database. It answers the questions the log
exists to answer but nothing currently asks it -- which answers grounded badly,
which tools they went through, and which questions belong in the eval set.

Run from the repo root:

    python scripts/review_queries.py
    python scripts/review_queries.py --threshold 0.9 --limit 40
    python scripts/review_queries.py --since 2026-07-20

Honors CHECKTHEBALL_DB the same way the app does.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db

# A perfect answer scores 1.0, so anything under it has at least one claim that
# did not trace back to the retrieved data. Overridable: raising it widens the
# reading list, lowering it narrows to the worst.
DEFAULT_THRESHOLD = 1.0
DEFAULT_LIMIT = 25


def _rows(db_path: str, since: str | None) -> list[sqlite3.Row]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        if since:
            return con.execute(
                "SELECT * FROM queries WHERE created_at >= ? ORDER BY id", (since,)
            ).fetchall()
        return con.execute("SELECT * FROM queries ORDER BY id").fetchall()
    finally:
        con.close()


def _calls(row: sqlite3.Row) -> list[dict]:
    # tool_calls is whatever log_query serialized; a malformed row shouldn't
    # take down a report over 269 good ones.
    try:
        parsed = json.loads(row["tool_calls"])
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def scores_section(rows: list[sqlite3.Row]) -> None:
    print("\n=== Eval scores ===\n")
    scored = [r["grounding_score"] for r in rows if r["grounding_score"] is not None]
    ungraded = len(rows) - len(scored)
    print(f"Questions logged:  {len(rows)}")
    print(f"Scored:            {len(scored)}" + (f"  ({ungraded} ungraded)" if ungraded else ""))
    if not scored:
        return
    print(f"Span:              {rows[0]['created_at'][:10]} to {rows[-1]['created_at'][:10]}")
    print(f"Mean score:        {_fmt(_mean(scored))}")
    perfect = sum(1 for s in scored if s >= 1.0)
    print(f"Perfect (1.0):     {perfect}/{len(scored)}  ({perfect / len(scored):.0%})")

    # Buckets rather than a histogram: the shape of the tail is the point, and
    # a mean alone hides whether misses are near-misses or total failures.
    buckets = [(0.0, 0.5, "0.00-0.49"), (0.5, 0.8, "0.50-0.79"), (0.8, 1.0, "0.80-0.99"), (1.0, 1.1, "1.00")]
    print("\nDistribution")
    for low, high, label in buckets:
        n = sum(1 for s in scored if low <= s < high)
        print(f"  {label:<10} {n:>4}  {'#' * round(40 * n / len(scored))}")

    by_day: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if r["grounding_score"] is not None:
            by_day[r["created_at"][:10]].append(r["grounding_score"])
    print("\nBy day")
    for day in sorted(by_day):
        day_scores = by_day[day]
        print(f"  {day}  n={len(day_scores):<4} mean={_fmt(_mean(day_scores))}")
    # Nothing records which model or prompt produced a row, so a change in this
    # column can't be attributed to a change you made. Read it as description.
    print("\n  (trend is descriptive only: rows carry no model or prompt version)")


def failures_section(rows: list[sqlite3.Row], threshold: float, limit: int) -> None:
    print(f"\n\n=== Failures to read (score < {threshold}) ===\n")
    bad = [r for r in rows if r["grounding_score"] is not None and r["grounding_score"] < threshold]
    bad.sort(key=lambda r: r["grounding_score"])
    if not bad:
        print("Nothing below the threshold.")
        return
    print(f"{len(bad)} below threshold; showing the {min(limit, len(bad))} worst.\n")
    for r in bad[:limit]:
        print(f"[{r['grounding_score']:.2f}] #{r['id']}  {r['created_at'][:16]}")
        print(f"  Q: {r['question']}")
        print(f"  A: {' '.join(r['answer'].split())[:260]}")
        calls = _calls(r)
        if not calls:
            print("  tools: (none called)")
        for c in calls:
            print(f"  tool: {c.get('name')}  {json.dumps(c.get('input', {}), sort_keys=True)}")
        print()


def tools_section(rows: list[sqlite3.Row]) -> None:
    print("\n=== Tool usage ===\n")
    calls_per_tool: collections.Counter = collections.Counter()
    questions_per_tool: collections.Counter = collections.Counter()
    scores_per_tool: dict[str, list[float]] = collections.defaultdict(list)

    for r in rows:
        names = [c.get("name") for c in _calls(r) if c.get("name")]
        for name in names:
            calls_per_tool[name] += 1
        for name in set(names):
            questions_per_tool[name] += 1
            if r["grounding_score"] is not None:
                scores_per_tool[name].append(r["grounding_score"])

    if not calls_per_tool:
        print("No tool calls logged.")
        return

    print(f"{'tool':<26} {'calls':>6} {'questions':>10} {'mean score':>11}")
    print(f"{'-' * 26} {'-' * 6} {'-' * 10} {'-' * 11}")
    for name, n in calls_per_tool.most_common():
        mean = _mean(scores_per_tool[name])
        print(f"{name:<26} {n:>6} {questions_per_tool[name]:>10} {_fmt(mean):>11}")
    # An answer can use several tools, so a low mean here points at a tool worth
    # looking into; it does not establish that the tool caused the misses.
    print("\n  (mean = score of answers that used the tool, not the tool's own accuracy)")


def missing_tool_section(rows: list[sqlite3.Row], limit: int) -> None:
    print("\n\n=== Missing-tool signals ===\n")
    no_calls = [r for r in rows if not _calls(r)]
    print(f"Questions answered with no tool call: {len(no_calls)}")
    if no_calls:
        # Either the model declined to guess (good) or nothing covered the ask
        # (a gap). Both are worth reading; only the transcript separates them.
        print("The model either declined to guess or had nothing that fit.\n")
        for r in no_calls[:limit]:
            score = _fmt(r["grounding_score"], 2)
            print(f"  [{score}] #{r['id']}  {r['question']}")

    repeats = [r for r in rows if len(_calls(r)) >= 4]
    if repeats:
        print(f"\nQuestions needing 4+ tool calls: {len(repeats)}")
        print("Many calls for one question can mean no single tool covers it.\n")
        for r in repeats[:limit]:
            names = ", ".join(sorted({c.get("name", "?") for c in _calls(r)}))
            print(f"  [{_fmt(r['grounding_score'], 2)}] #{r['id']}  {len(_calls(r))} calls ({names})")
            print(f"        {r['question']}")


def eval_candidates_section(rows: list[sqlite3.Row], threshold: float, limit: int) -> None:
    print("\n\n=== Eval candidates ===\n")
    bad = [r for r in rows if r["grounding_score"] is not None and r["grounding_score"] < threshold]
    bad.sort(key=lambda r: r["grounding_score"])
    if not bad:
        print("Nothing below the threshold.")
        return
    seen: set[str] = set()
    unique = []
    for r in bad:
        key = " ".join(r["question"].lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print("Paste into QUESTIONS in scripts/run_eval.py. Set the category, and set")
    print("expected only where one tool call is unambiguously correct (else None).\n")
    for r in unique[:limit]:
        print(f'    ("TODO", {json.dumps(r["question"])}, None),  # scored {r["grounding_score"]:.2f}')


def main() -> int:
    parser = argparse.ArgumentParser(description="Report on the CheckTheBall query log.")
    parser.add_argument("--db", default=db.DEFAULT_DB_PATH, help="database path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="score below which an answer counts as a failure")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max rows to print per section")
    parser.add_argument("--since", help="only rows created on or after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"No database at {args.db}", file=sys.stderr)
        return 1

    rows = _rows(args.db, args.since)
    if not rows:
        print("No queries logged yet.")
        return 0

    print(f"CheckTheBall query log: {args.db}")
    scores_section(rows)
    tools_section(rows)
    missing_tool_section(rows, args.limit)
    failures_section(rows, args.threshold, args.limit)
    eval_candidates_section(rows, args.threshold, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
