"""Verify the grounding layer with canned data and a scripted extractor.

The claim extractor (normally a Haiku call) is replaced with a fake client
returning pre-scripted claims, and the tool results are canned, so the
deterministic verification runs offline.

Run from the repo root:

    python scripts/verify_grounding.py

Exit code is 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grounding import ground_answer

_passed = 0
_failed = 0


def check(name: str, condition: bool) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}")


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, content: list) -> None:
        self.content = content


class _Messages:
    def __init__(self, claims: list) -> None:
        self._payload = json.dumps({"claims": claims})

    def create(self, **kwargs):
        return _Response([_Text(self._payload)])


class FakeExtractor:
    """Stands in for the Anthropic client; returns pre-scripted claims."""

    def __init__(self, claims: list) -> None:
        self.messages = _Messages(claims)


# Canned retrieved data, like two get_player_stat results.
TOOL_RESULTS = [
    {"name": "get_player_stat", "input": {}, "result": {
        "player": "Aaron Judge", "team": "New York Yankees",
        "stat": "homeRuns", "value": 58, "season": 2024, "avg": ".322"}},
    {"name": "get_player_stat", "input": {}, "result": {
        "player": "Shohei Ohtani", "team": "Los Angeles Dodgers",
        "stat": "homeRuns", "value": 54, "season": 2024}},
]


def all_grounded() -> None:
    claims = [
        {"text": "Judge hit 58 home runs", "values": ["58"]},
        {"text": "Ohtani hit 54 home runs", "values": ["54"]},
        {"text": "Judge plays for the Yankees", "values": ["New York Yankees"]},
        {"text": "Judge batted .322", "values": [".322"]},
    ]
    g = ground_answer("…", TOOL_RESULTS, client=FakeExtractor(claims))
    check("all-grounded: score is 1.0", g["grounding_score"] == 1.0)
    check("all-grounded: 4 of 4 supported", g["supported_claims"] == 4 and g["total_claims"] == 4)


def mixed_grounding() -> None:
    claims = [
        {"text": "Judge hit 58 home runs", "values": ["58"]},        # grounded
        {"text": "Ohtani hit 60 home runs", "values": ["60"]},        # wrong number
        {"text": "Judge won AL MVP", "values": ["AL MVP"]},           # not in data
    ]
    g = ground_answer("…", TOOL_RESULTS, client=FakeExtractor(claims))
    check("mixed: 1 of 3 supported", g["supported_claims"] == 1 and g["total_claims"] == 3)
    check("mixed: score is 0.333", g["grounding_score"] == round(1 / 3, 3))
    flagged = {c["text"] for c in g["claims"] if not c["supported"]}
    check("mixed: wrong number flagged", "Ohtani hit 60 home runs" in flagged)
    check("mixed: unsupported fact flagged", "Judge won AL MVP" in flagged)


def string_and_number_forms() -> None:
    claims = [
        {"text": "team name", "values": ["Los Angeles Dodgers"]},  # string match
        {"text": "avg as decimal", "values": ["0.322"]},           # ".322" stored -> 0.322
        {"text": "season", "values": ["2024"]},                    # int stored -> match
    ]
    g = ground_answer("…", TOOL_RESULTS, client=FakeExtractor(claims))
    check("forms: all three matched across string/number forms", g["supported_claims"] == 3)


def non_factual_skipped() -> None:
    claims = [
        {"text": "Judge hit 58 home runs", "values": ["58"]},   # counts
        {"text": "a historic season", "values": []},            # no values -> skipped
    ]
    g = ground_answer("…", TOOL_RESULTS, client=FakeExtractor(claims))
    check("skip: value-less claim not counted", g["total_claims"] == 1)


def no_claims() -> None:
    g = ground_answer("I can't answer that.", TOOL_RESULTS, client=FakeExtractor([]))
    check("no-claims: score defaults to 1.0", g["grounding_score"] == 1.0 and g["total_claims"] == 0)


def value_only_in_input() -> None:
    # The index is built from both a tool's `input` and its `result`. A value the
    # model passed as an argument (here a wrong season) is therefore treated as
    # supported even though no result carries it. This is the concrete version of
    # the query-faithfulness gap documented in docs/grounding.md.
    tool_results = [
        {
            "name": "get_player_stat",
            "input": {"player": "Aaron Judge", "season": 1999},
            "result": {"player": "Aaron Judge", "stat": "homeRuns", "value": 58, "season": 2024},
        }
    ]
    in_input = ground_answer("…", tool_results, client=FakeExtractor([{"text": "the 1999 season", "values": ["1999"]}]))
    check(
        "value present only in tool input still grounds (documents the faithfulness gap)",
        in_input["supported_claims"] == 1 and in_input["grounding_score"] == 1.0,
    )
    # Contrast: a value in neither the input nor the result is not supported, so
    # the grounding above is genuinely coming from the indexed input.
    absent = ground_answer("…", tool_results, client=FakeExtractor([{"text": "the 2001 season", "values": ["2001"]}]))
    check("value in neither input nor result does not ground (contrast)", absent["supported_claims"] == 0)


def multi_value_partial() -> None:
    # One value is backed by the data ("58"); one is not ("AL MVP"). The claim
    # should fail, and `missing` should list only the unsupported value.
    claims = [{"text": "Judge hit 58 home runs and won AL MVP", "values": ["58", "AL MVP"]}]
    g = ground_answer("…", TOOL_RESULTS, client=FakeExtractor(claims))
    c = g["claims"][0]
    check("multi-value: claim unsupported when one value is missing", c["supported"] is False)
    check("multi-value: missing lists only the unsupported value", c["missing"] == ["AL MVP"])


def negative_number() -> None:
    # A differential like -4 exercises the negative branch of the number regex.
    tool_results = [{"name": "compare_players", "input": {}, "result": {"stat": "homeRuns", "difference": -4}}]
    present = ground_answer("…", tool_results, client=FakeExtractor([{"text": "the gap was -4 home runs", "values": ["-4"]}]))
    check("negative number matches -4 in the data", present["supported_claims"] == 1)
    absent = ground_answer("…", tool_results, client=FakeExtractor([{"text": "the gap was -9", "values": ["-9"]}]))
    check("negative number not in the data is flagged", absent["supported_claims"] == 0)


def error_message_grounding() -> None:
    # A tool that errored still has its result indexed. This probes whether a
    # claim value can spuriously match text inside an {"error": ...} string.
    tool_results = [
        {
            "name": "get_player_stat",
            "input": {},
            "result": {"error": "No player found matching 'Zephyr'. Did you mean 42?"},
        }
    ]
    number = ground_answer("…", tool_results, client=FakeExtractor([{"text": "the value is 42", "values": ["42"]}]))
    check(
        "number embedded in an error string does NOT ground (strings aren't parsed for numbers)",
        number["supported_claims"] == 0,
    )
    word = ground_answer("…", tool_results, client=FakeExtractor([{"text": "it concerns Zephyr", "values": ["Zephyr"]}]))
    check(
        "word in an error string DOES ground (error text is indexed into the corpus)",
        word["supported_claims"] == 1,
    )


def date_range_year_claim() -> None:
    # Regression guard: a date-range answer states the queried year ("...in 2024").
    # Haiku reliably extracts that year as a value. The date-range tool result now
    # exposes the year as a number (window meta), so the whole claim grounds. Before
    # that, the year lived only inside date strings and produced a deterministic 0.0.
    claim = [{"text": "Aaron Judge hit 11 home runs between June 1 and June 30, 2024", "values": ["Aaron Judge", "11", "2024"]}]

    with_year = [{"name": "get_player_stat", "input": {}, "result": {
        "player": "Aaron Judge", "stat": "homeRuns", "value": 11,
        "scope": "date_range", "start_date": "2024-06-01", "end_date": "2024-06-30", "year": 2024}}]
    g = ground_answer("…", with_year, client=FakeExtractor(claim))
    check("date-range+year claim grounds when the result exposes the year", g["supported_claims"] == 1 and g["grounding_score"] == 1.0)

    # Contrast: with the year only inside date strings, it is unverifiable (the
    # pre-fix behavior this guard exists to catch if it regresses).
    no_year = [{"name": "get_player_stat", "input": {}, "result": {
        "player": "Aaron Judge", "stat": "homeRuns", "value": 11,
        "start_date": "2024-06-01", "end_date": "2024-06-30"}}]
    g2 = ground_answer("…", no_year, client=FakeExtractor(claim))
    c = g2["claims"][0]
    check("date-range+year claim: year unverifiable when result omits it", c["supported"] is False and c["missing"] == ["2024"])


def main() -> int:
    print("Grounding checks (scripted extractor, no API key needed)")
    all_grounded()
    mixed_grounding()
    string_and_number_forms()
    non_factual_skipped()
    no_claims()
    value_only_in_input()
    multi_value_partial()
    negative_number()
    error_message_grounding()
    date_range_year_claim()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
