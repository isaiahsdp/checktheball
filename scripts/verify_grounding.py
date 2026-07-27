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


def derived_rate_claim() -> None:
    # A rate the model computes itself: 18 strikeouts over 2 games -> 9.0 per
    # game. 9.0 was never returned by a tool, but both inputs are grounded, so
    # the ratio fallback supports it (mirrors fantasy.py's derived stats, but
    # verified after the fact instead of computed ahead of time).
    tool_results = [{"name": "get_player_stat", "input": {}, "result": {
        "player": "Tarik Skubal", "strikeOuts": 18, "gamesPlayed": 2}}]
    ok = ground_answer("…", tool_results, client=FakeExtractor(
        [{"text": "That is 9.0 strikeouts per game", "values": ["9.0"]}]))
    c = ok["claims"][0]
    check("derived rate: 18/2 = 9.0 grounds via a computed ratio", c["supported"] is True)
    check("derived rate: ratio match flagged in `derived`, not silent", c.get("derived") == ["9.0"])

    # Contrast (false-positive guard): a derived-looking number that is not any
    # real ratio of the corpus stays unsupported.
    bad = ground_answer("…", tool_results, client=FakeExtractor(
        [{"text": "That is 7.5 strikeouts per game", "values": ["7.5"]}]))
    check("derived rate: 7.5 is no ratio of 18 and 2, still flagged unsupported", bad["supported_claims"] == 0)

    # A directly-present number must not be mislabeled as derived.
    direct = ground_answer("…", tool_results, client=FakeExtractor(
        [{"text": "He had 18 strikeouts", "values": ["18"]}]))
    check("derived rate: a direct match is not tagged derived", "derived" not in direct["claims"][0])


def leaderboard_gap_claim() -> None:
    # A leaderboard answer states the gap behind the leader ("trailed by 14").
    # get_top_performers now precomputes gap_from_leader, so the number grounds
    # via the normal direct-match path, not grounding's division fallback. This
    # is the fix landing at the source (tools.py), with no grounding.py change.
    tool_results = [{"name": "get_top_performers", "input": {}, "result": {
        "scope": "season", "stat": "strikeOuts", "season": 2024, "leaders": [
            {"player": "Garrett Crochet", "value": 255, "gap_from_leader": 0},
            {"player": "Tarik Skubal", "value": 241, "gap_from_leader": 14},
        ]}}]
    claim = [{"text": "Skubal trailed the strikeout leader by 14", "values": ["Tarik Skubal", "14"]}]
    g = ground_answer("…", tool_results, client=FakeExtractor(claim))
    c = g["claims"][0]
    check("leaderboard gap: 'trailed by 14' grounds via precomputed gap_from_leader", c["supported"] is True)
    check("leaderboard gap: 14 matched directly, not via the derived-ratio fallback", "derived" not in c)


def derived_ratio_widens_with_corpus_size() -> None:
    # Documents a known limitation, in the style of value_only_in_input. The
    # ratio fallback tries every ordered pair, so the set of numbers it accepts
    # grows with the square of the retrieved data. Against a two-number result,
    # 22 is correctly rejected; against an ordinary five-row leaderboard it
    # grounds as a coincidental ratio (44 / 2) despite no tool returning it.
    # This is why a predictable derived value belongs in the tool output
    # (gap_from_leader) rather than in a wider arithmetic fallback.
    small = [{"name": "get_player_stat", "input": {}, "result": {"value": 44, "gamesPlayed": 2}}]
    tight = ground_answer("…", small, client=FakeExtractor([{"text": "he hit 22", "values": ["22"]}]))
    check("derived ratio: 22 grounds against a corpus that really contains 44 and 2", tight["supported_claims"] == 1)

    leaderboard = [{"name": "get_top_performers", "input": {"stat": "homeRuns", "season": 2024, "limit": 5}, "result": {
        "scope": "season", "stat": "homeRuns", "season": 2024, "limit": 5, "leaders": [
            {"rank": 1, "player": "A", "value": 58, "gap_from_leader": 0},
            {"rank": 2, "player": "B", "value": 54, "gap_from_leader": 4},
            {"rank": 3, "player": "C", "value": 48, "gap_from_leader": 10},
            {"rank": 4, "player": "D", "value": 47, "gap_from_leader": 11},
            {"rank": 5, "player": "E", "value": 44, "gap_from_leader": 14},
        ]}}]
    wide = ground_answer("…", leaderboard, client=FakeExtractor([{"text": "he hit 22", "values": ["22"]}]))
    c = wide["claims"][0]
    check("derived ratio: a leaderboard-sized corpus accepts 22 as a coincidental ratio", c["supported"] is True)
    check("derived ratio: the coincidental match is tagged derived, so it stays traceable", c.get("derived") == ["22"])


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
    derived_rate_claim()
    leaderboard_gap_claim()
    derived_ratio_widens_with_corpus_size()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
