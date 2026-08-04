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




def iso_date_is_context() -> None:
    # A refusal that cites today's date ("not playing in today's games
    # (2026-08-03)") scored 0.00 in production: the extractor put the date in
    # values despite being told not to, and the tool had returned only an error
    # string, so nothing matched. A date names when a stat was measured, so it
    # is context. Handled here as well as in the prompt because an ISO date is
    # recognizable by shape alone, with no sport vocabulary involved.
    errored = [{"name": "get_fantasy_points", "input": {"player": "Jung Hoo Lee"},
                "result": {"error": "No player matching 'Jung Hoo Lee' found in today's games."}}]
    g = ground_answer("…", errored, client=FakeExtractor(
        [{"text": "Jung Hoo Lee is not in today's games (2026-08-03)",
          "values": ["Jung Hoo Lee", "2026-08-03"]}]))
    check("iso date: a date in values does not sink an otherwise sound claim",
          g["claims"][0]["missing"] == [])

    # A bare year is a real fact the answer asserts, so it still has to ground.
    # Guards against widening this into "anything date-ish passes".
    year_only = [{"name": "get_player_stat", "input": {}, "result": {"value": 58, "season": 2024}}]
    right = ground_answer("…", year_only, client=FakeExtractor(
        [{"text": "in 2024 he hit 58", "values": ["2024", "58"]}]))
    check("iso date: a bare year is still checked", right["supported_claims"] == 1)
    wrong = ground_answer("…", year_only, client=FakeExtractor(
        [{"text": "in 1999 he hit 58", "values": ["1999", "58"]}]))
    check("iso date: a wrong year still fails", wrong["claims"][0]["missing"] == ["1999"])

    # Only a full ISO date passes. A number that merely contains digits and
    # dashes is not waved through.
    partial = ground_answer("…", year_only, client=FakeExtractor(
        [{"text": "over 2024-06", "values": ["2024-06"]}]))
    check("iso date: a partial date is not treated as one", partial["supported_claims"] == 0)


def model_arithmetic_does_not_ground() -> None:
    # The load-bearing rule after the derived-ratio fallback was removed: a
    # number the model works out itself is not verified data, no matter how
    # cleanly it follows from numbers we did retrieve. 41/222 is a correct
    # strikeout rate and still must not ground on its own. Re-adding an
    # after-the-fact arithmetic check would flip these, which is the point of
    # guarding them: on a leaderboard-sized result that check accepted roughly
    # half of all percentages, so it approved invented numbers as readily as
    # real ones. The supported path is core/compute.py.
    lookup = [{"name": "get_player_stat", "input": {}, "result": {
        "player": "Trent Grisham", "strikeOuts": 41, "atBats": 222}}]
    for label, value in (("the decimal ratio", "0.1847"), ("the percentage", "18.47%"),
                         ("the rounded percentage", "18%")):
        g = ground_answer("…", lookup, client=FakeExtractor(
            [{"text": f"a strikeout rate of {value}", "values": [value]}]))
        check(f"model math: {label} does not ground without a compute result",
              g["supported_claims"] == 0)

    # Subtraction likewise: 24 - 17 = 7 is correct and still unverified.
    pair = [{"name": "get_player_stat", "input": {}, "result": {"a": 24, "b": 17}}]
    gap = ground_answer("…", pair, client=FakeExtractor(
        [{"text": "a gap of 7 home runs", "values": ["7"]}]))
    check("model math: a difference the model subtracted does not ground",
          gap["supported_claims"] == 0)


def computed_value_grounds_directly() -> None:
    # The compute tool is what makes the rate above verifiable: the same claim
    # grounds once the number arrives as a tool result instead of out of the
    # model's head. Paired with model_arithmetic_does_not_ground, this is the
    # whole design in two checks.
    from core.compute import run_compute

    lookup = [{"name": "get_player_stat", "input": {}, "result": {
        "player": "Trent Grisham", "split": "vs_right", "strikeOuts": 41, "atBats": 222}}]
    computed = run_compute({"operation": "divide", "operands": [41, 222]}, lookup)
    tool_results = lookup + [{"name": "compute", "input": {}, "result": computed}]

    claim = [{"text": "Trent Grisham struck out in 18.47% of his at-bats against righties",
              "values": ["Trent Grisham", "18.47%"]}]
    g = ground_answer("…", tool_results, client=FakeExtractor(claim))
    check("computed value: the same rate grounds once compute returned it",
          g["claims"][0]["supported"] is True)

    # The whole-percent form the model usually reaches for comes back too, so
    # "about 18%" grounds without any rounding slack in the verifier.
    whole = ground_answer("…", tool_results, client=FakeExtractor(
        [{"text": "That is about an 18% strikeout rate", "values": ["18%"]}]))
    check("computed value: the whole-percent form grounds too", whole["supported_claims"] == 1)

    # A number compute never returned still fails, even next to a compute result.
    invented = ground_answer("…", tool_results, client=FakeExtractor(
        [{"text": "That is a 26% strikeout rate", "values": ["26%"]}]))
    check("computed value: a rate compute did not return stays unsupported",
          invented["supported_claims"] == 0)


def split_label_is_not_a_value() -> None:
    # A live 0.00: the answer said "44.74% vs. RHP", the math was right, and the
    # claim failed because "RHP" is uppercase and so was checked as a proper
    # noun against data that says "vs_right". A handedness or home/away label
    # names the slice a number came from, exactly like a date, so the extraction
    # prompt excludes it rather than the verifier special-casing abbreviations
    # (which would be baseball vocabulary in a sport-agnostic module).
    from core.grounding import _EXTRACTION_SYSTEM

    check("split label: the extractor is told to skip handedness labels",
          "RHP" in _EXTRACTION_SYSTEM and "LHP" in _EXTRACTION_SYSTEM)
    check("split label: and home/away labels",
          "at home" in _EXTRACTION_SYSTEM and "on the road" in _EXTRACTION_SYSTEM)
    check("split label: an opponent's team name is still a value",
          "opponent's team name IS a value" in _EXTRACTION_SYSTEM)

    # If one slips through anyway, an uppercase label still fails, which is why
    # the rule lives in the prompt. Documents the residual gap, in the style of
    # value_only_in_input.
    tool_results = [{"name": "get_player_stat", "input": {},
                     "result": {"player": "Spencer Jones", "split": "vs_right", "strikeOuts": 34}}]
    leaked = ground_answer("…", tool_results, client=FakeExtractor(
        [{"text": "34 strikeouts vs. RHP", "values": ["34", "RHP"]}]))
    check("split label: an abbreviation that slips through still fails (known gap)",
          leaked["claims"][0]["missing"] == ["RHP"])
    # Spelled out in lowercase it passes as a descriptive label, which is why
    # only the abbreviated answers failed in production.
    spelled = ground_answer("…", tool_results, client=FakeExtractor(
        [{"text": "34 strikeouts", "values": ["34", "right-handed pitching"]}]))
    check("split label: the lowercase spelled-out form was never affected",
          spelled["supported_claims"] == 1)


def leaderboard_gap_claim() -> None:
    # A leaderboard answer states the gap behind the leader ("trailed by 14").
    # get_top_performers precomputes gap_from_leader, so the number is retrieved
    # data rather than model subtraction. Same fix as compute, landed at the
    # source: with nothing precomputed, this claim would not ground at all.
    tool_results = [{"name": "get_top_performers", "input": {}, "result": {
        "scope": "season", "stat": "strikeOuts", "season": 2024, "leaders": [
            {"player": "Garrett Crochet", "value": 255, "gap_from_leader": 0},
            {"player": "Tarik Skubal", "value": 241, "gap_from_leader": 14},
        ]}}]
    claim = [{"text": "Skubal trailed the strikeout leader by 14", "values": ["Tarik Skubal", "14"]}]
    g = ground_answer("…", tool_results, client=FakeExtractor(claim))
    check("leaderboard gap: 'trailed by 14' grounds via precomputed gap_from_leader",
          g["claims"][0]["supported"] is True)

    # Contrast: strip the precomputed field and the same gap is model math again.
    bare = [{"name": "get_top_performers", "input": {}, "result": {
        "leaders": [{"player": "Garrett Crochet", "value": 255},
                    {"player": "Tarik Skubal", "value": 241}]}}]
    without = ground_answer("…", bare, client=FakeExtractor(claim))
    check("leaderboard gap: without the precomputed field the gap does not ground",
          without["supported_claims"] == 0)


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
    iso_date_is_context()
    model_arithmetic_does_not_ground()
    computed_value_grounds_directly()
    split_label_is_not_a_value()
    leaderboard_gap_claim()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
