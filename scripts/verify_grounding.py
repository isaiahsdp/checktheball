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


def main() -> int:
    print("Grounding checks (scripted extractor, no API key needed)")
    all_grounded()
    mixed_grounding()
    string_and_number_forms()
    non_factual_skipped()
    no_claims()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
