"""Grounding / verification layer (sport-agnostic).

Two stages: Haiku breaks an answer into factual claims and the values each
asserts (numbers, names, awards); a deterministic check then confirms each
value appears in the tool results retrieved this turn.

``ground_answer`` returns a grounding_score and a per-claim breakdown.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from dotenv import load_dotenv

GROUNDING_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# Numbers pulled from different sources ("58", 58, ".322") compare as floats.
_NUM_TOL = 1e-4
_NUMBER_RE = re.compile(r"-?\d+\.?\d*|-?\.\d+")

_EXTRACTION_SYSTEM = """\
You are a fact-checking assistant. Read the answer and list its verifiable
factual claims about sports data: statistics, results, players, teams, seasons,
and awards.

For each claim, put in "values" the specific data points it asserts that could
be checked against a database: the numeric value(s), player name(s), team
name(s), season year(s), and any named award or milestone. Do NOT include the
name of the stat category itself (e.g. "home runs", "batting average", "OPS",
"RBIs", "stolen bases"). Those are labels, not values.

Do NOT include calendar dates or date ranges (e.g. "June 30, 2025", "July 19 to
Sept 30", "since the All-Star break") as values. They describe the time window a
stat was measured over, not the stat itself.

Skip subjective or evaluative statements ("a historic season", "MVP-caliber")
and skip meta statements about your own ability ("I can't answer that", "my
tools only cover MLB"). Those are not verifiable factual claims.
"""

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "values"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def extract_claims(answer: str, client: anthropic.Anthropic) -> list[dict[str, Any]]:
    """Use Haiku to decompose an answer into claims with their asserted values."""
    response = client.messages.create(
        model=GROUNDING_MODEL,
        max_tokens=MAX_TOKENS,
        system=_EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": f"Answer:\n{answer}"}],
        output_config={"format": {"type": "json_schema", "schema": _CLAIMS_SCHEMA}},
    )
    text = next((b.text for b in response.content if b.type == "text"), "{}")
    return json.loads(text).get("claims", [])


def _to_float(token: str) -> float | None:
    cleaned = token.strip().lstrip("+").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumerics so 'home runs' matches 'homeRuns'."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _numbers_in(token: str) -> list[float]:
    found = []
    for match in _NUMBER_RE.findall(token):
        value = _to_float(match)
        if value is not None:
            found.append(value)
    return found


def _collect(value: Any, numbers: set[float], text_parts: list[str]) -> None:
    """Recursively gather numeric leaves and text (values + keys)."""
    if isinstance(value, bool):
        text_parts.append(str(value))
    elif isinstance(value, (int, float)):
        numbers.add(float(value))
    elif isinstance(value, str):
        text_parts.append(value)
        as_num = _to_float(value)
        if as_num is not None:
            numbers.add(as_num)
    elif isinstance(value, dict):
        for key, item in value.items():
            text_parts.append(key)  # index keys so stat labels can match
            _collect(item, numbers, text_parts)
    elif isinstance(value, list):
        for item in value:
            _collect(item, numbers, text_parts)


def _index_tool_results(tool_results: list[dict[str, Any]]) -> tuple[set[float], str]:
    """Build the searchable index (numbers + normalized text) from retrieved data."""
    numbers: set[float] = set()
    text_parts: list[str] = []
    for entry in tool_results:
        _collect(entry.get("result"), numbers, text_parts)
        _collect(entry.get("input"), numbers, text_parts)
    return numbers, _normalize(" ".join(text_parts))


def _value_supported(value: str, numbers: set[float], corpus: str) -> bool:
    nums = _numbers_in(value)
    if nums:
        # A value with numbers is backed only if every number is in the data.
        return all(any(abs(n - dn) <= _NUM_TOL for dn in numbers) for n in nums)
    # Lowercase descriptive phrases ("road games", "home runs") are labels, not
    # checkable data points, so they don't count against a claim.
    if value.strip() == value.strip().lower():
        return True
    # A proper-noun value (player, team, award) must appear in the data.
    normalized = _normalize(value)
    return bool(normalized) and normalized in corpus


def _verify_claim(
    claim: dict[str, Any], numbers: set[float], corpus: str
) -> dict[str, Any]:
    values = claim.get("values", [])
    missing = [v for v in values if not _value_supported(v, numbers, corpus)]
    return {
        "text": claim.get("text", ""),
        "values": values,
        "supported": len(missing) == 0,
        "missing": missing,
    }


def ground_answer(
    answer: str,
    tool_results: list[dict[str, Any]],
    *,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    """Extract claims from ``answer`` and verify each against ``tool_results``.

    Returns ``{grounding_score, total_claims, supported_claims, claims}``. A
    claim is supported when every value it asserts is found in the retrieved
    data. An answer with no verifiable claims scores 1.0.
    """
    if client is None:
        load_dotenv()
        client = anthropic.Anthropic()

    numbers, corpus = _index_tool_results(tool_results)

    checked = []
    for claim in extract_claims(answer, client):
        if not claim.get("values"):  # nothing checkable; not a factual claim
            continue
        checked.append(_verify_claim(claim, numbers, corpus))

    total = len(checked)
    supported = sum(1 for c in checked if c["supported"])
    score = round(supported / total, 3) if total else 1.0

    return {
        "grounding_score": score,
        "total_claims": total,
        "supported_claims": supported,
        "claims": checked,
    }
