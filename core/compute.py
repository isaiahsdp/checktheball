"""Arithmetic as a tool (sport-agnostic).

The model decides what to compute; this module produces the number. The result
lands in the turn's tool results like any other retrieved value, so grounding
verifies it by direct match instead of reconstructing the arithmetic after the
fact.

The operand check is what separates this from letting the model do the math in
its head: every operand must already appear in the data retrieved this turn, so
the model can combine verified numbers but never introduce one. A model that
passes a number it invented gets an error, not a result.

Lives in ``core`` because arithmetic is not sport-specific; the orchestrator
offers it alongside whatever tools the sport provider exposes.
"""

from __future__ import annotations

from typing import Any

from core.grounding import matches_retrieved, retrieved_numbers

TOOL_NAME = "compute"

# Left to right, so order carries meaning for subtract and divide:
# subtract [24, 17] is 7, divide [41, 222] is 0.1847.
_OPERATIONS = ("add", "subtract", "multiply", "divide")

# Counting stats divide into rates that need more than 2dp to stay distinct
# (a .308 batting average, an .1847 strikeout rate), so results keep four.
_RESULT_PLACES = 4


def _apply(operation: str, operands: list[float]) -> float:
    total = operands[0]
    for operand in operands[1:]:
        if operation == "add":
            total += operand
        elif operation == "subtract":
            total -= operand
        elif operation == "multiply":
            total *= operand
        else:
            total /= operand
    return total


def run_compute(
    arguments: dict[str, Any], tool_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run one arithmetic operation over numbers already retrieved this turn.

    Returns an ``{"error": ...}`` dict for anything the model can correct
    (unknown operation, an operand nothing returned, division by zero) so the
    orchestrator can hand it back rather than failing the request.
    """
    operation = arguments.get("operation")
    if operation not in _OPERATIONS:
        return {"error": f"Unknown operation '{operation}'. Use one of: {', '.join(_OPERATIONS)}."}

    operands = arguments.get("operands")
    if not isinstance(operands, list) or len(operands) < 2:
        return {"error": "'operands' must be a list of at least two numbers."}

    values: list[float] = []
    for operand in operands:
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            return {"error": f"Operand {operand!r} is not a number."}
        values.append(float(operand))

    # The whole point of the tool: arithmetic on verified numbers only.
    available = retrieved_numbers(tool_results)
    ungrounded = [v for v in values if not matches_retrieved(v, available)]
    if ungrounded:
        return {
            "error": (
                f"Operand(s) {ungrounded} were not returned by any tool this turn. "
                "compute only combines numbers already retrieved. Look the value up "
                "first, then compute with what the tool returned."
            )
        }

    if operation == "divide" and any(v == 0 for v in values[1:]):
        return {"error": "Cannot divide by zero."}

    raw = _apply(operation, values)
    result: dict[str, Any] = {
        "operation": operation,
        "operands": values,
        "result": round(raw, _RESULT_PLACES),
    }
    if operation == "divide":
        # A rate gets stated as a percentage far more often than as a decimal,
        # and the model rounds when it does. Which precision it picks varies
        # with the number ("18.5%", "roughly 21%", "44.74%"), so all three
        # observed forms come back. Returning them means whatever the answer
        # states is a number a tool actually returned, and grounding stays an
        # exact match instead of carrying a rounding tolerance of its own.
        percent = raw * 100
        result["as_percent"] = round(percent, 2)
        result["as_percent_1dp"] = round(percent, 1)
        result["as_percent_whole"] = round(percent)
    return result


COMPUTE_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Do arithmetic on numbers that other tools have already returned in this "
        "conversation. Use this instead of calculating in your head whenever an "
        "answer needs a value no tool returned directly: a rate or percentage "
        "(divide), a difference or gap (subtract), or a total across sources "
        "(add). Every operand must be a number a tool already returned this turn; "
        "passing a number from memory returns an error. Operands apply left to "
        "right, so subtract [24, 17] is 7 and divide [41, 222] is 0.1847. For "
        "divide, the result also comes back as a percentage at three "
        "precisions: 'as_percent', 'as_percent_1dp', and 'as_percent_whole'. "
        "State whichever returned value you use exactly as given; do not round "
        "it further."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS),
                "description": "The arithmetic to perform, applied left to right across operands.",
            },
            "operands": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    "Two or more numbers, each already returned by a tool this turn. "
                    "Order matters for subtract and divide."
                ),
            },
        },
        "required": ["operation", "operands"],
    },
}
