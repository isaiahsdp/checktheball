"""Verify the orchestration loop with a scripted fake Claude client.

Fully offline and hermetic: a fake client returns pre-scripted responses and a
fake tool provider returns canned data, so this exercises the loop's mechanics
(tool dispatch, multi-round chaining, error handling, the round cap) with no
API key, no network, and no cost. It tests the plumbing, not whether the real
model picks the right tool; that's what the live checks and grounding layer cover.

Run from the repo root:

    python scripts/verify_orchestrator.py

Exit code is 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compute import COMPUTE_SCHEMA, run_compute
from core.orchestrator import answer_question

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


# --- Fakes standing in for the Anthropic client and a sport's tool module ---


class _ToolUse:
    type = "tool_use"

    def __init__(self, block_id: str, name: str, arguments: dict) -> None:
        self.id = block_id
        self.name = name
        self.input = arguments


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, stop_reason: str, content: list) -> None:
        self.stop_reason = stop_reason
        self.content = content


def tool_use(name: str, arguments: dict, block_id: str = "t1") -> _Response:
    return _Response("tool_use", [_ToolUse(block_id, name, arguments)])


def answer(text: str) -> _Response:
    return _Response("end_turn", [_Text(text)])


class _Messages:
    def __init__(self, scripted: list) -> None:
        self._scripted = list(scripted)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted: list) -> None:
        self.messages = _Messages(scripted)


class FakeTools:
    TOOL_SCHEMAS = [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}]

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name == "raises":
            raise ValueError("boom")
        if name == "bad":
            return {"error": "not found"}
        return {"value": 58}


# --- Scenarios ---


def single_tool() -> None:
    tools = FakeTools()
    client = FakeClient([tool_use("lookup", {"q": 1}), answer("The value is 58.")])
    result = answer_question("q", tools, client=client)
    check("single: final answer returned", result["answer"] == "The value is 58.")
    check("single: one tool call recorded", len(result["tool_calls_made"]) == 1)
    check("single: tool actually executed", tools.calls == [("lookup", {"q": 1})])
    check("single: real result captured for grounding", result["tool_results"][0]["result"] == {"value": 58})


def multi_round_chain() -> None:
    tools = FakeTools()
    client = FakeClient(
        [tool_use("lookup", {"step": 1}), tool_use("lookup", {"step": 2}), answer("done")]
    )
    result = answer_question("q", tools, client=client)
    check("chain: two tool rounds executed", len(result["tool_calls_made"]) == 2)
    check("chain: three API calls made", len(client.messages.requests) == 3)
    check("chain: answer taken from final turn", result["answer"] == "done")


def out_of_scope() -> None:
    tools = FakeTools()
    client = FakeClient([answer("That's out of scope.")])
    result = answer_question("q", tools, client=client)
    check("out-of-scope: no tools called", result["tool_calls_made"] == [])
    check("out-of-scope: answer returned directly", result["answer"] == "That's out of scope.")


def tool_returns_error() -> None:
    tools = FakeTools()
    client = FakeClient([tool_use("bad", {}), answer("I couldn't find that.")])
    result = answer_question("q", tools, client=client)
    check("error-result: error surfaced in results", "error" in result["tool_results"][0]["result"])
    sent_block = client.messages.requests[1]["messages"][-1]["content"][0]
    check("error-result: is_error flagged back to the model", sent_block["is_error"] is True)
    check("error-result: loop still produced an answer", result["answer"] == "I couldn't find that.")


def tool_raises() -> None:
    tools = FakeTools()
    client = FakeClient([tool_use("raises", {}), answer("Something went wrong.")])
    result = answer_question("q", tools, client=client)
    err = result["tool_results"][0]["result"].get("error", "")
    check("exception: caught and returned as error", err.startswith("Tool execution failed"))
    check("exception: did not crash, produced answer", result["answer"] == "Something went wrong.")


def round_cap() -> None:
    tools = FakeTools()
    # Model keeps calling tools; with max_tool_rounds=2 the third call must forbid them.
    client = FakeClient(
        [tool_use("lookup", {"n": 1}), tool_use("lookup", {"n": 2}), answer("forced final")]
    )
    result = answer_question("q", tools, client=client, max_tool_rounds=2)
    check("cap: final call forbids further tools", client.messages.requests[2].get("tool_choice") == {"type": "none"})
    check("cap: earlier calls allow tools", "tool_choice" not in client.messages.requests[0])
    check("cap: answer still returned", result["answer"] == "forced final")


def parallel_tool_calls() -> None:
    tools = FakeTools()
    # A single assistant turn requests two tools at once (parallel tool use).
    two_at_once = _Response(
        "tool_use",
        [_ToolUse("a", "lookup", {"which": "first"}), _ToolUse("b", "lookup", {"which": "second"})],
    )
    client = FakeClient([two_at_once, answer("combined")])
    result = answer_question("q", tools, client=client)

    check(
        "parallel: both tools executed, in order, in one round",
        tools.calls == [("lookup", {"which": "first"}), ("lookup", {"which": "second"})],
    )
    check("parallel: both calls recorded", len(result["tool_calls_made"]) == 2)
    check("parallel: both results captured", len(result["tool_results"]) == 2)
    check("parallel: only two API calls (one round handled both tools)", len(client.messages.requests) == 2)

    # Both tool_result blocks go back in a single user turn before the next round.
    result_msg = client.messages.requests[1]["messages"][-1]
    check("parallel: two tool_result blocks in one user turn", len(result_msg["content"]) == 2)
    check(
        "parallel: result blocks reference both tool_use ids",
        {b["tool_use_id"] for b in result_msg["content"]} == {"a", "b"},
    )
    check("parallel: final answer returned", result["answer"] == "combined")


def system_prompt_has_date() -> None:
    # The prompt must carry the real current date so the model resolves "this
    # season" against reality. Guards against a revert to a static, date-blind
    # prompt (the bug where 2026 questions were answered with 2024 data, or
    # dismissed as "data doesn't exist yet").
    tools = FakeTools()
    client = FakeClient([answer("ok")])
    answer_question("q", tools, client=client)
    system_text = client.messages.requests[0]["system"][0]["text"]
    check("system: request carries today's date", date.today().isoformat() in system_text)
    check(
        "system: warns not to assume recent-season data is missing",
        'likely doesn\'t exist' in system_text,
    )


def system_prompt_forbids_own_math() -> None:
    # The rule that a number the model works out itself is not verified data. It
    # is what motivates gap_from_leader in the tools and the compute tool in
    # core, so a silent edit to it would quietly change what the rest of the
    # pipeline is built around.
    tools = FakeTools()
    client = FakeClient([answer("ok")])
    answer_question("q", tools, client=client)
    system_text = client.messages.requests[0]["system"][0]["text"]
    check(
        "system: a self-calculated number is not verified data",
        "A number you calculate yourself" in system_text,
    )
    check(
        "system: names the operations the rule covers",
        all(word in system_text for word in ("sum", "difference", "percentage", "per-game rate", "projection")),
    )
    check(
        "system: points the model at the compute tool instead",
        "Use the 'compute' tool" in system_text,
    )
    check(
        "system: tells the model not to round a computed value further",
        "do not round it further" in system_text,
    )


def compute_operand_validation() -> None:
    # The guard that makes arithmetic-as-a-tool safe: operands must already be
    # in this turn's results, so the model can combine verified numbers but
    # never introduce one. Without this, compute would launder a hallucination
    # into a grounded value.
    results = [{"name": "get_player_stat", "input": {},
                "result": {"player": "Trent Grisham", "strikeOuts": 41, "atBats": 222}}]

    ok = run_compute({"operation": "divide", "operands": [41, 222]}, results)
    check("compute: divides two retrieved numbers", ok["result"] == 0.1847)
    # All three precisions, because which one an answer states varies with the
    # number. A live 0.00 came from returning only 18.47 and 18 while the model
    # wrote "about 18.5%".
    check("compute: a rate also comes back as a percentage", ok["as_percent"] == 18.47)
    check("compute: at one decimal place, the form that caused a live miss", ok["as_percent_1dp"] == 18.5)
    check("compute: and as a whole percent", ok["as_percent_whole"] == 18)

    invented = run_compute({"operation": "divide", "operands": [50, 200]}, results)
    check("compute: operands no tool returned are refused", "error" in invented)
    check(
        "compute: the error names the offending operands",
        "50.0" in invented["error"] and "200.0" in invented["error"],
    )

    # Half-invented is still refused: 41 is real, 300 is not.
    partly = run_compute({"operation": "divide", "operands": [41, 300]}, results)
    check("compute: one bad operand fails the whole call", "error" in partly)

    # Nothing retrieved yet (compute called before its inputs) fails the same way.
    early = run_compute({"operation": "divide", "operands": [41, 222]}, [])
    check("compute: called before any lookup, nothing validates", "error" in early)


def compute_operations() -> None:
    results = [{"name": "compare_players", "input": {}, "result": {
        "players": [{"player": "Aaron Judge", "value": 17},
                    {"player": "Shohei Ohtani", "value": 24}], "atBats": 0}}]
    check(
        "compute: subtract is left to right, so order carries the sign",
        run_compute({"operation": "subtract", "operands": [24, 17]}, results)["result"] == 7,
    )
    check(
        "compute: add totals across sources",
        run_compute({"operation": "add", "operands": [24, 17]}, results)["result"] == 41,
    )
    check(
        "compute: non-divide results carry no percentage form",
        "as_percent" not in run_compute({"operation": "add", "operands": [24, 17]}, results),
    )
    check(
        "compute: dividing by a retrieved zero is refused, not a crash",
        "error" in run_compute({"operation": "divide", "operands": [17, 0]}, results),
    )
    check(
        "compute: an unknown operation is refused",
        "error" in run_compute({"operation": "power", "operands": [24, 17]}, results),
    )
    check(
        "compute: a single operand is not an operation",
        "error" in run_compute({"operation": "add", "operands": [24]}, results),
    )
    check(
        "compute: a non-numeric operand is refused",
        "error" in run_compute({"operation": "add", "operands": [24, "17"]}, results),
    )


def compute_in_the_loop() -> None:
    # End to end through the real loop: look a stat up, then compute a rate from
    # it. compute is handled by the orchestrator, never dispatched to the sport
    # provider, and its result lands in tool_results like any other retrieval.
    tools = FakeTools()
    client = FakeClient([
        tool_use("lookup", {"stat": "strikeOuts"}, "t1"),
        tool_use("compute", {"operation": "divide", "operands": [58, 58]}, "t2"),
        answer("That is a rate of 1.0."),
    ])
    result = answer_question("q", tools, client=client)
    check("compute loop: the sport provider never sees the compute call",
          [name for name, _ in tools.calls] == ["lookup"])
    check("compute loop: compute is recorded as a tool call like any other",
          [c["name"] for c in result["tool_calls_made"]] == ["lookup", "compute"])
    computed = result["tool_results"][1]["result"]
    check("compute loop: the computed value lands in tool_results", computed["result"] == 1.0)

    # The schema is offered alongside the provider's tools, not instead of them.
    offered = [t["name"] for t in client.messages.requests[0]["tools"]]
    check("compute loop: schema offered next to the sport's tools", offered == ["lookup", "compute"])
    check("compute loop: the provider's own schemas are untouched",
          [t["name"] for t in FakeTools.TOOL_SCHEMAS] == ["lookup"])


def compute_error_reaches_the_model() -> None:
    # A refused operand must come back as a tool_result flagged is_error, so the
    # model can correct itself instead of the request failing.
    tools = FakeTools()
    client = FakeClient([
        tool_use("compute", {"operation": "divide", "operands": [50, 200]}, "t1"),
        answer("I could not verify that number."),
    ])
    result = answer_question("q", tools, client=client)
    follow_up = client.messages.requests[1]["messages"][-1]["content"][0]
    check("compute error: fed back to the model as an error result", follow_up["is_error"] is True)
    check("compute error: the refusal names the operands, not a generic failure",
          "50.0" in json.loads(follow_up["content"])["error"])
    check("compute error: the loop still produces an answer",
          result["answer"] == "I could not verify that number.")


def compute_schema_shape() -> None:
    check("compute schema: named 'compute'", COMPUTE_SCHEMA["name"] == "compute")
    check(
        "compute schema: offers exactly the four arithmetic operations",
        COMPUTE_SCHEMA["input_schema"]["properties"]["operation"]["enum"]
        == ["add", "subtract", "multiply", "divide"],
    )
    check(
        "compute schema: tells the model operands must already be retrieved",
        "already returned" in COMPUTE_SCHEMA["description"],
    )


def main() -> int:
    print("Orchestrator checks (scripted fake client, no API key needed)")
    single_tool()
    multi_round_chain()
    out_of_scope()
    tool_returns_error()
    tool_raises()
    round_cap()
    parallel_tool_calls()
    system_prompt_has_date()
    system_prompt_forbids_own_math()
    compute_operand_validation()
    compute_operations()
    compute_in_the_loop()
    compute_error_reaches_the_model()
    compute_schema_shape()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
