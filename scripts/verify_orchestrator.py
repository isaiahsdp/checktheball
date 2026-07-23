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

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main() -> int:
    print("Orchestrator checks (scripted fake client, no API key needed)")
    single_tool()
    multi_round_chain()
    out_of_scope()
    tool_returns_error()
    tool_raises()
    round_cap()
    parallel_tool_calls()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
