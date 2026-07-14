"""LLM tool-use orchestration loop (sport-agnostic).

The two-call loop: the question and tool schemas go to Claude, Claude chooses
which tool(s) to call, our code executes the real functions, and the results
are fed back for Claude to write the final answer. Claude decides *which*
lookups to run; it never supplies the numbers itself.

This module never imports from ``sports``. It receives a tool provider (any
object exposing ``TOOL_SCHEMAS`` and ``call_tool``) as an argument, so the same
loop drives MLB, NBA, or any future sport.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import anthropic
from dotenv import load_dotenv

# Sonnet 5 handles the reasoning and tool selection.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
# Safety valve so a confused model can't loop forever calling tools.
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """\
You are CheckTheBall, a sports statistics assistant. You answer questions using \
ONLY verified data returned by your tools.

Rules:
- Never state a statistic, score, or factual claim from your own memory. Every \
number in your answer must come from a tool result in this conversation.
- Call the appropriate tool(s) to get the data you need. For compound questions, \
call multiple tools and combine their results.
- If a tool result contains an "error" field, do not invent an answer. Tell the \
user what went wrong or that the data isn't available.
- If the question cannot be answered with the available tools (a sport or data \
they don't cover, or a pure opinion with no statistical basis), say so plainly \
instead of guessing.
- For opinion-style questions (e.g. "who had the better season"), ground your \
reasoning in the specific stats you retrieved and name the numbers that support \
your conclusion.
- Be concise and factual. Any number you state must match a tool result exactly.
"""


class ToolProvider(Protocol):
    """Structural contract a sport module must satisfy to drive the loop."""

    TOOL_SCHEMAS: list[dict[str, Any]]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


def _extract_text(content: list[Any]) -> str:
    """Join the text blocks of a response into the final answer string."""
    return "".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def answer_question(
    question: str,
    tools: ToolProvider,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> dict[str, Any]:
    """Answer a question by letting Claude call ``tools`` against real data.

    Returns ``{answer, tool_calls_made, tool_results}``. ``tool_calls_made`` and
    ``tool_results`` expose exactly what was executed so the caller (and the
    grounding layer) can verify the answer against the real data.
    """
    if client is None:
        load_dotenv()  # pick up ANTHROPIC_API_KEY from .env
        client = anthropic.Anthropic()

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    tool_calls_made: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    response = None
    for round_index in range(max_tool_rounds + 1):
        # On the final allowed round, forbid further tool calls so the model
        # must produce a text answer from what it already has.
        force_answer = round_index == max_tool_rounds
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "tools": tools.TOOL_SCHEMAS,
            "messages": messages,
        }
        if force_answer:
            request["tool_choice"] = {"type": "none"}

        response = client.messages.create(**request)
        if response.stop_reason != "tool_use":
            break

        # Preserve the assistant turn (incl. any thinking / tool_use blocks).
        messages.append({"role": "assistant", "content": response.content})

        result_blocks = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_calls_made.append({"name": block.name, "input": block.input})
            try:
                result = tools.call_tool(block.name, block.input)
                is_error = isinstance(result, dict) and "error" in result
            except Exception as exc:  # bad args, etc.; feed back, don't crash
                result = {"error": f"Tool execution failed: {exc}"}
                is_error = True
            tool_results.append({"name": block.name, "input": block.input, "result": result})
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": result_blocks})

    answer = _extract_text(response.content) if response is not None else ""
    return {
        "answer": answer,
        "tool_calls_made": tool_calls_made,
        "tool_results": tool_results,
    }
