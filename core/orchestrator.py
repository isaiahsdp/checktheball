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
from datetime import date
from typing import Any, Protocol

import anthropic
from dotenv import load_dotenv

from core.compute import COMPUTE_SCHEMA, TOOL_NAME as COMPUTE_TOOL, run_compute

# Sonnet 5 handles the reasoning and tool selection.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
# Safety valve so a confused model can't loop forever calling tools.
MAX_TOOL_ROUNDS = 6

# {current_date} is filled in per request (see _system_blocks): the model must
# resolve "this season" against the real date, not its training horizon.
_SYSTEM_PROMPT_TEMPLATE = """\
You are CheckTheBall, a sports statistics assistant. You answer questions using \
ONLY verified data returned by your tools.

Today's date is {current_date}. Use it to resolve relative time references like \
"this season", "today", "current", or "latest" against the real world; do not \
assume a season based on your own training data.

Rules:
- Never state a statistic, score, or factual claim from your own memory. Every \
number in your answer must come from a tool result in this conversation.
- A number you calculate yourself (a sum, difference, percentage, per-game rate, \
or projection over extra games) is not verified data. Use the 'compute' tool to \
do the arithmetic instead: it works only on numbers other tools already returned \
this turn, and what it gives back is verified data you can state directly. State \
a computed value exactly as compute returned it; do not round it further. Never \
present a multi-step estimate or projection as a precise figure. If the number \
the question centers on cannot come from a tool, say so plainly instead of \
computing it yourself.
- Call the appropriate tool(s) to get the data you need. For compound questions, \
call multiple tools and combine their results.
- Do not assume that data for the current or a recent season doesn't exist yet. \
Your training has a cutoff, but the tools query live data that is more current \
than that. If you are unsure whether a season or date has data, call the tool \
and let the actual result tell you; never claim data "likely doesn't exist" \
without having tried the lookup.
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


def _system_blocks() -> list[dict[str, Any]]:
    """System block(s) for one request, with the current date embedded.

    Built per request (not once at import) so the date reflects the real request
    date, not process startup, which matters for a long-running server. The block
    stays cacheable: the date changes at most once a day, so embedding it turns
    the "never changes" prompt into one that changes daily. In practice that means
    a cache miss only on the first request after a date rollover (or normal TTL
    expiry), then cache hits again, rather than the previous never-miss case. The
    breakpoint still sits on the system block, so the tool schemas ahead of it and
    every extra loop round within a request reuse the cached prefix.
    """
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(current_date=date.today().isoformat())
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]


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

    # Built once per request so every round shares one date and reuses the cache.
    system_blocks = _system_blocks()

    response = None
    for round_index in range(max_tool_rounds + 1):
        # On the final allowed round, forbid further tool calls so the model
        # must produce a text answer from what it already has.
        force_answer = round_index == max_tool_rounds
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system_blocks,
            # compute rides alongside the sport's tools: it is sport-agnostic and
            # needs this turn's results, which the provider never sees.
            "tools": [*tools.TOOL_SCHEMAS, COMPUTE_SCHEMA],
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
                if block.name == COMPUTE_TOOL:
                    # Validated against what has been retrieved so far, so a
                    # compute call in an earlier round than its inputs errors out
                    # rather than laundering an invented number.
                    result = run_compute(block.input, tool_results)
                else:
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
