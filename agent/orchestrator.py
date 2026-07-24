"""The agent loop: plan -> call tools -> observe -> repeat -> final cited answer.

This is a small hand-rolled ReAct-style loop (not a framework like
LangGraph/CrewAI) so the control flow — how tool calls are dispatched, how
sources are tracked independently of what the model remembers to cite, and
how the loop terminates — is fully visible and testable rather than hidden
inside a library.
"""
from __future__ import annotations

import json

from agent.config import MAX_STEPS
from agent.llm_client import GroqLLMClient, LLMClient
from agent.report import SourceStore, build_report
from agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are a research assistant. Given a research question:
1. Break it into sub-questions if useful.
2. Use the web_search tool to find relevant sources, then use fetch_page on
   the most promising URLs to read them in full before relying on them.
3. Keep researching until you have enough evidence to answer confidently,
   using at most a handful of tool calls.
4. When ready, respond with your final answer as plain markdown prose
   (headings/bullets welcome). Do NOT write your own "Sources" section —
   a verified reference list is appended automatically from the pages you
   actually fetched. Be concise and factual; do not state anything you did
   not find in a tool result.
"""


def _tool_call_to_message_dict(tool_calls) -> list[dict]:
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in tool_calls
    ]


def _record_sources(sources: SourceStore, tool_name: str, result) -> None:
    if tool_name == "web_search" and isinstance(result, list):
        for r in result:
            if isinstance(r, dict) and r.get("url"):
                sources.add(r["url"], r.get("title", ""))
    elif tool_name == "fetch_page" and isinstance(result, dict) and result.get("url") and "error" not in result:
        sources.add(result["url"])


def _execute_tool_call(tc) -> dict:
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"name": name, "args": args, "result": {"error": f"unknown tool '{name}'"}}
    return {"name": name, "args": args, "result": fn(**args)}


def run_agent(question: str, llm_client: LLMClient | None = None, max_steps: int = MAX_STEPS) -> dict:
    """Run the research loop and return {answer, report, sources, steps, trace}."""
    client = llm_client or GroqLLMClient()
    sources = SourceStore()
    trace: list[dict] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        message = client.chat(messages, tools=TOOL_SCHEMAS)
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            answer = message.content or ""
            trace.append({"type": "final", "content": answer})
            return _finalize(answer, sources, step + 1, trace)

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": _tool_call_to_message_dict(tool_calls),
            }
        )

        for tc in tool_calls:
            call = _execute_tool_call(tc)
            _record_sources(sources, call["name"], call["result"])
            trace.append({"type": "tool_call", **call})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(call["result"])[:8000],
                }
            )

    # Ran out of steps: force a final answer from whatever evidence was gathered.
    messages.append(
        {
            "role": "user",
            "content": (
                "You have used all available research steps. Write your final "
                "answer now, based only on the evidence gathered so far."
            ),
        }
    )
    final_message = client.chat(messages, tools=None)
    answer = final_message.content or ""
    trace.append({"type": "final", "content": answer, "forced": True})
    return _finalize(answer, sources, max_steps, trace)


def _finalize(answer: str, sources: SourceStore, steps: int, trace: list[dict]) -> dict:
    return {
        "answer": answer,
        "report": build_report(answer, sources.as_list()),
        "sources": sources.as_list(),
        "steps": steps,
        "trace": trace,
    }
