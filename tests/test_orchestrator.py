import json
from unittest.mock import patch

from agent.orchestrator import run_agent


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeLLMClient:
    """Returns a scripted sequence of messages, one per .chat() call."""

    def __init__(self, scripted_messages):
        self._messages = list(scripted_messages)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._messages.pop(0)


def _fake_web_search(query):
    return [{"title": "Example result", "url": "https://example.com/a", "snippet": "..."}]


def test_agent_terminates_when_model_returns_no_tool_calls():
    client = FakeLLMClient([FakeMessage(content="Final answer, no research needed.")])
    result = run_agent("trivial question", llm_client=client, max_steps=8)

    assert result["answer"] == "Final answer, no research needed."
    assert result["steps"] == 1
    assert result["sources"] == []
    assert len(client.calls) == 1


def test_agent_executes_tool_call_then_returns_final_answer():
    tool_call = FakeToolCall("call_1", "web_search", json.dumps({"query": "test"}))
    client = FakeLLMClient(
        [
            FakeMessage(content=None, tool_calls=[tool_call]),
            FakeMessage(content="Based on my research, the answer is X."),
        ]
    )

    with patch.dict("agent.orchestrator.TOOL_REGISTRY", {"web_search": _fake_web_search}):
        result = run_agent("real question", llm_client=client, max_steps=8)

    assert result["answer"] == "Based on my research, the answer is X."
    assert result["steps"] == 2
    assert result["sources"] == [{"url": "https://example.com/a", "title": "Example result"}]
    assert "## Sources" in result["report"]

    # Second call to the LLM must include the tool result as a "tool" message
    second_call_messages = client.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"


def test_agent_forces_final_answer_after_max_steps():
    tool_call = FakeToolCall("call_1", "web_search", json.dumps({"query": "test"}))
    # Every scripted turn keeps calling the tool, never producing a final answer on its own.
    scripted = [FakeMessage(content=None, tool_calls=[tool_call]) for _ in range(2)]
    scripted.append(FakeMessage(content="Forced final answer."))
    client = FakeLLMClient(scripted)

    with patch.dict("agent.orchestrator.TOOL_REGISTRY", {"web_search": _fake_web_search}):
        result = run_agent("endless question", llm_client=client, max_steps=2)

    assert result["answer"] == "Forced final answer."
    assert result["steps"] == 2
    assert any(entry.get("forced") for entry in result["trace"])


def test_unknown_tool_name_is_reported_as_error_not_a_crash():
    tool_call = FakeToolCall("call_1", "not_a_real_tool", "{}")
    client = FakeLLMClient(
        [
            FakeMessage(content=None, tool_calls=[tool_call]),
            FakeMessage(content="done"),
        ]
    )
    result = run_agent("question", llm_client=client, max_steps=8)
    tool_trace = [e for e in result["trace"] if e["type"] == "tool_call"][0]
    assert "error" in tool_trace["result"]
