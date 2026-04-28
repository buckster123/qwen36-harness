"""Tests for the agent tool-loop driver."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from harness.agent import Agent, AgentLimits
from harness.client import HarnessClient
from harness.config import Endpoint
from harness.tools import Registry, ToolSpec


def _ep() -> Endpoint:
    return Endpoint(
        name="t",
        base_url="http://test.local/v1",
        model="qwen.gguf",
        default_max_tokens=128,
    )


def _sse(chunks: list[dict]) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_agent_terminates_on_first_final_answer() -> None:
    client = HarnessClient(_ep())
    registry = Registry()  # no tools
    agent = Agent(client, registry=registry)

    chunks = [
        {"id": "x", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello!"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=_sse(chunks))
        )
        events = [e async for e in agent.run([{"role": "user", "content": "hi"}])]

    kinds = [e.kind for e in events]
    assert kinds.count("llm_start") == 1
    assert "content" in kinds
    # stats is now the last event (comes after done)
    assert kinds[-1] == "stats"
    done_idx = kinds.index("done")
    assert done_idx == len(kinds) - 2
    assert events[done_idx].data["reason"] == "final"
    await client.aclose()


@pytest.mark.asyncio
async def test_agent_dispatches_tool_then_continues() -> None:
    client = HarnessClient(_ep())
    registry = Registry()
    registry.register(
        ToolSpec(
            name="add",
            description="add",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            fn=lambda a, b: a + b,
        )
    )

    # Turn 1: model emits tool_call
    turn1 = [
        {"id": "1", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [
            {"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "type": "function",
                 "function": {"name": "add", "arguments": '{"a":2,"b":3}'}}
            ]}}
        ]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    # Turn 2: model gives final answer
    turn2 = [
        {"id": "2", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "The sum is 5."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]

    bodies = [_sse(turn1), _sse(turn2)]
    call_count = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        body = bodies[call_count["n"]]
        call_count["n"] += 1
        return httpx.Response(200, text=body)

    agent = Agent(client, registry=registry)
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        events = [e async for e in agent.run([{"role": "user", "content": "what is 2+3"}])]

    kinds = [e.kind for e in events]
    assert kinds.count("llm_start") == 2  # two LLM turns
    tool_results = [e for e in events if e.kind == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].data["output"] == "5"
    final_content = "".join(e.text for e in events if e.kind == "content")
    assert "5" in final_content
    # done is second-to-last (stats comes after)
    done_idx = kinds.index("done")
    assert done_idx == len(events) - 2
    assert events[done_idx].data["reason"] == "final"
    assert call_count["n"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_agent_respects_max_turns() -> None:
    client = HarnessClient(_ep())
    registry = Registry()
    registry.register(
        ToolSpec(
            name="loop", description="x", parameters={}, fn=lambda: "again"
        )
    )

    # Every response asks for a tool call → would loop forever without max_turns
    turn = [
        {"id": "x", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [
            {"delta": {"tool_calls": [
                {"index": 0, "id": "c", "type": "function",
                 "function": {"name": "loop", "arguments": "{}"}}
            ]}}
        ]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]

    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(turn))

    agent = Agent(client, registry=registry, limits=AgentLimits(max_turns=3))
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        events = [e async for e in agent.run([{"role": "user", "content": "go"}])]

    # done is second-to-last (stats comes after)
    kinds = [e.kind for e in events]
    done_idx = kinds.index("done")
    assert done_idx == len(events) - 2
    assert events[done_idx].data["reason"] == "turn_limit"
    assert events[done_idx].data["turns"] == 3
    await client.aclose()
