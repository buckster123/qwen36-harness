"""Tests for harness.client — uses respx to mock the llama-server endpoint."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from harness.client import HarnessClient, _drain_messages
from harness.config import Endpoint


def _ep() -> Endpoint:
    return Endpoint(
        name="test",
        base_url="http://test.local/v1",
        model="qwen.gguf",
        api_key="sk-test",
        default_max_tokens=128,
        default_temperature=0.5,
        mode="thinking",
    )


def _sse(chunks: list[dict]) -> str:
    """Render a list of dicts as an SSE stream the way llama-server does."""
    out = []
    for c in chunks:
        out.append(f"data: {json.dumps(c)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out)


# ---------------- non-streaming ----------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_dual_content() -> None:
    client = HarnessClient(_ep())
    payload = {
        "id": "x",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "A hash table is...",
                    "reasoning_content": "Let me think about this.",
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        result = await client.complete([{"role": "user", "content": "What is a hash table?"}])

    assert result.content == "A hash table is..."
    assert result.reasoning_content == "Let me think about this."
    assert result.finish_reason == "stop"
    assert result.usage["completion_tokens"] == 20
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_passes_tools_and_parses_tool_calls() -> None:
    client = HarnessClient(_ep())
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Reykjavik","unit":"c"}',
                            },
                        }
                    ],
                },
            }
        ],
    }
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=payload)

    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        result = await client.complete(
            [{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
        )

    body = captured["body"]
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["tool_choice"] == "auto"
    assert body["max_tokens"] == 128  # endpoint default
    assert body["temperature"] == 0.5

    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "get_weather"
    assert call["arguments"] == {"city": "Reykjavik", "unit": "c"}
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_handles_unparseable_tool_args() -> None:
    client = HarnessClient(_ep())
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{not json"},
                        }
                    ],
                },
            }
        ],
    }
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        result = await client.complete([{"role": "user", "content": "x"}])
    assert result.tool_calls[0]["arguments"]["_error"]
    assert result.tool_calls[0]["arguments"]["_raw"] == "{not json"
    await client.aclose()


@pytest.mark.asyncio
async def test_nonthinking_mode_sets_template_kwarg() -> None:
    ep = Endpoint(
        name="t", base_url="http://test.local/v1", model="m", mode="nonthinking",
    )
    client = HarnessClient(ep)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        await client.complete([{"role": "user", "content": "x"}])

    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    await client.aclose()


# ---------------- streaming ---------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_dual_channel_events() -> None:
    client = HarnessClient(_ep())
    chunks = [
        {"id": "abc", "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"reasoning_content": "Think "}}]},
        {"choices": [{"delta": {"reasoning_content": "more."}}]},
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 5}},
    ]
    body = _sse(chunks)
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
        events = [ev async for ev in await client.stream([{"role": "user", "content": "hi"}])]

    kinds = [e.kind for e in events]
    assert kinds[0] == "start"
    assert "reasoning" in kinds and "content" in kinds and "usage" in kinds and "done" in kinds

    drained = _drain_messages(events)
    assert drained.content == "Hello world."
    assert drained.reasoning_content == "Think more."
    assert drained.finish_reason == "stop"
    assert drained.usage["completion_tokens"] == 5
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_aggregates_tool_calls_across_chunks() -> None:
    client = HarnessClient(_ep())
    chunks = [
        {"id": "x", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "get_w", "arguments": '{"ci'}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ty":"Oslo"}'}}
        ]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    body = _sse(chunks)
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
        events = [ev async for ev in await client.stream([{"role": "user", "content": "weather?"}])]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert len(tool_events) == 1
    tc = tool_events[0].data
    assert tc["name"] == "get_w"
    assert tc["arguments"] == {"city": "Oslo"}
    assert tc["id"] == "c1"
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_surfaces_http_error() -> None:
    client = HarnessClient(_ep())
    with respx.mock(base_url="http://test.local") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="boom")
        )
        events = [ev async for ev in await client.stream([{"role": "user", "content": "x"}])]
    assert any(e.kind == "error" for e in events)
    await client.aclose()
