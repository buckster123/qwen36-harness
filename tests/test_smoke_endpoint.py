"""Real-endpoint smoke test against a live Qwen3.6 endpoint.

Skipped unless ``HARNESS_SMOKE_URL`` is set to a base_url like
``http://127.0.0.1:8800/v1``. CI never runs this; you do, locally, when you
want to confirm a freshly-spun-up Vast box is wired correctly.
"""

from __future__ import annotations

import os

import pytest

from harness.client import HarnessClient
from harness.config import Endpoint

URL = os.environ.get("HARNESS_SMOKE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="HARNESS_SMOKE_URL not set")


def _ep() -> Endpoint:
    return Endpoint(
        name="smoke",
        base_url=URL or "",
        model=os.environ.get("HARNESS_SMOKE_MODEL", "auto"),
        default_max_tokens=2048,
        default_temperature=0.7,
        mode="thinking",
    )


@pytest.mark.asyncio
async def test_smoke_complete() -> None:
    async with HarnessClient(_ep()) as client:
        result = await client.complete(
            [{"role": "user", "content": "In one sentence, what is gradient descent?"}]
        )
    assert result.content, f"empty content; reasoning={result.reasoning_content[:120]!r}"
    assert result.finish_reason in ("stop", "tool_calls"), result.finish_reason


@pytest.mark.asyncio
async def test_smoke_stream() -> None:
    # Use nonthinking mode for this test — pure content stream, no thinking burn.
    ep = _ep()
    nonthinking_ep = Endpoint(
        name=ep.name, base_url=ep.base_url, model=ep.model, api_key=ep.api_key,
        default_max_tokens=ep.default_max_tokens,
        default_temperature=ep.default_temperature, mode="nonthinking",
    )
    async with HarnessClient(nonthinking_ep) as client:
        events = [
            ev
            async for ev in await client.stream(
                [{"role": "user", "content": "Say hello in five words."}],
                max_tokens=128,
            )
        ]
    kinds = [e.kind for e in events]
    assert "start" in kinds and "done" in kinds
    text = "".join(e.text for e in events if e.kind == "content")
    assert text.strip(), f"empty content stream; events: {kinds}"


@pytest.mark.asyncio
async def test_smoke_tool_call() -> None:
    async with HarnessClient(_ep()) as client:
        result = await client.complete(
            [{"role": "user", "content": "What is the weather in Reykjavik right now?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=2048,
        )
    assert result.tool_calls, "model did not invoke a tool"
    call = result.tool_calls[0]
    assert call["name"] == "get_weather"
    assert call["arguments"].get("city", "").lower().startswith("reykjav")
