"""Agent loop smoke test — calc + fs against a live endpoint.

Set ``HARNESS_SMOKE_URL=http://127.0.0.1:8800/v1`` to enable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from harness.agent import Agent, AgentLimits
from harness.client import HarnessClient
from harness.config import Endpoint
from harness.tools import Registry
from harness.tools.calc import register as register_calc
from harness.tools.filesystem import FsSandbox, register as register_fs

URL = os.environ.get("HARNESS_SMOKE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="HARNESS_SMOKE_URL not set")


def _ep(mode: str = "nonthinking") -> Endpoint:
    return Endpoint(
        name="smoke",
        base_url=URL or "",
        model=os.environ.get("HARNESS_SMOKE_MODEL", "auto"),
        default_max_tokens=4096,
        default_temperature=0.3,
        mode=mode,
    )


@pytest.mark.asyncio
async def test_agent_uses_calc_tool() -> None:
    registry = Registry()
    register_calc(registry)

    async with HarnessClient(_ep()) as client:
        agent = Agent(client, registry=registry, limits=AgentLimits(max_turns=4))
        events = [
            e
            async for e in agent.run(
                [
                    {
                        "role": "system",
                        "content": (
                            "You have a calc.eval tool. ALWAYS use it for arithmetic. "
                            "After getting the tool result, give a one-sentence answer."
                        ),
                    },
                    {"role": "user", "content": "What is 17 multiplied by 91?"},
                ]
            )
        ]

    tool_calls = [e for e in events if e.kind == "tool_call"]
    tool_results = [e for e in events if e.kind == "tool_result"]
    final_content = "".join(e.text for e in events if e.kind == "content")

    assert tool_calls, f"model did not call any tool. events: {[e.kind for e in events]}"
    assert any(tc.data["name"] == "calc.eval" for tc in tool_calls)
    assert any("1547" in tr.data["output"] for tr in tool_results)
    assert "1547" in final_content


@pytest.mark.asyncio
async def test_agent_writes_then_reads_a_file() -> None:
    registry = Registry()
    with tempfile.TemporaryDirectory() as tmp:
        register_fs(registry, FsSandbox(Path(tmp)))

        async with HarnessClient(_ep()) as client:
            agent = Agent(client, registry=registry, limits=AgentLimits(max_turns=6))
            events = [
                e
                async for e in agent.run(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You have fs.write, fs.read, fs.list tools that operate "
                                "on a sandbox directory. Use them to fulfill the user's "
                                "request without asking for confirmation."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Write the text 'hello fren' to a file named "
                                "greeting.txt in the sandbox, then read it back and "
                                "tell me what's in it."
                            ),
                        },
                    ]
                )
            ]

    tool_calls = [e for e in events if e.kind == "tool_call"]
    names_called = [tc.data["name"] for tc in tool_calls]
    final = "".join(e.text for e in events if e.kind == "content")

    assert "fs.write" in names_called, f"never wrote. names={names_called}"
    assert "fs.read" in names_called, f"never read back. names={names_called}"
    assert "hello fren" in final.lower() or "greeting" in final.lower()
