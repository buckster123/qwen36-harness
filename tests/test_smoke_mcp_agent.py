"""Live agentic-MCP smoke: model uses an MCP-registered tool via the agent loop.

This is the actual integration test we care about: the model, via tool calls,
invokes the filesystem MCP server (not our builtin fs.* tools). End-to-end:
LLM → tool_call → MCPManager → JSON-RPC → fs-mcp subprocess → response.

Opt-in: HARNESS_MCP_SMOKE=1 + HARNESS_SMOKE_URL set. Requires npx + a live
qwen36 endpoint via the SSH tunnel.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.agent import Agent, AgentLimits
from harness.client import HarnessClient
from harness.config import Endpoint
from harness.mcp import MCPManager, MCPServerConfig
from harness.tools import Registry

pytestmark = pytest.mark.skipif(
    os.environ.get("HARNESS_MCP_SMOKE") != "1" or not os.environ.get("HARNESS_SMOKE_URL"),
    reason="set HARNESS_MCP_SMOKE=1 and HARNESS_SMOKE_URL to run live agentic MCP smoke",
)


@pytest.mark.asyncio
async def test_model_calls_mcp_filesystem_tool(tmp_path: Path) -> None:
    # Seed sandbox
    (tmp_path / "secret.txt").write_text("the magic word is OCTOPUS\n", encoding="utf-8")

    reg = Registry()
    mgr = MCPManager(reg)
    cfg = MCPServerConfig(
        name="fs-mcp",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
        connect_timeout=120.0,
    )
    ep = Endpoint(
        name="smoke-mcp",
        base_url=os.environ["HARNESS_SMOKE_URL"],
        model=os.environ.get("HARNESS_SMOKE_MODEL", "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"),
        mode="nonthinking",
        default_max_tokens=2048,
        default_temperature=0.0,
    )
    client = HarnessClient(ep)
    agent = Agent(client, registry=reg, limits=AgentLimits(max_turns=4))
    try:
        await mgr.start(cfg)
        names = mgr.tools_for("fs-mcp")
        assert names

        # Compose a tight prompt that forces the model toward the MCP tool.
        prompt = (
            f"You have filesystem MCP tools available (names start with 'fs-mcp.'). "
            f"Read the file at '{tmp_path}/secret.txt' and report the magic word "
            f"in the form: MAGIC=<word>"
        )
        full_text = ""
        tool_calls_seen: list[str] = []
        async for ev in agent.run([{"role": "user", "content": prompt}]):
            if ev.kind == "content" and ev.text:
                full_text += ev.text
            if ev.kind == "tool_call" and ev.data:
                tool_calls_seen.append(ev.data.get("name", ""))
        assert tool_calls_seen, "model never called any tool"
        assert all(n.startswith("fs-mcp.") for n in tool_calls_seen), (
            f"model called non-MCP tools: {tool_calls_seen}"
        )
        assert "OCTOPUS" in full_text.upper(), f"model didn't surface secret: {full_text!r}"
    finally:
        await mgr.stop_all()
        await client.aclose()
