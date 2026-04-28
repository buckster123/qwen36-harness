"""Unit tests for harness.mcp — config loader + manager with mocked session.

The live MCP smoke test (against actual mcp-server-filesystem subprocess)
lives in ``test_smoke_mcp.py`` and is opt-in via env var.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harness.mcp import MCPManager, MCPServerConfig, _flatten_call_result, load_mcp_config
from harness.tools import Registry


# --- load_mcp_config -----------------------------------------------------------


def test_load_config_missing_returns_empty(tmp_path: Path) -> None:
    cfg = load_mcp_config(tmp_path / "does_not_exist.toml")
    assert cfg == {}


def test_load_config_parses_servers(tmp_path: Path) -> None:
    p = tmp_path / "mcp_servers.toml"
    p.write_text(
        """
[servers.fs-mcp]
command = ["uvx", "mcp-server-filesystem", "/tmp"]
auto_start = true

[servers.gh]
command = ["npx", "-y", "@modelcontextprotocol/server-github"]
auto_start = false
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_x" }
connect_timeout = 45.0
""".strip(),
        encoding="utf-8",
    )
    cfg = load_mcp_config(p)
    assert set(cfg.keys()) == {"fs-mcp", "gh"}
    assert cfg["fs-mcp"].command == ["uvx", "mcp-server-filesystem", "/tmp"]
    assert cfg["fs-mcp"].auto_start is True
    assert cfg["gh"].auto_start is False
    assert cfg["gh"].env == {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_x"}
    assert cfg["gh"].connect_timeout == 45.0


# --- _flatten_call_result ------------------------------------------------------


def test_flatten_text_blocks() -> None:
    res = SimpleNamespace(
        content=[SimpleNamespace(text="hello"), SimpleNamespace(text="world")],
        isError=False,
        structuredContent=None,
    )
    assert _flatten_call_result(res) == "hello\nworld"


def test_flatten_marks_errors() -> None:
    res = SimpleNamespace(
        content=[SimpleNamespace(text="boom")], isError=True, structuredContent=None
    )
    assert _flatten_call_result(res).startswith("[mcp error]")


def test_flatten_falls_back_to_structured() -> None:
    res = SimpleNamespace(
        content=[],
        isError=False,
        structuredContent={"answer": 42},
    )
    out = _flatten_call_result(res)
    assert "42" in out


# --- MCPManager._register_tools (no subprocess; mock session) ------------------


@pytest.mark.asyncio
async def test_register_tools_namespaces_into_registry() -> None:
    """list_tools returns 2 tools → registry gains 'mock.alpha' and 'mock.beta'."""
    reg = Registry()
    mgr = MCPManager(reg)

    fake_tools = [
        SimpleNamespace(
            name="alpha",
            description="alpha tool",
            inputSchema={"type": "object", "properties": {"x": {"type": "integer"}}},
        ),
        SimpleNamespace(
            name="beta",
            description=None,
            inputSchema=None,
        ),
    ]
    fake_result = SimpleNamespace(content=[SimpleNamespace(text="ok")], isError=False)

    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=fake_tools)),
        call_tool=AsyncMock(return_value=fake_result),
    )

    registered = await mgr._register_tools("mock", session)
    assert registered == ["mock.alpha", "mock.beta"]
    assert reg.get("mock.alpha").description == "alpha tool"
    # Description fallback when MCP tool returns description=None
    assert "MCP tool beta" in reg.get("mock.beta").description
    # Schema falls back to empty object schema
    assert reg.get("mock.beta").parameters == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_dispatch_round_trips_through_call_tool() -> None:
    reg = Registry()
    mgr = MCPManager(reg)
    fake_tools = [
        SimpleNamespace(
            name="echo",
            description="echo back",
            inputSchema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        )
    ]
    fake_result = SimpleNamespace(
        content=[SimpleNamespace(text="echoed: hi")], isError=False, structuredContent=None
    )
    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=fake_tools)),
        call_tool=AsyncMock(return_value=fake_result),
    )
    await mgr._register_tools("mock", session)

    result = await reg.dispatch("mock.echo", {"msg": "hi"})
    assert not result.is_error
    assert result.output == "echoed: hi"
    session.call_tool.assert_awaited_once_with("echo", {"msg": "hi"})


@pytest.mark.asyncio
async def test_dispatch_surfaces_call_tool_exception() -> None:
    reg = Registry()
    mgr = MCPManager(reg)
    fake_tools = [
        SimpleNamespace(name="fail", description="x", inputSchema={"type": "object"})
    ]
    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=fake_tools)),
        call_tool=AsyncMock(side_effect=RuntimeError("upstream gone")),
    )
    await mgr._register_tools("mock", session)
    result = await reg.dispatch("mock.fail", {})
    assert result.is_error
    assert "upstream gone" in result.output


@pytest.mark.asyncio
async def test_collision_does_not_crash() -> None:
    """If two MCP servers register a tool with the same full name, the
    second registration is logged + skipped, not raised."""
    reg = Registry()
    mgr = MCPManager(reg)
    fake_tools = [SimpleNamespace(name="a", description="x", inputSchema={"type": "object"})]
    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=fake_tools)),
        call_tool=AsyncMock(),
    )
    first = await mgr._register_tools("mock", session)
    assert first == ["mock.a"]
    second = await mgr._register_tools("mock", session)
    assert second == []  # collision, dropped
    # Original survives
    assert "mock.a" in reg.names()


# --- start() failure paths -----------------------------------------------------


@pytest.mark.asyncio
async def test_start_with_empty_command_fails_cleanly() -> None:
    reg = Registry()
    mgr = MCPManager(reg)
    cfg = MCPServerConfig(name="bad", command=[], connect_timeout=2.0)
    with pytest.raises(RuntimeError, match="empty command"):
        await mgr.start(cfg)
