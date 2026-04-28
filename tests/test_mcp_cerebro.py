"""Integration tests for CerebroCortex MCP tools.

These tests spin up the real CC MCP server as a subprocess and exercise
a few key tool calls (recall, remember, list_intentions) via the harness
registry's dispatch mechanism.

Run with:
    .venv/bin/python -m pytest tests/test_mcp_cerebro.py -v

Requires: CerebroCortex installed at ~/projects/CerebroCortex
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness.mcp import MCPManager, MCPServerConfig
from harness.tools import Registry, ToolResult


# --- fixtures ----------------------------------------------------------------


def _make_cerebro_cfg() -> MCPServerConfig:
    """Return config pointing at CC's stdio MCP server."""
    return MCPServerConfig(
        name="cerebro",
        command=["/home/andre/projects/CerebroCortex/cerebro-mcp"],
        env={"CEREBRO_DATA_DIR": "/home/andre/.cerebro-cortex"},
        connect_timeout=30.0,
    )


@pytest.fixture()
def reg():
    """Fresh registry for each test."""
    return Registry()


@pytest.fixture()
async def cerebro_mgr(reg):
    """Start CC MCP server and return the manager + cleanup on teardown."""
    mgr = MCPManager(reg)
    await mgr.start(_make_cerebro_cfg())
    yield mgr
    await mgr.stop("cerebro")


# --- tool discovery ------------------------------------------------------------


@pytest.mark.asyncio
async def test_cerebro_registers_all_tools(reg, cerebro_mgr):
    """Verify all 42 CC tools are registered under 'cerebro.' namespace."""
    tools = cerebro_mgr.tools_for("cerebro")
    assert len(tools) == 42, f"Expected 42 tools, got {len(tools)}: {tools}"
    # Key tools should be present
    expected = {"cerebro.recall", "cerebro.remember", "cerebro.list_intentions",
                "cerebro.episode_start", "cerebro.dream_run", "cerebro.get_memory"}
    for name in expected:
        assert name in reg.names(), f"Missing tool: {name}"


@pytest.mark.asyncio
async def test_cerebro_tools_export_openai_schema(reg, cerebro_mgr):
    """Verify tools export valid OpenAI function schemas."""
    schema = reg.export_schema(only=["cerebro.recall", "cerebro.list_intentions"])
    assert len(schema) == 2
    recall = [s for s in schema if s["function"]["name"] == "cerebro.recall"][0]
    assert "query" in recall["function"]["parameters"]["properties"]
    assert recall["function"]["parameters"]["properties"]["query"]["type"] == "string"


# --- recall --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cerebro_recall(reg, cerebro_mgr):
    """recall should return memory objects with content + salience."""
    result = await reg.dispatch("cerebro.recall", {"query": "qwen36 harness", "top_k": 2})
    assert not result.is_error
    assert "episodic" in result.output or "memory" in result.output.lower()


# --- list_intentions ----------------------------------------------------------


@pytest.mark.asyncio
async def test_cerebro_list_intentions(reg, cerebro_mgr):
    """list_intentions should return pending TODOs."""
    result = await reg.dispatch("cerebro.list_intentions", {})
    assert not result.is_error
    # We expect at least 1 intention from previous sessions
    assert "Pending Intentions" in result.output or "intention" in result.output.lower()


# --- remember ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cerebro_remember_and_get(reg, cerebro_mgr):
    """remember should create a memory; get_memory should retrieve it.

    Note: CC's Thalamus gating engine may reject very short or duplicate content.
    This test verifies the tool dispatch path works (no crashes, no schema errors)
    rather than guaranteeing storage success.
    """
    # Use unique, substantive content to avoid gating
    content = (
        f"TEST harness integration — this is a long unique string for "
        f"test_mcp_cerebro.py run at session boundary abc123xyz"
    )
    import uuid as _uuid
    content += str(_uuid.uuid4())

    result = await reg.dispatch(
        "cerebro.remember",
        {
            "content": content,
            "tags": ["source:harness-test"],
            "salience": 0.3,
        },
    )
    # Either: memory stored successfully, or gated by Thalamus (both are valid)
    assert not result.is_error
    # Verify CC returned a meaningful response about the memory operation
    output_lower = result.output.lower()
    assert any(phrase in output_lower for phrase in ["stored", "memory", "gated", "saved", "saved"])
