"""Tests for harness.tools.subagent — SubAgentManager lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.harness.client import HarnessClient
from src.harness.tools import Registry
from src.harness.tools.subagent import SubAgentManager


@pytest.fixture
def registry() -> Registry:
    reg = Registry()
    # Register a dummy tool so export_schema works
    reg.tool(
        name="dummy",
        description="A test tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )(lambda: "ok")
    return reg


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=HarnessClient)
    # Mock complete() to return a result with usage
    mock_result = MagicMock()
    mock_result.content = "Test response from sub-agent"
    mock_result.usage = {"prompt_tokens": 10, "completion_tokens": 20}
    client.complete = AsyncMock(return_value=mock_result)
    return client


@pytest.fixture
def manager(mock_client, registry) -> SubAgentManager:
    return SubAgentManager(mock_client, registry)


@pytest.mark.asyncio
async def test_spawn_and_result(manager):
    """Test basic spawn → result lifecycle."""
    result = await manager.spawn("tester", "Do a thing")
    assert result["agent_id"] == "agent-1"
    assert result["role"] == "tester"
    assert result["status"] in ("initializing", "running")

    # Wait for task to complete
    await asyncio.sleep(0.1)

    # Check result
    res = await manager.result("agent-1")
    assert res["status"] == "done"
    assert res["result"] == "Test response from sub-agent"


@pytest.mark.asyncio
async def test_spawn_max_concurrent(manager):
    """Test that we cannot exceed max concurrent agents."""
    # Spawn 3 (should work)
    for i in range(3):
        r = await manager.spawn("worker", f"Task {i}")
        assert "error" not in r, f"Failed to spawn agent-{i+1}"

    # 4th should fail
    r = await manager.spawn("overflow", "Too many")
    assert "error" in r or "Cannot spawn" in str(r)


@pytest.mark.asyncio
async def test_cancel(manager):
    """Test cancelling transitions state correctly."""
    # Manually set running state to simulate in-flight task
    mgr = manager
    from src.harness.tools.subagent import _AgentState, asyncio
    import uuid
    
    # Create a fake agent that's running
    mgr._agents["fake-1"] = _AgentState("fake-1", "worker", "task", None)
    mgr._agents["fake-1"].status = "running"
    
    # Cancel it
    r = await mgr.cancel("fake-1")
    assert r.get("status") == "cancelled" or "cancelled" in str(r).lower()


@pytest.mark.asyncio
async def test_unknown_agent(manager):
    """Test result/cancel on non-existent agent."""
    r = await manager.result("nonexistent")
    assert "error" in r

    r = await manager.cancel("nonexistent")
    assert "error" in r


@pytest.mark.asyncio
async def test_list_agents(manager):
    """Test listing multiple agents."""
    await manager.spawn("alpha", "First task")
    await manager.spawn("beta", "Second task")
    await asyncio.sleep(0.1)

    agents = manager.list_agents()
    assert len(agents) == 2
    assert any(a["role"] == "alpha" for a in agents)
    assert any(a["role"] == "beta" for a in agents)


@pytest.mark.asyncio
async def test_timeout(manager, mock_client):
    """Test that slow tasks get marked as timeout."""
    # Make complete() hang forever
    async def _hang():
        await asyncio.sleep(10_000)

    mock_client.complete = AsyncMock(side_effect=_hang)

    r = await manager.spawn("slow", "Take your time", timeout_s=1)
    assert r["agent_id"] == "agent-1"

    # Wait for timeout to fire
    await asyncio.sleep(1.5)

    res = await manager.result("agent-1")
    assert res["status"] == "timeout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
