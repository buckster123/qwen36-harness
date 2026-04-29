"""Sub-agent orchestration tools.

Spawns helper agents that run in separate llama.cpp slots for parallel work.
Each sub-agent gets its own isolated conversation with configurable tool access.

Usage from the model:
    agent.spawn(role="researcher", prompt="Find docs for X library")
    agent.list()
    agent.result(agent_id)  # poll for result
    agent.cancel(agent_id)  # stop early
"""

from __future__ import annotations

import asyncio
import itertools
import uuid
from datetime import datetime, timezone
from typing import Any

from ..client import HarnessClient
from . import Registry, default_registry


# ---------------------------------------------------------------------------
# Sub-agent state machine
# ---------------------------------------------------------------------------

class _AgentState:
    """Tracks a single sub-agent's lifecycle."""

    def __init__(self, agent_id: str, role: str, prompt: str, tools: list[str] | None) -> None:
        self.agent_id = agent_id
        self.role = role
        self.prompt = prompt
        self.tools = tools  # None = full access
        self.status: str = "initializing"  # initializing/running/done/cancelled/timeout/error
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: str | None = None
        self.result: str = ""
        self.error: str = ""
        self.turns: int = 0
        self.tokens_used: int = 0


# ---------------------------------------------------------------------------
# Manager — tracks and runs sub-agents
# ---------------------------------------------------------------------------

class SubAgentManager:
    """Manages a pool of sub-agents running against a shared LLM endpoint."""

    MAX_CONCURRENT = 3  # Reserve 1 slot for main agent

    def __init__(self, client: HarnessClient, registry: Registry) -> None:
        self.client = client
        self.registry = registry
        self._agents: dict[str, _AgentState] = {}
        self._tasks: dict[str, asyncio.Task] = {}  # Cancel handle
        self._counter = itertools.count(1)

    def _new_id(self) -> str:
        return f"agent-{next(self._counter)}"

    async def spawn(
        self,
        role: str,
        prompt: str,
        *,
        tools: list[str] | None = None,
        timeout_s: int = 300,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict:
        """Spawn a new sub-agent in a separate slot."""
        if len([a for a in self._agents.values() if a.status in ("initializing", "running")]) >= self.MAX_CONCURRENT:
            return {
                "error": f"Cannot spawn: max {self.MAX_CONCURRENT} concurrent agents reached. "
                         "Wait for existing agents to finish or cancel them."
            }

        agent_id = self._new_id()
        state = _AgentState(agent_id, role, prompt, tools)
        self._agents[agent_id] = state

        # Build isolated conversation
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a '{role}' sub-agent working on a delegated task. "
                    f"Be concise and focused. Return your work as structured output."
                )
            },
            {"role": "user", "content": prompt},
        ]

        # Start async task
        task = asyncio.create_task(
            self._run_session(agent_id, messages, tools, max_tokens, temperature)
        )
        self._tasks[agent_id] = task

        # Set timeout
        def _timeout_handler():
            state.status = "timeout"
            state.error = f"Sub-agent exceeded {timeout_s}s time limit"
            state.finished_at = datetime.now(timezone.utc).isoformat()

        try:
            asyncio.get_event_loop().call_later(timeout_s, lambda: (_timeout_handler(), None)[0])
        except Exception:
            pass  # Best-effort timeout

        return {
            "agent_id": agent_id,
            "role": role,
            "status": state.status,
            "created_at": state.created_at,
            "message": f"Sub-agent spawned. Use agent.result('{agent_id}') to check progress."
        }

    async def _run_session(
        self,
        agent_id: str,
        messages: list[dict[str, Any]],
        tools: list[str] | None,
        max_tokens: int,
        temperature: float,
    ) -> None:
        """Run a single-turn LLM session for a sub-agent."""
        state = self._agents.get(agent_id)
        if not state or state.status == "cancelled":
            return

        state.status = "running"
        # Small startup delay so cancel can interleave
        await asyncio.sleep(0.01)
        if state.status == "cancelled":
            return

        try:
            # Get tool schema if restricted tools are specified
            tools_schema = None
            if tools:
                tools_schema = self.registry.export_schema(only=tools)

            # Send to LLM
            result = await self.client.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools_schema,
            )

            state.result = result.content or ""
            state.turns = 1
            state.tokens_used = sum(result.usage.values()) if result.usage else 0
            state.status = "done"
            state.finished_at = datetime.now(timezone.utc).isoformat()

        except asyncio.CancelledError:
            state.status = "cancelled"
            state.error = "Task was cancelled"
            state.finished_at = datetime.now(timezone.utc).isoformat()
            raise  # Re-raise to cancel the task

        except Exception as e:
            state.status = "error"
            state.error = f"{type(e).__name__}: {e}"
            state.finished_at = datetime.now(timezone.utc).isoformat()

    async def send(self, agent_id: str, message: str) -> dict:
        """Send a follow-up message to a running sub-agent."""
        state = self._agents.get(agent_id)
        if not state:
            return {"error": f"Unknown agent: {agent_id}"}

        if state.status != "running":
            return {"error": f"Cannot send to {state.status} agent. Only 'running' agents accept messages."}

        # For now, we don't support multi-turn - just note that message was queued
        # Future: implement streaming append to conversation
        return {
            "message": "Multi-turn messaging not yet implemented. "
                       "Use agent.spawn() with updated prompts instead.",
            "status": state.status
        }

    async def result(self, agent_id: str) -> dict:
        """Poll for sub-agent result."""
        state = self._agents.get(agent_id)
        if not state:
            return {"error": f"Unknown agent: {agent_id}"}

        response: dict[str, Any] = {
            "agent_id": agent_id,
            "role": state.role,
            "status": state.status,
            "created_at": state.created_at,
        }

        if state.status in ("done", "cancelled", "timeout", "error"):
            response["finished_at"] = state.finished_at
            response["result"] = state.result or "(no output)"
            response["error"] = state.error or None
            response["turns"] = state.turns
            response["tokens_used"] = state.tokens_used

        return response

    async def cancel(self, agent_id: str) -> dict:
        """Cancel a running sub-agent."""
        state = self._agents.get(agent_id)
        if not state:
            return {"error": f"Unknown agent: {agent_id}"}

        if state.status != "running":
            return {"message": f"Agent was already '{state.status}'. No action needed."}

        # Cancel the task
        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        state.status = "cancelled"
        state.error = "Cancelled by user"
        state.finished_at = datetime.now(timezone.utc).isoformat()

        return {
            "agent_id": agent_id,
            "status": "cancelled",
            "message": f"Sub-agent '{agent_id}' has been cancelled."
        }

    def list_agents(self) -> list[dict]:
        """List all tracked sub-agents."""
        agents = []
        for state in sorted(self._agents.values(), key=lambda a: a.created_at):
            agents.append({
                "agent_id": state.agent_id,
                "role": state.role,
                "status": state.status,
                "created_at": state.created_at,
                "finished_at": state.finished_at,
                "result_preview": (state.result or "")[:100] if state.status == "done" else None,
            })
        return agents


# ---------------------------------------------------------------------------
# Global singleton - will be initialized when CLI/WebUI starts
# ---------------------------------------------------------------------------

_manager: SubAgentManager | None = None


def get_manager() -> SubAgentManager:
    """Get or create the global sub-agent manager."""
    global _manager
    if _manager is None:
        # Lazy init with a placeholder client (will be set by CLI/WebUI)
        raise RuntimeError(
            "SubAgentManager not initialized. Call init_manager() first."
        )
    return _manager


def init_manager(client: HarnessClient, registry: Registry) -> SubAgentManager:
    """Initialize the sub-agent manager with a client and registry."""
    global _manager
    _manager = SubAgentManager(client, registry)
    return _manager


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register(registry: Registry = default_registry) -> None:
    """Register agent.* tools on the registry."""

    @registry.tool(
        name="agent.spawn",
        description=(
            "Spawn a sub-agent to run a task in a separate slot. "
            "The main agent continues working while the sub-agent processes its task. "
            "Returns an agent_id for tracking. Max 3 concurrent sub-agents.\n\n"
            "Parameters:\n"
            "- role: Label describing what this agent does (e.g., 'researcher', 'coder', 'reviewer')\n"
            "- prompt: The task/instructions for the sub-agent\n"
            "- tools: Optional list of tool names to restrict access to (None = full access)\n"
            "- timeout_s: Maximum runtime in seconds (default 300)\n"
            "- max_tokens: Max tokens for response (default 1024)\n"
            "- temperature: Randomness 0-1 (default 0.7)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "Label for the agent role (e.g., 'researcher', 'coder', 'reviewer')",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task/instructions to give the sub-agent",
                },
                "tools": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Tool names to allow (None = all tools). Restrict for safety.",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Max runtime in seconds (default 300)",
                    "default": 300,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max tokens for response (default 1024)",
                    "default": 1024,
                },
                "temperature": {
                    "type": "number",
                    "description": "Randomness 0-1 (default 0.7)",
                    "default": 0.7,
                },
            },
            "required": ["role", "prompt"],
        },
    )
    def agent_spawn(
        role: str,
        prompt: str,
        tools: list[str] | None = None,
        timeout_s: int = 300,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict:
        """Spawn a new sub-agent."""
        mgr = get_manager()
        return asyncio.run(mgr.spawn(role, prompt, tools=tools, timeout_s=timeout_s, 
                                     max_tokens=max_tokens, temperature=temperature))

    @registry.tool(
        name="agent.list",
        description=(
            "List all active and recently completed sub-agents. "
            "Shows: agent_id, role, status (initializing/running/done/cancelled/error), created_at."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
    )
    def agent_list() -> list[dict]:
        """List all tracked sub-agents."""
        mgr = get_manager()
        return mgr.list_agents()

    @registry.tool(
        name="agent.result",
        description=(
            "Retrieve the latest result from a sub-agent. "
            "Non-blocking poll - returns current status and partial results if still running. "
            "Call repeatedly until status is 'done' or another terminal state."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent_id returned by agent.spawn()",
                },
            },
            "required": ["agent_id"],
        },
    )
    def agent_result(agent_id: str) -> dict:
        """Poll for sub-agent result."""
        mgr = get_manager()
        return asyncio.run(mgr.result(agent_id))

    @registry.tool(
        name="agent.cancel",
        description=(
            "Terminate a running sub-agent and free its slot. "
            "Use this to stop agents that are taking too long or no longer needed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent_id to cancel",
                },
            },
            "required": ["agent_id"],
        },
    )
    def agent_cancel(agent_id: str) -> dict:
        """Cancel a running sub-agent."""
        mgr = get_manager()
        return asyncio.run(mgr.cancel(agent_id))

    @registry.tool(
        name="agent.send",
        description=(
            "Send a follow-up message to a running sub-agent. "
            "Currently limited - for multi-turn conversations, use agent.spawn with updated prompts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent_id to send to",
                },
                "message": {
                    "type": "string",
                    "description": "The message/instruction to send",
                },
            },
            "required": ["agent_id", "message"],
        },
    )
    def agent_send(agent_id: str, message: str) -> dict:
        """Send a follow-up message."""
        mgr = get_manager()
        return asyncio.run(mgr.send(agent_id, message))
