# Phase 5: Sub-Agent Orchestration

## Goal
Enable the main agent to spawn helper agents that run in parallel slots on our multi-slot llama.cpp server.

## Architecture

```
Main Agent (slot 1)          Sub-Agent A (slot 2)    Sub-Agent B (slot 3)
┌──────────────┐             ┌──────────────┐       ┌──────────────┐
│ Conversation │              │ Task: research│       │ Task: coding │
│ + tool access│──spawn()───>│ + tool access│       │ + restricted │
│              │  <result>   │              │       │ tools        │
└──────────────┘             └──────────────┘       └──────────────┘
```

## Design Decisions

1. **Shared server, separate conversations**: Each sub-agent gets its own message buffer sent to the same llama-server endpoint. The server's slot system handles parallelism.

2. **Async dispatch**: Sub-agents run in `asyncio.create_task()` - non-blocking from the main agent's perspective.

3. **Tool access levels**:
   - Default: Full tool access (same as main agent)
   - Restricted: Configurable subset (e.g., only fs.read for code review)

4. **No nested spawning**: Sub-agents cannot spawn their own sub-agents to keep complexity manageable.

5. **Max concurrent limit**: 3 sub-agents max (1 per available slot minus main agent's slot)

## Implementation Plan

### Step 1: Core module `src/harness/tools/subagent.py`
- `SubAgentManager` class to track running agents
- Async task wrapper for LLM sessions
- State management (running/done/cancelled/timeout)

### Step 2: Tool functions
```python
@registry.tool("agent.spawn")
def agent_spawn(role: str, prompt: str, tools: list[str] | None = None, timeout_s: int = 300) -> dict

@registry.tool("agent.send") 
def agent_send(agent_id: str, message: str) -> dict

@registry.tool("agent.result")
def agent_result(agent_id: str) -> dict

@registry.tool("agent.cancel")
def agent_cancel(agent_id: str) -> dict

@registry.tool("agent.list")
def agent_list() -> list[dict]
```

### Step 3: Wire up in `__init__.py`
- Import and register the subagent module
- Add to system prompt

### Step 4: Tests
- Test spawn/cancel lifecycle
- Test concurrent execution
- Test timeout behavior

## Files to Create/Modify

New:
- `src/harness/tools/subagent.py` - Core implementation
- `tests/test_subagent.py` - Unit tests

Modified:
- `src/harness/tools/__init__.py` - Import new module
- `src/harness/cli.py` - Add `/agents` slash command
- `src/harness/tools/system_prompt.md` - Document agent.* tools
- `ROADMAP.md` - Update status to 🔄 In Progress

## Technical Details

### Slot Management
```python
class SubAgentManager:
    def __init__(self, client, registry):
        self.client = client  # HarnessClient
        self.registry = registry
        self._agents: dict[str, SubAgentTask] = {}
        self._next_id = itertools.count(1)
        self.max_concurrent = 3
    
    async def spawn(self, role, prompt, tools=None, timeout_s=300):
        # Create isolated message buffer
        # Start async task with agent loop
        # Return agent_id
```

### Concurrency Model
```python
# Each sub-agent runs as an asyncio.Task
async def _run_agent_loop(self, agent_id: str, messages: list, ...):
    """Run a single-turn or multi-turn agent session."""
    try:
        # Use HarnessClient.stream() with isolated messages
        result = await self.client.complete(messages, ...)
        self._agents[agent_id].result = result.content
        self._agents[agent_id].status = "done"
    except asyncio.TimeoutError:
        self._agents[agent_id].status = "timeout"
```

### Integration Points
- CLI: `harness chat` → `/agent on` enables tool loop → agent.* tools available
- WebUI: Same mechanism, rendered in chat interface
- Tests: Mock client + real registry for fast unit tests

## Open Questions
1. Should sub-agents have persistent memory across spawns? (Probably no - keep it simple)
2. How to handle tool confirmation for sub-agents? (Auto-confirm or skip)
3. Rate limiting? (Server handles this naturally via slot contention)

## Timeline
- Day 1: Core module + spawn/list/cancel
- Day 2: send/result + tests
- Day 3: CLI integration + polish
