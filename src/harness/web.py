"""FastAPI web UI server for the harness.

A 3-pane single-page app served at ``/`` with a JSON+SSE control API. The
server holds a single shared session: one HarnessClient, one Conversation,
one MCPManager. Concurrent /api/chat requests are serialised so we never
issue overlapping LLM streams (the model would just queue them anyway).

Endpoints
---------
GET  /                        static index.html
GET  /static/<path>           static files
GET  /api/state               settings, endpoints, tools, mcp servers, convo
POST /api/chat                kick off a turn → SSE stream of agent events
POST /api/clear               drop convo turns (system prompt kept)
POST /api/system              {text} update or clear system prompt
POST /api/use                 {endpoint} switch endpoint
POST /api/settings            {mode, max_tokens, temperature, agent_mode, show_thinking}
POST /api/tools/{name}        {enabled: bool} toggle a registered tool
POST /api/mcp/{name}/start    spawn an MCP server, register tools
POST /api/mcp/{name}/stop     stop and unregister
POST /api/cancel              best-effort cancel of in-flight chat

The SSE stream re-emits the existing ``AgentEvent`` shape verbatim (kind,
text, data) plus a final ``stats`` event with usage. The frontend picks
this apart into chat bubbles and tool-call panels.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .agent import Agent, AgentLimits
from .client import HarnessClient
from .config import Config, Endpoint, load_config
from .mcp import MCPManager, MCPServerConfig, load_mcp_config
from .tools import default_registry
from .tools.calc import register as register_calc
from .tools.cron import register as register_cron
from .tools.cron import scheduler as cron_scheduler
from .tools.filesystem import FsSandbox, register as register_fs
from .tools.subagent import register as register_subagent
from .tools.code_exec import init_sandbox, register as register_code_exec
from .tools.web_search import register as register_web_search
from .tools.vast_manager import VastManager

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class WebSession:
    cfg: Config
    ep: Endpoint
    sandbox: FsSandbox
    mcp_configs: dict[str, MCPServerConfig] = field(default_factory=dict)
    show_thinking: bool = True
    agent_mode: bool = True  # default ON in the UI — that's why we built tools
    max_tokens: int = 4096
    temperature: float = 0.7
    system: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    last_stats: dict[str, Any] = field(default_factory=dict)
    discovered_models: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    vast_manager: VastManager | None = None  # set in create_app factory

    def __post_init__(self) -> None:
        self.client = HarnessClient(self.ep)
        self.mcp = MCPManager(default_registry)
        self.lock = asyncio.Lock()
        self.cancel_event = asyncio.Event()

    def messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.extend(self.turns)
        return out

    async def aclose(self) -> None:
        await self.mcp.stop_all()
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    text: str = Field(min_length=1)


class SystemRequest(BaseModel):
    text: str | None = None


class UseRequest(BaseModel):
    endpoint: str


class SettingsRequest(BaseModel):
    mode: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    agent_mode: bool | None = None
    show_thinking: bool | None = None


class ToolToggleRequest(BaseModel):
    enabled: bool


# ---------------------------------------------------------------------------
# Model discovery — ping /v1/models on each configured endpoint
# ---------------------------------------------------------------------------


async def _discover_endpoint_models(client: httpx.AsyncClient, endpoint) -> list[dict[str, Any]]:
    """Return a list of {id, object, owned_by} from the endpoint's /models endpoint."""
    try:
        url = endpoint.models_url()  # uses Endpoint.models_url() — already includes /v1
        resp = await client.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", []) or []
    except Exception:  # noqa: BLE001
        pass
    return []


async def discover_models(session: WebSession) -> dict[str, list[dict[str, Any]]]:
    """Ping all configured endpoints' /v1/models and cache the results."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        tasks = [
            _discover_endpoint_models(client, ep)
            for ep in session.cfg.endpoints.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    discovered: dict[str, list[dict[str, Any]]] = {}
    for (ep_name, ep), result in zip(session.cfg.endpoints.items(), results):
        if isinstance(result, Exception):
            log.warning("discover '%s': %s", ep_name, result)
            continue
        discovered[ep_name] = result

    session.discovered_models = discovered
    return discovered


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _state_payload(session: WebSession) -> dict[str, Any]:
    tools = []
    for name in default_registry.names():
        spec = default_registry.get(name)
        tools.append(
            {
                "name": name,
                "description": spec.description,
                "enabled": spec.enabled,
                "is_mcp": "." in name,  # heuristic: dot = mcp_namespaced
            }
        )
    mcp_status = []
    running = set(session.mcp.server_names())
    for name, mc in session.mcp_configs.items():
        mcp_status.append(
            {
                "name": name,
                "running": name in running,
                "auto_start": mc.auto_start,
                "command": mc.command,
                "tools": session.mcp.tools_for(name) if name in running else [],
            }
        )
    return {
        "endpoints": [
            {
                "name": e.name,
                "model": e.model,
                "base_url": e.base_url,
                "mode": e.mode,
                "current": e.name == session.ep.name,
            }
            for e in session.cfg.endpoints.values()
        ],
        "current_endpoint": session.ep.name,
        "mode": session.ep.mode,
        "max_tokens": session.max_tokens,
        "temperature": session.temperature,
        "agent_mode": session.agent_mode,
        "show_thinking": session.show_thinking,
        "system": session.system,
        "sandbox_root": str(session.sandbox.root),
        "tools": tools,
        "mcp": mcp_status,
        "cron": cron_scheduler.list_jobs(),
        "turn_count": len(session.turns),
        "last_stats": session.last_stats,
        "discovered_models": {k: [m["id"] for m in v] for k, v in session.discovered_models.items()},
    }


def create_app(
    *,
    session: WebSession | None = None,
    register_builtins: bool = True,
) -> FastAPI:
    """Build the FastAPI app. ``session`` may be injected by tests; production
    callers (`harness serve`) get the default builder."""
    if session is None:
        cfg = load_config()
        ep = cfg.get()
        sandbox = FsSandbox()
        if register_builtins:
            register_fs(default_registry, sandbox=sandbox)
            register_calc(default_registry)
            register_cron(default_registry)
            register_subagent(default_registry)
            register_code_exec(default_registry)
            register_web_search(default_registry)
        session = WebSession(
            cfg=cfg,
            ep=ep,
            sandbox=sandbox,
            mcp_configs=load_mcp_config(),
            max_tokens=ep.default_max_tokens,
            temperature=ep.default_temperature,
            vast_manager=VastManager(),
        )

    app = FastAPI(title="qwen36-harness web", version="0.3.0")
    app.state.session = session

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # --- root ---------------------------------------------------------------

    @app.get("/")
    async def root() -> Any:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            return JSONResponse({"error": "index.html missing"}, status_code=500)
        return FileResponse(str(index))

    # --- state --------------------------------------------------------------

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return _state_payload(session)

    # --- conversation control ----------------------------------------------

    @app.post("/api/clear")
    async def clear_convo() -> dict[str, Any]:
        session.turns.clear()
        session.last_stats = {}
        return {"ok": True}

    @app.post("/api/system")
    async def set_system(req: SystemRequest) -> dict[str, Any]:
        session.system = req.text or None
        return {"ok": True, "system": session.system}

    @app.post("/api/use")
    async def use_endpoint(req: UseRequest) -> dict[str, Any]:
        try:
            new_ep = session.cfg.get(req.endpoint)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        await session.client.aclose()
        session.ep = new_ep
        session.client = HarnessClient(new_ep)
        session.max_tokens = new_ep.default_max_tokens
        session.temperature = new_ep.default_temperature
        return {"ok": True, "endpoint": new_ep.name}

    @app.post("/api/settings")
    async def update_settings(req: SettingsRequest) -> dict[str, Any]:
        if req.mode is not None:
            if req.mode not in ("thinking", "nonthinking", "coding"):
                raise HTTPException(400, "mode must be thinking|nonthinking|coding")
            # Endpoint is a frozen dataclass — swap to a new instance + new client.
            new_ep = dataclasses.replace(session.ep, mode=req.mode)
            await session.client.aclose()
            session.ep = new_ep
            session.client = HarnessClient(new_ep)
        if req.max_tokens is not None:
            session.max_tokens = max(1, int(req.max_tokens))
        if req.temperature is not None:
            session.temperature = float(req.temperature)
        if req.agent_mode is not None:
            session.agent_mode = bool(req.agent_mode)
        if req.show_thinking is not None:
            session.show_thinking = bool(req.show_thinking)
        return {"ok": True}

    # --- tools --------------------------------------------------------------

    @app.post("/api/tools/{name}/toggle")
    async def toggle_tool(name: str, req: ToolToggleRequest) -> dict[str, Any]:
        try:
            default_registry.set_enabled(name, req.enabled)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return {"ok": True, "name": name, "enabled": req.enabled}

    # --- MCP ----------------------------------------------------------------

    @app.post("/api/mcp/{name}/start")
    async def mcp_start(name: str) -> dict[str, Any]:
        mc = session.mcp_configs.get(name)
        if mc is None:
            raise HTTPException(404, f"no MCP server '{name}' in config")
        try:
            await session.mcp.start(mc)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, str(e)) from e
        return {"ok": True, "tools": session.mcp.tools_for(name)}

    @app.post("/api/mcp/{name}/stop")
    async def mcp_stop(name: str) -> dict[str, Any]:
        await session.mcp.stop(name)
        return {"ok": True}

    # --- cron ---------------------------------------------------------------

    @app.post("/api/cron/{name}/run_now")
    async def cron_run(name: str) -> dict[str, Any]:
        try:
            result = cron_scheduler.run_once(name)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, str(e)) from e
        return {"ok": True, "result": result}

    @app.post("/api/cron/{name}/stop")
    async def cron_stop(name: str) -> dict[str, Any]:
        try:
            cron_scheduler.stop(name)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, str(e)) from e
        return {"ok": True}

    # --- chat (SSE) ---------------------------------------------------------

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> EventSourceResponse:
        return EventSourceResponse(_chat_stream(session, req.text))

    @app.post("/api/cancel")
    async def cancel() -> dict[str, Any]:
        session.cancel_event.set()
        return {"ok": True}

    # --- model discovery ------------------------------------------------------

    @app.post("/api/discover")
    async def api_discover() -> dict[str, Any]:
        models = await discover_models(session)
        # Also refresh MCP configs so the full state is up to date
        session.mcp_configs = load_mcp_config()
        return {"ok": True, "models": models}
    # --- Vast.ai instance management ----------------------------------------

    @app.get("/api/vast/recipes")
    async def vast_recipes() -> dict[str, Any]:
        """List all available GPU/model recipes."""
        vm = session.vast_manager
        if vm is None:
            return {"ok": False, "error": "VastManager not configured"}
        return {"ok": True, "recipes": await asyncio.to_thread(vm.recipes)}

    @app.post("/api/vast/spinup")
    async def vast_spinup(req: dict[str, str] | None = None) -> dict[str, Any]:
        """Spin up a new Vast.ai instance. Body: {recipe, geo, ...overrides}."""
        vm = session.vast_manager
        if vm is None:
            return {"success": False, "error": "VastManager not configured"}
        body = req or {}
        recipe = body.get("recipe")
        if not recipe:
            raise HTTPException(400, "missing 'recipe' field")
        geo = body.get("geo", "EU_NORDIC")
        overrides = {k: v for k, v in body.items() if k not in ("recipe", "geo")}
        result = await asyncio.to_thread(vm.spinup, recipe, geo=geo, **overrides)
        return result

    @app.get("/api/vast/instances")
    async def vast_instances(state: str | None = None) -> dict[str, Any]:
        """List active Vast.ai instances."""
        vm = session.vast_manager
        if vm is None:
            return {"ok": False, "error": "VastManager not configured"}
        instances = await asyncio.to_thread(vm.list_instances, state=state)
        return {"ok": True, "instances": instances}

    @app.get("/api/vast/latest")
    async def vast_latest() -> dict[str, Any]:
        """Get the last spun-up instance."""
        vm = session.vast_manager
        if vm is None:
            return {"ok": False, "error": "VastManager not configured"}
        inst = await asyncio.to_thread(vm.latest_instance)
        return {"ok": True, "instance": inst}

    @app.get("/api/vast/status")
    async def vast_status(instance_id: str | None = None) -> dict[str, Any]:
        """Poll status of a specific instance (or .last_instance)."""
        vm = session.vast_manager
        if vm is None:
            return {"ok": False, "error": "VastManager not configured"}
        target = instance_id or ((await asyncio.to_thread(vm.latest_instance)) or {}).get("id")
        if not target:
            return {"ok": True, "status": "no active instance"}
        result = await asyncio.to_thread(vm.poll_status, target)
        return {"ok": True, "status": result}

    @app.post("/api/vast/down")
    async def vast_down(req: dict[str, str] | None = None) -> dict[str, Any]:
        """Destroy a Vast.ai instance. Body: {instance_id} (optional)."""
        vm = session.vast_manager
        if vm is None:
            return {"success": False, "error": "VastManager not configured"}
        body = req or {}
        inst_id = body.get("instance_id")
        result = await asyncio.to_thread(vm.down, instance_id=inst_id)
        return result

    @app.post("/api/vast/tunnel/{action}")
    async def vast_tunnel(action: str) -> dict[str, Any]:
        """Control SSH tunnel: up | status | down | logs."""
        if action not in ("up", "status", "down", "logs"):
            raise HTTPException(400, f"invalid action: {action}")
        vm = session.vast_manager
        if vm is None:
            return {"success": False, "error": "VastManager not configured"}
        result = await asyncio.to_thread(vm.tunnel, action)
        return result


    # --- lifecycle ----------------------------------------------------------

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Auto-start any flagged MCP servers
        for nm, mc in session.mcp_configs.items():
            if mc.auto_start:
                try:
                    await session.mcp.start(mc)
                    log.info(
                        "mcp '%s' auto-started: %s tools",
                        nm,
                        len(session.mcp.tools_for(nm)),
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("mcp '%s' auto_start failed: %s", nm, e)

        # Discover available models from all configured endpoints
        discovered = await discover_models(session)
        log.info("discovered models: %s", {k: len(v) for k, v in discovered.items()})

        try:
            yield
        finally:
            await session.aclose()

    app.router.lifespan_context = lifespan

    return app


# ---------------------------------------------------------------------------
# Chat SSE generator
# ---------------------------------------------------------------------------


async def _chat_stream(session: WebSession, user_text: str):
    """Acquire the session lock, stream events, append the assistant turn at end."""
    if session.lock.locked():
        yield {"event": "error", "data": json.dumps({"error": "another turn is in flight"})}
        return

    async with session.lock:
        session.cancel_event.clear()
        session.turns.append({"role": "user", "content": user_text})
        # Also tell the UI that we accepted the user message (helps the front-
        # end render the bubble immediately if it didn't already).
        yield {
            "event": "user",
            "data": json.dumps({"text": user_text, "turn_index": len(session.turns)}),
        }

        if session.agent_mode:
            agent = Agent(
                session.client,
                registry=default_registry,
                limits=AgentLimits(max_turns=8),
            )
            messages = session.messages()
            full_content = ""
            full_reasoning = ""
            last_stats = {}
            try:
                async for ev in agent.run(
                    messages,
                    max_tokens=session.max_tokens,
                    temperature=session.temperature,
                ):
                    if session.cancel_event.is_set():
                        yield {"event": "cancelled", "data": "{}"}
                        break
                    if ev.kind == "content":
                        full_content += ev.text
                    elif ev.kind == "reasoning":
                        full_reasoning += ev.text
                    elif ev.kind == "stats":
                        last_stats = ev.data
                    yield {
                        "event": ev.kind,
                        "data": json.dumps(
                            {"text": ev.text, "data": ev.data}, default=str
                        ),
                    }
                # The agent mutated ``messages`` in-place to include any tool
                # turns + the final assistant turn. Replace session.turns with
                # everything except the leading system prompt.
                if session.system and messages and messages[0].get("role") == "system":
                    session.turns = messages[1:]
                else:
                    session.turns = list(messages)
                # Save stats for /api/state display
                session.last_stats = last_stats
            except Exception as e:  # noqa: BLE001
                yield {
                    "event": "error",
                    "data": json.dumps({"error": f"{type(e).__name__}: {e}"}),
                }
        else:
            # Non-agent: a single streamed completion
            messages = session.messages()
            full_content = ""
            full_reasoning = ""
            last_stats = {}
            try:
                stream = await session.client.stream(
                    messages,
                    tools=None,
                    max_tokens=session.max_tokens,
                    temperature=session.temperature,
                )
                async for ev in stream:
                    if session.cancel_event.is_set():
                        yield {"event": "cancelled", "data": "{}"}
                        break
                    if ev.kind == "content":
                        full_content += ev.text
                    elif ev.kind == "reasoning":
                        full_reasoning += ev.text
                    elif ev.kind == "usage":
                        last_stats = ev.data
                    yield {
                        "event": ev.kind,
                        "data": json.dumps(
                            {"text": ev.text, "data": ev.data}, default=str
                        ),
                    }
                msg = {"role": "assistant", "content": full_content}
                if full_reasoning:
                    msg["reasoning_content"] = full_reasoning
                session.turns.append(msg)
                # Save stats for /api/state display
                session.last_stats = last_stats
            except Exception as e:  # noqa: BLE001
                yield {
                    "event": "error",
                    "data": json.dumps({"error": f"{type(e).__name__}: {e}"}),
                }

        yield {"event": "end", "data": "{}"}


__all__ = ["create_app", "WebSession"]
