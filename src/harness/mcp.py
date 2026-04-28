"""MCP client integration — register external MCP server tools into our registry.

The harness speaks JSON-RPC 2.0 to MCP servers (stdio subprocesses) via the
official ``mcp`` Python SDK. On startup we:

1. Read configs/mcp_servers.toml
2. Spawn each ``auto_start`` server as a subprocess
3. Call ``tools/list`` on each session and register each returned tool into
   ``default_registry`` with name ``<server>.<tool>``
4. Tool dispatch routes through ``ClientSession.call_tool`` over JSON-RPC

Lifecycle: connections are persistent for the lifetime of the CLI session.
``MCPManager.stop_all()`` is called on exit (or via ``/mcp stop <name>``).

Concurrency note — the SDK's ``stdio_client`` and ``ClientSession`` are async
context managers that must be entered and exited in the same task. To keep
them alive while we register tools and dispatch over the lifetime of the
REPL, we run each server as a long-lived asyncio Task that holds the context
managers open and signals readiness via an ``asyncio.Event``. Stopping a
server sets a stop event; the task tears down cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    # tomllib is stdlib in 3.11+; we target 3.13
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .tools import Registry, ToolError, ToolSpec

log = logging.getLogger(__name__)

# These are imported lazily so unit tests can monkey-patch / skip when SDK
# absent. ``MCPManager.start`` will raise a clean error if the SDK isn't
# available.
_MCP_SDK_AVAILABLE: bool | None = None


def _check_sdk() -> bool:
    global _MCP_SDK_AVAILABLE
    if _MCP_SDK_AVAILABLE is None:
        try:
            import mcp  # noqa: F401
            from mcp import ClientSession, StdioServerParameters  # noqa: F401
            from mcp.client.stdio import stdio_client  # noqa: F401

            _MCP_SDK_AVAILABLE = True
        except ImportError:
            _MCP_SDK_AVAILABLE = False
    return _MCP_SDK_AVAILABLE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MCPServerConfig:
    """Static config for one MCP stdio server."""

    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    auto_start: bool = True
    cwd: str | None = None
    connect_timeout: float = 30.0


def load_mcp_config(path: Path | str | None = None) -> dict[str, MCPServerConfig]:
    """Read configs/mcp_servers.toml. Missing file → empty dict (no error)."""
    if path is None:
        # Walk up from CWD to find the project root with configs/
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / "configs" / "mcp_servers.toml"
            if candidate.exists():
                path = candidate
                break
        else:
            return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    out: dict[str, MCPServerConfig] = {}
    for name, raw in (data.get("servers") or {}).items():
        out[name] = MCPServerConfig(
            name=name,
            command=list(raw["command"]),
            env=dict(raw.get("env") or {}),
            auto_start=bool(raw.get("auto_start", True)),
            cwd=raw.get("cwd"),
            connect_timeout=float(raw.get("connect_timeout", 30.0)),
        )
    return out


# ---------------------------------------------------------------------------
# Per-server runtime
# ---------------------------------------------------------------------------


# Safe baseline env passed to MCP subprocesses (mirrors hermes' native-mcp).
# We refuse to inherit the shell's full env to avoid leaking secrets.
_SAFE_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TERM",
    "SHELL",
    "TMPDIR",
    "LOGNAME",
}


def _build_env(extra: dict[str, str]) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS or k.startswith("XDG_")}
    base.update(extra)
    return base


@dataclass(slots=True)
class _ServerRuntime:
    cfg: MCPServerConfig
    task: asyncio.Task[None]
    ready: asyncio.Event
    stop: asyncio.Event
    session_holder: dict[str, Any]  # {"session": ClientSession | None, "error": str | None}
    tool_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class MCPManager:
    """Owns the lifecycle of one or more MCP stdio subprocesses and bridges
    their tools into a ``Registry``.

    Tool naming: ``<server>.<tool>``. Dots are LLM-friendly and match our
    plan's convention. Models tolerate dots in tool names; OpenAI's spec
    allows ``a-zA-Z0-9_-.``.
    """

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._servers: dict[str, _ServerRuntime] = {}

    # --- public API -----------------------------------------------------------

    def server_names(self) -> list[str]:
        return sorted(self._servers)

    def tools_for(self, server: str) -> list[str]:
        rt = self._servers.get(server)
        return list(rt.tool_names) if rt else []

    async def start(self, cfg: MCPServerConfig) -> None:
        """Spawn ``cfg`` and register its tools. Idempotent: re-start = noop."""
        if cfg.name in self._servers:
            log.debug("mcp server '%s' already running", cfg.name)
            return
        if not _check_sdk():
            raise RuntimeError(
                "mcp SDK not installed — `pip install mcp>=1.0` to use MCP servers"
            )

        ready = asyncio.Event()
        stop = asyncio.Event()
        holder: dict[str, Any] = {"session": None, "error": None}

        task = asyncio.create_task(
            self._run_server(cfg, holder, ready, stop),
            name=f"mcp-{cfg.name}",
        )

        # Wait for the session to be ready (initialize + list_tools done) OR
        # for the task to die in startup.
        try:
            await asyncio.wait_for(ready.wait(), timeout=cfg.connect_timeout)
        except asyncio.TimeoutError:
            stop.set()
            task.cancel()
            raise RuntimeError(
                f"mcp server '{cfg.name}' did not become ready within {cfg.connect_timeout}s"
            ) from None

        if holder.get("error"):
            raise RuntimeError(f"mcp server '{cfg.name}' failed to start: {holder['error']}")

        session = holder["session"]
        if session is None:
            raise RuntimeError(f"mcp server '{cfg.name}' has no session after ready")

        tool_names = await self._register_tools(cfg.name, session)
        rt = _ServerRuntime(
            cfg=cfg, task=task, ready=ready, stop=stop, session_holder=holder, tool_names=tool_names
        )
        self._servers[cfg.name] = rt

    async def stop(self, name: str) -> None:
        """Stop one server, unregister its tools, await task cleanup."""
        rt = self._servers.pop(name, None)
        if rt is None:
            return
        # Unregister tools. Registry has no public unregister; reach into
        # _tools dict (we own it).
        for tn in rt.tool_names:
            self.registry._tools.pop(tn, None)  # noqa: SLF001
        rt.stop.set()
        try:
            await asyncio.wait_for(rt.task, timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            rt.task.cancel()

    async def stop_all(self) -> None:
        for name in list(self._servers):
            try:
                await self.stop(name)
            except Exception as e:  # noqa: BLE001
                log.warning("mcp stop_all: %s failed: %s", name, e)

    # --- internal -------------------------------------------------------------

    async def _run_server(
        self,
        cfg: MCPServerConfig,
        holder: dict[str, Any],
        ready: asyncio.Event,
        stop: asyncio.Event,
    ) -> None:
        """Long-lived task: open stdio_client + ClientSession, hold open until
        ``stop`` is set. Sets ``ready`` when initialize() succeeds."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not cfg.command:
            holder["error"] = "empty command"
            ready.set()
            return

        params = StdioServerParameters(
            command=cfg.command[0],
            args=list(cfg.command[1:]),
            env=_build_env(cfg.env),
            cwd=cfg.cwd,
        )

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    holder["session"] = session
                    ready.set()
                    await stop.wait()
        except Exception as e:  # noqa: BLE001
            holder["error"] = f"{type(e).__name__}: {e}"
            log.warning("mcp server '%s' crashed: %s", cfg.name, holder["error"])
            ready.set()  # unblock waiter
            return
        finally:
            holder["session"] = None

    async def _register_tools(self, server_name: str, session: Any) -> list[str]:
        """Call list_tools and register each into our registry."""
        result = await session.list_tools()
        registered: list[str] = []
        for tool in result.tools:
            full_name = f"{server_name}.{tool.name}"
            schema = tool.inputSchema or {"type": "object", "properties": {}}
            description = tool.description or f"MCP tool {tool.name} from {server_name}"

            # Closure captures session + tool.name. Each call_tool is JSON-RPC.
            async def _dispatch(_session=session, _tool_name=tool.name, **kwargs: Any) -> str:
                try:
                    res = await _session.call_tool(_tool_name, kwargs)
                except Exception as e:  # noqa: BLE001
                    raise ToolError(f"mcp call failed: {type(e).__name__}: {e}") from e
                return _flatten_call_result(res)

            spec = ToolSpec(
                name=full_name,
                description=description,
                parameters=schema,
                fn=_dispatch,
            )
            try:
                self.registry.register(spec)
                registered.append(full_name)
            except ValueError:
                log.warning("mcp: tool name collision, skipping '%s'", full_name)
        return registered


def _flatten_call_result(res: Any) -> str:
    """Coerce an mcp CallToolResult into a string for the model.

    The SDK returns a CallToolResult with ``content`` (list of content blocks
    such as TextContent / ImageContent) plus ``isError`` and an optional
    ``structuredContent``. We surface text blocks first, fall back to a JSON
    dump of structuredContent, and prefix [error] if isError is True.
    """
    parts: list[str] = []
    for block in getattr(res, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
            continue
        # fallback: stringify
        parts.append(repr(block))
    body = "\n".join(parts).strip()
    if not body:
        sc = getattr(res, "structuredContent", None)
        if sc:
            import json as _json

            body = _json.dumps(sc, ensure_ascii=False, indent=2, default=str)
    if getattr(res, "isError", False):
        return f"[mcp error] {body}" if body else "[mcp error]"
    return body


__all__ = [
    "MCPManager",
    "MCPServerConfig",
    "load_mcp_config",
]
