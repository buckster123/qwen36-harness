"""Tool registry — decorator, dispatch, OpenAI schema export.

A tool is a Python callable with a JSON-Schema describing its parameters.
Registered tools become available to the model as ``functions`` in OpenAI
chat completion requests; when the model emits a ``tool_call`` we look up
the function by name and dispatch with the JSON-decoded arguments.

Design choices:

- **No magic introspection**: tools declare their schema explicitly. We don't
  try to derive it from type hints. Models are picky about schemas; explicit
  is reliable.
- **Sync OR async**: dispatch awaits async callables, calls sync ones in a
  threadpool. Tools mostly do quick I/O so this is fine.
- **Result is a string**: tools return whatever they want internally; the
  registry always coerces results to a UTF-8 string for the model's
  ``role: tool`` reply. JSON-serialisable dicts/lists get json.dumped.

Loaded tool modules (all register against ``default_registry``):

- **calc** — sympy expression evaluator
- **cron** — background scheduler (daemon threads + subprocess)
- **filesystem** — sandboxed fs.* read/list/write/search
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

ToolFn = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    requires_confirmation: bool = False
    enabled: bool = True

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI tools-schema entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolResult:
    """Result of a tool call dispatch."""

    name: str
    output: str
    is_error: bool = False
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    """Raised by tools to report a soft error the model should see."""


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' already registered")
        self._tools[spec.name] = spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        requires_confirmation: bool = False,
    ) -> Callable[[ToolFn], ToolFn]:
        """Decorator: register the wrapped function as a tool."""

        def deco(fn: ToolFn) -> ToolFn:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters,
                    fn=fn,
                    requires_confirmation=requires_confirmation,
                )
            )
            return fn

        return deco

    # -- introspection ---------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool '{name}'")
        return self._tools[name]

    def export_schema(self, *, only: list[str] | None = None) -> list[dict[str, Any]]:
        """Return the OpenAI tools list, optionally filtered to ``only`` and enabled."""
        out: list[dict[str, Any]] = []
        for name in self.names():
            spec = self._tools[name]
            if only is not None and name not in only:
                continue
            if not spec.enabled:
                continue
            out.append(spec.schema())
        return out

    def set_enabled(self, name: str, enabled: bool) -> None:
        self.get(name).enabled = enabled

    # -- dispatch --------------------------------------------------------------

    async def dispatch(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        """Dispatch a tool call. ``arguments`` may be a dict or a JSON string."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                return ToolResult(
                    name=name,
                    output=f"tool argument JSON parse error: {e}",
                    is_error=True,
                )

        try:
            spec = self.get(name)
        except KeyError as e:
            return ToolResult(name=name, output=str(e), is_error=True)

        if not spec.enabled:
            return ToolResult(
                name=name,
                output=f"tool '{name}' is currently disabled",
                is_error=True,
                arguments=arguments,
            )

        try:
            if inspect.iscoroutinefunction(spec.fn):
                raw = await spec.fn(**arguments)
            else:
                raw = await asyncio.to_thread(spec.fn, **arguments)
        except ToolError as e:
            return ToolResult(name=name, output=str(e), is_error=True, arguments=arguments)
        except TypeError as e:
            # Bad argument shape — surface to the model so it retries
            return ToolResult(
                name=name,
                output=f"tool '{name}' rejected arguments: {e}",
                is_error=True,
                arguments=arguments,
            )
        except Exception as e:  # noqa: BLE001 — surface ALL crashes to the model
            return ToolResult(
                name=name,
                output=f"tool '{name}' raised {type(e).__name__}: {e}",
                is_error=True,
                arguments=arguments,
            )

        output = _coerce_to_string(raw)
        return ToolResult(name=name, output=output, arguments=arguments)


def _coerce_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            return repr(value)
    return str(value)


# A module-level registry that builtin tools attach to. Apps can also
# instantiate their own ``Registry()`` for isolated test scenarios.
default_registry = Registry()

# Import tool modules — they register against default_registry at import time
from . import calc, cron, filesystem, subagent  # noqa: E402

__all__ = [
    "Registry",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "default_registry",
    "calc",
    "cron",
    "filesystem",
    "subagent",
]
