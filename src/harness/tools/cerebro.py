"""CerebroCortex tool — wires recall/remember/list_episodes into the harness.

This is a Phase-4 stub: when CC's Python API is importable AND a working
store is configured, we expose a few key functions to the model. When CC
is missing (e.g. the harness runs on a machine without it), tools register
as disabled with a clear error, so the model knows not to call them.

Doing it this way means the harness keeps working everywhere; CC just
becomes "available" wherever it's installed.
"""

from __future__ import annotations

from typing import Any

from . import Registry, ToolError, default_registry

_CC_IMPORT_ERROR: str | None = None
try:
    # Defer the import — CC has heavy deps (chromadb, sentence-transformers).
    # Importing at module-load would slow down ``harness chat`` startup.
    import importlib

    _ = importlib.util.find_spec("cerebro")
    if _ is None:
        _CC_IMPORT_ERROR = "package 'cerebro' not installed"
except Exception as e:  # noqa: BLE001
    _CC_IMPORT_ERROR = str(e)


def _stub_unavailable(*_args: Any, **_kw: Any) -> str:
    raise ToolError(
        f"cerebro is not available on this machine: {_CC_IMPORT_ERROR}. "
        "Install CerebroCortex (`pip install -e ~/projects/CerebroCortex`) "
        "and restart the harness to enable memory tools."
    )


def register(registry: Registry = default_registry) -> None:
    """Register CC tools. Tools register but are immediately disabled if CC is missing."""

    @registry.tool(
        name="cerebro.recall",
        description=(
            "Search Andre's long-term memory (CerebroCortex) by meaning. "
            "Returns the top-k most relevant memories with content and salience. "
            "Use whenever the user references past sessions or asks 'do you remember'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "top_k": {"type": "integer", "default": 5, "description": "Max results (1-20)."},
            },
            "required": ["query"],
        },
    )
    def cerebro_recall(query: str, top_k: int = 5) -> Any:
        if _CC_IMPORT_ERROR:
            return _stub_unavailable(query=query, top_k=top_k)
        # Lazy import so we don't pay the cost on harness startup
        from cerebro.cortex import Cortex  # type: ignore  # noqa: PLC0415

        cortex = Cortex()  # picks up settings from ~/.cerebro-cortex/settings.json
        results = cortex.recall(query, top_k=top_k)
        return [
            {
                "id": getattr(r, "id", ""),
                "content": getattr(r, "content", "")[:500],
                "salience": getattr(r, "salience", 0.0),
                "tags": getattr(r, "tags", []),
            }
            for r in results
        ]

    @registry.tool(
        name="cerebro.remember",
        description=(
            "Save a memory to CerebroCortex. Use for facts/insights worth keeping "
            "beyond this session. Auto-tagged with source='harness'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content to remember."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags.",
                },
                "salience": {
                    "type": "number",
                    "description": "Importance 0-1 (default auto).",
                },
            },
            "required": ["content"],
        },
        requires_confirmation=True,
    )
    def cerebro_remember(
        content: str, tags: list[str] | None = None, salience: float | None = None
    ) -> Any:
        if _CC_IMPORT_ERROR:
            return _stub_unavailable(content=content)
        from cerebro.cortex import Cortex  # type: ignore  # noqa: PLC0415

        cortex = Cortex()
        merged_tags = list(set((tags or []) + ["source:harness"]))
        memory = cortex.remember(
            content=content, tags=merged_tags, salience=salience
        )
        return {"id": getattr(memory, "id", ""), "tags": merged_tags}

    @registry.tool(
        name="cerebro.list_episodes",
        description="List recent episode summaries from CerebroCortex memory.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
            },
        },
    )
    def cerebro_list_episodes(limit: int = 10) -> Any:
        if _CC_IMPORT_ERROR:
            return _stub_unavailable()
        from cerebro.cortex import Cortex  # type: ignore  # noqa: PLC0415

        cortex = Cortex()
        episodes = cortex.list_episodes(limit=limit)
        return [
            {
                "id": getattr(e, "id", ""),
                "title": getattr(e, "title", ""),
                "valence": getattr(e, "valence", "neutral"),
                "step_count": len(getattr(e, "steps", []) or []),
            }
            for e in episodes
        ]

    if _CC_IMPORT_ERROR:
        # Disable now so the model doesn't waste tokens calling them.
        for name in ("cerebro.recall", "cerebro.remember", "cerebro.list_episodes"):
            registry.set_enabled(name, False)


__all__ = ["register"]
