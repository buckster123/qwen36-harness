"""Tests for the tool registry primitives."""

from __future__ import annotations

import pytest

from harness.tools import Registry, ToolError, ToolSpec


def test_register_and_dispatch_sync() -> None:
    r = Registry()

    @r.tool("upper", "uppercase a string", {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    })
    def upper(text: str) -> str:
        return text.upper()

    assert r.names() == ["upper"]
    schemas = r.export_schema()
    assert schemas[0]["function"]["name"] == "upper"


@pytest.mark.asyncio
async def test_dispatch_with_dict_args() -> None:
    r = Registry()
    r.register(
        ToolSpec(
            name="add",
            description="add two ints",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            fn=lambda a, b: a + b,
        )
    )
    result = await r.dispatch("add", {"a": 2, "b": 3})
    assert result.output == "5"
    assert not result.is_error


@pytest.mark.asyncio
async def test_dispatch_parses_string_args() -> None:
    r = Registry()
    r.register(ToolSpec(name="echo", description="echo", parameters={}, fn=lambda x: x))
    result = await r.dispatch("echo", '{"x": "hello"}')
    assert result.output == "hello"


@pytest.mark.asyncio
async def test_dispatch_async_tool() -> None:
    r = Registry()

    async def doubler(n: int) -> int:
        return n * 2

    r.register(ToolSpec(name="double", description="double", parameters={}, fn=doubler))
    result = await r.dispatch("double", {"n": 7})
    assert result.output == "14"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error() -> None:
    r = Registry()
    result = await r.dispatch("missing", {})
    assert result.is_error
    assert "unknown tool" in result.output.lower()


@pytest.mark.asyncio
async def test_dispatch_tool_error() -> None:
    r = Registry()

    def boom(x: int) -> int:
        raise ToolError("explicit failure")

    r.register(ToolSpec(name="boom", description="x", parameters={}, fn=boom))
    result = await r.dispatch("boom", {"x": 1})
    assert result.is_error
    assert result.output == "explicit failure"


@pytest.mark.asyncio
async def test_dispatch_unexpected_exception_surfaces_to_model() -> None:
    r = Registry()

    def crash(x: int) -> int:
        raise RuntimeError("kaboom")

    r.register(ToolSpec(name="crash", description="x", parameters={}, fn=crash))
    result = await r.dispatch("crash", {"x": 1})
    assert result.is_error
    assert "RuntimeError" in result.output


@pytest.mark.asyncio
async def test_disabled_tool() -> None:
    r = Registry()
    r.register(ToolSpec(name="t", description="x", parameters={}, fn=lambda: "ok"))
    r.set_enabled("t", False)
    schemas = r.export_schema()
    assert schemas == []  # disabled tools are excluded
    result = await r.dispatch("t", {})
    assert result.is_error
    assert "disabled" in result.output


@pytest.mark.asyncio
async def test_dict_result_serialized_to_json() -> None:
    r = Registry()
    r.register(ToolSpec(name="d", description="x", parameters={}, fn=lambda: {"a": 1, "b": [1, 2]}))
    result = await r.dispatch("d", {})
    assert '"a": 1' in result.output
    assert '"b"' in result.output
