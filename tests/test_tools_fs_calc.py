"""Tests for fs and calc tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools import Registry, ToolError
from harness.tools.calc import register as register_calc
from harness.tools.filesystem import FsSandbox, register as register_fs


def test_sandbox_resolves_relative(tmp_path: Path) -> None:
    sb = FsSandbox(tmp_path)
    p = sb.resolve("notes/foo.txt")
    assert p == (tmp_path / "notes" / "foo.txt").resolve()


def test_sandbox_rejects_dotdot_escape(tmp_path: Path) -> None:
    sb = FsSandbox(tmp_path)
    with pytest.raises(ToolError):
        sb.resolve("../outside.txt")


def test_sandbox_rejects_absolute_path(tmp_path: Path) -> None:
    sb = FsSandbox(tmp_path)
    with pytest.raises(ToolError):
        sb.resolve("/etc/passwd")


def test_sandbox_rejects_symlink_escape(tmp_path: Path) -> None:
    sb = FsSandbox(tmp_path)
    outside = tmp_path.parent / "real_file"
    outside.write_text("secret")
    link = tmp_path / "shortcut"
    link.symlink_to(outside)
    with pytest.raises(ToolError):
        sb.resolve("shortcut")


@pytest.mark.asyncio
async def test_fs_read_write_list_round_trip(tmp_path: Path) -> None:
    r = Registry()
    register_fs(r, FsSandbox(tmp_path))
    # write
    write_res = await r.dispatch("fs.write", {"path": "a/b.txt", "content": "hello fren"})
    assert not write_res.is_error
    # list
    list_res = await r.dispatch("fs.list", {"path": "a"})
    assert "b.txt" in list_res.output
    # read
    read_res = await r.dispatch("fs.read", {"path": "a/b.txt"})
    assert read_res.output == "hello fren"


@pytest.mark.asyncio
async def test_fs_read_truncates_large_files(tmp_path: Path) -> None:
    r = Registry()
    register_fs(r, FsSandbox(tmp_path))
    big = tmp_path / "big.txt"
    big.write_text("x" * 200_000)
    res = await r.dispatch("fs.read", {"path": "big.txt", "limit": 1024})
    assert "[... truncated" in res.output
    assert res.output.count("x") == 1024


@pytest.mark.asyncio
async def test_fs_write_blocked_outside_sandbox(tmp_path: Path) -> None:
    r = Registry()
    register_fs(r, FsSandbox(tmp_path))
    res = await r.dispatch("fs.write", {"path": "../escape.txt", "content": "no"})
    assert res.is_error
    assert "escapes sandbox" in res.output


@pytest.mark.asyncio
async def test_calc_exact_form() -> None:
    r = Registry()
    register_calc(r)
    res = await r.dispatch("calc.eval", {"expression": "sqrt(8)"})
    assert not res.is_error
    assert "2*sqrt(2)" in res.output


@pytest.mark.asyncio
async def test_calc_float_form() -> None:
    r = Registry()
    register_calc(r)
    res = await r.dispatch("calc.eval", {"expression": "pi", "as_float": True})
    assert not res.is_error
    assert "3.14159" in res.output


@pytest.mark.asyncio
async def test_calc_arithmetic() -> None:
    r = Registry()
    register_calc(r)
    res = await r.dispatch("calc.eval", {"expression": "17 * 91"})
    assert "1547" in res.output


@pytest.mark.asyncio
async def test_calc_invalid_expression_returns_error() -> None:
    r = Registry()
    register_calc(r)
    res = await r.dispatch("calc.eval", {"expression": "this is not math"})
    assert res.is_error
