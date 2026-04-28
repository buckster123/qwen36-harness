"""Live smoke test for MCP integration.

Spawns the real ``@modelcontextprotocol/server-filesystem`` via npx against
a temp directory, registers its tools into a fresh Registry, then dispatches
a real list_directory and verifies the result.

Opt-in: set HARNESS_MCP_SMOKE=1 to enable. Requires npx on PATH and network
the first time (npx fetches the package).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.mcp import MCPManager, MCPServerConfig
from harness.tools import Registry

pytestmark = pytest.mark.skipif(
    os.environ.get("HARNESS_MCP_SMOKE") != "1",
    reason="set HARNESS_MCP_SMOKE=1 to run live MCP smoke",
)


@pytest.mark.asyncio
async def test_filesystem_mcp_round_trip(tmp_path: Path) -> None:
    # Seed the sandbox so list returns something
    (tmp_path / "hello.txt").write_text("hi from smoke test\n", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    reg = Registry()
    mgr = MCPManager(reg)
    cfg = MCPServerConfig(
        name="fs-mcp",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
        connect_timeout=120.0,  # cold npx fetch
    )
    try:
        await mgr.start(cfg)
        names = mgr.tools_for("fs-mcp")
        assert names, "fs-mcp registered no tools"
        # The filesystem MCP exposes tools like read_file, list_directory, write_file.
        # Tool names come prefixed.
        list_tool = next(
            (n for n in names if n.endswith(".list_directory") or n.endswith(".list_dir")),
            None,
        )
        assert list_tool, f"no list tool found in {names}"
        result = await reg.dispatch(list_tool, {"path": str(tmp_path)})
        assert not result.is_error, f"list failed: {result.output}"
        assert "hello.txt" in result.output
        assert "subdir" in result.output
    finally:
        await mgr.stop_all()
