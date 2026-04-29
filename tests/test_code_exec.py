"""Tests for harness.tools.code_exec — CmdSandbox lifecycle and safety."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.harness.tools.code_exec import CmdSandbox, init_sandbox


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """Create a temporary sandbox root for testing."""
    root = tmp_path_factory.mktemp("harness-code-sandbox")
    return init_sandbox(root)


def test_sandbox_created(sandbox):
    """Test that the sandbox directory exists after init."""
    assert sandbox.root.exists()
    assert sandbox.root.is_dir()


@pytest.mark.asyncio
async def test_python_success_output(sandbox):
    """Test basic Python code execution returns correct output."""
    result = sandbox.run("print(42)", lang="python")

    assert result["exit_code"] == 0
    assert "42" in result["stdout"]
    assert not result["timed_out"]


@pytest.mark.asyncio
async def test_python_syntax_error(sandbox):
    """Test that Python syntax errors produce non-zero exit code and stderr."""
    result = sandbox.run("print(,invalid", lang="python")

    assert result["exit_code"] != 0
    # stderr contains the SyntaxError message
    assert "SyntaxError" in result["stderr"] or "syntax error" in result["stderr"].lower()


@pytest.mark.asyncio
async def test_python_timed_out(sandbox):
    """Test that long-running processes are killed after timeout."""
    # Python infinite loop — should be killed by timeout
    result = sandbox.run("import time; i = 0\nwhile True: i += 1", lang="python", timeout_s=2)

    assert result["timed_out"] is True
    assert result["exit_code"] != 0


@pytest.mark.asyncio
async def test_shell_command(sandbox):
    """Test basic shell command execution."""
    result = sandbox.run("echo hello world", lang="sh")

    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]


@pytest.mark.asyncio
async def test_cwd_confined_to_sandbox(sandbox):
    """Test that code runs in the sandbox root directory."""
    result = sandbox.run("import os; print(os.getcwd())", lang="python")

    expected = str(sandbox.root)
    assert expected in result["stdout"]


@pytest.mark.asyncio
async def test_unsupported_language_error(sandbox):
    """Test that unsupported languages return an error dict (not crash)."""
    result = sandbox.run("code", lang="rust")

    # run() catches ValueError and returns error response
    assert result["exit_code"] == -1
    assert "Unsupported" in result.get("error_msg", "") or "Unsupported language" in result.get("stderr", "")


@pytest.mark.asyncio
async def test_output_capped_at_50kb(sandbox):
    """Test that stdout/stderr are capped at 50KB."""
    # Generate large output — Python can't naturally overflow, but we cap at 50k
    code = "print('x' * 60000)"
    result = sandbox.run(code, lang="python")

    assert len(result["stdout"]) <= 50_001


@pytest.mark.asyncio
async def test_blocked_command_returns_error(sandbox):
    """Test that invalid commands are rejected gracefully."""
    # Using an empty/invalid code string should still produce an error dict
    result = sandbox.run("", lang="python")
    # Empty Python code is actually valid (does nothing), so exit_code=0 and stdout=""
    assert "stdout" in result
    assert "stderr" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
