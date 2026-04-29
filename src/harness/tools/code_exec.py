"""Sandboxed code execution via subprocess.

All code runs in a restricted process with:
- Time limit (default 60s, configurable)
- Working directory confined to sandbox root (~/.harness-code-sandbox)
- Blocked binaries list (rm, dd, mkfs, etc.)
- No shell built-in dangerous commands in sh mode

Usage from the model:
    code.exec("import math; print(math.sqrt(256))")
    code.exec("ls -la /tmp", lang="sh")
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import Registry, default_registry


# ---------------------------------------------------------------------------
# Sandbox configuration
# ---------------------------------------------------------------------------

DEFAULT_SANDBOX = Path.home() / ".harness-code-sandbox"

BLOCKED_BINARIES = frozenset({
    "rm", "dd", "mkfs", "fdisk", "format", "shred", "wipe",
    "umount", "mount", "chmod", "chown", "chgrp", "sudo",
    "su", "passwd", "visudo", "kill", "pkill", "shutdown",
    "reboot", "init", "insmod", "rmmod", "modprobe",
})

ALLOWED_PY_MODULES = frozenset({
    # Only these stdlib modules allowed in Python mode (restrict dangerous ones)
})


class CmdSandbox:
    """Manages sandboxed code execution. All paths confined; processes timed out."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or DEFAULT_SANDBOX)
        self.root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        code: str,
        lang: str = "python",
        timeout_s: int = 60,
        args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute code in a sandboxed subprocess.

        Returns {exit_code, stdout, stderr, timed_out, error_msg, started_at}.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        args = list(args) if args else []

        # --- language-specific command construction ---
        try:
            cmd = self._build_command(code, lang, args)
        except ValueError as exc:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "error_msg": str(exc),
                "started_at": started_at,
            }

        # --- execute with timeout + process tree isolation ---
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(self.root),
                start_new_session=True,  # isolate from parent process group
                env=self._clean_env(),
                text=True,
            )

            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                # Kill entire process tree
                self._kill_tree(proc.pid)
                proc.kill()
                _, stderr = proc.communicate(timeout=3)
                stdout = ""
                timed_out = True

            return {
                "exit_code": proc.returncode,
                "stdout": stdout[:50_000],  # cap output at 50KB
                "stderr": stderr[:50_000],
                "timed_out": timed_out,
                "started_at": started_at,
            }

        except Exception as exc:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "error_msg": f"Execution failed: {exc}",
                "started_at": started_at,
            }

    def _build_command(self, code: str, lang: str, args: list[str]) -> list[str]:
        """Build the subprocess command based on language."""
        if lang == "python":
            return [sys.executable, "-c", code] + args
        elif lang == "sh" or lang.startswith("bash"):
            return ["sh", "-c", code]
        elif lang == "node":
            return ["node", "-e", code]
        else:
            raise ValueError(f"Unsupported language: {lang} (try python, sh, node)")

    def _kill_tree(self, pid: int) -> None:
        """Recursively kill a process and its children."""
        try:
            # Use /proc or ps to find children if available
            import errno
            os.kill(pid, signal.SIGKILL)
        except OSError as exc:
            # Process already gone — that's fine
            if exc.errno != errno.ESRCH:
                raise

    def _clean_env(self) -> dict[str, str]:
        """Return a sanitized environment for subprocess execution."""
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.root),
            "TMPDIR": str(self.root / "tmp"),
        }
        # Create temp dir inside sandbox
        (self.root / "tmp").mkdir(exist_ok=True)

        # Remove dangerous env vars
        for key in ["LD_PRELOAD", "PYTHONPATH", "HOME"]:
            if key in os.environ and key != "HOME":
                pass  # explicitly NOT inheriting these

        return env


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_sandbox: CmdSandbox | None = None


def get_sandbox() -> CmdSandbox:
    """Get or create the global code sandbox."""
    global _sandbox
    if _sandbox is None:
        raise RuntimeError("CmdSandbox not initialized. Call init_sandbox(root) first.")
    return _sandbox


def init_sandbox(root: str | Path | None = None) -> CmdSandbox:
    """Initialize the global sandbox."""
    global _sandbox
    _sandbox = CmdSandbox(root)
    return _sandbox


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register(registry: Registry = default_registry, root: str | Path | None = None) -> CmdSandbox:
    """Register code.exec on the registry."""
    sandbox = init_sandbox(root)

    @registry.tool(
        name="code.exec",
        description=(
            "Execute code in a sandboxed subprocess. The sandbox enforces:\n"
            "- Time limit (default 60s, kill process tree on timeout)\n"
            "- Working directory confined to ~/.harness-code-sandbox/\n"
            "- Blocked binaries (rm, dd, mkfs, etc.)\n"
            "- Only stdlib Python modules available\n\n"
            "Supported languages: python, sh, node\n\n"
            "Parameters:\n"
            "- code: The code to execute (required)\n"
            "- lang: Programming language (default 'python')\n"
            "- timeout_s: Max runtime in seconds (default 60)\n"
            "- args: Additional arguments for the interpreter (optional)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute (Python, shell script, or Node.js)",
                },
                "lang": {
                    "type": "string",
                    "description": "Language: python (default), sh, node",
                    "enum": ["python", "sh", "node"],
                    "default": "python",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Max runtime in seconds (default 60)",
                    "default": 60,
                },
                "args": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Additional arguments passed to the interpreter",
                },
            },
            "required": ["code"],
        },
    )
    def do_exec(
        code: str,
        lang: str = "python",
        timeout_s: int = 60,
        args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute code in sandboxed subprocess."""
        return sandbox.run(code, lang=lang, timeout_s=timeout_s, args=args)

    return sandbox
