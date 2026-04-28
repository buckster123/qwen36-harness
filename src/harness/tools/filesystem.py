"""Filesystem tools — sandboxed under a configurable root.

All paths are resolved with strict containment checks so the model can't
escape the sandbox via ``..``, symlinks pointing outside, etc. The default
root is ``~/qwen36-sandbox`` (auto-created), but tests and callers can pass
any directory.

Tools intentionally cap result sizes — if the model needs more, it can
re-call with explicit ``offset`` / ``limit``. We never silently truncate
without telling the model.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import Registry, ToolError, default_registry

DEFAULT_SANDBOX = Path.home() / "qwen36-sandbox"
MAX_READ_BYTES = 64 * 1024  # 64 KB hard cap per call
MAX_LIST_ENTRIES = 200


class FsSandbox:
    """Path resolver bound to a single root directory."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or DEFAULT_SANDBOX).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel_path: str) -> Path:
        """Resolve ``rel_path`` strictly under ``self.root``. Raises ToolError on escape."""
        if not rel_path or rel_path == ".":
            return self.root
        # Reject absolute paths outright — the sandbox is rooted, every input is relative
        p = Path(rel_path)
        if p.is_absolute():
            raise ToolError(
                f"absolute paths not allowed in sandbox; got '{rel_path}'. "
                f"All paths are relative to the sandbox root."
            )
        candidate = (self.root / p).resolve()
        # Strict containment via os.path.commonpath
        try:
            common = os.path.commonpath([str(self.root), str(candidate)])
        except ValueError:
            common = ""
        if common != str(self.root):
            raise ToolError(f"path '{rel_path}' escapes sandbox root '{self.root}'")
        return candidate


# --- tool registration --------------------------------------------------------


def register(registry: Registry = default_registry, sandbox: FsSandbox | None = None) -> FsSandbox:
    """Register fs tools on ``registry`` bound to ``sandbox``.

    Returns the sandbox so callers can reuse it (e.g. for tests).
    """
    sb = sandbox or FsSandbox()

    @registry.tool(
        name="fs.read",
        description=(
            "Read a UTF-8 text file from the sandbox. Returns first ``limit`` bytes. "
            "Use this to inspect file contents the model is reasoning about."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to sandbox root."},
                "limit": {
                    "type": "integer",
                    "description": f"Max bytes to read (default 8192, hard cap {MAX_READ_BYTES}).",
                    "default": 8192,
                },
            },
            "required": ["path"],
        },
    )
    def fs_read(path: str, limit: int = 8192) -> str:
        p = sb.resolve(path)
        if not p.exists():
            raise ToolError(f"no such file: {path}")
        if p.is_dir():
            raise ToolError(f"'{path}' is a directory, not a file (use fs.list)")
        cap = min(int(limit), MAX_READ_BYTES)
        try:
            data = p.read_bytes()[:cap]
        except OSError as e:
            raise ToolError(f"read failed: {e}") from None
        truncated = p.stat().st_size > cap
        text = data.decode("utf-8", errors="replace")
        if truncated:
            text += f"\n\n[... truncated; full size {p.stat().st_size} bytes ...]"
        return text

    @registry.tool(
        name="fs.list",
        description=(
            "List entries in a sandbox directory. Returns up to "
            f"{MAX_LIST_ENTRIES} entries with kind (file/dir) and size."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to sandbox root (default '.').",
                    "default": ".",
                }
            },
        },
    )
    def fs_list(path: str = ".") -> dict:
        p = sb.resolve(path)
        if not p.exists():
            raise ToolError(f"no such directory: {path}")
        if not p.is_dir():
            raise ToolError(f"'{path}' is not a directory")
        try:
            entries = sorted(p.iterdir())[:MAX_LIST_ENTRIES]
        except OSError as e:
            raise ToolError(f"list failed: {e}") from None
        return {
            "root": str(sb.root),
            "path": path,
            "entries": [
                {
                    "name": e.name,
                    "kind": "dir" if e.is_dir() else "file",
                    "size": e.stat().st_size if e.is_file() else None,
                }
                for e in entries
            ],
            "count": len(entries),
        }

    @registry.tool(
        name="fs.write",
        description=(
            "Write a UTF-8 text file in the sandbox (overwrites if exists). "
            "Creates parent directories as needed. "
            "Use this when asked to save artifacts (notes, code, drafts)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to sandbox root."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
        },
        requires_confirmation=True,
    )
    def fs_write(path: str, content: str) -> str:
        p = sb.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            raise ToolError(f"write failed: {e}") from None
        return f"wrote {len(content.encode('utf-8'))} bytes to {path}"

    return sb


__all__ = ["FsSandbox", "register"]
