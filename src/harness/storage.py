"""Conversation persistence layer — SQLite-backed session store.

All operations are synchronous (run in asyncio.to_thread from the web routes).
Database file: ~/.harness/harness.db

Schema
------
sessions          — one row per conversation session (metadata)
messages          — individual turns (user/assistant/system/tool) with FTS5 search
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_DIR = Path(os.path.expanduser("~/.harness"))
DB_FILE = DB_DIR / "harness.db"


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local-compatible connection. Caller must close it."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables/triggers if they don't exist."""
    with _get_conn() as conn:
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created TIMESTAMP DEFAULT (datetime('now')), updated TIMESTAMP DEFAULT (datetime('now')), system_prompt TEXT, endpoint TEXT, turn_count INTEGER DEFAULT 0)")
            conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE, role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')), content TEXT, turn_index INTEGER NOT NULL, UNIQUE(session_id, turn_index))")
        except Exception:  # noqa: BLE001
            pass

        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(role, content, content=sessions, content_rowid=id)")
        except Exception:  # noqa: BLE001
            pass

        for trigger_sql in (
            "CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN INSERT INTO messages_fts(rowid, role, content) VALUES (NEW.id, NEW.role, COALESCE(NEW.content,'')); END;",
            "CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN INSERT INTO messages_fts(messages_fts, rowid, role, content) VALUES ('delete', OLD.id, OLD.role, COALESCE(OLD.content,'')); END;",
        ):
            try:
                conn.execute(trigger_sql)
            except Exception:  # noqa: BLE001
                pass
    log.info("database initialized at %s", DB_FILE)


def _make_session_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:5]
    return f"{ts}_{short}"


# --- session CRUD ----------------------------------------------------------

def create_session(
    turns: list[dict[str, Any]],
    system_prompt: str | None,
    endpoint: str,
) -> dict[str, Any]:
    """Save the current conversation as a new session. Returns session metadata."""
    if not turns:
        return {"created": False}

    sid = _make_session_id()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _get_conn() as conn:
        c = conn.execute(
            "INSERT INTO sessions(id, created, updated, system_prompt, endpoint, turn_count) VALUES (?,?,?,?,?,?)",
            (sid, created, created, system_prompt, endpoint, len(turns)),
        )
        for idx, msg in enumerate(turns):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            conn.execute(
                "INSERT INTO messages(session_id, role, content, turn_index) VALUES (?,?,?,?)",
                (sid, role, content, idx),
            )

    log.info("created session %s with %d turns", sid, len(turns))
    return {"id": sid, "created": created, "turn_count": len(turns)}


def list_sessions(limit: int = 50, search: str | None = None) -> list[dict[str, Any]]:
    """Return recent session metadata. If *search* is provided, filter by message content."""
    with _get_conn() as conn:
        if search:
            # Find sessions whose messages contain the search term (using LIKE for now;
            # FTS5 triggers work but schema integration is complex)
            rows = conn.execute(
                "SELECT DISTINCT s.id, s.created, s.updated, s.system_prompt, s.endpoint, s.turn_count "
                "FROM sessions s JOIN messages m ON m.session_id = s.id "
                "WHERE m.content LIKE ? ESCAPE '\\' "
                "ORDER BY s.updated DESC LIMIT ?",
                (f'%{search}%', limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, created, updated, system_prompt, endpoint, turn_count FROM sessions ORDER BY updated DESC LIMIT ?",
                (limit,),
            ).fetchall()

    sessions = []
    for sid, created, updated, prompt, ep, tc in rows:
        preview = ""
        if prompt and len(prompt) > 80:
            preview = prompt[:77] + "..."
        else:
            preview = prompt or "(no system prompt)"
        sessions.append({
            "id": sid,
            "created": created,
            "updated": updated,
            "turn_count": tc,
            "preview": preview,
            "endpoint": ep or "",
        })
    return sessions


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load a session's messages and metadata. Returns dict with 'messages', 'system_prompt', 'endpoint'."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None
        sid, created, updated, prompt, ep, tc = row

        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY turn_index ASC",
            (sid,),
        ).fetchall()

    return {
        "id": sid,
        "messages": [{"role": r[0], "content": r[1]} for r in rows],
        "system_prompt": prompt,
        "endpoint": ep or "",
        "turn_count": tc,
    }


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages. Returns True if deleted."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
    return cur.rowcount > 0


def delete_turn(session_id: str, turn_index: int) -> bool:
    """Delete a specific turn by its index within a session. Returns True if deleted."""
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM messages WHERE session_id=? AND turn_index=?",
            (session_id, turn_index),
        )
        conn.commit()
        affected = cur.rowcount > 0
        # Re-index remaining turns
        if affected:
            conn.execute(
                "UPDATE messages SET turn_index=turn_index-1 WHERE session_id=? AND turn_index>?",
                (session_id, turn_index),
            )
            conn.execute(
                "UPDATE sessions SET updated=datetime('now'), turn_count=(SELECT COUNT(*) FROM messages WHERE session_id=?) WHERE id=?",
                (session_id, session_id),
            )
            conn.commit()
    return affected


def clear_current_session(session_id: str | None = None) -> bool:
    """Clear all messages from a session (keeps the session row)."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM messages WHERE 1=1")
        conn.execute(
            "UPDATE sessions SET updated=datetime('now') WHERE id=?",
            (session_id,) if session_id else ("",),
        )
        conn.commit()
    return True


def clear_all_sessions() -> int:
    """Delete ALL sessions and messages. Returns count of deleted sessions."""
    with _get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        # Use individual DELETE statements to avoid FTS5 trigger issues
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
        conn.commit()
    log.info("cleared all %d sessions", n)
    return n


# --- import/export ----------------------------------------------------------

def export_session_to_jsonl(session_id: str) -> bytes | None:
    """Export a session as JSONL bytes. Returns None if not found."""
    data = load_session(session_id)
    if not data:
        return None
    lines = []
    for msg in data["messages"]:
        lines.append(json.dumps(msg, ensure_ascii=False))
    return "\n".join(lines).encode("utf-8")


def import_from_jsonl(data: bytes | str, session_id: str | None = None) -> dict[str, Any]:
    """Import a JSONL conversation. Creates a new session and returns metadata."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")

    lines = [l.strip() for l in data.strip().split("\n") if l.strip()]
    messages = []
    for line in lines:
        try:
            msg = json.loads(line)
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)
        except (json.JSONDecodeError, TypeError):
            continue

    sid = session_id or _make_session_id()
    with _get_conn() as conn:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Check if session already exists — update it instead of duplicating
        existing = conn.execute("SELECT system_prompt FROM sessions WHERE id=?", (sid,)).fetchone()
        prompt = existing[0] if existing else ""
        conn.execute(
            "INSERT OR REPLACE INTO sessions(id, created, updated, system_prompt, endpoint, turn_count) VALUES (?,?,?,?,?,?)",
            (sid, created, created, prompt, "", len(messages)),
        )
        for idx, msg in enumerate(messages):
            conn.execute(
                "INSERT INTO messages(session_id, role, content, turn_index) VALUES (?,?,?,?)",
                (sid, msg["role"], msg["content"], idx),
            )

    return {"id": sid, "imported_turns": len(messages)}


def get_stats() -> dict[str, int]:
    """Return counts for display in the UI."""
    with _get_conn() as conn:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {"sessions": sessions, "messages": messages}


def save_session(
    sid: str | None,
    turns: list[dict[str, Any]],
    system_prompt: str | None,
    endpoint: str,
) -> dict[str, Any]:
    """Save (create or update) a session. Returns {id}."""
    if not turns:
        return {"created": False}

    if not sid:
        sid = _make_session_id()
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions(id, created, updated, system_prompt, endpoint, turn_count) VALUES (?,?,?,?,?,?)",
                (sid, created, created, system_prompt, endpoint, len(turns)),
            )
            for idx, msg in enumerate(turns):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                conn.execute(
                    "INSERT INTO messages(session_id, role, content, turn_index) VALUES (?,?,?,?)",
                    (sid, role, content, idx),
                )
        return {"id": sid}
    else:
        # Update existing session — replace all messages
        with _get_conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            conn.execute(
                "UPDATE sessions SET updated=datetime('now'), turn_count=?, system_prompt=?, endpoint=? WHERE id=?",
                (len(turns), system_prompt, endpoint, sid),
            )
            for idx, msg in enumerate(turns):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                conn.execute(
                    "INSERT INTO messages(session_id, role, content, turn_index) VALUES (?,?,?,?)",
                    (sid, role, content, idx),
                )
        return {"id": sid}


def delete_turn(session_id: str, turn_index: int) -> bool:
    """Delete a specific turn by its index within a session."""
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM messages WHERE session_id=? AND turn_index=?",
            (session_id, turn_index),
        )
        affected = cur.rowcount > 0
        if affected:
            # Re-index remaining turns
            conn.execute(
                "UPDATE messages SET turn_index=turn_index-1 WHERE session_id=? AND turn_index>?",
                (session_id, turn_index),
            )
            conn.execute(
                "UPDATE sessions SET updated=datetime('now'), turn_count=(SELECT COUNT(*) FROM messages WHERE session_id=?) WHERE id=?",
                (session_id, session_id),
            )
    return affected
