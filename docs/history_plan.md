# Conversation History Manager — Implementation Plan

## Problem
- Conversations live only in `WebSession.turns` (in-memory Python list)
- No persistence — server restart = history lost
- No export, import, or individual turn deletion
- CLI has `/save` and `/load` but web UI has no wiring to that

## Goals
1. Persist conversations to disk between server sessions
2. Allow export/import of entire conversations (JSONL format)
3. Allow delete of individual turns within a session
4. Provide a clean history management UI in the web interface
5. Keep cross-session context for Cerebro-based agent workflows

## Design Decisions

### Storage Format
- **Primary**: JSONL files, one per conversation session
- Location: `~/.harness/sessions/` (created on first use)
- Filename format: `<YYYYMMDD_HHMMSS>_<short_id>.jsonl`
- Each line = one message in standard OpenAI-compatible format:
  ```json
  {"role": "user"|"assistant", "content": "..."}
  ```
- Session metadata stored in separate JSON file alongside each `.jsonl`:
  ```json
  {
    "id": "20260429_193800_x7k2m",
    "created": "2026-04-29T19:38:00+02:00",
    "updated": "2026-04-29T19:42:15+02:00",
    "turn_count": 12,
    "system_prompt": "...",
    "endpoint": "vast-qwen36-moe"
  }
  ```

### Backend Changes

#### New Files
- `src/harness/storage.py` — Conversation storage module
  - `save_turns(turns: list[dict], system: str, endpoint: str) -> session_id`
  - `load_session(session_id: str) -> (turns, system, metadata)`
  - `delete_turn(session_id: str, turn_index: int)`
  - `list_sessions(limit: int=50) -> list[session_metadata]`
  - `clear_all()` — wipe all sessions

#### API Endpoints to Add
```
GET    /api/history/list?limit=50     → {ok:true, sessions:[{id, created, updated, turn_count, preview}]}
POST   /api/history/load              → {session_id} → replaces session.turns + system
POST   /api/history/export            → {session_id} → JSONL file download
GET    /api/history/export_all        → zipped JSONL files (optional v2)
DELETE /api/history/delete            → {session_id}
DELETE /api/history/clear             → clears current session turns only
```

#### Integration Points
- Every chat turn appends to `session.turns` AND calls `save_turns()`
- `POST /api/clear` calls `clear_all_sessions()` (not just clearing turns)
- On server startup, auto-discover existing sessions in `~/.harness/sessions/`

### Frontend Changes

#### New UI Component: History Sidebar Panel
Add a tabbed panel to the **right sidebar** replacing "Cron Jobs":

```
┌─ Right Sidebar ─────────────┐
│ [Conversation] [History]     │ ← tabs
│                            │
│ Current session: 12 turns   │
│ System prompt preview...    │
│ Endpoint: vast-qwen36-moe   │
│                            │
│ Actions:                   │
│ ┌──────────┐ ┌───────────┐ │
│ │  Export  │ │  Clear All│ │
│ └──────────┘ └───────────┘ │
│                            │
│ Recent Conversations:      │
│ • Today 19:42 · 12 turns   │
│ • Today 18:30 · 5 turns    │
│ • Yesterday 14:15 · 28     │
│                            │
│ [Upload JSONL]             │
└────────────────────────────┘
```

#### History Tab Features
- Auto-refresh list every 10s while open
- Click a session → loads it into current conversation (replaces turns)
- Each entry shows: timestamp, turn count, system prompt snippet
- Hover actions per entry: Load, Delete
- "Export All" button downloads all sessions as JSONL files in zip

### Implementation Steps (Order Matters)

**Phase 1: Storage Layer (backend-only)**
1. Create `src/harness/storage.py` with save/load/delete/list/clear functions
2. Add API endpoints for export/import/delete/clear
3. Wire up auto-save on every chat turn completion
4. Test CLI → storage roundtrip compatibility

**Phase 2: History UI (frontend-only, no backend changes)**
1. Add "History" tab to right sidebar (next to existing "Conversation" tab)
2. Fetch and display session list from `/api/history/list`
3. Wire up click-to-load, delete buttons
4. Export button → download as JSONL

**Phase 3: Polish & Edge Cases**
1. Search/filter sessions by keyword (searches message content)
2. Auto-cleanup old sessions (>7 days or >50 sessions)
3. Session preview in modal before loading (show messages, confirm replacement)
4. Keyboard shortcut to toggle history panel

## Technical Notes

### Cross-Session Context
- Cerebro handles long-term memory across sessions
- This storage is for LOCAL conversation history within the harness UI
- When loading a past session, it replaces current turns — but doesn't clear Cerebro memories
- Users can resume a session and continue building context in Cerebro

### Performance Considerations
- JSONL format: streaming read/write, no need to load full file into memory for large conversations
- Session listing reads only metadata JSON (not all messages)
- Delete turns removes the line from JSONL and rewrites index

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Storage fills disk | Auto-cleanup after N days or M sessions; show warnings in UI |
| Corrupted session file | Validate JSON on load; keep .bak backup of last clean state |
| Large conversation slow to load | Limit preview to first/last N messages; full view available via export |
| Race condition (save while loading) | Use file locks during write; queue save requests |

## Future Enhancements (Out of Scope)
- Markdown rendering of conversation messages in history panel
- Tagging/saving favorite sessions
- Session comparison (diff between two versions)
- Search across all sessions with regex support
- Export as HTML for pretty-printed reading
