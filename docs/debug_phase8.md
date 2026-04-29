# Phase 8 Debug Plan — History Panel + MCP Context Leak

## Status (Apr 29)
- History UI shipped as commit `7eefc61` but server had NOT been restarted after routes were added, so all `/api/history/*` endpoints returned 404.
- Auto-save hook (`_save_current_session`) present in current code but was never exercised due to stale server process.
- Server restarted mid-session — history endpoints now return 200 OK with empty session list.

## Open Issues

### Issue A: History Panel Not Visible (FIXED)
**Symptom:** Clicking the `history-btn` produced no visible UI response.
**Root Cause:** Running stale server code missing `/api/history/list`, `/api/history/load`, etc. The frontend JS `api()` call got HTTP 404 → threw → caught silently.
**Fix Applied:** Restarted uvicorn server with latest code from HEAD. Verified endpoints return `{ok: true, sessions: []}`.
**Remaining Test:** Verify user can see panel after hard-refresh (Ctrl+Shift+R in Firefox). Confirm toast error handling if DB operations fail.

### Issue B: Conversations Don't Stick to History
**Symptom:** After chatting, history panel shows "No conversations saved yet."
**Root Cause:** Old server code never had `_save_current_session()` hook at line 769 of `web.py`. Auto-save was simply not wired in.
**Fix Applied:** Server now runs HEAD code where auto-save fires after every turn completes. The `_chat_stream` generator calls `await asyncio.to_thread(_save_current_session, session)` between the SSE "end" event and the final yield.
**Remaining Test:** Send a multi-turn conversation → check DB at `~/.harness/harness.db` → verify sessions + messages tables populated → confirm History panel lists them.

### Issue C: MCP File Tools Leak Context (NEW — needs full debug round)
**Symptom:** When harness calls MCP file tools (search_files, read_file, etc.), the tool responses include massive outputs like entire directory trees, OS filesystem listings, or Python venv contents. These blow up the conversation context window (we've exceeded 200k tokens).
**Root Cause Candidates:**
- **search_files / rg backend:** Ripgrep searches may match inside `node_modules/`, `.venv/lib/`, and system paths unless explicitly excluded via `.gitignore` or tool config
- **read_file without limit:** Reading large files (or reading the wrong file path) can return hundreds of KB
- **MCP tool definitions:** The MCP server's tool output schemas may not have size caps
- **FS sandbox boundary issues:** The FsSandbox in `tools/filesystem.py` may not be properly scoping MCP file paths

**Debug Steps:**
1. Inspect the MCP server configurations in `~/.harness/mcp_config.yaml` (or wherever configured)
2. Check if ripgrep is called with `--no-ignore`, `--hidden`, or without `.gitignore` exclusions
3. Add debug logging to trace which MCP tool is producing the oversized response
4. Implement response size limits on MCP tool output (e.g., cap at 10KB, truncate with notice)
5. Add auto-exclusion patterns for `.venv/`, `node_modules/`, `/usr/lib/`, etc. in search files tool
6. Consider adding a `limit` parameter to all MCP file tools with sensible defaults

**Potential Fixes:**
- In `search_files`: add `--no-ignore-vcs --hidden` removal, or explicitly pass `.gitignore` paths for exclusion
- In `read_file`: enforce max line count (already has 2000-line cap but default might be too high)
- In `tools/filesystem.py`: add path validation to prevent escaping the sandbox directory
- Add a context-budget tracker that warns or truncates when session turns exceed N tokens

### Issue D: History Load Overwrites Current Chat (UX concern)
**Symptom:** Clicking a session row loads it into the current chat, replacing all existing turns.
**Risk:** User accidentally loses current conversation while browsing history.
**Proposed Fix:** Show confirmation modal before loading past session: "Load this session? This will replace your current conversation." Or offer an "append" vs "replace" option.

## Files to Inspect Next Session
- `src/harness/mcp.py` — MCP server config and tool invocation
- `src/harness/tools/filesystem.py` — FsSandbox boundaries and search implementation
- `src/harness/tools/search_files_tool.py` (if exists as separate file)
- `~/.harness/mcp_config.yaml` (or wherever MCP servers are configured)
- `/home/andre/projects/qwen36-harness/src/harness/static/index.html` — toast visibility CSS

## Quick Smoke Tests to Run Next Session
1. `curl http://127.0.0.1:7777/api/history/list` → should return `{ok: true, sessions: [...]}` (after a conversation)
2. Check DB directly: `sqlite3 ~/.harness/harness.db "SELECT COUNT(*) FROM sessions; SELECT COUNT(*) FROM messages;"`
3. Open Firefox → hard refresh → send test message → verify auto-save in DB after turn completes
4. Test history panel: open it, see session list, click load, verify chat populates
