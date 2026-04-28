# Phase 4.4: Cron Jobs Web UI

**Date:** 2026-04-28  
**Status:** Planning — not yet implemented  

## Background

The cron scheduler (`src/harness/tools/cron.py`) exposes 5 tools via the tool registry:
`cron_register`, `cron_run_now`, `cron_list`, `cron_stop`, `cron_remove`.

They work correctly as agent-called tools, but the web UI has **no dedicated Cron Jobs panel**. Jobs only appear as raw tool-call panels in the chat stream. There is no at-a-glance dashboard showing what's running, intervals, run counts, or one-click actions.

## Goals

1. Add a "Cron Jobs" section to the right sidebar (between MCP Servers and Sandbox).
2. Show all jobs with name, status pill, interval, run count, and running indicator.
3. Provide inline action buttons: Run Now, Stop, Remove.
4. Auto-refresh on `refreshState()` (polls a new `/api/cron` endpoint).

## Scope

### Backend (`web.py`) — 3 new endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/cron` | GET | List all jobs (delegates to `scheduler.list_jobs()`) |
| `/api/cron/{name}/run_now` | POST | One-shot execute of a registered job |
| `/api/cron/{name}/stop` | POST | Stop a running job (keeps in registry) |

Note: No cron registration via API — jobs are only registered by the model calling `cron_register`. The UI is read + control only.

### Frontend (`index.html`) — new panel + JS

**HTML:** New `<h2>Cron Jobs</h2>` + `<div id="cron-list">` in right sidebar, below MCP servers and above Sandbox.

**JS functions (same pattern as `renderMCP`):**
- `renderCron()` — fetches state, populates job rows with status pills and action buttons
- Each row: name + pill (`up`/`down`/`warn`) + metadata + [run_now] [stop] [remove] buttons

**CSS:** Reuses existing `.row`, `.pill`, `aside h2` styles. Buttons inherit `button { cursor: pointer }` with `.primary` / `.danger` modifier classes matching the MCP server button pattern (small, 10px font).

### State payload (`_state_payload`)
Add `"cron": scheduler.list_jobs()` to the JSON returned by `/api/state`. This avoids an extra HTTP call — cron data is part of the normal state refresh.

## Implementation Order

1. Patch `web.py`: add `"cron"` key to `_state_payload`, import `scheduler` from `.tools.cron`.
2. Patch `index.html`: add "Cron Jobs" section HTML in right sidebar.
3. Patch `index.html`: add `renderCron()` JS function + action handlers.
4. Patch `index.html`: call `renderCron()` from within `refreshState()`.
5. Git commit + push.

## Files to modify

- `src/harness/web.py` — _state_payload cron key + import scheduler
- `src/harness/static/index.html` — sidebar HTML + JS render function
