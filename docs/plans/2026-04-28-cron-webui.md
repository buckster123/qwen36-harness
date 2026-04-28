# Phase 4.4: Cron Jobs Web UI

**Date:** 2026-04-28  
**Status: ✅ COMPLETE — committed as `4.4-ui` branch changes**

## Background

The cron scheduler (`src/harness/tools/cron.py`) exposes 5 tools via the tool registry:
`cron_register`, `cron_run_now`, `cron_list`, `cron_stop`, `cron_remove`.

They work correctly as agent-called tools, but the web UI has **no dedicated Cron Jobs panel**. Jobs only appear as raw tool-call panels in the chat stream. There is no at-a-glance dashboard showing what's running, intervals, run counts, or one-click actions.

## Goals (Shipped)

1. ✅ Added "Cron Jobs" panel to right sidebar (between MCP Servers and Sandbox).
2. ✅ Shows all jobs with name, status pill (`up`=running, `error`, `pending`), interval, run count.
3. ✅ Inline action buttons: ▶ Run Now, ■ Stop (when idle), ✕ Remove (with confirm).
4. ✅ Auto-refreshes on `refreshState()` — no separate endpoint needed.

## What was implemented

### Backend (`web.py`) — 1 change + 2 endpoints

| Change | Purpose |
|--------|---------|
| `"cron": cron_scheduler.list_jobs()` in `_state_payload` | Exposes all jobs via existing `/api/state` GET |
| `POST /api/cron/{name}/run_now` | One-shot execute of a registered job |
| `POST /api/cron/{name}/stop` | Stop a running job (keeps in registry) |

No separate cron registration endpoint — jobs are only created by the agent calling `cron_register`. The UI is read + control only.

### Frontend (`index.html`) — new panel + JS

**Sidebar:** `<h2>Cron Jobs</h2>` + `<div id="cron-list">` inserted between MCP Servers and Sandbox.

**JS:** `renderCron()` function (mirrors `renderMCP` pattern) with:
- Status pills: green `up` for running, red `error`, yellow `pending`
- Metadata: interval/schedule type + total runs counter
- Action buttons: ▶ run_now (always), ■ stop (when idle), ✕ remove (with confirmation)

**CSS:** New `.cron-row`, `.cron-head`, `.cron-meta`, `.cron-actions` styles with dark theme. Buttons match MCP button sizing (9px font, compact padding).
