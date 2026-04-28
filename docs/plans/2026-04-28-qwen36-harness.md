# Qwen3.6 Private Endpoint Harness — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan
> task-by-task once we agree on the shape. Until then, this is a roadmap doc.

**Goal:** Build a personal agentic harness around the private Qwen3.6 endpoint
running on rented Vast.ai 5090 (or any future local/cloud Qwen3.6) so Andre can
chat, run tool loops, and feed it into CerebroCortex — with privacy as a first-
class property and a UI that beats the borrowed ryzenai-serve chat.html.

**Architecture:**
A thin Python core (`harness/`) that wraps any OpenAI-compatible endpoint, plus
a curated tool registry (filesystem, web fetch, shell, cerebro, calculator…)
exposed via the OpenAI tools schema. Two surfaces consume the core: a
single-page web UI (one HTML file, no build step) and a CLI for headless runs.
Cerebro integration is bidirectional — harness can call cerebro tools (recall,
remember, dream) AND cerebro can call the harness as its primary LLM endpoint.

**Tech Stack:**
- Python 3.12+ (already in `~/ryzen_ai/venv` for hf, or fresh venv per project)
- `httpx` for the OpenAI client (async, streaming, no openai-package coupling)
- `rich` for CLI rendering
- single-file HTML/JS web UI (no node, no build, just open the file)
- pytest for tests
- CerebroCortex Python API (already installed in `~/projects/CerebroCortex`)

**Endpoint reality:** Qwen3.6-35B-A3B Q5_K_XL on Vast 5090 — 110 t/s decode,
128K ctx, OpenAI-compat at `http://<IP>:<PORT>/v1`. Thinking-mode default;
reasoning lands in `.message.reasoning_content`, content in `.message.content`.
See `mlops/qwen36-on-vast-5090` skill for spin-up.

**Privacy contract:** No telemetry from harness. No logging of prompts/responses
to disk unless user opts in via `--record <path>`. No external HTTP calls except
to user-explicit endpoints. Tool calls show what they're about to do before they
do it (interactive mode) or always-allow per session.

---

## Project layout (target)

```
~/projects/qwen36-harness/
  README.md
  pyproject.toml
  docs/
    plans/2026-04-28-qwen36-harness.md   ← this file
    architecture.md                       ← grows as we decide things
    cerebro-integration.md                ← phase 2 doc
  src/harness/
    __init__.py
    client.py        OpenAI-compat client (httpx, async, streaming, dual content)
    tools/
      __init__.py    tool registry, dispatch, schema export
      filesystem.py  read_file, list_dir, write_file (sandboxed)
      shell.py       run_command (allowlist + confirm)
      web.py         fetch_url, search (uses local hermes-style web tool)
      cerebro.py     remember, recall, list_episodes, dream_status
      calc.py        evaluate (sympy)
    agent.py         tool-loop driver: call → tool → call → ...
    config.py        endpoints, defaults, tool allowlists
    ui_server.py     tiny FastAPI/aiohttp serving static + SSE
  src/harness/static/
    index.html       single-page UI, dark theme, streaming, tool-call inspector
    app.js           if needed; otherwise inline
  tests/
    test_client.py
    test_tools_*.py
    test_agent_loops.py
  tools/                           devops scripts (vast endpoint sync, etc.)
  configs/
    endpoints.toml                 list of endpoints (vast, npu, future-local)
  cli.py                           entry point: harness chat, harness run, harness eval
```

## Out of scope (explicit YAGNI)

- Multi-user auth
- Database (config is files; conversation history is files)
- Image generation
- Audio in/out
- Custom training / fine-tuning
- Hosting this anywhere other than localhost
- Replacing CerebroCortex's own LLM client (we wrap, we don't replace)

---

## Phase 0 — Repo + skeleton (≈30 min)

### Task 0.1: Initialise repo and remote

**Files:**
- Create: `~/projects/qwen36-harness/.gitignore`
- Create: `~/projects/qwen36-harness/README.md`
- Create: `~/projects/qwen36-harness/pyproject.toml`

**Steps:**
1. `cd ~/projects/qwen36-harness && git init -b main` (already done)
2. Write `.gitignore` (Python, venv, .env, .ruff_cache, conversation logs)
3. Write minimal `README.md` (one-paragraph description + link to plan)
4. Write `pyproject.toml` with httpx, rich, pytest, ruff, mypy
5. `gh repo create buckster123/qwen36-harness --private --source=. --remote=origin`
6. `git add . && git commit -m "chore: initial scaffold" && git push -u origin main`

**Verification:** `gh repo view buckster123/qwen36-harness` shows the repo.

### Task 0.2: Python venv + deps

**Files:**
- Create: `~/projects/qwen36-harness/.venv/`

**Steps:**
1. `python3.12 -m venv .venv`
2. `.venv/bin/pip install -U pip httpx rich pytest pytest-asyncio ruff mypy aiohttp`
3. `.venv/bin/pip freeze > requirements-frozen.txt`
4. Commit `requirements-frozen.txt`.

**Verification:** `.venv/bin/python -c "import httpx, rich; print('ok')"` prints ok.

### Task 0.3: Endpoint config

**Files:**
- Create: `configs/endpoints.toml`
- Create: `src/harness/__init__.py`
- Create: `src/harness/config.py`

**Content of `configs/endpoints.toml`:**
```toml
[endpoints.vast-qwen36-moe]
base_url = "http://79.160.189.79:12515/v1"
model    = "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
api_key  = "sk-anything"
default_max_tokens = 2048
default_temperature = 0.7
mode = "thinking"           # thinking | nonthinking | coding
description = "Vast.ai NO 5090, private, ~$0.40/hr"

[endpoints.npu-llama]
base_url = "http://127.0.0.1:8000/v1"
model    = "Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K"
api_key  = "sk-local"
default_max_tokens = 1024
default_temperature = 0.7
mode = "nonthinking"
description = "Local Krackan NPU, ~26 t/s"

[default]
endpoint = "vast-qwen36-moe"
```

**Verification:** `.venv/bin/python -c "from harness.config import load_endpoints; print(load_endpoints())"` prints the dict.

---

## Phase 1 — Client + minimal CLI chat (≈1 hr)

### Task 1.1: OpenAI-compat client (TDD)

**Files:**
- Create: `src/harness/client.py`
- Create: `tests/test_client.py`

**Behavior to capture in tests:**
- `Client.complete(messages, **kw)` returns a structured result with both
  `content` and `reasoning_content` (split for Qwen3.6 thinking mode)
- Streaming: `Client.stream(messages, **kw)` yields `(kind, text)` where
  kind ∈ {'reasoning', 'content', 'tool_call', 'done'}
- Tool calls: passes `tools` list, gets back parsed `tool_calls` with id,
  name, parsed args
- Handles transient connection errors with exponential backoff (3 tries)
- Respects `max_tokens` from endpoint default if not passed

**TDD pattern:** mock httpx with `respx` so tests are offline. Real-endpoint
smoke is separate.

### Task 1.2: CLI `harness chat`

**Files:**
- Create: `cli.py`

**Behavior:**
- `python cli.py chat` opens an interactive REPL using `rich.console`
- Streams `reasoning_content` in dim grey, `content` in normal
- `:endpoints` lists configured endpoints
- `:use vast-qwen36-moe` switches endpoint
- `:mode thinking|nonthinking|coding` switches MODE
- `:max 4096` sets max_tokens
- `:save chat.json` persists transcript
- `:quit` exits

**Verification:** Manual smoke. Have a 3-turn conversation with the live Vast
endpoint, verify thinking shows dim and answer shows clear, total round-trip
feels fast.

### Task 1.3: Real-endpoint smoke test

**Files:**
- Create: `tests/test_smoke_vast.py` (skipped unless `VAST_BASE` env set)

**Steps:**
- `VAST_BASE=http://79.160.189.79:12515 .venv/bin/pytest tests/test_smoke_vast.py`
- Asserts: /v1/models reachable, completion returns content+reasoning,
  tool_calls round-trip works end to end.

---

## Phase 2 — Tool registry (≈2 hr)

### Task 2.1: Tool plumbing

**Files:**
- Create: `src/harness/tools/__init__.py`

**Design:**
- `@tool(name, description, schema)` decorator registers a Python function
- Each tool exports an OpenAI-tools-schema entry via `registry.export_schema()`
- `registry.dispatch(name, args)` calls the function, returns string result
- `registry.allowlist(["fs.read", "calc"])` filters which tools are exposed
- Permission gates: `confirm=True` tools prompt the user once; once-allowed
  for that session unless re-invoked with different args (configurable)

**TDD:** test registration, schema export shape, dispatch happy/error paths.

### Task 2.2: Filesystem tool (sandboxed)

**Files:**
- Create: `src/harness/tools/filesystem.py`
- Create: `tests/test_tools_fs.py`

**Functions:**
- `fs.read(path, lines: int=200)` — read first N lines
- `fs.list(path)` — list directory
- `fs.write(path, content, confirm=True)` — write file, requires confirmation
- All paths resolved against a configurable root (default `~/`); reject
  anything escaping it.

### Task 2.3: Calculator (safe, no `eval`)

**Files:**
- Create: `src/harness/tools/calc.py`
- Create: `tests/test_tools_calc.py`

**Functions:**
- `calc.evaluate(expression)` using `sympy.sympify` (no arbitrary code exec)
- Handles units optionally via `pint` (later)

### Task 2.4: Web fetch (HTML→markdown)

**Files:**
- Create: `src/harness/tools/web.py`
- Create: `tests/test_tools_web.py`

**Functions:**
- `web.fetch(url)` — GET via httpx, return markdown extract using
  `markdownify` or `readability-lxml` + plain text fallback
- `web.search(query)` — wraps DuckDuckGo HTML scraper for now
  (no API key needed; switch to a real provider later)

### Task 2.5: Shell tool (allowlist)

**Files:**
- Create: `src/harness/tools/shell.py`
- Create: `tests/test_tools_shell.py`

**Functions:**
- `shell.run(cmd)` only if `cmd[0]` is in an allowlist
  (`ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `git status`, ...)
- Anything else → error message + nudge model toward allowlisted variant
- Captures stdout/stderr, truncates to 8 KB

### Task 2.6: Wire registry into agent

**Files:**
- Create: `src/harness/agent.py`
- Create: `tests/test_agent_loops.py`

**Behavior:**
- `Agent.run(prompt, tools=[...])` runs the OpenAI tool-loop:
  - Send messages + tool schemas
  - If response has `tool_calls`, dispatch each, append `role: tool` results
  - Repeat until model emits final `content` without tool calls
  - Cap iterations (default 10) and total tokens (default 30k)
- Streaming events: `('llm_start'|'tool_call'|'tool_result'|'llm_end'|'done', payload)`

---

## Phase 3 — Web UI (≈2 hr)

### Task 3.1: Single-page UI with streaming

**Files:**
- Create: `src/harness/static/index.html`
- Create: `src/harness/ui_server.py`

**Server:**
- aiohttp app on `127.0.0.1:7777`
- `GET /` — serves `index.html`
- `POST /chat` — streams SSE: `reasoning_delta`, `content_delta`,
  `tool_call`, `tool_result`, `done`
- `GET /endpoints` — returns the configured endpoints list
- `GET /tools` — returns the registry's exported schemas

**UI:**
- Borrow chat.html's dark aesthetic (we already love it)
- Sidebar: endpoint selector, mode selector, tool-allowlist checkboxes
- Main: messages + thinking-bubble (collapsible) + tool-call cards
  (showing name + args + result, expandable)
- Bottom: textarea + send + clear + save-transcript

**Verification:** open `http://127.0.0.1:7777`, chat with the model, watch
thinking stream in dim, content stream after, click `+tool` to enable
calc, ask "what's 17 * 91?", see tool_call and tool_result render inline.

### Task 3.2: Tool-call confirmation UX

**Behavior:**
- Tools marked `confirm=True` produce a yellow "approve?" pill in the UI
  before dispatch. Click ✓ or ✗.
- Always-allow checkbox per tool for the session.

---

## Phase 4 — Cerebro integration (≈2 hr)

This is bidirectional and the most "this is the point of the project" phase.

### Task 4.1: Cerebro as a tool inside the harness

**Files:**
- Create: `src/harness/tools/cerebro.py`
- Create: `tests/test_tools_cerebro.py`

**Functions exposed to the model:**
- `cerebro.recall(query, top_k=5, memory_types=None)` — wraps
  `cerebro_cortex.api.recall(...)`
- `cerebro.remember(content, tags=None, salience=None)` — wraps
  `cerebro_cortex.api.remember(...)`
- `cerebro.episode_start(title)` / `episode_add_step` / `episode_end`
- `cerebro.list_intentions()` — pending TODOs
- `cerebro.dream_status()` — what dream cycle thinks

**Implementation:** Import the CC Python API directly (we own the install).
Tag every memory created via the harness with `source=harness` so we can
distinguish later.

**TDD:** uses a tmpdir-scoped CC store so we don't pollute the real one.

### Task 4.2: Harness as CC's primary LLM endpoint

**Files:**
- Modify: `~/.cerebro-cortex/settings.json` (manual or via CLI)

**Steps:**
1. Add a new endpoint entry pointing at the Vast box:
   ```json
   "qwen36-vast": {
       "base_url": "http://79.160.189.79:12515/v1",
       "model": "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",
       "max_tokens": 2048
   }
   ```
2. Test a manual `cerebro recall "anything"` command and verify it now
   uses Qwen3.6 instead of NPU-Llama.
3. Run a small dream cycle: `cerebro dream --max-llm-calls 4`. Verify it
   uses the Vast endpoint, completes successfully, doesn't fall back.

**Why both directions?** Cerebro-as-tool gives the harness contextual memory
across sessions (the model can ask itself "what did Andre say about X last
week?"). Harness-as-CC's-LLM upgrades CC's reasoning quality from 3B-NPU to
35B-MoE for offline analysis tasks.

### Task 4.3: Documentation

**Files:**
- Create: `docs/cerebro-integration.md`

Diagram + a few example prompts that exercise the bidirectional path.

---

## Phase 5 — Tool-piling experiments (open-ended)

Each "tool" added is a mini-experiment: drop it in, run a representative
agentic prompt, observe quality, log results in `docs/experiments/NN-toolname.md`.

Suggested order:
1. **calc + fs.read** — baseline tool-loop sanity
2. **web.fetch + web.search** — research-y queries, "summarize this URL"
3. **shell + git** — code-aware questions about ~/projects/
4. **cerebro.recall** — long-term memory probe
5. **cerebro.remember + episode_*** — write-back behavior, schema formation
6. **multi-tool: web + cerebro** — "research X, then remember key points"
7. **vision (requires MMPROJ=F16 spin-up)** — drop screenshot, ask questions

For each: capture token counts, t/s observed, tool-call success rate, any
schema mangling on nested args, model behavior under failure (does it retry,
does it hallucinate). Goal is empirical confidence in Qwen3.6-35B-A3B as a
real agentic driver before we trust it for anything serious.

---

## Open questions (decide before phase 1 starts)

1. **Streaming UX**: should tool calls render BEFORE the model's outer
   response is complete, or buffer until done? Recommend: render tool
   cards live as they arrive, model's wrapper text continues streaming
   below. Mirrors what Claude / ChatGPT do.

2. **Conversation persistence**: jsonl per session? sqlite? plain markdown?
   Recommend: jsonl, one file per session in `~/.local/share/qwen36-harness/`,
   no compression, never deleted automatically.

3. **Endpoint health checks on startup**: fail loud if Vast endpoint is down
   (we destroyed the instance), or fall back to NPU? Recommend: fail loud
   with a clear message and a one-liner suggestion to either spin up Vast
   or `:use npu-llama`.

4. **Tool execution sandbox**: shell tool runs in a subprocess with the
   harness's user, full home-dir access. Should it run under firejail or
   in a container? Recommend: defer until phase 5 — for now allowlist is
   the only safety, and we trust ourselves not to ask the model to
   `rm -rf ~`.

5. **Authentication on the Vast endpoint**: anyone on the internet can
   currently hit `http://79.160.189.79:12515` and use OUR rented GPU.
   Recommend: add a `--api-key` to llama-server's launch.sh OR run a
   tiny SSH-tunnel pattern (`ssh -L 8000:localhost:8000 root@vast`) so
   the endpoint isn't internet-facing. **This should be a phase-1.5 task
   before we start hitting it heavily.**

---

## Risk register

| Risk | Mitigation |
|---|---|
| Vast box dies mid-session (host reboot, network blip) | Health check on send; "endpoint dead" message with reconnect button. Save transcript on every turn. |
| Qwen3.6 mangles tool schemas with deep nesting | Empirical test in Phase 5; if it happens, add a "flatten args" pass before sending |
| CerebroCortex API drifts (we patched CC commit `98156a1` for settings load) | Pin CC version in pyproject.toml extras; add a smoke test that imports CC fresh each test run |
| 128K ctx + heavy thinking at temp=1.0 burns wallclock | Default to 16K ctx in Phase 1, raise per-request via `:ctx 65536` etc. |
| Endpoint exposed unauthenticated to internet | Phase 1.5 — wrap with API key OR SSH tunnel before tool-pile experiments |
| Andre forgets the box is running and burns $9.60/day | `harness status` prints meter time + estimated cost + "destroy?" prompt; cron warns if up >12h |

---

## Success criteria

By end of Phase 4 we have:

- [ ] `harness chat` opens a fast, private chat with the Vast endpoint that
      feels qualitatively better than the borrowed chat.html (thinking
      visualization, tool calls, transcript save, endpoint hot-swap)
- [ ] Web UI at `http://127.0.0.1:7777` provides the same UX in a browser
- [ ] Five tools wired in: filesystem, calc, web, shell, cerebro
- [ ] CC dream cycles can run on Qwen3.6-35B-A3B as primary
- [ ] One end-to-end agentic loop (e.g. "summarize the last 3 sessions in my
      cerebro memory and write the result to a markdown file") works
      reliably 5/5 attempts
- [ ] Repo is on GitHub, commits per task, README explains how to run

By end of Phase 5 we have:

- [ ] `docs/experiments/` filled with 7-10 mini-experiment writeups
- [ ] A clear sense of what Qwen3.6 is and isn't good at agentically
- [ ] A reusable harness Andre wants to keep using even after this evening
