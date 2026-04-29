# Phase 6: Web Search & Code Execution

## Goal
Give the agent two fundamental capabilities: **look up external information** and **run arbitrary code**. Both tools run local (no auth, no APIs) with strict safety boundaries.

```
Agent ──spawns──→ web.search(query)    → DuckDuckGo HTML snippet
                ──runs──→  code.exec(code)  → sandboxed subprocess output
```

## Design Decisions

### 1. Web Search via DuckDuckGo HTML
- **Why:** Zero config, no API keys, no rate-limit friction for basic queries
- **Method:** GET `https://html.duckduckgo.com/html/?q=<query>` — scrape `<a class="result__url">` + `<a class="result__snippet">` from response
- **Result format:** List of {url, title, snippet} — max 5 results to keep context tight
- **Dependence on httpx** which is already a harness dep

### 2. Code Execution via Subprocess Sandbox
- **Why:** Fast (no container startup), fits existing FsSandbox philosophy, fully local
- **Mechanism:** `subprocess.run()` with timeout + blocked-binary list + restricted cwd
- **Safety:** Write operations allowed only inside sandbox root (`~/.harness-code-sandbox/` or configurable)
- **Timeout:** 60s default (configurable per-call), kills process tree on timeout
- **Shell vs Python:** Supports both. Default = `python3 -c "..."`. Shell mode available via explicit flag.

### 3. Tool Namespacing
- `web.search(query, max_results=5)` — search the web
- `code.exec(code, lang="python", args=None, timeout_s=60)` — run code in sandbox

### 4. Sandbox Root Directory
- Default: `~/.harness-code-sandbox/` (auto-created)
- Configurable via CLI flag: `--code-sandbox /path`
- Both `FsSandbox` and `CmdSandbox` use same confinement philosophy but are separate concerns

## Implementation Plan

### Step 1: Core modules
**`src/harness/tools/web_search.py`**
```python
import httpx
from bs4 import BeautifulSoup  # OR: stdlib html.parser (to keep deps zero)
# Actually: use regex/simple string parsing — no new deps
# Parse the DuckDuckGo HTML response manually

@registry.tool("web.search")
def search(query, max_results=5): ...
```

**`src/harness/tools/code_exec.py`**
```python
import subprocess, asyncio, signal, os, tempfile
from pathlib import Path

class CmdSandbox:
    def __init__(self, root=None):
        self.root = Path(root) or (Path.home() / ".harness-code-sandbox")
        self.root.mkdir(parents=True, exist_ok=True)
        self.blocked_binaries = {"rm", "dd", "mkfs", "fdisk", "format"}

    def run(self, code: str, lang: str = "python", timeout_s: int = 60, args: list[str] | None = None) -> dict:
        """Execute code in sandbox. Returns {exit_code, stdout, stderr, timed_out}."""
        # Build command based on lang
        # Enforce cwd=self.root
        # Timeout + process tree kill
        ...

@registry.tool("code.exec")
def execute(code: str, lang: str = "python", timeout_s: int = 60, args: list[str] | None = None) -> dict: ...
```

### Step 2: Register on default_registry
- Import modules in `tools/__init__.py` (same pattern as calc/cron/fs/subagent)
- Call `register()` in `cli.py` startup alongside other tool registration
- Call `register()` in `web.py` create_app() alongside other tool registration

### Step 3: Update system prompt
Add section to `tools/system_prompt.md`:
```markdown
### web.search (Web Search — NEW)
Query DuckDuckGo for information. Returns list of {url, title, snippet}.
- web.search(query) → [results]

### code.exec (Code Execution — NEW)
Run arbitrary code in a sandboxed subprocess. Safe for:
- Math/data processing (pandas, numpy, math)
- File reads/writes inside the sandbox root
- Network calls to external APIs
- Shell commands (explicit via lang="sh")

Constraints:
- Writes only allowed in ~/.harness-code-sandbox/
- Processes killed after 60s timeout
- Blocked: rm, dd, mkfs, fdisk, format, and system-modifying commands
```

### Step 4: CLI integration
- Add `--code-sandbox` argument to argparse (optional)
- Pass to CmdSandbox constructor at startup (in cli.py)
- Show sandbox path in `/info` output

### Step 5: Tests
**`tests/test_web_search.py`** (3-4 tests)
- Parse valid DDG HTML response → extract results
- Empty/fail response → empty list
- max_results limit works

**`tests/test_code_exec.py`** (4-5 tests)
- Python syntax error → exit_code != 0, stderr populated
- Successful math computation → correct stdout
- Timeout enforcement → timed_out = True
- Blocked binary → denied/error message
- Wrote outside sandbox → permission denied / path error

## Files to Create/Modify

### New:
- `src/harness/tools/web_search.py` — web search tool + registration
- `src/harness/tools/code_exec.py` — code exec tool + CmdSandbox class
- `tests/test_web_search.py` — search tests
- `tests/test_code_exec.py` — code exec tests

### Modified:
- `src/harness/tools/__init__.py` — import + register web_search, code_exec
- `src/harness/cli.py` — add --code-sandbox arg + register tools at startup
- `src/harness/web.py` — register tools in create_app()
- `src/harness/tools/system_prompt.md` — document new tools
- `ROADMAP.md` (if exists) — update status

## Technical Details

### web_search parsing strategy
DuckDuckGo HTML response structure:
```html
<a class="result__url" href="...">...</a>
<a class="result__snippet" href="...">...</a>
```
We iterate the response body with regex to find pairs of result__url + result__snippet — no external parser needed.

### code_exec safety model
| Layer | Protection | How |
|-------|-----------|-----|
| Process isolation | Subprocess + PID namespace | `subprocess.Popen(..., start_new_session=True)` |
| Time limit | kill after N seconds | `signal.SIGKILL` after timeout_s |
| Path confinement | cwd = sandbox root | Only reads/writes allowed inside sandbox.root |
| Binary blocking | Pre-flight check | Reject if first argument (command name) is in blocked list |
| Memory | OS limits (default ~256MB per process) | No explicit cgroups, but OS naturally caps it |
