# qwen36-harness

Personal agentic harness around a private Qwen3.6 endpoint (currently a Vast.ai
RTX 5090 running llama.cpp + Unsloth UD GGUFs). Chat UI, tool registry, and
bidirectional CerebroCortex integration.

**Status:** Phase 2.5 SHIPPED (Apr 28, 2026, evening session continues). See
`docs/plans/2026-04-28-qwen36-harness.md` for the full implementation plan.

**Done:**
- Phase 0   repo, venv, endpoint config loader (5 tests)
- Phase 1   streaming httpx client, rich CLI chat (12 tests + 3 smoke)
- Phase 1.5 SSH-tunnel lockdown, llama-server bound to 127.0.0.1, vast_tunnel.sh helper
- Phase 2   tool registry, fs/calc/cerebro tools, agent loop, CLI integration
            (35 unit tests + 5 smoke tests passing)
- Phase 2.5 MCP client (`mcp>=1.0`): MCPManager spawns stdio servers,
            registers their tools as `<server>.<tool>`, `/mcp` slash commands,
            configs/mcp_servers.toml. fs-mcp verified end-to-end —
            Qwen3.6-35B-A3B successfully calls real `@modelcontextprotocol/server-filesystem`
            via JSON-RPC and answers from the file contents.
            (45 unit tests + 7 smoke tests, 52/52 green with full env)

**Live state:** Vast.ai NO 5090 instance 35758586 still running ($0.40/hr,
~$1.50 burned across the evening). Tunnel up at `127.0.0.1:8800`. Default
endpoint `vast-qwen36-moe` works end-to-end with agentic tool calls
(verified via `tests/test_smoke_agent.py` and `tests/test_smoke_mcp_agent.py`).

**Next up:** Phase 3 web UI → Phase 4 real Cerebro wiring → Phase 5 tool-piling.

## Why this exists

- Privacy first — no third-party telemetry on prompts/responses
- High-quality local-equivalent inference (110 t/s decode on rented 5090)
- Tool-loop substrate for experimenting with what Qwen3.6 can do agentically
- A bridge between cloud GPU and CerebroCortex's memory layer

## Spin-up dependency

Endpoint is brought up by `~/Projects/qwen36-vast/vast_up.sh` (separate repo).
See the `qwen36-on-vast-5090` skill for details.

## Quick start (once Phase 1 ships)

```bash
.venv/bin/python cli.py chat
# or
.venv/bin/python -m harness.ui_server
```

## License

Private — Andre's tools.
