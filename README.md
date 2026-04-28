# qwen36-harness

Personal agentic harness around a private Qwen3.6 endpoint (currently a Vast.ai
RTX 5090 running llama.cpp + Unsloth UD GGUFs). Chat UI, tool registry, and
bidirectional CerebroCortex integration.

**Status:** Phase 0 — scaffolding. See `docs/plans/2026-04-28-qwen36-harness.md`
for the full implementation plan.

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
