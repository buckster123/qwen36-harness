# Architecture (living doc)

> Updated as design decisions are made. Each section starts as a stub and
> grows with the corresponding phase's work.

## High-level

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Web UI     │      │  CLI (harness)   │      │  CerebroCortex  │
│  127.0.0.1  │      │   chat / run     │      │  dream cycles   │
└──────┬──────┘      └────────┬─────────┘      └────────┬────────┘
       │                      │                         │
       └──────────────┬───────┴─────────────────────────┘
                      │
              ┌───────▼────────┐
              │    Agent       │  tool-loop driver
              │   (agent.py)   │
              └───────┬────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼────┐  ┌─────▼────┐  ┌────▼─────┐
   │ Client  │  │ Tools    │  │ Config   │
   │ (httpx) │  │ Registry │  │ (toml)   │
   └────┬────┘  └─────┬────┘  └──────────┘
        │             │
        │             └─→ filesystem, calc, web, shell, cerebro
        │
        └─→ Vast 5090  (or)  Krackan NPU  (or)  future-local
            OpenAI-compat /v1/chat/completions
```

## Decisions log

(Empty — fill in as we make calls)

- 2026-04-28: chose `httpx` over `openai` python package — fewer deps, async
  native, no risk of `openai` lib changing tool-call shapes from under us
- 2026-04-28: single-file HTML UI, no node build — keeps the bar to "open
  the file" instead of npm install
