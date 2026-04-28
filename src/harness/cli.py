"""Interactive CLI chat REPL for qwen36-harness.

Run with:
    .venv/bin/python -m harness.cli chat
or (after `pip install -e .`):
    harness chat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from .agent import Agent, AgentEvent, AgentLimits
from .client import HarnessClient, StreamEvent
from .config import Config, Endpoint, load_config
from .tools import default_registry
from .tools.calc import register as register_calc
from .tools.cerebro import register as register_cerebro
from .tools.filesystem import FsSandbox, register as register_fs
from .mcp import MCPManager, load_mcp_config

# --- styling ------------------------------------------------------------------

console = Console()

C_USER = "bold cyan"
C_ASSISTANT = "white"
C_REASONING = "dim italic grey50"
C_TOOL = "yellow"
C_SYS = "dim"
C_ENDPOINT = "magenta"
C_OK = "green"
C_ERR = "bold red"


def banner(cfg: Config, ep: Endpoint, *, show_thinking: bool) -> None:
    console.print()
    console.print(
        Panel(
            (
                f"[{C_OK}]qwen36-harness[/]    "
                f"endpoint=[{C_ENDPOINT}]{ep.name}[/]   "
                f"model=[white]{ep.model}[/]\n"
                f"url={ep.base_url}    mode=[bold]{ep.mode}[/]    "
                f"max_tokens={ep.default_max_tokens}    "
                f"temp={ep.default_temperature}    "
                f"thinking-visible={'yes' if show_thinking else 'no'}"
            ),
            title="ready",
            border_style=C_OK,
            padding=(0, 1),
        )
    )
    console.print(
        f"[{C_SYS}]/help for commands  ·  /quit to exit  ·  Ctrl-C cancels in-flight gen[/]"
    )


HELP = """
[bold]Slash commands[/]
  /help                show this
  /quit | /exit        leave
  /endpoints           list configured endpoints
  /use <name>          switch endpoint  (e.g. /use vast-qwen36-moe)
  /mode <m>            thinking | nonthinking | coding
  /max <n>             set max_tokens for next message
  /temp <f>            set temperature for next message
  /system <text>       set/replace system prompt (also: /system to clear)
  /think on|off        show or hide reasoning_content stream
  /tools               list registered tools and their state
  /tools on|off <name> enable/disable a specific tool
  /agent on|off        toggle tool-loop mode (default off = plain chat)
  /mcp                 list configured MCP servers + state
  /mcp start <name>    spawn an MCP server, register its tools
  /mcp stop <name>     stop an MCP server, unregister its tools
  /sandbox             show filesystem sandbox root
  /clear               drop conversation history (keeps system prompt)
  /save [path]         save transcript as JSON
  /info                current state, last response stats
  /retry               regenerate the last assistant turn
"""


# --- transcript helpers -------------------------------------------------------


class Conversation:
    def __init__(self, system: str | None = None) -> None:
        self.system = system
        self.turns: list[dict[str, Any]] = []
        self.last_stats: dict[str, Any] = {}

    def messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.extend(self.turns)
        return out

    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})

    def add_assistant(self, text: str, reasoning: str = "") -> None:
        msg = {"role": "assistant", "content": text}
        if reasoning:
            # NOTE: We intentionally do NOT send reasoning back to the model in
            # subsequent turns by default — Qwen3.6 supports `preserve_thinking`
            # but it bloats context. Stash it locally for transcript saves only.
            msg["_reasoning"] = reasoning
        self.turns.append(msg)

    def pop_last_assistant(self) -> None:
        while self.turns and self.turns[-1]["role"] != "user":
            self.turns.pop()

    def save(self, path: Path) -> None:
        data = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "system": self.system,
            "turns": self.turns,
            "last_stats": self.last_stats,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# --- interaction loop ---------------------------------------------------------


async def stream_one_turn(
    client: HarnessClient,
    convo: Conversation,
    *,
    max_tokens: int,
    temperature: float,
    show_thinking: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Stream a single assistant turn (no tools). Returns (content, reasoning, stats)."""
    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    started_content = False
    started_reasoning = False
    t_start = time.perf_counter()
    t_first_token: float | None = None
    n_completion = 0
    finish_reason = ""

    stream = await client.stream(
        convo.messages(),
        max_tokens=max_tokens,
        temperature=temperature,
    )

    try:
        async for ev in stream:
            if ev.kind == "reasoning" and show_thinking:
                if not started_reasoning:
                    console.print(Rule(title="reasoning", style="dim"))
                    started_reasoning = True
                console.print(Text(ev.text, style=C_REASONING), end="")
                reasoning_buf.append(ev.text)
                if t_first_token is None:
                    t_first_token = time.perf_counter()
            elif ev.kind == "reasoning":
                reasoning_buf.append(ev.text)
                if t_first_token is None:
                    t_first_token = time.perf_counter()
            elif ev.kind == "content":
                if started_reasoning:
                    console.print()
                    console.print(Rule(title="answer", style=C_OK))
                    started_reasoning = False
                started_content = True
                console.print(Text(ev.text, style=C_ASSISTANT), end="")
                content_buf.append(ev.text)
                if t_first_token is None:
                    t_first_token = time.perf_counter()
            elif ev.kind == "usage":
                n_completion = ev.data.get("completion_tokens", 0)
            elif ev.kind == "done":
                finish_reason = ev.data.get("finish_reason", "")
            elif ev.kind == "error":
                console.print()
                console.print(Panel(ev.text, title="error", border_style=C_ERR))
    except KeyboardInterrupt:
        console.print()
        console.print(f"[{C_SYS}](generation cancelled)[/]")
        finish_reason = "cancelled"

    elapsed = time.perf_counter() - t_start
    ttft = (t_first_token - t_start) if t_first_token else 0.0
    tps = (n_completion / elapsed) if (elapsed > 0 and n_completion) else 0.0
    stats = {
        "completion_tokens": n_completion,
        "elapsed_s": round(elapsed, 3),
        "ttft_s": round(ttft, 3),
        "tokens_per_second": round(tps, 1),
        "finish_reason": finish_reason,
    }
    console.print()
    console.print(
        f"[{C_SYS}]"
        f"completion={n_completion}t  "
        f"ttft={stats['ttft_s']}s  "
        f"elapsed={stats['elapsed_s']}s  "
        f"speed={stats['tokens_per_second']} t/s  "
        f"finish={finish_reason}"
        f"[/]"
    )
    return "".join(content_buf), "".join(reasoning_buf), stats


async def stream_agent_turn(
    client: HarnessClient,
    convo: Conversation,
    *,
    max_tokens: int,
    temperature: float,
    show_thinking: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Run an agent (tool-using) turn. Returns the FINAL (content, reasoning, stats)."""
    agent = Agent(client, registry=default_registry, limits=AgentLimits(max_turns=8))
    messages = convo.messages()  # snapshot list to mutate
    started_reasoning = False
    started_content = False
    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    t_start = time.perf_counter()
    t_first_token: float | None = None
    finish_reason = ""
    total_tokens = 0
    turns_observed = 0

    try:
        async for ev in agent.run(
            messages, max_tokens=max_tokens, temperature=temperature
        ):
            if ev.kind == "llm_start":
                if ev.data["turn"] > 1:
                    console.print()
                    console.print(Rule(title=f"turn {ev.data['turn']}", style=C_SYS))
                turns_observed = ev.data["turn"]
                started_reasoning = False
                started_content = False

            elif ev.kind == "reasoning":
                if show_thinking:
                    if not started_reasoning:
                        console.print(Rule(title="reasoning", style="dim"))
                        started_reasoning = True
                    console.print(Text(ev.text, style=C_REASONING), end="")
                reasoning_buf.append(ev.text)
                if t_first_token is None:
                    t_first_token = time.perf_counter()

            elif ev.kind == "content":
                if started_reasoning:
                    console.print()
                    console.print(Rule(title="answer", style=C_OK))
                    started_reasoning = False
                started_content = True
                console.print(Text(ev.text, style=C_ASSISTANT), end="")
                content_buf.append(ev.text)
                if t_first_token is None:
                    t_first_token = time.perf_counter()

            elif ev.kind == "tool_call":
                if started_content or started_reasoning:
                    console.print()
                args_repr = json.dumps(ev.data.get("arguments", {}), ensure_ascii=False)
                if len(args_repr) > 200:
                    args_repr = args_repr[:200] + "..."
                console.print(
                    f"[{C_TOOL}]→ {ev.data['name']}({args_repr})[/]"
                )

            elif ev.kind == "tool_result":
                output = ev.data.get("output", "")
                preview = output[:400] + ("..." if len(output) > 400 else "")
                style = C_ERR if ev.data.get("is_error") else C_SYS
                console.print(
                    Panel(
                        preview,
                        title=f"← {ev.data['name']}{'  (error)' if ev.data.get('is_error') else ''}",
                        border_style=style,
                        padding=(0, 1),
                    )
                )
                # reset state since model will start a new turn after
                started_content = False
                started_reasoning = False

            elif ev.kind == "llm_end":
                finish_reason = ev.data.get("finish_reason", "")

            elif ev.kind == "done":
                finish_reason = "agent:" + ev.data.get("reason", "?")
                total_tokens = ev.data.get("total_tokens", 0)

            elif ev.kind == "error":
                console.print()
                console.print(Panel(ev.text, title="error", border_style=C_ERR))
    except KeyboardInterrupt:
        console.print()
        console.print(f"[{C_SYS}](agent cancelled)[/]")
        finish_reason = "cancelled"

    # Replay messages back into convo (the agent appended to a snapshot)
    # We discard the snapshot and just record the LAST assistant content as
    # the turn for transcript purposes; full tool messages live only in the
    # snapshot and get GC'd. Cleaner UX, no schema mismatch on /save.
    elapsed = time.perf_counter() - t_start
    ttft = (t_first_token - t_start) if t_first_token else 0.0
    stats = {
        "completion_tokens": total_tokens,  # rough — agent sums total
        "elapsed_s": round(elapsed, 3),
        "ttft_s": round(ttft, 3),
        "turns": turns_observed,
        "finish_reason": finish_reason,
    }
    console.print()
    console.print(
        f"[{C_SYS}]"
        f"agent: turns={turns_observed}  "
        f"total_tokens={total_tokens}  "
        f"ttft={stats['ttft_s']}s  "
        f"elapsed={stats['elapsed_s']}s  "
        f"finish={finish_reason}"
        f"[/]"
    )
    return "".join(content_buf), "".join(reasoning_buf), stats


# --- slash command handler ----------------------------------------------------


class State:
    def __init__(self, cfg: Config, ep: Endpoint, sandbox: FsSandbox) -> None:
        self.cfg = cfg
        self.ep = ep
        self.client = HarnessClient(ep)
        self.convo = Conversation()
        self.show_thinking = True
        self.max_tokens = ep.default_max_tokens
        self.temperature = ep.default_temperature
        self.sandbox = sandbox
        self.agent_mode = False  # toggle with /agent on
        self.mcp = MCPManager(default_registry)
        self.mcp_configs: dict[str, Any] = {}  # populated in chat_loop()

    async def switch_endpoint(self, name: str) -> None:
        try:
            new_ep = self.cfg.get(name)
        except KeyError as e:
            console.print(f"[{C_ERR}]{e}[/]")
            return
        await self.client.aclose()
        self.ep = new_ep
        self.client = HarnessClient(new_ep)
        self.max_tokens = new_ep.default_max_tokens
        self.temperature = new_ep.default_temperature
        console.print(
            f"[{C_OK}]switched to[/] [{C_ENDPOINT}]{new_ep.name}[/] "
            f"({new_ep.model}, mode={new_ep.mode})"
        )


async def handle_slash(state: State, line: str) -> bool:
    """Return True if we should keep the loop, False to exit."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        return False

    if cmd == "/help":
        console.print(HELP)

    elif cmd == "/endpoints":
        for name, ep in state.cfg.endpoints.items():
            marker = "*" if name == state.ep.name else " "
            console.print(
                f" {marker} [{C_ENDPOINT}]{name:<22}[/] "
                f"[white]{ep.model}[/]   "
                f"[{C_SYS}]{ep.description}[/]"
            )

    elif cmd == "/use":
        if not arg:
            console.print(f"[{C_ERR}]usage: /use <endpoint-name>[/]")
        else:
            await state.switch_endpoint(arg.strip())

    elif cmd == "/mode":
        if arg in ("thinking", "nonthinking", "coding"):
            # Mutate a NEW Endpoint with the desired mode (Endpoint is frozen)
            ep = state.ep
            new_ep = Endpoint(
                name=ep.name, base_url=ep.base_url, model=ep.model, api_key=ep.api_key,
                description=ep.description, default_max_tokens=ep.default_max_tokens,
                default_temperature=ep.default_temperature, mode=arg,
            )
            await state.client.aclose()
            state.ep = new_ep
            state.client = HarnessClient(new_ep)
            console.print(f"[{C_OK}]mode = {arg}[/]")
        else:
            console.print(f"[{C_ERR}]mode must be thinking | nonthinking | coding[/]")

    elif cmd == "/max":
        try:
            state.max_tokens = int(arg)
            console.print(f"[{C_OK}]max_tokens = {state.max_tokens}[/]")
        except ValueError:
            console.print(f"[{C_ERR}]usage: /max <int>[/]")

    elif cmd == "/temp":
        try:
            state.temperature = float(arg)
            console.print(f"[{C_OK}]temperature = {state.temperature}[/]")
        except ValueError:
            console.print(f"[{C_ERR}]usage: /temp <float>[/]")

    elif cmd == "/system":
        if not arg:
            state.convo.system = None
            console.print(f"[{C_OK}]system prompt cleared[/]")
        else:
            state.convo.system = arg
            console.print(f"[{C_OK}]system prompt set ({len(arg)} chars)[/]")

    elif cmd == "/think":
        if arg in ("on", "off"):
            state.show_thinking = arg == "on"
            console.print(f"[{C_OK}]thinking-visible = {state.show_thinking}[/]")
        else:
            console.print(f"[{C_ERR}]usage: /think on|off[/]")

    elif cmd == "/tools":
        sub = arg.split(maxsplit=1) if arg else []
        if not sub:
            for name in default_registry.names():
                spec = default_registry.get(name)
                state_str = (
                    f"[{C_OK}]on[/] " if spec.enabled else f"[{C_ERR}]off[/]"
                )
                cf = " (confirm)" if spec.requires_confirmation else ""
                console.print(f" {state_str} [{C_TOOL}]{name:<22}[/]{cf}  [{C_SYS}]{spec.description[:80]}[/]")
        elif len(sub) == 2 and sub[0] in ("on", "off"):
            tname = sub[1].strip()
            try:
                default_registry.set_enabled(tname, sub[0] == "on")
                console.print(f"[{C_OK}]{tname} = {sub[0]}[/]")
            except KeyError as e:
                console.print(f"[{C_ERR}]{e}[/]")
        else:
            console.print(f"[{C_ERR}]usage: /tools  OR  /tools on|off <name>[/]")

    elif cmd == "/agent":
        if arg in ("on", "off"):
            state.agent_mode = arg == "on"
            console.print(f"[{C_OK}]agent-mode = {state.agent_mode}[/]")
        else:
            console.print(f"[{C_ERR}]usage: /agent on|off[/]")

    elif cmd == "/mcp":
        sub = arg.split(maxsplit=1) if arg else []
        if not sub:
            # list configured + running
            running = set(state.mcp.server_names())
            if not state.mcp_configs:
                console.print(f"[{C_SYS}]no MCP servers configured (configs/mcp_servers.toml)[/]")
            for name, mc in state.mcp_configs.items():
                if name in running:
                    tools = state.mcp.tools_for(name)
                    console.print(
                        f" [{C_OK}]up[/]   [{C_TOOL}]{name:<14}[/]"
                        f"  [{C_SYS}]{len(tools)} tool(s): {', '.join(tools) or '-'}[/]"
                    )
                else:
                    cmdstr = " ".join(mc.command)
                    console.print(
                        f" [{C_ERR}]down[/] [{C_TOOL}]{name:<14}[/]"
                        f"  [{C_SYS}]{cmdstr[:80]}[/]"
                    )
        elif len(sub) == 2 and sub[0] == "start":
            name = sub[1].strip()
            mc = state.mcp_configs.get(name)
            if mc is None:
                console.print(f"[{C_ERR}]no such MCP server in config: {name}[/]")
            else:
                try:
                    await state.mcp.start(mc)
                    tools = state.mcp.tools_for(name)
                    console.print(
                        f"[{C_OK}]mcp '{name}' up[/]  ({len(tools)} tools: {', '.join(tools)})"
                    )
                except Exception as e:  # noqa: BLE001
                    console.print(f"[{C_ERR}]start failed: {e}[/]")
        elif len(sub) == 2 and sub[0] == "stop":
            name = sub[1].strip()
            try:
                await state.mcp.stop(name)
                console.print(f"[{C_OK}]mcp '{name}' stopped[/]")
            except Exception as e:  # noqa: BLE001
                console.print(f"[{C_ERR}]stop failed: {e}[/]")
        else:
            console.print(f"[{C_ERR}]usage: /mcp  OR  /mcp start|stop <name>[/]")

    elif cmd == "/sandbox":
        console.print(f"[{C_OK}]fs sandbox root: {state.sandbox.root}[/]")

    elif cmd == "/clear":
        state.convo.turns.clear()
        console.print(f"[{C_OK}]history cleared[/]")

    elif cmd == "/save":
        path = Path(arg or f"transcript-{datetime.now():%Y%m%d-%H%M%S}.json").expanduser()
        state.convo.save(path)
        console.print(f"[{C_OK}]saved {len(state.convo.turns)} turns to {path}[/]")

    elif cmd == "/info":
        console.print(
            Panel(
                (
                    f"endpoint   [{C_ENDPOINT}]{state.ep.name}[/]\n"
                    f"model      {state.ep.model}\n"
                    f"url        {state.ep.base_url}\n"
                    f"mode       {state.ep.mode}\n"
                    f"max_tokens {state.max_tokens}\n"
                    f"temp       {state.temperature}\n"
                    f"thinking   {state.show_thinking}\n"
                    f"turns      {len(state.convo.turns)}\n"
                    f"last       {state.convo.last_stats or '<none>'}"
                ),
                title="state",
                border_style=C_SYS,
            )
        )

    elif cmd == "/retry":
        if not state.convo.turns or state.convo.turns[-1]["role"] != "assistant":
            console.print(f"[{C_ERR}]nothing to retry[/]")
        else:
            state.convo.pop_last_assistant()
            console.print(f"[{C_SYS}](retrying last turn...)[/]")
            content, reasoning, stats = await stream_one_turn(
                state.client, state.convo,
                max_tokens=state.max_tokens, temperature=state.temperature,
                show_thinking=state.show_thinking,
            )
            state.convo.add_assistant(content, reasoning)
            state.convo.last_stats = stats

    else:
        console.print(f"[{C_ERR}]unknown command: {cmd}  (try /help)[/]")

    return True


# --- main ---------------------------------------------------------------------


async def chat_loop(args: argparse.Namespace) -> int:
    cfg = load_config()
    if w := os.environ.get("HARNESS_CONFIG_WARNING"):
        console.print(f"[{C_ERR}]config warning: {w}[/]")
    try:
        ep = cfg.get(args.endpoint) if args.endpoint else cfg.get()
    except KeyError as e:
        console.print(f"[{C_ERR}]{e}[/]")
        return 1

    # Register builtin tools on the default registry. Idempotent-ish: failure
    # in cerebro just disables those tools.
    sandbox = register_fs(default_registry, sandbox=FsSandbox(args.sandbox or None))
    register_calc(default_registry)
    register_cerebro(default_registry)

    state = State(cfg, ep, sandbox)
    if args.system:
        state.convo.system = args.system
    if args.mode:
        await handle_slash(state, f"/mode {args.mode}")
    if args.agent:
        state.agent_mode = True

    # Load MCP server configs and auto_start any flagged servers.
    state.mcp_configs = load_mcp_config()
    for name, mc in state.mcp_configs.items():
        if mc.auto_start:
            try:
                await state.mcp.start(mc)
                tools = state.mcp.tools_for(name)
                console.print(
                    f"[{C_OK}]mcp '{name}' up[/]  ({len(tools)} tools: {', '.join(tools) or '-'})"
                )
            except Exception as e:  # noqa: BLE001
                console.print(f"[{C_ERR}]mcp '{name}' auto_start failed: {e}[/]")

    banner(cfg, state.ep, show_thinking=state.show_thinking)
    if state.agent_mode:
        console.print(f"[{C_OK}]agent-mode ON[/]  — tools: " + ", ".join(default_registry.names()))
    console.print(f"[{C_SYS}]fs sandbox: {sandbox.root}[/]")

    while True:
        try:
            line = Prompt.ask(f"[{C_USER}]you[/]")
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[{C_SYS}]bye[/]")
            break

        if not line.strip():
            continue
        if line.startswith("/"):
            keep_going = await handle_slash(state, line)
            if not keep_going:
                break
            continue

        state.convo.add_user(line)
        console.print()
        if state.agent_mode:
            content, reasoning, stats = await stream_agent_turn(
                state.client, state.convo,
                max_tokens=state.max_tokens, temperature=state.temperature,
                show_thinking=state.show_thinking,
            )
        else:
            content, reasoning, stats = await stream_one_turn(
                state.client, state.convo,
                max_tokens=state.max_tokens, temperature=state.temperature,
                show_thinking=state.show_thinking,
            )
        state.convo.add_assistant(content, reasoning)
        state.convo.last_stats = stats
        console.print()

    await state.mcp.stop_all()
    await state.client.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness", description="qwen36-harness CLI")
    sub = p.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive REPL chat with an endpoint")
    chat.add_argument("--endpoint", "-e", help="endpoint name (default from config)")
    chat.add_argument("--mode", choices=["thinking", "nonthinking", "coding"])
    chat.add_argument("--system", "-s", help="system prompt")
    chat.add_argument("--agent", action="store_true", help="enable tool-loop agent mode")
    chat.add_argument("--sandbox", help="fs sandbox root (default ~/qwen36-sandbox)")

    args = p.parse_args(argv)
    if args.command == "chat":
        try:
            return asyncio.run(chat_loop(args))
        except KeyboardInterrupt:
            return 130
    return 2


if __name__ == "__main__":
    sys.exit(main())
