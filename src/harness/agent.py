"""Agent — tool-loop driver.

Wraps a HarnessClient and a tool Registry. ``run`` streams the model's
output, dispatches tool calls when they arrive, appends their results as
``role: tool`` messages, and continues looping until the model emits a
final answer (no tool_calls in finish_reason) or limits are hit.

Stream contract — yields tagged events for any UI to render:

  ``llm_start``        — model is about to think/answer
  ``reasoning``        — chain-of-thought delta (text)
  ``content``          — visible answer delta (text)
  ``tool_call``        — model invoked a tool (data: id, name, arguments)
  ``tool_result``      — dispatch finished (data: name, output, is_error)
  ``llm_end``          — model finished this turn (data: finish_reason)
  ``done``             — agent loop terminated (data: turns, total_tokens)
  ``error``            — fatal error (text)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .client import HarnessClient, StreamEvent
from .tools import Registry, default_registry


@dataclass(slots=True)
class AgentEvent:
    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentLimits:
    max_turns: int = 8
    max_total_tokens: int = 100_000


class Agent:
    def __init__(
        self,
        client: HarnessClient,
        *,
        registry: Registry | None = None,
        tools_enabled: list[str] | None = None,
        limits: AgentLimits | None = None,
    ) -> None:
        self.client = client
        self.registry = registry or default_registry
        self.tools_enabled = tools_enabled  # None = all enabled tools
        self.limits = limits or AgentLimits()

    async def run(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream an agent loop. Yields ``AgentEvent``s; mutates ``messages`` in-place."""
        tools_schema = self.registry.export_schema(only=self.tools_enabled)
        total_tokens = 0
        turn = 0

        while turn < self.limits.max_turns:
            turn += 1
            yield AgentEvent(kind="llm_start", data={"turn": turn})

            # Stream the LLM step
            tool_calls_this_turn: list[dict[str, Any]] = []
            finish_reason = ""
            content_buf: list[str] = []
            reasoning_buf: list[str] = []

            stream = await self.client.stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools_schema or None,
                tool_choice="auto" if tools_schema else None,
            )
            async for ev in stream:
                if ev.kind == "reasoning":
                    reasoning_buf.append(ev.text)
                    yield AgentEvent(kind="reasoning", text=ev.text)
                elif ev.kind == "content":
                    content_buf.append(ev.text)
                    yield AgentEvent(kind="content", text=ev.text)
                elif ev.kind == "tool_call":
                    tool_calls_this_turn.append(ev.data)
                    yield AgentEvent(kind="tool_call", data=ev.data)
                elif ev.kind == "usage":
                    total_tokens += int(ev.data.get("total_tokens") or 0)
                elif ev.kind == "done":
                    finish_reason = ev.data.get("finish_reason", "")
                elif ev.kind == "error":
                    yield AgentEvent(kind="error", text=ev.text, data=ev.data)
                    return

            yield AgentEvent(
                kind="llm_end",
                data={"finish_reason": finish_reason, "turn": turn},
            )

            # Append the assistant message we just received to the conversation.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content_buf),
            }
            if tool_calls_this_turn:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc.get("raw_arguments")
                            or json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in tool_calls_this_turn
                ]
            messages.append(assistant_msg)

            # If no tool calls, we're done.
            if not tool_calls_this_turn or finish_reason != "tool_calls":
                yield AgentEvent(
                    kind="done",
                    data={"turns": turn, "total_tokens": total_tokens, "reason": "final"},
                )
                return

            # Dispatch every tool call, append results.
            for call in tool_calls_this_turn:
                result = await self.registry.dispatch(call["name"], call.get("arguments", {}))
                yield AgentEvent(
                    kind="tool_result",
                    data={
                        "id": call["id"],
                        "name": result.name,
                        "output": result.output,
                        "is_error": result.is_error,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result.output,
                    }
                )

            if total_tokens >= self.limits.max_total_tokens:
                yield AgentEvent(
                    kind="done",
                    data={
                        "turns": turn,
                        "total_tokens": total_tokens,
                        "reason": "token_limit",
                    },
                )
                return

        yield AgentEvent(
            kind="done",
            data={"turns": turn, "total_tokens": total_tokens, "reason": "turn_limit"},
        )


__all__ = ["Agent", "AgentEvent", "AgentLimits"]
