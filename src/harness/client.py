"""OpenAI-compatible streaming chat client built on httpx.

Designed against llama.cpp's `llama-server` (Qwen3.6 specifically, but the
shape is plain OpenAI-compat). Models like Qwen3.6 emit dual-channel deltas:

  - ``delta.reasoning_content`` — internal thinking (chain-of-thought)
  - ``delta.content``           — user-visible answer
  - ``delta.tool_calls``        — function-call invocations

This module exposes a stream of typed events so any UI (CLI, web) can
render thinking, answer, and tool calls independently.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .config import Endpoint

# --- event types --------------------------------------------------------------

EventKind = Literal[
    "start",          # before first byte; emitted with the request id once the server sends it
    "reasoning",      # delta of internal thinking text
    "content",        # delta of user-visible text
    "tool_call",      # tool call (id + name + accumulated args, fired only on completion)
    "tool_call_delta",  # streaming tool-call argument fragment (rare with llama.cpp)
    "usage",          # token-usage stats (often arrives in the last chunk)
    "done",           # final event; finish_reason and any totals
    "error",          # transport or upstream error
]


@dataclass(slots=True)
class StreamEvent:
    kind: EventKind
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResult:
    """Aggregated result of a non-streaming completion (or a fully drained stream)."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


# --- helpers ------------------------------------------------------------------

ToolCallList = list[dict[str, Any]]


def _accumulate_tool_call_delta(buffer: ToolCallList, delta_calls: list[dict[str, Any]]) -> None:
    """Apply OpenAI-style streaming tool_call deltas to ``buffer`` in-place.

    Each delta carries an ``index`` plus partial fields. We grow the matching
    slot in ``buffer`` so each tool call accumulates id/name/arguments across
    chunks. llama.cpp tends to send the whole call in one chunk, but we handle
    the partial case for robustness.
    """
    for piece in delta_calls:
        idx = piece.get("index", 0)
        while len(buffer) <= idx:
            buffer.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        slot = buffer[idx]
        if piece.get("id"):
            slot["id"] = piece["id"]
        if piece.get("type"):
            slot["type"] = piece["type"]
        fn = piece.get("function") or {}
        if "name" in fn and fn["name"]:
            slot["function"]["name"] += fn["name"]
        if "arguments" in fn and fn["arguments"] is not None:
            slot["function"]["arguments"] += fn["arguments"]


def _parse_tool_args(call: dict[str, Any]) -> dict[str, Any]:
    """Best-effort JSON-parse of a tool call's accumulated argument string.

    Returns an empty dict on parse failure rather than raising — the caller
    can decide whether to surface that to the model as a tool error.
    """
    raw = call.get("function", {}).get("arguments", "") or ""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw, "_error": "could not parse arguments as JSON"}


# --- client -------------------------------------------------------------------


class HarnessClient:
    """Async OpenAI-compatible chat client. One instance per endpoint.

    Holds a long-lived httpx.AsyncClient — call ``.aclose()`` (or use
    ``async with``) to release sockets cleanly.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        *,
        timeout: float = 600.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint = endpoint
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={"authorization": f"Bearer {endpoint.api_key}"},
        )

    async def __aenter__(self) -> HarnessClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- non-streaming ---------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Single non-streaming completion. Returns aggregated ChatResult."""
        body = self._build_body(messages, max_tokens, temperature, tools, tool_choice, extra)
        body["stream"] = False
        resp = await self._post(body)
        return self._parse_completion(resp)

    # -- streaming -------------------------------------------------------------

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream completion as typed events. Aggregates tool calls across chunks."""
        body = self._build_body(messages, max_tokens, temperature, tools, tool_choice, extra)
        body["stream"] = True
        body.setdefault("stream_options", {"include_usage": True})

        return self._stream_iterator(body)

    async def _stream_iterator(self, body: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        tool_calls: ToolCallList = []
        finish_reason = ""
        emitted_start = False

        try:
            async with self._client.stream(
                "POST", self.endpoint.chat_url(), json=body
            ) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    yield StreamEvent(
                        kind="error",
                        text=f"HTTP {resp.status_code}",
                        data={"body": text.decode("utf-8", "replace")[:500]},
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if not emitted_start:
                        yield StreamEvent(kind="start", data={"id": chunk.get("id", "")})
                        emitted_start = True

                    # Token-usage chunk (sent at the end when include_usage)
                    if chunk.get("usage"):
                        yield StreamEvent(kind="usage", data=chunk["usage"])

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    if rc := delta.get("reasoning_content"):
                        yield StreamEvent(kind="reasoning", text=rc)
                    if c := delta.get("content"):
                        yield StreamEvent(kind="content", text=c)
                    if tc := delta.get("tool_calls"):
                        _accumulate_tool_call_delta(tool_calls, tc)
                        # surface a tool_call_delta for any UI that wants to render live
                        yield StreamEvent(kind="tool_call_delta", data={"calls": tc})

                    if fr := choice.get("finish_reason"):
                        finish_reason = fr

                # Emit fully-formed tool_calls (one event each) before done
                for call in tool_calls:
                    yield StreamEvent(
                        kind="tool_call",
                        data={
                            "id": call.get("id", ""),
                            "name": call.get("function", {}).get("name", ""),
                            "arguments": _parse_tool_args(call),
                            "raw_arguments": call.get("function", {}).get("arguments", ""),
                        },
                    )

                yield StreamEvent(
                    kind="done",
                    data={"finish_reason": finish_reason, "tool_call_count": len(tool_calls)},
                )

        except httpx.HTTPError as exc:
            yield StreamEvent(kind="error", text=str(exc), data={"type": type(exc).__name__})

    # -- internals -------------------------------------------------------------

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "max_tokens": max_tokens or self.endpoint.default_max_tokens,
            "temperature": temperature
            if temperature is not None
            else self.endpoint.default_temperature,
        }
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if self.endpoint.mode == "nonthinking":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            body.update(extra)
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(self.endpoint.chat_url(), json=body)
                if resp.status_code >= 500 and attempt < self._max_retries - 1:
                    last_exc = httpx.HTTPStatusError(
                        f"server {resp.status_code}", request=resp.request, response=resp
                    )
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                continue
        raise last_exc or RuntimeError("post failed without an exception (impossible)")

    @staticmethod
    def _parse_completion(resp: dict[str, Any]) -> ChatResult:
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        raw_calls = msg.get("tool_calls") or []
        return ChatResult(
            content=msg.get("content") or "",
            reasoning_content=msg.get("reasoning_content") or "",
            tool_calls=[
                {
                    "id": c.get("id", ""),
                    "name": c.get("function", {}).get("name", ""),
                    "arguments": _parse_tool_args(c),
                    "raw_arguments": c.get("function", {}).get("arguments", ""),
                }
                for c in raw_calls
            ],
            finish_reason=choice.get("finish_reason") or "",
            usage=resp.get("usage") or {},
            raw=resp,
        )


__all__ = [
    "ChatResult",
    "HarnessClient",
    "StreamEvent",
]


def _drain_messages(events: Iterable[StreamEvent]) -> ChatResult:
    """Synchronous helper used by tests to fold a list of events into a ChatResult."""
    out = ChatResult()
    for ev in events:
        if ev.kind == "reasoning":
            out.reasoning_content += ev.text
        elif ev.kind == "content":
            out.content += ev.text
        elif ev.kind == "tool_call":
            out.tool_calls.append(ev.data)
        elif ev.kind == "usage":
            out.usage = ev.data
        elif ev.kind == "done":
            out.finish_reason = ev.data.get("finish_reason", "")
    return out
