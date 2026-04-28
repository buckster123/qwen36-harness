"""Tests for harness.web — FastAPI control endpoints + SSE shape.

The chat SSE test mocks the agent loop so we don't need a live endpoint.
The live web smoke (real endpoint, real Qwen3.6) lives in test_smoke_web.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from harness.config import Config, Endpoint
from harness.tools import Registry, ToolSpec, default_registry
from harness.tools.filesystem import FsSandbox
from harness.web import WebSession, create_app


@pytest.fixture
def session(tmp_path) -> WebSession:
    # Clean slate: nuke any leftover registry state from earlier tests
    default_registry._tools.clear()  # noqa: SLF001
    default_registry.register(
        ToolSpec(
            name="testing.echo",
            description="echo arg",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            fn=lambda x="": x,
        )
    )
    cfg = Config(
        endpoints={
            "alpha": Endpoint(
                name="alpha",
                base_url="http://alpha.example/v1",
                model="alpha-model",
                mode="nonthinking",
                default_max_tokens=512,
                default_temperature=0.0,
            ),
            "beta": Endpoint(
                name="beta",
                base_url="http://beta.example/v1",
                model="beta-model",
                mode="thinking",
                default_max_tokens=1024,
                default_temperature=0.7,
            ),
        },
        default_endpoint="alpha",
    )
    return WebSession(
        cfg=cfg,
        ep=cfg.get(),
        sandbox=FsSandbox(root=tmp_path),
        max_tokens=512,
        temperature=0.0,
    )


@pytest.fixture
def client(session: WebSession) -> TestClient:
    app = create_app(session=session, register_builtins=False)
    return TestClient(app)


# --- state ---------------------------------------------------------------------


def test_state_returns_endpoints_tools_and_settings(client: TestClient) -> None:
    r = client.get("/api/state")
    assert r.status_code == 200
    s = r.json()
    assert s["current_endpoint"] == "alpha"
    assert s["mode"] == "nonthinking"
    assert s["agent_mode"] is True  # web defaults agent ON
    assert {e["name"] for e in s["endpoints"]} == {"alpha", "beta"}
    assert any(t["name"] == "testing.echo" for t in s["tools"])
    assert s["mcp"] == []  # no servers configured in this test session
    assert s["turn_count"] == 0


def test_use_switches_endpoint(client: TestClient, session: WebSession) -> None:
    r = client.post("/api/use", json={"endpoint": "beta"})
    assert r.status_code == 200
    assert session.ep.name == "beta"
    assert session.max_tokens == 1024


def test_use_unknown_endpoint_404(client: TestClient) -> None:
    r = client.post("/api/use", json={"endpoint": "nonexistent"})
    assert r.status_code == 404


def test_settings_update(client: TestClient, session: WebSession) -> None:
    r = client.post("/api/settings", json={
        "mode": "thinking", "max_tokens": 100, "temperature": 0.5,
        "agent_mode": False, "show_thinking": False,
    })
    assert r.status_code == 200
    assert session.ep.mode == "thinking"
    assert session.max_tokens == 100
    assert session.temperature == 0.5
    assert session.agent_mode is False
    assert session.show_thinking is False


def test_settings_rejects_bad_mode(client: TestClient) -> None:
    r = client.post("/api/settings", json={"mode": "bogus"})
    assert r.status_code == 400


def test_clear_drops_turns(client: TestClient, session: WebSession) -> None:
    session.turns.append({"role": "user", "content": "hi"})
    r = client.post("/api/clear")
    assert r.status_code == 200
    assert session.turns == []


def test_system_set_and_clear(client: TestClient, session: WebSession) -> None:
    r = client.post("/api/system", json={"text": "you are a pirate"})
    assert r.status_code == 200
    assert session.system == "you are a pirate"
    r = client.post("/api/system", json={"text": None})
    assert session.system is None


# --- tool toggle ---------------------------------------------------------------


def test_tool_toggle(client: TestClient) -> None:
    r = client.post("/api/tools/testing.echo/toggle", json={"enabled": False})
    assert r.status_code == 200
    spec = default_registry.get("testing.echo")
    assert spec.enabled is False
    r = client.post("/api/tools/testing.echo/toggle", json={"enabled": True})
    assert spec.enabled is True


def test_tool_toggle_unknown_404(client: TestClient) -> None:
    r = client.post("/api/tools/nope/toggle", json={"enabled": False})
    assert r.status_code == 404


# --- chat SSE (with mocked agent) ---------------------------------------------


@pytest.mark.asyncio
async def test_chat_streams_agent_events(session: WebSession) -> None:
    """Mock the agent so we don't need a live endpoint, verify SSE shape."""

    async def fake_run(self, messages, **kwargs):
        # Simulate the events an agent would emit
        from harness.agent import AgentEvent

        yield AgentEvent(kind="llm_start", data={"turn": 1})
        yield AgentEvent(kind="content", text="Hello ")
        yield AgentEvent(kind="content", text="world.")
        # Agent appends the assistant message to mutated messages
        messages.append({"role": "assistant", "content": "Hello world."})
        yield AgentEvent(kind="done", data={"stats": {"completion_tokens": 2}})

    app = create_app(session=session, register_builtins=False)
    client = TestClient(app)
    with patch("harness.web.Agent.run", new=fake_run):
        with client.stream("POST", "/api/chat", json={"text": "hi"}) as r:
            assert r.status_code == 200
            body = b""
            for chunk in r.iter_bytes():
                body += chunk
    text = body.decode()
    assert "event: user" in text
    assert "event: llm_start" in text
    assert "event: content" in text
    assert '"text": "Hello "' in text
    assert "event: done" in text
    assert "event: end" in text
    # session.turns should contain user + assistant after the run
    assert session.turns[0]["role"] == "user"
    assert session.turns[-1]["role"] == "assistant"
    assert "Hello world." in session.turns[-1]["content"]


# --- root --------------------------------------------------------------------


def test_root_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "qwen36-harness" in r.text
    assert "<html" in r.text.lower()
