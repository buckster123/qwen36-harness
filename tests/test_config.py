"""Tests for harness.config endpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import Config, Endpoint, load_config


def write(tmp_path: Path, name: str, content: str) -> None:
    (tmp_path / name).write_text(content)


def test_loads_default_endpoint(tmp_path: Path) -> None:
    write(
        tmp_path,
        "endpoints.toml",
        """
[default]
endpoint = "vast"

[endpoints.vast]
base_url = "http://1.2.3.4:9000/v1"
model = "qwen.gguf"
""",
    )
    cfg = load_config(tmp_path)
    assert cfg.default_endpoint == "vast"
    ep = cfg.get()
    assert isinstance(ep, Endpoint)
    assert ep.chat_url() == "http://1.2.3.4:9000/v1/chat/completions"
    assert ep.api_key == "sk-anything"
    assert ep.default_max_tokens == 1024


def test_local_overrides_base(tmp_path: Path) -> None:
    write(
        tmp_path,
        "endpoints.toml",
        """
[default]
endpoint = "vast"

[endpoints.vast]
base_url = "http://old:9000/v1"
model = "old.gguf"
default_max_tokens = 1024
""",
    )
    write(
        tmp_path,
        "endpoints.local.toml",
        """
[endpoints.vast]
base_url = "http://new:9000/v1"
default_max_tokens = 4096
""",
    )
    cfg = load_config(tmp_path)
    ep = cfg.get("vast")
    assert ep.base_url == "http://new:9000/v1"
    assert ep.model == "old.gguf"  # not overridden
    assert ep.default_max_tokens == 4096


def test_unknown_endpoint_raises(tmp_path: Path) -> None:
    write(
        tmp_path,
        "endpoints.toml",
        """
[default]
endpoint = "vast"

[endpoints.vast]
base_url = "http://x/v1"
model = "m"
""",
    )
    cfg = load_config(tmp_path)
    with pytest.raises(KeyError):
        cfg.get("does-not-exist")


def test_no_default_when_missing(tmp_path: Path) -> None:
    write(tmp_path, "endpoints.toml", "[endpoints.x]\nbase_url='http://y/v1'\nmodel='m'\n")
    cfg = load_config(tmp_path)
    # default_endpoint will be empty string; cfg.get() with no name should fail
    assert cfg.default_endpoint == ""
    with pytest.raises(KeyError):
        cfg.get()


def test_real_repo_config_parses() -> None:
    """Smoke: the actual endpoints.toml in this repo loads cleanly."""
    cfg = load_config()  # uses CONFIG_DIR
    assert cfg.default_endpoint
    ep = cfg.get()
    assert ep.base_url.startswith("http")
    assert ep.model
