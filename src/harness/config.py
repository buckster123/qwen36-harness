"""Endpoint and runtime configuration loading.

Endpoints live in ``configs/endpoints.toml`` (committed) plus an optional
``configs/endpoints.local.toml`` (gitignored, overrides). We deliberately do
NOT use environment variables for endpoint URLs — config is the source of
truth so the same setup is reproducible across sessions.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A single OpenAI-compatible endpoint."""

    name: str
    base_url: str
    model: str
    api_key: str = "sk-anything"
    description: str = ""
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    mode: str = "thinking"  # thinking | nonthinking | coding

    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"


@dataclass(slots=True)
class Config:
    endpoints: dict[str, Endpoint] = field(default_factory=dict)
    default_endpoint: str = ""

    def get(self, name: str | None = None) -> Endpoint:
        key = name or self.default_endpoint
        if key not in self.endpoints:
            available = ", ".join(sorted(self.endpoints)) or "<none>"
            raise KeyError(f"endpoint '{key}' not configured (have: {available})")
        return self.endpoints[key]


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(config_dir: Path | None = None) -> Config:
    """Load ``endpoints.toml`` and merge optional ``endpoints.local.toml`` on top."""
    cdir = config_dir or CONFIG_DIR
    base = _load_toml(cdir / "endpoints.toml")
    local = _load_toml(cdir / "endpoints.local.toml")

    cfg = Config()
    cfg.default_endpoint = (
        local.get("default", {}).get("endpoint")
        or base.get("default", {}).get("endpoint", "")
    )

    raw_endpoints: dict[str, dict] = dict(base.get("endpoints", {}))
    for name, body in (local.get("endpoints") or {}).items():
        merged = {**raw_endpoints.get(name, {}), **body}
        raw_endpoints[name] = merged

    for name, body in raw_endpoints.items():
        cfg.endpoints[name] = Endpoint(
            name=name,
            base_url=body["base_url"],
            model=body["model"],
            api_key=body.get("api_key", "sk-anything"),
            description=body.get("description", ""),
            default_max_tokens=int(body.get("default_max_tokens", 1024)),
            default_temperature=float(body.get("default_temperature", 0.7)),
            mode=body.get("mode", "thinking"),
        )

    if cfg.default_endpoint and cfg.default_endpoint not in cfg.endpoints:
        # Fall back to the first endpoint defined; surface a soft warning via env.
        first = next(iter(cfg.endpoints), "")
        os.environ.setdefault(
            "HARNESS_CONFIG_WARNING",
            f"default endpoint '{cfg.default_endpoint}' not found; using '{first}'",
        )
        cfg.default_endpoint = first

    return cfg
