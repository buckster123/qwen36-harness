"""Vast.ai GPU instance lifecycle manager for qwen36-harness.

Wraps the scripts at ~/Projects/qwen36-vast/ (vast_up.sh, vast_down.sh,
tools/vast_tunnel.sh) into structured Python APIs that the WebUI and CLI
can call without digging through shell scripts or docs.

Public API
----------
VastManager
    .recipes()                    → list of recipe dicts
    .latest_instance()            → {id, created, config} from .last_instance
    .list_instances(state=None)   → [{id, status, gpu, price/hr, geo, ports}]
    .spinup(recipe, **overrides)  → {success, instance_id, logs} or raises
    .down(instance_id=None)       → {success, message}
    .tunnel(action)               → {success, pid, msg}
    .poll_status(instance_id)     → {status, ssh_host, ssh_port, public_ip, ports, model_loaded, health}

Recipe selection is the "menu" the user sees — each recipe bundles GPU model,
LLM model/quant, context size, KV cache type, parallel slots, and price ceiling
into a single named option.

Usage from WebUI: POST /api/vast/spinup {recipe: "moe-256k", geo: "EU_NORDIC"}
Usage from CLI:    harness vast spinup moe-256k --geo EU
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recipe definitions — each is a "preset" the user picks from the menu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VastRecipe:
    """A pre-configured spin-up recipe."""
    name: str
    display: str            # What shows in the UI menu
    model: str              # dense | moe | moe-256k | moe-beast
    gpu: str                # 5090 | 4090 | 6000pro
    max_tokens_display: int = 0  # Max tokens this config supports (1 slot)

# All available recipes — sorted by GPU tier then complexity
ALL_RECIPES = [
    VastRecipe("dense-4090",     "RTX 4090 | Qwen3.6-27B | Q4_K_XL | 64K ctx",           "dense",      "4090"),
    VastRecipe("dense-5090",     "RTX 5090 | Qwen3.6-27B | Q6_K_XL | 96K ctx",            "dense",      "5090"),
    VastRecipe("moe-4090",       "RTX 4090 | Qwen3.6-35B-A3B | Q4_K_XL | 48K ctx",        "moe",        "4090"),
    VastRecipe("moe-128k",       "RTX 5090 | Qwen3.6-35B-A3B | Q5_K_XL | 128K ctx",       "moe",        "5090"),
    VastRecipe("moe-256k",       "RTX 5090 | Qwen3.6-35B-A3B | Q5_K_XL | 256K q4 KV",     "moe-256k",   "5090"),
    VastRecipe("moe-beast",      "PRO 6000 WS | Qwen3.6-35B-A3B | Q8_K_XL | 6×256K q8 KV", "moe-beast",  "6000pro"),
    VastRecipe("moe-6000-pro",   "PRO 6000 WS | Qwen3.6-35B-A3B | Q6_K_XL | 4×256K q8 KV", "moe",       "6000pro"),
    VastRecipe("dense-6000-pro", "PRO 6000 WS | Qwen3.6-27B | Q8_K_XL | 4×128K q8 KV",     "dense",      "6000pro"),
]


@dataclass
class VastManager:
    """Orchestrates Vast.ai GPU instance lifecycle."""

    # Where the qwen36-vast scripts live (override via env or constructor)
    vast_project: Path = field(default_factory=lambda: _default_vast_dir())
    tunnel_script: Path = field(default_factory=lambda: _tunnel_script_path())
    max_log_lines: int = 500

    # --- discovery --------------------------------------------------------

    def recipes(self) -> list[dict[str, Any]]:
        """Return all available recipes as dicts."""
        return [
            {"name": r.name, "display": r.display}
            for r in ALL_RECIPES
        ]

    def latest_instance(self) -> dict[str, Any] | None:
        """Read .last_instance from the qwen36-vast project dir.

        Returns {id, config} or None if no instance has been spun up yet.
        """
        last_file = self.vast_project / ".last_instance"
        if not last_file.exists():
            return None
        inst_id = last_file.read_text().strip()
        return {"id": inst_id}

    def list_instances(self, state: str | None = None) -> list[dict[str, Any]]:
        """Query Vast.ai for active instances via `vastai show instance`.

        When ``state`` is given (e.g. "running"), only matching instances are returned.
        """
        cmd = ["vastai", "show", "instance"]
        if state:
            cmd.append(state)
        try:
            result = asyncio.get_event_loop().run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            )
            raw = ""
            try:
                raw = result.stdout or ""
            except Exception:
                raw = ""

            if not raw.strip():
                return []

            # Parse JSON array from vastai output
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                # Fallback: parse line-based output
                instances = []
                current: dict[str, Any] = {}
                for line in raw.strip().split("\n"):
                    m = re.match(r"^(\w+)\s*:\s*(.*)", line)
                    if m:
                        key, val = m.group(1).strip(), m.group(2).strip()
                        current[key] = val
                    elif line.startswith("  ") and current.get("id"):
                        # Sub-item — skip for now
                        pass
                if "id" in current:
                    instances.append(current)
                return instances

            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                return []

            out = []
            for inst in data:
                ports_raw = inst.get("ports", {})
                if isinstance(ports_raw, str):
                    try:
                        ports_raw = json.loads(ports_raw)
                    except (json.JSONDecodeError, TypeError):
                        ports_raw = {}

                out.append({
                    "id": str(inst.get("id", "")),
                    "status": inst.get("actual_status", inst.get("intended_status", "unknown")),
                    "gpu": inst.get("gpu_name", inst.get("gpu_ram", "?")),
                    "vram_gb": inst.get("gpu_ram", 0) / 1024 if isinstance(inst.get("gpu_ram"), (int, float)) else "?",
                    "price_hr": inst.get("dph_total", "?"),
                    "geo": inst.get("geolocation", "?"),
                    "ssh_host": inst.get("ssh_host", ""),
                    "ssh_port": inst.get("ports", {}).get("22/tcp", [{}])[0].get("HostPort", "") if isinstance(inst.get("ports"), dict) else "",
                    "ports": {k: v[0].get("HostPort", "?") for k, v in ports_raw.items() if isinstance(v, list)},
                    "rel": inst.get("reliability2", "?"),
                })
            return out

        except Exception as e:  # noqa: BLE001
            log.warning("list_instances failed: %s", e)
            return []

    def poll_status(self, instance_id: str) -> dict[str, Any]:
        """Poll a single instance for detailed status including health check."""
        cmd = ["vastai", "show", "instance", instance_id, "--raw"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            raw = (result.stdout or "").strip()
            if not raw:
                return {"error": "no data from vastai show"}

            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {"raw_output": raw}

            ports_raw = data.get("ports", {})
            if isinstance(ports_raw, str):
                try:
                    ports_raw = json.loads(ports_raw)
                except (json.JSONDecodeError, TypeError):
                    ports_raw = {}

            # Extract port 8000 mapping
            pub_port = "?"
            for k, v in ports_raw.items():
                if "8000" in k and isinstance(v, list) and len(v) > 0:
                    pub_port = str(v[0].get("HostPort", "?"))
                    break

            # Health check against the public port (may fail if HOST=127.0.0.1 inside container)
            health = "unknown"
            model_loaded = "?"
            ip = data.get("public_ipaddr", "")
            try:
                resp = subprocess.run(
                    ["curl", "-s", "--max-time", "3", f"http://{ip}:{pub_port}/health"],
                    capture_output=True, text=True, timeout=5
                )
                if "ok" in (resp.stdout or "").lower():
                    health = "ok"
                    # Check model loaded
                    mr = subprocess.run(
                        ["curl", "-s", "--max-time", "3", f"http://{ip}:{pub_port}/v1/models"],
                        capture_output=True, text=True, timeout=5
                    )
                    try:
                        md = json.loads(mr.stdout or "{}")
                        model_loaded = (md.get("data", [{}])[0] or {}).get("id", "?")
                    except (json.JSONDecodeError, TypeError):
                        pass
                else:
                    health = resp.stdout.strip()[:120]
            except Exception:  # noqa: BLE001
                health = "unreachable"

            return {
                "id": str(data.get("id", "")),
                "status": data.get("actual_status", data.get("intended_status", "unknown")),
                "gpu": data.get("gpu_name", ""),
                "vram_gb": data.get("gpu_ram", 0) / 1024 if isinstance(data.get("gpu_ram"), (int, float)) else "?",
                "price_hr": data.get("dph_total", "?"),
                "geo": data.get("geolocation", ""),
                "ssh_host": data.get("ssh_host", ""),
                "ssh_port": str(data.get("ports", {}).get("22/tcp", [{}])[0].get("HostPort", "")) if isinstance(data.get("ports"), dict) else "",
                "public_ip": ip,
                "pub_port_8000": pub_port,
                "health": health,
                "model_loaded": model_loaded,
            }

        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    # --- spinup -----------------------------------------------------------

    def spinup(self, recipe_name: str, geo: str = "EU_NORDIC", **overrides) -> dict[str, Any]:
        """Spin up a new instance using the given recipe.

        Returns {success, instance_id, logs, error} dict.
        Raises if the script fails unexpectedly (not matched-offer).
        """
        # Resolve recipe name to env vars
        env = {
            "GEO": geo,
            "MODEL": "",
            "GPU": "",
        }

        for r in ALL_RECIPES:
            if r.name == recipe_name:
                env["MODEL"] = r.model
                env["GPU"] = r.gpu
                break
        else:
            return {"success": False, "error": f"unknown recipe '{recipe_name}'"}

        # Apply overrides
        for k, v in overrides.items():
            env[k.upper()] = str(v)

        vast_up = self.vast_project / "vast_up.sh"
        if not vast_up.exists():
            return {"success": False, "error": f"vast_up.sh not found at {vast_up}"}

        # Run the script
        log.info("spinup recipe=%s geo=%s env=%s", recipe_name, geo, env)
        proc = subprocess.run(
            ["bash", str(vast_up)],
            capture_output=True, text=True, timeout=600,  # 10 min max for full spinup
            env={**os.environ, **env},
        )

        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return {
            "success": proc.returncode == 0,
            "output": output[-4096:],  # last 4KB of log
            "instance_id": self._extract_instance_id(output),
            "error": output if proc.returncode != 0 else None,
        }

    def _extract_instance_id(self, output: str) -> str | None:
        """Parse instance ID from vast_up.sh stdout."""
        # Try JSON contract ID first
        m = re.search(r'"new_contract"\s*:\s*"(\d+)"', output)
        if m:
            return m.group(1)
        # Fallback: ".last_instance" line or number pattern after "created"
        m = re.search(r'instance\s+(\d+)\s+created', output, re.I)
        if m:
            return m.group(1)
        return None

    # --- shutdown ---------------------------------------------------------

    def down(self, instance_id: str | None = None) -> dict[str, Any]:
        """Destroy the given instance (or .last_instance)."""
        target = instance_id or self._read_last_instance()
        if not target:
            return {"success": False, "error": "no instance specified and no .last_instance"}

        vast_down = self.vast_project / "vast_down.sh"
        if not vast_down.exists():
            # Fallback: direct vastai destroy
            cmd = ["bash", "-c", f'echo y | vastai destroy instance {target}']
        else:
            cmd = ["bash", str(vast_down)]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                env={**os.environ, "INSTANCE_ID": target},
            )
            # Also remove .last_instance
            last_file = self.vast_project / ".last_instance"
            if last_file.exists():
                last_file.unlink(missing_ok=True)

            return {
                "success": result.returncode == 0,
                "output": (result.stdout or "") + "\n" + (result.stderr or ""),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "vast down timed out after 30s"}

    # --- tunnel -----------------------------------------------------------

    def tunnel(self, action: str = "status") -> dict[str, Any]:
        """Control the SSH tunnel to the active instance.

        Actions: up, status, down, logs (tail last lines of launch.log)
        """
        ts = self.tunnel_script
        if not ts.exists():
            return {"success": False, "error": f"tunnel script not found at {ts}"}

        try:
            result = subprocess.run(
                ["bash", str(ts), action],
                capture_output=True, text=True, timeout=60,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")

            if action == "status":
                pid_match = re.search(r'(?:pid|PID)\s*(\d+)', output)
                return {
                    "success": result.returncode == 0,
                    "running": pid_match is not None,
                    "pid": int(pid_match.group(1)) if pid_match else None,
                    "output": output.strip(),
                }

            if action == "logs":
                # Return last N lines from tunnel logs or launch.log
                return {
                    "success": result.returncode == 0,
                    "lines": output.strip().split("\n")[-self.max_log_lines:],
                    "output": output.strip(),
                }

            return {
                "success": result.returncode == 0,
                "output": output.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "tunnel command timed out"}

    # --- internals --------------------------------------------------------

    def _read_last_instance(self) -> str | None:
        last_file = self.vast_project / ".last_instance"
        if not last_file.exists():
            return None
        return last_file.read_text().strip() or None


def _default_vast_dir() -> Path:
    p = Path.home() / "Projects" / "qwen36-vast"
    if p.exists():
        return p
    # Fallback to project-local
    p2 = Path(__file__).resolve().parent.parent.parent / "projects" / "qwen36-vast"
    if p2.exists():
        return p2
    return Path.home() / "Projects" / "qwen36-vast"


def _tunnel_script_path() -> Path:
    base = _default_vast_dir()
    p = base / "tools" / "vast_tunnel.sh"
    if p.exists():
        return p
    # Also check harness-local if we copied scripts there
    p2 = base.parent / "qwen36-harness" / "projects" / "qwen36-vast" / "tools" / "vast_tunnel.sh"
    if p2.exists():
        return p2
    return Path.home() / "Projects" / "qwen36-vast" / "tools" / "vast_tunnel.sh"
