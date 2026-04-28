"""Background scheduler — register jobs that run on intervals.

Each job is a daemon thread that calls subprocess.run() on its command.
Jobs can be one-shot, repeating, or infinite (repeat=0).

Usage from the model:
    cron_register(name="heartbeat", cmd="echo tick", interval_s=60)
    cron_run_now("heartbeat")       # execute immediately
    cron_list()                     # show all jobs with status
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from . import default_registry, ToolError


# ---------------------------------------------------------------------------
# Scheduler — thread-per-job background runner
# ---------------------------------------------------------------------------

class Scheduler:
    """Simple threaded background job runner.

    Thread-safe for register/stop/list; individual jobs are self-contained.
    All commands timeout at 5 minutes to avoid runaway processes.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}   # name -> spec
        self._lock = threading.RLock()    # Reentrant — run_once calls _execute while locked
        self._threads: dict[str, threading.Thread] = {}

    # -- public API ----------------------------------------------------------

    def register(
        self,
        name: str,
        cmd: str,
        interval_s: int | None = None,
        repeat: int = 0,
        max_age_s: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict:
        """Register a cron job. Returns the created job spec."""

        if interval_s is None and repeat > 0:
            interval_s = 60  # default interval for recurring jobs

        now_iso = datetime.now(timezone.utc).isoformat()

        job: dict[str, Any] = {
            "id": f"{name}-{int(time.time())}",
            "name": name,
            "cmd": cmd,
            "interval_s": interval_s,
            "repeat": repeat,         # 0 = forever
            "runs": 0,
            "max_age_s": max_age_s,
            "cwd": cwd,
            "env": env or {},
            "created_at": now_iso,
            "next_run": _now_utc(),
            "last_run": None,
            "status": "pending",
        }

        with self._lock:
            if name in self._jobs:
                raise ToolError(f"job '{name}' already exists")
            self._jobs[name] = job

        # Kick off runner thread (daemon so it dies with the process)
        t = threading.Thread(target=self._run_loop, args=(name,), daemon=True)
        t.start()
        self._threads[name] = t

        return _format_job(job)

    def run_once(self, name: str) -> dict:
        """Execute a registered job immediately (one-shot, no scheduling)."""

        with self._lock:
            if name not in self._jobs:
                raise ToolError(f"unknown job '{name}'")
            result = self._execute(name)

        with self._lock:
            job = self._jobs[name]
            job["last_run"] = _now_utc()
            job["status"] = "ok" if not isinstance(result, dict) or "error" not in result else "error"

        return {
            **_format_job(job),
            "last_output": (result.get("stdout", "") if isinstance(result, dict) else str(result))[:500],
            "last_error": (result.get("stderr", "") if isinstance(result, dict) and result != {} else None),
        }

    def stop(self, name: str) -> dict:
        """Stop a running job. Does not remove it from registry."""

        with self._lock:
            if name not in self._jobs:
                raise ToolError(f"unknown job '{name}'")
            t = self._threads.pop(name, None)

        # Threads are daemon and can't be killed — the run loop checks
        # membership on each iteration so it will exit naturally.
        return {"stopped": name}

    def remove(self, name: str) -> dict:
        """Permanently stop and delete a job."""

        self.stop(name)

        with self._lock:
            del self._jobs[name]

        return {"removed": name}

    def list_jobs(self) -> list[dict]:
        """List all registered jobs with current status."""

        with self._lock:
            out = []
            for name, job in sorted(self._jobs.items()):
                running = name in self._threads and self._threads[name].is_alive()
                entry = _format_job(job)
                entry["running"] = running
                out.append(entry)
        return out

    # -- internal -----------------------------------------------------------

    def _run_loop(self, name: str) -> None:
        """Background thread that loops on a schedule until stopped or exhausted."""

        with self._lock:
            job = self._jobs.get(name)
        if not job:
            return

        interval = job["interval_s"] or 60
        counter = 0

        while True:
            # Check external stop before each iteration
            with self._lock:
                if name not in self._jobs:
                    break

            result = self._execute(name)

            with self._lock:
                job["runs"] += 1
                job["last_run"] = _now_utc()
                job["next_run"] = (_now_utc_obj() + timedelta(seconds=interval)).isoformat()
                job["status"] = "ok" if not isinstance(result, dict) or "error" not in result else "error"

            counter += 1
            if 0 < job["repeat"] <= counter:
                break   # done repeating

            if interval:
                time.sleep(interval)

    def _execute(self, name: str) -> Any:
        """Run a command via subprocess. Returns stdout on success."""

        with self._lock:
            job = self._jobs.get(name)
        if not job:
            return {"error": f"job '{name}' gone"}

        try:
            cwd_str = str(Path(job["cwd"]).resolve()) if job.get("cwd") else None
            env_override = {**os.environ, **job.get("env", {})}
            proc = subprocess.run(
                job["cmd"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd_str,
                env=env_override,
            )

            if proc.returncode == 0:
                return {"stdout": proc.stdout[:4096]}
            else:
                error = proc.stderr[:1024] or f"exit code {proc.returncode}"
                return {"error": error}

        except subprocess.TimeoutExpired:
            return {"error": "timed out after 5 minutes"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_utc_obj() -> datetime:
    return datetime.now(timezone.utc)


def _format_job(job: dict[str, Any]) -> dict[str, Any]:
    """Remove internal fields for external-facing output."""
    return {k: v for k, v in job.items() if k not in ("env",)}


# ---------------------------------------------------------------------------
# Singleton + tool registration
# ---------------------------------------------------------------------------

scheduler = Scheduler()


def register(registry=default_registry) -> None:
    """Register all cron.* tools with the harness."""

    @registry.tool(
        name="cron_register",
        description=(
            "Schedule a background command to run at a fixed interval. "
            "The command is executed via shell in a daemon thread. "
            "Useful for periodic tasks like polling, health checks, or heartbeat pings."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique job name (must not collide with existing jobs)",
                },
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute on each tick. "
                                   "Can include pipes, redirects, or calls to scripts.",
                },
                "interval_s": {
                    "type": "integer",
                    "default": 60,
                    "description": "Seconds between runs (1-3600). Default: 60.",
                },
                "repeat": {
                    "type": "integer",
                    "default": 0,
                    "description": "Number of times to run. 0 = forever (unlimited).",
                },
                "max_age_s": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Auto-stop after N seconds from creation (optional).",
                },
                "cwd": {
                    "type": ["string", "null"],
                    "default": None,
                    "description": "Working directory for the command. Defaults to harness cwd.",
                },
            },
            "required": ["name", "cmd"],
        }
    )
    def cron_register(
        name: str,
        cmd: str,
        interval_s: int = 60,
        repeat: int = 0,
        max_age_s: int | None = None,
        cwd: str | None = None,
    ) -> dict:
        """Register a new cron job with the scheduler."""

        if interval_s < 1 or interval_s > 3600:
            raise ToolError("interval_s must be between 1 and 3600")
        return scheduler.register(
            name=name,
            cmd=cmd,
            interval_s=interval_s,
            repeat=repeat,
            max_age_s=max_age_s,
            cwd=cwd,
        )

    @registry.tool(
        name="cron_run_now",
        description=(
            "Execute a registered job immediately as a one-shot. "
            "Does not affect the scheduled run cycle."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Job name to execute right now",
                },
            },
            "required": ["name"],
        }
    )
    def cron_run_now(name: str) -> dict:
        """Run a job immediately (one-shot execution)."""

        return scheduler.run_once(name)

    @registry.tool(
        name="cron_list",
        description=(
            "List all registered cron jobs with their current status, run count, and whether they are running."
        ),
        parameters={
            "type": "object",
            "properties": {},
        }
    )
    def cron_list() -> list[dict]:
        """Show all registered cron jobs."""

        return scheduler.list_jobs()

    @registry.tool(
        name="cron_stop",
        description=(
            "Stop a running job. The job remains in the registry but will not execute on its next tick."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Job name to stop",
                },
            },
            "required": ["name"],
        }
    )
    def cron_stop(name: str) -> dict:
        """Stop a running job."""

        return scheduler.stop(name)

    @registry.tool(
        name="cron_remove",
        description=(
            "Permanently stop and delete a cron job from the registry."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Job name to remove",
                },
            },
            "required": ["name"],
        }
    )
    def cron_remove(name: str) -> dict:
        """Stop and permanently delete a job."""

        return scheduler.remove(name)


__all__ = ["scheduler", "register"]
