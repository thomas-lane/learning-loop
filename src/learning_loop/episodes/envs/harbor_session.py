"""EnvironmentSession over a Harbor `BaseEnvironment` (the task's Docker container).

- Tools run through Harbor's `exec` / `upload_file`, exactly as the original
  agent did. Commands with a timeout are additionally wrapped in coreutils
  `timeout` inside the container (when available) so a timed-out command does
  not keep running as an unmodeled background process; Harbor's own exec
  timeout remains as a backstop.
- Transport failures (any exception from Harbor's exec/upload, including
  Harbor's backstop exec timeout) raise `EnvInfraError`: the episode stops as
  INFRA. Command-level results (non-zero exit, exit 124 from the in-container
  `timeout`) are observations. Without an in-container `timeout`, Harbor's
  exec timeout *is* the command timeout and is reported as a timed-out command.
- Image identity: `image_identity()` hashes the task's build context
  (Dockerfile + files, whose base images are pinned by digest) and records the
  locally built image id when `docker image inspect` can reach it.
- Fingerprints run `learning_loop/episodes/fingerprint.py` inside the container with
  `python3 -I -B` (isolated: no cwd imports, no bytecode writes) from `/`, so
  they are read-only with respect to the task state. Tasks declaring
  deterministic replay must ship `python3` in their image.
- CPU time per tool call is the delta of the container cgroup's
  `cpu.stat:usage_usec` (scope: the whole container, measured by two extra
  execs around the call). Peak memory per call is not measurable this way
  (cgroup v2 `memory.peak` is container-lifetime), so it is None.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
from pathlib import Path
from typing import Any

from evaluation.agents.tools import ToolConfig

from ...core.interfaces import EnvCapabilities, StateSpec
from ...core.records import RestoreCapability
from .base import EnvInfraError, FingerprintError, ToolSession, build_inputs_identity

_FINGERPRINT_SOURCE = (Path(__file__).resolve().parents[1] / "fingerprint.py").read_text()
_FINGERPRINT_B64 = base64.b64encode(_FINGERPRINT_SOURCE.encode()).decode()
_PROBE_SOURCE = (Path(__file__).resolve().parents[2] / "tasks" / "runtime" / "probe.py").read_text()
_PROBE_B64 = base64.b64encode(_PROBE_SOURCE.encode()).decode()
_TIMEOUT_GRACE_SEC = 5
_BACKSTOP_EXTRA_SEC = 30


class _HarborExec:
    """Exec target that enforces command timeouts inside the container."""

    def __init__(self, env: Any):
        self.env = env
        self.enforces_timeout = False
        self._probed = False

    async def probe(self) -> None:
        if self._probed:
            return
        self._probed = True
        try:
            r = await self.env.exec("command -v timeout >/dev/null 2>&1", cwd="/", timeout_sec=30)
            self.enforces_timeout = r.return_code == 0
        except Exception:
            self.enforces_timeout = False

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout_sec: int | None = None, user: str | int | None = None) -> Any:
        await self.probe()
        if timeout_sec and self.enforces_timeout:
            wrapped = f"timeout -k {_TIMEOUT_GRACE_SEC} {int(timeout_sec)} bash -c {shlex.quote(command)}"
            try:
                return await self.env.exec(wrapped, cwd=cwd, env=env, timeout_sec=int(timeout_sec) + _BACKSTOP_EXTRA_SEC, user=user)
            except Exception as e:  # incl. the backstop timeout: the container did not answer
                raise EnvInfraError(f"harbor exec failed: {type(e).__name__}: {e}") from e
        try:
            return await self.env.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        except Exception as e:
            if timeout_sec and "timed out" in str(e).lower():
                raise TimeoutError(str(e)) from e  # Harbor's timeout is the command timeout here
            raise EnvInfraError(f"harbor exec failed: {type(e).__name__}: {e}") from e

    async def upload_file(self, source_path: Path | str, target_path: str) -> Any:
        try:
            return await self.env.upload_file(source_path, target_path)
        except Exception as e:
            raise EnvInfraError(f"harbor upload failed: {type(e).__name__}: {e}") from e


class HarborSession(ToolSession):
    def __init__(self, environment: Any, tool_cfg: ToolConfig, restore: RestoreCapability = RestoreCapability.NONE, measure_cpu: bool = True):
        self.environment = environment
        self.tool_cfg = tool_cfg
        self.target = _HarborExec(environment)
        self._cpu_ok = measure_cpu
        self.capabilities = EnvCapabilities(
            restore=restore,
            fingerprint=True,
            measures_cpu=measure_cpu,
            measures_memory=False,
            workdir=tool_cfg.workdir,
        )

    async def image_identity(self) -> dict[str, Any] | None:
        env_dir = getattr(self.environment, "environment_dir", None)
        if env_dir is None:
            return None
        cfg = getattr(self.environment, "task_env_config", None)
        prebuilt = getattr(cfg, "docker_image", None) if cfg is not None else None
        name = prebuilt or getattr(getattr(self.environment, "_env_vars", None), "main_image_name", None)
        return build_inputs_identity("harbor_docker", Path(env_dir), prebuilt_image=prebuilt, image_id=await _docker_image_id(name) if isinstance(name, str) else None)

    async def _cpu_usage_sec(self) -> float | None:
        if not self._cpu_ok:
            return None
        try:
            r = await self.environment.exec("cat /sys/fs/cgroup/cpu.stat", cwd="/", timeout_sec=30)
        except Exception:
            r = None
        if r is None or r.return_code != 0 or not r.stdout:
            self._cpu_ok = False
            self.capabilities.measures_cpu = False
            return None
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "usage_usec":
                return int(parts[1]) / 1e6
        self._cpu_ok = False
        self.capabilities.measures_cpu = False
        return None

    async def probe(self, expect: dict[str, Any]) -> dict[str, Any]:
        """Run tasks/runtime/probe.py in the container: {"ok", "violations", "observed"}."""
        cmd = f"echo {_PROBE_B64} | base64 -d | python3 -I -B - {shlex.quote(json.dumps(expect, sort_keys=True))}"
        try:
            r = await self.environment.exec(cmd, cwd="/", timeout_sec=60)
        except Exception as e:
            raise EnvInfraError(f"harbor exec failed (env probe): {type(e).__name__}: {e}") from e
        try:
            result = json.loads(r.stdout or "")
        except json.JSONDecodeError:
            return {"ok": False, "violations": [f"probe_unavailable:exit_{r.return_code}"], "observed": {"stderr": (r.stderr or "")[-500:]}}
        return result

    async def fingerprint_with_detail(self, spec: StateSpec) -> tuple[str, list[str]]:
        arg = json.dumps({"paths": spec.fingerprint_paths, "exclude": spec.fingerprint_exclude, "cwd": self.tool_cfg.workdir})
        cmd = f"echo {_FINGERPRINT_B64} | base64 -d | python3 -I -B - {shlex.quote(arg)}"
        try:
            r = await self.environment.exec(cmd, cwd="/", timeout_sec=120)
        except Exception as e:
            raise EnvInfraError(f"harbor exec failed (fingerprint): {type(e).__name__}: {e}") from e
        if r.return_code != 0:
            raise FingerprintError(f"fingerprint script failed (exit {r.return_code}): {(r.stderr or '').strip()[:500]}")
        try:
            data = json.loads(r.stdout or "")
        except json.JSONDecodeError as e:
            raise FingerprintError(f"fingerprint output is not JSON: {e}") from e
        return data["sha256"], data["lines"]


async def _docker_image_id(name: str) -> str | None:
    """`docker image inspect` id of a local image (best effort; None when unreachable)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "image", "inspect", "--format", "{{.Id}}", name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
    except (OSError, TimeoutError):
        return None
    text = out.decode().strip()
    return text if proc.returncode == 0 and text.startswith("sha256:") else None
