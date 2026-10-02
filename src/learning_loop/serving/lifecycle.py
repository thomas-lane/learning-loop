"""Inference lifecycle: which checkpoint is being served, by whom, and where.

Single-GPU-host lifecycle is sequential: serve learner -> collect -> serve the
fixed editor (only if it differs) -> restore the learner for verification ->
stop serving -> train -> serve the next checkpoint. `ensure()` swaps servers
only when the requested checkpoint differs from the one being served, and only
ever stops processes this run started (tracked by handle / remote PID).

External endpoints are never restarted or re-pointed; they may only be used
for the single checkpoint the machine profile declares they serve.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.config import REPO_ROOT, InferenceProfile, ModelProfile
from ..core.records import CheckpointRef, PolicySpec, SamplingConfig
from ..hosts.remote import Remote


class InferenceError(RuntimeError):
    pass


def served_name(ckpt: CheckpointRef) -> str:
    return ckpt.checkpoint_id


def _get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (local/tunneled endpoint)
        return json.loads(r.read().decode())


def endpoint_models(api_base: str) -> list[str]:
    data = _get_json(api_base.rstrip("/") + "/models")
    return [m.get("id") for m in data.get("data", [])]


@dataclass
class ServerHandle:
    checkpoint_id: str
    api_base: str
    popen: subprocess.Popen | None = None
    remote_pid: int | None = None
    log_path: str | None = None
    remote_log_rel: str | None = None  # server log on the remote host (copied back on stop)
    tunnel: subprocess.Popen | None = None  # SSH tunnel this manager opened (runpod hosts)
    started_at: float = field(default_factory=time.time)


class InferenceManager:
    """One manager per role (learner / editor) per coordinator process."""

    def __init__(self, profile: InferenceProfile, model: ModelProfile, log_dir: Path, role: str = "learner",
                 zero_lora: dict[str, Any] | None = None):
        self.profile = profile
        self.zero_lora = zero_lora  # serve base checkpoints through an all-zero LoRA of this shape
        self.model = model
        self.log_dir = Path(log_dir)
        self.role = role
        self.current: ServerHandle | None = None
        self.events: list[dict[str, Any]] = []  # lifecycle log (saved by the coordinator)

    # -- policy specs ------------------------------------------------------- #

    def policy_spec(self, ckpt: CheckpointRef, sampling: SamplingConfig, scripted_path: str | None = None) -> PolicySpec:
        if self.profile.mode == "scripted":
            return PolicySpec(kind="scripted", checkpoint=ckpt, served_model_name=ckpt.checkpoint_id, scripted_path=scripted_path, sampling=sampling, seed_supported=True)
        api_base = self.current.api_base if self.current else self.profile.api_base
        return PolicySpec(
            kind="openai",
            checkpoint=ckpt,
            served_model_name=served_name(ckpt) if self.profile.mode == "managed" else self._external_model_name(),
            api_base=api_base,
            api_key_env=self.profile.api_key_env,
            send_seed=True,
            seed_supported=self.model.seed_supported,
            sampling=sampling,
            request_timeout_sec=float(self.profile.request_timeout_sec),
            max_concurrent_requests=self.profile.request_concurrency,
        )

    def _external_model_name(self) -> str:
        if self.profile.served_model_name:
            return self.profile.served_model_name
        names = endpoint_models(self.profile.api_base)  # type: ignore[arg-type]
        if len(names) != 1:
            # Router endpoints list several models; the served id must be explicit.
            if self.profile.served_checkpoint_id in names:
                return self.profile.served_checkpoint_id  # type: ignore[return-value]
            raise InferenceError(f"external endpoint lists {names}; set served_checkpoint_id to one of them")
        return names[0]

    # -- lifecycle ---------------------------------------------------------- #

    def ensure(self, ckpt: CheckpointRef) -> None:
        if self.profile.mode == "scripted":
            return
        if self.profile.mode == "external":
            if self.profile.served_checkpoint_id != ckpt.checkpoint_id:
                raise InferenceError(
                    f"external endpoint declares it serves {self.profile.served_checkpoint_id!r}, "
                    f"but {self.role} needs {ckpt.checkpoint_id!r}; external endpoints are never re-pointed"
                )
            return
        if self.current and self.current.checkpoint_id == ckpt.checkpoint_id and self._healthy(self.current):
            return
        self.stop()
        self.current = self._start(ckpt)

    def _backend_profile(self):
        sb = self.model.serving.get(self.profile.backend)
        if sb is None:
            raise InferenceError(f"model profile {self.model.name} declares no {self.profile.backend!r} serving backend")
        return sb

    def _adapter_path_on_host(self, ckpt: CheckpointRef) -> str | None:
        """Where the server host finds the adapter. Local hosts use the published path; an SSH host
        gets the same repository-relative path under its workdir, pushed there if it is missing
        (it is already there when that host also trained it)."""
        if not ckpt.adapter_path:
            return None
        host = self.profile.host
        if host.kind == "local":
            return ckpt.adapter_path
        local = Path(ckpt.adapter_path).resolve()
        try:
            rel = local.relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError as e:
            raise InferenceError(f"{local} is outside the repository; remote serving maps checkpoints by repository-relative path") from e
        remote = Remote(host)
        if remote.read_text(f"{rel}/checkpoint.json") is None:
            remote.run(["mkdir", "-p", str(Path(rel).parent)])
            remote.push(local, f"{Path(rel).parent.as_posix()}/")
        return f"{host.workdir.rstrip('/')}/{rel}"

    def _server_argv(self, ckpt: CheckpointRef) -> list[str]:
        backend = self.profile.backend
        adapter = self._adapter_path_on_host(ckpt)
        if backend == "hf_transformers":
            argv = ["-m", "learning_loop.serving.hf_server", "--profile", self.model.name, "--port", str(self.profile.port), "--device", self.profile.device]
            if adapter:
                argv += ["--checkpoint-dir", adapter]  # served name = the checkpoint id
            else:
                argv += ["--base-checkpoint-id", ckpt.checkpoint_id]
                if self.zero_lora:
                    argv += ["--zero-lora", json.dumps(self.zero_lora, sort_keys=True)]
            return argv
        if backend == "vllm":
            # Untested integration: requires a Linux GPU host with vLLM installed separately.
            argv = ["vllm", "serve", ckpt.base_model, "--revision", ckpt.base_revision, "--port", str(self.profile.port), "--served-model-name", ckpt.checkpoint_id if not adapter else "base"]
            if adapter:
                argv += ["--enable-lora", "--lora-modules", f"{ckpt.checkpoint_id}={adapter}"]
            return argv + list(self._backend_profile().launch_args)
        raise InferenceError(f"managed serving is not implemented for backend {backend!r} (see configs/models/{self.model.name}.yaml)")

    def _start(self, ckpt: CheckpointRef) -> ServerHandle:
        argv = self._server_argv(ckpt)
        host = self.profile.host
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"{self.role}-server-{ckpt.checkpoint_id.replace('/', '_')}-{int(time.time())}.log"
        api_base = self.profile.api_base or f"http://127.0.0.1:{self.profile.port}/v1"
        if host.kind == "local":
            full = [sys.executable, *argv] if argv[0] == "-m" else argv
            log = open(self.log_dir / log_name, "w")
            popen = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ})
            handle = ServerHandle(ckpt.checkpoint_id, api_base, popen=popen, log_path=str(self.log_dir / log_name))
        else:
            remote = Remote(host)
            remote.push_repo()  # serve exactly the coordinator's code (training does the same)
            full = ["uv", "run", "--extra", "train", "python", *argv] if argv[0] == "-m" else argv
            remote_log = f"runs/_servers/{log_name}"
            pid = remote.start_detached(full, remote_log)
            handle = ServerHandle(ckpt.checkpoint_id, api_base, remote_pid=pid, log_path=f"{remote.alias}:{remote_log}",
                                  remote_log_rel=remote_log)
            if host.kind == "runpod":
                # A pod's address changes across restarts, so the run owns the tunnel; for `ssh`
                # hosts the user keeps their own tunnel to api_base open (see docs/runpod.md).
                handle.tunnel = subprocess.Popen(
                    remote.tunnel_argv(self.profile.port, self.profile.port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        self.events.append({"event": "start", "checkpoint_id": ckpt.checkpoint_id, "argv": full, "log": handle.log_path, "t": time.time(),
                            "pid": handle.popen.pid if handle.popen else handle.remote_pid, "host": host.ssh_alias or "local"})
        self._wait_ready(handle, ckpt)
        return handle

    def _healthy(self, h: ServerHandle) -> bool:
        if h.popen is not None and h.popen.poll() is not None:
            return False
        try:
            return served_name_ok(endpoint_models(h.api_base), h.checkpoint_id)
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _wait_ready(self, h: ServerHandle, ckpt: CheckpointRef) -> None:
        deadline = time.time() + self.profile.startup_timeout_sec
        while time.time() < deadline:
            if h.popen is not None and h.popen.poll() is not None:
                raise InferenceError(f"{self.role} server exited with {h.popen.returncode} while loading {ckpt.checkpoint_id}; log: {h.log_path}")
            if h.remote_pid is not None and not Remote(self.profile.host).pid_alive(h.remote_pid):
                self._close(h)
                raise InferenceError(f"{self.role} server on {self.profile.host.kind} host exited while loading {ckpt.checkpoint_id}; log: {self._fetch_log(h)}")
            if h.tunnel is not None and h.tunnel.poll() is not None:
                h.tunnel = subprocess.Popen(
                    Remote(self.profile.host).tunnel_argv(self.profile.port, self.profile.port),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                )  # the server may not have been listening yet; re-open
            if self._healthy(h):
                self.events.append({"event": "ready", "checkpoint_id": ckpt.checkpoint_id, "t": time.time()})
                return
            time.sleep(2)
        self._terminate(h)
        raise InferenceError(f"{self.role} server for {ckpt.checkpoint_id} not ready after {self.profile.startup_timeout_sec}s; log: {h.log_path}")

    def _terminate(self, h: ServerHandle) -> None:
        if h.popen is not None and h.popen.poll() is None:
            os.killpg(h.popen.pid, signal.SIGTERM)
            try:
                h.popen.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(h.popen.pid, signal.SIGKILL)
                h.popen.wait(timeout=30)
        elif h.remote_pid is not None:
            stopped = Remote(self.profile.host).stop_group(h.remote_pid)
            self._close(h)
            if not stopped:
                raise InferenceError(f"remote server group {h.remote_pid} on {self.profile.host.kind} host did not stop; not starting another")

    def _close(self, h: ServerHandle) -> None:
        """Close the tunnel and copy the remote server log next to the local logs."""
        if h.tunnel is not None and h.tunnel.poll() is None:
            h.tunnel.terminate()
            try:
                h.tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                h.tunnel.kill()
        h.tunnel = None
        self._fetch_log(h)

    def _fetch_log(self, h: ServerHandle) -> str | None:
        if not h.remote_log_rel:
            return h.log_path
        local = self.log_dir / Path(h.remote_log_rel).name
        try:
            Remote(self.profile.host).pull(h.remote_log_rel, local)
            h.log_path = str(local)
        except Exception as e:  # noqa: BLE001 - logs are diagnostics; never fail a stop over them
            self.events.append({"event": "log_copy_failed", "checkpoint_id": h.checkpoint_id, "error": str(e), "t": time.time()})
        return h.log_path

    def stop(self) -> None:
        if self.current is None:
            return
        self._terminate(self.current)
        self.events.append({"event": "stop", "checkpoint_id": self.current.checkpoint_id, "t": time.time()})
        self.current = None


def served_name_ok(names: list[str], checkpoint_id: str) -> bool:
    return checkpoint_id in names
