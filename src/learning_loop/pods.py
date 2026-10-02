"""Runpod pod lifecycle for `kind: runpod` hosts: start or create, reach, prepare, watch, stop or terminate.

Two kinds of pods:
- existing pods (`pod_id`): only started and stopped, never created or terminated;
- created pods (`pod: <spec>`, `runpod.create`): each command creates one pod per spec, named
  `lfe-<spec>-<UTC stamp>`, records it in a local ledger (artifacts/runpod/created.jsonl) and
  terminates it however the command ends. `loop pod cleanup` terminates any ledger pod that
  is still alive (e.g. after the laptop died). Pods not in the ledger are never terminated.
The account API key is read on the coordinator from the environment variable named in the
profile (e.g. via `.env`) and is never sent to a pod; created pods get only your SSH public key.

`PodLifecycle` wraps every command that needs the pods (run, resume, stage, evaluate,
edit-replay, sync-hosts):

1. create each spec's pod (retrying while no GPU of the listed types is free) or start each
   existing pod that is not running, wait until it reports a public IP and an SSH port mapping
   and accepts SSH (a pod's address changes across restarts, so it is looked up every time);
2. prepare it: the setup script (tools, caches), the code checkout and the locked environment;
3. start the pod-side idle watchdog (scripts/pod_watchdog.py) and refresh its heartbeat from a
   background thread, so a pod stops itself if this process dies or the laptop sleeps;
4. on exit (success, failure or Ctrl-C), terminate created pods, and stop existing pods when
   `stop_when_done` is set.

Pod host keys are regenerated on every pod start, so each command trusts the key it first sees
(a per-pod known_hosts file under artifacts/runpod/, reset at the start of every command).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import REPO_ROOT, HostRef, MachineProfile, RunpodConfig
from .storage import JsonlAppender, now_iso

# Runpod's Cloudflare front end rejects urllib's default agent (error 1010).
USER_AGENT = "learn-from-experience-loop/1"
NO_FREE_GPU = "not enough free GPUs"  # Runpod's start error while the pod's host is fully used
GPU_RETRY_MIN_SEC = 30.0
KNOWN_HOSTS_DIR = REPO_ROOT / "artifacts" / "runpod"
HEARTBEAT_REL = "runs/_pod/heartbeat"  # relative to the pod workdir (runs/ is never synced)
WATCHDOG_LOG_REL = "runs/_pod/watchdog.jsonl"
WATCHDOG_PID_REL = "runs/_pod/watchdog.pid"
LEDGER_NAME = "created.jsonl"  # under KNOWN_HOSTS_DIR
CREATED_PREFIX = "lfe-"
# Runpod's create errors while no machine has a free GPU of the requested types (retried)
NO_CAPACITY = ("no longer any instances available", "not enough free gpus", "no instances available", "could not find any")


class RunpodError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# REST client
# --------------------------------------------------------------------------- #


class RunpodClient:
    """Minimal Runpod REST client: get / list / start / stop / create / delete. Never logs the key."""

    def __init__(self, api_key: str, api_base: str = "https://rest.runpod.io/v1", timeout_sec: float = 30.0):
        self._key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout_sec = timeout_sec

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json", "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"{self.api_base}{path}", method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as r:  # noqa: S310 - fixed https API base
                body = r.read().decode() or "{}"
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RunpodError(f"Runpod API {method} {path}: HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise RunpodError(f"Runpod API {method} {path}: {e}") from None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body[:300]}

    def get(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")

    def start(self, pod_id: str) -> dict[str, Any]:
        return self._request("POST", f"/pods/{pod_id}/start")

    def stop(self, pod_id: str) -> dict[str, Any]:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def list(self) -> list[dict[str, Any]]:
        out = self._request("GET", "/pods")
        return out if isinstance(out, list) else []

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/pods", body)

    def delete(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")


# --------------------------------------------------------------------------- #
# Ledger of created pods (the only pods `loop` may terminate)
# --------------------------------------------------------------------------- #


def ledger_append(event: str, pod_id: str, **data: Any) -> None:
    JsonlAppender(KNOWN_HOSTS_DIR / LEDGER_NAME).append({"t": now_iso(), "event": event, "pod_id": pod_id, **data})


def ledger_open_pods() -> dict[str, dict[str, Any]]:
    """Created pods with no recorded termination: pod id -> its `created` entry."""
    from .storage import read_jsonl

    path = KNOWN_HOSTS_DIR / LEDGER_NAME
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for e in read_jsonl(path):
        if e.get("event") == "created":
            out[e["pod_id"]] = e
        elif e.get("event") == "terminated":
            out.pop(e.get("pod_id"), None)
    return out


def _runpod_host(ref: str, workdir: str) -> HostRef:
    if ref.startswith("new-"):
        return HostRef(kind="runpod", pod=ref[len("new-"):], workdir=workdir)
    return HostRef(kind="runpod", pod_id=ref, workdir=workdir)


def _public_key(identity_file: str) -> str:
    path = Path(os.path.expanduser(identity_file) + ".pub")
    try:
        return path.read_text().strip()
    except OSError as e:
        raise RunpodError(f"cannot read {path} (the public half of runpod.identity_file), which created pods need for SSH: {e}") from None


def client_for(cfg: RunpodConfig) -> RunpodClient:
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise RunpodError(f"{cfg.api_key_env} is not set: add it to .env (see .env.example) or the environment")
    return RunpodClient(key, cfg.api_base)


def pod_address(info: dict[str, Any]) -> tuple[str, int] | None:
    """(public IP, public SSH port) once the pod reports both, else None."""
    ip = info.get("publicIp") or ""
    port = (info.get("portMappings") or {}).get("22")
    return (ip, int(port)) if ip and port else None


# --------------------------------------------------------------------------- #
# Current addresses (used by remote.Remote for kind: runpod)
# --------------------------------------------------------------------------- #


@dataclass
class Endpoint:
    pod_id: str
    ip: str
    port: int
    user: str
    identity_file: str
    known_hosts: Path

    def ssh_options(self) -> list[str]:
        return [
            "-p", str(self.port),
            "-i", os.path.expanduser(self.identity_file),
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30",
        ]


_ENDPOINTS: dict[str, Endpoint] = {}


def register(ep: Endpoint) -> None:
    _ENDPOINTS[ep.pod_id] = ep


def endpoint(pod_id: str | None) -> Endpoint:
    if pod_id not in _ENDPOINTS:
        raise RunpodError(f"pod {pod_id} has no known address in this process; commands that use it start it first")
    return _ENDPOINTS[pod_id]


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


class PodLifecycle:
    """Context manager around one command. No-op when the machine profile has no runpod hosts."""

    def __init__(
        self,
        machine: MachineProfile,
        log_dir: Path,
        log: Callable[[str], None] | None = None,
        client: RunpodClient | None = None,
        prepare: bool = True,
        poll_sec: float = 10.0,
    ):
        self.machine = machine
        self.cfg = machine.runpod
        self.pods = machine.pod_ids()
        self.events = JsonlAppender(Path(log_dir) / "pod-lifecycle.jsonl")
        self.log = log or (lambda m: print(m, flush=True))
        self._client = client
        self.prepare = prepare
        self.poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_by_us: set[str] = set()
        self.created: dict[str, str] = {}  # ref (new-<spec>) -> created pod id

    def pod_id(self, ref: str) -> str:
        """The Runpod pod id behind a ref (created pods get theirs at creation)."""
        return self.created.get(ref, ref) if ref.startswith("new-") else ref

    @property
    def client(self) -> RunpodClient:
        if self._client is None:
            self._client = client_for(self.cfg)  # type: ignore[arg-type]
        return self._client

    def _event(self, pod_id: str, event: str, **data: Any) -> None:
        self.events.append({"t": now_iso(), "pod_id": pod_id, "event": event, **data})

    def _workdirs(self, pod_id: str) -> dict[str, set[str]]:
        """workdir -> roles for this pod (a pod may host several roles, normally one workdir)."""
        out: dict[str, set[str]] = {}
        for role, h in self.machine.hosts():
            if h.kind == "runpod" and h.pod_ref == pod_id:
                out.setdefault(h.workdir, set()).add(role)  # type: ignore[arg-type]
        return out

    # -- enter / exit ------------------------------------------------------- #

    def __enter__(self) -> "PodLifecycle":
        if not self.pods:
            return self
        try:
            for pod in self.pods:
                self._up(pod)
            if self.prepare:
                for pod in self.pods:
                    self._prepare(pod)
            for pod in self.pods:
                self._start_watchdog(pod)
            self._thread = threading.Thread(target=self._heartbeat_loop, name="pod-heartbeat", daemon=True)
            self._thread.start()
        except BaseException:
            self.__exit__(None, None, None)  # never leave a pod we touched running on a failed start
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.pods:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        for pod in self.pods:
            if pod.startswith("new-"):
                self._terminate(pod, "error" if exc_type else "success")
                _ENDPOINTS.pop(pod, None)
                continue
            if not self.cfg.stop_when_done:  # type: ignore[union-attr]
                self._event(pod, "left_running", reason="stop_when_done=false; the watchdog stops it when idle")
                continue
            try:
                self.client.stop(pod)
                self._event(pod, "stop_requested", after=("error" if exc_type else "success"))
                self.log(f"pod {pod}: stop requested")
            except RunpodError as e:
                self._event(pod, "stop_failed", error=str(e))
                self.log(f"WARNING: could not stop pod {pod}: {e} -- stop it in the Runpod console; its watchdog stops it after {self.cfg.idle_stop_minutes} idle minutes")  # type: ignore[union-attr]
            _ENDPOINTS.pop(pod, None)
        return False

    def _terminate(self, ref: str, after: str) -> None:
        pid = self.created.get(ref)
        if pid is None:
            return  # creation never succeeded
        try:
            self.client.delete(pid)
            ledger_append("terminated", pid, ref=ref)
            self._event(ref, "terminate_requested", pod=pid, after=after)
            self.log(f"pod {pid} ({ref}): terminated")
        except RunpodError as e:
            self._event(ref, "terminate_failed", pod=pid, error=str(e))
            self.log(f"WARNING: could not terminate pod {pid}: {e} -- run `loop pod cleanup --machines ...` or terminate it in the "
                     f"Runpod console; its watchdog terminates it after {self.cfg.idle_stop_minutes} idle minutes")  # type: ignore[union-attr]

    def _create(self, ref: str, deadline: float) -> str:
        """Create a pod from the spec, retrying while no GPU of the listed types is free."""
        name = ref[len("new-"):]
        spec = self.cfg.create[name]  # type: ignore[union-attr]
        body = {
            "name": f"{CREATED_PREFIX}{name}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}",
            "imageName": spec.image,
            "gpuTypeIds": list(spec.gpu_types),
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "cloudType": spec.cloud_type,
            "containerDiskInGb": spec.container_disk_gb,
            "volumeInGb": 0,
            "ports": ["22/tcp"],
            "allowedCudaVersions": list(spec.allowed_cuda_versions),
            "env": {"PUBLIC_KEY": _public_key(self.cfg.identity_file)},  # type: ignore[union-attr]
        }
        waiting = False
        while True:
            try:
                info = self.client.create(body)
                break
            except RunpodError as e:
                msg = str(e).lower()
                if not any(t in msg for t in NO_CAPACITY) and "http 5" not in msg:
                    raise
                if time.monotonic() > deadline:
                    raise RunpodError(f"no {' / '.join(spec.gpu_types)} GPU became available in {spec.cloud_type} cloud within "
                                      f"{self.cfg.start_timeout_sec}s; retry later or add GPU types to runpod.create.{name}") from None  # type: ignore[union-attr]
                if not waiting:
                    waiting = True
                    self._event(ref, "waiting_for_gpu", detail=str(e)[-200:])
                    self.log(f"{ref}: no {' / '.join(spec.gpu_types)} available yet; retrying until the start timeout")
                time.sleep(max(self.poll_sec, GPU_RETRY_MIN_SEC))
        pid = info.get("id")
        if not pid:
            raise RunpodError(f"Runpod create returned no pod id: {str(info)[:300]}")
        self.created[ref] = pid  # from here on, exit terminates it
        cost = info.get("costPerHr") or info.get("adjustedCostPerHr")
        gpu = (info.get("gpu") or {}).get("displayName") or (info.get("machine") or {}).get("gpuDisplayName")
        ledger_append("created", pid, ref=ref, name=body["name"], gpu=gpu, cost_per_hr=cost)
        self._event(ref, "created", pod=pid, name=body["name"], gpu=gpu, cost_per_hr=cost)
        self.log(f"{ref}: created pod {pid} ({gpu or '?'}, ${cost}/hr)")
        if cost is not None and float(cost) > spec.max_cost_per_hr:
            raise RunpodError(f"created pod {pid} costs ${cost}/hr, above runpod.create.{name}.max_cost_per_hr={spec.max_cost_per_hr}; terminated")
        return pid

    # -- steps -------------------------------------------------------------- #

    def _up(self, ref: str) -> None:
        cfg = self.cfg
        deadline = time.monotonic() + cfg.start_timeout_sec  # type: ignore[union-attr]
        if ref.startswith("new-"):
            pod = self._create(ref, deadline)
        else:
            pod = ref
            info = self.client.get(pod)
            status = info.get("desiredStatus")
            if status == "TERMINATED":
                raise RunpodError(f"pod {pod} is terminated; create a new pod and update the machine profile")
            if status != "RUNNING":
                self.log(f"pod {pod}: {status or 'unknown'} -> starting")
                self._start_when_gpu_free(pod, deadline)
                self.started_by_us.add(pod)
                self._event(pod, "start_requested", previous=status)
        addr = None
        while True:
            info = self.client.get(pod)
            addr = pod_address(info) if info.get("desiredStatus") == "RUNNING" else None
            if addr:
                break
            if time.monotonic() > deadline:
                raise RunpodError(f"pod {pod} did not report a public IP and SSH port within {cfg.start_timeout_sec}s (status {info.get('desiredStatus')}); its GPU may be taken")  # type: ignore[union-attr]
            time.sleep(self.poll_sec)
        KNOWN_HOSTS_DIR.mkdir(parents=True, exist_ok=True)
        known = KNOWN_HOSTS_DIR / f"{pod}.known_hosts"
        known.unlink(missing_ok=True)  # host keys are regenerated on every pod start
        ep = Endpoint(ref, addr[0], addr[1], cfg.ssh_user, cfg.identity_file, known)  # type: ignore[union-attr]
        register(ep)
        from .remote import Remote, RemoteError

        probe = Remote(_runpod_host(ref, "/"))
        while True:
            try:
                probe.run(["true"], timeout=30)
                break
            except RemoteError as e:
                if time.monotonic() > deadline:
                    raise RunpodError(f"pod {pod} at {ep.ip}:{ep.port} does not accept SSH: {e}") from None
                time.sleep(self.poll_sec)
        self._event(ref, "ready", pod=pod, ip=ep.ip, port=ep.port, started_by_us=pod in self.started_by_us or ref in self.created)
        self.log(f"pod {pod}: reachable at {ep.ip}:{ep.port}")

    def _start_when_gpu_free(self, pod: str, deadline: float) -> None:
        """A stopped pod restarts only on its original host machine; while other users hold that
        host's GPUs the API refuses the start. Retry until the start timeout."""
        waiting = False
        while True:
            try:
                self.client.start(pod)
                return
            except RunpodError as e:
                if NO_FREE_GPU not in str(e):
                    raise
                if time.monotonic() > deadline:
                    raise RunpodError(
                        f"pod {pod} cannot start: its host machine has had no free GPU for {self.cfg.start_timeout_sec}s "  # type: ignore[union-attr]
                        "(a stopped pod can only restart on the machine it was created on). Retry later, raise "
                        "runpod.start_timeout_sec, or deploy a new pod in the Runpod console and put its id in the machine profile."
                    ) from None
                if not waiting:
                    waiting = True
                    self._event(pod, "waiting_for_gpu", detail=str(e)[-200:])
                    self.log(f"pod {pod}: no free GPU on its host yet; retrying until the start timeout")
                time.sleep(max(self.poll_sec, GPU_RETRY_MIN_SEC))

    def _prepare(self, pod: str) -> None:
        """Setup script (idempotent), code checkout and locked environment for each workdir."""
        from .remote import Remote

        script = (REPO_ROOT / "scripts" / "setup_gpu_host.sh").read_text()
        root = Remote(_runpod_host(pod, "/"))
        for workdir, roles in self._workdirs(pod).items():
            out = root._ssh(f"WORKDIR={workdir} bash -s", stdin=script, timeout=900)
            warnings = [ln for ln in (out.stdout + out.stderr).splitlines() if ln.startswith("WARNING")]
            remote = Remote(_runpod_host(pod, workdir))
            remote.push_repo()
            argv = ["uv", "sync", "--frozen"] + (["--extra", "train"] if roles & {"inference", "training", "editor_inference"} else [])
            remote.run(argv, timeout=None)
            gpu = root._ssh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", check=False).stdout.strip()
            self._event(pod, "prepared", workdir=workdir, roles=sorted(roles), warnings=warnings, gpu=gpu or None)
            self.log(f"pod {pod}: prepared ({gpu or 'GPU unknown'})")
            for w in warnings:
                self.log(f"pod {pod}: {w}")

    def _start_watchdog(self, pod: str) -> None:
        from .remote import Remote

        workdir = next(iter(self._workdirs(pod)))
        remote = Remote(_runpod_host(pod, workdir))
        remote.run(["mkdir", "-p", str(Path(HEARTBEAT_REL).parent)])
        remote.run(["touch", HEARTBEAT_REL])
        # Replace an earlier watchdog (e.g. from a previous command on a pod that kept running).
        # (Only if that PID really is a watchdog: after a pod restart the saved PID may belong to
        # an unrelated process.)
        remote._ssh(
            f"p=$(cat {WATCHDOG_PID_REL} 2>/dev/null) && ps -o args= -p \"$p\" | grep -qF pod_watchdog.py && kill \"$p\"; true",
            check=False,
        )
        idle = self.cfg.idle_stop_minutes * 60  # type: ignore[union-attr]
        pid = remote.start_detached(
            ["python3", "scripts/pod_watchdog.py", HEARTBEAT_REL, str(idle), "30", WATCHDOG_LOG_REL,
             "terminate" if pod.startswith("new-") else "stop"],
            "runs/_pod/watchdog.out",
        )
        remote._ssh(f"echo {pid} > {WATCHDOG_PID_REL}")
        self._event(pod, "watchdog_started", pid=pid, idle_stop_minutes=self.cfg.idle_stop_minutes)  # type: ignore[union-attr]

    def _heartbeat_loop(self) -> None:
        from .remote import Remote, RemoteError

        warned: set[str] = set()
        while not self._stop.wait(self.cfg.heartbeat_sec):  # type: ignore[union-attr]
            for pod in self.pods:
                try:
                    workdir = next(iter(self._workdirs(pod)))
                    Remote(_runpod_host(pod, workdir)).run(["touch", HEARTBEAT_REL], timeout=30)
                    warned.discard(pod)
                except (RemoteError, RunpodError) as e:
                    if pod not in warned:
                        self._event(pod, "heartbeat_failed", error=str(e))
                        warned.add(pod)


def pod_lifecycle(machine: MachineProfile, log_dir: Path, **kw: Any) -> PodLifecycle:
    return PodLifecycle(machine, log_dir, **kw)


def pod_status(machine: MachineProfile) -> list[dict[str, Any]]:
    """Read-only: status and address of the profile's existing pods and of every created pod the
    ledger still lists as alive."""
    if not machine.pod_ids():
        return []
    client = client_for(machine.runpod)  # type: ignore[arg-type]
    out = []
    for pod in [*machine.existing_pod_ids(), *ledger_open_pods()]:
        try:
            info = client.get(pod)
        except RunpodError as e:
            out.append({"pod_id": pod, "status": f"unknown ({str(e)[:80]})", "address": None, "gpu": None})
            continue
        addr = pod_address(info)
        out.append({"pod_id": pod, "status": info.get("desiredStatus"), "address": f"{addr[0]}:{addr[1]}" if addr else None,
                    "gpu": (info.get("machine") or {}).get("gpuDisplayName") or info.get("gpuDisplayName")})
    return out


def cleanup(machine: MachineProfile, dry_run: bool = False) -> list[str]:
    """Terminate created pods that are still alive: only pods in the local ledger whose Runpod name
    still starts with `lfe-` (a pod reused or renamed elsewhere is left alone)."""
    client = client_for(machine.runpod)  # type: ignore[arg-type]
    live = {p.get("id"): p for p in client.list()}
    out = []
    for pid, entry in ledger_open_pods().items():
        pod = live.get(pid)
        if pod is None or pod.get("desiredStatus") == "TERMINATED":
            out.append(f"{pid}: already gone")
            if not dry_run:
                ledger_append("terminated", pid, note="not found at cleanup")
            continue
        if not str(pod.get("name", "")).startswith(CREATED_PREFIX):
            out.append(f"{pid}: name {pod.get('name')!r} lacks the {CREATED_PREFIX} prefix; left alone")
            continue
        if dry_run:
            out.append(f"{pid} ({pod.get('name')}, {pod.get('desiredStatus')}): would terminate")
            continue
        client.delete(pid)
        ledger_append("terminated", pid, note="loop pod cleanup")
        out.append(f"{pid} ({pod.get('name')}): terminated")
    return out
