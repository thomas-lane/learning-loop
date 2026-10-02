"""Plain SSH/rsync helpers for the three machine roles.

A remote host is either an alias from the user's own ~/.ssh/config (keys, ports
and jump hosts live there, never in this repo), or a Runpod pod whose current
address comes from `pods.endpoint()` (the pod lifecycle resolves it through the
Runpod API). Every remote command runs in the host's declared `workdir`.

Ambiguous interruptions: a local timeout or dropped connection does NOT prove
the remote process stopped. `remote_run_state()` asks the host whether the run
lock is still held before anything is resumed or resubmitted.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..core.config import REPO_ROOT, HostRef

# Never synced to remote hosts: outputs, environments, caches, private profiles and secrets.
RSYNC_EXCLUDES = [".venv", "runs", "artifacts", "evaluation/jobs", "__pycache__", ".pytest_cache", "configs/machines/local", ".DS_Store", ".env", ".git", ".engines"]


class RemoteError(RuntimeError):
    pass


# Non-destructive: takes the lock only momentarily (never while a coordinator holds it).
LOCK_PROBE = """\
import fcntl, os, sys
p = os.path.join(sys.argv[1], ".lock")
if not os.path.exists(p):
    print("missing"); sys.exit(0)
f = open(p, "a+")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("free")
except BlockingIOError:
    print("locked")
"""


@dataclass
class Remote:
    host: HostRef
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.host.kind not in ("ssh", "runpod"):
            raise ValueError("Remote requires an ssh or runpod host")
        self.workdir = self.host.workdir
        if self.host.kind == "ssh":
            self.alias = self.host.ssh_alias
            self._opts: list[str] = []
            self._dest = self.alias
        else:
            from .pods import endpoint

            ep = endpoint(self.host.pod_ref)  # current address of a running pod
            self.alias = f"runpod:{self.host.pod_ref}"
            self._opts = ep.ssh_options()
            self._dest = f"{ep.user}@{ep.ip}"

    def ssh_argv(self, *extra: str) -> list[str]:
        """`ssh [options] destination [extra...]` for this host."""
        return ["ssh", "-o", "BatchMode=yes", *self._opts, self._dest, *extra]

    def _ssh(self, command: str, timeout: int | None = 60, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
        full = f"cd {shlex.quote(self.workdir)} && {command}"
        argv = self.ssh_argv(full)
        if self.dry_run:
            return subprocess.CompletedProcess(argv, 0, stdout=" ".join(map(shlex.quote, argv)), stderr="")
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=stdin)
        except subprocess.TimeoutExpired as e:
            raise RemoteError(f"ssh {self.alias}: timed out after {timeout}s (remote state unknown; reconcile before retrying)") from e
        if check and out.returncode != 0:
            raise RemoteError(f"ssh {self.alias}: exit {out.returncode}: {out.stderr.strip()[-2000:]}")
        return out

    def run(self, argv: list[str], timeout: int | None = 60, check: bool = True) -> subprocess.CompletedProcess:
        return self._ssh(" ".join(shlex.quote(a) for a in argv), timeout=timeout, check=check)

    def push_repo(self) -> None:
        """Sync the code checkout (not runs/artifacts/private profiles) to the host."""
        excludes = sum((["--exclude", e] for e in RSYNC_EXCLUDES), [])
        self._rsync([*excludes, f"{REPO_ROOT}/", f"{self._dest}:{self.workdir}/"])

    def push(self, local: Path, remote_rel: str) -> None:
        self._rsync([str(local), f"{self._dest}:{self.workdir}/{remote_rel}"])

    def pull(self, remote_rel: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._rsync([f"{self._dest}:{self.workdir}/{remote_rel}", str(local)])

    def _rsync(self, args: list[str]) -> None:
        rsh = ["-e", " ".join(shlex.quote(a) for a in ["ssh", "-o", "BatchMode=yes", *self._opts])] if self._opts else []
        argv = ["rsync", "-az", "--partial", *rsh, *args]
        if self.dry_run:
            return
        out = subprocess.run(argv, capture_output=True, text=True)
        if out.returncode != 0:
            raise RemoteError(f"rsync failed ({out.returncode}): {out.stderr.strip()[-2000:]}")

    def start_detached(self, argv: list[str], log_rel: str) -> int:
        """Start a command in its own session (setsid) so the returned PID is the leader of a
        process group this run owns; `stop_group` signals exactly that group."""
        cmd = " ".join(shlex.quote(a) for a in argv)
        out = self._ssh(
            f"mkdir -p $(dirname {shlex.quote(log_rel)}) && "
            f"{{ setsid nohup {cmd} > {shlex.quote(log_rel)} 2>&1 < /dev/null & echo $!; }}"
        )
        pid = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if not pid.isdigit():
            raise RemoteError(f"could not read remote pid from {out.stdout!r}")
        return int(pid)

    def stop_group(self, pid: int, grace_sec: int = 60) -> bool:
        """TERM the process group led by `pid`, wait, then KILL. Returns True once it is gone."""
        p = int(pid)
        script = (
            f"kill -TERM -- -{p} 2>/dev/null; for i in $(seq {grace_sec}); do kill -0 {p} 2>/dev/null || exit 0; sleep 1; done; "
            f"kill -KILL -- -{p} 2>/dev/null; sleep 2; kill -0 {p} 2>/dev/null && exit 1 || exit 0"
        )
        return self._ssh(script, timeout=grace_sec + 30, check=False).returncode == 0

    def read_text(self, rel: str) -> str | None:
        out = self._ssh(f"cat {shlex.quote(rel)} 2>/dev/null", check=False)
        return out.stdout if out.returncode == 0 else None

    def pid_alive(self, pid: int, needle: str | None = None) -> bool:
        """True if `pid` runs (and, with `needle`, its command line contains it, which guards
        against PID reuse after a pod restart)."""
        check = f"kill -0 {int(pid)} 2>/dev/null"
        if needle:
            check += f" && ps -o args= -p {int(pid)} | grep -qF -- {shlex.quote(needle)}"
        out = self._ssh(f"{check} && echo alive || echo gone", check=False)
        return out.stdout.strip().endswith("alive")

    def run_state(self, run_rel: str) -> str:
        """'locked' if a coordinator still holds the run lock, 'free', or 'missing'."""
        probe = LOCK_PROBE
        out = self._ssh(f"python3 -c {shlex.quote(probe)} {shlex.quote(run_rel)}", check=False)
        state = out.stdout.strip()
        if state not in {"locked", "free", "missing"}:
            raise RemoteError(f"could not determine remote run state: {out.stderr.strip()}")
        return state

    def tunnel_argv(self, local_port: int, remote_port: int) -> list[str]:
        """argv for an SSH local-forward to a remote inference server (user runs/keeps it)."""
        return ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", *self._opts,
                "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", self._dest]
