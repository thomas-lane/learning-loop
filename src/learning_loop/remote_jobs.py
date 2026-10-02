"""Laptop -> Linux coordinator job submission over SSH.

The coordinator host runs Docker itself; the laptop only syncs code, starts
`loop run` detached, and later pulls the run directory back. Submission
records live in runs/_submissions/<run_id>.json.

Reconciliation rule: before (re)submitting a run id, ask the host whether the
run lock is held. A lost SSH session or local timeout never counts as proof
that the remote coordinator stopped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import REPO_ROOT, HostRef, load_machine
from .coordinator import load_all, new_run_id
from .remote import Remote, RemoteError
from .storage import atomic_write_json, now_iso, read_json

SUBMISSIONS = REPO_ROOT / "runs" / "_submissions"


def _coordinator(machines: str) -> Remote:
    m = load_machine(machines)
    if m.coordinator.kind != "ssh":
        raise RemoteError(f"machine profile {m.name} has a local coordinator; use `loop run` directly")
    return Remote(m.coordinator)


def submit(experiment: str, machines: str, run_id: str | None = None) -> str:
    exp, *_ = load_all(experiment, machines)  # validate locally before touching the host
    run_id = run_id or new_run_id(exp)
    remote = _coordinator(machines)
    state = remote.run_state(f"runs/{run_id}")
    if state == "locked":
        raise RemoteError(f"run {run_id} is still locked on {remote.alias}: a coordinator is running; not resubmitting")
    if state == "free":
        raise RemoteError(f"run {run_id} already exists on {remote.alias}; use `ssh {remote.alias}` + `loop resume runs/{run_id}`")
    remote.push_repo()
    remote.run(["mkdir", "-p", f"runs/{run_id}"])
    remote.push(Path(machines).resolve(), f"runs/{run_id}/submitted-machine.yaml")
    exp_rel = str(Path(experiment).resolve().relative_to(REPO_ROOT))
    pid = remote.start_detached(
        ["uv", "run", "loop", "run", exp_rel, "--machines", f"runs/{run_id}/submitted-machine.yaml", "--run-id", run_id],
        f"runs/{run_id}/coordinator.log",
    )
    rec = {"run_id": run_id, "host": remote.alias, "workdir": remote.workdir, "pid": pid, "submitted_at": now_iso(), "experiment": exp_rel}
    atomic_write_json(SUBMISSIONS / f"{run_id}.json", rec)
    return f"submitted {run_id} to {remote.alias} (pid {pid}); log: runs/{run_id}/coordinator.log"


def remote_status(run_id: str, machines: str) -> dict[str, Any]:
    remote = _coordinator(machines)
    sub = SUBMISSIONS / f"{run_id}.json"
    rec = read_json(sub) if sub.exists() else {}
    out: dict[str, Any] = {"run_id": run_id, "lock": remote.run_state(f"runs/{run_id}")}
    if rec.get("pid"):
        out["pid"] = rec["pid"]
        out["pid_alive"] = remote.pid_alive(rec["pid"])
    st = remote.run(["uv", "run", "loop", "status", f"runs/{run_id}"], check=False, timeout=120)
    out["status"] = st.stdout.strip() or st.stderr.strip()[-500:]
    return out


def fetch(run_id: str, machines: str) -> str:
    remote = _coordinator(machines)
    dest = REPO_ROOT / "runs" / run_id
    remote.pull(f"runs/{run_id}/", dest)
    return f"fetched to {dest}"


def sync_hosts(machines: str, dry_run: bool = False) -> list[str]:
    """Push the code checkout to every SSH host the machine profile names and build the locked
    environment there (`uv sync --frozen`, with the `train` extra on GPU roles), so the first
    server start or training stage does not spend its timeout installing packages."""
    m = load_machine(machines)
    out: list[str] = []
    if m.pod_ids():
        def what(p: str) -> str:
            return "create, prepare, then terminate" if p.startswith("new-") else f"start (if stopped), prepare, then stop (stop_when_done={m.runpod.stop_when_done})"

        if dry_run:
            out += [f"runpod:{p}: would {what(p)}" for p in m.pod_ids()]
        else:
            from .pods import PodLifecycle

            with PodLifecycle(m, REPO_ROOT / "runs" / "_pods"):  # prepare = setup + code + environment
                pass
            out += [f"runpod:{p}: done: {what(p)}" for p in m.pod_ids()]
    roles: dict[tuple[str, str], set[str]] = {}
    for role, host in [("coordinator", m.coordinator), ("inference", m.inference.host), ("training", m.training.host)] + (
        [("editor_inference", m.editor_inference.host)] if m.editor_inference else []
    ):
        if host.kind == "ssh":
            roles.setdefault((host.ssh_alias, host.workdir), set()).add(role)
    for (alias, workdir), rs in roles.items():
        # the workdir must exist before commands can `cd` into it
        Remote(HostRef(kind="ssh", ssh_alias=alias, workdir="/"), dry_run=dry_run).run(["mkdir", "-p", workdir])
        remote = Remote(HostRef(kind="ssh", ssh_alias=alias, workdir=workdir), dry_run=dry_run)
        remote.push_repo()
        argv = ["uv", "sync", "--frozen"] + (["--extra", "train"] if rs & {"inference", "training", "editor_inference"} else [])
        remote.run(argv, timeout=None)
        verb = "would push code and run" if dry_run else "code pushed; ran"
        out.append(f"{alias}:{workdir} ({', '.join(sorted(rs))}): {verb} `{' '.join(argv)}`")
    return out or ["no SSH hosts in this machine profile; nothing to do"]

