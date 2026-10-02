"""Atomic files, append-only logs, run locks and stage manifests.

Run layout (owned by the coordinator):

    runs/<run_id>/
      run.json                     immutable: resolved config + provenance
      .lock                        exclusive coordinator lock (flock)
      tasks/<instance>/...         generated task instances (content-hashed)
      checkpoints/<ckpt>/          immutable published adapters + checkpoint.json
      cycles/cycle-NNN/<stage>/    stage outputs; manifest.json per stage
      reports/                     generated summaries (CSV + markdown)

Stage manifests track stable work-item IDs. Completed items are never redone;
interrupted items keep their directory (renamed `*.interrupted-N`) before retry.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, indent=2, sort_keys=False, default=str)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + fsync + rename, so readers never see partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, _dump(obj) + "\n")


def write_once_json(path: Path, obj: Any) -> None:
    """Immutable inputs: refuse to overwrite with different content."""
    text = _dump(obj) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to overwrite immutable file with different content: {path}")
        return
    atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


class JsonlAppender:
    """Append-only JSONL. Each line is flushed and fsynced before returning."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, obj: Any) -> None:
        line = obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj, default=str)
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, tolerating a torn final line from an interrupted writer."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn tail from an interrupted append
            raise
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path, exclude: tuple[str, ...] = ("__pycache__", ".DS_Store")) -> str:
    """Hash relative paths + file bytes + exec bit, in sorted order."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if any(part in exclude for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            h.update(f"L {rel} -> {os.readlink(p)}\n".encode())
        elif p.is_file():
            mode = "x" if os.access(p, os.X_OK) else "-"
            h.update(f"F {rel} {mode} {sha256_file(p)}\n".encode())
        elif p.is_dir():
            h.update(f"D {rel}\n".encode())
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Locking
# --------------------------------------------------------------------------- #


class RunLockedError(RuntimeError):
    pass


@contextlib.contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Exclusive, non-blocking lock so two coordinators never mutate one run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".lock"
    f = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            f.seek(0)
            holder = f.read().strip()
            raise RunLockedError(f"{run_dir} is locked by another coordinator ({holder or 'unknown'})") from e
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "since": now_iso()}))
        f.flush()
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


# --------------------------------------------------------------------------- #
# Stage manifests
# --------------------------------------------------------------------------- #

ItemStatus = Literal["pending", "running", "done", "failed", "infra_failed"]


class WorkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str
    status: ItemStatus = "pending"
    attempts: int = 0  # executions started (infra retries included)
    infra_failures: int = 0
    output: str | None = None  # relative path of the item's output
    error: str | None = None
    interrupted_dirs: list[str] = Field(default_factory=list)
    updated_at: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class StageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    stage: str
    cycle: int | None
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    items: dict[str, WorkItem] = Field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    notes: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    active_intervals: list[list[str]] = Field(default_factory=list)  # [start, end] per execution (resume-safe durations)

    @classmethod
    def load_or_create(cls, path: Path, stage: str, cycle: int | None) -> "StageManifest":
        if path.exists():
            return cls.model_validate(read_json(path))
        return cls(stage=stage, cycle=cycle)

    def save(self, path: Path) -> None:
        atomic_write_json(path, self)

    def ensure_items(self, item_ids: list[str], meta: dict[str, dict[str, Any]] | None = None) -> None:
        for iid in item_ids:
            if iid not in self.items:
                self.items[iid] = WorkItem(item_id=iid, meta=(meta or {}).get(iid, {}))

    def pending(self) -> list[WorkItem]:
        """Items still to execute. `infra_failed` is terminal: its retry budget is spent, and
        re-running it on resume would silently change denominators after later stages froze."""
        return [w for w in self.items.values() if w.status not in ("done", "failed", "infra_failed")]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for w in self.items.values():
            out[w.status] = out.get(w.status, 0) + 1
        return out


def preserve_interrupted(item_dir: Path, item: WorkItem) -> None:
    """Keep artifacts of an interrupted/failed execution instead of overwriting them."""
    if not item_dir.exists():
        return
    n = len(item.interrupted_dirs) + 1
    target = item_dir.with_name(f"{item_dir.name}.interrupted-{n}")
    while target.exists():
        n += 1
        target = item_dir.with_name(f"{item_dir.name}.interrupted-{n}")
    item_dir.rename(target)
    item.interrupted_dirs.append(target.name)
