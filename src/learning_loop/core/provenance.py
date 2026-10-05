"""Secret-free provenance bundle saved with every run.

Records code identity (git revision + dirty-tree fingerprint, or a source-tree
hash when the checkout is not a git repository), dependency and runtime
versions, OS/hardware, and the dependency lockfile hash. Environment variables
are never copied wholesale.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .storage import sha256_file, sha256_tree

PACKAGES = ["harbor", "openai", "pydantic", "transformers", "torch", "peft", "trl", "accelerate", "datasets"]
SOURCE_DIRS = ["src/learning_loop", "evaluation/agents", "evaluation/families", "prompts", "configs/models"]


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 10) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    tree = {d: sha256_tree(root / d) for d in SOURCE_DIRS if (root / d).exists()}
    ident: dict[str, Any] = {"source_tree_sha256": tree}
    rev = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if rev is None:
        ident["git"] = None
        ident["note"] = "not a git checkout; identity is the source-tree hashes"
        return ident
    status = _run(["git", "status", "--porcelain"], cwd=root) or ""
    diff = _run(["git", "diff", "HEAD"], cwd=root) or ""
    import hashlib

    ident["git"] = {
        "revision": rev,
        "dirty": bool(status),
        "dirty_fingerprint": hashlib.sha256((status + "\n" + diff).encode()).hexdigest() if status else None,
    }
    return ident


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_bytes"] = int(mem) if mem and mem.isdigit() else None
    else:
        try:
            meminfo = Path("/proc/meminfo").read_text().splitlines()[0]
            info["memory_bytes"] = int(meminfo.split()[1]) * 1024
        except (OSError, IndexError, ValueError):
            info["memory_bytes"] = None
    if shutil.which("nvidia-smi"):
        info["gpus"] = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    info["docker_server"] = _run(["docker", "version", "--format", "{{.Server.Version}} {{.Server.Arch}}"], timeout=15)
    return info


def collect(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    lock = REPO_ROOT / "uv.lock"
    return {
        "code": code_identity(),
        "packages": package_versions(),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "hardware": hardware(),
        "argv": sys.argv,
        **(extra or {}),
    }
