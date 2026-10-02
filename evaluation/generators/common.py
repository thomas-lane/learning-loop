"""Helpers shared by the task generators.

Every generator writes a complete Harbor task directory:

    task.toml        Harbor config + [metadata.learning_loop] replay contract
    instruction.md   learner-visible instruction
    environment/     Dockerfile + learner-visible files (files/ for the local fixture backend)
    tests/           hidden grader, with its own Dockerfile (separate verifier container)
    solution/        reference solution (oracle)

Output must be byte-for-byte deterministic for (family, version, difficulty,
seed): no timestamps, sorted iteration, gzip mtime=0, `random.Random` seeded
from a string.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

LS_MTIME_NORMALIZER = {
    "pattern": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [ \d]\d (?:\d\d:\d\d| \d{4})\b",
    "replacement": "<mtime>",
    "reason": "ls -l shows modification times of files written during the episode; they depend on wall-clock time",
}

PYCACHE_EXCLUDES = ["*/__pycache__"]

# Base images pinned by (multi-arch index) digest, so an image is identified by its build
# context. Bump every generator VERSION when these change.
PYTHON_BASE = "python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217"
UBUNTU_BASE = "ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3"


def rng_for(family: str, version: int, difficulty: str, seed: int) -> random.Random:
    """`version` is the generator's RNG_VERSION (the random stream), not its output VERSION."""
    return random.Random(f"{family}|v{version}|{difficulty}|{seed}")


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755 if executable else 0o644)


def write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz:
        gz.write(text.encode())
    path.chmod(0o644)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def task_toml(
    *,
    family: str,
    generator: str | None,
    generator_seed: int | None,
    difficulty: str,
    category: str,
    tags: list[str],
    skills: list[str],
    artifacts: list[str],
    fingerprint_exclude: list[str] | None = None,
    normalizers: list[dict[str, str]] | None = None,
    params: dict[str, Any] | None = None,
    success_threshold: float = 1.0,
    agent_timeout_sec: float = 300.0,
    comment: str | None = None,
) -> str:
    lines = []
    if comment:
        lines += [f"# {c}" if c else "#" for c in comment.splitlines()]
    lines += [
        'schema_version = "1.4"',
        "# Separate verifier: only these files are copied from the agent's container into a",
        "# fresh grading container built from tests/ (hidden tests never enter the agent's container).",
        f"artifacts = {_toml_value(artifacts)}",
        "",
        "[metadata]",
        f"difficulty = {_toml_value(difficulty)}",
        f"category = {_toml_value(category)}",
        f"tags = {_toml_value(tags)}",
        f"family = {_toml_value(family)}",
        f"skills = {_toml_value(skills)}",
    ]
    if generator:
        lines.append(f"generator = {_toml_value(generator)}")
    if generator_seed is not None:
        lines.append(f"generator_seed = {generator_seed}")
    if params:
        lines.append(f"params_json = {_toml_value(json.dumps(params, sort_keys=True))}")
    lines += [
        "",
        "# Replay contract used by the learning loop (see evaluation/README.md).",
        "[metadata.learning_loop]",
        'restore = "deterministic_replay"',
        'fingerprint_paths = ["/app"]',
        f"fingerprint_exclude = {_toml_value(fingerprint_exclude or [])}",
        f"success_threshold = {success_threshold}",
        'reward_key = "reward"',
        f"observation_normalizers = {_toml_value(normalizers if normalizers is not None else [LS_MTIME_NORMALIZER])}",
        "",
        "[agent]",
        f"timeout_sec = {agent_timeout_sec}",
        "",
        "[verifier]",
        "timeout_sec = 60.0",
        'environment_mode = "separate"',
        "",
        "[environment]",
        "build_timeout_sec = 300.0",
        "cpus = 1",
        "memory_mb = 1024",
        "",
    ]
    return "\n".join(lines)
