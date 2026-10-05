"""Task families as Python specs.

A family module defines `FAMILY = Family(...)` whose `build(ctx)` returns a `TaskSpec`: the
whole task in one place (instruction, learner-visible files, grader and answer key, the
reference solution and the known wrong shortcuts). `tasks.generate` turns
(family, difficulty, seed) into a checked spec; `tasks.render` writes the Harbor task
directory from it. Nothing else writes task files.

Paths: `TaskSpec.files` keys are relative to the working directory `/app`; grader paths and
the artifacts a `Solution.model` returns are absolute container paths.

A model returns the files the solution leaves behind that differ from the initial files:
{absolute path: content}, where content is text, bytes, `FileState(content, mode)`, or None
for a file the solution deleted (or moved away).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

WORKDIR = "/app"

Files = Mapping[str, bytes]  # path relative to /app -> content


@dataclass(frozen=True)
class FileState:
    """A file a solution model predicts, with its mode (e.g. 0o640)."""

    content: str | bytes
    mode: int


Artifacts = Mapping[str, "str | bytes | FileState | None"]  # absolute container path -> content; None = deleted


@dataclass(frozen=True)
class Profile:
    """The container environment a family runs in. The renderer derives the agent and
    verifier images, compose overlays and probe expectation from it; a profile never
    changes after use (a change is a new name/version)."""

    name: str
    base_image: str  # pinned by digest
    env: Mapping[str, str]
    tools: tuple[str, ...]  # must be on PATH; checked by the probe
    hostname: str = "task"
    cpus: int = 1
    memory_mb: int = 1024
    allowed_processes: tuple[str, ...] = ("sh", "sleep")  # Harbor runs `sh -c "sleep infinity"` as the container command

    def probe_expectation(self) -> dict[str, Any]:
        return {
            "network": "none",
            "env": dict(self.env),
            "tools": list(self.tools),
            "hostname": self.hostname,
            "allowed_processes": list(self.allowed_processes),
        }


PYTHON_BASE = "python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217"

PROFILES: dict[str, Profile] = {
    "python@1": Profile(
        name="python@1",
        base_image=PYTHON_BASE,
        # TZ/LANG/LC_ALL fix time zone and collation (sort order); PYTHONHASHSEED fixes set and
        # dict-of-str iteration order in agent one-liners, which replay compares;
        # PYTHONDONTWRITEBYTECODE keeps mtime-stamped bytecode caches out of the task state.
        env={
            "TZ": "UTC",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": "/root",
        },
        tools=("bash", "python3", "timeout", "gzip", "zcat", "awk", "sort", "grep", "sed", "find", "tar", "diff"),
    ),
}


class Reject(Exception):
    """Raised by a family's `build` (or its solution models) when this draw is unusable,
    e.g. the answer is tied or a shortcut happens to give the right answer. Generation
    draws again from the same random stream."""


# --------------------------------------------------------------------------- #
# Grader kinds (each renders to the key that tests/grade.py reads)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExactAnswer:
    """Stripped text of `path` equals `expected`."""

    path: str
    expected: str

    def key(self) -> dict[str, Any]:
        return {"kind": "exact", "path": self.path, "expected": self.expected}

    @property
    def artifacts(self) -> list[str]:
        return [self.path]


@dataclass(frozen=True)
class NumericAnswer:
    """The number in `path` is within max(abs_tol, rel_tol * |expected|) of `expected`."""

    path: str
    expected: float
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    def key(self) -> dict[str, Any]:
        return {"kind": "numeric", "path": self.path, "expected": self.expected, "abs_tol": self.abs_tol, "rel_tol": self.rel_tol}

    @property
    def artifacts(self) -> list[str]:
        return [self.path]


@dataclass(frozen=True)
class ParsedAnswer:
    """`path` parsed as `format` equals `expected`: "json", "toml" (object key order ignored),
    "dotenv" (KEY=VALUE lines into a dict; duplicate keys fail), "lines" (non-empty stripped
    lines, in order) or "line-set" (the same, sorted)."""

    path: str
    format: str
    expected: Any

    def key(self) -> dict[str, Any]:
        return {"kind": "parsed", "path": self.path, "format": self.format, "expected": self.expected}

    @property
    def artifacts(self) -> list[str]:
        return [self.path]


@dataclass(frozen=True)
class FileTree:
    """The regular files under the directory `root` are exactly `expected`:
    {relative path: {"sha256": hex, optional "mode": "0644"}}. Reward = correct entries /
    (expected + unexpected entries). Build `expected` with `tree_entry`."""

    root: str
    expected: Mapping[str, Mapping[str, str]]

    def key(self) -> dict[str, Any]:
        return {"kind": "tree", "root": self.root, "expected": {k: dict(v) for k, v in self.expected.items()}}

    @property
    def artifacts(self) -> list[str]:
        return [self.root]


def tree_entry(content: str | bytes, mode: int | None = None) -> dict[str, str]:
    data = content.encode() if isinstance(content, str) else content
    entry = {"sha256": hashlib.sha256(data).hexdigest()}
    if mode is not None:
        entry["mode"] = f"{mode:04o}"
    return entry


@dataclass(frozen=True)
class Checks:
    """Call functions of the Python module at `path` (importable as `module`) with hidden
    inputs; the reward is the fraction of checks passed. Each check is
    `{"name", "func", "kind", "args", ...}` with kind `value` (number; "expected", optional
    "abs_tol"), `equal` (JSON-equal "expected"), `raises` (optional "exception" class name,
    default any ValueError) or `no_mutation` (the first argument is unchanged)."""

    path: str
    module: str
    checks: tuple[Mapping[str, Any], ...]

    def key(self) -> dict[str, Any]:
        return {"kind": "checks", "path": self.path, "module": self.module, "checks": [dict(c) for c in self.checks]}

    @property
    def artifacts(self) -> list[str]:
        return [self.path]


@dataclass(frozen=True)
class Commands:
    """Run commands against the agent's programs: the artifacts `files` (absolute /app paths)
    are copied to the same paths relative to a scratch directory, and each check
    `{"name", "argv", optional "stdin", "inputs" {rel: text}, "stdout", "exit", "outputs" {rel: text}}`
    runs `argv` there; the given stdout (trailing newlines ignored), exit code and output files
    must match. Reward = fraction of checks passed."""

    files: tuple[str, ...]
    checks: tuple[Mapping[str, Any], ...]
    timeout_sec: float = 10.0

    def key(self) -> dict[str, Any]:
        return {"kind": "commands", "files": list(self.files), "checks": [dict(c) for c in self.checks], "timeout_sec": self.timeout_sec, "workdir": WORKDIR}

    @property
    def artifacts(self) -> list[str]:
        return list(self.files)


Grader = ExactAnswer | NumericAnswer | ParsedAnswer | FileTree | Checks | Commands
LOCAL_FIXTURE_GRADERS = (ExactAnswer, NumericAnswer, ParsedAnswer)  # read one file; no agent code, no directories


@dataclass(frozen=True)
class Solution:
    """A way of solving the task, in two forms. `shell` runs in the container (the oracle is
    Harbor's reference solution; shortcuts are run by the family's Docker test). `model`
    predicts, in Python, the artifacts `shell` produces from the task's files; generation
    grades these predictions for every instance, and the family's Docker test checks that
    the shell form really produces them."""

    shell: str
    model: Callable[[Files], Artifacts]


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    files: Mapping[str, str | bytes]  # relative to /app
    grader: Grader
    oracle: Solution
    shortcuts: Mapping[str, Solution] = field(default_factory=dict)
    modes: Mapping[str, int] = field(default_factory=dict)  # file modes (relative to /app); others are 0o644
    params: Mapping[str, Any] = field(default_factory=dict)  # recorded in task.toml (params_json)


@dataclass
class GenContext:
    family: str
    difficulty: str
    seed: int
    params: Mapping[str, Any]  # the family's settings for this difficulty
    rng: random.Random  # the only source of randomness a family may use


@dataclass(frozen=True)
class Family:
    name: str
    version: int  # bump whenever output for an existing (difficulty, seed) changes
    cluster: str
    skills: tuple[str, ...]
    difficulties: Mapping[str, Mapping[str, Any]]
    build: Callable[[GenContext], TaskSpec]
    category: str = "general"
    profile: str = "python@1"
    rng_version: int = 1  # version of the random stream; bump only when draws change
    agent_timeout_sec: float = 300.0
    verifier_timeout_sec: float = 60.0
    success_threshold: float = 1.0
    local_fixture: bool = False  # also runnable on the local fixture backend (LOCAL_FIXTURE_GRADERS only)

    @property
    def generator_id(self) -> str:
        return f"{self.name}@v{self.version}"
