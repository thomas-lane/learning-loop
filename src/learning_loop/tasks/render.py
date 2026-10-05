"""Render a checked TaskSpec into a Harbor task directory.

This is the only code that writes task directories, so the environment invariants hold by
construction rather than by convention:

    task.toml                    Harbor config + [metadata.learning_loop] replay contract
    instruction.md
    environment/Dockerfile       FROM <pinned base>, ENV <profile>, COPY files/ -> /app; no RUN
    environment/docker-compose.yaml   network_mode "none" and the profile hostname
    environment/files/...        the learner-visible files (become /app)
    tests/Dockerfile             same base and ENV; COPY grade.py probe.py key.json test.sh
    tests/docker-compose.yaml    the same overlay for the separate verifier container
    tests/key.json               the hidden answer key (only in the verifier's build context)
    solution/solve.sh            the oracle (Harbor's reference solution)
    solution/shortcuts/<name>.sh known wrong solutions, run by the family's Docker test

Because no image runs RUN, the hash of the build context (which contains the pinned FROM)
identifies the image exactly. Every rendered file and directory gets the mtime
`FIXED_MTIME`; Docker's COPY keeps it, so `ls -l` and `stat` show the same times on any
host and on every re-render. Harbor appends a task's `docker-compose.yaml` after its own
compose files, for the agent container (context `environment/`) and for the separate
verifier (context `tests/`), which is how both get `network_mode: none`.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from .generate import Generated, describe, generate_spec
from .spec import PROFILES, WORKDIR, Family, Profile

FIXED_MTIME = 1767225600  # 2026-01-01T00:00:00Z
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_FILES = ("grade.py", "probe.py")

LS_MTIME_NORMALIZER = {
    "pattern": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [ \d]\d (?:\d\d:\d\d| \d{4})\b",
    "replacement": "<mtime>",
    "reason": "ls -l shows modification times of files written during the episode; they depend on wall-clock time",
}


def _toml(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def _write(path: Path, content: str | bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    path.chmod(mode)


def _env_lines(profile: Profile) -> str:
    return "ENV " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(profile.env.items())) + "\n"


def _header(family: Family, difficulty: str, seed: int) -> str:
    return f"Rendered by learning_loop.tasks.render from {family.generator_id} ({difficulty}, seed {seed}). Do not edit."


def _compose(profile: Profile) -> str:
    return (
        "# No network and a fixed hostname for this container (learning_loop.tasks.render).\n"
        "services:\n"
        "  main:\n"
        '    network_mode: "none"\n'
        f"    hostname: {json.dumps(profile.hostname)}\n"
    )


def task_toml(family: Family, difficulty: str, seed: int, gen: Generated, profile: Profile) -> str:
    params = dict(gen.spec.params) | describe(gen)
    lines = [
        f"# {_header(family, difficulty, seed)}",
        'schema_version = "1.4"',
        "# Separate verifier: only these files are copied from the agent's container into a",
        "# fresh grading container built from tests/.",
        f"artifacts = {_toml(gen.spec.grader.artifacts)}",
        "",
        "[metadata]",
        f"family = {_toml(family.name)}",
        f"cluster = {_toml(family.cluster)}",
        f"difficulty = {_toml(difficulty)}",
        f"category = {_toml(family.category)}",
        f"skills = {_toml(list(family.skills))}",
        f"generator = {_toml(family.generator_id)}",
        f"generator_seed = {seed}",
        f"profile = {_toml(profile.name)}",
        'network = "none"',
        f"params_json = {_toml(json.dumps(params, sort_keys=True))}",
        "",
        "# Replay contract used by the learning loop (see evaluation/README.md).",
        "[metadata.learning_loop]",
        'restore = "deterministic_replay"',
        f"fingerprint_paths = {_toml([WORKDIR])}",
        "fingerprint_exclude = []",
        f"success_threshold = {family.success_threshold}",
        'reward_key = "reward"',
        f"observation_normalizers = {_toml([LS_MTIME_NORMALIZER])}",
    ]
    if family.local_fixture:
        lines += [
            "",
            "# Local fixture backend (host subprocesses; NOT a sandbox - scripted policies only).",
            "[metadata.local_fixture]",
            'files = "environment/files"',
            'grader = "tests/grade.py"',
        ]
    lines += [
        "",
        "[agent]",
        f"timeout_sec = {family.agent_timeout_sec}",
        "",
        "[verifier]",
        f"timeout_sec = {family.verifier_timeout_sec}",
        'environment_mode = "separate"',
        "",
        "[environment]",
        "build_timeout_sec = 300.0",
        f"cpus = {profile.cpus}",
        f"memory_mb = {profile.memory_mb}",
        "",
    ]
    return "\n".join(lines)


def _fix_mtimes(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames + dirnames:
            os.utime(os.path.join(dirpath, name), (FIXED_MTIME, FIXED_MTIME), follow_symlinks=False)
    os.utime(root, (FIXED_MTIME, FIXED_MTIME))


def render_generated(family: Family, difficulty: str, seed: int, gen: Generated, out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to render into non-empty {out_dir}")
    if family.local_fixture and gen.spec.grader.key()["kind"] == "checks":
        raise ValueError(f"{family.name}: the local fixture backend cannot run the checks grader (it needs root to sandbox)")
    profile = PROFILES[family.profile]
    spec = gen.spec
    header = _header(family, difficulty, seed)
    env = out_dir / "environment"
    tests = out_dir / "tests"

    (env / "files").mkdir(parents=True, exist_ok=True)
    for rel, content in sorted(spec.files.items()):
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise ValueError(f"{family.name}: file path {rel!r} must be relative to {WORKDIR}")
        _write(env / "files" / rel, content, 0o755 if rel in spec.executables else 0o644)
    _write(env / "Dockerfile", f"# {header}\nFROM {profile.base_image}\n{_env_lines(profile)}WORKDIR {WORKDIR}\nCOPY files/ {WORKDIR}/\n")
    _write(env / "docker-compose.yaml", _compose(profile))

    for name in RUNTIME_FILES:
        _write(tests / name, (RUNTIME_DIR / name).read_bytes())
    _write(tests / "key.json", json.dumps(spec.grader.key(), sort_keys=True, indent=1) + "\n")
    probe = shlex.quote(json.dumps(profile.probe_expectation(), sort_keys=True, separators=(",", ":")))
    _write(
        tests / "test.sh",
        f"#!/bin/bash\n# {header}\n"
        f"exec python3 -I -B /tests/grade.py --root / --out /logs/verifier/reward.json --probe {probe}\n",
        0o755,
    )
    _write(tests / "Dockerfile", f"# {header}\nFROM {profile.base_image}\n{_env_lines(profile)}COPY grade.py probe.py key.json test.sh /tests/\n")
    _write(tests / "docker-compose.yaml", _compose(profile))

    _write(out_dir / "solution" / "solve.sh", "#!/bin/bash\n" + spec.oracle.shell, 0o755)
    for name, sol in sorted(spec.shortcuts.items()):
        _write(out_dir / "solution" / "shortcuts" / f"{name}.sh", "#!/bin/bash\n" + sol.shell, 0o755)
    _write(out_dir / "instruction.md", spec.instruction)
    _write(out_dir / "task.toml", task_toml(family, difficulty, seed, gen, profile))
    _fix_mtimes(out_dir)
    return dict(spec.params) | describe(gen)


def render(family: Family, difficulty: str, seed: int, out_dir: Path) -> dict[str, Any]:
    """Generate (with the per-instance checks) and render; returns the recorded params."""
    return render_generated(family, difficulty, seed, generate_spec(family, difficulty, seed), out_dir)
