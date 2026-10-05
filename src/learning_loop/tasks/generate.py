"""Generation: (family, difficulty, seed) -> a checked TaskSpec.

Every instance is checked here, in-process, before it can be rendered. The checks grade
Python predictions with the same grader code the verifier runs (`runtime/grade.py`):

- the oracle's predicted artifacts reach the success threshold (otherwise the family is
  wrong: `GenerationError`);
- every shortcut's predicted artifacts stay below it, and so does doing nothing (the
  initial files as they are). The predicted rewards are recorded (`oracle_reward`,
  `nop_reward`, `shortcut_rewards`) for the family's Docker test to compare against.

A family signals an unusable draw (a tied answer, a shortcut that happens to be right) by
raising `Reject`, from `build` or from a solution model; a shortcut or no-op that reaches
the threshold also counts as a rejected draw. Generation then draws again from the same
random stream, up to `MAX_DRAWS` times, so the result is still a pure function of
(family, rng_version, difficulty, seed).
"""

from __future__ import annotations

import posixpath
import random
from dataclasses import dataclass
from typing import Any

from .runtime import grade as grade_runtime
from .spec import WORKDIR, Artifacts, Family, Files, GenContext, Reject, TaskSpec

MAX_DRAWS = 100


class GenerationError(RuntimeError):
    """The family cannot produce a valid instance (a bug in the family, not a bad draw)."""


@dataclass(frozen=True)
class Generated:
    spec: TaskSpec
    draws: int
    oracle_reward: float
    nop_reward: float
    shortcut_rewards: dict[str, float]


def rng_for(family: Family, difficulty: str, seed: int) -> random.Random:
    return random.Random(f"{family.name}|v{family.rng_version}|{difficulty}|{seed}")


def _bytes(v: str | bytes) -> bytes:
    return v.encode() if isinstance(v, str) else v


def file_bytes(spec: TaskSpec) -> dict[str, bytes]:
    return {k: _bytes(v) for k, v in spec.files.items()}


def _reader(files: Files, artifacts: Artifacts | None):
    """`read(path)` over the container state: the initial files under /app, overlaid with
    the artifacts a solution wrote."""
    state = {posixpath.join(WORKDIR, rel): content for rel, content in files.items()}
    for path, content in (artifacts or {}).items():
        state[path] = _bytes(content)
    return lambda path: state.get(path)


def grade_in_process(spec: TaskSpec, files: Files, artifacts: Artifacts | None) -> float:
    reward, _ = grade_runtime.grade(spec.grader.key(), _reader(files, artifacts), trusted=True)
    return reward


def check_spec(family: Family, spec: TaskSpec) -> tuple[float, float, dict[str, float]]:
    """(oracle reward, nop reward, shortcut rewards). Raises Reject for a bad draw, GenerationError for a bad family."""
    files = file_bytes(spec)
    threshold = family.success_threshold
    for path in spec.grader.artifacts:
        if not path.startswith(WORKDIR + "/"):
            raise GenerationError(f"{family.name}: artifact {path!r} is outside {WORKDIR}")
    oracle = grade_in_process(spec, files, spec.oracle.model(files))
    if oracle < threshold:
        raise GenerationError(f"{family.name}: the oracle model scores {oracle} < {threshold}")
    nop = grade_in_process(spec, files, None)
    if nop >= threshold:
        raise Reject("doing nothing reaches the success threshold")
    shortcuts = {}
    for name, sol in sorted(spec.shortcuts.items()):
        shortcuts[name] = grade_in_process(spec, files, sol.model(files))
        if shortcuts[name] >= threshold:
            raise Reject(f"shortcut {name!r} reaches the success threshold")
    return oracle, nop, shortcuts


def generate_spec(family: Family, difficulty: str, seed: int) -> Generated:
    if difficulty not in family.difficulties:
        raise KeyError(f"{family.name}: unknown difficulty {difficulty!r} (have {sorted(family.difficulties)})")
    rng = rng_for(family, difficulty, seed)
    last: Exception | None = None
    for draw in range(1, MAX_DRAWS + 1):
        ctx = GenContext(family=family.name, difficulty=difficulty, seed=seed, params=family.difficulties[difficulty], rng=rng)
        try:
            spec = family.build(ctx)
            oracle, nop, shortcuts = check_spec(family, spec)
        except Reject as e:
            last = e
            continue
        return Generated(spec=spec, draws=draw, oracle_reward=oracle, nop_reward=nop, shortcut_rewards=shortcuts)
    raise GenerationError(f"{family.name}/{difficulty}/seed {seed}: no valid instance in {MAX_DRAWS} draws (last: {last})")


def describe(generated: Generated) -> dict[str, Any]:
    return {"draws": generated.draws, "oracle_reward": round(generated.oracle_reward, 6), "nop_reward": round(generated.nop_reward, 6), "shortcut_rewards": {k: round(v, 6) for k, v in generated.shortcut_rewards.items()}}
