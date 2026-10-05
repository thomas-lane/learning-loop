"""Shared checks for rendered task families in real Docker (used by the `docker` tests).

`check_predictions` is the per-family Docker check: every solution's shell form (oracle,
each shortcut) and doing nothing must score exactly what generation predicted from the
Python models, with the environment probe passing in the verifier.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from learning_loop.tasks.render import render
from learning_loop.tasks.spec import Family


async def trial(task_dir: Path, agent: str, trials: Path):
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    cfg = TrialConfig(task=TaskConfig(path=task_dir), trials_dir=trials, agent=AgentConfig(name=agent))
    return await (await Trial.create(cfg)).run()


def verifier_stdout(result) -> str:
    return "\n".join(p.read_text() for p in Path(result.trial_uri.removeprefix("file://")).rglob("test-stdout.txt"))


def _with_solution(task_dir: Path, script: Path, dest: Path) -> Path:
    shutil.copytree(task_dir, dest)
    shutil.copy(script, dest / "solution" / "solve.sh")
    return dest


async def _one(label: str, task_dir: Path, agent: str, expected: float, trials: Path) -> list[str]:
    r = await trial(task_dir, agent, trials)
    if r.exception_info is not None:
        return [f"{label}: exception {r.exception_info}"]
    problems = []
    got = (r.verifier_result.rewards or {}).get("reward") if r.verifier_result else None
    if got is None or abs(got - expected) > 1e-6:
        problems.append(f"{label}: reward {got}, predicted {expected}")
    if r.verifier_environment_mode.value != "separate":
        problems.append(f"{label}: verifier not separate")
    out = verifier_stdout(r)
    if "verifier files match the rendered task" not in out:
        problems.append(f"{label}: verifier did not confirm the rendered digest")
    probe = [ln for ln in out.splitlines() if ln.startswith("probe: ")]
    if not probe or json.loads(probe[0][len("probe: "):])["ok"] is not True:
        problems.append(f"{label}: probe {probe}")
    return problems


async def check_predictions(fam: Family, difficulty: str, seed: int, root: Path, gate: asyncio.Semaphore | None = None) -> list[str]:
    """Problems (empty when every shell solution scored its predicted reward)."""
    # Harbor names a task's image after its directory, so every directory name is unique.
    base = f"{fam.name}__{difficulty}__s{seed}"
    task = root / base
    params = render(fam, difficulty, seed, task)
    runs = [("oracle", task, "oracle", params["oracle_reward"]), ("nop", task, "nop", params["nop_reward"])]
    for name, predicted in params["shortcut_rewards"].items():
        runs.append((f"shortcut:{name}", _with_solution(task, task / "solution" / "shortcuts" / f"{name}.sh", root / f"{base}__{name}"), "oracle", predicted))
    gate = gate or asyncio.Semaphore(1)

    async def guarded(label, task_dir, agent, expected):
        async with gate:
            return await _one(f"{fam.name}/{difficulty}/s{seed} {label}", task_dir, agent, expected, root / "trials")

    results = await asyncio.gather(*(guarded(*r) for r in runs))
    return [p for ps in results for p in ps]
