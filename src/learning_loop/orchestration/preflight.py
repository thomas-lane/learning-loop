"""Docker-host preflight: the reference solution of one task per family, before any episode.

Generation proves each instance correct in-process, and each family's Docker test proves its
solution models right, but both ran on some other machine. The preflight checks the machine
this run actually uses: for every family in the run's panels, the oracle of one instance (the
lowest instance id) runs as a Harbor trial on this Docker host and must score the reward
generation predicted (`oracle_reward` in the instance's params), graded by the separate
verifier after its environment probe and file digest pass. A host whose Docker, kernel or
build cache behaves differently therefore stops the run before any GPU time is spent,
instead of surfacing later as failed or, worse, silently mis-graded episodes.

It costs one short trial per family (a few seconds each, run `backend_concurrency()` at a
time) and runs while the first model server starts (`preflight_while_serving`), so it adds no
wall time when a server has to start. Every coordinator start that runs episodes writes one
record to `logs/preflight.jsonl` (also when skipped: the local fixture backend runs no
containers).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..core.records import TaskInstance
from ..core.storage import JsonlAppender, now_iso

if TYPE_CHECKING:
    from .coordinator import RunContext

TrialFn = Callable[[Path, Path], Awaitable[dict[str, Any]]]


class PreflightError(RuntimeError):
    """The Docker host failed the preflight; the run did not start any episode."""


async def harbor_oracle_trial(task_dir: Path, trials_dir: Path) -> dict[str, Any]:
    """Run Harbor's oracle agent on one task: {"reward", "error", "verifier_mode", "trial_dir"}."""
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    cfg = TrialConfig(task=TaskConfig(path=task_dir), trials_dir=trials_dir, agent=AgentConfig(name="oracle"))
    try:
        r = await (await Trial.create(cfg)).run()
    except Exception as e:  # the trial could not run at all
        return {"reward": None, "error": f"{type(e).__name__}: {e}", "verifier_mode": None, "trial_dir": None}
    rewards = r.verifier_result.rewards if r.verifier_result is not None else None
    err = f"{r.exception_info.exception_type}: {r.exception_info.exception_message}" if r.exception_info is not None else None
    return {
        "reward": (rewards or {}).get("reward"),
        "error": err,
        "verifier_mode": r.verifier_environment_mode.value if r.verifier_environment_mode else None,
        "trial_dir": r.trial_uri.removeprefix("file://") if getattr(r, "trial_uri", None) else None,
    }


def preflight_instances(instances: dict[str, TaskInstance]) -> dict[str, TaskInstance]:
    """Family -> the instance with the lowest id among the run's instances."""
    out: dict[str, TaskInstance] = {}
    for iid in sorted(instances):
        out.setdefault(instances[iid].family, instances[iid])
    return out


async def run_preflight(ctx: "RunContext", log: Callable[[str], None], trial: TrialFn = harbor_oracle_trial) -> dict[str, Any]:
    record: dict[str, Any] = {"started_at": now_iso(), "backend": ctx.machine.environment_backend}
    appender = JsonlAppender(ctx.run_dir / "logs" / "preflight.jsonl")
    if ctx.machine.environment_backend != "harbor_docker":
        record |= {"status": "skipped", "reason": "the local fixture backend runs no containers", "finished_at": now_iso()}
        appender.append(record)
        return record
    picks = preflight_instances(ctx.instances)
    trials_dir = ctx.run_dir / "preflight" / record["started_at"].replace(":", "")
    gate = asyncio.Semaphore(ctx.backend_concurrency())
    log(f"preflight: oracle of one instance per family on this Docker host ({len(picks)} {'trial' if len(picks) == 1 else 'trials'})")

    async def one(family: str, inst: TaskInstance) -> dict[str, Any]:
        expected = inst.params.get("oracle_reward")
        async with gate:
            r = await trial(Path(inst.task_dir), trials_dir)
        problems = []
        if r["error"]:
            problems.append(r["error"])
        if expected is None:
            problems.append("the instance records no predicted oracle_reward (not rendered by tasks.render)")
        elif r["reward"] is None or abs(r["reward"] - expected) > 1e-6:
            problems.append(f"reward {r['reward']}, predicted {expected}")
        if not r["error"] and r["verifier_mode"] != "separate":
            problems.append(f"verifier mode {r['verifier_mode']!r}, expected 'separate'")
        return {"family": family, "instance": inst.instance_id, "expected": expected, **r, "ok": not problems, "problems": problems}

    results = await asyncio.gather(*(one(f, i) for f, i in sorted(picks.items())))
    failed = [r for r in results if not r["ok"]]
    record |= {"status": "failed" if failed else "passed", "results": results, "finished_at": now_iso()}
    appender.append(record)
    if failed:
        lines = [f"  {r['family']} ({r['instance']}): {'; '.join(r['problems'])} [{r['trial_dir']}]" for r in failed]
        raise PreflightError(
            "the Docker host failed the preflight; no episode was started (details: logs/preflight.jsonl):\n" + "\n".join(lines)
        )
    log(f"preflight: passed ({len(results)} families)")
    return record


async def preflight_while_serving(
    ctx: "RunContext",
    log: Callable[[str], None],
    warm: Callable[[], None] | None = None,
    trial: TrialFn = harbor_oracle_trial,
) -> dict[str, Any]:
    """Run the preflight while `warm` (e.g. starting the first model server, which blocks)
    runs in a worker thread; return once both are done. Either failing stops the run."""
    task = asyncio.create_task(run_preflight(ctx, log, trial))
    try:
        if warm is not None:
            await asyncio.to_thread(warm)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    return await task
