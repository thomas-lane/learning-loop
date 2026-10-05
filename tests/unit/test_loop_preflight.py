"""The Docker-host preflight at the start of a run: one oracle per family must score its
predicted reward, it overlaps the first server start, and a failure stops the run."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from learning_loop.core.records import TaskInstance
from learning_loop.orchestration.preflight import PreflightError, preflight_instances, preflight_while_serving, run_preflight


def _ctx(tmp_path, backend="harbor_docker"):
    insts = {
        f"{fam}/{d}/s{s}": TaskInstance(instance_id=f"{fam}/{d}/s{s}", family=fam, task_dir=str(tmp_path / f"{fam}-{d}-{s}"), content_hash="x", params={"oracle_reward": 1.0})
        for fam in ("count-errors", "fix-stats")
        for d, s in (("easy", 2), ("easy", 1), ("hard", 1))
    }
    return SimpleNamespace(run_dir=tmp_path / "run", machine=SimpleNamespace(environment_backend=backend), instances=insts, backend_concurrency=lambda: 4)


def _records(ctx):
    return [json.loads(line) for line in (ctx.run_dir / "logs" / "preflight.jsonl").read_text().splitlines()]


def _trial(rewards, delay=0.0):
    calls = []

    async def trial(task_dir, trials_dir):
        calls.append(task_dir.name)
        await asyncio.sleep(delay)
        return {"reward": rewards.get(task_dir.name, 1.0), "error": None, "verifier_mode": "separate", "trial_dir": str(trials_dir)}

    return trial, calls


def test_one_instance_per_family_with_the_lowest_id(tmp_path):
    picks = preflight_instances(_ctx(tmp_path).instances)
    assert {f: i.instance_id for f, i in picks.items()} == {"count-errors": "count-errors/easy/s1", "fix-stats": "fix-stats/easy/s1"}


async def test_passing_preflight_runs_one_oracle_per_family_and_records_it(tmp_path):
    ctx = _ctx(tmp_path)
    trial, calls = _trial({})
    rec = await run_preflight(ctx, lambda m: None, trial)
    assert sorted(calls) == ["count-errors-easy-1", "fix-stats-easy-1"]
    assert rec["status"] == "passed" and _records(ctx)[-1]["status"] == "passed"


async def test_a_reward_other_than_predicted_stops_the_run(tmp_path):
    ctx = _ctx(tmp_path)
    trial, _ = _trial({"fix-stats-easy-1": 0.5})
    with pytest.raises(PreflightError, match=r"fix-stats \(fix-stats/easy/s1\): reward 0.5, predicted 1.0"):
        await run_preflight(ctx, lambda m: None, trial)
    rec = _records(ctx)[-1]
    assert rec["status"] == "failed" and [r["ok"] for r in rec["results"]] == [True, False]


async def test_local_fixture_runs_record_a_skip(tmp_path):
    ctx = _ctx(tmp_path, backend="local_fixture")
    trial, calls = _trial({})
    assert (await run_preflight(ctx, lambda m: None, trial))["status"] == "skipped" and calls == []
    assert _records(ctx)[-1]["status"] == "skipped"


async def test_preflight_overlaps_the_first_server_start(tmp_path):
    trial, _ = _trial({}, delay=0.4)
    t0 = time.monotonic()
    await preflight_while_serving(_ctx(tmp_path), lambda m: None, warm=lambda: time.sleep(0.4), trial=trial)
    assert time.monotonic() - t0 < 0.7  # not 0.8: the blocking server start ran in a worker thread


async def test_a_failing_server_start_cancels_the_preflight(tmp_path):
    trial, _ = _trial({}, delay=5)

    def warm():
        raise RuntimeError("server did not become healthy")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="healthy"):
        await preflight_while_serving(_ctx(tmp_path), lambda m: None, warm=warm, trial=trial)
    assert time.monotonic() - t0 < 2
