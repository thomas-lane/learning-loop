"""Bounded smoke tests (`loop smoke <level>`). All are engineering checks, never evidence of
learning efficacy.

fixture         scripted learner/editor + fixture trainer, local fixture backend, 2 cycles (seconds)
fixture-docker  the same on real Harbor Docker containers (a few minutes)
train           REAL LoRA DPO on labeled fixture preferences (Qwen3-0.6B, MPS, 2 optimizer steps),
                publish + reload check, then serve the adapter and run one real Docker task episode
live            the small real learner/editor loop (experiments/smoke-mac.yaml); zero accepted
                edits is a valid, recorded outcome
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import coordinator as co
from .config import REPO_ROOT
from .interfaces import TrainRequest
from .storage import atomic_write_json, read_json

SMOKE_ROOT = REPO_ROOT / "runs" / "_smoke"
FIXTURE_PREFS = REPO_ROOT / "tests" / "fixtures" / "train" / "fixture_prefs"
MAC = REPO_ROOT / "configs" / "machines" / "examples" / "mac-local.yaml"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _fixture_loop(machine: Path, keep: bool) -> int:
    ctx = co.create_run(REPO_ROOT / "experiments" / "fixture-two-cycles.yaml", machine, run_id=f"fixture-{_stamp()}", runs_dir=SMOKE_ROOT)
    asyncio.run(co.run_all(ctx))
    states = [read_json(co.cycle_state_path(ctx, c)) for c in range(ctx.exp.cycles + 1)]
    ok = all(s["status"] == "done" for s in states)
    for c, s in enumerate(states[:-1]):
        ok &= states[c + 1]["learner_in"] == s["learner_out"]
        if s["update"] == "trained":
            ok &= s["reference"] == s["learner_in"]["checkpoint_id"]
    print(("PASS" if ok else "FAIL") + f": fixture loop lineage/reference checks; report: {ctx.run_dir / 'reports' / 'report.md'}")
    if ok and not keep:
        shutil.rmtree(ctx.run_dir)
    return 0 if ok else 1


def _train_smoke(machine: Path, keep: bool) -> int:
    """Real DPO on FIXTURE preferences, then an actual Docker environment interaction with the
    reloaded adapter. Task success is not required (a 2-step update on fixture data)."""
    from .training import base_checkpoint_ref

    root = SMOKE_ROOT / f"train-{_stamp()}"
    root.mkdir(parents=True)
    exp_path = REPO_ROOT / "experiments" / "smoke-mac.yaml"
    exp, *_ = co.load_all(exp_path, machine)
    req = TrainRequest(
        run_id=root.name,
        cycle=0,
        dataset_dir=str(FIXTURE_PREFS),
        incoming=base_checkpoint_ref(exp.learner.model_profile),
        model_profile=exp.learner.model_profile,
        training_config=exp.training.model_dump(mode="json"),
        seed=1,
        output_root=str(root / "checkpoints"),
        device="auto",
    )
    atomic_write_json(root / "request.json", req)
    print("training (real TRL DPO, labeled fixture preferences) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "learning_loop.training.run", "--request", str(root / "request.json"), "--work-dir", str(root / "work")],
        stdout=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        print(f"FAIL: training exited {proc.returncode}; see {root / 'work'}")
        return 1
    ckpt_json = Path(proc.stdout.strip().splitlines()[-1])
    rec = read_json(ckpt_json)
    print(f"published {rec['checkpoint']['checkpoint_id']} (load check: {rec.get('load_check')})")
    print("serving the adapter and running one Docker task episode ...", flush=True)
    out = co.evaluate_checkpoint(exp_path, machine, str(ckpt_json.parent), ["dev"], run_id="eval", runs_dir=root)
    m = read_json(out / "cycles" / "cycle-000" / "eval" / "manifest.json")
    (item,) = m["items"].values()
    summary = read_json(out / item["output"] / "summary.json")
    interacted = summary["n_requests"] >= 1 and summary["stop_category"] != "infra"
    print(
        f"{'PASS' if interacted else 'FAIL'}: episode with adapter {summary['checkpoint_id']}: requests={summary['n_requests']} "
        f"tool_calls={summary['n_tool_calls']} stop={summary['stop_reason']} reward={summary['partial_reward']} (success not required)"
    )
    print(f"artifacts: {root}")
    return 0 if interacted else 1


def _live(machine: Path, keep: bool) -> int:
    ctx = co.create_run(REPO_ROOT / "experiments" / "smoke-mac.yaml", machine, run_id=f"live-{_stamp()}", runs_dir=SMOKE_ROOT)
    asyncio.run(co.run_all(ctx))
    c0 = read_json(co.cycle_state_path(ctx, 0))
    print(f"cycle 0 update: {c0['update']} (dataset pairs: {c0.get('dataset', {}).get('n_examples')}); report: {ctx.run_dir / 'reports' / 'report.md'}")
    return 0


def run_smoke(level: str, machines: str | None = None, keep: bool = False) -> int:
    if level == "fixture":
        return _fixture_loop(Path(machines) if machines else REPO_ROOT / "configs/machines/examples/fixture.yaml", keep)
    if level == "fixture-docker":
        return _fixture_loop(Path(machines) if machines else REPO_ROOT / "configs/machines/examples/fixture-docker.yaml", keep)
    if level == "train":
        return _train_smoke(Path(machines) if machines else MAC, keep)
    if level == "live":
        return _live(Path(machines) if machines else MAC, keep)
    raise ValueError(level)
