"""FixtureTrainer: lineage, continued weights, reference transitions, immutable publication, resume, CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from learning_loop.config import TrainingConfig
from learning_loop.interfaces import TrainRequest
from learning_loop.records import CheckpointRecord
from learning_loop.storage import read_json
from learning_loop.training import FixtureTrainer, NoTrainableExamples, TrainingRequestError, base_checkpoint_ref
from learning_loop.training.fixture import FIXTURE_ADAPTER, SimulatedInterruption, _weights_sha, load_fixture_weights

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "train" / "fixture_prefs"


def request(tmp: Path, incoming, cycle: int = 1, steps: int = 3, dataset: Path = FIX, seed: int = 7, out: str = "ckpts") -> TrainRequest:
    tc = TrainingConfig(trainer="fixture", optimizer_steps=steps)
    return TrainRequest(
        run_id="run-a", cycle=cycle, dataset_dir=str(dataset), incoming=incoming, model_profile="qwen3-0.6b",
        training_config=tc.model_dump(), seed=seed, output_root=str(tmp / out), device="cpu",
    )


def test_two_cycles_continue_weights_and_move_reference(tmp_path):
    base = base_checkpoint_ref("qwen3-0.6b")
    r1 = FixtureTrainer().train(request(tmp_path, base, cycle=1), tmp_path / "w1")
    r2 = FixtureTrainer().train(request(tmp_path, r1.checkpoint, cycle=2), tmp_path / "w2")

    assert r1.reference_checkpoint_id == "base" and r1.checkpoint.parent_checkpoint_id == "base"
    assert r2.reference_checkpoint_id == r1.checkpoint.checkpoint_id == r2.checkpoint.parent_checkpoint_id
    assert r1.checkpoint.checkpoint_id != r2.checkpoint.checkpoint_id
    assert r1.optimizer_steps == 3 and r1.trainer == "fixture"
    # cycle 2 started from exactly cycle 1's weights (continuation, not re-init)
    w1 = load_fixture_weights(r1.checkpoint)
    assert r2.metrics["initial_weights_sha256"] == _weights_sha(w1)
    assert r1.metrics["initial_weights_sha256"] == _weights_sha([0.0] * 8)
    assert load_fixture_weights(r2.checkpoint) != w1
    payload = read_json(Path(r2.checkpoint.adapter_path) / FIXTURE_ADAPTER)
    assert payload["kind"] == "fixture_pseudo_adapter" and "FIXTURE" in payload["label"]
    assert r2.load_check["ok"] is True
    # no staging dirs left behind
    assert not [p for p in (tmp_path / "ckpts").iterdir() if p.name.startswith(".tmp-")]


def test_checkpoint_id_is_content_derived_and_publication_immutable(tmp_path):
    base = base_checkpoint_ref("qwen3-0.6b")
    req = request(tmp_path, base)
    r1 = FixtureTrainer().train(req, tmp_path / "w1")
    cj = Path(r1.checkpoint.adapter_path) / "checkpoint.json"
    mtime = cj.stat().st_mtime_ns
    again = FixtureTrainer().train(req, tmp_path / "w-other")  # idempotent: returns the published record
    assert again == r1 and cj.stat().st_mtime_ns == mtime
    other_seed = FixtureTrainer().train(request(tmp_path, base, seed=8), tmp_path / "w3")
    assert other_seed.checkpoint.checkpoint_id != r1.checkpoint.checkpoint_id
    # a foreign directory at the target id is never overwritten
    req2 = request(tmp_path, base, out="ckpts2")
    target = Path(req2.output_root) / r1.checkpoint.checkpoint_id
    target.mkdir(parents=True)
    (target / "junk").write_text("x")
    with pytest.raises(FileExistsError):
        FixtureTrainer().train(req2, tmp_path / "w4")
    assert (target / "junk").read_text() == "x"


def test_resume_continues_from_stage_state(tmp_path):
    base = base_checkpoint_ref("qwen3-0.6b")
    req = request(tmp_path, base, steps=4)
    with pytest.raises(SimulatedInterruption):
        FixtureTrainer(_test_interrupt_after_step=2).train(req, tmp_path / "w")
    assert not (Path(req.output_root)).exists() or not any(Path(req.output_root).iterdir())
    resumed = FixtureTrainer().train(req, tmp_path / "w")
    assert resumed.metrics["resumed_from_step"] == 2 and resumed.optimizer_steps == 4
    fresh = FixtureTrainer().train(request(tmp_path, base, steps=4, out="ckpts-fresh"), tmp_path / "w-fresh")
    assert load_fixture_weights(resumed.checkpoint) == load_fixture_weights(fresh.checkpoint)
    # a work dir belonging to another request is refused
    with pytest.raises(TrainingRequestError):
        FixtureTrainer().train(request(tmp_path, base, steps=4, seed=99, out="x"), tmp_path / "w")


def test_invalid_inputs(tmp_path):
    base = base_checkpoint_ref("qwen3-0.6b")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "preferences.jsonl").write_text("")
    with pytest.raises(NoTrainableExamples):
        FixtureTrainer().train(request(tmp_path, base, dataset=empty), tmp_path / "w")
    r = request(tmp_path, base)
    r.training_config["trainer"] = "trl_dpo"
    with pytest.raises(TrainingRequestError):
        FixtureTrainer().train(r, tmp_path / "w2")
    # tampered incoming adapter is detected by its recorded hash
    r1 = FixtureTrainer().train(request(tmp_path, base), tmp_path / "w3")
    copy = tmp_path / "tampered"
    shutil.copytree(r1.checkpoint.adapter_path, copy)
    p = copy / FIXTURE_ADAPTER
    assert not os.access(Path(r1.checkpoint.adapter_path) / FIXTURE_ADAPTER, os.W_OK)  # published files are read-only
    p.chmod(0o644)
    d = json.loads(p.read_text())
    d["weights"][0] += 1
    p.write_text(json.dumps(d))
    bad_ref = r1.checkpoint.model_copy(update={"adapter_path": str(copy)})
    with pytest.raises(TrainingRequestError):
        FixtureTrainer().train(request(tmp_path, bad_ref, cycle=2), tmp_path / "w5")


def _run_cli(req: TrainRequest, tmp: Path, name: str) -> subprocess.CompletedProcess:
    rp = tmp / f"{name}.json"
    rp.write_text(req.model_dump_json())
    return subprocess.run(
        [sys.executable, "-m", "learning_loop.training.run", "--request", str(rp), "--work-dir", str(tmp / f"work-{name}")],
        capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def test_cli_prints_only_the_checkpoint_path(tmp_path):
    base = base_checkpoint_ref("qwen3-0.6b")
    out = _run_cli(request(tmp_path, base), tmp_path, "ok")
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert len(lines) == 1 and lines[0].endswith("checkpoint.json")
    rec = CheckpointRecord.model_validate(read_json(Path(lines[0])))
    assert rec.trainer == "fixture"
    assert read_json(tmp_path / "work-ok" / "result.json")["status"] == "published"

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "preferences.jsonl").write_text("")
    out = _run_cli(request(tmp_path, base, dataset=empty), tmp_path, "empty")
    assert out.returncode == 3 and out.stdout == ""
    assert read_json(tmp_path / "work-empty" / "result.json")["status"] == "no_trainable_examples"


def test_cli_refuses_a_work_dir_held_by_another_trainer(tmp_path):
    import os as _os

    from learning_loop.training.run import EXIT_WORK_DIR_LOCKED, WorkDirLocked, lock_work_dir

    base = base_checkpoint_ref("qwen3-0.6b")
    work = tmp_path / "work-locked"
    work.mkdir()
    fd = lock_work_dir(work)  # stands in for a live trainer process
    try:
        with pytest.raises(WorkDirLocked, match=f"pid {_os.getpid()}"):
            lock_work_dir(work)  # a second open file description conflicts even in-process
        out = _run_cli(request(tmp_path, base), tmp_path, "locked")
        assert out.returncode == EXIT_WORK_DIR_LOCKED == 4
        assert out.stdout == "" and "work dir locked" in out.stderr
        assert not (work / "result.json").exists() and not (work / "request.json").exists()  # nothing touched
        assert not (tmp_path / "ckpts").exists()
    finally:
        _os.close(fd)
    out = _run_cli(request(tmp_path, base), tmp_path, "locked")  # lock released (holder gone) -> runs
    assert out.returncode == 0, out.stderr
    assert read_json(work / "result.json")["status"] == "published"
