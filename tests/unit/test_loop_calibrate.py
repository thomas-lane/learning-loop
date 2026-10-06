"""Calibration runs: reserved-seed instances of each family and difficulty, run through the
evaluation path, summarized per (family, difficulty) against the target range."""

from __future__ import annotations

import csv

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.core.storage import read_json
from learning_loop.orchestration import coordinator as co
from learning_loop.tasks.instances import CALIBRATION_SEEDS, calibration_ids

EXP = REPO_ROOT / "experiments" / "fixture-two-cycles.yaml"
MACHINE = REPO_ROOT / "configs" / "machines" / "examples" / "fixture.yaml"


def test_calibration_instances_use_only_reserved_seeds():
    defs = calibration_ids(["count-errors"], None, 3)
    assert [d.instance_id for d in defs][:3] == ["count-errors/easy/s900000", "count-errors/easy/s900001", "count-errors/easy/s900002"]
    assert len(defs) == 9 and all(d.seed in CALIBRATION_SEEDS for d in defs)
    with pytest.raises(ValueError, match="unknown family"):
        calibration_ids(["nope"], None, 1)
    with pytest.raises(ValueError, match="unknown difficulty"):
        calibration_ids(["count-errors"], ["brutal"], 1)


def test_calibrate_runs_and_reports_per_family_and_difficulty(tmp_path):
    run_dir = co.calibrate(EXP, MACHINE, "base", families=["count-errors"], difficulties=["easy", "medium"], instances_per_difficulty=2, attempts=2, runs_dir=tmp_path)
    meta = read_json(run_dir / "run.json")
    assert meta["kind"] == "calibration" and meta["calibration"]["attempts"] == 2
    assert all(i["generator_seed"] in CALIBRATION_SEEDS for i in meta["instances"].values())
    rows = list(csv.DictReader(open(run_dir / "reports" / "calibration.csv")))
    assert [(r["family"], r["difficulty"]) for r in rows] == [("count-errors", "easy"), ("count-errors", "medium")]
    for r in rows:
        assert int(r["episodes"]) + int(r["infra_excluded"]) == 4  # 2 instances x 2 attempts
        rate = float(r["success_rate"])
        assert r["verdict"] == ("too hard" if rate < 0.2 else "too easy" if rate > 0.8 else "in range")
    assert (run_dir / "logs" / "preflight.jsonl").exists()  # skipped on the local fixture backend, but recorded
