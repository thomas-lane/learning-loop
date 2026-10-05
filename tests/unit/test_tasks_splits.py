"""Split/panel files: shipped files valid; leakage and overlap rejected."""

from __future__ import annotations

from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.core.records import Split
from learning_loop.tasks.instances import (
    SplitValidationError,
    assert_exportable,
    load_splits,
    materialize,
    validate_splits,
)

SPLITS = REPO_ROOT / "evaluation" / "splits"


@pytest.mark.parametrize("name", ["pilot", "smoke", "fixture"])
def test_shipped_split_files_are_valid(tmp_path, name):
    s = load_splits(SPLITS / f"{name}.yaml")
    inst = materialize(s, tmp_path)
    validate_splits(s, inst)
    assert set(inst) == set(s.instances)
    for iid, ti in inst.items():
        assert Path(ti.task_dir).is_dir() and len(ti.content_hash) == 64 and ti.public_content_hash
        assert ti.family == s.instances[iid].family
    assert (tmp_path / "instances.json").exists()


def test_pilot_layout():
    s = load_splits(SPLITS / "pilot.yaml")
    assert s.panels["train"].split == Split.TRAIN and s.panels["dev"].split == Split.DEV
    assert {i.difficulty for i in s.panels["train-easy-only"].instances} == {"easy"}
    assert {i.difficulty for i in s.panels["dev-medium-hard"].instances} == {"medium", "hard"}
    held = {i.family for i in s.panels["final-held-out-family"].instances}
    assert held == set(s.held_out_families) == {"csv-revenue"}
    trained = {i.family for p in s.panels.values() if p.split == Split.TRAIN for i in p.instances}
    assert not trained & held
    assert {i.family for i in s.panels["final-same-family"].instances} <= trained


def test_materialize_is_stable_and_reuses_dirs(tmp_path):
    s = load_splits(SPLITS / "fixture.yaml")
    a = materialize(s, tmp_path, panels=["train"])
    b = materialize(s, tmp_path, panels=["train"])
    assert {k: v.content_hash for k, v in a.items()} == {k: v.content_hash for k, v in b.items()}
    c = materialize(s, tmp_path / "other", panels=["train"])
    assert {k: v.content_hash for k, v in a.items()} == {k: v.content_hash for k, v in c.items()}


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "s.yaml"
    p.write_text("schema_version: 1\nname: t\n" + body)
    return p


BASE = """
held_out_families: [csv-revenue]
instances:
  - {id: a, family: count-errors, difficulty: easy, seed: 1, split: train}
  - {id: b, family: count-errors, difficulty: easy, seed: 2, split: dev}
  - {id: h, family: csv-revenue, difficulty: easy, seed: 1, split: final}
"""


def test_rejects_instance_declared_twice(tmp_path):
    with pytest.raises(SplitValidationError, match="more than once"):
        load_splits(_write(tmp_path, BASE + "  - {id: a, family: count-errors, difficulty: easy, seed: 1, split: dev}\npanels: {}\n"))


def test_rejects_panel_mixing_splits(tmp_path):
    with pytest.raises(SplitValidationError, match="contains 'b' of split dev"):
        load_splits(_write(tmp_path, BASE + "panels:\n  train: {split: train, instances: [a, b]}\n"))


def test_rejects_held_out_family_in_train(tmp_path):
    body = BASE.replace("{id: h, family: csv-revenue, difficulty: easy, seed: 1, split: final}", "{id: h, family: csv-revenue, difficulty: easy, seed: 1, split: train}")
    with pytest.raises(SplitValidationError, match="held-out family 'csv-revenue'"):
        load_splits(_write(tmp_path, body + "panels:\n  train: {split: train, instances: [a, h]}\n"))


def test_rejects_same_generator_coordinates_under_two_ids(tmp_path):
    body = BASE + "  - {id: c, family: count-errors, difficulty: easy, seed: 1, split: final}\npanels: {}\n"
    with pytest.raises(SplitValidationError, match="same task"):
        load_splits(_write(tmp_path, body))


def test_rejects_unknown_panel_instance_and_generator_problems(tmp_path):
    with pytest.raises(SplitValidationError, match="unknown instance 'zz'"):
        load_splits(_write(tmp_path, BASE + "panels:\n  train: {split: train, instances: [zz]}\n"))
    with pytest.raises(SplitValidationError, match="unknown difficulty"):
        load_splits(_write(tmp_path, "instances:\n  - {id: a, family: count-errors, difficulty: brutal, seed: 1, split: train}\npanels: {}\n"))
    with pytest.raises(Exception):
        load_splits(_write(tmp_path, "instances:\n  - {id: a, family: count-errors, difficulty: easy, seed: 1, split: train, bogus: 1}\npanels: {}\n"))


def test_rejects_identical_content_across_splits(tmp_path):
    # different seeds can still produce the same learner-visible task; that must not cross splits
    body = """
instances:
  - {id: x, family: count-errors, difficulty: easy, seed: 1, split: train}
  - {id: y, family: count-errors, difficulty: easy, seed: 2, split: dev}
panels:
  train: {split: train, instances: [x]}
  dev: {split: dev, instances: [y]}
"""
    s = load_splits(_write(tmp_path, body))
    inst = materialize(s, tmp_path / "mat")
    assert inst["x"].public_content_hash != inst["y"].public_content_hash
    validate_splits(s, inst)
    same = {"x": inst["x"], "y": inst["y"].model_copy(update={"public_content_hash": inst["x"].public_content_hash})}
    with pytest.raises(SplitValidationError, match="identical learner-visible content in different splits"):
        validate_splits(s, same)


def test_export_boundary():
    s = load_splits(SPLITS / "pilot.yaml")
    assert_exportable(s, "log-triage/easy/s1")
    for bad in ("log-triage/easy/s101", "csv-revenue/easy/s301", "nope"):
        with pytest.raises(SplitValidationError):
            assert_exportable(s, bad)
    with pytest.raises(SplitValidationError, match="held out"):
        assert_exportable(s, "fix-stats/easy/s1", forbidden_families=["fix-stats"])
