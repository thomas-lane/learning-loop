"""Seeds/IDs, strict config validation, storage atomicity/locking, usage arithmetic."""

import json
import subprocess
import sys
import textwrap

import pytest
import yaml
from pydantic import ValidationError

from learning_loop import seeds
from learning_loop.config import ExperimentConfig, assert_no_secrets, load_experiment, load_machine
from learning_loop.records import Usage
from learning_loop.storage import (
    JsonlAppender,
    StageManifest,
    atomic_write_json,
    preserve_interrupted,
    read_jsonl,
    run_lock,
    write_once_json,
)


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def test_seeds_are_stable_across_processes():
    code = "from learning_loop import seeds; print(seeds.derive_seed(7, 'learner_attempt', 'log-triage/easy/s1', 3))"
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": str(h), "PATH": ""}).stdout
        for h in (0, 1, 12345)
    }
    assert len(outs) == 1
    assert int(outs.pop()) == seeds.derive_seed(7, "learner_attempt", "log-triage/easy/s1", 3)


def test_seed_streams_are_distinct_and_attempts_differ():
    a = seeds.derive_seed(1, "learner_attempt", "x", 0)
    assert a != seeds.derive_seed(1, "editor_proposal", "x", 0)
    assert a != seeds.derive_seed(1, "learner_attempt", "x", 1)
    assert 0 <= a < 2**31


def test_attempt_seed_independent_of_scheduling_order():
    items = [("i1", 0), ("i1", 1), ("i2", 0)]
    fwd = {k: seeds.attempt_seed(5, *k) for k in items}
    rev = {k: seeds.attempt_seed(5, *k) for k in reversed(items)}
    assert fwd == rev


def test_continuation_seed_is_matched_across_branches_and_purposes_differ():
    s = seeds.continuation_seed(3, "prop-abc", 0)
    assert s == seeds.continuation_seed(3, "prop-abc", 0)  # no branch label input exists
    assert s != seeds.continuation_seed(3, "prop-abc", 1)
    assert s != seeds.continuation_seed(3, "prop-abc", 0, purpose="audit")


def test_request_seed_none_passthrough():
    assert seeds.request_seed(None, 0) is None
    assert seeds.request_seed(10, 0) != seeds.request_seed(10, 1)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #

MINIMAL = {
    "name": "t",
    "learner": {"model_profile": "qwen3-0.6b"},
    "cycles": 1,
    "seeds": {"root": 1},
    "tasks": {"splits": "evaluation/splits/fixture.yaml", "collection_panel": "train", "attempts_per_instance": 1},
    "evaluation": {"dev_panels": ["dev"], "attempts_per_instance": 1},
    "episode": {"sampling": {"temperature": 0.7}, "max_turns": 5},
    "editor": {"mode": "initial_policy"},
    "training": {"optimizer_steps": 2},
}


def test_minimal_experiment_validates():
    cfg = ExperimentConfig.model_validate(MINIMAL)
    assert cfg.editor.proposals_per_source == 1
    assert cfg.verification.continuations_per_branch == 1
    assert cfg.verification.acceptance_rule == "strict_all_success_v1"
    assert cfg.training.reference == "incoming_checkpoint"


@pytest.mark.parametrize(
    "patch",
    [
        {"unknown_key": 1},
        {"editor": {"mode": "initial_policy", "temprature": 0.1}},
        {"condition": "fixed_dataset"},  # missing fixed_dataset
        {"training": {"optimizer_steps": 2, "fixed_dataset": "x"}},  # fixed dataset outside its condition
        {"editor": {"mode": "external"}},  # missing profile
        {"editor": {"mode": "initial_policy", "proposals_per_source": 3}},  # needs confirmation seeds
        {"verification": {"mode": "local", "continuations_per_branch": 2}},
        {"name": "Bad Name"},
    ],
)
def test_invalid_experiments_rejected(patch):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**MINIMAL, **patch})


def test_base_inheritance_one_level(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(MINIMAL))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"base": "base.yaml", "name": "child", "verification": {"continuations_per_branch": 2}}))
    cfg, raw = load_experiment(tmp_path / "child.yaml")
    assert cfg.name == "child" and cfg.verification.continuations_per_branch == 2
    assert cfg.episode.max_turns == 5
    (tmp_path / "grand.yaml").write_text(yaml.safe_dump({"base": "child.yaml", "name": "g"}))
    with pytest.raises(ValueError, match="nested"):
        load_experiment(tmp_path / "grand.yaml")


def test_literal_credentials_rejected(tmp_path):
    with pytest.raises(ValueError, match="credential"):
        assert_no_secrets({"inference": {"api_key": "sk-123"}})
    assert_no_secrets({"inference": {"api_key_env": "LLM_API_KEY"}})
    p = tmp_path / "m.yaml"
    p.write_text(textwrap.dedent("""
        name: m
        inference: {mode: external, backend: llama_cpp, api_base: "http://x/v1", api_key: "secret"}
    """))
    with pytest.raises(ValueError):
        load_machine(p)


def test_machine_ssh_alias_validated(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(textwrap.dedent("""
        name: m
        coordinator: {kind: ssh, ssh_alias: "bad; rm -rf /", workdir: /x}
        inference: {mode: scripted, backend: scripted}
    """))
    with pytest.raises(ValidationError):
        load_machine(p)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def test_write_once_refuses_different_content(tmp_path):
    p = tmp_path / "run.json"
    write_once_json(p, {"a": 1})
    write_once_json(p, {"a": 1})  # identical is fine (idempotent resume)
    with pytest.raises(FileExistsError):
        write_once_json(p, {"a": 2})


def test_jsonl_tolerates_torn_tail(tmp_path):
    p = tmp_path / "e.jsonl"
    app = JsonlAppender(p)
    app.append({"x": 1})
    app.append({"x": 2})
    with open(p, "a") as f:
        f.write('{"x": 3')  # interrupted append
    assert read_jsonl(p) == [{"x": 1}, {"x": 2}]


def test_run_lock_is_exclusive(tmp_path):
    with run_lock(tmp_path):
        code = textwrap.dedent(f"""
            from pathlib import Path
            from learning_loop.storage import run_lock, RunLockedError
            try:
                with run_lock(Path({str(tmp_path)!r})):
                    print("acquired")
            except RunLockedError:
                print("locked")
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.stdout.strip() == "locked", out.stderr
    with run_lock(tmp_path):  # released afterwards
        pass


def test_manifest_resume_and_interrupted_preserved(tmp_path):
    mpath = tmp_path / "manifest.json"
    m = StageManifest.load_or_create(mpath, "collect", 0)
    m.ensure_items(["a", "b"])
    m.items["a"].status = "done"
    m.items["b"].status = "running"
    m.save(mpath)
    item_dir = tmp_path / "b"
    item_dir.mkdir()
    (item_dir / "partial.txt").write_text("x")

    m2 = StageManifest.load_or_create(mpath, "collect", 0)
    m2.ensure_items(["a", "b"])  # idempotent
    assert [w.item_id for w in m2.pending()] == ["b"]
    preserve_interrupted(item_dir, m2.items["b"])
    assert not item_dir.exists()
    assert (tmp_path / "b.interrupted-1" / "partial.txt").exists()
    assert m2.items["b"].interrupted_dirs == ["b.interrupted-1"]


def test_atomic_write_leaves_no_temp(tmp_path):
    atomic_write_json(tmp_path / "x.json", {"k": [1, 2]})
    assert json.loads((tmp_path / "x.json").read_text()) == {"k": [1, 2]}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# --------------------------------------------------------------------------- #
# Usage arithmetic
# --------------------------------------------------------------------------- #


def test_usage_sum_and_unavailable_propagates():
    a = Usage(input_tokens=100, output_tokens=10, cached_input_tokens=80)
    b = Usage(input_tokens=120, output_tokens=5, cached_input_tokens=None)
    s = Usage.sum([a, b])
    assert s.input_tokens == 220 and s.output_tokens == 15 and s.total == 235
    assert s.cached_input_tokens is None  # unavailable, not 0
    assert Usage(input_tokens=None, output_tokens=3).total is None
