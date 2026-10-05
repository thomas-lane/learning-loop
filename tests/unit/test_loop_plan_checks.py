"""Plan checks before any run directory exists: serving backends a model profile marks
`unsupported`, and the checkpoint each external endpoint must declare for standalone
evaluations and editor replays."""

import asyncio

import pytest
import yaml

from learning_loop.core.config import REPO_ROOT
from learning_loop.core.storage import read_json
from learning_loop.orchestration import coordinator as co

EXP = REPO_ROOT / "experiments" / "fixture-two-cycles.yaml"
M = REPO_ROOT / "configs" / "machines" / "examples"
REAL_LEARNER = ["learner.scripted_policy=null"]  # a served learner instead of the scripted fixture policy
MODEL_EDITOR = ["editor.mode=initial_policy", "editor.scripted_path=null"]


def machine(tmp_path, name: str, **changes) -> str:
    m = yaml.safe_load((M / "fixture.yaml").read_text())
    m.update(changes)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(m))
    return str(p)


def external(served: str) -> dict:
    return {"mode": "external", "backend": "hf_transformers", "api_base": "http://127.0.0.1:9/v1", "served_checkpoint_id": served}


@pytest.fixture(scope="module")
def source_run(tmp_path_factory):
    """A fixture run whose cycle-1 learner is a trained checkpoint (not the initial one)."""
    ctx = co.create_run(EXP, M / "fixture.yaml", run_id="src", runs_dir=tmp_path_factory.mktemp("runs"))
    asyncio.run(co.run_all(ctx, log=lambda m: None))
    initial = read_json(ctx.run_dir / "run.json")["initial_checkpoint"]["checkpoint_id"]
    learner_c1 = read_json(co.cycle_state_path(ctx, 1))["learner_in"]["checkpoint_id"]
    assert learner_c1 != initial
    return ctx.run_dir, initial, learner_c1


def unsupported(profile, backend: str):
    sb = profile.serving[backend].model_copy(update={"status": "unsupported"})
    return profile.model_copy(update={"serving": {**profile.serving, backend: sb}})


def test_unsupported_serving_backend_is_refused():
    exp, _, mac, learner, editor = co.load_all(EXP, M / "mac-local.yaml", REAL_LEARNER)
    co.check_compatibility(exp, mac, learner, editor)
    bad = unsupported(learner, "hf_transformers")
    with pytest.raises(co.PlanError, match="unsupported; the learner"):
        co.check_compatibility(exp, mac, bad, bad)
    # the editor's endpoint is checked against the editor's model profile
    exp, _, live, learner, _ = co.load_all(EXP, M / "fixture-docker-live-editor.yaml", MODEL_EDITOR)
    co.check_compatibility(exp, live, learner, learner)
    with pytest.raises(co.PlanError, match="unsupported; the editor"):
        co.check_compatibility(exp, live, learner, unsupported(learner, "hf_transformers"))


def test_managed_lora_runs_need_a_declared_peft_backend():
    exp, _, mac, learner, editor = co.load_all(REPO_ROOT / "experiments" / "smoke-mac.yaml", M / "mac-local.yaml")
    co.check_compatibility(exp, mac, learner, editor)
    undeclared = learner.model_copy(update={"serving": {k: v for k, v in learner.serving.items() if k != "hf_transformers"}})
    with pytest.raises(co.PlanError, match="declares no 'hf_transformers'"):
        co.check_compatibility(exp, mac, undeclared, undeclared)


def test_evaluation_checks_the_external_endpoint_before_creating_the_run(tmp_path):
    exp, _, _, learner, _ = co.load_all(EXP, M / "fixture.yaml")
    base_id = co.base_checkpoint(learner).checkpoint_id
    wrong = machine(tmp_path, "wrong", inference=external("base:other@000000000000"))
    with pytest.raises(co.PlanError, match=f"needs '{base_id}'"):
        co.evaluate_checkpoint(EXP, wrong, "base", ["dev"], run_id="ev", runs_dir=tmp_path / "runs", overrides=REAL_LEARNER)
    assert not (tmp_path / "runs").exists()


def test_edit_replay_checks_endpoints_against_the_source_cycle(source_run, tmp_path, monkeypatch):
    src_dir, initial, learner_c1 = source_run

    async def no_replay(*args):  # the checks are the subject; nothing is served here
        return None

    monkeypatch.setattr(co, "_run_edit_replay", no_replay)
    runs = tmp_path / "runs"
    # The learner continues branches as the source cycle's learner, not the initial checkpoint.
    at_initial = machine(tmp_path, "learner-initial", inference=external(initial))
    with pytest.raises(co.PlanError, match=f"needs '{learner_c1}'"):
        co.edit_replay(src_dir, 1, EXP, at_initial, run_id="er0", runs_dir=runs, overrides=REAL_LEARNER)
    assert not (runs / "er0").exists()
    at_c1 = machine(tmp_path, "learner-c1", inference=external(learner_c1))
    co.edit_replay(src_dir, 1, EXP, at_c1, run_id="er1", runs_dir=runs, overrides=REAL_LEARNER)

    # A separate external editor endpoint must serve the editor: the source run's initial
    # checkpoint for initial_policy, the source cycle's learner for current_learner.
    def editor_at(served: str) -> str:
        return machine(tmp_path, f"editor-{served.replace(':', '_').replace('@', '_')}", editor_inference=external(served))

    with pytest.raises(co.PlanError, match="editor endpoint declares"):
        co.edit_replay(src_dir, 1, EXP, editor_at(learner_c1), run_id="er2", runs_dir=runs, overrides=MODEL_EDITOR)
    co.edit_replay(src_dir, 1, EXP, editor_at(initial), run_id="er3", runs_dir=runs, overrides=MODEL_EDITOR)
    current = ["editor.mode=current_learner", "editor.scripted_path=null"]
    with pytest.raises(co.PlanError, match="editor endpoint declares"):
        co.edit_replay(src_dir, 1, EXP, editor_at(initial), run_id="er4", runs_dir=runs, overrides=current)
    co.edit_replay(src_dir, 1, EXP, editor_at(learner_c1), run_id="er5", runs_dir=runs, overrides=current)
