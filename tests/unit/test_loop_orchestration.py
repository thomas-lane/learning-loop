"""Cycle orchestration on cheap fixtures (local fixture backend, scripted learner/editor,
fixture trainer): lineage, reference transitions, fixed editor identity, no-update cycles,
frozen/fixed-dataset controls, resume, infra retries, locking and editor replays."""

import asyncio
import json
import stat
from pathlib import Path

import pytest
import yaml

from learning_loop.core.config import REPO_ROOT
from learning_loop.core.records import EpisodeSummary, StopCategory
from learning_loop.core.storage import RunLockedError, StageManifest, read_json, read_jsonl, run_lock
from learning_loop.orchestration import coordinator as co

EXP = REPO_ROOT / "experiments" / "fixture-two-cycles.yaml"
MACHINE = REPO_ROOT / "configs" / "machines" / "examples" / "fixture.yaml"


def run(ctx):
    asyncio.run(co.run_all(ctx, log=lambda m: None))
    return ctx


def cycle(ctx, c):
    return read_json(co.cycle_state_path(ctx, c))


def manifest(ctx, c, stage):
    return StageManifest.model_validate(read_json(ctx.stage_dir(c, stage) / "manifest.json"))


@pytest.fixture(scope="module")
def two_cycles(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    return run(co.create_run(EXP, MACHINE, run_id="fx", runs_dir=runs))


def test_lineage_and_reference_transitions(two_cycles):
    ctx = two_cycles
    c0, c1, c2 = cycle(ctx, 0), cycle(ctx, 1), cycle(ctx, 2)
    base = c0["learner_in"]["checkpoint_id"]
    assert base.startswith("base:")
    assert c0["update"] == c1["update"] == "trained"
    # cycle k's output is cycle k+1's input; the DPO reference is the learner frozen at cycle start
    assert c1["learner_in"] == c0["learner_out"]
    assert c2["learner_in"] == c1["learner_out"]
    assert c0["reference"] == base
    assert c1["reference"] == c0["learner_out"]["checkpoint_id"]
    rec1 = c1["checkpoint_record"]
    assert rec1["checkpoint"]["parent_checkpoint_id"] == c0["learner_out"]["checkpoint_id"]
    assert rec1["metrics"]["initial_weights_match_incoming"] is True  # continued, not re-initialized
    assert rec1["optimizer_steps"] == ctx.exp.training.optimizer_steps
    # immutable, unique checkpoint dirs
    ids = {c0["learner_out"]["checkpoint_id"], c1["learner_out"]["checkpoint_id"]}
    assert len(ids) == 2
    for cid in ids:
        d = ctx.checkpoints_dir / cid
        assert (d / "checkpoint.json").exists()
        assert not (d / "checkpoint.json").stat().st_mode & stat.S_IWUSR


def test_eval_is_paired_and_branches_never_count_as_eval(two_cycles):
    ctx = two_cycles
    seeds_by_cycle = []
    for c in (0, 1, 2):
        m = manifest(ctx, c, "eval")
        sums = [EpisodeSummary.model_validate(read_json(ctx.run_dir / w.output / "summary.json")) for w in m.items.values()]
        assert all(s.role.value == "eval" for s in sums)
        seeds_by_cycle.append(sorted((s.instance_id, s.attempt_index, s.seed) for s in sums))
    assert seeds_by_cycle[0] == seeds_by_cycle[1] == seeds_by_cycle[2]  # same declared schedule
    ver = manifest(ctx, 0, "verify")
    for w in ver.items.values():
        rec = read_json(ctx.run_dir / w.output / "verification.json")
        assert {b["episode"]["role"] for b in rec["branches"]} == {"branch"}
        seeds = {b["branch"]: b["continuation_seed"] for b in rec["branches"]}
        assert seeds["original"] == seeds["edited"]  # matched continuation seeds


def test_fixed_editor_identity_and_one_proposal_per_source(two_cycles):
    ctx = two_cycles
    ids = set()
    for c in (0, 1):
        m = manifest(ctx, c, "edit")
        sources = [w for w in manifest(ctx, c, "collect").items.values()]
        assert len(m.items) == len(sources)  # one proposal per successful source
        ids |= {w.meta["editor_id"] for w in m.items.values()}
    assert len(ids) == 1  # the editor did not change when the learner did


def test_datasets_only_train_instances_and_verified(two_cycles):
    ctx = two_cycles
    for c in (0, 1):
        d = ctx.stage_dir(c, "dataset")
        man = read_json(d / "manifest.json")
        assert man["n_examples"] > 0
        for prov in read_jsonl(d / "provenance.jsonl"):
            assert prov["split"] == "train"
            assert ctx.instance_split[prov["instance_id"]].value == "train"
            assert prov["verification_mode"] == "continuation"
            assert prov["kind"] == "fixture"  # scripted learner/editor: never labeled verified
        for ex in read_jsonl(d / "preferences.jsonl"):
            assert set(ex) == {"schema_version", "pair_id", "prompt", "chosen", "rejected", "tools"}
            assert ex["chosen"] != ex["rejected"]
    # cycle 1 mixes current and history pairs (bounded buffer)
    assert read_json(ctx.stage_dir(1, "dataset") / "manifest.json")["n_examples"] > read_json(ctx.stage_dir(1, "dataset") / "current" / "manifest.json")["n_examples"]


def test_report_written(two_cycles):
    text = (two_cycles.run_dir / "reports" / "report.md").read_text()
    assert "Paired comparison" in text and "lost" in text


def test_no_update_cycle_when_editor_abstains(tmp_path):
    edits = tmp_path / "abstain.yaml"
    edits.write_text(yaml.safe_dump({"editor_id": "abstain-fixture", "proposals": [{"family": "count-errors", "decision": "abstain"}]}))
    ctx = run(co.create_run(EXP, MACHINE, run_id="noupd", runs_dir=tmp_path, overrides=[f"editor.scripted_path={edits}", "cycles=1"]))
    c0 = cycle(ctx, 0)
    assert c0["update"] == "no_update"
    assert c0["learner_out"] == c0["learner_in"]
    assert c0["dataset"]["n_examples"] == 0
    assert not ctx.checkpoints_dir.exists() or not any(ctx.checkpoints_dir.iterdir())
    assert cycle(ctx, 1)["learner_in"]["checkpoint_id"].startswith("base:")


def test_frozen_baseline_and_compare(two_cycles, tmp_path):
    ctx = run(co.create_run(EXP, MACHINE, run_id="frozen", runs_dir=tmp_path, overrides=["condition=frozen_baseline", "cycles=0"]))
    assert sorted(p.name for p in ctx.cycle_dir(0).iterdir() if p.is_dir()) == ["eval"]
    from learning_loop.reporting.report import compare

    res = compare(ctx.run_dir, two_cycles.run_dir)
    assert res is not None


def test_fixed_dataset_control_never_collects(two_cycles, tmp_path):
    frozen = two_cycles.stage_dir(0, "dataset")
    ctx = run(
        co.create_run(EXP, MACHINE, run_id="fixed", runs_dir=tmp_path, overrides=["condition=fixed_dataset", f"training.fixed_dataset={frozen}"])
    )
    shas = set()
    for c in (0, 1):
        stages = {p.name for p in ctx.cycle_dir(c).iterdir() if p.is_dir()}
        assert not stages & {"collect", "edit", "verify"}
        st = cycle(ctx, c)
        assert st["update"] == "trained"
        assert st["checkpoint_record"]["optimizer_steps"] == two_cycles.exp.training.optimizer_steps  # matched effort
        shas.add(st["checkpoint_record"]["dataset_sha256"])
    assert shas == {read_json(frozen / "manifest.json")["dataset_sha256"]}


def test_edit_replay_uses_same_sources_without_collecting(two_cycles, tmp_path):
    out = co.edit_replay(two_cycles.run_dir, 0, EXP, MACHINE, run_id="er", runs_dir=tmp_path)
    meta = read_json(out / "run.json")
    assert meta["kind"] == "edit_replay"
    assert meta["imported_sources"]["learner"]["checkpoint_id"] == cycle(two_cycles, 0)["learner_in"]["checkpoint_id"]
    assert not (out / "cycles" / "cycle-000" / "collect").exists()
    m = StageManifest.model_validate(read_json(out / "cycles" / "cycle-000" / "edit" / "manifest.json"))
    src_dirs = {w.meta["source_dir"] for w in m.items.values()}
    assert all(Path(s).is_absolute() and str(two_cycles.run_dir) in s for s in src_dirs)


def test_resume_skips_done_items_and_preserves_interrupted(tmp_path):
    ctx = co.create_run(EXP, MACHINE, run_id="res", runs_dir=tmp_path, overrides=["cycles=0"])
    run(ctx)
    m = manifest(ctx, 0, "eval")
    (item,) = m.items.values()
    first_attempts = item.attempts
    # simulate a crash mid-item: status running, partial output on disk, cycle not done
    item.status = "running"
    m.status = "running"
    m.save(ctx.stage_dir(0, "eval") / "manifest.json")
    st = cycle(ctx, 0)
    st["status"] = "running"
    (co.cycle_state_path(ctx, 0)).write_text(json.dumps(st))
    ctx2 = co.open_run(ctx.run_dir)
    run(ctx2)
    m2 = manifest(ctx2, 0, "eval")
    (item2,) = m2.items.values()
    assert item2.status == "done"
    assert item2.attempts == first_attempts + 1
    assert item2.interrupted_dirs == [f"{item2.item_id}.interrupted-1"]
    assert (ctx.stage_dir(0, "eval") / "items" / f"{item2.item_id}.interrupted-1" / "summary.json").exists()


def test_concurrent_coordinators_refused(tmp_path):
    ctx = co.create_run(EXP, MACHINE, run_id="lock", runs_dir=tmp_path, overrides=["cycles=0"])
    with run_lock(ctx.run_dir):
        with pytest.raises(RunLockedError):
            asyncio.run(co.run_all(co.open_run(ctx.run_dir), log=lambda m: None))


def test_cli_stage_records_provenance_and_refuses_a_locked_run(tmp_path, capsys):
    """`loop stage` appends its invocation like `resume`; a second coordinator on a locked run gets
    `error: ...` and exit 2, and leaves no invocation record."""
    from learning_loop.cli import main

    ctx = run(co.create_run(EXP, MACHINE, run_id="stg", runs_dir=tmp_path, overrides=["cycles=0"]))
    inv = ctx.run_dir / "invocations.jsonl"
    assert not inv.exists()
    assert main(["stage", str(ctx.run_dir), "--cycle", "0", "--stage", "eval"]) == 0
    assert len(read_jsonl(inv)) == 1 and read_jsonl(inv)[0]["run_id"] == "stg"
    capsys.readouterr()
    with run_lock(ctx.run_dir):
        assert main(["stage", str(ctx.run_dir), "--cycle", "0", "--stage", "eval"]) == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "locked by another coordinator" in err
        assert len(read_jsonl(inv)) == 1  # the refused invocation changed nothing
        assert main(["resume", str(ctx.run_dir)]) == 2
        assert "locked by another coordinator" in capsys.readouterr().err
        assert len(read_jsonl(inv)) == 1  # nor did the refused resume
        with pytest.raises(RunLockedError):  # nor does re-running the same --run-id
            co.create_run(EXP, MACHINE, run_id="stg", runs_dir=tmp_path, overrides=["cycles=0"])
        assert len(read_jsonl(inv)) == 1


def test_cli_validate_checks_the_initial_checkpoint_profile(two_cycles, tmp_path, capsys):
    """`validate` refuses an initial checkpoint trained for another model profile, as `run` does."""
    from learning_loop.cli import main

    src = two_cycles.checkpoints_dir / cycle(two_cycles, 0)["learner_out"]["checkpoint_id"]
    rec = read_json(src / "checkpoint.json")
    assert main(["validate", str(EXP), "--machines", str(MACHINE), "--set", f"learner.initial_checkpoint={src}"]) == 0
    rec["checkpoint"]["model_profile"] = "some-other-profile"
    other = tmp_path / "other-ckpt"
    other.mkdir()
    (other / "checkpoint.json").write_text(json.dumps(rec))
    capsys.readouterr()
    assert main(["validate", str(EXP), "--machines", str(MACHINE), "--set", f"learner.initial_checkpoint={other}"]) == 2
    assert "was trained for some-other-profile" in capsys.readouterr().err


class FlakyBackend:
    """Wraps the real fixture backend; the first `n_fail` episodes report an infra failure."""

    name = "flaky"

    def __init__(self, inner, n_fail):
        self.inner, self.n_fail = inner, n_fail

    def restore_capability(self, instance):
        return self.inner.restore_capability(instance)

    async def run(self, instance, plan, out_dir):
        res = await self.inner.run(instance, plan, out_dir)
        if self.n_fail > 0:
            self.n_fail -= 1
            res.summary = res.summary.model_copy(update={"stop_category": StopCategory.INFRA, "stop_reason": "infra:injected", "infra_error": "injected", "success": None})
        return res


def test_infra_failures_retried_then_visible(tmp_path):
    ctx = co.create_run(EXP, MACHINE, run_id="infra", runs_dir=tmp_path, overrides=["cycles=0", "runtime.infra_retries=1"])
    ctx._backend = FlakyBackend(co.make_backend(ctx), n_fail=1)
    run(ctx)
    (item,) = manifest(ctx, 0, "eval").items.values()
    assert item.status == "done" and item.attempts == 2 and item.infra_failures == 1
    assert len(item.interrupted_dirs) == 1  # the failed attempt is kept, not overwritten

    ctx2 = co.create_run(EXP, MACHINE, run_id="infra2", runs_dir=tmp_path, overrides=["cycles=0", "runtime.infra_retries=1"])
    ctx2._backend = FlakyBackend(co.make_backend(ctx2), n_fail=5)
    run(ctx2)
    (item,) = manifest(ctx2, 0, "eval").items.values()
    assert item.status == "infra_failed" and item.attempts == 2  # bounded, and visible in the manifest
    assert "infra" in (ctx2.run_dir / "reports" / "report.md").read_text().lower()


def test_validation_rejects_incompatible_plans(tmp_path):
    with pytest.raises(co.PlanError):
        co.create_run(EXP, MACHINE, run_id="bad1", runs_dir=tmp_path, overrides=["condition=frozen_baseline"])  # cycles must be 0
    with pytest.raises(co.PlanError):
        co.create_run(EXP, MACHINE, run_id="bad2", runs_dir=tmp_path, overrides=["tasks.collection_panel=dev"])  # dev panel for training
    with pytest.raises(co.PlanError):
        co.create_run(EXP, MACHINE, run_id="bad3", runs_dir=tmp_path, overrides=["evaluation.dev_panels=[train]"])
    with pytest.raises(co.PlanError):
        co.create_run(EXP, MACHINE, run_id="bad4", runs_dir=tmp_path, overrides=["learner.scripted_policy=null"])


def test_evaluate_saved_checkpoint_requires_final_flag(two_cycles, tmp_path):
    ckpt = two_cycles.checkpoints_dir / cycle(two_cycles, 1)["learner_out"]["checkpoint_id"]
    with pytest.raises(co.PlanError, match="--final"):
        co.evaluate_checkpoint(EXP, MACHINE, str(ckpt), ["final"], run_id="ev0", runs_dir=tmp_path)
    out = co.evaluate_checkpoint(EXP, MACHINE, str(ckpt), ["final"], final=True, run_id="ev1", runs_dir=tmp_path)
    meta = read_json(out / "run.json")
    assert meta["kind"] == "evaluation" and meta["final"] is True
    assert meta["evaluated_checkpoint"]["checkpoint_id"] == ckpt.name
    m = StageManifest.model_validate(read_json(out / "cycles" / "cycle-000" / "eval" / "manifest.json"))
    assert m.status == "done" and len(m.items) == 1


def test_resume_dispatches_by_run_kind(two_cycles, tmp_path):
    ckpt = two_cycles.checkpoints_dir / cycle(two_cycles, 0)["learner_out"]["checkpoint_id"]
    out = co.evaluate_checkpoint(EXP, MACHINE, str(ckpt), ["dev"], run_id="evk", runs_dir=tmp_path)
    with pytest.raises(co.PlanError):
        asyncio.run(co.run_all(co.open_run(out), log=lambda m: None))  # never the learning loop
    co.resume_run(out)  # re-enters the evaluation only; idempotent
    assert not (out / "checkpoints").exists()
    assert not (out / "cycles" / "cycle-000" / "collect").exists()
    assert (out / "invocations.jsonl").exists()  # resume provenance appended, original kept


def test_infra_failed_items_stay_terminal_on_resume(tmp_path):
    ctx = co.create_run(EXP, MACHINE, run_id="term", runs_dir=tmp_path, overrides=["cycles=0", "runtime.infra_retries=0"])
    ctx._backend = FlakyBackend(co.make_backend(ctx), n_fail=10)
    run(ctx)
    (item,) = manifest(ctx, 0, "eval").items.values()
    assert item.status == "infra_failed" and item.attempts == 1
    ctx2 = co.open_run(ctx.run_dir)
    st = cycle(ctx2, 0)
    st["status"] = "running"
    co.cycle_state_path(ctx2, 0).write_text(json.dumps(st))
    run(ctx2)
    (item2,) = manifest(ctx2, 0, "eval").items.values()
    assert item2.attempts == 1  # retry budget is not silently extended by a resume


def test_audit_uses_its_own_repetitions(tmp_path):
    ctx = run(co.create_run(EXP, MACHINE, run_id="aud", runs_dir=tmp_path,
                            overrides=["cycles=1", "verification.audit.fraction=1.0", "verification.audit.continuations_per_branch=2"]))
    m = manifest(ctx, 0, "audit")
    assert m.items
    for w in m.items.values():
        rec = read_json(ctx.run_dir / w.output / "verification.json")
        assert rec["purpose"] == "audit"
        assert sorted({b["repetition"] for b in rec["branches"]}) == [0, 1]
    # audits never change the frozen dataset
    assert read_json(ctx.stage_dir(0, "dataset") / "manifest.json")["n_examples"] == len(manifest(ctx, 0, "verify").items)


def test_documented_manual_reopen_of_failed_eval_item(tmp_path):
    """docs/operations.md: set the item to pending in the manifest, then `loop stage ... eval`."""
    ctx = co.create_run(EXP, MACHINE, run_id="reopen", runs_dir=tmp_path, overrides=["cycles=0", "runtime.infra_retries=0"])
    ctx._backend = FlakyBackend(co.make_backend(ctx), n_fail=1)
    run(ctx)
    mpath = ctx.stage_dir(0, "eval") / "manifest.json"
    m = manifest(ctx, 0, "eval")
    (item,) = m.items.values()
    assert item.status == "infra_failed"
    item.status = "pending"
    m.save(mpath)
    ctx2 = co.open_run(ctx.run_dir)
    asyncio.run(co.run_single_stage(ctx2, 0, "eval"))
    (item2,) = manifest(ctx2, 0, "eval").items.values()
    assert item2.status == "done" and item2.attempts == 2 and len(item2.interrupted_dirs) == 1


def test_dropped_training_pairs_are_recorded_and_reported(tmp_path, monkeypatch):
    """Pairs the trainer drops (e.g. longer than dpo.max_length) change the training data: they are
    counted in cycle.json, `loop status` and the report, with reasons and pair ids."""
    real = co.train_checkpoint
    render = {"n_input": 2, "n_kept": 0, "n_dropped": 2, "dropped_by_reason": {"oversize": 2}, "max_length": 64,
              "dropped": [{"pair_id": "p-a", "reason": "oversize:90>64"}, {"pair_id": "p-b", "reason": "oversize:70>64"}],
              "kept_tokens": None}

    def train(ctx, c, incoming, ds_dir):
        if c == 0:
            raise co.NoUpdate("all 2 examples dropped", render=render)
        return real(ctx, c, incoming, ds_dir)

    monkeypatch.setattr(co, "train_checkpoint", train)
    ctx = run(co.create_run(EXP, MACHINE, run_id="drops", runs_dir=tmp_path))
    c0, c1 = cycle(ctx, 0), cycle(ctx, 1)
    assert c0["update"] == "no_update"
    td = c0["training_data"]
    assert (td["n_trained"], td["n_dropped"], td["dropped_by_reason"]) == (0, 2, {"oversize": 2})
    assert [d["pair_id"] for d in td["dropped"]] == ["p-a", "p-b"] and td["n_exported"] == c0["dataset"]["n_examples"]
    assert c1["update"] == "trained" and c1["training_data"]["rendered"] is False  # fixture trainer renders nothing
    assert co.run_status(ctx.run_dir)["cycles"]["cycle-000"]["training_data"]["n_dropped"] == 2
    co.write_report(ctx)
    report = (ctx.run_dir / "reports" / "report.md").read_text()
    assert "## Training data actually used" in report and "**2 exported pair(s) were dropped before training**" in report
    rows = (ctx.run_dir / "reports" / "training_data.csv").read_text()
    assert "p-a: oversize:90>64" in rows
