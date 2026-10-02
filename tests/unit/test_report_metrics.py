"""Aggregation, uncertainty over instances, paired comparisons, reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from learning_loop.metrics import (
    SMALL_N_NOTE,
    CostRate,
    EpisodeRow,
    StageEffort,
    append_rows,
    group_summaries,
    instance_level,
    paired_comparison,
    proposal_stats,
    stage_effort_table,
    summarize,
)
from learning_loop.records import (
    EditProposal,
    EpisodeRole,
    EpisodeSummary,
    StopCategory,
    Timing,
    Usage,
    VerificationRecord,
)
from learning_loop.report import collect_run, compare, compare_runs, write_report, write_run_report
from learning_loop.storage import StageManifest, WorkItem, atomic_write_json


def ep(iid="i0", attempt=0, success=True, total=(100, 10), ckpt="base", role=EpisodeRole.EVAL, stop=StopCategory.MODEL, seed=None, **kw) -> EpisodeSummary:
    u = Usage(input_tokens=total[0], output_tokens=total[1]) if total else Usage(input_tokens=None, output_tokens=None)
    return EpisodeSummary(
        episode_id=f"{ckpt}-{iid}-{attempt}-{role.value}",
        role=role,
        instance_id=iid,
        attempt_index=attempt,
        seed=seed if seed is not None else 1000 + attempt,
        checkpoint_id=ckpt,
        stop_reason="model_finished" if stop == StopCategory.MODEL else f"{stop.value}:x",
        stop_category=stop,
        reward={"reward": 1.0 if success else 0.3} if success is not None else None,
        partial_reward=(1.0 if success else 0.3) if success is not None else None,
        success=success,
        usage=u,
        n_requests=3,
        n_tool_calls=2,
        **kw,
    )


def row(e, cycle=0, panel="dev", family="fam", difficulty="easy", skills=("logs",)) -> EpisodeRow:
    return EpisodeRow(episode=e, run_id="r", cycle=cycle, panel=panel, family=family, difficulty=difficulty, skills=list(skills))


def test_summary_missing_values_are_not_zero():
    rows = [
        row(ep("a", total=(100, 10))),
        row(ep("b", total=None, success=None, stop=StopCategory.INFRA)),
        row(ep("c", total=(200, 20), success=False, timing=Timing(tool_sec=1.5))),
    ]
    rows[0].episode.usage.cached_input_tokens = 80
    s = summarize(rows)
    assert s["n_episodes"] == 3 and s["n_success"] == 1 and s["success_rate"] == pytest.approx(1 / 3)  # infra stays in the denominator
    assert s["n_ungraded"] == 1 and s["n_infra_failures"] == 1
    assert s["total_tokens_sum"] == 330 and s["n_with_tokens"] == 2 and s["total_tokens_mean"] == 165
    assert s["input_tokens_sum"] == 300  # cached subset NOT added
    assert s["cached_input_tokens_sum (subset of input)"] == 80 and s["n_with_cached"] == 1
    assert s["tool_sec_sum"] == 1.5 and s["endpoint_sec_sum"] is None and s["tool_cpu_sec_sum"] is None
    assert s["mean_partial_reward"] == pytest.approx((1.0 + 0.3) / 2)
    assert s["instance_success"]["interval"] == SMALL_N_NOTE


def test_instance_level_uncertainty_unit():
    # 3 attempts of one instance must not look like 3 independent samples
    s = instance_level({"a": [1.0, 1.0, 1.0], "b": [0.0]})
    assert s["mean"] == 0.5 and s["n_instances"] == 2 and s["ci95_low"] is None and s["interval"] == SMALL_N_NOTE
    big = instance_level({f"i{k}": [float(k % 2)] for k in range(8)})
    assert big["n_instances"] == 8 and big["ci95_low"] is not None and big["ci95_low"] <= big["mean"] <= big["ci95_high"]
    assert instance_level({f"i{k}": [float(k % 2)] for k in range(8)}) == big  # seeded, reproducible


def test_group_by_skill_and_checkpoint():
    rows = [row(ep("a"), skills=("logs", "gzip")), row(ep("b", success=False), skills=("python",)), row(ep("a", ckpt="c1"), cycle=1)]
    by_skill = {g["skill"]: g for g in group_summaries(rows, ("skill",))}
    assert by_skill["logs"]["n_episodes"] == 2 and by_skill["gzip"]["n_episodes"] == 1 and by_skill["python"]["n_success"] == 0
    by_ck = {(g["checkpoint"], g["cycle"]): g for g in group_summaries(rows, ("checkpoint", "cycle"))}
    assert by_ck[("base", 0)]["n_episodes"] == 2 and by_ck[("c1", 1)]["n_episodes"] == 1


def test_paired_comparison_transitions_never_hide_lost_successes():
    a = [row(ep("i1")), row(ep("i2")), row(ep("i3", success=False)), row(ep("i4", success=False)), row(ep("i5")), row(ep("i6", attempt=0))]
    b = [
        row(ep("i1", total=(50, 5), ckpt="c1")),  # both succeed, cheaper
        row(ep("i2", success=False, total=(10, 1), ckpt="c1")),  # LOST (and very cheap)
        row(ep("i3", ckpt="c1")),  # gained
        row(ep("i4", success=False, ckpt="c1")),  # both fail
        row(ep("i5", success=None, total=None, ckpt="c1", stop=StopCategory.INFRA)),  # undetermined
        row(ep("i6", attempt=0, seed=777, ckpt="c1")),  # seed mismatch -> not paired
    ]
    c = paired_comparison(a, b, "base", "c1")
    assert c["transitions"] == {"both_succeed": 1, "gained": 1, "lost": 1, "both_fail": 1, "undetermined": 1}
    assert c["n_pairs"] == 5 and c["n_seed_mismatch"] == 1 and c["n_only_a"] == 1 and c["n_only_b"] == 1
    assert c["token_delta_coverage"] == {"n_both_succeed_with_tokens": 1, "n_pairs": 5, "fraction": 0.2}
    assert c["token_delta_instance_level"]["mean"] == -55.0  # only the both-succeed pair
    assert c["token_delta_instance_level"]["interval"] == SMALL_N_NOTE
    lost = [p for p in c["pairs"] if p["transition"] == "lost"][0]
    assert lost["token_delta_both_succeed"] is None
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison(a + [row(ep("i1"))], b)


def test_stage_effort_money_requires_explicit_rate():
    efforts = [StageEffort(stage="collection", cycle=0, usage=Usage(input_tokens=1_000_000, output_tokens=500_000), duration_sec=3600), StageEffort(stage="training", cycle=0, duration_sec=None, optimizer_steps=3)]
    rows = stage_effort_table(efforts)
    assert all(r["cost_usd"] is None for r in rows)
    rates = {"collection": CostRate(usd_per_million_input_tokens=1.0, usd_per_million_output_tokens=2.0, provenance="test rate"), "training": CostRate(usd_per_hour=2.0, provenance="gpu quote")}
    rows = stage_effort_table(efforts, rates)
    assert rows[0]["cost_usd"] == pytest.approx(2.0) and rows[0]["cost_provenance"] == "test rate"
    assert rows[1]["cost_usd"] is None and "incomplete" in rows[1]["cost_provenance"]  # missing duration is not free


def _prop(i, status, reasons=()):
    return EditProposal(proposal_id=f"p{i}", source_episode_id=f"s{i}", instance_id="i", editor_id="e", status=status, rejection_reasons=list(reasons), usage=Usage(input_tokens=100, output_tokens=10))


def test_proposal_stats():
    props = [_prop(1, "proposed"), _prop(2, "proposed"), _prop(3, "abstained"), _prop(4, "invalid", ["ungrounded_constant:1.2.3.4", "hidden_path:/tests"])]
    vers = [
        VerificationRecord(verification_id="v1", proposal_id="p1", mode="continuation", acceptance_rule="r", accepted=True, reasons=["accepted"], mean_saving=40, operational_usage=Usage(input_tokens=10, output_tokens=1)),
        VerificationRecord(verification_id="v2", proposal_id="p2", mode="continuation", acceptance_rule="r", accepted=False, reasons=["not_success:edited:r0"], mean_saving=90, operational_usage=Usage(input_tokens=10, output_tokens=1)),
        VerificationRecord(verification_id="v3", proposal_id="p1", mode="continuation", acceptance_rule="r", accepted=False, reasons=["tie"], purpose="audit"),
    ]
    s = proposal_stats(props, vers)
    assert s["n_sources"] == 4 and s["n_proposed_valid"] == 2 and s["n_abstained"] == 1 and s["n_invalid"] == 1
    assert s["invalid_reasons"] == {"ungrounded_constant": 1, "hidden_path": 1}
    assert s["n_verified"] == 2 and s["n_accepted"] == 1 and s["rejection_reasons"] == {"not_success": 1}
    assert s["yield_per_source"] == 0.25 and s["audits"] == {"n": 1, "n_accepted": 0}
    assert s["editor_usage"]["input_tokens"] == 400


def test_write_report_and_compare(tmp_path):
    rows = [row(ep(f"i{k}")) for k in range(3)] + [row(ep(f"i{k}", ckpt="c1", success=k != 1, total=(80, 8)), cycle=1) for k in range(3)]
    rows.append(row(ep("i0", role=EpisodeRole.COLLECT), panel="train"))
    cmp = paired_comparison(rows[:3], rows[3:6], "base", "c1")
    paths = write_report(tmp_path / "rep", rows, comparisons=[cmp], stage_efforts=[StageEffort(stage="training", cycle=0, optimizer_steps=2)], title="T")
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "lost in c1" in md and "1 previously successful pair(s) failed under c1" in md
    assert SMALL_N_NOTE in md and "No cost rates configured" in md
    with open(paths["episodes.csv"]) as f:
        assert len(list(csv.DictReader(f))) == 7
    with open(paths["summary_by_checkpoint_cycle_panel.csv"]) as f:
        assert {r["checkpoint"] for r in csv.DictReader(f)} == {"base", "c1"}  # eval rows only
    # compare() over run directories with an episode index
    for name, rs in (("run-a", rows[:3]), ("run-b", rows[3:6])):
        append_rows(tmp_path / name / "episodes.jsonl", rs)
    c = compare(tmp_path / "run-a", tmp_path / "run-b", out_dir=tmp_path / "cmp")
    assert c["transitions"]["lost"] == 1 and (tmp_path / "cmp" / "comparison.md").exists()
    assert json.loads((tmp_path / "cmp" / "comparison.json").read_text())["n_pairs"] == 3


def _item(run: Path, rel: str, name: str, obj) -> str:
    atomic_write_json(run / rel / name, obj)
    return rel


def test_write_run_report_from_coordinator_layout(tmp_path):
    run = tmp_path / "run-x"
    atomic_write_json(run / "run.json", {"run_id": "run-x", "instances": {}, "experiment": {"tasks": {"collection_panel": "train-panel"}}})
    for cycle, ck, succ in ((0, "base", [True, True]), (1, "c1", [True, False])):
        m = StageManifest(stage="eval", cycle=cycle, status="done", started_at="2026-01-01T00:00:00+00:00", finished_at="2026-01-01T00:01:00+00:00")
        for k, s in enumerate(succ):
            rel = _item(run, f"cycles/cycle-{cycle:03d}/eval/items/e{k}", "summary.json", ep(f"i{k}", ckpt=ck, success=s))
            m.items[f"e{k}"] = WorkItem(item_id=f"e{k}", status="done", output=rel, meta={"panel": "dev"})
        m.save(run / f"cycles/cycle-{cycle:03d}/eval/manifest.json")
    mv = StageManifest(stage="verify", cycle=0, status="done")
    rel = _item(run, "cycles/cycle-000/verify/items/v0", "verification.json", VerificationRecord(verification_id="v0", proposal_id="p", mode="continuation", acceptance_rule="r", accepted=True, reasons=["accepted"], operational_usage=Usage(input_tokens=50, output_tokens=5)))
    mv.items["v0"] = WorkItem(item_id="v0", status="done", output=rel)
    mv.save(run / "cycles/cycle-000/verify/manifest.json")
    atomic_write_json(run / "cycles/cycle-000/cycle.json", {"cycle": 0, "status": "done", "update": "trained", "reference": "base", "learner_in": {"checkpoint_id": "base"}, "learner_out": {"checkpoint_id": "c1"}, "checkpoint_record": {"optimizer_steps": 2, "n_train_examples": 1, "trainer": "fixture", "metrics": {"duration_sec": 0.5}}})
    rec = collect_run(run)
    assert len(rec.rows) == 4 and len(rec.verifications) == 1
    eff = {(e.stage, e.cycle): e for e in rec.efforts}
    assert eff[("evaluation", 0)].duration_sec == 60 and eff[("replay_verification", 0)].usage.total == 55
    assert eff[("training", 0)].optimizer_steps == 2
    md_path = write_run_report(run)
    md = md_path.read_text()
    assert md_path == run / "reports" / "report.md"
    assert "Paired comparison: cycle 0 -> cycle 1" in md and "lost in cycle 1" in md
    assert "DPO reference base" in md
    text = compare_runs(run, run)
    assert "Paired comparison" in text
