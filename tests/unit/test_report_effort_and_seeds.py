"""Infra retries / usage coverage in stage effort, editor infra errors, success-conditioned
branch savings, replay vs continuation effort, token-unit checks and multi-seed comparisons."""

from __future__ import annotations

import pytest

from learning_loop.metrics import (
    FEW_SEEDS_NOTE,
    INCOMPARABLE_TOKENS_NOTE,
    CostRate,
    EpisodeRow,
    across_seeds,
    append_rows,
    branch_saving_summary,
    paired_comparison,
    proposal_stats,
    stage_effort_table,
    verification_effort_split,
)
from learning_loop.records import BranchCost, BranchResult, EditProposal, EpisodeRole, EpisodeSummary, StopCategory, Timing, Usage, VerificationRecord
from learning_loop.report import collect_run, compare_conditions, load_run_rows, write_run_report
from learning_loop.storage import StageManifest, WorkItem, atomic_write_json
from test_report_metrics import ep, row


# --------------------------------------------------------------------------- #
# Stage effort: retries, coverage, active time
# --------------------------------------------------------------------------- #


def _eval_run(tmp_path):
    run = tmp_path / "run-r"
    atomic_write_json(run / "run.json", {"run_id": "run-r", "instances": {}, "experiment": {"seeds": {"loop_seed": 3}}, "model_profiles": {"learner": {"name": "qwen-x", "base_revision": "abcdef0123456789"}}})
    stage = run / "cycles/cycle-000/eval"
    m = StageManifest(
        stage="eval", cycle=0, status="done", started_at="2026-01-01T00:00:00+00:00", finished_at="2026-01-01T05:00:00+00:00",
        active_intervals=[["2026-01-01T00:00:00+00:00", "2026-01-01T00:01:00+00:00"], ["2026-01-01T04:59:00+00:00", "2026-01-01T05:00:00+00:00"]],
    )
    atomic_write_json(stage / "items/e0/summary.json", ep("i0", total=(100, 10)))
    atomic_write_json(stage / "items/e1/summary.json", ep("i1", total=None, success=None, stop=StopCategory.INFRA))  # no usage
    # e0 was retried once after an infra failure; the failed attempt spent 40+2 tokens
    atomic_write_json(stage / "items/e0.interrupted-1/summary.json", ep("i0", total=(40, 2), success=None, stop=StopCategory.INFRA))
    m.items["e0"] = WorkItem(item_id="e0", status="done", output="cycles/cycle-000/eval/items/e0", attempts=2, infra_failures=1, interrupted_dirs=["e0.interrupted-1"], meta={"panel": "dev"})
    m.items["e1"] = WorkItem(item_id="e1", status="done", output="cycles/cycle-000/eval/items/e1", attempts=1, meta={"panel": "dev"})
    m.save(stage / "manifest.json")
    return run


def test_stage_effort_reports_retries_coverage_and_active_time(tmp_path):
    run = _eval_run(tmp_path)
    rec = collect_run(run)
    (eff,) = rec.efforts
    assert eff.extra["attempts"] == 3 and eff.extra["infra_failures"] == 1 and eff.extra["n_retried_attempts"] == 1
    assert eff.duration_sec == 120 and eff.extra["wall_span_sec"] == 5 * 3600  # the resume gap is not active time
    (r,) = stage_effort_table(rec.efforts)
    # one item without usage no longer blanks the stage: available sum + coverage
    assert r["total_tokens"] == 110 and r["tokens_coverage"] == "1/2"
    assert r["retried_total_tokens"] == 42 and r["retried_tokens_coverage"] == "1/1"
    # token money is not computed from partial coverage
    (priced,) = stage_effort_table(rec.efforts, {"evaluation": CostRate(usd_per_million_input_tokens=1.0, provenance="quote")})
    assert priced["cost_usd"] is None and "incomplete" in priced["cost_provenance"]
    md = write_run_report(run).read_text()
    assert "infra retries" in md and "retried attempts" in md and "items with tokens" in md
    assert all(r.model_identity == "qwen-x@abcdef012345" for r in rec.rows)


# --------------------------------------------------------------------------- #
# Proposals: editor infra errors are not invalid proposals
# --------------------------------------------------------------------------- #


def _prop(i, status, reasons=(), src=None):
    return EditProposal(proposal_id=f"p{i}", source_episode_id=src or f"s{i}", instance_id="i", editor_id="e", status=status, rejection_reasons=list(reasons))


def test_proposal_stats_separates_editor_infra_errors():
    props = [_prop(1, "proposed"), _prop(2, "invalid", ["ungrounded_constant:17"]), _prop(3, "invalid", ["editor_infra_error:ConnectError: refused"])]
    ver = [VerificationRecord(verification_id="v", proposal_id="p1", mode="continuation", acceptance_rule="r", accepted=True, reasons=["accepted"])]
    s = proposal_stats(props, ver)
    assert s["n_invalid"] == 1 and s["invalid_reasons"] == {"ungrounded_constant": 1}
    assert s["n_editor_infra_errors"] == 1 and s["editor_infra_errors"] == ["editor_infra_error:ConnectError: refused"]
    assert s["yield_per_source"] == pytest.approx(1 / 3)  # lost proposals are lost yield
    assert s["n_sources_with_editor_output"] == 2 and s["yield_per_source_with_editor_output"] == 0.5


# --------------------------------------------------------------------------- #
# Branch savings conditioned on success, split by mode; replay vs continuation
# --------------------------------------------------------------------------- #


def _br(label, success, cont, rep=0, stop=StopCategory.MODEL, extra=None, timing=None):
    e = EpisodeSummary(
        episode_id=f"{label}{rep}", role=EpisodeRole.BRANCH, instance_id="i", checkpoint_id="base", stop_reason="x", stop_category=stop,
        success=success, usage=Usage(input_tokens=cont, output_tokens=0), n_requests=1, n_tool_calls=1, extra=extra or {}, timing=timing or Timing(),
    )
    c = BranchCost(shared_prefix_tokens=100, intervention_request_input_tokens=10, intervention_tokens=10, intervention_tokens_source="t", continuation_tokens=cont)
    return BranchResult(branch=label, repetition=rep, continuation_seed=1, episode=e, replay_ok=True, cost=c)


def _ver(vid, branches, accepted=False, mode="continuation", saving=None):
    o = [b.cost.total for b in branches if b.branch == "original"]
    e = [b.cost.total for b in branches if b.branch == "edited"]
    ms = saving if saving is not None else (sum(o) / len(o) - sum(e) / len(e))
    return VerificationRecord(verification_id=vid, proposal_id=vid, mode=mode, acceptance_rule="r", branches=branches, accepted=accepted, reasons=["accepted"] if accepted else ["x"], mean_saving=ms)


def test_branch_saving_is_conditioned_on_success_and_split_by_mode():
    vs = [
        _ver("ok", [_br("original", True, 500), _br("edited", True, 300)], accepted=True),  # saving 200
        _ver("lost", [_br("original", True, 500), _br("edited", False, 10)]),  # "saving" 490 from a FAILED edit
        _ver("rev", [_br("original", False, 500), _br("edited", True, 400)]),
        _ver("early", [_br("original", True, 500), _br("edited", True, 5, stop=StopCategory.BUDGET)]),
        _ver("loc", [_br("original", None, 0, extra={"executed": True}), _br("edited", None, 0, extra={"executed": True})], accepted=True, mode="local", saving=7.0),
    ]
    s = branch_saving_summary(vs)
    by = {r["mode"]: r for r in s["by_mode"]}
    c = by["continuation"]
    assert c["n_verifications"] == 4 and c["n_all_branches_ok"] == 1 and c["mean_saving_all_branches_ok"] == 200
    assert c["n_edited_failed_original_ok"] == 2  # failed + budget-stopped edits
    assert c["n_original_failed_edited_ok"] == 1 and c["n_branch_pairs"] == 4
    loc = by["local"]
    assert loc["n_all_branches_ok"] == 1 and loc["mean_saving_all_branches_ok"] == 7.0 and "intervention tokens only" in loc["saving_unit"]


def test_replay_vs_continuation_effort_split():
    plain = [_ver("a", [_br("original", True, 500, timing=Timing(tool_sec=3.0)), _br("edited", True, 300, timing=Timing(tool_sec=2.0))])]
    out = verification_effort_split(plain)
    assert not out["available"] and "unavailable" in out["note"] and out["continuation_tokens"]["sum"] == 800
    timed = [_ver("b", [
        _br("original", True, 500, extra={"replay_tool_sec": 1.0, "replay_cpu_sec": 0.5}, timing=Timing(tool_sec=3.0)),
        _br("edited", True, 300, extra={"replay_tool_sec": 1.0}, timing=Timing(tool_sec=2.0)),
    ])]
    out = verification_effort_split(timed)
    assert out["available"] and out["replay_tool_sec"]["sum"] == 2.0 and out["continuation_tool_sec"]["sum"] == 3.0
    assert out["replay_tokens"] == 0 and out["replay_cpu_sec"] == {"sum": 0.5, "mean": 0.5, "n": 1, "n_missing": 1}


# --------------------------------------------------------------------------- #
# Token units
# --------------------------------------------------------------------------- #


def test_token_delta_none_for_different_models_or_usage_sources():
    a = [row(ep("i1", ckpt="base:qwen-a@111111111111")), row(ep("i2", ckpt="base:qwen-a@111111111111"))]
    b = [row(ep("i1", total=(50, 5), ckpt="base:llama-b@222222222222")), row(ep("i2", success=False, ckpt="base:llama-b@222222222222"))]
    c = paired_comparison(a, b, "A", "B")
    assert not c["token_units"]["comparable"] and c["token_units"]["note"].startswith(INCOMPARABLE_TOKENS_NOTE)
    assert c["token_ratio_both_succeed"] is None and c["token_delta_instance_level"]["mean"] is None
    assert c["token_delta_instance_level"]["interval"] == INCOMPARABLE_TOKENS_NOTE
    assert all(p["token_delta_both_succeed"] is None for p in c["pairs"])
    assert c["transitions"]["lost"] == 1  # success transitions are still reported
    # same model, different usage source (provider vs local recount)
    b2 = [row(ep("i1", total=(50, 5), ckpt="base:qwen-a@111111111111").model_copy(update={"usage": Usage(input_tokens=50, output_tokens=5, source="local_recount")}))]
    c2 = paired_comparison(a[:1], b2)
    assert c2["token_ratio_both_succeed"] is None and "usage source differs" in c2["token_units"]["note"]
    # explicit identity on trained-checkpoint rows
    ra = [EpisodeRow(episode=ep("i1", ckpt="run/c001-aa"), panel="dev", model_identity="qwen-a@111111111111")]
    rb = [EpisodeRow(episode=ep("i1", total=(50, 5), ckpt="run/c002-bb"), panel="dev", model_identity="qwen-a@111111111111")]
    c3 = paired_comparison(ra, rb)
    assert c3["token_units"]["comparable"] and c3["token_units"]["note"] is None and c3["token_ratio_both_succeed"] == pytest.approx(55 / 110)
    # unknown identity: computed, but flagged as unverified
    c4 = paired_comparison([row(ep("i1", ckpt="run/c1"))], [row(ep("i1", total=(50, 5), ckpt="run/c2"))])
    assert c4["token_ratio_both_succeed"] is not None and "unknown" in c4["token_units"]["note"]


def test_episode_index_rows_get_identity_from_run_json(tmp_path):
    run = tmp_path / "r"
    atomic_write_json(run / "run.json", {"run_id": "r", "model_profiles": {"learner": {"name": "p", "base_revision": "0123456789abcdef"}}})
    append_rows(run / "episodes.jsonl", [row(ep("i1"))])
    assert load_run_rows(run)[0].model_identity == "p@0123456789ab"


# --------------------------------------------------------------------------- #
# Multi-seed comparison
# --------------------------------------------------------------------------- #


def _cond_rows(ckpt, successes, tokens=100):
    return [row(ep(f"i{k}", ckpt=ckpt, success=s, total=(tokens, 0))) for k, s in enumerate(successes)]


def test_across_seeds_interval_threshold():
    assert across_seeds([0.1, 0.3])["interval"].startswith(FEW_SEEDS_NOTE) and across_seeds([0.1, 0.3])["ci95_low"] is None
    x = across_seeds([0.1, 0.2, 0.3])
    assert x["mean"] == pytest.approx(0.2) and x["sd"] == pytest.approx(0.1) and x["ci95_low"] < 0.2 < x["ci95_high"]
    assert across_seeds([None, 0.5])["n_missing"] == 1 and across_seeds([])["interval"] == "no data"


def test_compare_conditions_per_seed_then_across_seeds(tmp_path):
    base = _cond_rows("c0", [True, False, False, True], tokens=200)
    runs_a, runs_b = [], []
    for seed, succ in ((0, [True, True, False, True]), (1, [True, False, True, True]), (2, [True, True, True, True])):
        for cond, rows, lst in (("learn", _cond_rows(f"c{seed}", succ, tokens=150), runs_a), ("frozen", base, runs_b)):
            d = tmp_path / f"{cond}-s{seed}"
            atomic_write_json(d / "run.json", {"run_id": d.name, "experiment": {"seeds": {"loop_seed": seed}}, "model_profiles": {"learner": {"name": "p", "base_revision": "r" * 12}}})
            append_rows(d / "episodes.jsonl", rows)
            lst.append(d)
    c = compare_conditions(list(reversed(runs_a)), runs_b, label_a="learning", label_b="frozen", out_dir=tmp_path / "out")
    assert c["pairing"] == "matched by loop seed" and [p["seed"] for p in c["per_seed"]] == [0, 1, 2]
    # frozen - learning per seed: seed0 -1/4, seed1 -1/4, seed2 -2/4
    assert [p["success_delta"] for p in c["per_seed"]] == [-0.25, -0.25, -0.5]
    assert c["success_delta"]["mean"] == pytest.approx(-1 / 3) and c["success_delta"]["ci95_low"] is not None
    assert c["token_delta_both_succeed"]["mean"] == 50  # frozen used more tokens on both-succeed pairs
    assert (tmp_path / "out" / "conditions.md").exists() and "across loop seeds" in c["markdown"]
    # two seeds: no interval, explicit note
    c2 = compare_conditions(runs_a[:2], runs_b[:2])
    assert c2["success_delta"]["ci95_low"] is None and FEW_SEEDS_NOTE in c2["markdown"]
    assert "not evidence about the method" in c2["markdown"]
    # one frozen baseline run against several learning seeds
    c3 = compare_conditions(runs_a, runs_b[:1])
    assert c3["n_seeds"] == 3 and c3["pairing"] == "every a run vs the single b run"
    with pytest.raises(ValueError):
        compare_conditions(runs_a[:2], runs_b)
