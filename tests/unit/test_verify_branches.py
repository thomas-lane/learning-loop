"""Replay specs, matched continuation seeds, branch cost arithmetic and acceptance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures" / "verify"))
import verify_builders as vb  # noqa: E402

from learning_loop.config import VerificationConfig  # noqa: E402
from learning_loop.editor import SourceContext  # noqa: E402
from learning_loop.events import read_events  # noqa: E402
from learning_loop.records import (  # noqa: E402
    BranchCost,
    EditProposal,
    EpisodeRole,
    ProposedCall,
    RestoreCapability,
    Usage,
)
from learning_loop.seeds import continuation_seed  # noqa: E402
from learning_loop.verify import (  # noqa: E402
    ContinuationVerifier,
    LocalVerifier,
    UnsupportedLocalContract,
    branch_replay_specs,
    build_replay_spec,
    check_local_contract,
    make_verifier,
)

EDIT_CMD = vb.PIPELINE + " | awk '{print $2}' > /app/answer.txt"
ROOT = 99


def counter(prompt, message, tools):
    """Deterministic test counter: 50 tokens for the original turn-0 call, 30 for anything else."""
    args = message["tool_calls"][0]["function"]["arguments"]
    return (50 if args == '{"command":"ls /app/logs"}' else 30), "test_counter"


def setup(tmp_path, turn_index=0, replacement=None, **src_kw):
    summary, path = vb.write_source_episode(tmp_path / "src", **src_kw)
    inst = vb.make_instance(tmp_path)
    src = SourceContext.load(summary, inst, path)
    prop = EditProposal(
        proposal_id="prop-1",
        source_episode_id=summary.episode_id,
        instance_id=inst.instance_id,
        editor_id="ed",
        status="proposed",
        turn_index=turn_index,
        tool_call_id=f"call_{turn_index}",
        replacement=replacement or ProposedCall(name="bash", arguments={"command": EDIT_CMD}),
        justification="merge",
    )
    return src, prop


def outcomes(orig=None, edit=None):
    """Default: original needs 3 continuation requests, edited needs 1; both succeed."""
    table = {
        "original": orig or {"success": True, "requests": [(1000, 20), (1100, 20), (1200, 10)]},
        "edited": edit or {"success": True, "requests": [(900, 10)]},
    }
    return lambda label, seed, plan: table[label]


def verifier(tmp_path, backend, src, **kw):
    return ContinuationVerifier(backend, {src.summary.episode_id: src}, root_seed=ROOT, out_root=tmp_path / "ver", intervention_counter=counter, **kw)


# --------------------------------------------------------------------------- #
# Replay specs
# --------------------------------------------------------------------------- #


def test_replay_spec_uses_only_the_source_prefix(tmp_path):
    src, prop = setup(tmp_path, turn_index=2)
    specs = branch_replay_specs(prop, src)
    events = read_events(Path(src.summary.events_path))
    req2 = next(e for e in events if e.kind.value == "request" and e.turn_index == 2)
    for label, spec in specs.items():
        assert spec.intervention_label == label and spec.intervention_turn == 2
        assert spec.history_prefix == req2.data["messages"]  # exact conversation before turn 2
        assert [t.turn_index for t in spec.prefix_turns] == [0, 1]
        assert spec.prefix_turns[1].expected_observations == [vb.DEFAULT_TURNS[1][2]]
        assert spec.prefix_turns[0].executed_arguments == [{"command": "ls /app/logs"}]
        assert spec.prefix_turns[1].expected_fingerprint_before == "fp-1"
        assert spec.expected_fingerprint_before_intervention == "fp-2"
        blob = json.dumps(spec.model_dump())
        # nothing observed at/after the decision (turn 2 result, turn 3, final answer)
        assert "42 10.0.0.7" not in blob and "Wrote 8 characters" not in blob and vb.FINAL_ANSWER not in blob
        assert spec.prefix_usage.total == vb.usage_for(0).total + vb.usage_for(1).total
        assert spec.intervention_request_input_tokens == vb.usage_for(2).input_tokens
    assert specs["original"].intervention_message == src.turns[2].assistant_message
    edited = specs["edited"].intervention_message
    assert edited["tool_calls"][0]["id"] == "call_2" and json.loads(edited["tool_calls"][0]["function"]["arguments"]) == {"command": EDIT_CMD}
    assert specs["original"].history_prefix == specs["edited"].history_prefix


def test_replay_spec_rejects_bad_turn(tmp_path):
    src, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        build_replay_spec(src.events, src.turns, 17, "original", {}, "src-ep")


# --------------------------------------------------------------------------- #
# Seeds, plans and cost arithmetic
# --------------------------------------------------------------------------- #


async def test_matched_seeds_equal_budgets_fresh_branches(tmp_path):
    src, prop = setup(tmp_path)
    backend = vb.FakeBackend(outcomes())
    v = verifier(tmp_path, backend, src, continuations_per_branch=2)
    rec = await v.verify(prop)
    assert len(backend.calls) == 4
    by = {(p.replay.intervention_label, p.seed): p for p, _ in backend.calls}
    for rep in (0, 1):
        seed = continuation_seed(ROOT, "prop-1", rep, "acceptance")
        o, e = by[("original", seed)], by[("edited", seed)]
        assert o.budgets == e.budgets == src.plan.budgets
        assert o.role == e.role == EpisodeRole.BRANCH and o.episode_id != e.episode_id
        assert o.replay.history_prefix == e.replay.history_prefix
    assert len({p.seed for p, _ in backend.calls}) == 2  # repetitions differ, labels do not
    assert len({d for _, d in backend.calls}) == 4  # fresh output dir per branch
    assert rec.accepted and rec.evidence_label == "all_2_continuation_pairs_successful"


async def test_cost_arithmetic_and_operational_usage(tmp_path):
    src, prop = setup(tmp_path, turn_index=1)
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes()), src).verify(prop)
    prefix = vb.usage_for(0).total  # request 0 in+out
    req_in = vb.usage_for(1).input_tokens  # request 1 input, shared, counted once
    o = next(b for b in rec.branches if b.branch == "original")
    e = next(b for b in rec.branches if b.branch == "edited")
    assert o.cost == BranchCost(shared_prefix_tokens=prefix, intervention_request_input_tokens=req_in, intervention_tokens=30, intervention_tokens_source="test_counter", continuation_tokens=3350)
    assert e.cost.continuation_tokens == 910 and e.cost.intervention_tokens == 30
    # the source's generated completion of turn 1 (output_tokens=21) is NOT added on top
    assert o.cost.total == prefix + req_in + 30 + 3350
    assert rec.mean_saving == o.cost.total - e.cost.total == 3350 - 910
    # verification spend = continuation tokens actually generated/consumed, separate from costs
    assert rec.operational_usage.total == 3350 + 910
    assert rec.accepted and rec.reasons == ["accepted"] and rec.evidence_label == "one_observed_successful_preference"
    assert rec.acceptance_rule == "strict_all_success_v1" and rec.mode == "continuation"


async def test_intervention_tokens_differ_by_branch(tmp_path):
    src, prop = setup(tmp_path, turn_index=0)
    same = {"success": True, "requests": [(500, 10)]}
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(same, same)), src).verify(prop)
    o = next(b for b in rec.branches if b.branch == "original")
    assert o.cost.intervention_tokens == 50 and rec.mean_saving == 20
    assert rec.accepted  # identical continuations; the shorter fixed action alone saves 20


# --------------------------------------------------------------------------- #
# Acceptance rule
# --------------------------------------------------------------------------- #


async def test_cheaper_but_failing_edit_is_rejected(tmp_path):
    src, prop = setup(tmp_path)
    fail = {"success": False, "requests": [(100, 5)]}
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(edit=fail)), src).verify(prop)
    assert not rec.accepted and "not_success:edited:r0" in rec.reasons
    assert rec.mean_saving and rec.mean_saving > 0  # it WAS cheaper; still rejected
    assert rec.evidence_label is None


async def test_original_failure_also_rejects(tmp_path):
    src, prop = setup(tmp_path)
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(orig={"success": False, "requests": [(5000, 5)]})), src).verify(prop)
    assert not rec.accepted and "not_success:original:r0" in rec.reasons


async def test_tie_rejected(tmp_path):
    src, prop = setup(tmp_path, replacement=ProposedCall(name="bash", arguments={"command": "ls -1 /app/logs"}))
    same = {"success": True, "requests": [(500, 10)]}

    def count_same(prompt, message, tools):
        return 40, "test_counter"

    v = ContinuationVerifier(vb.FakeBackend(outcomes(same, same)), {"src-ep": src}, root_seed=ROOT, out_root=tmp_path / "v", intervention_counter=count_same)
    rec = await v.verify(prop)
    assert not rec.accepted and rec.reasons == ["tie"] and rec.mean_saving == 0


async def test_negative_and_below_threshold_savings(tmp_path):
    src, prop = setup(tmp_path)
    slow = {"success": True, "requests": [(5000, 50)]}
    rec = await verifier(tmp_path / "a", vb.FakeBackend(outcomes(edit=slow)), src).verify(prop)
    assert not rec.accepted and any(r.startswith("no_saving:") for r in rec.reasons)
    rec = await verifier(tmp_path / "b", vb.FakeBackend(outcomes()), src, min_token_saving=10_000).verify(prop)
    assert not rec.accepted and any(r.startswith("saving_below_min:") for r in rec.reasons)
    rec = await verifier(tmp_path / "c", vb.FakeBackend(outcomes()), src, min_relative_saving=0.9).verify(prop)
    assert not rec.accepted and any(r.startswith("relative_saving_below_min:") for r in rec.reasons)


async def test_missing_cost_rejected(tmp_path):
    src, prop = setup(tmp_path)
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(edit={"success": True, "requests": [None]})), src).verify(prop)
    assert not rec.accepted and "missing_cost:edited:r0" in rec.reasons and rec.mean_saving is None
    # missing usage in the SOURCE prefix also makes every cost unavailable
    src2, prop2 = setup(tmp_path / "m", turn_index=2, missing_usage_turns=(0,))
    rec2 = await verifier(tmp_path / "m", vb.FakeBackend(outcomes()), src2).verify(prop2)
    assert not rec2.accepted and "missing_cost:original:r0" in rec2.reasons


async def test_replay_failure_and_bad_stops_rejected(tmp_path):
    src, prop = setup(tmp_path)
    rec = await verifier(tmp_path / "a", vb.FakeBackend(outcomes(edit={"replay_ok": False})), src).verify(prop)
    assert not rec.accepted and "replay_failed:edited:r0" in rec.reasons
    budget = {"success": True, "requests": [(100, 5)], "stop_category": "budget", "stop_reason": "budget:max_turns"}
    rec = await verifier(tmp_path / "b", vb.FakeBackend(outcomes(edit=budget)), src).verify(prop)
    assert not rec.accepted and "bad_stop:edited:r0:budget:max_turns" in rec.reasons
    infra = {"success": True, "requests": [(100, 5)], "stop_category": "infra", "stop_reason": "infra:docker"}
    rec = await verifier(tmp_path / "c", vb.FakeBackend(outcomes(orig=infra)), src).verify(prop)
    assert not rec.accepted and "bad_stop:original:r0:infra:docker" in rec.reasons


async def test_all_repetitions_must_succeed(tmp_path):
    src, prop = setup(tmp_path)
    seeds_seen = []

    def fn(label, seed, plan):
        seeds_seen.append(seed)
        first = seed == continuation_seed(ROOT, "prop-1", 0, "acceptance")
        if label == "edited" and not first:
            return {"success": False, "requests": [(10, 1)]}
        return {"success": True, "requests": [(1000, 10)] if label == "original" else [(100, 10)]}

    rec = await verifier(tmp_path, vb.FakeBackend(fn), src, continuations_per_branch=2).verify(prop)
    assert not rec.accepted and rec.reasons == ["not_success:edited:r1"]


async def test_restore_unsupported_fails_closed_without_running(tmp_path):
    src, prop = setup(tmp_path)
    backend = vb.FakeBackend(outcomes(), restore=RestoreCapability.NONE)
    rec = await verifier(tmp_path, backend, src).verify(prop)
    assert not rec.accepted and "restore_unsupported:none" in rec.reasons and backend.calls == []
    src2, prop2 = setup(tmp_path / "ap", spec=vb.state_spec(restore=RestoreCapability.APPROXIMATE_REPLAY))
    rec = await verifier(tmp_path / "ap", vb.FakeBackend(outcomes()), src2).verify(prop2)
    assert "state_spec_restore:approximate_replay" in rec.reasons


async def test_idempotent_and_audit_is_separate(tmp_path):
    src, prop = setup(tmp_path)
    backend = vb.FakeBackend(outcomes())
    v = verifier(tmp_path, backend, src)
    a = await v.verify(prop)
    b = await v.verify(prop)
    assert a == b and len(backend.calls) == 2  # completed logical work is not redone
    audit = await v.verify(prop, purpose="audit")
    assert audit.purpose == "audit" and audit.verification_id != a.verification_id
    audit_seeds = {p.seed for p, _ in backend.calls[2:]}
    assert audit_seeds == {continuation_seed(ROOT, "prop-1", 0, "audit")} and audit_seeds.isdisjoint({p.seed for p, _ in backend.calls[:2]})
    assert (tmp_path / "ver" / "audit" / audit.verification_id / "verification.json").exists()


async def test_invalid_proposal_is_never_executed(tmp_path):
    src, prop = setup(tmp_path)
    backend = vb.FakeBackend(outcomes())
    with pytest.raises(ValueError):
        await verifier(tmp_path, backend, src).verify(prop.model_copy(update={"status": "invalid"}))
    assert backend.calls == []


async def test_coordinator_entry_points(tmp_path):
    """make_verifier + verify(source_dir=..., out_dir=...) as the coordinator calls them."""
    summary, path = vb.write_source_episode(tmp_path / "item")
    (tmp_path / "item" / "summary.json").write_text(summary.model_copy(update={"events_path": "events.jsonl"}).model_dump_json())
    inst = vb.make_instance(tmp_path)
    _, prop = setup(tmp_path / "x")
    backend = vb.FakeBackend(outcomes())
    seen = []

    def plan_factory(instance, role, eid, seed, replay):
        seen.append(eid)
        return vb.make_plan(eid).model_copy(update={"role": role, "seed": seed, "replay": replay})

    v = make_verifier(mode="continuation", backend=backend, config=VerificationConfig(), policy_spec=None, plan_factory=plan_factory, root_seed=ROOT, scripted=True)
    rec = await v.verify(proposal=prop, verification_id="ver-x", source_dir=tmp_path / "item", instance=inst, out_dir=tmp_path / "out", purpose="acceptance")
    assert rec.verification_id == "ver-x" and rec.accepted and len(seen) == 2
    assert rec.branches[0].cost.intervention_tokens_source.startswith("fixture_estimate")
    assert (tmp_path / "out" / "verification.json").exists()

    def drifting(instance, role, eid, seed, replay):
        return vb.make_plan(eid).model_copy(update={"instruction": "something else", "replay": replay})

    v2 = make_verifier(mode="continuation", backend=backend, config=VerificationConfig(), plan_factory=drifting, root_seed=ROOT, scripted=True)
    with pytest.raises(ValueError, match="differs from the source plan"):
        await v2.verify(proposal=prop, source_dir=tmp_path / "item", instance=inst, out_dir=tmp_path / "out2")


# --------------------------------------------------------------------------- #
# Local verification
# --------------------------------------------------------------------------- #

LOCAL_TURNS = [
    ("bash", {"command": "ls /app/logs"}, "[exit code 0]\naccess.log\naccess.log.1\naccess.log.2.gz"),
    ("bash", {"command": "echo        10.0.0.7         > /app/answer.txt"}, "[exit code 0]"),
]


async def test_local_verifier_same_state(tmp_path):
    initial = await vb.FakeSession().fingerprint(None)  # type: ignore[arg-type]
    spec = vb.state_spec(local_equivalence="same_state_after_action")
    summary, path = vb.write_source_episode(tmp_path / "src", turns=LOCAL_TURNS, spec=spec, fingerprints=[initial, initial, "after"])
    inst = vb.make_instance(tmp_path)
    src = SourceContext.load(summary, inst, path)

    def prop(cmd):
        return EditProposal(proposal_id="lp", source_episode_id="src-ep", instance_id=inst.instance_id, editor_id="ed", status="proposed", turn_index=1, tool_call_id="call_1", replacement=ProposedCall(name="bash", arguments={"command": cmd}))

    factory = vb.fake_session_factory()
    lv = LocalVerifier(factory, {"src-ep": src}, out_root=tmp_path / "lv")
    rec = await lv.verify(prop("echo 10.0.0.7 > /app/answer.txt"))
    assert rec.mode == "local" and rec.acceptance_rule == "local_same_state_v1"
    assert rec.accepted, rec.reasons
    assert rec.evidence_label == "local_same_state_after_action" and rec.mean_saving > 0
    assert all(b.cost.continuation_tokens is None for b in rec.branches)
    assert len(factory.sessions) == 2  # two freshly restored environments
    assert all(s.executed[0] == {"name": "bash", "command": "ls /app/logs"} for s in factory.sessions)

    rec = await LocalVerifier(vb.fake_session_factory(), {"src-ep": src}, out_root=tmp_path / "lv2").verify(prop("echo 10.0.0.8 > /app/answer.txt"))
    assert not rec.accepted and "state_differs_after_action" in rec.reasons

    rec = await LocalVerifier(vb.fake_session_factory(fail_on="10.0.0.9"), {"src-ep": src}, out_root=tmp_path / "lv3").verify(prop("echo 10.0.0.9 > /app/answer.txt"))
    assert not rec.accepted and "action_error:edited" in rec.reasons


async def test_local_verifier_replay_mismatch_fails_closed(tmp_path):
    spec = vb.state_spec(local_equivalence="same_state_after_action")
    summary, path = vb.write_source_episode(tmp_path / "src", turns=LOCAL_TURNS, spec=spec, fingerprints=["wrong", "wrong", "x"])
    src = SourceContext.load(summary, vb.make_instance(tmp_path), path)
    p = EditProposal(proposal_id="lp", source_episode_id="src-ep", instance_id="log-triage/easy/s0", editor_id="ed", status="proposed", turn_index=1, tool_call_id="call_1", replacement=ProposedCall(name="bash", arguments={"command": "echo 10.0.0.7 > /app/answer.txt"}))
    factory = vb.fake_session_factory()
    rec = await LocalVerifier(factory, {"src-ep": src}, out_root=tmp_path / "lv").verify(p)
    assert not rec.accepted and "replay_failed:original:r0" in rec.reasons
    assert all(s.executed == [] for s in factory.sessions)  # nothing executed after the mismatch


def test_unsupported_local_contract_is_a_validation_error():
    with pytest.raises(UnsupportedLocalContract):
        check_local_contract(vb.state_spec())
    with pytest.raises(UnsupportedLocalContract):
        check_local_contract(vb.state_spec(local_equivalence="exit_code_zero"))
    assert check_local_contract(vb.state_spec(local_equivalence="same_state_after_action")) == "same_state_after_action"
    with pytest.raises(UnsupportedLocalContract):
        make_verifier(mode="local", backend=vb.FakeBackend(outcomes()), config=VerificationConfig(mode="local"), root_seed=0)


def test_usage_sum_keeps_missing_missing():
    assert Usage.sum([Usage(input_tokens=1, output_tokens=2), Usage(input_tokens=None, output_tokens=3)]).total is None
