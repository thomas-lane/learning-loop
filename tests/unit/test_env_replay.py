"""Deterministic replay + branching on the local fixture backend (scripted policy)."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
from _env_helpers import count_errors_instance, make_plan, scripted_spec, tool_call_message

from learning_loop.backends import LocalFixtureBackend
from learning_loop.episode import apply_normalizers, build_replay_spec
from learning_loop.events import load_turns, read_events
from learning_loop.records import EventKind, RestoreCapability, StopCategory
from learning_loop.tasks import load_state_spec

EDIT = '{"command": "grep -h ERROR data/*.log | wc -l"}'


@pytest.fixture()
async def source(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    res = await LocalFixtureBackend().run(inst, make_plan(inst, "ep-src", seed=11), tmp_path / "src")
    assert res.summary.success
    return inst, res


def _obs(events_path: Path) -> list[str]:
    return [e.data["observation"] for e in read_events(events_path) if e.kind == EventKind.TOOL_RESULT]


async def test_original_branch_reproduces_prefix_and_state(tmp_path, source):
    inst, src = source
    ev = src.out_dir / "events.jsonl"
    src_turns = load_turns(ev)
    spec = build_replay_spec(ev, 1, "original", src_turns[1].assistant_message)
    assert spec.expected_fingerprint_before_intervention == src_turns[1].fingerprint_before
    assert spec.prefix_usage.total == src_turns[0].usage.total
    assert spec.intervention_request_input_tokens == src_turns[1].usage.input_tokens
    # the prefix is exactly the request of turn 1: nothing downstream of the intervention
    assert len(spec.history_prefix) == 4 and spec.history_prefix[-1]["role"] == "tool"
    res = await LocalFixtureBackend().run(inst, make_plan(inst, "br-orig", replay=spec, seed=99), tmp_path / "orig")
    s = res.summary
    assert res.replay_ok is True and s.success is True
    evs = read_events(res.out_dir / "events.jsonl")
    check = next(e for e in evs if e.kind == EventKind.REPLAY_CHECK)
    assert check.data["ok"] and check.data["mismatches"] == []
    turns = load_turns(evs)
    assert [t.origin for t in turns][:2] == ["replayed", "intervention_original"]
    # replayed/intervention turns made no model requests; fresh observations match the source
    assert turns[0].request_seq is None and turns[1].request_seq is None
    assert turns[0].tool_executions[0].observation == src_turns[0].tool_executions[0].observation
    assert turns[1].fingerprint_before == src_turns[1].fingerprint_before
    assert turns[1].tool_executions[0].observation == src_turns[1].tool_executions[0].observation
    # continuation: deterministic fixture policy -> same actions as the source suffix
    assert [t.assistant_message for t in turns[2:]] == [t.assistant_message for t in src_turns[2:]]
    assert s.n_requests == len(src_turns) - 2
    assert s.extra["n_prefix_turns"] == 1


async def test_edited_branch_continues_sensibly(tmp_path, source):
    inst, src = source
    ev = src.out_dir / "events.jsonl"
    spec = build_replay_spec(ev, 1, "edited", tool_call_message("call_1_0", "bash", EDIT))
    res = await LocalFixtureBackend().run(inst, make_plan(inst, "br-edit", replay=spec, seed=99), tmp_path / "edit")
    s = res.summary
    assert res.replay_ok and s.success and s.stop_reason == "model_finished"
    turns = load_turns(res.out_dir / "events.jsonl")
    assert turns[1].origin == "intervention_edited"
    assert turns[1].tool_executions[0].executed_arguments == {"command": "grep -h ERROR data/*.log | wc -l"}
    # fresh observation after the edit (not the source's downstream observation)
    assert turns[1].tool_executions[0].observation.split()[-1] == str(inst.params["expected"])
    assert s.n_requests == 2  # write + finish; the source needed 3 after turn 1
    assert s.usage.total < src.summary.usage.total


async def test_perturbed_environment_is_rejected_before_branching(tmp_path, source):
    inst, src = source
    ev = src.out_dir / "events.jsonl"
    spec = build_replay_spec(ev, 2, "original", load_turns(ev)[2].assistant_message)
    # a "fresh" environment whose inputs differ from the source (one extra ERROR line)
    bad = inst.model_copy(update={"task_dir": str(tmp_path / "bad-task")})
    shutil.copytree(inst.task_dir, bad.task_dir)
    log = Path(bad.task_dir) / "environment/files/data/app.log"
    log.write_text(log.read_text() + "2026-09-01T00:00:00Z ERROR injected\n")
    res = await LocalFixtureBackend().run(bad, make_plan(bad, "br-bad", replay=spec), tmp_path / "bad")
    assert res.replay_ok is False
    assert res.summary.stop_category == StopCategory.REPLAY
    # the environment's build inputs differ from the source's recorded image identity
    assert spec.expected_image_identity is not None and spec.expected_image_identity["kind"] == "local_fixture"
    assert res.summary.stop_reason == "replay:image_mismatch"
    assert res.summary.extra["image_identity_check"] == "mismatch"
    # without a recorded source identity the check is skipped (recorded), and the fresh
    # environment's declared state already differs before the first replayed action
    spec.expected_image_identity = None
    res1 = await LocalFixtureBackend().run(bad, make_plan(bad, "br-bad1", replay=spec), tmp_path / "bad1")
    assert res1.summary.stop_reason == "replay:fingerprint:turn=0"
    assert res1.summary.extra["image_identity_check"].startswith("skipped")
    # with per-turn source fingerprints removed, the replayed observation catches it instead
    for t in spec.prefix_turns:
        t.expected_fingerprint_before = None
    res2 = await LocalFixtureBackend().run(bad, make_plan(bad, "br-bad2", replay=spec), tmp_path / "bad2")
    assert res2.summary.stop_reason == "replay:observation:turn=1:call=0"
    detail = next(e for e in read_events(res2.out_dir / "events.jsonl") if e.kind == EventKind.REPLAY_CHECK).data["details"][0]
    assert "injected" in detail["actual"] and "injected" not in detail["expected"]
    kinds = [e.kind for e in read_events(res.out_dir / "events.jsonl")]
    assert EventKind.REQUEST not in kinds and EventKind.INTERVENTION not in kinds
    assert res.summary.n_requests == 0


async def test_fingerprint_and_prefix_tampering_rejected(tmp_path, source):
    inst, src = source
    ev = src.out_dir / "events.jsonl"
    turns = load_turns(ev)
    msg = tool_call_message("call_1_0", "bash", EDIT)

    spec = build_replay_spec(ev, 1, "edited", msg)
    spec.expected_fingerprint_before_intervention = "0" * 64
    r1 = await LocalFixtureBackend().run(inst, make_plan(inst, "fp", replay=spec), tmp_path / "fp")
    assert r1.summary.stop_reason == "replay:fingerprint:before_intervention"
    assert EventKind.INTERVENTION not in [e.kind for e in read_events(r1.out_dir / "events.jsonl")]

    spec2 = build_replay_spec(ev, 1, "edited", msg)
    spec2.history_prefix = copy.deepcopy(spec2.history_prefix)
    spec2.history_prefix[-1]["content"] = "[exit code 0]\nsomething else"
    r2 = await LocalFixtureBackend().run(inst, make_plan(inst, "hp", replay=spec2), tmp_path / "hp")
    assert r2.summary.stop_reason.startswith("replay:history_prefix_observations")

    spec3 = build_replay_spec(ev, 1, "edited", {"role": "assistant", "content": "", "tool_calls": msg["tool_calls"] * 2})
    r3 = await LocalFixtureBackend().run(inst, make_plan(inst, "two", replay=spec3), tmp_path / "two")
    assert r3.summary.stop_reason == "replay:invalid_intervention"

    plan4 = make_plan(inst, "nr", replay=build_replay_spec(ev, 1, "edited", msg))
    plan4.state_spec = plan4.state_spec.model_copy(update={"restore": RestoreCapability.NONE})
    r4 = await LocalFixtureBackend().run(inst, plan4, tmp_path / "nr")
    assert r4.summary.stop_reason == "replay:restore_unsupported:none"
    assert turns  # source unchanged


async def test_budgets_count_prefix_and_intervention(tmp_path, source):
    inst, src = source
    ev = src.out_dir / "events.jsonl"
    spec = build_replay_spec(ev, 2, "original", load_turns(ev)[2].assistant_message)
    # turns: 2 replayed + 1 intervention = 3 -> max_turns=3 leaves no model turn
    r1 = await LocalFixtureBackend().run(inst, make_plan(inst, "b1", replay=spec, max_turns=3), tmp_path / "b1")
    assert r1.replay_ok and r1.summary.stop_reason == "budget:max_turns" and r1.summary.n_requests == 0
    used = spec.prefix_usage.total + spec.intervention_request_input_tokens
    r2 = await LocalFixtureBackend().run(inst, make_plan(inst, "b2", replay=spec, max_episode_tokens=used), tmp_path / "b2")
    assert r2.summary.stop_reason == "budget:max_episode_tokens" and r2.summary.n_requests == 0
    r3 = await LocalFixtureBackend().run(inst, make_plan(inst, "b3", replay=spec, max_episode_tokens=used + 10_000), tmp_path / "b3")
    assert r3.summary.success and r3.summary.n_requests == 2


async def test_matched_seeds_across_branches(tmp_path, source):
    inst, src = source
    ev = src.out_dir / "events.jsonl"
    outs = {}
    for label, msg in (("original", load_turns(ev)[1].assistant_message), ("edited", tool_call_message("call_1_0", "bash", EDIT))):
        res = await LocalFixtureBackend().run(inst, make_plan(inst, f"s-{label}", replay=build_replay_spec(ev, 1, label, msg), seed=1234), tmp_path / label)
        outs[label] = [e.data["seed"] for e in read_events(res.out_dir / "events.jsonl") if e.kind == EventKind.REQUEST]
    n = min(len(outs["original"]), len(outs["edited"]))
    assert outs["original"][:n] == outs["edited"][:n] and n >= 2


async def test_normalizers_are_task_declared_and_recorded(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    spec = load_state_spec(Path(inst.task_dir))
    assert [n["replacement"] for n in spec.observation_normalizers] == ["<mtime>"]
    assert apply_normalizers("-rw-r--r-- 1 u g 3 Sep 28 17:14 answer.txt", spec.observation_normalizers) == "-rw-r--r-- 1 u g 3 <mtime> answer.txt"
    assert apply_normalizers("ERROR 12:30 happened", spec.observation_normalizers) == "ERROR 12:30 happened"

    # A volatile, ls-style timestamp that is guaranteed to differ between source and replay
    # (a host-side counter outside the fingerprinted work dir).
    counter = tmp_path / "counter"
    cmd = f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}; echo "answer.txt Jan  $n 10:00"'
    script = tmp_path / "ls.yaml"
    script.write_text(f"""
schema: scripted_policy/v1
rules:
  - {{name: done, when: {{min_turn: 2}}, respond: {{content: done}}}}
  - {{name: stamp, when: {{turn: 0}}, respond: {{tool_calls: [{{name: bash, arguments: {{command: '{cmd}'}}}}]}}}}
  - {{name: next, when: {{turn: 1}}, respond: {{tool_calls: [{{name: bash, arguments: {{command: "true"}}}}]}}}}
""")
    backend = LocalFixtureBackend()
    src = await backend.run(inst, make_plan(inst, "n-src", scripted_spec(script)), tmp_path / "nsrc")
    ev = src.out_dir / "events.jsonl"
    assert "Jan  1 10:00" in load_turns(ev)[0].tool_executions[0].observation
    rspec = build_replay_spec(ev, 1, "original", load_turns(ev)[1].assistant_message)
    res = await backend.run(inst, make_plan(inst, "n-br", scripted_spec(script), replay=rspec), tmp_path / "nbr")
    assert res.replay_ok, res.replay_mismatches  # "Jan  2 10:00" vs "Jan  1 10:00": only the declared field differs
    check = next(e for e in read_events(res.out_dir / "events.jsonl") if e.kind == EventKind.REPLAY_CHECK)
    assert check.data["normalizers"] == spec.observation_normalizers
    # without the declared normalizer the same replay fails closed
    plan = make_plan(inst, "n-br2", scripted_spec(script), replay=rspec)
    plan.state_spec = plan.state_spec.model_copy(update={"observation_normalizers": []})
    res2 = await backend.run(inst, plan, tmp_path / "nbr2")
    assert res2.replay_ok is False and res2.summary.stop_reason == "replay:observation:turn=0:call=0"
