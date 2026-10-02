"""Episode loop on the local fixture backend with scripted policies (no Docker, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _env_helpers import COUNT_POLICY, FIXTURES, count_errors_instance, make_plan, scripted_spec

from learning_loop.core.records import EventKind, PolicySpec, StopCategory
from learning_loop.episodes.backends import LocalFixtureBackend
from learning_loop.episodes.events import load_turns, read_events


@pytest.fixture()
def instance(tmp_path):
    return count_errors_instance(tmp_path / "tasks")


def _script(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / f"{name}.yaml"
    p.write_text(text)
    return p


async def test_source_episode_succeeds_and_records_everything(tmp_path, instance):
    backend = LocalFixtureBackend()
    res = await backend.run(instance, make_plan(instance, "ep-src"), tmp_path / "out")
    s = res.summary
    assert s.stop_reason == "model_finished" and s.stop_category == StopCategory.MODEL
    assert s.reward == {"reward": 1.0} and s.partial_reward == 1.0 and s.success is True
    assert s.n_requests == 5 and s.n_tool_calls == 4
    assert s.usage.source == "fixture_estimate" and s.usage.total is not None
    evs = read_events(res.out_dir / "events.jsonl")
    kinds = [e.kind for e in evs]
    assert kinds[0] == EventKind.EPISODE_START and kinds[-1] == EventKind.EPISODE_END
    assert kinds.count(EventKind.REQUEST) == 5 and kinds.count(EventKind.RESPONSE) == 5
    assert kinds.count(EventKind.FINGERPRINT) == 5  # before every model turn
    req0 = next(e for e in evs if e.kind == EventKind.REQUEST)
    assert req0.data["messages"][0]["role"] == "system" and req0.data["seed"] is not None
    turns = load_turns(evs)
    assert [t.origin for t in turns] == ["model"] * 5
    assert all(t.fingerprint_before for t in turns)
    te = turns[1].tool_executions[0]
    assert te.executed and te.executed_arguments == {"command": "cat data/*.log"}
    assert te.observation == te.raw_output  # short output: not truncated
    for f in ("events.jsonl", "episode.json", "messages.json", "summary.json", "grading.json"):
        assert (res.out_dir / f).exists(), f
    msgs = json.loads((res.out_dir / "messages.json").read_text())["messages"]
    assert msgs[-1]["role"] == "assistant" and "tool_calls" not in msgs[-1]


async def test_refuses_to_overwrite_existing_record(tmp_path, instance):
    backend = LocalFixtureBackend()
    await backend.run(instance, make_plan(instance, "ep-a"), tmp_path / "out")
    with pytest.raises(FileExistsError):
        await backend.run(instance, make_plan(instance, "ep-a"), tmp_path / "out")


async def test_malformed_call_is_visible_and_not_a_successful_action(tmp_path, instance):
    plan = make_plan(instance, "ep-mal", scripted_spec(FIXTURES / "malformed_policy.yaml"))
    res = await LocalFixtureBackend().run(instance, plan, tmp_path / "out")
    evs = read_events(res.out_dir / "events.jsonl")
    pe = [e for e in evs if e.kind == EventKind.PARSE_ERROR]
    rep = [e for e in evs if e.kind == EventKind.REPAIR]
    assert len(pe) == 1 and "could not parse arguments" in pe[0].data["error"]
    assert len(rep) == 1 and rep[0].data["field"] == "function.arguments" and rep[0].data["replacement"] == "{}"
    assert rep[0].data["original"].startswith('{"command"')
    turns = load_turns(evs)
    t0 = turns[0]
    assert t0.malformed and t0.repaired
    te = t0.tool_executions[0]
    assert te.executed is False and te.executed_arguments is None
    assert te.requested_arguments_raw == rep[0].data["original"]  # exactly what the model emitted
    assert t0.assistant_message["tool_calls"][0]["function"]["arguments"] == "{}"
    assert res.summary.n_malformed_turns == 1
    assert res.summary.success is False  # it then wrote a wrong answer


async def test_budget_stops(tmp_path, instance):
    b = LocalFixtureBackend()
    r1 = await b.run(instance, make_plan(instance, "e1", max_turns=2), tmp_path / "o1")
    assert r1.summary.stop_reason == "budget:max_turns" and r1.summary.stop_category == StopCategory.BUDGET
    assert r1.summary.n_requests == 2
    r2 = await b.run(instance, make_plan(instance, "e2", max_episode_tokens=50), tmp_path / "o2")
    assert r2.summary.stop_reason == "budget:max_episode_tokens" and r2.summary.n_requests == 1


async def test_output_truncated_and_endpoint_errors(tmp_path, instance):
    trunc = _script(tmp_path, "trunc", """
schema: scripted_policy/v1
rules:
  - name: cut
    respond:
      finish_reason: length
      tool_calls: [{name: write_file, raw_arguments: '{"path": "/app/answer.txt", "content": "12'}]
""")
    res = await LocalFixtureBackend().run(instance, make_plan(instance, "t", scripted_spec(trunc)), tmp_path / "o1")
    assert res.summary.stop_reason == "budget:output_truncated"
    obs = load_turns(res.out_dir / "events.jsonl")[0].tool_executions[0].observation
    assert "hit the max_tokens limit" in obs

    for kind, cat in (("request_error", StopCategory.MODEL_ERROR), ("infra_error", StopCategory.INFRA)):
        p = _script(tmp_path, kind, f"""
schema: scripted_policy/v1
rules:
  - name: fail
    respond: {{error: {{kind: {kind}, message: "400 context overflow"}}}}
""")
        res = await LocalFixtureBackend().run(instance, make_plan(instance, kind, scripted_spec(p)), tmp_path / kind)
        assert res.summary.stop_category == cat
        assert res.summary.n_requests == 1 and res.summary.extra["n_failed_requests"] == 1
        if cat == StopCategory.INFRA:
            assert res.summary.infra_error


async def test_agent_timeout_is_a_safety_stop(tmp_path, instance):
    slow = _script(tmp_path, "slow", "schema: scripted_policy/v1\ndelay_sec: 5\nrules:\n  - name: any\n    respond: {content: done}\n")
    res = await LocalFixtureBackend().run(instance, make_plan(instance, "slow", scripted_spec(slow), agent_timeout_sec=0.5), tmp_path / "o")
    assert res.summary.stop_reason == "safety:agent_timeout" and res.summary.stop_category == StopCategory.SAFETY
    assert (res.out_dir / "episode.json").exists()


async def test_unknown_tool_and_truncation(tmp_path, instance):
    p = _script(tmp_path, "unk", """
schema: scripted_policy/v1
rules:
  - name: done
    when: {min_turn: 2}
    respond: {content: done}
  - name: unknown
    when: {turn: 0}
    respond: {tool_calls: [{name: python, arguments: {code: "print(1)"}}]}
  - name: long
    when: {turn: 1}
    respond: {tool_calls: [{name: bash, arguments: {command: "for i in $(seq 1 400); do echo line-$i; done"}}]}
""")
    res = await LocalFixtureBackend().run(instance, make_plan(instance, "u", scripted_spec(p), max_output_chars=200), tmp_path / "o")
    turns = load_turns(res.out_dir / "events.jsonl")
    te0 = turns[0].tool_executions[0]
    assert te0.executed is False and te0.observation.startswith("[error] unknown tool 'python'")
    te1 = turns[1].tool_executions[0]
    assert te1.truncated and "characters omitted" in te1.observation
    assert te1.observation.startswith("[exit code 0]\nline-1") and te1.observation.endswith("line-400")
    assert len(te1.raw_output) > 2000 and "line-200" in te1.raw_output


async def test_live_policy_refused_on_local_backend(tmp_path, instance):
    spec = PolicySpec(kind="openai", api_base="http://127.0.0.1:9/v1", served_model_name="x")
    with pytest.raises(PermissionError):
        await LocalFixtureBackend().run(instance, make_plan(instance, "live", spec), tmp_path / "o")


def test_count_policy_fixture_exists():
    assert COUNT_POLICY.exists()


async def test_unparsed_tool_call_stops_as_model_error(tmp_path, instance):
    """A tool-call block the server could not parse is an explicit malformed turn, not a finish."""
    from evaluation.agents.tools import ToolConfig
    from learning_loop.core.interfaces import PolicyDecision
    from learning_loop.core.records import Usage
    from learning_loop.episodes.envs.local_session import LocalSession
    from learning_loop.episodes.episode import run_episode
    from learning_loop.episodes.events import EventLog

    class Unparsed:
        spec = scripted_spec()

        async def decide(self, messages, tools, seed):
            msg = {"role": "assistant", "content": '<tool_call>{"name": "bash", "arguments": {"command": "ls \\*"}}</tool_call>'}
            return PolicyDecision(raw_request={}, raw_response={}, history_message=msg, finish_reason="stop",
                                  usage=Usage(input_tokens=10, output_tokens=5), latency_sec=0.0,
                                  parse_errors=["unparsed_tool_call: invalid JSON"])

    plan = make_plan(instance, "ep-unparsed")
    log = EventLog(tmp_path / "events.jsonl", plan.episode_id)
    files = Path(instance.task_dir) / "environment" / "files"
    async with LocalSession(files, ToolConfig(workdir="/app", command_timeout_sec=10, max_output_chars=2000)) as session:
        run = await run_episode(plan, session, Unparsed(), log)
    assert run.core.stop_reason == "model_error:unparsed_tool_call"
    assert run.core.stop_category == StopCategory.MODEL_ERROR
    turns = load_turns(read_events(tmp_path / "events.jsonl"))
    assert turns[0].malformed
