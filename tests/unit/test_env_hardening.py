"""Episode/environment hardening: malformed mixed turns, environment INFRA stops,
intervention outcome records, usage-less budgets, bash timeout clamping, stub cores,
image identity, request timeout/concurrency, and symlink-safe agent log copies."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
from _env_helpers import count_errors_instance, make_plan, scripted_spec, tool_call_message
from openai import AsyncOpenAI

from evaluation.agents.tool_agent import ToolAgent, copy_into_untrusted_dir
from evaluation.agents.tools import ToolConfig, effective_timeout
from harbor.models.agent.context import AgentContext
from learning_loop.core.interfaces import PolicyDecision
from learning_loop.core.records import EventKind, PolicySpec, StopCategory, Usage
from learning_loop.core.storage import atomic_write_json
from learning_loop.episodes.backends import LocalFixtureBackend, _stub_core
from learning_loop.episodes.envs.harbor_session import HarborSession
from learning_loop.episodes.envs.local_session import LocalSession, _LocalExec
from learning_loop.episodes.episode import build_replay_spec, run_episode
from learning_loop.episodes.events import EventLog, load_turns, read_events
from learning_loop.episodes.policy import OpenAIChatPolicy
from learning_loop.tasks.instances import NETWORK_CAVEAT, load_state_spec


@pytest.fixture()
def instance(tmp_path):
    return count_errors_instance(tmp_path / "tasks")


def _files(instance) -> Path:
    return Path(instance.task_dir) / "environment" / "files"


def _cfg(timeout: int = 20) -> ToolConfig:
    return ToolConfig(workdir="/app", command_timeout_sec=timeout, max_output_chars=4000)


class _ListPolicy:
    """Returns the given (history message, usage, parse_errors) decisions in order, then 'done'."""

    spec = scripted_spec()

    def __init__(self, steps):
        self.steps = list(steps)

    async def decide(self, messages, tools, seed):
        if self.steps:
            msg, usage, perr = self.steps.pop(0)
        else:
            msg, usage, perr = {"role": "assistant", "content": "done"}, Usage(input_tokens=1, output_tokens=1), []
        return PolicyDecision(raw_request={}, raw_response={}, history_message=msg, finish_reason="stop", usage=usage, latency_sec=0.0, parse_errors=perr)

    async def aclose(self):
        return None


class _FakeHarborEnv:
    """A Harbor-like environment over a local temp dir; `fail` makes exec/upload raise like Harbor."""

    def __init__(self, session: LocalSession, fail: str | None = None, has_timeout: bool = True):
        self._x = _LocalExec(session)
        self.fail = fail
        self.has_timeout = has_timeout

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if command.startswith("command -v timeout"):
            return await self._x.exec("true" if self.has_timeout else "false")
        if self.fail == "exec" and "cpu.stat" not in command:
            raise RuntimeError("Docker compose command failed for environment x. Return code: 1")
        if self.fail == "backstop" and command.startswith("timeout "):
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        if self.fail == "harbor_timeout" and timeout_sec:
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        if command.startswith("timeout "):  # the in-container wrapper: run the inner command
            command = command.split(" bash -c ", 1)[1]
            import shlex

            command = shlex.split(command)[0]
        return await self._x.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec)

    async def upload_file(self, source_path, target_path):
        if self.fail == "upload":
            raise RuntimeError("docker cp failed")
        return await self._x.upload_file(source_path, target_path)


# --------------------------------------------------------------------------- #
# 3. mixed valid + unparseable tool calls
# --------------------------------------------------------------------------- #


async def test_turn_with_valid_and_unparsed_call_is_malformed(tmp_path, instance):
    valid = tool_call_message("c0", "bash", '{"command": "echo ok"}')
    valid["content"] = '<tool_call>{"name": "bash", "arguments": {"command": "ls \\*"}}</tool_call>'
    pol = _ListPolicy([(valid, Usage(input_tokens=5, output_tokens=5), ["unparsed_tool_call: invalid JSON in block 1"])])
    log = EventLog(tmp_path / "events.jsonl", "mixed")
    async with LocalSession(_files(instance), _cfg()) as s:
        run = await run_episode(make_plan(instance, "mixed"), s, pol, log)
    evs = read_events(tmp_path / "events.jsonl")
    pe = [e for e in evs if e.kind == EventKind.PARSE_ERROR]
    assert len(pe) == 1 and pe[0].data["tool_call_id"] is None and pe[0].data["error"].startswith("unparsed_tool_call:")
    turns = load_turns(evs)
    assert turns[0].malformed and turns[0].tool_executions[0].executed  # the valid call still ran
    assert run.core.n_malformed_turns == 1 and run.core.stop_reason == "model_finished"


# --------------------------------------------------------------------------- #
# 4. environment transport failures are INFRA
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fail", ["exec", "backstop", "upload"])
async def test_environment_transport_failure_is_infra(tmp_path, instance, fail):
    call = ("write_file", '{"path": "/app/a.txt", "content": "1"}') if fail == "upload" else ("bash", '{"command": "echo hi"}')
    pol = _ListPolicy([(tool_call_message("c0", *call), Usage(input_tokens=5, output_tokens=5), [])])
    log = EventLog(tmp_path / "events.jsonl", "infra")
    async with LocalSession(_files(instance), _cfg()) as local:
        sess = HarborSession(_FakeHarborEnv(local, fail=fail), _cfg(), measure_cpu=False)
        plan = make_plan(instance, "infra")
        plan.record_fingerprints = False
        run = await run_episode(plan, sess, pol, log)
    assert run.core.stop_category == StopCategory.INFRA, run.core.stop_reason
    assert run.core.stop_reason.startswith("infra:EnvInfraError") and run.core.infra_error
    evs = read_events(tmp_path / "events.jsonl")
    assert any(e.kind == EventKind.INFRA_ERROR and e.data["where"] == "environment" for e in evs)
    assert not any(e.kind == EventKind.TOOL_RESULT for e in evs)  # never shown to the learner


async def test_command_errors_stay_observations(tmp_path, instance):
    async with LocalSession(_files(instance), _cfg()) as local:
        sess = HarborSession(_FakeHarborEnv(local), _cfg(), measure_cpu=False)
        te = await sess.execute("c", "bash", {"command": "exit 3"}, None)
        assert te.executed and te.exit_code == 3 and te.observation.startswith("[exit code 3]")
        te = await sess.execute("c", "write_file", {"path": "/app/data", "content": "x"}, None)  # a directory
        assert te.executed is False and te.error and te.observation.startswith("[error]")
        te = await sess.execute("c", "read_file", {"path": "/app/nope"}, None)
        assert te.error and te.observation.startswith("[error]")
        # without an in-container `timeout`, Harbor's exec timeout is the command's own timeout
        sess2 = HarborSession(_FakeHarborEnv(local, fail="harbor_timeout", has_timeout=False), _cfg(3), measure_cpu=False)
        te = await sess2.execute("c", "bash", {"command": "echo hi"}, None)
        assert te.timed_out is True and te.executed and "timed out after 3s" in te.observation


async def test_environment_failure_during_replay_is_infra_not_replay(tmp_path, instance):
    src = await LocalFixtureBackend().run(instance, make_plan(instance, "src"), tmp_path / "src")
    ev = src.out_dir / "events.jsonl"
    spec = build_replay_spec(ev, 2, "original", load_turns(ev)[2].assistant_message)
    spec.expected_image_identity = None
    log = EventLog(tmp_path / "br.jsonl", "br")
    async with LocalSession(_files(instance), _cfg()) as local:
        sess = HarborSession(_FakeHarborEnv(local, fail="exec"), _cfg(), restore=spec_restore(instance), measure_cpu=False)
        run = await run_episode(make_plan(instance, "br", replay=spec), sess, _ListPolicy([]), log)
    assert run.core.stop_category == StopCategory.INFRA and run.core.replay_ok is None


def spec_restore(instance):
    return load_state_spec(Path(instance.task_dir)).restore


# --------------------------------------------------------------------------- #
# 5. intervention outcome record
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "args,expected",
    [
        ('{"command": "echo fine"}', {"executed": True, "error": None, "exit_code": 0, "timed_out": False}),
        ('{"command": "sleep 5", "timeout_sec": 1}', {"executed": True, "error": None, "exit_code": 124, "timed_out": True}),
        ('{"cmd": "oops"}', {"executed": False, "exit_code": None, "timed_out": False}),
    ],
)
async def test_intervention_tool_outcome_recorded(tmp_path, instance, args, expected):
    src = await LocalFixtureBackend().run(instance, make_plan(instance, "src"), tmp_path / "src")
    ev = src.out_dir / "events.jsonl"
    spec = build_replay_spec(ev, 1, "edited", tool_call_message("call_1_0", "bash", args))
    res = await LocalFixtureBackend().run(instance, make_plan(instance, "br", replay=spec), tmp_path / "br")
    rec = res.summary.extra["intervention_tool"]
    assert set(rec) == {"executed", "error", "exit_code", "timed_out"}
    for k, v in expected.items():
        assert rec[k] == v, (k, rec)
    if not expected["executed"]:
        assert "missing required argument" in rec["error"]


# --------------------------------------------------------------------------- #
# 6. token budget without usage
# --------------------------------------------------------------------------- #


async def test_missing_usage_stops_when_token_budget_is_set(tmp_path, instance):
    step = (tool_call_message("c0", "bash", '{"command": "echo hi"}'), Usage(source="none"), [])
    async with LocalSession(_files(instance), _cfg()) as s:
        run = await run_episode(make_plan(instance, "nu", max_episode_tokens=10_000), s, _ListPolicy([step]), EventLog(tmp_path / "a.jsonl", "nu"))
        assert run.core.stop_reason == "budget:usage_unavailable" and run.core.stop_category == StopCategory.BUDGET
        assert run.core.n_requests == 1
        # without a token budget, missing usage does not stop the episode
        run2 = await run_episode(make_plan(instance, "nu2"), s, _ListPolicy([step]), EventLog(tmp_path / "b.jsonl", "nu2"))
        assert run2.core.stop_reason == "model_finished"


# --------------------------------------------------------------------------- #
# 7. bash timeout clamp
# --------------------------------------------------------------------------- #


async def test_bash_timeout_is_clamped_and_recorded(instance):
    cfg = _cfg(5)
    assert effective_timeout({}, cfg) == 5 and effective_timeout({"timeout_sec": 3600}, cfg) == 5
    assert effective_timeout({"timeout_sec": 2}, cfg) == 2 and effective_timeout({"timeout_sec": 0}, cfg) == 5
    async with LocalSession(_files(instance), cfg) as s:
        te = await s.execute("c", "bash", {"command": "echo hi", "timeout_sec": 3600}, None)
        assert te.timeout_sec == 5 and te.timed_out is False
        te = await s.execute("c", "bash", {"command": "sleep 3", "timeout_sec": 1}, None)
        assert te.timeout_sec == 1 and te.timed_out is True and te.exit_code == 124 and "timed out after 1s" in te.observation
        te = await s.execute("c", "bash", {"command": "echo hi", "timeout_sec": "soon"}, None)
        assert te.executed is False and "timeout_sec must be an integer" in te.observation
        te = await s.execute("c", "read_file", {"path": "/app/data/app.log"}, None)
        assert te.timeout_sec is None and te.timed_out is None


# --------------------------------------------------------------------------- #
# 8. stub core counts
# --------------------------------------------------------------------------- #


async def test_stub_core_counts_from_events_or_unavailable(tmp_path, instance):
    from _env_helpers import FIXTURES

    res = await LocalFixtureBackend().run(instance, make_plan(instance, "mal", scripted_spec(FIXTURES / "malformed_policy.yaml")), tmp_path / "o")
    plan = make_plan(instance, "mal")
    core = _stub_core(plan, res.out_dir, "safety:agent_timeout", StopCategory.SAFETY, "x")
    assert core.n_turns == res.summary.extra["n_turns"] and core.n_turns >= 2
    assert core.n_malformed_turns == 1 and core.extra["counts_source"] == "events.jsonl"
    empty = tmp_path / "empty"
    empty.mkdir()
    core2 = _stub_core(plan, empty, "infra:x", StopCategory.INFRA, "x")
    assert core2.n_turns is None and "n_turns" in core2.extra["unavailable_counts"] and core2.usage.total is None


# --------------------------------------------------------------------------- #
# 9/10. network caveat, image identity
# --------------------------------------------------------------------------- #


def test_network_caveat_recorded_for_public_network_replay(instance):
    spec = load_state_spec(Path(instance.task_dir))
    assert NETWORK_CAVEAT in spec.caveats
    toml = Path(instance.task_dir) / "task.toml"
    toml.write_text(toml.read_text().replace("memory_mb = 1024", 'memory_mb = 1024\nnetwork_mode = "no-network"', 1))
    assert load_state_spec(Path(instance.task_dir)).caveats == []


async def test_image_identity_recorded_and_carried_into_replay_spec(tmp_path, instance):
    src = await LocalFixtureBackend().run(instance, make_plan(instance, "src"), tmp_path / "src")
    ident = src.summary.extra["image_identity"]
    assert ident["kind"] == "local_fixture" and ident["identity"].startswith("sha256:")
    start = read_events(src.out_dir / "events.jsonl")[0]
    assert start.kind == EventKind.EPISODE_START and start.data["image_identity"] == ident
    ev = src.out_dir / "events.jsonl"
    spec = build_replay_spec(ev, 1, "original", load_turns(ev)[1].assistant_message)
    assert spec.expected_image_identity == ident
    res = await LocalFixtureBackend().run(instance, make_plan(instance, "br", replay=spec), tmp_path / "br")
    assert res.replay_ok and res.summary.extra["image_identity_check"] == "ok"


def test_task_dockerfiles_pin_base_images_by_digest(tmp_path):
    from evaluation.generators import GENERATORS, generate
    from learning_loop.core.config import REPO_ROOT
    from learning_loop.episodes.envs.base import build_inputs_identity

    dirs = [REPO_ROOT / "evaluation/tasks/fix-stats", REPO_ROOT / "evaluation/tasks/log-triage"]
    for fam in GENERATORS:
        generate(fam, "easy", 1, tmp_path / fam)
        dirs.append(tmp_path / fam)
    for d in dirs:
        for sub in ("environment", "tests"):
            ident = build_inputs_identity("harbor_docker", d / sub)
            assert ident["base_pinned"] is True and all("@sha256:" in b for b in ident["base_images"]), (d, sub, ident)


# --------------------------------------------------------------------------- #
# policy: request timeout and per-endpoint concurrency
# --------------------------------------------------------------------------- #


def _completion():
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def test_request_timeout_from_spec():
    spec = PolicySpec(kind="openai", api_base="http://t/v1", served_model_name="m", request_timeout_sec=12.5)
    pol = OpenAIChatPolicy(spec)
    assert pol.timeout_sec == 12.5 and pol.client.timeout == 12.5
    assert OpenAIChatPolicy(spec.model_copy(update={"request_timeout_sec": None})).timeout_sec == 600.0
    with pytest.raises(ValueError):
        OpenAIChatPolicy(spec.model_copy(update={"max_concurrent_requests": 0}))


async def test_max_concurrent_requests_per_endpoint():
    state = {"now": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.05)
        state["now"] -= 1
        return httpx.Response(200, json=_completion())

    def make(limit):
        client = AsyncOpenAI(base_url="http://sem/v1", api_key="k", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        spec = PolicySpec(kind="openai", api_base="http://sem/v1", served_model_name="m", max_concurrent_requests=limit)
        return OpenAIChatPolicy(spec, client=client)

    msgs = [{"role": "user", "content": "u"}]
    pols = [make(2) for _ in range(6)]  # separate policy objects share the endpoint's semaphore
    await asyncio.gather(*(p.decide(msgs, [], None) for p in pols))
    assert state["max"] == 2
    state["max"] = 0
    pols = [make(None) for _ in range(6)]
    await asyncio.gather(*(p.decide(msgs, [], None) for p in pols))
    assert state["max"] == 6


# --------------------------------------------------------------------------- #
# 2. agent log dir: never written through planted symlinks
# --------------------------------------------------------------------------- #


def test_copy_into_untrusted_dir_replaces_symlinks(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("host secret")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "messages.json").symlink_to(victim)
    (logs / "trajectory.json").mkdir()
    src = tmp_path / "src.json"
    src.write_text("{}")
    assert copy_into_untrusted_dir(src, logs, "messages.json") is None
    assert victim.read_text() == "host secret"
    assert not (logs / "messages.json").is_symlink() and (logs / "messages.json").read_text() == "{}"
    assert copy_into_untrusted_dir(src, logs, "trajectory.json") is not None  # planted dir: refused
    link_dir = tmp_path / "linkdir"
    link_dir.symlink_to(tmp_path / "logs")
    assert copy_into_untrusted_dir(src, link_dir, "x.json") is not None  # the dir itself must not be a link
    assert copy_into_untrusted_dir(src, logs, "../x.json") is not None


class _AgentEnv:
    def __init__(self, session: LocalSession, logs: Path, victim: Path):
        self._x = _LocalExec(session)
        self.logs, self.victim = logs, victim
        self.planted = False

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self.planted and not command.startswith("command -v"):
            # the "container" plants symlinks in the mounted agent log dir during the episode
            for name in ("messages.json", "trajectory.json", "events.jsonl", "episode.json"):
                p = self.logs / name
                if p.exists() or p.is_symlink():
                    p.unlink()
                p.symlink_to(self.victim)
            self.planted = True
        return await self._x.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec)

    async def upload_file(self, source_path, target_path):
        return await self._x.upload_file(source_path, target_path)


async def test_tool_agent_never_writes_through_planted_symlinks(tmp_path, instance):
    plan = make_plan(instance, "ep-agent")
    plan_path = tmp_path / "plan.json"
    atomic_write_json(plan_path, plan)
    logs = tmp_path / "trial" / "agent"
    logs.mkdir(parents=True)
    victim = tmp_path / "host_file.txt"
    victim.write_text("do not touch")
    rec = tmp_path / "host-only"
    agent = ToolAgent(logs_dir=logs, model_name="scripted-fixture", episode_plan_path=str(plan_path), record_dir=str(rec))
    async with LocalSession(_files(instance), _cfg()) as sess:
        env = _AgentEnv(sess, logs, victim)
        await agent.run(plan.instruction, env, AgentContext())
    assert env.planted and victim.read_text() == "do not touch"
    for name in ("messages.json", "trajectory.json", "events.jsonl", "episode.json"):
        assert not (logs / name).is_symlink() and (logs / name).is_file(), name
        assert (logs / name).read_bytes() == (rec / name).read_bytes()
    assert json.loads((rec / "episode.json").read_text())["stop_reason"] == "model_finished"
    assert not [p for p in os.listdir(logs) if p.endswith(".tmp")]
