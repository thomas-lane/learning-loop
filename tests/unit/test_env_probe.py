"""The environment probe at episode start: a session that reports violations stops the
episode as infra before anything is executed or requested; sessions without a probe skip it."""

from __future__ import annotations

from pathlib import Path

from _env_helpers import count_errors_instance, make_plan

from learning_loop.core.records import EventKind, StopCategory
from learning_loop.episodes.episode import run_episode
from learning_loop.episodes.envs.local_session import LocalSession
from learning_loop.episodes.events import EventLog, read_events
from learning_loop.episodes.policy import make_policy

from evaluation.agents.tools import ToolConfig


class _ProbingSession(LocalSession):
    def __init__(self, *a, result, **kw):
        super().__init__(*a, **kw)
        self.result = result
        self.calls = 0

    async def probe(self, expect):
        self.calls += 1
        assert "app_digest" in expect and expect["network"] == "none"
        return self.result


async def _run(tmp_path, result):
    inst = count_errors_instance(tmp_path / "tasks")
    plan = make_plan(inst, "p")
    sess = _ProbingSession(Path(inst.task_dir) / "environment" / "files", ToolConfig("/app", 20, 8000), result=result)
    log = EventLog(tmp_path / "events.jsonl", "p")
    async with sess:
        run = await run_episode(plan, sess, make_policy(plan.policy), log)
    return run, sess, read_events(tmp_path / "events.jsonl")


async def test_a_failing_probe_stops_before_any_request(tmp_path):
    run, sess, events = await _run(tmp_path, {"ok": False, "violations": ["egress:dns", "app_content"], "observed": {}})
    assert run.core.stop_reason == "infra:env_probe:egress:dns,app_content" and run.core.stop_category == StopCategory.INFRA
    assert sess.calls == 1 and run.core.n_requests == 0
    kinds = [e.kind for e in events]
    assert EventKind.ENV_PROBE in kinds and EventKind.REQUEST not in kinds and EventKind.TOOL_CALL not in kinds


async def test_a_passing_probe_is_recorded_and_the_episode_runs(tmp_path):
    run, sess, events = await _run(tmp_path, {"ok": True, "violations": [], "observed": {}})
    assert run.core.stop_reason == "model_finished" and run.core.extra["env_probe"] == {"ok": True, "violations": []}


async def test_sessions_without_a_probe_record_the_skip(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    plan = make_plan(inst, "p")
    log = EventLog(tmp_path / "events.jsonl", "p")
    async with LocalSession(Path(inst.task_dir) / "environment" / "files", ToolConfig("/app", 20, 8000)) as sess:
        run = await run_episode(plan, sess, make_policy(plan.policy), log)
    assert run.core.stop_reason == "model_finished" and run.core.extra["env_probe"].startswith("skipped")
