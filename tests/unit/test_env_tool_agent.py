"""ToolAgent (Harbor agent class): CLI-compatible options and plan mode, without Docker."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from _env_helpers import count_errors_instance, make_plan

from evaluation.agents.tool_agent import ToolAgent
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Trajectory
from learning_loop.core.config import REPO_ROOT
from learning_loop.core.storage import atomic_write_json
from learning_loop.episodes.envs.local_session import LocalSession, _LocalExec
from learning_loop.episodes.episode import load_core
from learning_loop.episodes.events import load_turns
from evaluation.agents.tools import ToolConfig


def test_cli_kwargs_from_existing_job_config(tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "evaluation/configs/local-llama.yaml").read_text())
    agent_cfg = cfg["agents"][0]
    assert agent_cfg["name"] == "evaluation.agents.tool_agent:ToolAgent"
    a = ToolAgent(logs_dir=tmp_path, model_name=agent_cfg["model_name"], **agent_cfg["kwargs"])
    plan = a._cli_plan("Do the task.")
    assert plan.policy.api_base == "http://localhost:9931/v1" and plan.policy.send_seed is False
    assert plan.policy.sampling.max_output_tokens == 8192 and plan.budgets.max_turns == 30
    assert plan.system_prompt == (REPO_ROOT / "evaluation/agents/system_prompt.md").read_text().format(workdir="/app")
    assert [t["function"]["name"] for t in plan.tools] == ["bash", "read_file", "write_file"]
    assert plan.record_fingerprints is False and plan.state_spec.fingerprint_paths == []


class _FakeHarborEnv:
    """Exec/upload on a local temp dir mapped to /app (stands in for a container)."""

    def __init__(self, session: LocalSession):
        self._x = _LocalExec(session)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        return await self._x.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec)

    async def upload_file(self, source_path, target_path):
        return await self._x.upload_file(source_path, target_path)


async def test_plan_mode_writes_records_and_atif(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    plan = make_plan(inst, "ep-agent", env_probe=False)
    plan_path = tmp_path / "plan.json"
    atomic_write_json(plan_path, plan)
    trial = tmp_path / "trial"
    logs = trial / "agent"
    logs.mkdir(parents=True)
    rec = tmp_path / "host-only"
    agent = ToolAgent(logs_dir=logs, model_name="scripted-fixture", episode_plan_path=str(plan_path), record_dir=str(rec))
    sess = LocalSession(Path(inst.task_dir) / "environment/files", ToolConfig("/app", 20, 8000))
    await sess.start()
    try:
        ctx = AgentContext()
        await agent.run(plan.instruction, _FakeHarborEnv(sess), ctx)
        answer = (sess.real_workdir / "answer.txt").read_text().strip()
    finally:
        await sess.close()
    assert answer == str(inst.params["expected"])
    core = load_core(rec / "episode.json")
    assert core.stop_reason == "model_finished" and core.n_requests == 5
    assert ctx.metadata["stop_reason"] == "model_finished" and ctx.n_input_tokens == core.usage.input_tokens
    turns = load_turns(rec / "events.jsonl")
    assert len(turns) == 5 and all(t.fingerprint_before for t in turns)  # fingerprints via HarborSession path
    for f in ("trajectory.json", "messages.json", "events.jsonl", "episode.json"):
        assert (logs / f).exists(), f
    traj = Trajectory.model_validate_json((logs / "trajectory.json").read_text())
    assert [s.source for s in traj.steps] == ["system", "user"] + ["agent"] * 5
    assert traj.steps[2].tool_calls[0].arguments == {"command": "ls data"}
    msgs = json.loads((logs / "messages.json").read_text())
    assert msgs["messages"][:2] == [{"role": "system", "content": plan.system_prompt}, {"role": "user", "content": plan.instruction}]


async def test_plan_instruction_mismatch_refused(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    plan_path = tmp_path / "plan.json"
    atomic_write_json(plan_path, make_plan(inst, "ep", env_probe=False))
    agent = ToolAgent(logs_dir=tmp_path, model_name="x", episode_plan_path=str(plan_path), record_dir=str(tmp_path / "r"))
    try:
        await agent.run("a different instruction", object(), AgentContext())
    except ValueError as e:
        assert "instruction" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected refusal")
    shutil.rmtree(tmp_path / "r", ignore_errors=True)
