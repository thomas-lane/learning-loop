"""Harbor + Docker checks (marker `docker`; run with `uv run pytest -m docker tests/integration`).

- oracle/nop baselines for the original tasks (separate verifier mode)
- oracle on one generated hard instance per trained/held-out family
- scripted episode through HarborDockerBackend + ToolAgent; replay of the original
  and an edited branch; timing independence with inserted delays
- malicious outputs (planted reward files, peeking at /tests) get no reward; a forged
  fix-stats artifact (atexit/daemon reward overwrite, __eq__-always-true values) does not
  score 1.0 under the isolated grader
- image identity (digest-pinned build inputs) is recorded and checked on replay
Containers are removed after every trial (Harbor `delete=True`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from _env_helpers import FIXTURES, count_errors_instance, make_plan, scripted_spec, tool_call_message  # noqa: E402

from learning_loop.backends import HarborDockerBackend  # noqa: E402
from learning_loop.config import REPO_ROOT  # noqa: E402
from learning_loop.episode import build_replay_spec  # noqa: E402
from learning_loop.events import load_turns, read_events  # noqa: E402
from learning_loop.records import EventKind, StopCategory  # noqa: E402
from learning_loop.tasks import load_splits, materialize  # noqa: E402

pytestmark = pytest.mark.docker


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _docker_ok():  # pragma: no cover
    pytest.skip("Docker daemon not available", allow_module_level=True)


def _leftover_containers(prefix: str) -> list[str]:
    out = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
    return [n for n in out if prefix.lower() in n.lower()]


async def _baseline(task_dir: Path, agent: str, trials: Path) -> dict | None:
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    cfg = TrialConfig(task=TaskConfig(path=task_dir), trials_dir=trials, agent=AgentConfig(name=agent))
    r = await (await Trial.create(cfg)).run()
    assert r.exception_info is None, r.exception_info
    assert r.verifier_environment_mode.value == "separate"
    return r.verifier_result.rewards if r.verifier_result else None


@pytest.mark.parametrize(
    "task,agent,expected",
    [("log-triage", "oracle", 1.0), ("log-triage", "nop", 0.0), ("fix-stats", "oracle", 1.0), ("fix-stats", "nop", 3 / 9)],
)
async def test_original_task_baselines(tmp_path, task, agent, expected):
    rewards = await _baseline(REPO_ROOT / "evaluation" / "tasks" / task, agent, tmp_path)
    assert rewards is not None and abs(rewards["reward"] - expected) < 1e-6


@pytest.mark.parametrize("iid", ["log-triage/hard/s1", "fix-stats/hard/s1", "csv-revenue/hard/s301"])
async def test_generated_hard_instances_oracle(tmp_path, iid):
    s = load_splits(REPO_ROOT / "evaluation" / "splits" / "pilot.yaml")
    inst = materialize(s, tmp_path / "tasks", ids=[iid])[iid]
    rewards = await _baseline(Path(inst.task_dir), "oracle", tmp_path / "trials")
    assert rewards == {"reward": 1.0}
    if iid.startswith("fix-stats"):
        nop = await _baseline(Path(inst.task_dir), "nop", tmp_path / "trials")
        assert abs(nop["reward"] - inst.params["nop_reward"]) < 1e-6


async def test_scripted_episode_replay_branch_and_timing(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    backend = HarborDockerBackend()
    src = await backend.run(inst, make_plan(inst, "dk-src", seed=3), tmp_path / "src")
    s = src.summary
    assert s.stop_reason == "model_finished" and s.success is True and s.reward == {"reward": 1.0}
    assert s.extra["verifier_environment_mode"] == "separate"
    ident = s.extra["image_identity"]
    assert ident["kind"] == "harbor_docker" and ident["base_pinned"] is True and ident["identity"].startswith("sha256:")
    ev = src.out_dir / "events.jsonl"
    turns = load_turns(ev)
    assert len(turns) == 5 and all(t.fingerprint_before for t in turns)
    assert turns[1].tool_executions[0].cpu_time_sec is not None  # container cgroup delta
    # host-only record is outside the trial's container-mounted dirs
    assert not str(src.out_dir).startswith(str(Path(s.trial_dir)))
    assert (Path(s.trial_dir) / "agent" / "trajectory.json").exists()

    orig = await backend.run(inst, make_plan(inst, "dk-orig", replay=build_replay_spec(ev, 1, "original", turns[1].assistant_message), seed=5), tmp_path / "orig")
    edit_msg = tool_call_message("call_1_0", "bash", '{"command": "grep -h ERROR data/*.log | wc -l"}')
    edit = await backend.run(inst, make_plan(inst, "dk-edit", replay=build_replay_spec(ev, 1, "edited", edit_msg), seed=5), tmp_path / "edit")
    for r in (orig, edit):
        assert r.replay_ok is True and r.summary.success is True, (r.summary.stop_reason, r.replay_mismatches)
        assert r.summary.extra["image_identity_check"] == "ok"
        assert r.summary.extra["intervention_tool"] == {"executed": True, "error": None, "exit_code": 0, "timed_out": False}
    assert edit.summary.n_requests < orig.summary.n_requests
    bt = load_turns(orig.out_dir / "events.jsonl")
    assert bt[1].fingerprint_before == turns[1].fingerprint_before
    assert bt[0].tool_executions[0].observation == turns[0].tool_executions[0].observation

    # timing independence: the same fixed actions with inserted inference delays
    slow_script = tmp_path / "slow.yaml"
    slow_script.write_text((FIXTURES / "count_errors_policy.yaml").read_text().replace("schema: scripted_policy/v1", "schema: scripted_policy/v1\ndelay_sec: 1.5"))
    slow = await backend.run(inst, make_plan(inst, "dk-slow", scripted_spec(slow_script), seed=3), tmp_path / "slow")

    def trace(p):
        evs = read_events(p)
        return (
            [e.data["observation"] for e in evs if e.kind == EventKind.TOOL_RESULT],
            [e.data["fingerprint"] for e in evs if e.kind == EventKind.FINGERPRINT],
        )

    assert slow.summary.timing.endpoint_sec >= 1.5 * slow.summary.n_requests - 0.1
    assert trace(slow.out_dir / "events.jsonl") == trace(ev)
    assert slow.summary.reward == s.reward
    assert _leftover_containers("dk-") == []


async def test_malicious_outputs_get_no_reward_on_docker(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    res = await HarborDockerBackend().run(inst, make_plan(inst, "dk-evil", scripted_spec(FIXTURES / "malicious_policy.yaml")), tmp_path / "evil")
    s = res.summary
    assert s.stop_category == StopCategory.MODEL
    assert s.reward == {"reward": 0.0} and s.success is False
    turns = load_turns(res.out_dir / "events.jsonl")
    assert "planted" in turns[0].tool_executions[0].observation
    peek = turns[1].tool_executions[0].observation
    assert "No such file or directory" in peek and "EXPECTED" not in peek
    assert _leftover_containers("dk-evil") == []


async def test_planted_reward_file_is_ignored_on_log_triage(tmp_path):
    from learning_loop.records import TaskInstance
    from learning_loop.storage import sha256_tree

    d = REPO_ROOT / "evaluation" / "tasks" / "log-triage"
    inst = TaskInstance(instance_id="log-triage/static", family="log-triage", task_dir=str(d), content_hash=sha256_tree(d))
    res = await HarborDockerBackend().run(inst, make_plan(inst, "dk-plant", scripted_spec(FIXTURES / "plant_reward_policy.yaml")), tmp_path / "o")
    assert res.summary.extra["verifier_environment_mode"] == "separate"
    assert res.summary.reward == {"reward": 0.0}


async def test_forged_fix_stats_artifact_does_not_score(tmp_path):
    from learning_loop.records import TaskInstance
    from learning_loop.storage import sha256_tree

    d = REPO_ROOT / "evaluation" / "tasks" / "fix-stats"
    inst = TaskInstance(instance_id="fix-stats/static", family="fix-stats", task_dir=str(d), content_hash=sha256_tree(d))
    res = await HarborDockerBackend().run(inst, make_plan(inst, "dk-forge", scripted_spec(FIXTURES / "fix_stats_forge_policy.yaml")), tmp_path / "o")
    s = res.summary
    assert s.extra["verifier_environment_mode"] == "separate"
    assert s.reward is not None and s.reward["reward"] < 1.0 and s.success is False, s.reward
    # the forged extra observation lines (atexit hook, daemon) make the output ambiguous: every check fails
    out = (Path(s.trial_dir) / "verifier" / "test-stdout.txt").read_text()
    assert s.reward == {"reward": 0.0} and "expected exactly one observation line" in out, out
    assert _leftover_containers("dk-forge") == []
