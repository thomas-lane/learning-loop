"""Every evaluation family in real Docker (marker `docker`): the per-family check.

Generation checks each instance against Python models of its solutions; this test checks
those models against reality. For every difficulty (seed 1), the oracle, doing nothing and
every declared shortcut must score exactly the reward generation predicted, with the
environment probe passing in the verifier. Then an episode that writes and runs the oracle
is branched at the oracle's run: the branch re-executes the earlier turn in a fresh
container, and every fingerprint and the oracle's own output must match the source.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env_helpers import make_plan, scripted_spec  # noqa: E402
from _task_docker import check_predictions  # noqa: E402

from evaluation.families import FAMILIES  # noqa: E402
from learning_loop.core.records import TaskInstance  # noqa: E402
from learning_loop.core.storage import sha256_tree  # noqa: E402
from learning_loop.episodes.backends import HarborDockerBackend  # noqa: E402
from learning_loop.episodes.episode import build_replay_spec  # noqa: E402
from learning_loop.episodes.events import load_turns  # noqa: E402
from learning_loop.tasks.render import render  # noqa: E402

pytestmark = pytest.mark.docker
CONCURRENCY = 4


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _docker_ok():  # pragma: no cover
    pytest.skip("Docker daemon not available", allow_module_level=True)


@pytest.mark.parametrize("family", sorted(FAMILIES))
async def test_shell_solutions_score_what_generation_predicted(tmp_path, family):
    fam = FAMILIES[family]
    gate = asyncio.Semaphore(CONCURRENCY)
    problems = await asyncio.gather(*(check_predictions(fam, d, 1, tmp_path / d, gate) for d in fam.difficulties))
    assert [p for ps in problems for p in ps] == []


def _oracle_policy(path: Path, shell: str) -> Path:
    """FIXTURE scripted policy: write the oracle to /tmp, run it, finish."""
    rules = [
        {"name": "write-oracle", "when": {"turn": 0}, "respond": {"tool_calls": [{"name": "write_file", "arguments": {"path": "/tmp/oracle.sh", "content": shell}}]}},
        {"name": "run-oracle", "when": {"turn": 1}, "respond": {"tool_calls": [{"name": "bash", "arguments": {"command": "bash /tmp/oracle.sh"}}]}},
        {"name": "done", "when": {"min_turn": 2}, "respond": {"content": "done"}},
    ]
    path.write_text(yaml.safe_dump({"schema": "scripted_policy/v1", "label": "fixture", "description": "runs the task's oracle", "rules": rules}))
    return path


@pytest.mark.parametrize("family", sorted(FAMILIES))
async def test_oracle_episode_replays_identically(tmp_path, family):
    task = tmp_path / f"{family}__medium__s1__oracle-replay"  # unique: Harbor names the image after the directory
    render(FAMILIES[family], "medium", 1, task)
    inst = TaskInstance(instance_id=f"{family}/medium/s1", family=family, task_dir=str(task), content_hash=sha256_tree(task))
    policy = scripted_spec(_oracle_policy(tmp_path / "oracle.yaml", (task / "solution" / "solve.sh").read_text()))
    backend = HarborDockerBackend()
    src = await backend.run(inst, make_plan(inst, f"orc-{family}", policy, agent_timeout_sec=120.0), tmp_path / "src")
    assert src.summary.success is True and src.summary.stop_reason == "model_finished", src.summary.stop_reason
    assert src.summary.extra["env_probe"] == {"ok": True, "violations": []}  # the agent container matched its profile
    ev = src.out_dir / "events.jsonl"
    turns = load_turns(ev)
    assert len(turns) == 3
    # branch at turn 1 (running the oracle): turn 0 is re-executed in a fresh container and checked
    br = await backend.run(inst, make_plan(inst, f"orc-br-{family}", policy, replay=build_replay_spec(ev, 1, "original", turns[1].assistant_message), agent_timeout_sec=120.0), tmp_path / "br")
    assert br.replay_ok is True and br.replay_mismatches == [], br.replay_mismatches
    assert br.summary.success is True and br.summary.extra["image_identity_check"] == "ok"
    branch_turns = load_turns(br.out_dir / "events.jsonl")
    assert [t.fingerprint_before for t in branch_turns] == [t.fingerprint_before for t in turns]
    assert branch_turns[1].tool_executions[0].observation == turns[1].tool_executions[0].observation
