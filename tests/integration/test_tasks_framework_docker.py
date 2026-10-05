"""Rendered tasks in real Docker (marker `docker`), on the FIXTURE families.

The shell form of every solution must score what generation predicted from its Python
model (oracle, each shortcut, doing nothing), with the environment probe passing in the
verifier. A verifier container that has network access refuses to grade.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from _task_helpers import fixture_family  # noqa: E402

from learning_loop.tasks.render import render  # noqa: E402

pytestmark = pytest.mark.docker


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _docker_ok():  # pragma: no cover
    pytest.skip("Docker daemon not available", allow_module_level=True)


async def _trial(task_dir: Path, agent: str, trials: Path):
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    cfg = TrialConfig(task=TaskConfig(path=task_dir), trials_dir=trials, agent=AgentConfig(name=agent))
    return await (await Trial.create(cfg)).run()


def _with_solution(task_dir: Path, script: Path, dest: Path) -> Path:
    shutil.copytree(task_dir, dest)
    shutil.copy(script, dest / "solution" / "solve.sh")
    return dest


def _verifier_stdout(result) -> str:
    return "\n".join(p.read_text() for p in Path(result.trial_uri.removeprefix("file://")).rglob("test-stdout.txt"))


@pytest.mark.parametrize("module,difficulty", [("sum_numbers", "hard"), ("fix_add", "easy")])
async def test_shell_solutions_score_what_their_models_predicted(tmp_path, module, difficulty):
    fam = fixture_family(module)
    task = tmp_path / "task"
    params = render(fam, difficulty, 1, task)
    runs = {"oracle": (task, "oracle", 1.0), "nop": (task, "nop", params["nop_reward"])}
    for name, predicted in params["shortcut_rewards"].items():
        runs[f"shortcut:{name}"] = (_with_solution(task, task / "solution" / "shortcuts" / f"{name}.sh", tmp_path / name), "oracle", predicted)
    for label, (task_dir, agent, expected) in runs.items():
        r = await _trial(task_dir, agent, tmp_path / "trials")
        assert r.exception_info is None, (label, r.exception_info)
        assert r.verifier_environment_mode.value == "separate"
        assert r.verifier_result.rewards == {"reward": pytest.approx(expected)}, label
        probe_line = next(ln for ln in _verifier_stdout(r).splitlines() if ln.startswith("probe: "))
        assert json.loads(probe_line[len("probe: "):])["ok"] is True, label


async def test_a_verifier_with_network_refuses_to_grade(tmp_path):
    task = tmp_path / "task"
    render(fixture_family("sum_numbers"), "easy", 1, task)
    (task / "tests" / "docker-compose.yaml").unlink()
    r = await _trial(task, "oracle", tmp_path / "trials")
    assert r.exception_info is not None and not (r.verifier_result and r.verifier_result.rewards)
    assert "environment probe failed; not grading: egress:dns, egress:tcp" in _verifier_stdout(r)
