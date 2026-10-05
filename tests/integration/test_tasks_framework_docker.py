"""Rendered tasks in real Docker (marker `docker`), on the FIXTURE families.

The shell form of every solution must score what generation predicted from its Python
model (oracle, each shortcut, doing nothing), with the environment probe passing in the
verifier. A verifier container that has network access refuses to grade.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env_helpers import make_plan  # noqa: E402
from _task_docker import check_predictions, trial, verifier_stdout  # noqa: E402
from _task_helpers import fixture_family  # noqa: E402

from learning_loop.core.records import StopCategory, TaskInstance  # noqa: E402
from learning_loop.core.storage import sha256_tree  # noqa: E402
from learning_loop.episodes.backends import HarborDockerBackend  # noqa: E402

from learning_loop.tasks.render import render  # noqa: E402

pytestmark = pytest.mark.docker


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _docker_ok():  # pragma: no cover
    pytest.skip("Docker daemon not available", allow_module_level=True)


@pytest.mark.parametrize("module,difficulty", [("sum_numbers", "hard"), ("fix_add", "easy")])
async def test_shell_solutions_score_what_their_models_predicted(tmp_path, module, difficulty):
    assert await check_predictions(fixture_family(module), difficulty, 1, tmp_path) == []


async def test_a_verifier_with_network_refuses_to_grade(tmp_path):
    task = tmp_path / "sum-numbers__easy__s1__networked-verifier"  # unique: Harbor names the image after the directory
    render(fixture_family("sum_numbers"), "easy", 1, task)
    (task / "tests" / "docker-compose.yaml").unlink()
    r = await trial(task, "oracle", tmp_path / "trials")
    assert r.exception_info is not None and not (r.verifier_result and r.verifier_result.rewards)
    assert "environment probe failed; not grading: egress:dns, egress:tcp" in verifier_stdout(r)


async def test_a_verifier_built_from_other_files_refuses_to_grade(tmp_path):
    task = tmp_path / "sum-numbers__easy__s1__stale-key"  # unique: Harbor names the image after the directory
    render(fixture_family("sum_numbers"), "easy", 1, task)
    key = task / "tests" / "key.json"
    key.write_text(key.read_text().replace('"expected": "', '"expected": "9'))  # as if a stale build mixed in another key
    r = await trial(task, "oracle", tmp_path / "trials")
    assert r.exception_info is not None and not (r.verifier_result and r.verifier_result.rewards)
    assert "verifier files differ from the rendered task (stale image build); not grading" in verifier_stdout(r)


async def test_an_agent_image_with_other_files_stops_before_turn_0(tmp_path):
    task = tmp_path / "sum-numbers__easy__s1__stale-data"  # unique: Harbor names the image after the directory
    render(fixture_family("sum_numbers"), "easy", 1, task)
    (task / "environment" / "files" / "data" / "part0.txt").write_text("99\n")  # as if a stale build kept another task's file
    inst = TaskInstance(instance_id="sum-numbers/easy/s1", family="sum-numbers", task_dir=str(task), content_hash=sha256_tree(task))
    res = await HarborDockerBackend().run(inst, make_plan(inst, "stale-data"), tmp_path / "ep")
    s = res.summary
    assert s.stop_reason == "infra:env_probe:app_content" and s.stop_category == StopCategory.INFRA
    assert s.n_requests == 0
