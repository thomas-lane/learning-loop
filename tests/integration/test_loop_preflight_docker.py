"""The run preflight against real Docker (marker `docker`): the oracle of a rendered task
passes; a task whose verifier files differ from its rendered digest fails the preflight."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.families import FAMILIES
from learning_loop.core.records import TaskInstance
from learning_loop.core.storage import sha256_tree
from learning_loop.orchestration.preflight import PreflightError, run_preflight
from learning_loop.tasks.render import render

pytestmark = pytest.mark.docker


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _docker_ok():  # pragma: no cover
    pytest.skip("Docker daemon not available", allow_module_level=True)


def _ctx(tmp_path: Path, name: str) -> SimpleNamespace:
    task = tmp_path / name  # unique: Harbor names the image after the directory
    params = render(FAMILIES["count-errors"], "easy", 3, task)
    inst = TaskInstance(instance_id="count-errors/easy/s3", family="count-errors", task_dir=str(task), content_hash=sha256_tree(task), params=params)
    return SimpleNamespace(run_dir=tmp_path / "run", machine=SimpleNamespace(environment_backend="harbor_docker"), instances={inst.instance_id: inst}, backend_concurrency=lambda: 2)


async def test_preflight_passes_on_a_rendered_task(tmp_path):
    rec = await run_preflight(_ctx(tmp_path, "count-errors__easy__s3__preflight"), lambda m: None)
    assert rec["status"] == "passed" and rec["results"][0]["reward"] == 1.0


async def test_preflight_fails_when_the_verifier_files_differ(tmp_path):
    ctx = _ctx(tmp_path, "count-errors__easy__s3__preflight-stale")
    key = Path(ctx.instances["count-errors/easy/s3"].task_dir) / "tests" / "key.json"
    key.write_text(key.read_text().replace('"expected": "', '"expected": "9'))
    with pytest.raises(PreflightError, match="count-errors"):
        await run_preflight(ctx, lambda m: None)
