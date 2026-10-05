"""Load the FIXTURE task families in tests/fixtures/tasks (framework tests only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from learning_loop.tasks.spec import Family

TASK_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tasks"


def fixture_family(module: str) -> Family:
    spec = importlib.util.spec_from_file_location(f"task_fixture_{module}", TASK_FIXTURES / f"{module}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAMILY
