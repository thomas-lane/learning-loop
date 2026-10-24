"""Families draw randomness only from GenContext.rng."""

from __future__ import annotations

import pytest
from _task_helpers import TASK_FIXTURES

from learning_loop.tasks.family_lint import nondeterminism


@pytest.mark.parametrize(
    "source,hit",
    [
        ("import random\n", "import random"),
        ("import time as t\n", "import time"),
        ("from secrets import token_hex\n", "from secrets"),
        ("import uuid\n", "import uuid"),
        ("import os\nos.urandom(4)\n", ".urandom()"),
        ("from datetime import datetime\ndatetime.now()\n", ".now()"),
        ("import datetime\ndatetime.date.today()\n", ".today()"),
    ],
)
def test_nondeterministic_sources_are_flagged(source, hit):
    assert any(hit in v for v in nondeterminism(source)), nondeterminism(source)


def test_clean_source_passes():
    assert nondeterminism("from datetime import datetime, timedelta\nx = datetime(2026, 1, 1) + timedelta(days=1)\n") == []


@pytest.mark.parametrize("path", sorted(TASK_FIXTURES.glob("*.py")), ids=lambda p: p.name)
def test_fixture_families_are_clean(path):
    assert nondeterminism(path.read_text(), str(path)) == []
