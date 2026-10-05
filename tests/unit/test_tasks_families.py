"""The evaluation families: deterministic, valid Harbor tasks, clean of stray randomness,
and their traps re-derived here from the rendered files, independently of each family's
own solution models."""

from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest
from harbor.models.task.config import TaskConfig

from evaluation.families import FAMILIES
from learning_loop.core.config import REPO_ROOT
from learning_loop.core.records import RestoreCapability
from learning_loop.core.storage import sha256_tree
from learning_loop.tasks.family_lint import nondeterminism
from learning_loop.tasks.instances import load_state_spec
from learning_loop.tasks.render import render

CASES = [(name, d) for name, fam in FAMILIES.items() for d in fam.difficulties]


def _expected(d: Path) -> str:
    return json.loads((d / "tests" / "key.json").read_text())["expected"]


def _files(d: Path, sub: str) -> dict[str, str]:
    root = d / "environment" / "files" / sub
    return {
        str(p.relative_to(root)): (gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()).decode()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("family,difficulty", CASES)
def test_deterministic_valid_and_network_isolated(tmp_path, family, difficulty):
    fam = FAMILIES[family]
    a = render(fam, difficulty, 5, tmp_path / "a")
    assert render(fam, difficulty, 5, tmp_path / "b") == a and sha256_tree(tmp_path / "a") == sha256_tree(tmp_path / "b")
    render(fam, difficulty, 6, tmp_path / "c")
    assert sha256_tree(tmp_path / "c") != sha256_tree(tmp_path / "a")
    d = tmp_path / "a"
    cfg = TaskConfig.model_validate_toml((d / "task.toml").read_text())
    assert cfg.verifier.environment_mode.value == "separate" and cfg.metadata["generator"] == fam.generator_id
    spec = load_state_spec(d)
    assert spec.restore == RestoreCapability.DETERMINISTIC_REPLAY and spec.caveats == []
    assert ("local_fixture" in tomllib.loads((d / "task.toml").read_text())["metadata"]) == (family == "count-errors")


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "evaluation" / "families").glob("*.py")), ids=lambda p: p.name)
def test_family_modules_use_only_the_generation_rng(path):
    assert nondeterminism(path.read_text(), str(path)) == []


def _top(ips: list[str]) -> str | None:
    best = Counter(ips).most_common(2)
    return best[0][0] if best and (len(best) == 1 or best[0][1] != best[1][1]) else None


@pytest.mark.parametrize("difficulty", list(FAMILIES["log-triage"].difficulties))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_log_triage_traps(tmp_path, difficulty, seed):
    render(FAMILIES["log-triage"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path, "logs"), _expected(tmp_path)

    def five(names):
        return _top([ln.split()[0] for n in names for ln in files[n].splitlines() if 500 <= int(ln.split()[8]) < 600])

    assert five(list(files)) == expected
    for k in range(1, len(files)):
        for combo in itertools.combinations(files, k):
            assert five(combo) != expected, combo
    assert _top([ln.split()[0] for t in files.values() for ln in t.splitlines() if int(ln.split()[8]) >= 400]) != expected
    if FAMILIES["log-triage"].difficulties[difficulty]["size_trap"]:
        any_field = [ln.split()[0] for t in files.values() for ln in t.splitlines() if any(500 <= int(f) < 600 for f in ln.split()[8:10])]
        assert _top(any_field) != expected


@pytest.mark.parametrize("difficulty", list(FAMILIES["csv-revenue"].difficulties))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_csv_revenue_traps(tmp_path, difficulty, seed):
    render(FAMILIES["csv-revenue"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path, "data"), _expected(tmp_path)
    rows = {n: list(csv.DictReader(io.StringIO(t))) for n, t in files.items()}

    def top(names, value, completed=True):
        t: Counter = Counter()
        for n in names:
            for r in rows[n]:
                if not completed or r["status"] == "completed":
                    t[r["region"]] += value(r)
        best = t.most_common(2)
        return best[0][0] if best and (len(best) == 1 or best[0][1] != best[1][1]) else None

    revenue = lambda r: int(r["quantity"]) * round(float(r["unit_price"]) * 100)  # noqa: E731 - cents, exact
    assert top(files, revenue) == expected
    assert top(files, revenue, completed=False) != expected
    assert top(files, lambda r: int(r["quantity"])) != expected
    assert top(files, lambda r: 1) != expected
    for k in range(1, len(files)):
        for combo in itertools.combinations(files, k):
            assert top(combo, revenue) != expected, combo
    p = FAMILIES["csv-revenue"].difficulties[difficulty]
    if p["quoted_commas"]:
        assert any('"' in t for t in files.values())
    if p["reorder"]:
        assert len({next(csv.reader(io.StringIO(t)))[1] for t in files.values()}) > 1


@pytest.mark.parametrize("difficulty", list(FAMILIES["count-errors"].difficulties))
def test_count_errors_traps(tmp_path, difficulty):
    render(FAMILIES["count-errors"], difficulty, 1, tmp_path)
    files, expected = _files(tmp_path, "data"), int(_expected(tmp_path))

    def count(take_file, take_line):
        return sum(1 for n, t in files.items() if take_file(n) for ln in t.splitlines() if take_line(ln))

    assert count(lambda n: n.endswith(".log"), lambda ln: "ERROR" in ln) == expected
    assert count(lambda n: n.endswith(".log"), lambda ln: "error" in ln.lower()) > expected
    assert count(lambda n: True, lambda ln: "ERROR" in ln) > expected
    if difficulty == "hard":
        assert count(lambda n: n.endswith(".log") and "/" not in n, lambda ln: "ERROR" in ln) < expected


@pytest.mark.parametrize("difficulty", list(FAMILIES["fix-stats"].difficulties))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fix_stats_visible_tests_and_partial_nop(tmp_path, difficulty, seed):
    params = render(FAMILIES["fix-stats"], difficulty, seed, tmp_path / "t")
    assert 0 < params["nop_reward"] < 1
    if difficulty != "easy":
        assert params["shortcut_rewards"]["visible-bugs-only"] < 1
    fixed = (tmp_path / "t" / "solution" / "solve.sh").read_text().split("<<'PY'\n")[1].split("PY\n")[0]
    env = tmp_path / "t" / "environment" / "files"
    for lib, want in ((fixed, 0), ((env / "stats.py").read_text(), 1)):
        d = tmp_path / f"run{want}"
        d.mkdir()
        (d / "stats.py").write_text(lib)
        (d / "test_stats.py").write_text((env / "test_stats.py").read_text())
        assert subprocess.run([sys.executable, "-B", "test_stats.py"], cwd=d, capture_output=True).returncode == want
