"""Generators: determinism, Harbor validity, separate-verifier layout, shortcut resistance."""

from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from evaluation.generators import GENERATORS, generate
from evaluation.generators import csv_revenue, fix_stats, log_triage
from harbor.models.task.config import TaskConfig
from learning_loop.config import REPO_ROOT
from learning_loop.records import RestoreCapability
from learning_loop.storage import sha256_tree
from learning_loop.tasks import load_state_spec

CASES = [(fam, d) for fam, m in GENERATORS.items() for d in m.DIFFICULTIES]


@pytest.mark.parametrize("family,difficulty", CASES)
def test_deterministic_and_harbor_valid(tmp_path, family, difficulty):
    a = generate(family, difficulty, 5, tmp_path / "a")
    b = generate(family, difficulty, 5, tmp_path / "b")
    assert a == b and sha256_tree(tmp_path / "a") == sha256_tree(tmp_path / "b")
    c = generate(family, difficulty, 6, tmp_path / "c")
    assert sha256_tree(tmp_path / "c") != sha256_tree(tmp_path / "a")
    d = tmp_path / "a"
    cfg = TaskConfig.model_validate_toml((d / "task.toml").read_text())
    assert cfg.verifier.environment_mode.value == "separate" and cfg.artifacts
    assert (d / "tests" / "Dockerfile").exists() and (d / "environment" / "Dockerfile").exists()
    assert (d / "solution" / "solve.sh").stat().st_mode & 0o111
    spec = load_state_spec(d)
    assert spec.restore == RestoreCapability.DETERMINISTIC_REPLAY and spec.fingerprint_paths == ["/app"]
    assert cfg.metadata["family"] == family and cfg.metadata["generator"] == f"{family}@v{GENERATORS[family].VERSION}"
    # hidden grading material is not in the learner-visible build context
    env_files = {p.name for p in (d / "environment").rglob("*")}
    assert not env_files & {"test.sh", "test_hidden.py", "grade.py", "solve.sh"}
    assert c is not None


@pytest.mark.parametrize("difficulty", list(log_triage.DIFFICULTIES))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_log_triage_every_shortcut_fails(tmp_path, difficulty, seed):
    generate("log-triage", difficulty, seed, tmp_path)
    logs = tmp_path / "environment" / "logs"
    files = {}
    for p in sorted(x for x in logs.rglob("*") if x.is_file()):
        raw = p.read_bytes()
        files[str(p.relative_to(logs))] = gzip.decompress(raw).decode() if p.suffix == ".gz" else raw.decode()
    expected = (tmp_path / "tests" / "test.sh").read_text().split('EXPECTED="')[1].split('"')[0]

    def top(rows):
        c = Counter(rows)
        (ip, n), *rest = c.most_common()
        return ip if not rest or rest[0][1] < n else "<tie>"

    def five(names):
        return top([ln.split()[0] for n in names for ln in files[n].splitlines() if 500 <= int(ln.split()[8]) < 600])

    assert five(list(files)) == expected
    for k in range(1, len(files)):
        for combo in itertools.combinations(files, k):
            assert five(combo) != expected, combo
    ge400 = top([ln.split()[0] for t in files.values() for ln in t.splitlines() if int(ln.split()[8]) >= 400])
    assert ge400 != expected
    if log_triage.DIFFICULTIES[difficulty]["size_trap"]:
        grep5 = top([ln.split()[0] for t in files.values() for ln in t.splitlines() if any(500 <= int(f) < 600 for f in ln.split()[8:10])])
        assert grep5 != expected


def test_static_log_triage_data_unchanged():
    chk = log_triage.check({p.name: (gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()) for p in (REPO_ROOT / "evaluation/tasks/log-triage/environment/logs").iterdir()})
    assert chk["truth"] == "10.0.0.99" and all(v != "10.0.0.99" for v in chk["subsets"].values())


@pytest.mark.parametrize("difficulty", list(fix_stats.DIFFICULTIES))
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_fix_stats_oracle_and_nop(tmp_path, difficulty, seed):
    params = generate("fix-stats", difficulty, seed, tmp_path)
    assert 0 < params["nop_reward"] < 1
    hidden = (tmp_path / "tests" / "test_hidden.py").read_text()
    run = fix_stats._exec(fix_stats.RUNNER)["run_checks"]
    checks = json.loads(eval(hidden.split("CHECKS = json.loads(")[1].split(")\n")[0]))  # noqa: S307 - our generated literal
    fixed = (tmp_path / "solution" / "solve.sh").read_text().split("<<'PY'\n")[1].split("PY\n")[0]
    buggy = (tmp_path / "environment" / "stats.py").read_text()
    assert all(run(fix_stats._exec(fixed), checks).values())
    res = run(fix_stats._exec(buggy), checks)
    assert abs(sum(res.values()) / len(res) - params["nop_reward"]) < 1e-6
    # the visible tests pass on the reference and fail on the buggy library
    for lib, want in ((fixed, 0), (buggy, 1)):
        d = tmp_path / f"run{want}"
        d.mkdir()
        (d / "stats.py").write_text(lib)
        (d / "test_stats.py").write_text((tmp_path / "environment" / "test_stats.py").read_text())
        rc = subprocess.run([sys.executable, "-B", "test_stats.py"], cwd=d, capture_output=True).returncode
        assert rc == want


def _static_fix_stats_checks() -> list[dict]:
    hidden = (REPO_ROOT / "evaluation/tasks/fix-stats/tests/test_hidden.py").read_text()
    return json.loads(eval(hidden.split("CHECKS = json.loads(")[1].split(")\n")[0]))  # noqa: S307 - our literal


def test_static_fix_stats_nop_baseline_is_partial():
    checks = _static_fix_stats_checks()
    run = fix_stats._exec(fix_stats.RUNNER)["run_checks"]
    buggy = fix_stats._exec((REPO_ROOT / "evaluation/tasks/fix-stats/environment/stats.py").read_text())
    res = run(buggy, checks)
    assert len(checks) == 9 and sum(res.values()) == 3  # recorded nop baseline 3/9
    assert {k for k, v in res.items() if v} == {"mean", "median_odd", "percentile_0"}


def test_static_fix_stats_grader_is_the_rendered_template():
    """The hand-written task's grader is exactly the generator's isolated grader (no drift)."""
    path = REPO_ROOT / "evaluation/tasks/fix-stats/tests/test_hidden.py"
    origin = path.read_text().split("reward.json. ", 1)[1].split("\n", 1)[0]
    assert path.read_text() == fix_stats.hidden_test_source(_static_fix_stats_checks(), origin=origin)


def test_fix_stats_judge_accepts_only_strict_observations():
    judge = fix_stats._exec(fix_stats.JUDGE)["judge"]
    checks = [
        {"name": "v", "func": "mean", "kind": "value", "args": [[1, 2]], "expected": 1.5},
        {"name": "r", "func": "mean", "kind": "raises", "args": [[]]},
        {"name": "m", "func": "median", "kind": "no_mutation", "args": [[3, 1, 2]]},
    ]
    good = {"v": {"value": 1.5}, "r": {"raised": "ValueError", "value_error": True}, "m": {"after": [3, 1, 2]}}
    assert judge(checks, good) == {"v": True, "r": True, "m": True}
    forged = [
        {"v": {"value": True}, "r": {"raised": "X", "value_error": 1}, "m": {"after": [1, 2, 3]}},
        {"v": {"value": "1.5"}, "r": {"raised": "X", "value_error": "true"}, "m": {"after": [3, 1, 2], "x": 1}},
        {"v": {"value": 1.5, "ok": True}, "r": {"returned": True}, "m": None},
        {"v": True, "r": True, "m": True},
        {"v": {"value": float("nan")}},
        [],
        "all passed",
    ]
    for obs in forged:
        assert not any(judge(checks, obs).values()), obs


def test_fix_stats_observe_serializes_values_not_objects():
    """An __eq__-always-true return value is reported as its plain float value (or nothing)."""
    observe = fix_stats._exec(fix_stats.OBSERVE)["observe"]

    class AlwaysEqual(float):
        def __eq__(self, other):
            return True

        __hash__ = float.__hash__

        def __sub__(self, other):
            return 0.0

    class NotANumber:
        def __eq__(self, other):
            return True

    lib = {"mean": lambda xs: AlwaysEqual(0), "median": lambda xs: NotANumber()}
    checks = [{"name": "a", "func": "mean", "kind": "value", "args": [[1, 2]]}, {"name": "b", "func": "median", "kind": "value", "args": [[1]]}]
    obs = json.loads(json.dumps(observe(lib, checks)))
    assert obs == {"a": {"value": 0.0}, "b": {"value": None}}
    judge = fix_stats._exec(fix_stats.JUDGE)["judge"]
    assert judge([{**checks[0], "expected": 1.5}, {**checks[1], "expected": 1}], obs) == {"a": False, "b": False}


def _safe(f) -> bool:
    try:
        return bool(f())
    except Exception:
        return False


@pytest.mark.parametrize("difficulty", list(csv_revenue.DIFFICULTIES))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_csv_revenue_traps(tmp_path, difficulty, seed):
    generate("csv-revenue", difficulty, seed, tmp_path)
    files = {p.name: p.read_text() for p in sorted((tmp_path / "environment" / "data").iterdir())}
    chk = csv_revenue.check(files)
    expected = (tmp_path / "tests" / "test.sh").read_text().split('EXPECTED="')[1].split('"')[0]
    assert chk["truth"] == expected
    for key in ("no_status_filter", "quantity_sum", "row_count"):
        assert chk[key] != expected, key
    assert all(v != expected for v in chk["subsets"].values())
    spec = csv_revenue.DIFFICULTIES[difficulty]
    if spec["quoted_commas"]:
        assert chk["naive_split"] != expected and any('"' in t for t in files.values())
    if spec["reorder"]:
        headers = {next(csv.reader(io.StringIO(t)))[1] for t in files.values()}
        assert len(headers) > 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_count_errors_expected(tmp_path, difficulty):
    params = generate("count-errors", difficulty, 3, tmp_path)
    root = tmp_path / "environment" / "files" / "data"
    n = sum(ln.count("ERROR") > 0 for p in root.rglob("*.log") for ln in p.read_text().splitlines())
    assert n == params["expected"]
    lower = sum("error" in ln.lower() for p in root.rglob("*.log") for ln in p.read_text().splitlines())
    assert lower > n  # grep -i overcounts
    assert "ERROR" in (root / "notes.txt").read_text()  # non-.log decoy


def test_generate_refuses_non_empty_dir(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_text("")
    with pytest.raises(FileExistsError):
        generate("count-errors", "easy", 1, tmp_path / "x")
    assert Path(tmp_path / "x" / "f").exists()
