"""The fix-code families fix-stats, fix-parser, fix-pagination, fix-dates and
fix-shell-script: their traps re-derived from the rendered files with reference code written
here, independently of each family's own models, and the properties every instance shares.

For every instance checked: the answer key agrees with an independent reference
implementation; the reference fix passes every hidden check and the visible tests; the
original files and every declared shortcut fail some hidden check, each shortcut a different
set than the original files (no shortcut is doing nothing in disguise); the visible tests
fail on the original files but pass once only the hidden bugs are left
(`visible-bugs-only`); every function has two visible tests, not both error checks, and only
tests of functions with a visible bug fail on the original files (so the tested functions do
not give away where the hidden bugs are); fix-shell-script leaves out the visible cases of
at least as many working behaviors as hidden-bug ones; and no hidden check reuses a visible test's inputs (so hard-coding them
cannot pass). Each family's instruction is the same at every difficulty, apart from the list
of the module's functions.
"""

from __future__ import annotations

import ast
import calendar
import copy
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import date, timedelta
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.generate import generate_spec, rng_for
from learning_loop.tasks.render import render
from learning_loop.tasks.runtime import grade as grade_runtime
from learning_loop.tasks.spec import GenContext, Reject

FAMILY_DIR = REPO_ROOT / "evaluation" / "families"
MODULES = {"fix-stats": "fix_stats", "fix-parser": "fix_parser", "fix-pagination": "fix_pagination", "fix-dates": "fix_dates", "fix-shell-script": "fix_shell_script"}
PYTHON_FAMILIES = ["fix-stats", "fix-parser", "fix-pagination", "fix-dates"]
# hidden checks whose input is the same in every instance, so a visible test may share it
FIXED_INPUTS = {"dumps_empty", "mean_empty_raises"}


def _load(module: str):
    path = FAMILY_DIR / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"family_{module}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAMILY


FAMILIES = {name: _load(module) for name, module in MODULES.items()}


# --------------------------------------------------------------------------- #
# Reading a rendered task
# --------------------------------------------------------------------------- #


def _heredocs(script: str) -> dict[str, str]:
    """{path: content} written by `cat > PATH <<'TAG' ... TAG` blocks."""
    return {m.group(1): m.group(3) for m in re.finditer(r"cat > (\S+) <<'(\w+)'\n(.*?\n)\2\n", script, re.DOTALL)}


def _task(d: Path):
    key = json.loads((d / "tests" / "key.json").read_text())
    params = json.loads(tomllib.loads((d / "task.toml").read_text())["metadata"]["params_json"])
    fixed = _heredocs((d / "solution" / "solve.sh").read_text())
    shortcuts = {p.stem: _heredocs(p.read_text()) for p in sorted((d / "solution" / "shortcuts").glob("*.sh"))}
    return key, params, fixed, shortcuts


def _reward(key, files: dict[str, str]) -> float:
    return _graded(key, files)[0]


def _graded(key, files: dict[str, str]) -> tuple[float, frozenset[str]]:
    """(reward, names of the failed checks)."""
    view = grade_runtime.MemoryView({path: (text.encode(), 0o755) for path, text in files.items()})
    reward, log = grade_runtime.grade(key, view, trusted=True)
    return reward, frozenset(ln.split()[1].rstrip(":") for ln in log if ln.startswith("FAIL "))


def _env_files(d: Path) -> dict[str, str]:
    root = d / "environment" / "files"
    return {"/app/" + str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def _visible_results(d: Path, tmp: Path, replace: dict[str, str]) -> dict[str, bool]:
    """Visible test (or case) name -> passed, running the task's visible tests with `replace`
    {/app path: content} in place of its files."""
    shutil.copytree(d / "environment" / "files", tmp)
    for path, text in replace.items():
        target = tmp / path[len("/app/") :]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    test = next(p.name for p in tmp.glob("test_*.py"))
    out = subprocess.run([sys.executable, "-B", test], cwd=tmp, capture_output=True, text=True, check=False).stdout
    return {ln.split()[1].rstrip(":"): ln.startswith("PASS ") for ln in out.splitlines() if ln.startswith(("PASS ", "FAIL "))}


def _visible_tests_pass(d: Path, tmp: Path, replace: dict[str, str]) -> bool:
    results = _visible_results(d, tmp, replace)
    return bool(results) and all(results.values())


def _check_traps(d: Path, tmp_path: Path):
    key, params, fixed, shortcuts = _task(d)
    original = _env_files(d)
    assert _reward(key, fixed) == 1.0
    nop, nop_failed = _graded(key, original)
    assert nop < 1.0
    for name, files in shortcuts.items():
        reward, failed = _graded(key, files)
        assert reward < 1.0, name
        # A shortcut other than a wrong fix (which may fail exactly where its bug did) must
        # fail differently from doing nothing. Hard-coding the visible tests' answers would
        # not (hidden inputs never repeat them), so it is not a declared shortcut.
        assert name.startswith("wrong-fix-") or failed != nop_failed, name
    assert "hardcode-visible" not in shortcuts
    assert not _visible_tests_pass(d, tmp_path / "original", {})
    assert _visible_tests_pass(d, tmp_path / "fixed", fixed)
    hidden_only = {f for f in params["bugs"] if f not in params["visible_bug_tests"]}
    assert ("visible-bugs-only" in shortcuts) == bool(hidden_only)
    if hidden_only:
        assert _visible_tests_pass(d, tmp_path / "partial", shortcuts["visible-bugs-only"])
    return key, params


# --------------------------------------------------------------------------- #
# Checks families: independent references for every hidden check
# --------------------------------------------------------------------------- #


def _visible_calls(test_source: str, funcs) -> set[str]:
    """JSON of (function, literal arguments) for every call to a module function in the visible tests."""
    out = set()
    for node in ast.walk(ast.parse(test_source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            try:
                args = [ast.literal_eval(a) for a in node.args]
            except ValueError:
                continue  # e.g. no_mutation tests pass a variable
            out.add(json.dumps([node.func.id, args], sort_keys=True))
    return out


def _check_against(key, refs, exceptions):
    for c in key["checks"]:
        ref = refs[c["func"]]
        args = copy.deepcopy(c["args"])
        if c["kind"] == "value":
            assert math.isclose(ref(*args), c["expected"], rel_tol=1e-9, abs_tol=1e-9), c["name"]
        elif c["kind"] == "equal":
            assert json.dumps(ref(*args), sort_keys=True) == json.dumps(c["expected"], sort_keys=True), c["name"]
        elif c["kind"] == "raises":
            with pytest.raises(exceptions.get(c.get("exception", "ValueError"), ValueError)):
                ref(*args)
        else:
            ref(*args)
            assert args == c["args"], c["name"]


def _check_visible_by_function(d: Path, tmp_path: Path, params, test_source: str):
    """Every function has visible tests, each calling only that function, and on the original
    files exactly the functions with a visible bug have a failing test."""
    funcs = set(params["functions"])
    calls = {
        node.name: {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs}
        for node in ast.parse(test_source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    assert all(len(called) == 1 for called in calls.values()), calls
    # the same number of tests for every function, never only error checks, so a function
    # whose tests had to avoid its hidden bug does not stand out
    per_function = {f: [t for t, called in calls.items() if f in called] for f in funcs}
    assert all(len(tests) == 2 for tests in per_function.values()), per_function
    bodies = {node.name: ast.unparse(node) for node in ast.parse(test_source).body if isinstance(node, ast.FunctionDef)}
    assert all(any("except" not in bodies[t] for t in tests) for tests in per_function.values())
    results = _visible_results(d, tmp_path / "by-function", {})
    assert set(results) == set(calls)
    assert {f for t, ok in results.items() if not ok for f in calls[t]} == set(params["visible_bug_tests"])


def _checks_family(name, difficulty, seed, tmp_path, refs, exceptions=None):
    d = tmp_path / "task"
    render(FAMILIES[name], difficulty, seed, d)
    key, params = _check_traps(d, tmp_path)
    _, _, _, shortcuts = _task(d)
    if name != "fix-stats":  # every catalogued bug has a plausible wrong fix
        assert {n for n in shortcuts if n.startswith("wrong-fix-")} == {f"wrong-fix-{f}" for f in params["bugs"]}
    assert len({c["name"] for c in key["checks"]}) == len(key["checks"])
    _check_against(key, refs, exceptions or {})
    test_source = next((d / "environment" / "files").glob("test_*.py")).read_text()
    _check_visible_by_function(d, tmp_path, params, test_source)
    visible = _visible_calls(test_source, refs)
    assert visible, "the visible tests call the module"
    hidden = {json.dumps([c["func"], c["args"]], sort_keys=True) for c in key["checks"] if c["name"] not in FIXED_INPUTS}
    assert not visible & hidden
    return key, params


# fix-stats ------------------------------------------------------------------ #


def _s_need(xs, n=1):
    if len(xs) < n:
        raise ValueError


def _s_mean(xs):
    _s_need(xs)
    return math.fsum(xs) / len(xs)


def _s_median(xs):
    _s_need(xs)
    s = sorted(xs)
    return (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2


def _s_variance(xs):
    _s_need(xs, 2)
    m = _s_mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _s_stdev(xs):
    return math.sqrt(_s_variance(xs))


def _s_percentile(xs, p):
    _s_need(xs)
    s = sorted(xs)
    pos = (len(s) - 1) * p / 100
    below = int(pos)
    return s[below] if below == len(s) - 1 else s[below] + (s[below + 1] - s[below]) * (pos - below)


def _s_mode(xs):
    _s_need(xs)
    return min(xs, key=lambda x: (-xs.count(x), x))


STATS = {"mean": _s_mean, "median": _s_median, "variance": _s_variance, "stdev": _s_stdev, "percentile": _s_percentile, "mode": _s_mode}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fix_stats_traps(tmp_path, difficulty, seed):
    key, params = _checks_family("fix-stats", difficulty, seed, tmp_path, STATS)
    assert ("visible-bugs-only" in _task(tmp_path / "task")[3]) == (difficulty != "easy")


# fix-pagination ------------------------------------------------------------- #


def _p_count(total, per_page):
    if total < 0 or per_page < 1:
        raise ValueError
    return math.ceil(total / per_page)


def _p_bounds(page, per_page, total):
    if not 1 <= page <= _p_count(total, per_page):
        raise ValueError
    return [(page - 1) * per_page, min(page * per_page, total)]


def _p_of(index, per_page):
    if index < 0 or per_page < 1:
        raise ValueError
    return math.floor(index / per_page) + 1


def _p_label(page, per_page, total):
    if total == 0:
        return "No items"
    first, stop = _p_bounds(page, per_page, total)
    return f"Items {first + 1}-{stop} of {total}"


def _p_window(current, last, width):
    if not 1 <= current <= last or width < 1:
        raise ValueError
    if last <= width:
        return list(range(1, last + 1))
    ideal_before = (width - 1) // 2  # one more page after current when the width is even
    starts = [s for s in range(1, last - width + 2) if s <= current <= s + width - 1]
    best = min(starts, key=lambda s: abs((current - s) - ideal_before))
    return list(range(best, best + width))


def _p_paginate(items, page, per_page):
    first, stop = _p_bounds(page, per_page, len(items))
    pages = _p_count(len(items), per_page)
    return {"items": list(items[first:stop]), "page": page, "pages": pages, "has_prev": page != 1, "has_next": page != pages}


PAGING = {"page_count": _p_count, "page_bounds": _p_bounds, "page_of": _p_of, "page_label": _p_label, "page_window": _p_window, "paginate": _p_paginate}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fix_pagination_traps(tmp_path, difficulty, seed):
    key, _ = _checks_family("fix-pagination", difficulty, seed, tmp_path, PAGING)
    names = {c["name"] for c in key["checks"]}
    # the boundaries every instance checks
    assert {"page_count_exact", "page_count_zero", "page_bounds_last_partial", "page_bounds_last_exact", "page_of_boundary", "page_label_no_items"} <= names
    exact = next(c for c in key["checks"] if c["name"] == "page_count_exact")
    assert exact["args"][0] % exact["args"][1] == 0


# fix-dates ------------------------------------------------------------------ #


def _d_days(year, month):
    return calendar.monthrange(year, month)[1]


def _d_end(day):
    d = date.fromisoformat(day)
    return date(d.year, d.month, _d_days(d.year, d.month)).isoformat()


def _d_add_months(day, months):
    d = date.fromisoformat(day)
    y, m = d.year, d.month
    for _ in range(abs(months)):
        y, m = (y + (m == 12), m % 12 + 1) if months > 0 else (y - (m == 1), (m - 2) % 12 + 1)
    return date(y, m, min(d.day, _d_days(y, m))).isoformat()


def _d_iso_week(day):
    d = date.fromisoformat(day)
    thursday = d + timedelta(days=3 - d.weekday())  # the ISO year is the year of the week's Thursday
    return f"{thursday.year}-W{(thursday.timetuple().tm_yday - 1) // 7 + 1:02d}"


def _d_business(day, n, holidays):
    if n < 0:
        raise ValueError
    d, off = date.fromisoformat(day), set(holidays)
    while n:
        d += timedelta(days=1)
        if d.isoweekday() < 6 and d.isoformat() not in off:
            n -= 1
    return d.isoformat()


def _d_age(born, on):
    b, d = date.fromisoformat(born), date.fromisoformat(on)
    if d < b:
        raise ValueError

    def birthday(year):
        return date(year, 3, 1) if (b.month, b.day) == (2, 29) and not calendar.isleap(year) else date(year, b.month, b.day)

    return sum(1 for y in range(b.year + 1, d.year + 1) if birthday(y) <= d)


def _d_weeks(year):
    jan1 = date(year, 1, 1)
    return sum(1 for i in range(366 if calendar.isleap(year) else 365) if (jan1 + timedelta(days=i)).weekday() == 3)  # one Thursday per ISO week


DATES = {
    "days_in_month": _d_days,
    "month_end": _d_end,
    "add_months": _d_add_months,
    "iso_week": _d_iso_week,
    "add_business_days": _d_business,
    "age": _d_age,
    "weeks_in_year": _d_weeks,
}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fix_dates_traps(tmp_path, difficulty, seed):
    key, _ = _checks_family("fix-dates", difficulty, seed, tmp_path, DATES)
    by_name = {c["name"]: c for c in key["checks"]}
    # the boundaries every instance checks: century and 400-year leap rules, ISO years that
    # differ from the calendar year at both ends, a December target month
    assert by_name["days_in_month_feb_century"]["args"][0] % 400 != 0 and by_name["days_in_month_feb_400"]["args"][0] % 400 == 0
    late, early = by_name["iso_week_late_december"], by_name["iso_week_early_january"]
    assert late["expected"][:4] == str(int(late["args"][0][:4]) + 1) and early["expected"][:4] == str(int(early["args"][0][:4]) - 1)
    assert by_name["add_months_into_december"]["expected"][5:7] == "12"
    source = (tmp_path / "task" / "environment" / "files" / "dates.py").read_text()
    assert ".today(" not in source and ".now(" not in source  # the task's code never reads the clock


# fix-parser ----------------------------------------------------------------- #


class _ParseError(ValueError):
    pass


def _ini_value(raw):
    v = raw.strip()
    m = re.fullmatch(r'"(.*)"', v, re.DOTALL)
    return m.group(1) if m else v


def _ini_parse(text):
    cfg, current = {}, None
    for line in (ln.strip() for ln in text.split("\n")):
        if line == "" or re.match(r"[#;]", line):
            continue
        if m := re.fullmatch(r"\[(.*)\]", line):
            name = m.group(1).strip()
            if name in cfg:
                raise _ParseError
            current = cfg[name] = {}
        elif m := re.fullmatch(r"([^=]*)=(.*)", line):
            key = m.group(1).strip().lower()
            if current is None or key in current:
                raise _ParseError
            current[key] = _ini_value(m.group(2))
        else:
            raise _ParseError
    return cfg


def _ini_bool(cfg, section, key):
    v = cfg[section][key].casefold()
    if v in {"1", "yes", "true", "on"} or v in {"0", "no", "false", "off"}:
        return v in {"1", "yes", "true", "on"}
    raise ValueError


def _ini_list(cfg, section, key):
    return [s for s in (part.strip() for part in re.split(",", cfg[section].get(key, ""))) if s]


def _ini_merge(base, override):
    out = copy.deepcopy(base)
    for name, keys in override.items():
        out[name] = {**out.get(name, {}), **keys}
    return out


def _ini_dumps(cfg):
    def fmt(v):
        return f'"{v}"' if v[:1].isspace() or v[-1:].isspace() else v

    return "\n".join(f"[{s}]\n" + "".join(f"{k} = {fmt(v)}\n" for k, v in keys.items()) for s, keys in cfg.items())


INI = {"parse_value": _ini_value, "parse": _ini_parse, "get_bool": _ini_bool, "get_list": _ini_list, "merge": _ini_merge, "dumps": _ini_dumps}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fix_parser_traps(tmp_path, difficulty, seed):
    key, _ = _checks_family("fix-parser", difficulty, seed, tmp_path, INI, {"ParseError": _ParseError, "KeyError": KeyError, "ValueError": ValueError})
    full = next(c for c in key["checks"] if c["name"] == "parse_full")["args"][0]
    stripped = [ln.strip() for ln in full.splitlines()]
    # every feature of the format is in the parsed text
    assert any(ln.startswith("#") for ln in stripped) and any(ln.startswith(";") for ln in stripped)
    assert any(ln != raw and ln[:1] in "#;" for ln, raw in zip(stripped, full.splitlines()))  # an indented comment
    assert any("=" in ln.split("=", 1)[1] for ln in stripped if "=" in ln and not ln.startswith(("#", ";")))
    assert any(" # " in ln or "#" in ln.split("=", 1)[-1] for ln in stripped if "=" in ln)
    assert any(ln.split("=", 1)[0].strip() != ln.split("=", 1)[0].strip().lower() for ln in stripped if "=" in ln and ln[0] not in "#;[")
    parsed = _ini_parse(full)
    assert any(name != name.lower() for name in parsed)  # a mixed-case section name
    first, second = list(parsed)[:2]
    shared = [k for k in parsed[first] if k in parsed[second]]
    assert shared  # the same key in two sections is not a duplicate
    if any(c["func"] == "dumps" for c in key["checks"]):
        assert "dumps_empty" in {c["name"] for c in key["checks"]}


# --------------------------------------------------------------------------- #
# fix-shell-script
# --------------------------------------------------------------------------- #


def _logsum(inputs, log_dir, pattern):
    """(OUT_FILE, stdout) of the specified script, computed here."""
    entries = []
    for rel, text in inputs.items():
        parent, _, name = rel.rpartition("/")
        if parent == log_dir and name.endswith(".log"):
            entries.append((name.encode(), name, sum(1 for line in text.split("\n") if pattern in line)))
    entries.sort()
    return "".join(f"{n}: {c}\n" for _, n, c in entries), f"total: {sum(c for *_, c in entries)}\n"


# bug -> the check kinds it breaks (each check uses the plain form of every other feature)
SHELL_EXPOSES = {
    "unset_default": {"default_pattern"},
    "error_not_fatal": {"missing_directory"},
    "unquoted_test": {"missing_directory"},
    "no_match_exits": {"file_without_matches"},
    "regex_pattern": {"fixed_string_pattern"},
    "unquoted_dir": {"spaces_in_directory"},
    "ls_words": {"spaces_in_file_names", "spaces_in_directory"},
    "no_empty_guard": {"no_log_files"},
}
SHELL_UNTOUCHED = {"skips_other_files", "case_sensitive_pattern", "lines_not_occurrences", "pattern_with_spaces"}  # no bug breaks these
SHELL_KINDS = {k for kinds in SHELL_EXPOSES.values() for k in kinds} | SHELL_UNTOUCHED


def _shell_missing(params, kinds):
    """(behaviors without a visible case, those a hidden bug breaks, those no bug breaks)."""
    hidden = {k for s, b in params["bugs"].items() if s not in params["visible_bug_tests"] for k in SHELL_EXPOSES[b]}
    untouched = SHELL_KINDS - {k for b in params["bugs"].values() for k in SHELL_EXPOSES[b]}
    missing = SHELL_KINDS - kinds
    return missing, hidden, missing & untouched


@pytest.mark.parametrize(("difficulty", "seed"), [("easy", 1), ("medium", 1), ("medium", 2), ("hard", 1), ("hard", 2)])  # each render runs bash ~100 times
def test_fix_shell_script_traps(tmp_path, difficulty, seed):
    d = tmp_path / "task"
    render(FAMILIES["fix-shell-script"], difficulty, seed, d)
    key, params = _check_traps(d, tmp_path)
    assert key["kind"] == "commands"
    assert ("/app/lib/count.sh" in key["files"]) == (difficulty == "hard")
    runs = [c for c in key["checks"] if c["argv"][:2] == ["bash", "logsum.sh"]]
    for c in runs:
        log_dir = c["argv"][2]
        pattern = c["argv"][4] if len(c["argv"]) > 4 else "ERROR"
        if any(rel.startswith(log_dir + "/") for rel in c["inputs"]):
            out, stdout = _logsum(c["inputs"], log_dir, pattern)
            assert (c["outputs"]["out.txt"], c["stdout"], c["exit"]) == (out, stdout, 0), c["name"]
        else:  # LOG_DIR does not exist
            assert (c["stdout"], c["exit"], "outputs" in c) == ("", 2, False) and " " in log_dir
    # one check per trap: the default PATTERN, a regex-looking PATTERN, a log without a match,
    # spaces in file and directory names, no .log file at all, the strict-mode line kept
    assert any(len(c["argv"]) == 4 for c in runs)
    regex = [c for c in runs if len(c["argv"]) > 4 and re.search(r"[.*\[]", c["argv"][4])]
    assert regex and any(
        re.search(c["argv"][4], line) and c["argv"][4] not in line for c in regex for text in c["inputs"].values() for line in text.splitlines()
    )
    assert any(c["outputs"]["out.txt"].count(": 0\n") for c in runs if "outputs" in c and c["outputs"]["out.txt"])
    assert any(" " in rel.rpartition("/")[2] and rel.endswith(".log") for c in runs for rel in c["inputs"])
    assert any(" " in c["argv"][2] and c["inputs"] for c in runs)
    assert any("outputs" in c and c["outputs"]["out.txt"] == "" for c in runs)
    assert any(c["argv"][:4] == ["grep", "-q", "-x", "set -euo pipefail"] for c in key["checks"])
    default = next(c for c in runs if len(c["argv"]) == 4)
    assert any("/archive/" in rel or rel.endswith((".txt", ".log.1")) for rel in default["inputs"])  # files the script must skip
    # in the regex-looking PATTERN check, every log's regex count differs from its fixed-string count
    for c in regex:
        for rel, text in c["inputs"].items():
            lines = text.splitlines()
            assert sum(bool(re.search(c["argv"][4], ln)) for ln in lines) != sum(c["argv"][4] in ln for ln in lines), (c["name"], rel)
    # no hidden run repeats a visible case's arguments
    test_source = (d / "environment" / "files" / "test_logsum.py").read_text()
    cases = json.loads(test_source.split('CASES = json.loads(r"""\n', 1)[1].split('\n""")', 1)[0])
    assert cases and not {json.dumps(case["args"]) for case in cases} & {json.dumps(c["argv"][2:]) for c in runs}
    # a visible case for every behavior a visible bug breaks; the behaviors without a case are
    # those hidden bugs break plus at least as many that work; on the original files only
    # cases of visible bugs fail, and every visible bug fails one
    shown = [b for s, b in params["bugs"].items() if s in params["visible_bug_tests"]]
    kinds = {case["name"].removeprefix("visible_") for case in cases}
    missing, hidden_kinds, missing_working = _shell_missing(params, kinds)
    assert missing == hidden_kinds | missing_working and len(missing_working) >= max(2, len(hidden_kinds))
    results = {n.removeprefix("visible_"): ok for n, ok in _visible_results(d, tmp_path / "by-case", {}).items()}
    assert set(results) == kinds
    assert {k for k, ok in results.items() if not ok} <= {k for b in shown for k in SHELL_EXPOSES[b]}
    assert all(any(not results.get(k, True) for k in SHELL_EXPOSES[b]) for b in shown)
    assert {n for n in _task(d)[3] if n.startswith("wrong-fix-")} == {f"wrong-fix-{s}" for s, b in params["bugs"].items() if b != "regex_pattern"}


# --------------------------------------------------------------------------- #
# Every family: one instruction, the bug mix, redraws
# --------------------------------------------------------------------------- #


def _template(name, spec) -> str:
    """The instruction with the list of the module's functions (instance data) blanked out."""
    if name == "fix-shell-script":
        return spec.instruction
    listed = ", ".join(f"`{f}`" for f in spec.params["functions"])
    assert spec.instruction.count(listed) == 1
    return spec.instruction.replace(listed, "{functions}")


@pytest.mark.parametrize("name", list(MODULES))
def test_instruction_is_the_same_at_every_difficulty(name):
    seeds = [1] if name == "fix-shell-script" else [1, 2]  # shell generation runs bash
    templates = {_template(name, generate_spec(FAMILIES[name], d, seed).spec) for d in FAMILIES[name].difficulties for seed in seeds}
    assert len(templates) == 1, templates
    # every statement holds at every level: no count of bugs or failing tests, and only
    # behavior every level has (fix-dates easy has no business days, so no weekends)
    text = templates.pop()
    assert "may not cover every bug" in text and "weekend" not in text and "One function" not in text


def _catalog(name):
    return FAMILIES[name].build.__globals__["BUGGY" if name == "fix-stats" else "BUGS"]


@pytest.mark.parametrize("name", ["fix-parser", "fix-pagination", "fix-dates"])
def test_every_python_bug_has_a_wrong_fix(name):
    # a draw without a catalogued wrong fix used to be redrawn, which skewed the draws towards
    # the bugs that had one
    assert all(entry["naive"] for bugs in _catalog(name).values() for entry in bugs.values())


def test_every_shell_bug_pair_has_a_wrong_fix():
    bugs = _catalog("fix-shell-script")
    with_fix = {(s, b) for s, kinds in bugs.items() for b, (_, fix) in kinds.items() if fix is not None}
    assert {b for _, b in set((s, b) for s, kinds in bugs.items() for b in kinds) - with_fix} == {"regex_pattern"}
    # so no draw is redrawn for lack of a wrong-fix shortcut


@pytest.mark.parametrize("name", PYTHON_FAMILIES)
def test_easy_draws_every_catalogued_bug(name):
    catalog, funcs = _catalog(name), FAMILIES[name].difficulties["easy"]["functions"]
    drawn = set()
    for seed in range(1, 121):
        drawn |= set(generate_spec(FAMILIES[name], "easy", seed).spec.params["bugs"].items())
    assert drawn == {(f, b) for f in funcs for b in catalog[f]}


# (bugs, hidden bugs) per level: the hidden bugs set the difficulty
BUG_MIX = {
    "fix-stats": {"easy": (1, 0), "medium": (2, 1), "hard": (4, 3)},
    "fix-parser": {"easy": (3, 2), "medium": (4, 3), "hard": (5, 4)},
    "fix-pagination": {"easy": (3, 0), "medium": (2, 1), "hard": (3, 2)},
    "fix-dates": {"easy": (2, 0), "medium": (2, 1), "hard": (3, 2)},
    "fix-shell-script": {"easy": (1, 0), "medium": (2, 1), "hard": (2, 1)},
}


@pytest.mark.parametrize(("name", "difficulty"), [(n, d) for n in MODULES for d in ("easy", "medium", "hard")])
def test_bugs_and_hidden_bugs_per_level(name, difficulty):
    for seed in [1] if name == "fix-shell-script" else [1, 2, 3]:  # shell generation runs bash
        params = generate_spec(FAMILIES[name], difficulty, seed).spec.params
        assert (len(params["bugs"]), len(params["bugs"]) - len(params["visible_bug_tests"])) == BUG_MIX[name][difficulty], seed


def test_fix_stats_hard_always_has_a_percentile_bug_that_is_often_missed():
    # the other catalogued bugs are found almost every time, so without it hard is no harder than medium
    drawn = {generate_spec(FAMILIES["fix-stats"], "hard", seed).spec.params["bugs"]["percentile"] for seed in range(1, 21)}
    assert drawn == {"rank_off_by_one", "unsorted"}


@pytest.mark.parametrize(("difficulty", "seed"), [("easy", 500123), ("easy", 500208), ("medium", 500118)])
def test_fix_shell_script_regex_pattern_seeds_generate(difficulty, seed):
    # these draws had as many decoy lines as matching lines, so the regex count equalled the
    # fixed-string count and `build` raised RuntimeError
    generate_spec(FAMILIES["fix-shell-script"], difficulty, seed)


def test_fix_pagination_reference_failing_a_hidden_check_is_a_redraw(monkeypatch):
    family = FAMILIES["fix-pagination"]
    namespace = family.build.__globals__
    cases = namespace["_cases"]

    def with_impossible_check(rng, func):
        return cases(rng, func) + [{"name": f"{func}_impossible", "func": func, "kind": "raises", "args": [1, 1]}]

    monkeypatch.setitem(namespace, "_cases", with_impossible_check)
    ctx = GenContext(family=family.name, difficulty="easy", seed=1, params=family.difficulties["easy"], rng=rng_for(family, "easy", 1))
    with pytest.raises(Reject, match="correct module fails hidden checks"):
        family.build(ctx)


def test_fix_parser_accepts_a_parse_error_for_an_invalid_boolean(tmp_path):
    render(FAMILIES["fix-parser"], "easy", 1, tmp_path)
    key, _, fixed, _ = _task(tmp_path)
    (path, source), = fixed.items()
    old = 'raise ValueError(f"not a boolean'
    assert source.count(old) == 1 and any(c["name"] == "get_bool_invalid" and c["exception"] == "ValueError" for c in key["checks"])
    # ParseError subclasses ValueError, so it is the ValueError the docstring specifies
    assert _graded(key, {path: source.replace(old, 'raise ParseError(f"not a boolean')}) == (1.0, frozenset())
    assert "get_bool_invalid" in _graded(key, {path: source.replace(old, 'raise KeyError(f"not a boolean')})[1]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_fix_shell_script_missing_cases_do_not_point_at_hidden_bugs(difficulty):
    family = FAMILIES["fix-shell-script"]
    for seed in range(1, 7):
        rng = rng_for(family, difficulty, seed)
        for _ in range(10):
            try:
                spec = family.build(GenContext(family=family.name, difficulty=difficulty, seed=seed, params=family.difficulties[difficulty], rng=rng))
                break
            except Reject:
                continue
        source = spec.files["test_logsum.py"]
        cases = json.loads(source.split('CASES = json.loads(r"""\n', 1)[1].split('\n""")', 1)[0])
        missing, hidden, missing_working = _shell_missing(spec.params, {c["name"].removeprefix("visible_") for c in cases})
        # behaviors that work go without a case at least as often as broken ones
        assert missing != hidden and len(missing_working) >= max(2, len(hidden)), (seed, missing, hidden)
