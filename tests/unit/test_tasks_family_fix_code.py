"""The fix-code families fix-parser, fix-pagination, fix-dates and fix-shell-script:
deterministic rendering, clean randomness, and their traps re-derived from the rendered
files with reference code written here, independently of each family's own models.

For every instance checked: the answer key agrees with an independent reference
implementation; the reference fix passes every hidden check and the visible tests; the
original files and every declared shortcut fail some hidden check; the visible tests fail on
the original files but pass once only the hidden bugs are left (`visible-bugs-only`); and no
hidden check reuses a visible test's inputs (so hard-coding them cannot pass).
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
from learning_loop.tasks.render import render
from learning_loop.tasks.runtime import grade as grade_runtime

FAMILY_DIR = REPO_ROOT / "evaluation" / "families"
MODULES = {"fix-parser": "fix_parser", "fix-pagination": "fix_pagination", "fix-dates": "fix_dates", "fix-shell-script": "fix_shell_script"}


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
    view = grade_runtime.MemoryView({path: (text.encode(), 0o755) for path, text in files.items()})
    return grade_runtime.grade(key, view, trusted=True)[0]


def _env_files(d: Path) -> dict[str, str]:
    root = d / "environment" / "files"
    return {"/app/" + str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def _visible_tests_pass(d: Path, tmp: Path, replace: dict[str, str]) -> bool:
    """Run the task's visible tests with `replace` {/app path: content} in place of its files."""
    shutil.copytree(d / "environment" / "files", tmp)
    for path, text in replace.items():
        target = tmp / path[len("/app/") :]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    test = next(p.name for p in tmp.glob("test_*.py"))
    return subprocess.run([sys.executable, "-B", test], cwd=tmp, capture_output=True, check=False).returncode == 0


def _check_traps(d: Path, tmp_path: Path):
    key, params, fixed, shortcuts = _task(d)
    original = _env_files(d)
    assert _reward(key, fixed) == 1.0
    assert _reward(key, original) < 1.0
    assert shortcuts and all(_reward(key, files) < 1.0 for files in shortcuts.values()), {n: _reward(key, f) for n, f in shortcuts.items()}
    assert any(n.startswith("wrong-fix-") for n in shortcuts) and "hardcode-visible" in shortcuts
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
        if c["kind"] == "equal":
            assert json.dumps(ref(*args), sort_keys=True) == json.dumps(c["expected"], sort_keys=True), c["name"]
        elif c["kind"] == "raises":
            with pytest.raises(exceptions.get(c.get("exception", "ValueError"), ValueError)):
                ref(*args)
        else:
            ref(*args)
            assert args == c["args"], c["name"]


def _checks_family(name, difficulty, seed, tmp_path, refs, exceptions=None):
    d = tmp_path / "task"
    render(FAMILIES[name], difficulty, seed, d)
    key, params = _check_traps(d, tmp_path)
    assert len({c["name"] for c in key["checks"]}) == len(key["checks"])
    _check_against(key, refs, exceptions or {})
    test_source = next((d / "environment" / "files").glob("test_*.py")).read_text()
    visible = _visible_calls(test_source, refs)
    assert visible, "the visible tests call the module"
    assert not visible & {json.dumps([c["func"], c["args"]], sort_keys=True) for c in key["checks"]}
    return key, params


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


@pytest.mark.parametrize(("difficulty", "seed"), [("easy", 1), ("medium", 1), ("hard", 1), ("hard", 2)])  # each render runs bash ~60 times
def test_fix_shell_script_traps(tmp_path, difficulty, seed):
    d = tmp_path / "task"
    render(FAMILIES["fix-shell-script"], difficulty, seed, d)
    key, _ = _check_traps(d, tmp_path)
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
    # no hidden run repeats a visible case's arguments
    test_source = (d / "environment" / "files" / "test_logsum.py").read_text()
    cases = json.loads(test_source.split('CASES = json.loads(r"""\n', 1)[1].split('\n""")', 1)[0])
    assert cases and not {json.dumps(case["args"]) for case in cases} & {json.dumps(c["argv"][2:]) for c in runs}
