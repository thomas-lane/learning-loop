"""The diagnosis families traceback-locate, slow-query, broken-symlinks, import-cycle and
config-drift: deterministic rendering, a clean randomness lint, and their answers and traps
re-derived here from the rendered files, independently of each family's own solver source
(traceback-locate by running the rendered package)."""

from __future__ import annotations

import ast
import csv
import gzip
import importlib.util
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.render import render

MODULES = {
    "traceback-locate": REPO_ROOT / "evaluation" / "families" / "traceback_locate.py",
    "slow-query": REPO_ROOT / "evaluation" / "families" / "slow_query.py",
    "broken-symlinks": REPO_ROOT / "evaluation" / "families" / "broken_symlinks.py",
    "import-cycle": REPO_ROOT / "evaluation" / "families" / "import_cycle.py",
    "config-drift": REPO_ROOT / "evaluation" / "families" / "config_drift.py",
}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_family_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FAMILY


FAMILIES = {name: _load(path) for name, path in MODULES.items()}
CASES = [(name, d) for name, fam in FAMILIES.items() for d in fam.difficulties]
SEEDS = [1, 2, 3]


def _expected(d: Path):
    return json.loads((d / "tests" / "key.json").read_text())["expected"]


def _app(d: Path) -> Path:
    return d / "environment" / "files"


# --------------------------------------------------------------------------- #
# traceback-locate: run the package; only patching the expected line fixes it
# --------------------------------------------------------------------------- #

_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in ')


def _tb_lines(text: str) -> list[str]:
    """Traceback lines without frozen-runpy frames (their numbers vary by Python version) and caret lines."""
    out, skip_source = [], False
    for ln in text.splitlines():
        if "<frozen runpy>" in ln or "runpy.py" in ln:
            skip_source = True
            continue
        if skip_source and ln.startswith("    "):
            continue
        skip_source = False
        if ln.strip() and set(ln.strip()) <= set("^~"):
            continue
        out.append(ln)
    return out


def _run(app: Path, pkg: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C.UTF-8"}
    return subprocess.run([sys.executable, "-B", "-m", f"{pkg}.main"], cwd=app, capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize("difficulty", list(FAMILIES["traceback-locate"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_traceback_locate_traps(tmp_path, difficulty, seed):
    params = render(FAMILIES["traceback-locate"], difficulty, seed, tmp_path / "t")
    app, expected, pkg = _app(tmp_path / "t"), _expected(tmp_path / "t"), params["package"]
    log = (app / "incident" / "traceback.txt").read_text()
    recorded = log[log.index("Traceback") : log.rindex("\n", 0, log.rindex("cron:")) + 1]

    # The recorded traceback is what Python prints for this code.
    run = _run(app, pkg)
    assert run.returncode == 1
    assert _tb_lines(run.stderr.replace(str(app), "/app")) == _tb_lines(recorded)

    # The answer is a line of the package that holds the key, and is not a traceback frame.
    frames = [(m.group(1), int(m.group(2))) for ln in recorded.splitlines() if (m := _FRAME.match(ln))]
    assert (expected["file"], expected["line"]) not in frames
    rel = expected["file"].removeprefix("/app/")
    key = params["key"]
    assert f'"{key}":' in (app / rel).read_text().splitlines()[expected["line"] - 1]

    # Patching the expected line to a valid value fixes the run; patching any other entry of
    # the key alone does not.
    candidates = [(p, i) for p in sorted((app / pkg).glob("*.py")) for i, ln in enumerate(p.read_text().splitlines(), 1) if ln.strip().startswith(f'"{key}":')]
    assert (app / rel, expected["line"]) in candidates and len(candidates) > 1
    for path, line in candidates:
        work = tmp_path / f"w{len(str(path))}_{line}_{path.stem}"
        shutil.copytree(app, work)
        target = work / path.relative_to(app)
        lines = target.read_text().splitlines(keepends=True)
        lines[line - 1] = re.sub(r'(":\s*).*,', r"\g<1>7,", lines[line - 1])
        target.write_text("".join(lines))
        fixed = _run(work, pkg).returncode == 0
        assert fixed == ((path, line) == (app / rel, expected["line"])), (path.name, line)

    # The bad literal (when the message shows it) appears earlier than the answer in a grep.
    m = re.search(r"invalid literal for int\(\) with base 10: '(.*)'", recorded)
    if m:
        hits = [(str(p.relative_to(app)), i) for p in sorted((app / pkg).glob("*.py")) for i, ln in enumerate(p.read_text().splitlines(), 1) if m.group(1) in ln]
        assert len(hits) > 1 and hits[0] != (rel, expected["line"])
    if difficulty != "easy":
        assert params["active_profile"] != "prod"


# --------------------------------------------------------------------------- #
# slow-query
# --------------------------------------------------------------------------- #


def _top(scores: dict) -> str | None:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return None if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]) else ranked[0][0]


@pytest.mark.parametrize("difficulty", list(FAMILIES["slow-query"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_slow_query_traps(tmp_path, difficulty, seed):
    render(FAMILIES["slow-query"], difficulty, seed, tmp_path)
    db, expected = _app(tmp_path) / "db", _expected(tmp_path)
    stats = json.loads((db / "table_stats.json").read_text())
    queries = {}
    for block in (db / "queries.sql").read_text().split("-- query: ")[1:]:
        qid, sql = block.split("\n", 1)
        table = sql.split(" FROM ")[1].split()[0]
        col = sql.split(" WHERE ")[1].split()[0]
        queries[qid.strip()] = (table, col)
    logs = {p.name: (gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()).decode() for p in sorted(db.glob("query_log*"))}
    rows = {name: list(csv.DictReader(io.StringIO(text))) for name, text in logs.items()}

    def scan(table, col, rule):
        if rule == "first":
            return not any(ix["columns"][0] == col for ix in stats[table]["indexes"])
        if rule == "anywhere":
            return not any(col in ix["columns"] for ix in stats[table]["indexes"])
        return not any(col in ix["columns"] for t in stats.values() for ix in t["indexes"])

    def total(names, keep=lambda q: True):
        t = Counter()
        for n in names:
            for r in rows[n]:
                if keep(r["query_id"]):
                    t[r["query_id"]] += int(r["duration_ms"])
        return t

    full = {q for q, (t, c) in queries.items() if scan(t, c, "first")}
    assert _top(total(logs, lambda q: q in full)) == expected
    # rows_scanned agrees with the index rule: full scans read the whole table.
    for r in (r for rs in rows.values() for r in rs):
        table, _ = queries[r["query_id"]]
        assert (int(r["rows_scanned"]) == stats[table]["rows"]) == (r["query_id"] in full)
    # Each wrong method names another query.
    counts = Counter(r["query_id"] for rs in rows.values() for r in rs)
    assert counts.most_common(1)[0][0] != expected
    slowest = max((int(r["duration_ms"]), r["query_id"]) for rs in rows.values() for r in rs)
    assert slowest[1] != expected
    assert _top(total(logs)) != expected
    if difficulty != "easy":
        table, col = queries[expected]
        assert not scan(table, col, "anywhere") and not scan(table, col, "any-table")
    if len(logs) > 1:
        for name in logs:
            assert _top(total([name], lambda q: q in full)) != expected, name


# --------------------------------------------------------------------------- #
# broken-symlinks: the real filesystem and a follow-until-it-exists walk of the log
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("difficulty", list(FAMILIES["broken-symlinks"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_broken_symlinks_traps(tmp_path, difficulty, seed):
    render(FAMILIES["broken-symlinks"], difficulty, seed, tmp_path)
    app, expected = _app(tmp_path), _expected(tmp_path)
    srv = app / "srv"
    renames = [ln.split()[1:] for ln in (app / "renames.log").read_text().splitlines()]
    links = {}
    for dirpath, dirnames, filenames in os.walk(srv):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            if p.is_symlink():
                links[p.relative_to(srv).as_posix()] = p

    def target_of(p: Path) -> str:
        text = os.readlink(p)
        full = text if text.startswith("/") else posixpath.join("/app/srv", posixpath.dirname(p.relative_to(srv).as_posix()), text)
        return posixpath.relpath(posixpath.normpath(full), "/app/srv")

    # Absolute targets name /app/srv inside the container, so they are resolved under `srv` here.
    broken = {k: p for k, p in links.items() if not (srv / target_of(p)).is_file()}
    working = set(links) - set(broken)
    assert all(p.exists() for k, p in links.items() if k in working and not os.readlink(p).startswith("/"))
    assert all(not p.exists() for k, p in broken.items() if not os.readlink(p).startswith("/"))

    def follow(path: str, steps: int | None = None, dirs: bool = True) -> str:
        n = 0
        while not (srv / path).is_file() and (steps is None or n < steps):
            for kind, old, _, new in renames:
                if kind == "file" and path == old or kind == "dir" and dirs and path.startswith(old + "/"):
                    path = new + path[len(old):]
                    break
            else:
                break
            n += 1
        return path

    assert expected == {k: follow(target_of(p)) for k, p in broken.items()}
    assert all((srv / v).is_file() for v in expected.values())
    assert working, "working links make 'report every link' fail"
    assert any("/" in k for k in broken), "a broken link in a subdirectory"
    assert all(os.readlink(broken[k]) != v and target_of(broken[k]) != v for k, v in expected.items())
    if difficulty != "easy":
        assert any(follow(target_of(p), steps=1) != expected[k] for k, p in broken.items()), "a chain longer than one rename"
        assert any(not os.readlink(p).startswith("/") and "/" in k and posixpath.normpath(os.readlink(p)) != target_of(p) for k, p in broken.items())
    if difficulty == "hard":
        assert any(follow(target_of(p), dirs=False) != expected[k] for k, p in broken.items()), "a directory rename matters"


# --------------------------------------------------------------------------- #
# import-cycle: the import graph, its elementary cycles, and the fake imports
# --------------------------------------------------------------------------- #


def _modules(app: Path, pkg: str) -> dict[str, str]:
    return {p.relative_to(app).with_suffix("").as_posix().replace("/", "."): p.read_text() for p in sorted((app / pkg).rglob("*.py")) if p.name != "__init__.py"}


def _import_graph(mods: dict[str, str], top_level_only: bool = False) -> dict[str, set[str]]:
    g = {}
    for name, src in mods.items():
        tree = ast.parse(src)
        out = set()
        for node in tree.body if top_level_only else ast.walk(tree):
            if isinstance(node, ast.Import):
                out |= {a.name for a in node.names if a.name in mods}
            elif isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), name.rsplit(".", 1)[0]) if node.level else node.module
                for a in node.names:
                    if f"{base}.{a.name}" in mods:
                        out.add(f"{base}.{a.name}")
                    elif base in mods:
                        out.add(base)
        g[name] = out
    return g


def _cycles(g: dict[str, set[str]]) -> set[tuple[str, ...]]:
    """Every elementary cycle, each rotated to start at its smallest member."""
    found = set()

    def walk(start, node, path):
        for nxt in g.get(node, ()):
            if nxt == start:
                i = path.index(min(path))
                found.add(tuple(path[i:] + path[:i]))
            elif nxt > start and nxt not in path:
                walk(start, nxt, path + [nxt])

    for s in g:
        walk(s, s, [s])
    return found


@pytest.mark.parametrize("difficulty", list(FAMILIES["import-cycle"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_import_cycle_traps(tmp_path, difficulty, seed):
    params = render(FAMILIES["import-cycle"], difficulty, seed, tmp_path)
    app, expected, pkg = _app(tmp_path), _expected(tmp_path), params["package"]
    mods = _modules(app, pkg)
    g = _import_graph(mods)
    assert _cycles(g) == {tuple(expected)}
    assert len(expected) >= 3 and min(mods) not in expected
    # Fake imports in comments and strings add cycles.
    fake = {n: set(e) for n, e in g.items()}
    for name, src in mods.items():
        for m in re.finditer(r"(?:from|import)\s+([\w.]+)", src):
            if m.group(1) in mods and m.group(1) not in g[name] and m.group(1) != name:
                fake[name].add(m.group(1))
    assert len(_cycles(fake)) > 1
    # The cycle uses a `from <package> import <module>` form (or its relative equivalent).
    edges = list(zip(expected, expected[1:] + expected[:1]))
    pkg_forms = []
    for a, b in edges:
        tree = ast.parse(mods[a])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(x.name == b.rsplit(".", 1)[1] for x in node.names):
                pkg_forms.append((a, b, node.level))
    assert pkg_forms
    if difficulty != "easy":
        assert any(level for *_, level in pkg_forms) or any(
            isinstance(n, ast.ImportFrom) and n.level for a, _ in edges for n in ast.walk(ast.parse(mods[a]))
        )
    if difficulty == "hard":
        assert _cycles(_import_graph(mods, top_level_only=True)) == set()


# --------------------------------------------------------------------------- #
# config-drift
# --------------------------------------------------------------------------- #


def _drift(base, host):
    out = {}

    def walk(b, h, prefix):
        for k in sorted(set(b) | set(h)):
            bv, hv = b.get(k), h.get(k)
            if isinstance(bv, dict) or isinstance(hv, dict):
                walk(bv if isinstance(bv, dict) else {}, hv if isinstance(hv, dict) else {}, prefix + k + ".")
            elif json.dumps(bv) != json.dumps(hv):
                out[prefix + k] = {"baseline": bv, "host": hv}

    walk(base, host, "")
    return out


@pytest.mark.parametrize("difficulty", list(FAMILIES["config-drift"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_config_drift_traps(tmp_path, difficulty, seed):
    render(FAMILIES["config-drift"], difficulty, seed, tmp_path)
    cfg, expected = _app(tmp_path) / "configs", _expected(tmp_path)
    base_text = (cfg / "baseline.json").read_text()
    base = json.loads(base_text)
    hosts = {p.stem: p for p in sorted((cfg / "hosts").rglob("*.json"))}
    computed = {name: d for name, p in hosts.items() if (d := _drift(base, json.loads(p.read_text())))}
    assert computed == expected
    assert any(p.read_text() != base_text and name not in expected for name, p in hosts.items()), "a format-only host"
    assert all("." in k for d in expected.values() for k in d), "every drift is nested"
    found = [v for d in expected.values() for v in d.values()]
    if difficulty != "easy":
        assert any(v["baseline"] is None for v in found) and any(v["host"] is None for v in found)
    if difficulty == "hard":
        assert any(str(v["baseline"]) == str(v["host"]) for v in found)
        assert any(hosts[n].parent != cfg / "hosts" for n in expected)
