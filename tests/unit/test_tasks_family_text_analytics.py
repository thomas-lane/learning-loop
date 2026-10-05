"""The text-analytics families latency-p95, session-count and error-burst: deterministic
rendering, a clean randomness lint, and their traps re-derived here from the rendered files,
independently of each family's own solver source."""

from __future__ import annotations

import gzip
import importlib.util
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.render import render

MODULES = {
    "latency-p95": REPO_ROOT / "evaluation" / "families" / "latency_p95.py",
    "session-count": REPO_ROOT / "evaluation" / "families" / "session_count.py",
    "error-burst": REPO_ROOT / "evaluation" / "families" / "error_burst.py",
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


def _files(d: Path, sub: str) -> dict[str, str]:
    root = d / "environment" / "files" / sub
    return {
        str(p.relative_to(root)): (gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()).decode()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _unique_max(scores: dict):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


# --------------------------------------------------------------------------- #
# latency-p95
# --------------------------------------------------------------------------- #


def _latency_requests(files):
    """[(file, url, latency in ms, latency as written)]"""
    out = []
    for name, text in files.items():
        for line in text.splitlines():
            if name.split("/")[-1].startswith("app.jsonl"):
                rec = json.loads(line)
                out.append((name, rec["url"], rec["duration_ms"], rec["duration_ms"]))
            else:
                raw = float(line.rsplit(" ", 1)[1])
                out.append((name, line.split('"')[1].split(" ")[1], round(raw * 1000), raw))
    return out


def _nearest_rank(v):
    v = sorted(v)
    return v[math.ceil(len(v) * 95 / 100) - 1]


def _numpy_linear(v):
    v = sorted(v)
    k = (len(v) - 1) * 0.95
    f = math.floor(k)
    return v[f] if f + 1 >= len(v) else v[f] * (f + 1 - k) + v[f + 1] * (k - f)


def _latency_top(reqs, stat, names=None, key=lambda url: url.split("?")[0], value=lambda r: r[2]):
    groups = defaultdict(list)
    for r in reqs:
        if names is None or r[0] in names:
            groups[key(r[1])].append(value(r))
    return _unique_max({k: stat(v) for k, v in groups.items()})


@pytest.mark.parametrize("difficulty", list(FAMILIES["latency-p95"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_latency_p95_traps(tmp_path, difficulty, seed):
    render(FAMILIES["latency-p95"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path, "logs"), _expected(tmp_path)
    reqs = _latency_requests(files)
    assert "?" not in expected
    assert _latency_top(reqs, _nearest_rank) == expected
    assert _latency_top(reqs, statistics.mean) != expected
    assert _latency_top(reqs, max) != expected
    assert _latency_top(reqs, lambda v: sorted(v)[int(0.95 * len(v))]) != expected
    assert _latency_top(reqs, _numpy_linear) != expected
    assert _latency_top(reqs, lambda v: statistics.quantiles(v, n=20)[-1]) != expected
    assert _latency_top(reqs, _nearest_rank, value=lambda r: r[3]) != expected  # seconds and ms mixed
    text = {n for n in files if n.split("/")[-1].startswith("access.log")}
    assert text and text != set(files)
    assert _latency_top(reqs, _nearest_rank, text) != expected
    assert _latency_top(reqs, _nearest_rank, set(files) - text) != expected
    for k in range(1, len(files)):
        for combo in itertools.combinations(files, k):
            assert _latency_top(reqs, _nearest_rank, set(combo)) != expected, combo
    if FAMILIES["latency-p95"].difficulties[difficulty]["queries"]:
        assert any("?" in r[1] for r in reqs)
        assert _latency_top(reqs, _nearest_rank, key=lambda url: url) != expected


# --------------------------------------------------------------------------- #
# session-count
# --------------------------------------------------------------------------- #


def _events(files):
    """[(file, aware datetime, user)] in file order."""
    out = []
    for name, text in files.items():
        for line in text.splitlines():
            stamp, user = line.split()[:2]
            assert user.startswith("user=")
            out.append((name, datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc), user[5:]))
    return out


def _sessions(evs, is_new):
    """Sessions in (time, user) pairs taken in the given order."""
    last, start, n = {}, {}, 0
    for t, u in evs:
        if u not in last or is_new(t - last[u], t - start[u]):
            n += 1
            start[u] = t
        last[u] = t
    return n


GAP = timedelta(minutes=30)


def _truth(evs):
    by_user = defaultdict(list)
    for _, t, u in evs:
        by_user[u].append(t)
    total = 0
    for times in by_user.values():
        times.sort()
        total += 1 + sum(1 for a, b in itertools.pairwise(times) if b - a > GAP)
    return total


@pytest.mark.parametrize("difficulty", list(FAMILIES["session-count"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_session_count_traps(tmp_path, difficulty, seed):
    render(FAMILIES["session-count"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path, "events"), _expected(tmp_path)
    evs = _events(files)
    truth = _truth(evs)
    assert truth == expected
    timeline = sorted((t, u) for _, t, u in evs)
    by_user = defaultdict(list)
    for t, u in timeline:
        by_user[u].append(t)
    gaps = [b - a for times in by_user.values() for a, b in itertools.pairwise(times)]
    assert GAP in gaps and GAP + timedelta(seconds=1) in gaps  # both sides of the boundary occur
    assert _sessions(timeline, lambda gap, since_start: gap >= GAP) > truth
    assert _sessions(timeline, lambda gap, since_start: since_start > GAP) > truth
    assert len(evs) != truth and len(by_user) < truth
    everyone = [t for t, _ in timeline]
    assert 1 + sum(1 for a, b in itertools.pairwise(everyone) if b - a > GAP) != truth
    per_file = sum(_truth([e for e in evs if e[0] == name]) for name in files)
    assert per_file > truth
    in_file_order = [(t, u) for _, t, u in evs]  # files concatenated in name order, not sorted
    assert _sessions(in_file_order, lambda gap, since_start: gap > GAP) != truth
    if FAMILIES["session-count"].difficulties[difficulty]["mobile_upload_order"]:
        mobile = [t for name, t, _ in evs if name == "mobile.log"]
        assert mobile != sorted(mobile)
    if FAMILIES["session-count"].difficulties[difficulty]["archive"]:
        assert _truth([e for e in evs if not e[0].endswith(".gz")]) != truth
        assert _truth([e for e in evs if "/" not in e[0]]) != truth


# --------------------------------------------------------------------------- #
# error-burst
# --------------------------------------------------------------------------- #


def _error_lines(files):
    """[(file, aware datetime or naive for a header file's declared offset, line)] for every log line."""
    out = []
    for name, text in files.items():
        declared = None
        for line in text.splitlines():
            if line.startswith("# timezone: UTC"):
                declared = line.split()[2][len("UTC"):]
                continue
            stamp = datetime.fromisoformat(line.split()[0])
            if stamp.tzinfo is None:
                assert declared is not None
                stamp = stamp.replace(tzinfo=datetime.strptime(declared, "%z").tzinfo)
            out.append((name, stamp, line))
    return out


def _minute(dt):
    return dt.strftime("%Y-%m-%d %H:%M")


def _burst(lines, when, names=None, take=lambda line: line.split()[1] == "ERROR"):
    return _unique_max(Counter(_minute(when(dt)) for n, dt, line in lines if (names is None or n in names) and take(line)))


def _utc(dt):
    return dt.astimezone(timezone.utc)


@pytest.mark.parametrize("difficulty", list(FAMILIES["error-burst"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_error_burst_traps(tmp_path, difficulty, seed):
    render(FAMILIES["error-burst"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path, "logs"), _expected(tmp_path)
    lines = _error_lines(files)
    offsets = {n: {dt.utcoffset() for f, dt, _ in lines if f == n} for n in files}
    assert all(len(o) == 1 for o in offsets.values())
    assert len({next(iter(o)) for o in offsets.values()}) == len(files)  # every file in its own zone
    assert _burst(lines, _utc) == expected
    assert _burst(lines, lambda dt: dt.replace(tzinfo=None)) != expected  # local clock readings
    assert _burst(lines, lambda dt: (dt.replace(tzinfo=None) + dt.utcoffset())) != expected  # sign flipped
    for k in range(1, len(files)):
        for combo in itertools.combinations(files, k):
            assert _burst(lines, _utc, set(combo)) != expected, combo
    if FAMILIES["error-burst"].difficulties[difficulty]["lookalikes"]:
        assert any("ERROR" in line and line.split()[1] != "ERROR" for _, _, line in lines)
        assert _burst(lines, _utc, take=lambda line: "ERROR" in line) != expected
    half_hour = [n for n, o in offsets.items() if next(iter(o)).total_seconds() % 3600]
    if difficulty != "easy":
        assert half_hour

        def hours_only(dt):
            off = dt.utcoffset()
            whole = timedelta(hours=int(off.total_seconds() / 3600))
            return dt.replace(tzinfo=None) - whole

        assert _burst(lines, hours_only) != expected
    if difficulty == "hard":
        header_files = {n for n, t in files.items() if t.startswith("# timezone:")}
        assert len(header_files) == 1
        as_utc = [(n, dt.replace(tzinfo=timezone.utc) if n in header_files else dt, line) for n, dt, line in lines]
        assert _burst(as_utc, _utc) != expected  # the header file's local times read as UTC
