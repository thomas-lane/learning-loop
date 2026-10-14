"""The text-analytics families: one instruction at every difficulty, and their traps
re-derived here from the rendered files, independently of each family's own solver source
(count-errors' and log-triage's basic traps are in test_tasks_families.py)."""

from __future__ import annotations

import gzip
import importlib.util
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.generate import generate_spec
from learning_loop.tasks.render import render

MODULES = {
    "count-errors": REPO_ROOT / "evaluation" / "families" / "count_errors.py",
    "log-triage": REPO_ROOT / "evaluation" / "families" / "log_triage.py",
    "latency-p95": REPO_ROOT / "evaluation" / "families" / "latency_p95.py",
    "session-count": REPO_ROOT / "evaluation" / "families" / "session_count.py",
    "error-burst": REPO_ROOT / "evaluation" / "families" / "error_burst.py",
}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_family_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOADED = {name: _load(path) for name, path in MODULES.items()}
FAMILIES = {name: module.FAMILY for name, module in LOADED.items()}
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


@pytest.mark.parametrize("name", list(MODULES))
@pytest.mark.parametrize("seed", SEEDS)
def test_instruction_is_the_same_at_every_difficulty(name, seed):
    """Difficulty comes from the data alone: these instructions draw no instance values, so
    every level gets the module's one INSTRUCTION, which states every rule of every level."""
    fam = FAMILIES[name]
    texts = {d: generate_spec(fam, d, seed).spec.instruction for d in fam.difficulties}
    assert set(texts.values()) == {LOADED[name].INSTRUCTION}, name


@pytest.mark.parametrize(
    "name,phrases",
    [
        ("count-errors", ["including any subdirectories", "`ERROR` (case-sensitive)"]),
        ("log-triage", ["gzip-compressed and archived logs in any subdirectories"]),
        ("latency-p95", ["gzip-compressed and archived logs in any subdirectories"]),
        ("session-count", ["separated by single spaces", "**all** the files", "strictly more than 1800 seconds"]),
        ("error-burst", ["`2026-01-31T09:05:00+02:00` is 07:05 UTC", "declares, in its first line, the UTC offset", "gzip-compressed"]),
    ],
)
def test_instruction_states_every_level_s_rules(name, phrases):
    """Rules that used to appear only at some levels (subdirectories, compressed files, the
    header offset, the worked offset example, the session gap) appear at every level."""
    fam = FAMILIES[name]
    for d in fam.difficulties:
        text = generate_spec(fam, d, 1).spec.instruction
        assert all(p in text for p in phrases), (d, [p for p in phrases if p not in text])


# --------------------------------------------------------------------------- #
# count-errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("difficulty", list(FAMILIES["count-errors"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_count_errors_in_word_and_twice_error_lines(tmp_path, difficulty, seed):
    """Medium and hard have lines with `ERROR` only inside a longer word and lines with it
    twice, so whole-word matching, the level field and occurrence counts all miss; easy has
    neither, and only hard has a log in a subdirectory."""
    render(FAMILIES["count-errors"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path, "data"), int(_expected(tmp_path))
    logs = [ln for n, t in files.items() if n.endswith(".log") for ln in t.splitlines()]
    assert sum("ERROR" in ln for ln in logs) == expected
    in_word = [ln for ln in logs if "ERROR" in ln and not re.search(r"\bERROR\b", ln)]
    twice = [ln for ln in logs if ln.count("ERROR") > 1]
    whole_word = sum(re.search(r"\bERROR\b", ln) is not None for ln in logs)
    level = sum(ln.split()[1] == "ERROR" for ln in logs)
    occurrences = sum(ln.count("ERROR") for ln in logs)
    assert any("/" in n for n in files) == (difficulty == "hard")
    if difficulty == "easy":
        assert not in_word and not twice and whole_word == level == occurrences == expected
    else:
        assert in_word and twice
        assert whole_word < expected and level < expected and occurrences > expected


# --------------------------------------------------------------------------- #
# log-triage
# --------------------------------------------------------------------------- #


def _ip_top(lines, take):
    return _unique_max(Counter(ln.split()[0] for ln in lines if take(ln)))


def _size_decoy_sizes(lines):
    return [int(ln.split()[9]) for ln in lines if ln.split()[8] == "200" and ln.split()[9].startswith("5")]


@pytest.mark.parametrize("difficulty", list(FAMILIES["log-triage"].difficulties))
@pytest.mark.parametrize("seed", SEEDS)
def test_log_triage_size_decoy_is_graded(tmp_path, difficulty, seed):
    """Every level has a size decoy. With 4-digit sizes 5000-5999 only a grep not anchored
    after the status fails; with 3-digit sizes 500-599 (hard only) a grep with spaces on both
    sides fails too. No other response size starts with 5."""
    render(FAMILIES["log-triage"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path, "logs"), _expected(tmp_path)
    lines = [ln for t in files.values() for ln in t.splitlines()]
    decoy_sizes = _size_decoy_sizes(lines)
    assert all(not ln.split()[9].startswith("5") for ln in lines if ln.split()[8] != "200")
    assert _ip_top(lines, lambda ln: ln.split()[8].startswith("5")) == expected
    unanchored = _ip_top(lines, lambda ln: re.search(r" 5[0-9]{2}", ln) is not None)
    spaced = _ip_top(lines, lambda ln: re.search(r" 5[0-9]{2} ", ln) is not None)
    assert decoy_sizes and unanchored != expected
    if all(5000 <= s <= 5999 for s in decoy_sizes):
        assert spaced == expected
    else:
        assert difficulty == "hard" and all(500 <= s <= 599 for s in decoy_sizes)
        assert spaced != expected


def test_log_triage_hard_mixes_3_and_4_digit_size_decoys():
    """Easy and medium always draw the 4-digit size decoy; hard draws the 3-digit one for
    about half of its instances and the 4-digit one for the rest."""
    def digits(difficulty, seed):
        spec = generate_spec(FAMILIES["log-triage"], difficulty, seed).spec
        lines = [ln for rel, data in spec.files.items() for ln in (gzip.decompress(data) if rel.endswith(".gz") else data.encode()).decode().splitlines()]
        return {len(str(s)) for s in _size_decoy_sizes(lines)}

    seeds = range(1, 21)
    for d in ("easy", "medium"):
        assert all(digits(d, seed) == {4} for seed in seeds), d
    hard = Counter(frozenset(digits("hard", seed)) for seed in seeds)
    assert set(hard) == {frozenset({3}), frozenset({4})} and min(hard.values()) >= 5, hard


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
                out.append((name, rec["path"], rec["duration_ms"], rec["duration_ms"]))
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


@pytest.mark.parametrize("difficulty", list(FAMILIES["latency-p95"].difficulties))
def test_latency_p95_json_logs_hold_the_url_in_path(tmp_path, difficulty):
    """The JSON logs carry the URL in `path`, the field name solvers assume, so a solver that
    does not read the file first still counts the JSON requests; only the declared traps decide."""
    render(FAMILIES["latency-p95"], difficulty, 1, tmp_path)
    files = _files(tmp_path, "logs")
    json_logs = [t for n, t in files.items() if n.split("/")[-1].startswith("app.jsonl")]
    assert json_logs
    for text in json_logs:
        for line in text.splitlines():
            assert set(json.loads(line)) == {"ts", "level", "method", "path", "status", "duration_ms"}


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
    for text in files.values():  # the line format the instruction states: single-space fields
        for line in text.splitlines():
            fields = line.split(" ")
            assert "" not in fields and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", fields[0]) and fields[1].startswith("user=")
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


def test_error_burst_hard_is_medium_with_the_header_file():
    """Hard differs from medium only by the header-offset file; no level has a compressed log,
    so the header offset is hard's one extra trap."""
    fam = FAMILIES["error-burst"]
    medium, hard = fam.difficulties["medium"], fam.difficulties["hard"]
    assert set(medium["files"]) ^ set(hard["files"]) == {"orders-us.log", "orders-us-legacy.log"}
    assert medium["lookalikes"] == hard["lookalikes"]
    for d in fam.difficulties:
        generated = generate_spec(fam, d, 1)
        assert not any(rel.endswith(".gz") for rel in generated.spec.files), d
        assert "plain-only" not in generated.spec.shortcuts
