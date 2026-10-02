"""fix-stats family: fix injected bugs in a tiny statistics library (partial credit).

Difficulty = library size and number of injected bugs, and how many of the
bugged functions the *visible* tests cover (hard: some bugs are only described
by docstrings). Hidden checks use seed-specific inputs that differ from the
visible tests, so hard-coding the visible expectations fails.

Checked at generation time by running the exact hidden-test runner in-process:
the reference library scores 1.0, every injected bug fails at least one
hidden check, and the no-op baseline (buggy library) score is recorded in
params as `nop_reward`.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from .common import PYCACHE_EXCLUDES, PYTHON_BASE, rng_for, task_toml, write

FAMILY = "fix-stats"
VERSION = 2  # v2: base images pinned by digest
RNG_VERSION = 1  # random stream; unchanged since v1, so instances keep their content
SKILLS = ["python", "debugging", "reading-docs"]

DIFFICULTIES = {
    "easy": {"functions": ["mean", "median", "stdev", "percentile"], "n_bugs": 1, "visible_bug_tests": 1.0},
    "medium": {"functions": ["mean", "median", "variance", "stdev", "percentile"], "n_bugs": 2, "visible_bug_tests": 0.5},
    "hard": {"functions": ["mean", "median", "variance", "stdev", "percentile", "mode"], "n_bugs": 4, "visible_bug_tests": 0.25},
}

HEADER = '"""Tiny statistics helpers."""\n\nimport math\n'

CORRECT = {
    "mean": '''def mean(xs: list[float]) -> float:
    """Arithmetic mean. Raises ValueError on an empty list."""
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)
''',
    "median": '''def median(xs: list[float]) -> float:
    """Median. For an even number of values, the mean of the two middle values.
    Does not modify the input. Raises ValueError on an empty list."""
    if not xs:
        raise ValueError("median of empty list")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
''',
    "variance": '''def variance(xs: list[float]) -> float:
    """*Sample* variance (divides by n - 1). Needs at least 2 values."""
    if len(xs) < 2:
        raise ValueError("variance needs at least two values")
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
''',
    "stdev": '''def stdev(xs: list[float]) -> float:
    """*Sample* standard deviation (divides by n - 1). Needs at least 2 values."""
    if len(xs) < 2:
        raise ValueError("stdev needs at least two values")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
''',
    "percentile": '''def percentile(xs: list[float], p: float) -> float:
    """p-th percentile (0 <= p <= 100) using linear interpolation between
    closest ranks, i.e. the same as numpy's default ("linear") method."""
    if not xs:
        raise ValueError("percentile of empty list")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
''',
    "mode": '''def mode(xs: list[float]) -> float:
    """Most common value. Ties are broken by returning the *smallest* of the
    most common values. Raises ValueError on an empty list."""
    if not xs:
        raise ValueError("mode of empty list")
    counts: dict[float, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return min(x for x, c in counts.items() if c == best)
''',
}

def _swap(func: str, old: str, new: str) -> str:
    src = CORRECT[func]
    assert old in src, (func, old)
    return src.replace(old, new)


# Several realistic bug variants per function (more distinct instances per difficulty).
BUGGY: dict[str, dict[str, str]] = {
    "mean": {
        "floor_division": _swap("mean", "return sum(xs) / len(xs)", "return sum(xs) // len(xs)"),
        "skips_first": _swap("mean", "return sum(xs) / len(xs)", "return sum(xs[1:]) / len(xs)"),
    },
    "median": {
        "mutates_upper_middle": _swap(
            "median",
            "    s = sorted(xs)\n    n = len(s)\n    mid = n // 2\n    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2\n",
            "    xs.sort()\n    return xs[len(xs) // 2]\n",
        ),
        "upper_middle": _swap("median", "return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2", "return s[mid]"),
        "always_averages": _swap("median", "return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2", "return (s[mid - 1] + s[mid]) / 2"),
    },
    "variance": {
        "population": _swap("variance", "/ (len(xs) - 1)", "/ len(xs)"),
        "absolute_deviation": _swap("variance", "(x - m) ** 2", "abs(x - m)"),
    },
    "stdev": {
        "population": _swap("stdev", "/ (len(xs) - 1))", "/ len(xs))"),
        "missing_sqrt": _swap("stdev", "return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))", "return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)"),
    },
    "percentile": {
        "rank_off_by_one": _swap("percentile", "k = (len(s) - 1) * p / 100", "k = (len(s)) * p / 100"),
        "no_interpolation": _swap("percentile", "return s[lo] + (s[hi] - s[lo]) * (k - lo)", "return s[round(k)]"),
        "unsorted": _swap("percentile", "s = sorted(xs)", "s = list(xs)"),
    },
    "mode": {
        "first_seen_tie": _swap("mode", "    best = max(counts.values())\n    return min(x for x, c in counts.items() if c == best)\n", "    return max(counts, key=counts.get)\n"),
        "largest_tie": _swap("mode", "return min(x for x, c in counts.items() if c == best)", "return max(x for x, c in counts.items() if c == best)"),
    },
}

# Grading is split in two so that agent code never runs where the reward is written:
#
#   OBSERVE  runs in a child process (unprivileged user, copy of /app/stats.py). It gets the
#            check *inputs* only (never expected values) and reports raw observations:
#            the returned number, the exception type, or the list after the call.
#   JUDGE    runs in the grading process. It never imports the artifact; it accepts only
#            strictly typed observations (plain JSON numbers, literal true/false) for known
#            checks, compares them with the hidden expected values, and anything else fails.
#
# The generator self-check runs exactly this code in-process (on our own reference and
# buggy libraries) through `run_checks`, with a JSON round trip in between.
OBSERVE = '''
def observe(lib, checks):
    import copy
    import json

    def plain(v):
        try:
            return json.loads(json.dumps(v, allow_nan=False))
        except Exception:
            return None

    out = {}
    for c in checks:
        fn = lib.get(c["func"]) if isinstance(lib, dict) else None
        if not callable(fn):
            out[c["name"]] = {"missing": True}
            continue
        args = copy.deepcopy(c["args"])
        try:
            got = fn(*args)
        except BaseException as e:  # noqa: BLE001 - anything the artifact raises is an observation
            out[c["name"]] = {"raised": type(e).__name__[:200], "value_error": isinstance(e, ValueError)}
            continue
        if c["kind"] == "no_mutation":
            out[c["name"]] = {"after": plain(args[0])}
        elif c["kind"] == "raises":
            out[c["name"]] = {"returned": True}
        else:
            ok_type = isinstance(got, (int, float)) and not isinstance(got, bool)
            out[c["name"]] = {"value": plain(got) if ok_type else None}
    return out
'''

JUDGE = '''
def judge(checks, observations):
    import json
    import math

    def num(x):
        return type(x) in (int, float) and math.isfinite(x)

    results = {}
    obs_all = observations if isinstance(observations, dict) else {}
    for c in checks:
        o = obs_all.get(c["name"])
        ok = False
        if isinstance(o, dict):
            if c["kind"] == "raises":
                ok = set(o) == {"raised", "value_error"} and o["value_error"] is True
            elif c["kind"] == "no_mutation":
                ok = set(o) == {"after"} and json.dumps(o["after"]) == json.dumps(c["args"][0])
            elif set(o) == {"value"} and num(o["value"]):
                exp = c["expected"]
                tol = c.get("abs_tol")
                ok = abs(o["value"] - exp) <= (tol if tol is not None else 1e-9 * max(1.0, abs(exp)))
        results[c["name"]] = ok is True
    return results
'''

RUNNER = OBSERVE + JUDGE + '''
def run_checks(lib, checks):
    import json
    inputs = [{k: v for k, v in c.items() if k not in ("expected", "abs_tol")} for c in checks]
    return judge(checks, json.loads(json.dumps(observe(lib, inputs))))
'''

# The child process: imports the copied artifact and prints one marker line of observations.
CHILD = '''import json
import sys

MARKER = "@@FIX_STATS_OBSERVATIONS@@ "
''' + OBSERVE + '''
sys.path.insert(0, sys.argv[1])
checks = json.loads(sys.stdin.read())
try:
    import stats

    lib = dict(vars(stats))
except BaseException as e:  # noqa: BLE001
    print("import failed: " + type(e).__name__)
    lib = None
obs = observe(lib, checks) if lib is not None else {}
sys.stdout.write("\\n" + MARKER + json.dumps(obs) + "\\n")
sys.stdout.flush()
'''

TEST_HIDDEN = '''"""Hidden verifier tests (separate verifier container). Writes partial credit to
/logs/verifier/reward.json. __ORIGIN__

The agent's artifact (/app/stats.py) is never imported by this process. A copy runs in a
child `python3 -I -B` process as the unprivileged user `nobody`, which cannot read /tests
(made 0700 first) and gets only the check inputs. The child prints raw observations as one
marker line; this process accepts only strictly typed observations for known checks,
compares them with the expected values, kills every process left by that user, removes any
reward file it may have planted, and only then writes reward.json.
"""

import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CHECKS = json.loads(__CHECKS_JSON__)
ARTIFACT = "/app/stats.py"
VERIFIER_DIR = "/logs/verifier"
CHILD_TIMEOUT_SEC = 30
CHILD_SOURCE = __CHILD_SOURCE__
MARKER = "@@FIX_STATS_OBSERVATIONS@@ "
__JUDGE__

def kill_user_processes(uid):
    for _ in range(50):
        found = False
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/status") as f:
                    uids = next(line for line in f if line.startswith("Uid:")).split()[1:]
            except (OSError, StopIteration):
                continue
            if str(uid) in uids:
                found = True
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
        if not found:
            return
        time.sleep(0.05)


def observe_isolated():
    if os.geteuid() != 0:
        print("grader must run as root to isolate the artifact; failing closed")
        return {}
    nobody = pwd.getpwnam("nobody")
    os.chmod("/tests", 0o700)
    work = tempfile.mkdtemp(prefix="grade-")
    stdout = ""
    try:
        os.chmod(work, 0o755)
        if os.path.isfile(ARTIFACT) and not os.path.islink(ARTIFACT):
            with open(ARTIFACT, "rb") as src, open(os.path.join(work, "stats.py"), "wb") as dst:
                dst.write(src.read())
        else:
            print("missing artifact: " + ARTIFACT)
        with open(os.path.join(work, "child.py"), "w") as f:
            f.write(CHILD_SOURCE)
        for name in os.listdir(work):
            os.chmod(os.path.join(work, name), 0o644)
        inputs = [{k: v for k, v in c.items() if k not in ("expected", "abs_tol")} for c in CHECKS]
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", os.path.join(work, "child.py"), work],
                input=json.dumps(inputs), capture_output=True, text=True, cwd=work,
                user=nobody.pw_uid, group=nobody.pw_gid, extra_groups=[], start_new_session=True,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8", "HOME": work},
                timeout=CHILD_TIMEOUT_SEC,
            )
            stdout = proc.stdout
            print(f"child exit code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print(f"child timed out after {CHILD_TIMEOUT_SEC}s")
    finally:
        kill_user_processes(nobody.pw_uid)
        shutil.rmtree(work, ignore_errors=True)
    lines = [ln[len(MARKER):] for ln in stdout.splitlines() if ln.startswith(MARKER)]
    if len(lines) != 1:
        print(f"expected exactly one observation line, got {len(lines)}")
        return {}
    try:
        return json.loads(lines[0])
    except ValueError:
        print("observation line is not JSON")
        return {}


def write_reward(value):
    for name in ("reward.json", "reward.txt"):
        path = os.path.join(VERIFIER_DIR, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.unlink(path)
    fd = os.open(os.path.join(VERIFIER_DIR, "reward.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as f:
        # reward.json may hold several numeric keys; Harbor averages each key across
        # trials (a key missing from some tasks counts as 0 there, so keep keys uniform).
        json.dump({"reward": value}, f)


results = judge(CHECKS, observe_isolated())
for name, ok in results.items():
    print(("PASS " if ok else "FAIL ") + name)
passed = sum(1 for ok in results.values() if ok is True)
write_reward(passed / len(CHECKS))
print(f"{passed}/{len(CHECKS)} checks passed")
'''


def hidden_test_source(checks: list[dict[str, Any]], origin: str = "Generated - do not edit by hand.") -> str:
    """The separate-verifier grader for `checks` (also renders the hand-written task's grader)."""
    return (
        TEST_HIDDEN.replace("__ORIGIN__", origin)
        .replace("__CHECKS_JSON__", repr(json.dumps(checks, sort_keys=True)))
        .replace("__CHILD_SOURCE__", repr(CHILD))
        .replace("__JUDGE__", JUDGE)
    )


def _visible_case(rng: random.Random, func: str) -> tuple[str, str, dict[str, Any]]:
    """(test name, assert line, check dict) with seed-specific inputs."""
    def ints(n: int, lo: int = 1, hi: int = 40) -> list[int]:
        return [rng.randint(lo, hi) for _ in range(n)]

    if func == "mean":
        xs = ints(4)
        while sum(xs) % 4 == 0 or xs[0] == 0:
            xs = ints(4)
        args: list[Any] = [xs]
    elif func == "median":
        args = [ints(rng.choice([4, 5]))]
    elif func in ("variance", "stdev"):
        args = [ints(rng.randint(5, 8))]
    elif func == "percentile":
        xs = sorted(set(ints(6, 1, 90)), reverse=True)
        while len(xs) < 4:
            xs = sorted(set(ints(6, 1, 90)), reverse=True)
        args = [xs, rng.choice([25, 50, 75])]
    else:  # mode
        a, b = sorted(rng.sample(range(1, 30), 2))
        args = [[b, a, b, a, rng.randint(31, 60)]]
    exp = _reference(func, args)
    call = f"{func}({', '.join(repr(a) for a in args)})"
    line = f"assert abs({call} - {exp!r}) < 1e-9" if isinstance(exp, float) else f"assert {call} == {exp!r}"
    check = {"name": f"visible_{func}", "func": func, "kind": "value", "args": args, "expected": exp}
    return f"test_{func}", line, check


def _module(funcs: list[str], bugs: dict[str, str]) -> str:
    return HEADER + "".join("\n\n" + (BUGGY[f][bugs[f]] if f in bugs else CORRECT[f]) for f in funcs)


def _reference(func: str, args: list[Any]) -> float:
    xs = args[0]
    if func == "mean":
        return statistics.fmean(xs)
    if func == "median":
        return statistics.median(xs)
    if func == "variance":
        return statistics.variance(xs)
    if func == "stdev":
        return statistics.stdev(xs)
    if func == "percentile":
        s = sorted(xs)
        k = (len(s) - 1) * args[1] / 100
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    if func == "mode":
        counts = {x: xs.count(x) for x in xs}
        best = max(counts.values())
        return min(x for x, c in counts.items() if c == best)
    raise KeyError(func)


def _checks(rng: random.Random, funcs: list[str]) -> list[dict[str, Any]]:
    def ints(n: int, lo: int = 1, hi: int = 60) -> list[int]:
        return [rng.randint(lo, hi) for _ in range(n)]

    out: list[dict[str, Any]] = []

    def value(name: str, func: str, args: list[Any]) -> None:
        out.append({"name": name, "func": func, "kind": "value", "args": args, "expected": _reference(func, args)})

    for f in funcs:
        if f == "mean":
            xs = ints(5)
            while sum(xs) % len(xs) == 0:
                xs = ints(5)
            value("mean", f, [xs])
            out.append({"name": "mean_empty_raises", "func": f, "kind": "raises", "args": [[]]})
        elif f == "median":
            value("median_odd", f, [ints(7)])
            xs = ints(6)
            value("median_even", f, [xs])
            out.append({"name": "median_no_mutation", "func": f, "kind": "no_mutation", "args": [ints(5)]})
        elif f in ("variance", "stdev"):
            value(f"{f}_sample", f, [ints(rng.randint(6, 9))])
            out.append({"name": f"{f}_single_raises", "func": f, "kind": "raises", "args": [[rng.randint(1, 9)]]})
        elif f == "percentile":
            xs = sorted(set(ints(6, 1, 99)))
            while len(xs) < 4:
                xs = sorted(set(ints(6, 1, 99)))
            rng.shuffle(xs)  # the function must sort its input
            value("percentile_0", f, [xs, 0])
            value("percentile_100", f, [xs, 100])
            value("percentile_mid", f, [xs, rng.choice([30, 40, 60, 70, 90])])
        elif f == "mode":
            a, b = sorted(rng.sample(range(1, 50), 2))
            xs = [b, b, a, a] + ints(3, 51, 99)  # tie: larger value appears first
            value("mode_tie_smallest", f, [xs])
            value("mode_simple", f, [[rng.randint(1, 9)] * 3 + ints(3, 10, 30)])
    return out


def _exec(src: str) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 - generator self-check on our own source
    return ns


def generate(out_dir: Path, difficulty: str, seed: int) -> dict[str, Any]:
    spec = DIFFICULTIES[difficulty]
    funcs: list[str] = spec["functions"]
    rng = rng_for(FAMILY, RNG_VERSION, difficulty, seed)
    bugged_funcs = sorted(rng.sample(funcs, spec["n_bugs"]))
    bugs = {f: rng.choice(sorted(BUGGY[f])) for f in bugged_funcs}
    n_visible = max(1, round(len(bugged_funcs) * spec["visible_bug_tests"]))
    visible_bugs = sorted(rng.sample(bugged_funcs, n_visible))
    run_checks = _exec(RUNNER)["run_checks"]
    ref_lib, bug_lib = _exec(_module(funcs, {})), _exec(_module(funcs, bugs))
    for _ in range(100):
        cases = {f: _visible_case(rng, f) for f in funcs if f not in bugs or f in visible_bugs}
        vis_checks = [c for _, _, c in cases.values()]
        vis_ref, vis_bug = run_checks(ref_lib, vis_checks), run_checks(bug_lib, vis_checks)
        if all(vis_ref.values()) and all(not vis_bug[f"visible_{f}"] for f in visible_bugs):
            break
    else:  # pragma: no cover
        raise RuntimeError(f"{FAMILY}: could not build visible tests that expose the visible bugs (seed {seed})")
    for _ in range(100):
        checks = _checks(rng, funcs)
        ok_ref = run_checks(ref_lib, checks)
        res_bug = run_checks(bug_lib, checks)
        caught = all(any(not res_bug[c["name"]] for c in checks if c["func"] == f) for f in bugs)
        if all(ok_ref.values()) and caught:
            break
    else:  # pragma: no cover
        raise RuntimeError(f"{FAMILY}: could not build checks that catch every bug (seed {seed})")
    nop_reward = sum(res_bug.values()) / len(res_bug)

    # Visible tests: the non-bugged functions plus the chosen bugged ones (seed-specific inputs).
    tests = "\n\n".join(f"def {name}():\n    {line}\n" for name, line, _ in cases.values())
    test_stats = (
        '"""Run with: python3 test_stats.py"""\n\n'
        f"from stats import {', '.join(funcs)}  # noqa: F401\n\n\n"
        f"{tests}\n\n"
        'if __name__ == "__main__":\n'
        "    failed = 0\n"
        "    for name, fn in list(globals().items()):\n"
        '        if name.startswith("test_"):\n'
        "            try:\n"
        "                fn()\n"
        '                print(f"PASS {name}")\n'
        "            except Exception as e:  # noqa: BLE001\n"
        "                failed += 1\n"
        '                print(f"FAIL {name}: {e!r}")\n'
        "    raise SystemExit(1 if failed else 0)\n"
    )
    out_dir = Path(out_dir)
    buggy_src = _module(funcs, bugs)
    fixed_src = _module(funcs, {})
    write(out_dir / "environment" / "stats.py", buggy_src)
    write(out_dir / "environment" / "test_stats.py", test_stats)
    write(out_dir / "environment" / "Dockerfile", "FROM " + PYTHON_BASE + "\n\nWORKDIR /app\nCOPY stats.py test_stats.py /app/\n")
    names = ", ".join(f"`{f}`" for f in funcs)
    write(
        out_dir / "instruction.md",
        f"The small statistics library in `/app/stats.py` ({names}) has bugs. Some of the tests in `/app/test_stats.py` may fail, "
        "and some bugs may not be covered by those tests at all.\n\n"
        "Fix `stats.py` so that all functions behave as their docstrings describe. You may add tests, but do not change what the "
        "existing tests assert. Only use the Python standard library.\n",
    )
    write(out_dir / "tests" / "test_hidden.py", hidden_test_source(checks))
    write(out_dir / "tests" / "test.sh", "#!/bin/bash\npython3 /tests/test_hidden.py\n", executable=True)
    write(out_dir / "tests" / "Dockerfile", "FROM " + PYTHON_BASE + "\nCOPY test.sh test_hidden.py /tests/\n")
    write(
        out_dir / "solution" / "solve.sh",
        "#!/bin/bash\n# Reference solution: the unbugged library.\ncat > /app/stats.py <<'PY'\n" + fixed_src + "PY\ncd /app && python3 test_stats.py\n",
        executable=True,
    )
    params = {"functions": funcs, "bugs": bugs, "visible_bug_tests": visible_bugs, "n_checks": len(checks), "nop_reward": round(nop_reward, 6)}
    write(
        out_dir / "task.toml",
        task_toml(
            family=FAMILY,
            generator=f"{FAMILY}@v{VERSION}",
            generator_seed=seed,
            difficulty=difficulty,
            category="debugging",
            tags=["python", "partial-credit"],
            skills=SKILLS,
            artifacts=["/app/stats.py"],
            fingerprint_exclude=PYCACHE_EXCLUDES,
            params=params,
            comment=f"Generated by evaluation/generators ({FAMILY}@v{VERSION}, {difficulty}, seed {seed}). Do not edit by hand.",
        ),
    )
    return params
