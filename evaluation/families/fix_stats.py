"""fix-stats: fix the bugs injected into a small statistics library (partial credit).

Difficulty sets the library size, the number of injected bugs and how many of the bugged
functions the *visible* tests cover (on hard, some bugs are only described by docstrings).
Hidden checks use seed-specific inputs that differ from the visible tests, so hard-coding
the visible expectations fails. `build` redraws visible tests until they expose the
visible bugs and hidden checks until the reference library passes all of them and every
injected bug fails at least one. When some bugs are not covered by visible tests, the
shortcut `visible-bugs-only` fixes just the covered ones and must fail.
"""

import statistics

from learning_loop.tasks.runtime.grade import run_checks
from learning_loop.tasks.spec import Checks, Family, Reject, Solution, TaskSpec

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

def _visible_case(rng, func: str):
    """(test name, assert line, check dict) with seed-specific inputs."""
    def ints(n: int, lo: int = 1, hi: int = 40) -> list[int]:
        return [rng.randint(lo, hi) for _ in range(n)]

    if func == "mean":
        xs = ints(4)
        while sum(xs) % 4 == 0 or xs[0] == 0:
            xs = ints(4)
        args = [xs]
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


def _reference(func: str, args) -> float:
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


def _checks(rng, funcs: list[str]):
    def ints(n: int, lo: int = 1, hi: int = 60) -> list[int]:
        return [rng.randint(lo, hi) for _ in range(n)]

    out = []

    def value(name: str, func: str, args) -> None:
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


def _exec(src):
    ns = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 - our own reference/buggy sources
    return ns


TEST_RUNNER = """

if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    raise SystemExit(1 if failed else 0)
"""


def _write(source, then=""):
    return "cat > /app/stats.py <<'PY'\n" + source + "PY\n" + then


def build(ctx):
    rng, p = ctx.rng, ctx.params
    funcs = p["functions"]
    bugged = sorted(rng.sample(funcs, p["n_bugs"]))
    bugs = {f: rng.choice(sorted(BUGGY[f])) for f in bugged}
    n_visible = max(1, round(len(bugged) * p["visible_bug_tests"]))
    visible_bugs = sorted(rng.sample(bugged, n_visible))
    ref_lib, bug_lib = _exec(_module(funcs, {})), _exec(_module(funcs, bugs))
    for _ in range(100):
        cases = {f: _visible_case(rng, f) for f in funcs if f not in bugs or f in visible_bugs}
        vis_checks = [c for _, _, c in cases.values()]
        vis_ref, vis_bug = run_checks(ref_lib, vis_checks), run_checks(bug_lib, vis_checks)
        if all(vis_ref.values()) and all(not vis_bug[f"visible_{f}"] for f in visible_bugs):
            break
    else:
        raise Reject("no visible tests expose the visible bugs")
    for _ in range(100):
        checks = _checks(rng, funcs)
        ok_ref, res_bug = run_checks(ref_lib, checks), run_checks(bug_lib, checks)
        if all(ok_ref.values()) and all(any(not res_bug[c["name"]] for c in checks if c["func"] == f) for f in bugs):
            break
    else:
        raise Reject("no hidden checks catch every bug")
    tests = "\n\n".join(f"def {name}():\n    {line}\n" for name, line, _ in cases.values())
    test_stats = '"""Run with: python3 test_stats.py"""\n\n' + f"from stats import {', '.join(funcs)}  # noqa: F401\n\n\n{tests}" + TEST_RUNNER
    fixed = _module(funcs, {})
    shortcuts = {}
    hidden_only = {f: b for f, b in bugs.items() if f not in visible_bugs}
    if hidden_only:
        partial = _module(funcs, hidden_only)
        shortcuts["visible-bugs-only"] = Solution(_write(partial), lambda f, s=partial: {"/app/stats.py": s})
    names = ", ".join(f"`{f}`" for f in funcs)
    return TaskSpec(
        instruction=(
            f"The small statistics library in `/app/stats.py` ({names}) has bugs. Some of the tests in `/app/test_stats.py` may fail, "
            "and some bugs may not be covered by those tests at all.\n\n"
            "Fix `stats.py` so that all functions behave as their docstrings describe. You may add tests, but do not change what the "
            "existing tests assert. Only use the Python standard library.\n"
        ),
        files={"stats.py": _module(funcs, bugs), "test_stats.py": test_stats},
        grader=Checks("/app/stats.py", "stats", tuple(checks)),
        oracle=Solution(_write(fixed, "cd /app && python3 test_stats.py\n"), lambda f: {"/app/stats.py": fixed}),
        shortcuts=shortcuts,
        params={"functions": funcs, "bugs": bugs, "visible_bug_tests": visible_bugs, "n_checks": len(checks)},
    )


FAMILY = Family(
    name="fix-stats",
    version=3,
    cluster="fix-code",
    category="debugging",
    skills=("python", "debugging", "reading-docs"),
    difficulties={
        "easy": {"functions": ["mean", "median", "stdev", "percentile"], "n_bugs": 1, "visible_bug_tests": 1.0},
        "medium": {"functions": ["mean", "median", "variance", "stdev", "percentile"], "n_bugs": 2, "visible_bug_tests": 0.5},
        "hard": {"functions": ["mean", "median", "variance", "stdev", "percentile", "mode"], "n_bugs": 4, "visible_bug_tests": 0.25},
    },
    build=build,
)
