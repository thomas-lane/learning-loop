"""fix-stats: fix the bugs injected into a small statistics library (partial credit).

`/app/stats.py` holds 4 / 5 / 6 functions (mean, median, sample standard deviation,
percentile; medium adds sample variance, hard the mode), and 1 / 2 / 4 of them carry a bug
drawn uniformly from that function's catalog (floor division, a skipped element, the upper
middle or a sort in place for the median, population instead of sample variance, a missing
square root, percentile ranks off by one, uninterpolated or unsorted, the wrong tie-break for
the mode). The instruction is the same at every difficulty apart from the function list:
some visible tests fail, and they may not cover every bug.

`/app/test_stats.py` holds two visible tests per function, of different kinds and at least
one of them not an error check, so neither the tested functions nor their number of tests
say where the bugs are. All / half / a quarter of the bugs (at least one) are *visible*: one
test of that function fails because of its bug, both with that bug alone and with every bug
in place. Every other function's tests pass on the library as given; for a function with a
*hidden* bug they exercise behavior the bug does not touch (on hard, most bugs are only
described by docstrings). Every visible test passes once only the hidden bugs are left.
Hidden checks use fresh seed-specific inputs (never a visible test's, except the empty list
for `mean`, whose input is always the same), and `build` redraws them until the reference
library passes all of them and every injected bug fails at least one.

Trap, declared as a shortcut that must fail: `visible-bugs-only` (when some bug is hidden)
fixes only the bugs the visible tests expose.
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


# Kinds of check whose input is the same in every instance, so a visible test and a hidden
# check of that kind may share it (hard-coding it gives the specified answer anyway).
FIXED_INPUTS = ("mean_empty_raises",)


def _cases(rng, func: str):
    """The hidden checks for `func`, with fresh seed-specific inputs; the visible tests draw
    from the same kinds."""

    def ints(n: int, lo: int = 1, hi: int = 60) -> list[int]:
        return [rng.randint(lo, hi) for _ in range(n)]

    def value(name: str, args) -> dict:
        return {"name": name, "func": func, "kind": "value", "args": args, "expected": _reference(func, args)}

    def raises(name: str, args) -> dict:
        return {"name": name, "func": func, "kind": "raises", "args": args}

    if func == "mean":
        xs = ints(5)
        while sum(xs) % len(xs) == 0:
            xs = ints(5)
        whole = ints(4)
        whole.append(5 * rng.randint(1, 11) + (-sum(whole)) % 5)  # the mean is a whole number
        return [value("mean", [xs]), value("mean_whole", [whole]), value("mean_from_zero", [[0] + ints(4)]), raises("mean_empty_raises", [[]])]
    if func == "median":
        return [
            value("median_odd", [ints(7)]),
            value("median_even", [ints(6)]),
            {"name": "median_no_mutation", "func": func, "kind": "no_mutation", "args": [ints(5)]},
        ]
    if func in ("variance", "stdev"):
        return [
            value(f"{func}_sample", [ints(rng.randint(6, 9))]),
            value(f"{func}_constant", [[rng.randint(1, 60)] * rng.randint(3, 6)]),
            raises(f"{func}_single_raises", [[rng.randint(1, 9)]]),
        ]
    if func == "percentile":
        xs = sorted(set(ints(6, 1, 99)))
        while len(xs) < 4:
            xs = sorted(set(ints(6, 1, 99)))
        rng.shuffle(xs)  # the function must sort its input
        return [
            value("percentile_0", [xs, 0]),
            value("percentile_100", [xs, 100]),
            value("percentile_mid", [xs, rng.choice([30, 40, 60, 70, 90])]),
            value("percentile_one_value", [[rng.randint(1, 99)], rng.choice([10, 25, 50, 75, 90])]),
        ]
    if func == "mode":
        a, b = sorted(rng.sample(range(1, 50), 2))
        xs = [b, b, a, a] + ints(3, 51, 99)  # tie: larger value appears first
        win, *others = rng.sample(range(1, 60), 4)
        clear = [others[0], win, others[1], win, others[2], win]  # one most common value, not first
        return [
            value("mode_tie_smallest", [xs]),
            value("mode_simple", [[rng.randint(1, 9)] * 3 + ints(3, 10, 30)]),
            value("mode_clear_winner", [clear]),
        ]
    raise KeyError(func)


def _exec(src):
    ns = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 - our own reference/buggy sources
    return ns


def _passes(lib, checks) -> bool:
    return all(run_checks(lib, checks).values())


def _test_function(c) -> str:
    call = f"{c['func']}({', '.join(repr(a) for a in c['args'])})"
    if c["kind"] == "value":
        exp = c["expected"]
        body = f"    got = {call}\n    assert abs(got - {exp!r}) <= 1e-9 * max(1.0, abs({exp!r})), f\"got {{got!r}}\"\n"
    elif c["kind"] == "raises":
        body = f"    try:\n        {call}\n    except ValueError:\n        return\n    raise AssertionError(\"expected ValueError\")\n"
    else:  # no_mutation
        body = f"    arg = {c['args'][0]!r}\n    {c['func']}(arg)\n    assert arg == {c['args'][0]!r}, \"the argument was modified\"\n"
    return f"def test_{c['name'][len('visible_'):]}():\n{body}"


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

VISIBLE_PER_FUNCTION = 2  # visible tests per function, each of a different kind
DRAWS_PER_KIND = 20


def _visible_tests(rng, f, ok, exposing):
    """VISIBLE_PER_FUNCTION visible tests for `f`, each of a different kind of `_cases`, at
    least one of them not a `raises` check, drawn with fresh inputs until `ok(check,
    exposing)` holds; with `exposing`, the first must expose `f`'s bug. Every function gets
    the same number and mix, so a function whose bug is hidden (whose tests must avoid it)
    does not stand out. Returns fewer when that is impossible. The tests are in the order
    `_cases` lists their kinds, so the failing one is not always first."""
    first = _cases(rng, f)
    order, kind_of = [c["name"] for c in first], {c["name"]: c["kind"] for c in first}
    kinds = list(order)
    rng.shuffle(kinds)

    def draw(kind, expose):
        for _ in range(DRAWS_PER_KIND):
            c = dict(next(c for c in _cases(rng, f) if c["name"] == kind), name=f"visible_{kind}")
            if ok(c, expose):
                return c
        return None

    chosen = []
    while len(chosen) < VISIBLE_PER_FUNCTION:
        expose = exposing and not chosen
        need_value = not expose and all(c["kind"] == "raises" for c in chosen) and len(chosen) == VISIBLE_PER_FUNCTION - 1
        for kind in kinds:
            if need_value and kind_of[kind] == "raises":
                continue
            if c := draw(kind, expose):
                chosen.append(c)
                kinds.remove(kind)
                break
        else:
            break
    return sorted(chosen, key=lambda c: order.index(c["name"][len("visible_") :]))


def _write(source, then=""):
    return "cat > /app/stats.py <<'PY'\n" + source + "PY\n" + then


def build(ctx):
    rng, p = ctx.rng, ctx.params
    funcs = p["functions"]
    bugged = sorted(rng.sample(funcs, p["n_bugs"]))
    bugs = {f: rng.choice(sorted(BUGGY[f])) for f in bugged}
    visible_bugs = sorted(rng.sample(bugged, max(1, round(len(bugged) * p["visible_bug_tests"]))))
    hidden_only = {f: b for f, b in bugs.items() if f not in visible_bugs}
    ref, buggy, partial = _exec(_module(funcs, {})), _exec(_module(funcs, bugs)), _exec(_module(funcs, hidden_only))

    # Every function gets visible tests, and all of them pass once only the hidden bugs are
    # left. A function with a visible bug gets one test that fails because of that bug (with
    # it alone and with every bug in place); every other function's tests pass on the library
    # as given, so they exercise behavior its hidden bug (if any) does not touch.
    visible = []
    for f in funcs:
        if f in visible_bugs:
            alone = _exec(_module(funcs, {f: bugs[f]}))

            def ok(c, exposing, alone=alone):
                if not (_passes(ref, [c]) and _passes(partial, [c])):
                    return False
                return not _passes(alone, [c]) and not _passes(buggy, [c]) if exposing else True

        else:

            def ok(c, exposing):
                return _passes(ref, [c]) and _passes(partial, [c]) and _passes(buggy, [c])

        tests = _visible_tests(rng, f, ok, exposing=f in visible_bugs)
        if len(tests) < VISIBLE_PER_FUNCTION:
            raise Reject(f"too few visible tests for {f}")
        visible += tests

    seen = [(c["func"], c["args"]) for c in visible if c["name"][len("visible_") :] not in FIXED_INPUTS]
    for _ in range(100):
        checks = [c for f in funcs for c in _cases(rng, f)]
        if any((c["func"], c["args"]) in seen for c in checks):
            continue  # hidden inputs differ from every visible test's, so hard-coding those cannot pass
        res_bug = run_checks(buggy, checks)
        if _passes(ref, checks) and all(any(not res_bug[c["name"]] for c in checks if c["func"] == f) for f in bugs):
            break
    else:
        raise Reject("no hidden checks with fresh inputs catch every bug")
    tests = "\n\n".join(_test_function(c) for c in visible)
    test_stats = '"""Run with: python3 test_stats.py"""\n\n' + f"from stats import {', '.join(funcs)}  # noqa: F401\n\n\n{tests}" + TEST_RUNNER
    fixed = _module(funcs, {})
    shortcuts = {}
    if hidden_only:
        partial_src = _module(funcs, hidden_only)
        shortcuts["visible-bugs-only"] = Solution(_write(partial_src), lambda f, s=partial_src: {"/app/stats.py": s})
    names = ", ".join(f"`{f}`" for f in funcs)
    return TaskSpec(
        instruction=INSTRUCTION.format(functions=names),
        files={"stats.py": _module(funcs, bugs), "test_stats.py": test_stats},
        grader=Checks("/app/stats.py", "stats", tuple(checks)),
        oracle=Solution(_write(fixed, "cd /app && python3 test_stats.py\n"), lambda f: {"/app/stats.py": fixed}),
        shortcuts=shortcuts,
        params={"functions": funcs, "bugs": bugs, "visible_bug_tests": visible_bugs, "n_checks": len(checks)},
    )


INSTRUCTION = """The small statistics library in `/app/stats.py` ({functions}) has bugs. Some of the tests in `/app/test_stats.py` fail, and those tests may not cover every bug.

Fix `stats.py` so that all functions behave as their docstrings describe. You may add tests, but do not change what the existing tests assert. Only use the Python standard library.
"""


FAMILY = Family(
    name="fix-stats",
    version=4,
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
