"""fix-pagination: fix the off-by-one and boundary bugs injected into pagination helpers
(partial credit).

`/app/paging.py` holds 4 / 5 / 6 small functions (page count, slice bounds, the page of an
item, an "Items 11-20 of 45" label; medium adds a window of page links, hard a page dict),
and 1 / 2 / 4 of them carry a bug from a catalog of classic pagination mistakes. The visible
tests in `/app/test_paging.py` expose all / half / a quarter of the bugs; a function with a
bug no visible test exposes gets no visible test. Hidden checks use fresh seed-specific
inputs and always include the boundaries: an exactly full and a partly filled last page, zero
items, the first page, an item index on a page boundary and link windows at either end.

Traps, each declared as a shortcut that must fail:
- `visible-bugs-only` (when some bug has no visible test): fix only the bugs the visible
  tests expose;
- `hardcode-visible`: return the visible tests' expected values for their exact inputs and
  keep the buggy code for everything else;
- `wrong-fix-<function>`: every bug fixed correctly except one, which gets a plausible wrong
  fix from the catalog (e.g. `total // per_page + 1` for a floor-divided page count, wrong for
  an exactly full last page and for zero items).

Expected values come from reference implementations written independently of the module's
source, and `build` checks that the correct module passes every hidden check. `build`
redraws a visible test until it exposes its function's bug (or, for a function without a
visible bug, passes with the hidden-only bugs still in place).
"""

import json

from learning_loop.tasks.runtime.grade import run_checks
from learning_loop.tasks.spec import Checks, Family, Reject, Solution, TaskSpec

MODULE = "paging"
PATH = f"/app/{MODULE}.py"

HEADER = '''"""Pagination helpers for list endpoints.

Pages are numbered from 1 and item indices from 0. The functions never modify their
arguments.
"""

import math
'''

CORRECT = {
    "page_count": '''def page_count(total: int, per_page: int) -> int:
    """Number of pages needed to show `total` items, `per_page` per page (the last page may be
    partly filled); 0 when there are no items. Raises ValueError if total < 0 or per_page < 1."""
    if total < 0 or per_page < 1:
        raise ValueError("need total >= 0 and per_page >= 1")
    return (total + per_page - 1) // per_page
''',
    "page_bounds": '''def page_bounds(page: int, per_page: int, total: int) -> list[int]:
    """[start, stop] such that items[start:stop] is page `page` of `total` items, `per_page`
    per page; the last page holds whatever remains. Raises ValueError unless
    1 <= page <= page_count(total, per_page)."""
    if not 1 <= page <= page_count(total, per_page):
        raise ValueError(f"page {page} is out of range")
    start = (page - 1) * per_page
    return [start, min(start + per_page, total)]
''',
    "page_of": '''def page_of(index: int, per_page: int) -> int:
    """The page that shows the item at 0-based `index`. Raises ValueError if index < 0 or
    per_page < 1."""
    if index < 0 or per_page < 1:
        raise ValueError("need index >= 0 and per_page >= 1")
    return index // per_page + 1
''',
    "page_label": '''def page_label(page: int, per_page: int, total: int) -> str:
    """The caption under a page: the 1-based numbers of its first and last item, inclusive,
    e.g. "Items 11-20 of 45". Returns "No items" when total is 0, whatever the page;
    otherwise raises ValueError unless 1 <= page <= page_count(total, per_page)."""
    if total == 0:
        return "No items"
    start, stop = page_bounds(page, per_page, total)
    return f"Items {start + 1}-{stop} of {total}"
''',
    "page_window": '''def page_window(current: int, last: int, width: int) -> list[int]:
    """The page numbers to show as links for page `current` of pages 1..last: `width`
    consecutive pages, or all pages when there are no more than `width`. The window puts
    `current` as close to its middle as possible (with an even width, one more page after
    current than before it) and is shifted as needed to stay within 1..last. Raises
    ValueError unless 1 <= current <= last and width >= 1."""
    if not 1 <= current <= last or width < 1:
        raise ValueError("need 1 <= current <= last and width >= 1")
    if last <= width:
        return list(range(1, last + 1))
    start = current - (width - 1) // 2
    start = max(1, min(start, last - width + 1))
    return list(range(start, start + width))
''',
    "paginate": '''def paginate(items: list, page: int, per_page: int) -> dict:
    """Page `page` of `items`: {"items": the items on that page (a new list), "page": page,
    "pages": page_count(len(items), per_page), "has_prev": whether an earlier page exists,
    "has_next": whether a later page exists}. Raises ValueError unless
    1 <= page <= pages, so an empty list has no valid page."""
    start, stop = page_bounds(page, per_page, len(items))
    pages = page_count(len(items), per_page)
    return {"items": items[start:stop], "page": page, "pages": pages, "has_prev": page > 1, "has_next": page < pages}
''',
}

# func -> bug -> {"bug": swaps applied to CORRECT[func], "naive": swaps for a plausible wrong fix, or None}
BUGS = {
    "page_count": {
        "floor": {
            "bug": [("return (total + per_page - 1) // per_page", "return total // per_page")],
            "naive": [("return (total + per_page - 1) // per_page", "return total // per_page + 1")],
        },
        "zero_is_one_page": {
            "bug": [("return (total + per_page - 1) // per_page", "return max(1, (total + per_page - 1) // per_page)")],
            "naive": None,
        },
    },
    "page_bounds": {
        "no_clamp": {
            "bug": [("return [start, min(start + per_page, total)]", "return [start, start + per_page]")],
            "naive": [("return [start, min(start + per_page, total)]", "return [start, min(start + per_page, total - 1)]")],
        },
        "last_page_rejected": {
            "bug": [("if not 1 <= page <= page_count(total, per_page):", "if not 1 <= page < page_count(total, per_page):")],
            "naive": [("if not 1 <= page <= page_count(total, per_page):", "if not 1 <= page <= total // per_page:")],
        },
    },
    "page_of": {
        "ceil": {
            "bug": [("return index // per_page + 1", "return math.ceil(index / per_page)")],
            "naive": [("return index // per_page + 1", "return max(1, math.ceil(index / per_page))")],
        },
        "one_based_index": {
            "bug": [("return index // per_page + 1", "return (index - 1) // per_page + 1")],
            "naive": [("return index // per_page + 1", "return max(1, (index - 1) // per_page + 1)")],
        },
    },
    "page_label": {
        "zero_based_start": {
            "bug": [('return f"Items {start + 1}-{stop} of {total}"', 'return f"Items {start}-{stop} of {total}"')],
            "naive": [('return f"Items {start + 1}-{stop} of {total}"', 'return f"Items {start + 1}-{stop + 1} of {total}"')],
        },
        "no_empty_case": {
            "bug": [('    if total == 0:\n        return "No items"\n', "")],
            "naive": None,
        },
    },
    "page_window": {
        "no_right_clamp": {
            "bug": [("start = max(1, min(start, last - width + 1))", "start = max(1, start)")],
            "naive": [
                ("start = max(1, min(start, last - width + 1))", "start = max(1, start)"),
                ("return list(range(start, start + width))", "return [p for p in range(start, start + width) if p <= last]"),
            ],
        },
        "even_width": {
            "bug": [("start = current - (width - 1) // 2", "start = current - width // 2")],
            "naive": None,
        },
        "few_pages": {
            "bug": [("    if last <= width:\n        return list(range(1, last + 1))\n", "")],
            "naive": None,
        },
    },
    "paginate": {
        "has_next_on_last": {
            "bug": [('"has_next": page < pages}', '"has_next": page <= pages}')],
            "naive": [('"has_next": page < pages}', '"has_next": stop - start == per_page}')],
        },
        "has_prev_on_first": {
            "bug": [('"has_prev": page > 1,', '"has_prev": page > 0,')],
            "naive": None,
        },
    },
}


# --------------------------------------------------------------------------- #
# Reference implementations (independent of CORRECT) and the checks
# --------------------------------------------------------------------------- #


def _count(total, per_page):
    return -(-total // per_page)


def _bounds(page, per_page, total):
    return [(page - 1) * per_page, min(page * per_page, total)]


def _window(current, last, width):
    if last <= width:
        return list(range(1, last + 1))
    before = (width - 1) // 2
    lo = min(max(1, current - before), last - width + 1)
    return list(range(lo, lo + width))


def _eq(name, func, args, expected):
    return {"name": name, "func": func, "kind": "equal", "args": args, "expected": expected}


def _raises(name, func, args):
    return {"name": name, "func": func, "kind": "raises", "args": args}


def _cases(rng, func):
    """The hidden checks for `func`, with fresh inputs; the visible tests draw from the same kinds."""
    pp = rng.randint(4, 12)
    full = rng.randint(2, 6)  # number of full pages
    partial_total = full * pp + rng.randint(1, pp - 1)  # last page partly filled
    exact_total = rng.randint(2, 6) * pp  # last page exactly full
    if func == "page_count":
        return [
            _eq("page_count_partial", func, [partial_total, pp], _count(partial_total, pp)),
            _eq("page_count_exact", func, [exact_total, pp], _count(exact_total, pp)),
            _eq("page_count_zero", func, [0, pp], 0),
            _raises("page_count_zero_per_page", func, [partial_total, 0]),
            _raises("page_count_negative_total", func, [-rng.randint(1, 9), pp]),
        ]
    if func == "page_bounds":
        last_p, last_e = _count(partial_total, pp), _count(exact_total, pp)
        mid = rng.randint(2, last_p - 1)
        return [
            _eq("page_bounds_first", func, [1, pp, partial_total], _bounds(1, pp, partial_total)),
            _eq("page_bounds_middle", func, [mid, pp, partial_total], _bounds(mid, pp, partial_total)),
            _eq("page_bounds_last_partial", func, [last_p, pp, partial_total], _bounds(last_p, pp, partial_total)),
            _eq("page_bounds_last_exact", func, [last_e, pp, exact_total], _bounds(last_e, pp, exact_total)),
            _raises("page_bounds_past_end", func, [last_p + 1, pp, partial_total]),
            _raises("page_bounds_page_zero", func, [0, pp, partial_total]),
        ]
    if func == "page_of":
        k = rng.randint(1, 6)
        inside = k * pp + rng.randint(1, pp - 1)
        return [
            _eq("page_of_first_item", func, [0, pp], 1),
            _eq("page_of_boundary", func, [k * pp, pp], k + 1),
            _eq("page_of_last_on_page", func, [k * pp - 1, pp], k),
            _eq("page_of_inside", func, [inside, pp], inside // pp + 1),
            _raises("page_of_negative", func, [-rng.randint(1, 9), pp]),
        ]
    if func == "page_label":
        last_p = _count(partial_total, pp)
        mid = rng.randint(1, last_p - 1)
        s, e = _bounds(mid, pp, partial_total)
        ls, le = _bounds(last_p, pp, partial_total)
        return [
            _eq("page_label_full_page", func, [mid, pp, partial_total], f"Items {s + 1}-{e} of {partial_total}"),
            _eq("page_label_last_partial", func, [last_p, pp, partial_total], f"Items {ls + 1}-{le} of {partial_total}"),
            _eq("page_label_no_items", func, [1, pp, 0], "No items"),
            _raises("page_label_past_end", func, [last_p + 1, pp, partial_total]),
        ]
    if func == "page_window":
        width_odd, width_even = rng.choice([5, 7]), rng.choice([4, 6])
        last = rng.randint(16, 30)
        mid = rng.randint(width_odd + 1, last - width_odd)
        mid_even = rng.randint(width_even + 1, last - width_even)
        end = last - rng.randint(0, 1)
        start = 1 + rng.randint(0, 1)
        few = rng.randint(2, width_odd - 1)
        return [
            _eq("page_window_middle_odd", func, [mid, last, width_odd], _window(mid, last, width_odd)),
            _eq("page_window_middle_even", func, [mid_even, last, width_even], _window(mid_even, last, width_even)),
            _eq("page_window_near_start", func, [start, last, width_odd], _window(start, last, width_odd)),
            _eq("page_window_near_end", func, [end, last, width_odd], _window(end, last, width_odd)),
            _eq("page_window_few_pages", func, [rng.randint(1, few), few, width_odd], list(range(1, few + 1))),
            _raises("page_window_current_past_last", func, [last + 1, last, width_odd]),
        ]
    if func == "paginate":
        base = rng.randint(100, 900)
        items_p = [f"row-{base + i}" for i in range(partial_total)]
        items_e = [f"row-{base + i}" for i in range(exact_total)]

        def page(items, p):
            n = _count(len(items), pp)
            s, e = _bounds(p, pp, len(items))
            return {"items": items[s:e], "page": p, "pages": n, "has_prev": p > 1, "has_next": p < n}

        mid = rng.randint(2, _count(partial_total, pp) - 1)
        return [
            _eq("paginate_first", func, [items_p, 1, pp], page(items_p, 1)),
            _eq("paginate_middle", func, [items_p, mid, pp], page(items_p, mid)),
            _eq("paginate_last_partial", func, [items_p, _count(partial_total, pp), pp], page(items_p, _count(partial_total, pp))),
            _eq("paginate_last_exact", func, [items_e, _count(exact_total, pp), pp], page(items_e, _count(exact_total, pp))),
            _raises("paginate_empty", func, [[], 1, pp]),
            {"name": "paginate_no_mutation", "func": func, "kind": "no_mutation", "args": [items_p, mid, pp]},
        ]
    raise KeyError(func)


# --------------------------------------------------------------------------- #
# The fix-code engine: buggy module, visible tests, shortcuts
# --------------------------------------------------------------------------- #


def _apply(func, swaps):
    src = CORRECT[func]
    for old, new in swaps:
        assert src.count(old) == 1, (func, old)
        src = src.replace(old, new)
    return src


def _module(funcs, chosen):
    """The module with `chosen` {func: source} in place of the correct functions."""
    return HEADER + "".join("\n\n" + chosen.get(f, CORRECT[f]) for f in funcs)


def _exec(src):
    ns = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 - our own reference/buggy sources
    return ns


def _canonical(c):
    """`c` with its arguments as the verifier sees them: key.json is written with sorted keys,
    so dict arguments arrive in sorted key order."""
    return dict(c, args=json.loads(json.dumps(c["args"], sort_keys=True)))


def _passes(lib, checks):
    return all(run_checks(lib, checks).values())


def _test_function(c):
    call = f"{c['func']}({', '.join(repr(a) for a in c['args'])})"
    if c["kind"] == "equal":
        body = f"    got = {call}\n    assert got == {c['expected']!r}, f\"got {{got!r}}\"\n"
    elif c["kind"] == "raises":
        exc = c.get("exception", "ValueError")
        body = f"    try:\n        {call}\n    except {exc}:\n        return\n    raise AssertionError(\"expected {exc}\")\n"
    else:  # no_mutation
        rest = "".join(", " + repr(a) for a in c["args"][1:])
        body = f"    arg = {c['args'][0]!r}\n    {c['func']}(arg{rest})\n    assert arg == {c['args'][0]!r}, \"the argument was modified\"\n"
    return f"def test_{c['func']}():\n{body}"


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


def _hardcoded(src, visible):
    """`src` plus wrappers that answer the given visible tests from their expectations."""
    for c in visible:
        f = c["func"]
        if c["kind"] == "equal":
            then = f"return {c['expected']!r}"
        elif c["kind"] == "raises":
            then = f"raise {c.get('exception', 'ValueError')}(\"invalid input\")"
        else:
            continue
        src += f"\n\n_{f}_original = {f}\n\n\ndef {f}(*args):\n    if list(args) == {c['args']!r}:\n        {then}\n    return _{f}_original(*args)\n"
    return src


def _write(source, then=""):
    return f"cat > {PATH} <<'PY'\n{source}PY\n{then}"


def _solution(source, then=""):
    return Solution(_write(source, then), lambda files, s=source: {PATH: s})


def build_fix(ctx, instruction, extra_imports=()):
    rng, p = ctx.rng, ctx.params
    funcs = p["functions"]
    bugged = sorted(rng.sample(funcs, p["n_bugs"]))
    bugs = {f: rng.choice(sorted(BUGS[f])) for f in bugged}
    if not any(BUGS[f][b]["naive"] for f, b in bugs.items()):
        raise Reject("no injected bug has a catalogued wrong fix")
    visible_bugs = sorted(rng.sample(bugged, max(1, round(len(bugged) * p["visible_bug_tests"]))))
    bug_src = {f: _apply(f, BUGS[f][b]["bug"]) for f, b in bugs.items()}
    hidden_only = {f: s for f, s in bug_src.items() if f not in visible_bugs}
    ref, buggy, partial = _exec(_module(funcs, {})), _exec(_module(funcs, bug_src)), _exec(_module(funcs, hidden_only))

    visible = []
    for f in funcs:
        if f in bugs and f not in visible_bugs:
            continue  # a hidden bug gets no visible test
        for _ in range(100):
            c = _canonical(dict(rng.choice(_cases(rng, f)), name=f"visible_{f}"))
            if _passes(ref, [c]) and _passes(partial, [c]) and (f not in visible_bugs or not _passes(buggy, [c])):
                visible.append(c)
                break
        else:
            raise Reject(f"no visible test for {f}")

    seen = [(c["func"], c["args"]) for c in visible]
    for _ in range(100):
        checks = [_canonical(c) for f in funcs for c in _cases(rng, f)]
        if not any((c["func"], c["args"]) in seen for c in checks):
            break  # hidden inputs differ from every visible test's, so hard-coding those cannot pass
    else:
        raise Reject("hidden checks keep repeating a visible test")
    failed = [c["name"] for c in checks if not _passes(ref, [c])]
    if failed:
        raise RuntimeError(f"{ctx.family}: the correct module fails hidden checks {failed}")

    names = ", ".join(funcs + list(extra_imports))
    tests = "\n\n".join(_test_function(c) for c in visible)
    test_src = f'"""Run with: python3 test_{MODULE}.py"""\n\nfrom {MODULE} import {names}  # noqa: F401\n\n\n{tests}' + TEST_RUNNER
    fixed = _module(funcs, {})
    shortcuts = {"hardcode-visible": _solution(_hardcoded(_module(funcs, bug_src), [c for c in visible if c["func"] in visible_bugs]))}
    if hidden_only:
        shortcuts["visible-bugs-only"] = _solution(_module(funcs, hidden_only))
    for f, b in bugs.items():
        if BUGS[f][b]["naive"]:
            shortcuts[f"wrong-fix-{f}"] = _solution(_module(funcs, {f: _apply(f, BUGS[f][b]["naive"])}))
    listed = ", ".join(f"`{f}`" for f in funcs)
    return TaskSpec(
        instruction=instruction.format(functions=listed, hint=p["hint"]),
        files={f"{MODULE}.py": _module(funcs, bug_src), f"test_{MODULE}.py": test_src},
        grader=Checks(PATH, MODULE, tuple(checks)),
        oracle=_solution(fixed, f"cd /app && python3 test_{MODULE}.py\n"),
        shortcuts=shortcuts,
        params={"functions": funcs, "bugs": bugs, "visible_bug_tests": visible_bugs, "n_checks": len(checks)},
    )


INSTRUCTION = """The pagination helpers in `/app/paging.py` ({functions}) are used by our list endpoints, and users report wrong page counts, missing items and broken page links. {hint}

Fix `paging.py` so that every function behaves exactly as its docstring describes. Keep the function names and signatures, and use only the Python standard library. You may add tests to `/app/test_paging.py` (run it with `python3 test_paging.py`), but do not change what the existing tests assert.
"""

HINTS = {
    "easy": "One function has a bug, and a test in `/app/test_paging.py` fails because of it.",
    "medium": "Some tests in `/app/test_paging.py` fail, and some bugs may not be covered by those tests at all.",
    "hard": "Several functions have bugs; the tests in `/app/test_paging.py` catch only a few of them.",
}


def build(ctx):
    return build_fix(ctx, INSTRUCTION)


FAMILY = Family(
    name="fix-pagination",
    version=1,
    cluster="fix-code",
    category="debugging",
    skills=("python", "debugging", "off-by-one", "reading-docs"),
    difficulties={
        "easy": {"functions": ["page_count", "page_bounds", "page_of", "page_label"], "n_bugs": 1, "visible_bug_tests": 1.0, "hint": HINTS["easy"]},
        "medium": {"functions": ["page_count", "page_bounds", "page_of", "page_label", "page_window"], "n_bugs": 2, "visible_bug_tests": 0.5, "hint": HINTS["medium"]},
        "hard": {
            "functions": ["page_count", "page_bounds", "page_of", "page_label", "page_window", "paginate"],
            "n_bugs": 4,
            "visible_bug_tests": 0.25,
            "hint": HINTS["hard"],
        },
    },
    build=build,
)
