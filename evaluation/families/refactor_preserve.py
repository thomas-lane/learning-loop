"""refactor-preserve: extract a duplicated parser into one helper, keep behavior, then extend it.

/app/<module>.py has two or three public functions that each contain their own copy of the
code that parses a string (a duration such as "1h30m" in `timesheet.py`, or a size such as
"4kb" in `quota.py`). The instruction asks for a preparatory refactoring: add a helper with a
given name and signature that parses one string exactly as the copies do, make every existing
function call it while keeping its behavior (results, tie-breaking, which inputs raise or are
skipped), then extend the helper with a new input form (a new unit, or spaces inside the
string), so every function accepts the new form.

Why behavior can see the structure: the `Checks` grader can only call functions, so the
requirement "every function uses the helper" is made observable through the extension.
Hidden checks call the helper (old forms with every legacy rule, new forms, invalid strings)
and every existing function on inputs that contain the new form. A function that kept its own
copy rejects the new form, so the shortcuts `helper-unused` (helper added, callers unchanged)
and `misses-a-caller` (the last function keeps its copy) fail; doing nothing fails because the
helper is missing. What behavior cannot see is a function that re-implements the extended
grammar in its own copy instead of calling the helper; that takes more work than the refactor
and is not declared as a shortcut.

Other traps (declared shortcuts, each failing a planted check): `no-extension` (pure
refactor), `hardcode-examples` (the helper special-cases the visible tests' new-form examples),
`case-sensitive` and `drops-legacy-form` (a helper rewritten from scratch that loses a legacy
rule from the docstrings: upper case, bare numbers of minutes, or the optional "b" of a size),
`ties-last` (rewriting `longest`/`largest` with `>=`, so ties return the last entry). Hard adds
a third function that skips unparsable entries instead of raising (`skip-becomes-raise`) and
asks for two extensions at once (`one-extension-only`).

Difficulty: easy and medium have two functions and one extension; easy's visible tests show
two new-form examples and every legacy rule, medium's one example and fewer legacy cases;
hard has three functions, two extensions and one new-form example.
"""

import re
import textwrap

from learning_loop.tasks.runtime.grade import run_checks
from learning_loop.tasks.spec import Checks, Family, Reject, Solution, TaskSpec

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

INSTRUCTION = """`/app/{module}` has {n} functions ({names}) that each contain their own copy of the code that parses a {noun} string. Refactor the module:

1. Add a function `{helper}(text)` that parses one {noun} string exactly the way the existing functions do (the module and function docstrings describe the format) and returns the number of {unit}. It raises `ValueError` for a string that is not a {noun}.
2. Make every one of the existing functions call `{helper}` instead of using its own copy of the parsing code. Keep their behavior otherwise unchanged: the same results, the same tie-breaking, and the same handling of entries that are not {noun}s.
3. Then extend `{helper}` so that it also accepts {extensions}Because every function goes through `{helper}`, they all accept the new form{s} too.

`/app/{tests}` has some tests (run `python3 {tests}` in `/app`); keep them passing. Use only the Python standard library.
"""

# --------------------------------------------------------------------------- #
# Domains: how a string is parsed, before and after the extension
# --------------------------------------------------------------------------- #


class Durations:
    module, helper, noun, unit, var, error = "timesheet.py", "parse_minutes", "duration", "minutes", "minutes", "not a duration"
    funcs = ("total_minutes", "longest", "over_limit")
    items = "entries"
    legacy_form = "bare numbers of minutes"
    extensions = {
        "days": 'a number of days with the unit `d`, written before the hours (1d is 24h, i.e. 1440 minutes), such as "2d", "1d4h" or "3d30m"',
        "spaces": 'spaces between the parts of a duration, such as "1h 30m" (one or more spaces between any two parts)',
    }
    doc = (
        '"""Timesheet helpers.\n\n'
        'A duration is a string of hours and/or minutes such as "1h30m", "2h" or "45m", in upper or\n'
        'lower case, or a bare number of minutes such as "90". Spaces around it are ignored.\n"""\n\nimport re\n'
    )

    @staticmethod
    def parse_lines(v, r, exts, case=True, legacy=True):
        units = ([("d", 1440)] if "days" in exts else []) + [("h", 60), ("m", 1)]
        regex = (" *" if "spaces" in exts else "").join(f"(?:([0-9]+){u})?" for u, _ in units)
        expr = " + ".join(f"int(match.group({i + 1}) or 0)" + (f" * {m}" if m != 1 else "") for i, (_, m) in enumerate(units))
        lines = [f"text = {v}.strip(){'.lower()' if case else ''}"]
        body = [f'match = re.fullmatch(r"{regex}", text)', "if not text or not match:", f'    raise ValueError("not a duration: %r" % {v})', f"{r} = {expr}"]
        if legacy:
            return lines + ["if text.isdigit():", f"    {r} = int(text)", "else:"] + ["    " + b for b in body]
        return lines + body


    @staticmethod
    def old_form(rng, value=None):
        """An old-format string (optionally for `value` minutes)."""
        v = rng.randint(1, 600) if value is None else value
        h, m = divmod(v, 60)
        forms = [str(v)] + ([f"{h}h{m}m"] if h and m else []) + ([f"{h}h"] if h and not m else []) + ([f"{v}m"])
        s = rng.choice(forms)
        if rng.random() < 0.3:
            s = s.upper()
        if rng.random() < 0.2:
            s = " " + s + " "
        return s

    @staticmethod
    def new_form(rng, ext):
        if ext == "days":
            d, h, m = rng.randint(1, 4), rng.randint(0, 9), rng.randint(0, 59)
            s = rng.choice([f"{d}d", f"{d}d{h + 1}h", f"{d}d{m + 1}m", f"{d}d{h + 1}h{m + 1}m"])
        else:
            h, m = rng.randint(1, 9), rng.randint(1, 59)
            s = f"{h}h" + " " * rng.randint(1, 2) + f"{m}m"
        return s.upper() if rng.random() < 0.25 else s

    @staticmethod
    def invalid(rng):
        return ["", "   ", "h", f"{rng.randint(2, 9)}x", "1.5h", f"{rng.randint(5, 50)}m{rng.randint(1, 5)}h", "-5m", "1hh"]

    tie_pairs = staticmethod(lambda rng: (lambda v: (str(v), f"{v // 60}h{v % 60}m"))(rng.randint(2, 9) * 60 + rng.randint(1, 59)))


class Sizes:
    module, helper, noun, unit, var, error = "quota.py", "parse_size", "size", "bytes", "size", "not a size"
    funcs = ("total_bytes", "largest", "over_quota")
    items = "sizes"
    legacy_form = 'the optional "b" after the unit'
    extensions = {
        "tera": 'the unit `t` for tebibytes (1t is 1024g), also with the optional `b`, such as "2t" or "1TB"',
        "spaces": 'spaces between the number and its unit, such as "4 kb", "3 G" or "512 b"',
    }
    doc = (
        '"""Storage quota helpers.\n\n'
        'A size is a string: a number of bytes such as "512", or a number followed by the unit k, m\n'
        'or g (powers of 1024: 1k is 1024 bytes), optionally followed by "b", such as "4k", "4kb" or\n'
        '"2GB", in upper or lower case. A bare "b" also means bytes ("512b"). Spaces around a size\n'
        'are ignored.\n"""\n\nimport re\n'
    )

    @staticmethod
    def _units(exts):
        return "kmgt" if "tera" in exts else "kmg"

    @staticmethod
    def parse_lines(v, r, exts, case=True, legacy=True):
        units = Sizes._units(exts)
        sp = " *" if "spaces" in exts else ""
        regex = f"([0-9]+){sp}([{units}]?)" + ("b?" if legacy else "")
        return [
            f"text = {v}.strip(){'.lower()' if case else ''}",
            f'match = re.fullmatch(r"{regex}", text)',
            "if not match:",
            f'    raise ValueError("not a size: %r" % {v})',
            f'{r} = int(match.group(1)) * 1024 ** "_{units}".index(match.group(2) or "_")',
        ]


    @staticmethod
    def old_form(rng, value=None):
        if value is None:
            n, u = rng.randint(1, 900), rng.choice(["", "k", "k", "m", "m", "g"])
        else:
            n, u = value, ""
        s = f"{n}{u}" + (rng.choice(["", "b"]) if u else rng.choice(["", "", "b"]))
        if rng.random() < 0.3:
            s = s.upper()
        if rng.random() < 0.2:
            s = " " + s + " "
        return s

    @staticmethod
    def new_form(rng, ext):
        if ext == "tera":
            s = f"{rng.randint(1, 9)}t" + rng.choice(["", "b"])
        else:
            s = f"{rng.randint(1, 900)}" + " " * rng.randint(1, 2) + rng.choice(["k", "kb", "m", "mb", "g", "b"])
        return s.upper() if rng.random() < 0.25 else s

    @staticmethod
    def invalid(rng):
        return ["", "  ", "k", f"{rng.randint(2, 9)}x", "1.5k", "4kbb", "-1k", "kb4"]

    tie_pairs = staticmethod(lambda rng: (lambda n: (f"{n}k", str(n * 1024)))(rng.randint(2, 900)))


DOMAINS = {"durations": Durations, "sizes": Sizes}

# --------------------------------------------------------------------------- #
# Module sources
# --------------------------------------------------------------------------- #

FUNC_DOCS = {
    "total": "The total of the {noun}s in `{items}`, in {unit}. Raises ValueError if an entry is not a {noun}.",
    "best": "The entry of `{items}` with the largest {noun}, exactly as given (the first such entry if several\n    tie), or None for an empty list. Raises ValueError if an entry is not a {noun}.",
    "over": "The entries of `{items}` larger than the {noun} `limit`, in their original order. Entries that\n    are not {noun}s are skipped; a `limit` that is not a {noun} raises ValueError.",
}


def _indent(lines, n):
    return "".join(" " * n + ln + "\n" for ln in lines)


def _parse_into(dom, mode, v, r, n, exts):
    """The lines that set `r` from the string `v` at indent `n`: a call or an inline copy."""
    if mode == "helper":
        return _indent([f"{r} = {dom.helper}({v})"], n)
    return _indent(dom.parse_lines(v, r, (), True, True), n)


def _total(dom, mode, exts):
    name, r, it = dom.funcs[0], dom.var, dom.items
    doc = FUNC_DOCS["total"].format(noun=dom.noun, items=it, unit=dom.unit)
    return f'def {name}({it}):\n    """{doc}"""\n    total = 0\n    for entry in {it}:\n' + _parse_into(dom, mode, "entry", r, 8, exts) + f"        total += {r}\n    return total\n"


def _best(dom, mode, exts, ties_last=False):
    name, r, it = dom.funcs[1], dom.var, dom.items
    doc = FUNC_DOCS["best"].format(noun=dom.noun, items=it)
    cmp = ">=" if ties_last else ">"
    return (
        f'def {name}({it}):\n    """{doc}\n    """\n    best, best_{r} = None, -1\n    for entry in {it}:\n'
        + _parse_into(dom, mode, "entry", r, 8, exts)
        + f"        if {r} {cmp} best_{r}:\n            best, best_{r} = entry, {r}\n    return best\n"
    )


def _over(dom, mode, exts, skip=True):
    name, r, it = dom.funcs[2], dom.var, dom.items
    doc = FUNC_DOCS["over"].format(noun=dom.noun, items=it)
    head = f'def {name}({it}, limit):\n    """{doc}\n    """\n' + _parse_into(dom, mode, "limit", f"limit_{r}", 4, exts) + f"    result = []\n    for entry in {it}:\n"
    if skip:
        loop = "        try:\n" + _parse_into(dom, mode, "entry", r, 12, exts) + "        except ValueError:\n            continue\n"
    else:
        loop = _parse_into(dom, mode, "entry", r, 8, exts)
    return head + loop + f"        if {r} > limit_{r}:\n            result.append(entry)\n    return result\n"


def _helper_doc(dom, exts):
    if dom is Durations:
        text = (
            'The number of minutes in the duration string `text`. A duration is hours and/or minutes such as "1h30m", "2h" or "45m"'
            + (', optionally preceded by days such as "1d4h"' if "days" in exts else "")
            + (', with spaces allowed between its parts ("1h 30m")' if "spaces" in exts else "")
            + ', in upper or lower case, or a bare number of minutes such as "90".'
        )
    else:
        text = (
            "The number of bytes in the size string `text`. A size is a number, optionally followed by the unit "
            + ("k, m, g or t" if "tera" in exts else "k, m or g")
            + ' (powers of 1024) and then an optional "b"'
            + (', with spaces allowed between the number and the unit ("4 kb")' if "spaces" in exts else "")
            + ", in upper or lower case."
        )
    text += " Spaces around it are ignored. Raises ValueError for anything else."
    return textwrap.fill(text, width=92, initial_indent="", subsequent_indent="    ")


def _helper(dom, exts, case=True, legacy=True, examples=None):
    lines = dom.parse_lines("text", dom.var, exts, case, legacy)
    src = f'def {dom.helper}(text):\n    """{_helper_doc(dom, exts)}\n    """\n'
    if examples:
        src += f"    known = {examples!r}\n    if text in known:\n        return known[text]\n"
    return src + _indent(lines, 4) + f"    return {dom.var}\n"


def _module(dom, n_funcs, exts, helper=None, callers=None, ties_last=False, skip=True):
    """The module: an optional helper source, then the functions, each calling the helper
    (`callers[i]` true) or keeping its own copy of the original parsing code."""
    callers = callers if callers is not None else [helper is not None] * n_funcs
    mode = ["helper" if c else "inline" for c in callers]
    parts = ([helper] if helper else []) + [_total(dom, mode[0], exts), _best(dom, mode[1], exts, ties_last)]
    if n_funcs == 3:
        parts.append(_over(dom, mode[2], exts, skip))
    return dom.doc + "\n\n" + "\n\n".join(parts)


def _exec(src):
    ns = {}
    exec(compile(src, "<module>", "exec"), ns)  # noqa: S102 - our own reference/wrong sources
    return ns


# --------------------------------------------------------------------------- #
# Checks and visible tests
# --------------------------------------------------------------------------- #


def _cases(rng, dom, n_funcs, exts, ref):
    """[(name, func, args)] hidden calls; expected values come from the reference module."""
    h, f = dom.helper, dom.funcs
    out = []
    for i in range(5):
        out.append((f"helper_old_{i}", h, [dom.old_form(rng)]))
    if dom is Durations:
        out.append(("helper_bare", h, [str(rng.randint(61, 400))]))
        out.append(("helper_upper", h, [f"{rng.randint(1, 9)}H{rng.randint(1, 59)}M"]))
    else:
        out.append(("helper_kb", h, [f"{rng.randint(2, 900)}kb"]))
        out.append(("helper_upper", h, [f"{rng.randint(2, 900)}MB"]))
        out.append(("helper_bare_b", h, [f"{rng.randint(2, 900)}b"]))
    for e in exts:
        for i in range(3):
            out.append((f"helper_new_{e}_{i}", h, [dom.new_form(rng, e)]))
    for i, bad in enumerate(dom.invalid(rng)):
        out.append((f"helper_invalid_{i}", h, [bad]))
    old = [dom.old_form(rng) for _ in range(5)]
    new = [dom.new_form(rng, e) for e in exts]
    out.append(("total_old", f[0], [old]))
    out.append(("total_new", f[0], [old[:3] + new]))
    out.append(("total_invalid", f[0], [old[:2] + [dom.invalid(rng)[3]]]))
    a, b = dom.tie_pairs(rng)
    v = ref[h](a)
    small = [dom.old_form(rng) for _ in range(3)]
    small = [s for s in small if ref[h](s) < v] or [dom.old_form(rng, 1)]
    out.append(("best_tie", f[1], [small[:1] + [a] + small[1:] + [b]]))
    out.append(("best_tie_reversed", f[1], [[b] + small + [a]]))
    big_new = max(new, key=lambda s: ref[h](s))
    out.append(("best_new", f[1], [[dom.old_form(rng, 1), big_new, dom.old_form(rng, 2)]]))
    out.append(("best_empty", f[1], [[]]))
    out.append(("best_invalid", f[1], [[old[0], dom.invalid(rng)[4]]]))
    if n_funcs == 3:
        limit = dom.old_form(rng)
        entries = old + new + [dom.invalid(rng)[3], dom.new_form(rng, exts[0])]
        rng.shuffle(entries)
        out.append(("over_mixed", f[2], [entries, limit]))
        out.append(("over_new_limit", f[2], [entries, dom.new_form(rng, exts[-1])]))
        out.append(("over_invalid_limit", f[2], [old, dom.invalid(rng)[5]]))
    checks = []
    for name, func, args in out:
        try:
            result = ref[func](*[list(a) if isinstance(a, list) else a for a in args])
        except ValueError:
            checks.append({"name": name, "func": func, "kind": "raises", "args": args})
            continue
        checks.append({"name": name, "func": func, "kind": "equal", "args": args, "expected": result})
    return checks


def _visible_tests(rng, dom, n_funcs, exts, ref, p, hidden_new):
    mod = dom.module[:-3]
    f, h = dom.funcs, dom.helper
    tests = []

    def add(name, call, args):
        result = ref[call.split(".")[-1]](*args)
        tests.append(f"def test_{name}():\n    assert {mod}.{call}({', '.join(repr(a) for a in args)}) == {result!r}\n")

    old = [dom.old_form(rng) for _ in range(4)]
    add("total", f[0], [old])
    a, b = dom.tie_pairs(rng)
    add("tie_returns_first", f[1], [[a, b]])
    if p["legacy_tests"]:
        add("legacy_forms", f[0], [[a, b, a.upper()]])
    if n_funcs == 3:
        add("over_skips_invalid", f[2], [[old[0], "oops", old[1]], dom.old_form(rng, 1)])
    examples = {}
    for e in exts[: p["new_examples"]] if len(exts) > 1 else exts * p["new_examples"]:
        for _ in range(20):
            s = dom.new_form(rng, e)
            if s not in hidden_new and s not in examples:
                break
        examples[s] = ref[h](s)
    for i, (s, v) in enumerate(examples.items()):
        tests.append(f"def test_{h}_new_form_{i}():\n    assert {mod}.{h}({s!r}) == {v!r}\n")
    src = f'"""Run with: python3 test_{dom.module}"""\n\nimport {mod}\n\n\n' + "\n\n".join(tests) + TEST_RUNNER
    return src, examples


def _write(module, source):
    return f"cat > /app/{module} <<'PY'\n{source}PY\n"


def build(ctx):
    rng, p = ctx.rng, ctx.params
    dom = DOMAINS[rng.choice(sorted(DOMAINS))]
    n = p["functions"]
    exts = sorted(dom.extensions) if p["extensions"] == 2 else [rng.choice(sorted(dom.extensions))]
    original = _module(dom, n, exts)
    oracle = _module(dom, n, exts, helper=_helper(dom, exts))
    ref = _exec(oracle)
    checks = _cases(rng, dom, n, exts, ref)
    hidden_new = {c["args"][0] for c in checks if c["name"].startswith("helper_new")}
    tests, examples = _visible_tests(rng, dom, n, exts, ref, p, hidden_new)
    shortcuts = {
        "helper-unused": _module(dom, n, exts, helper=_helper(dom, exts), callers=[False] * n),
        "misses-a-caller": _module(dom, n, exts, helper=_helper(dom, exts), callers=[True] * (n - 1) + [False]),
        "no-extension": _module(dom, n, exts, helper=_helper(dom, ())),
        "hardcode-examples": _module(dom, n, exts, helper=_helper(dom, (), examples=examples)),
        "case-sensitive": _module(dom, n, exts, helper=_helper(dom, exts, case=False)),
        "drops-legacy-form": _module(dom, n, exts, helper=_helper(dom, exts, legacy=False)),
        "ties-last": _module(dom, n, exts, helper=_helper(dom, exts), ties_last=True),
    }
    if n == 3:
        shortcuts["skip-becomes-raise"] = _module(dom, n, exts, helper=_helper(dom, exts), skip=False)
    if len(exts) == 2:
        shortcuts["one-extension-only"] = _module(dom, n, exts, helper=_helper(dom, exts[:1]))
    for name, src in shortcuts.items():  # by construction every shortcut fails a planted check
        if all(run_checks(_exec(src), checks).values()):
            raise Reject(f"no hidden check catches {name}")
    test_file = f"test_{dom.module}"
    if len(exts) == 1:
        ext_text = dom.extensions[exts[0]] + ". "
    else:
        ext_text = "both of these new forms:\n   - " + ";\n   - ".join(dom.extensions[e] for e in exts) + ".\n\n   "
    names = ", ".join(f"`{x}`" for x in dom.funcs[:n])
    instruction = INSTRUCTION.format(
        module=dom.module, n="two" if n == 2 else "three", names=names, noun=dom.noun, helper=dom.helper, unit=dom.unit,
        extensions=ext_text, s="s" if len(exts) > 1 else "", tests=test_file,
    )
    oracle_shell = _write(dom.module, oracle) + f"cd /app && python3 {test_file}\n"
    return TaskSpec(
        instruction=instruction,
        files={dom.module: original, test_file: tests},
        grader=Checks(f"/app/{dom.module}", dom.module[:-3], tuple(checks)),
        oracle=Solution(oracle_shell, lambda f, s=oracle: {f"/app/{dom.module}": s}),
        shortcuts={k: Solution(_write(dom.module, v), lambda f, s=v: {f"/app/{dom.module}": s}) for k, v in shortcuts.items()},
        params={"domain": dom.module[:-3], "functions": list(dom.funcs[:n]), "extensions": exts, "n_checks": len(checks)},
    )


FAMILY = Family(
    name="refactor-preserve",
    version=1,
    cluster="write-code",
    category="coding",
    skills=("python", "refactoring", "regression-safety"),
    difficulties={
        "easy": {"functions": 2, "extensions": 1, "new_examples": 2, "legacy_tests": True},
        "medium": {"functions": 2, "extensions": 1, "new_examples": 1, "legacy_tests": False},
        "hard": {"functions": 3, "extensions": 2, "new_examples": 1, "legacy_tests": False},
    },
    build=build,
)
