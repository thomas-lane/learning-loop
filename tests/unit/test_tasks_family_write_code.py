"""The write-code families (implement-function, format-converter, write-validator,
refactor-preserve, cli-flags): deterministic, clean of stray randomness, and their answer
keys and traps re-derived here from the rendered files with independent implementations,
never with the families' own reference or wrong solutions. Each family's instruction is the
same at every difficulty, apart from values drawn for the instance (names, schemas, the
example output); difficulty lives in the data."""

from __future__ import annotations

import ast
import csv
import datetime
import doctest
import importlib.util
import io
import itertools
import json
import posixpath
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.generate import rng_for, verifier_key
from learning_loop.tasks.render import render
from learning_loop.tasks.runtime import grade as grade_runtime
from learning_loop.tasks.spec import GenContext, Reject

FAMILY_DIR = REPO_ROOT / "evaluation" / "families"
MODULES = {
    "implement-function": "implement_function",
    "format-converter": "format_converter",
    "write-validator": "write_validator",
    "refactor-preserve": "refactor_preserve",
    "cli-flags": "cli_flags",
}
SEEDS = [1, 2, 3]
DIFFICULTIES = ["easy", "medium", "hard"]


def _load_module(module: str):
    path = FAMILY_DIR / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"write_code_{module}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAMILY_MODULES = {name: _load_module(module) for name, module in MODULES.items()}
FAMILIES = {name: mod.FAMILY for name, mod in FAMILY_MODULES.items()}
VERSIONS = {"implement-function": 2, "format-converter": 2, "write-validator": 2, "refactor-preserve": 2, "cli-flags": 2}


def _build(family: str, difficulty: str, seed: int):
    """The first spec `build` returns for (difficulty, seed), without generation's grading
    (fast enough to scan many seeds); draws that `build` itself rejects are skipped."""
    fam = FAMILIES[family]
    rng = rng_for(fam, difficulty, seed)
    for _ in range(100):
        try:
            return fam.build(GenContext(family=fam.name, difficulty=difficulty, seed=seed, params=fam.difficulties[difficulty], rng=rng))
        except Reject:
            continue
    raise AssertionError("no draw")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """(family, difficulty, seed) -> (task dir, params), rendered once per test module."""
    cache: dict = {}

    def get(family: str, difficulty: str, seed: int):
        key = (family, difficulty, seed)
        if key not in cache:
            d = tmp_path_factory.mktemp(f"{family}-{difficulty}-{seed}")
            cache[key] = (d, render(FAMILIES[family], difficulty, seed, d))
        return cache[key]

    return get


def _key(d: Path) -> dict:
    return json.loads((d / "tests" / "key.json").read_text())


def _files(d: Path) -> dict[str, str]:
    root = d / "environment" / "files"
    return {str(p.relative_to(root)): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- #
# Every family
# --------------------------------------------------------------------------- #


def test_family_metadata():
    for name, fam in FAMILIES.items():
        assert fam.name == name and fam.cluster == "write-code" and fam.version == VERSIONS[name] and fam.profile == "python@1"
        assert list(fam.difficulties) == DIFFICULTIES


# --------------------------------------------------------------------------- #
# implement-function
# --------------------------------------------------------------------------- #


def _compress(nums, p):
    runs = [[v for _, v in g] for _, g in itertools.groupby(enumerate(sorted(set(nums))), key=lambda iv: iv[1] - iv[0])]
    items = []
    for r in runs:
        items += [f"{r[0]}{p['sep']}{r[-1]}"] if len(r) >= p["min_run"] else [str(v) for v in r]
    return p["joiner"].join(items)


def _rle(s, p):
    out = ""
    for ch, g in itertools.groupby(s):
        n = len(list(g))
        count = "" if p["omit_one"] and n == 1 else str(n)
        out += count + ch if p["count_first"] else ch + count
    return out


def _window(xs, k, p):
    if k < 1:
        raise ValueError
    pre = [0, *itertools.accumulate(xs)]
    return [pre[i + k] - pre[i] for i in range(len(xs) - k + 1)]


def _merge(intervals, p):
    events = sorted(intervals, key=lambda iv: (iv[0], iv[1]))
    out: list[list[int]] = []
    for s, e in events:
        joins = out and (s <= out[-1][1] if p["touching"] == "merge" else s < out[-1][1])
        if joins:
            out[-1] = [out[-1][0], max(out[-1][1], e)]
        else:
            out.append([s, e])
    return out


def _top(text, k, p):
    words = [w for w in re.findall("[a-z]+", text.lower()) if len(w) >= p["min_len"]]
    return [[w, c] for w, c in sorted(Counter(words).items(), key=lambda wc: (-wc[1], wc[0]))[:k]]


def _version(a, b, p):
    def nums(v):
        if p["allow_v"] and v[:1] in "vV" and v[:1]:
            v = v[1:]
        return [int(x) for x in v.split(".")]

    for x, y in itertools.zip_longest(nums(a), nums(b), fillvalue=0):
        if x != y:
            return -1 if x < y else 1
    return 0


def _normalize(path, p):
    r = posixpath.normpath(path)
    if r.startswith("//"):
        r = "/" + r.lstrip("/")
    if p["keep_trailing"] and path.endswith("/") and r not in ("/", "."):
        r += "/"
    return r


def _split(line, p):
    if line == "":
        return [""]
    try:
        return next(csv.reader([line], delimiter=p["sep"], quotechar='"', strict=True))
    except csv.Error as e:
        raise ValueError(str(e)) from e


REFS = {
    "compress_ranges": _compress,
    "rle_encode": _rle,
    "window_sums": _window,
    "merge_intervals": _merge,
    "top_words": _top,
    "version_compare": _version,
    "normalize_path": _normalize,
    "split_fields": _split,
}


RAISES = object()


def _examples(source: str) -> dict[str, list[tuple[list, object]]]:
    """Function name -> [(args, result)] from the doctest examples in its docstring; the result
    of an example that shows an exception is `RAISES` (it must name ValueError)."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            ex = []
            for e in doctest.DocTestParser().get_examples(ast.get_docstring(node) or ""):
                call = ast.parse(e.source, mode="eval").body
                if e.exc_msg is not None:
                    assert e.exc_msg.strip() == "ValueError" and e.options.get(doctest.IGNORE_EXCEPTION_DETAIL)
                    ex.append(([ast.literal_eval(a) for a in call.args], RAISES))
                else:
                    ex.append(([ast.literal_eval(a) for a in call.args], ast.literal_eval(e.want.strip())))
            out[node.name] = ex
    return out


def _traps_present(func, checks, p, mutation=True):
    """Every trap of `func` has an input among `checks` (`mutation`: including the
    no_mutation check, which examples cannot show)."""
    args = [c["args"] for c in checks]
    if func == "compress_ranges":
        assert any(a[0] != sorted(set(a[0])) and len(set(a[0])) < len(a[0]) for a in args)  # unsorted with duplicates
        assert any(len(r) == p["min_run"] for a in args for r in _runs(a[0]))
    elif func == "rle_encode":
        assert any(max((len(list(g)) for _, g in itertools.groupby(a[0])), default=0) >= 10 for a in args) and [""] in args
    elif func == "window_sums":
        assert any(c["kind"] == "raises" and c["args"][1] < 1 for c in checks)
        assert any(c["kind"] == "equal" and c["args"][1] > len(c["args"][0]) and c["expected"] == [] for c in checks)
    elif func == "merge_intervals":
        assert any(any(x[1] == y[0] for x in a[0] for y in a[0]) for a in args)  # intervals that only touch
        assert any(any(x != y and x[0] < y[0] and y[1] < x[1] for x in a[0] for y in a[0]) for a in args)  # one inside another
        assert not mutation or any(c["kind"] == "no_mutation" and c["args"][0] != sorted(c["args"][0]) for c in checks)
    elif func == "top_words":
        def first_seen_differs(c):
            pos = {w: i for i, w in reversed(list(enumerate(re.findall("[a-z]+", c["args"][0].lower()))))}
            return any(a[1] == b[1] and pos[b[0]] < pos[a[0]] for a, b in zip(c["expected"], c["expected"][1:]))

        assert any(first_seen_differs(c) for c in checks)  # a tie whose alphabetical order is not the order of first use
        assert any(any(len(w) == p["min_len"] for w, _ in c["expected"]) for c in checks)
        assert any(re.search(r"[A-Za-z][0-9_]|[0-9_][A-Za-z]", a[0]) for a in args)  # \w+ would join these
    elif func == "version_compare":
        assert any(c["expected"] == 0 and c["args"][0] != c["args"][1] for c in checks)
        assert any(c["expected"] == -1 and c["args"][1].startswith(c["args"][0] + ".") for c in checks)
    elif func == "normalize_path":
        assert any(a[0].startswith("//") and not a[0].startswith("///") for a in args)
        assert any(c["kind"] == "equal" and c["expected"].startswith("..") for c in checks)
    elif func == "split_fields":
        assert any(c["kind"] == "raises" for c in checks) and [""] in args
        assert any('""' in a[0] for a in args)


def _runs(nums):
    return [[v for _, v in g] for _, g in itertools.groupby(enumerate(sorted(set(nums))), key=lambda iv: iv[1] - iv[0])]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_implement_function_key_and_traps(rendered, difficulty, seed):
    d, params = rendered("implement-function", difficulty, seed)
    key, files = _key(d), _files(d)
    ((module, source),) = files.items()
    assert module == "toolkit.py" and "raise NotImplementedError" in source and key["module"] == "toolkit"
    examples = _examples(source)
    assert sorted(examples) == sorted(params["functions"]) and len(examples) == 2
    for func in params["functions"]:
        p, ref = params["spec"][func], REFS[func]
        for args, want in examples[func]:
            if want is RAISES:
                with pytest.raises(ValueError):
                    ref(*args, p)
            else:
                assert ref(*args, p) == want, (func, args)
        checks = [c for c in key["checks"] if c["func"] == func]
        for c in checks:
            if c["kind"] == "raises":
                with pytest.raises(ValueError):
                    ref(*c["args"], p)
            elif c["kind"] == "equal":
                assert ref(*c["args"], p) == c["expected"], (func, c)
        shown = [a for a, _ in examples[func]]
        assert sum(c["args"] not in shown for c in checks) >= 4  # hard-coding the examples fails
        _traps_present(func, checks, p)


# --------------------------------------------------------------------------- #
# format-converter
# --------------------------------------------------------------------------- #

KV_PAIR = re.compile(r'([a-z0-9_]+)=(?:"((?:[^"\\]|\\.)*)"|([^ "]*))')


def _kv_csv(text, columns):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rec = {m[1]: re.sub(r"\\(.)", r"\1", m[2]) if m[2] is not None else m[3] for m in KV_PAIR.finditer(line)}
        w.writerow([rec.get(c, "") for c in columns])
    return buf.getvalue()


def _fixed_jsonl(text, fields):
    out = ""
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rec = {}
        for name, start, end, kind in fields:
            raw = line.ljust(end)[start - 1 : end].strip()
            rec[name] = (int(raw) if raw else None) if kind == "int" else raw
        out += json.dumps(rec) + "\n"
    return out


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_format_converter_key_and_traps(rendered, difficulty, seed):
    d, params = rendered("format-converter", difficulty, seed)
    key, files = _key(d), _files(d)
    assert key["files"] == ["/app/convert.py"] and "convert.py" not in files
    kv = params["conversion"] == "kv-csv"
    convert = (lambda t: _kv_csv(t, params["columns"])) if kv else (lambda t: _fixed_jsonl(t, params["fields"]))
    src, out = ("examples/sample.log", "examples/sample.csv") if kv else ("examples/sample.txt", "examples/sample.jsonl")
    assert convert(files[src]) == files[out]
    inputs = []
    for c in key["checks"]:
        ((name, text),) = c["inputs"].items()
        assert c["argv"] == ["python3", "convert.py", name] and c["exit"] == 0
        assert convert(text) == c["stdout"]
        assert c["stdout"] != files[out]  # printing the example's output fails
        if text:
            inputs.append(text)
    assert len(inputs) == 3
    for text in inputs:
        lines = text.splitlines()
        if kv:
            assert re.search(r'="[^"]* [^"]*"', text) and re.search(r'="[^"]*,[^"]*"', text)  # spaces (split) and commas (CSV quoting)
            if difficulty != "easy":
                assert '\\"' in text and any(ln.startswith("#") for ln in lines) and any(not ln.strip() for ln in lines)
            if difficulty == "hard":
                assert re.search(r"=[^\" ]*['\\\\][^ ]*", text)  # a bare value shlex would misread
        else:
            assert '"' in text  # hand-built JSON must escape it
            ints = [(s, e) for n, s, e, k in params["fields"] if k == "int" and n != "id"]
            if difficulty != "easy":
                assert any(ln.startswith("#") for ln in lines)
                assert any(not ln.startswith("#") and ln.ljust(e)[s - 1 : e].strip() == "" for ln in lines for s, e in ints)
            if difficulty == "hard":
                width = max(e for _, _, e, _ in params["fields"])
                assert any(ord(ch) > 127 for ch in text) and any(len(ln) < width for ln in lines if not ln.startswith("#"))


# --------------------------------------------------------------------------- #
# write-validator
# --------------------------------------------------------------------------- #


def _schema_from_instruction(text):
    """[(name, kind, required, opts)] parsed from the instruction's schema list."""
    out = []
    for m in re.finditer(r"^- `(\w+)` \((required|optional)\): (.*)$", text, re.M):
        name, req, desc = m[1], m[2] == "required", m[3]
        if desc.startswith("a string of"):
            lo, hi = map(int, re.match(r"a string of (\d+) to (\d+)", desc).groups())
            sep = "_" if "underscores" in desc else "-"
            out.append((name, "ident", req, {"min": lo, "max": hi, "chars": set("abcdefghijklmnopqrstuvwxyz0123456789" + sep)}))
        elif desc.startswith("an integer from"):
            lo, hi = map(int, re.match(r"an integer from (-?\d+) to (-?\d+)", desc).groups())
            out.append((name, "int", req, {"min": lo, "max": hi}))
        elif desc.startswith("one of"):
            out.append((name, "enum", req, {"values": re.findall(r'`"([^"]+)"`', desc.split("(exactly")[0])}))
        elif desc.startswith("an email"):
            out.append((name, "email", req, {}))
        elif desc.startswith("a list of at most"):
            out.append((name, "tags", req, {"max": int(re.match(r"a list of at most (\d+)", desc)[1])}))
        elif desc.startswith("a calendar date"):
            out.append((name, "date", req, {}))
        else:
            raise AssertionError(desc)
    return out


def _codes(record, schema, unknown_rule, not_object):
    if not_object and type(record) is not dict:
        return ["not_an_object"]
    codes = set()
    for name, kind, req, o in schema:
        if name not in record:
            if req:
                codes.add(f"{name}_missing")
            continue
        v = record[name]
        if kind == "int":
            if type(v) is not int:
                codes.add(f"{name}_type")
            elif not o["min"] <= v <= o["max"]:
                codes.add(f"{name}_range")
        elif kind == "enum":
            if not (type(v) is str and v in o["values"]):
                codes.add(f"{name}_invalid")
        elif type(v) is not (list if kind == "tags" else str):
            codes.add(f"{name}_type")
        elif kind == "ident":
            if not o["min"] <= len(v) <= o["max"]:
                codes.add(f"{name}_length")
            if not (v[:1].isascii() and v[:1].isalpha() and v[:1].islower() and set(v) <= o["chars"]):
                codes.add(f"{name}_format")
        elif kind == "email":
            if not re.fullmatch(r"[^@ ]+@[a-z0-9-]+(\.[a-z0-9-]+)+", v):
                codes.add(f"{name}_format")
        elif kind == "date":
            try:
                ok = len(v) == 10 and v[4] == v[7] == "-" and all(ch.isascii() and ch.isdigit() for ch in v[:4] + v[5:7] + v[8:])
                ok = ok and datetime.date(int(v[:4]), int(v[5:7]), int(v[8:])) is not None
            except ValueError:
                ok = False
            if not ok:
                codes.add(f"{name}_format")
        elif kind == "tags":
            if any(type(t) is not str or not t for t in v):
                codes.add(f"{name}_item")
            strings = [t for t in v if type(t) is str]
            if len(strings) != len(set(strings)):
                codes.add(f"{name}_duplicate")
            if len(v) > o["max"]:
                codes.add(f"{name}_too_many")
    if unknown_rule and any(k not in {f[0] for f in schema} for k in record):
        codes.add("unknown_field")
    return sorted(codes)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_write_validator_key_and_traps(rendered, difficulty, seed):
    d, params = rendered("write-validator", difficulty, seed)
    key, files = _key(d), _files(d)
    instruction = (d / "instruction.md").read_text()
    schema = _schema_from_instruction(instruction)
    assert [f[0] for f in schema] == params["fields"] and sorted(f[1] for f in schema) == sorted(["ident", "int", "enum", "email", "tags", "date"])
    assert "`validate(record)`" in instruction and "`unknown_field`" in instruction and "not_an_object" in instruction

    def result(record):
        return _codes(record, schema, True, True)

    for e in json.loads(files["examples.json"]):
        assert result(e["record"]) == e["expected"]
    shown = [e["record"] for e in json.loads(files["examples.json"])]
    records = [c["args"][0] for c in key["checks"]]
    assert all(c["func"] == "validate" and c["kind"] == "equal" for c in key["checks"])
    for c in key["checks"]:
        assert result(c["args"][0]) == c["expected"], c
    assert not any(r in shown for r in records)
    ident = next(f for f in schema if f[1] == "ident")
    num = next(f for f in schema if f[1] == "int")
    dicts = [r for r in records if type(r) is dict]
    # the planted traps
    assert any(r.get(num[0]) is True for r in dicts) and num[3]["min"] <= 1 <= num[3]["max"]  # true would pass as 1
    assert any(type(r.get(num[0])) is str and r[num[0]].isdigit() for r in dicts)
    for bound in (num[3]["min"], num[3]["max"]):
        assert any(type(r.get(num[0])) is int and r[num[0]] == bound and result(r) == [] for r in dicts)
    assert any(isinstance(r.get(ident[0]), str) and r[ident[0]].endswith("\n") for r in dicts)
    assert any(isinstance(r.get(ident[0]), str) and re.match(r"[a-z][a-z0-9_-]*", r[ident[0]]) and not set(r[ident[0]]) <= ident[3]["chars"] for r in dicts)
    names = [f[0] for f in schema]

    def schema_order(code):
        return next((i for i, n in enumerate(names) if code.startswith(n + "_")), len(names))

    multi = [result(r) for r in dicts if len(result(r)) >= 2]
    assert multi and any(sorted(cs, key=schema_order) != cs for cs in multi)  # checking in schema order is not enough
    assert any("unknown_field" in result(r) for r in dicts) and any(type(r) is not dict for r in records)
    date = next(f[0] for f in schema if f[1] == "date")
    library_traps = any(r.get(date) == "2026-2-03" for r in dicts) and any(r.get(date) == "20260203" for r in dicts)
    assert library_traps == (difficulty != "easy")


# --------------------------------------------------------------------------- #
# refactor-preserve
# --------------------------------------------------------------------------- #


def _duration(text, exts):
    t = text.strip().lower()
    if t.isdigit():
        return int(t)
    units = (["d"] if "days" in exts else []) + ["h", "m"]
    mult = {"d": 1440, "h": 60, "m": 1}
    total, seen, pos = 0, -1, 0
    if not t:
        raise ValueError(text)
    while pos < len(t):
        m = re.match(r"([0-9]+)([a-z])", t[pos:])
        if not m or m[2] not in units or units.index(m[2]) <= seen:
            raise ValueError(text)
        seen = units.index(m[2])
        total += int(m[1]) * mult[m[2]]
        pos += m.end()
        if "spaces" in exts:
            while pos < len(t) and t[pos] == " " and pos + 1 < len(t):
                pos += 1
    return total


def _size(text, exts):
    t = text.strip().lower()
    m = re.fullmatch(r"([0-9]+)( *)([kmgt]?)(b?)", t)
    units = "kmgt" if "tera" in exts else "kmg"
    if not m or (m[2] and "spaces" not in exts) or (m[3] and m[3] not in units):
        raise ValueError(text)
    return int(m[1]) * 1024 ** (units.index(m[3]) + 1 if m[3] else 0)


def _call_or_error(fn, args):
    try:
        return ("ok", fn(*[list(a) if isinstance(a, list) else a for a in args]))
    except ValueError:
        return ("raises", None)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_refactor_preserve_key_and_traps(rendered, difficulty, seed):
    d, params = rendered("refactor-preserve", difficulty, seed)
    key, files = _key(d), _files(d)
    module = params["domain"] + ".py"
    exts = params["extensions"]
    parse = (lambda s: _duration(s, exts)) if params["domain"] == "timesheet" else (lambda s: _size(s, exts))
    original: dict = {}
    exec(compile(files[module], module, "exec"), original)  # noqa: S102 - the rendered, agent-visible module
    helper = next(c["func"] for c in key["checks"] if c["func"].startswith("parse_"))
    assert helper not in original  # doing nothing fails
    total, best, *over = params["functions"]

    def expected(func, args):
        if func == helper:
            return _call_or_error(parse, args)
        if func == total:
            return _call_or_error(lambda xs: sum(parse(x) for x in xs), args)
        if func == best:
            return _call_or_error(lambda xs: max(xs, key=parse) if xs else None, args)  # max keeps the first of equal keys

        def over_fn(xs, limit):
            lim, out = parse(limit), []
            for x in xs:
                try:
                    if parse(x) > lim:
                        out.append(x)
                except ValueError:
                    pass
            return out

        return _call_or_error(over_fn, args)

    changed = Counter()
    for c in key["checks"]:
        want = expected(c["func"], c["args"])
        assert want == (("raises", None) if c["kind"] == "raises" else ("ok", c["expected"])), c
        if c["func"] != helper:
            got = _call_or_error(original[c["func"]], c["args"])
            changed[c["func"]] += got != want
            legacy = all(_call_or_error(lambda s: _duration(s, ()) if params["domain"] == "timesheet" else _size(s, ()), [s])[0] == "ok" for s in _strings(c["args"]))
            if legacy:
                assert got == want, c  # on old-format inputs the original behavior is the answer
    assert all(changed[f] >= 1 for f in params["functions"])  # every function must reach the extended helper
    helper_args = [c["args"][0] for c in key["checks"] if c["func"] == helper]
    assert any(s != s.lower() for s in helper_args)
    if params["domain"] == "timesheet":
        assert any(s.strip().isdigit() for s in helper_args)
    else:
        assert any(re.fullmatch(r"\s*[0-9]+[kmgt]?b\s*", s.lower()) for s in helper_args)
    best_lists = [c["args"][0] for c in key["checks"] if c["func"] == best and c["kind"] == "equal" and c["args"][0]]
    assert any(sorted((parse(x) for x in xs), reverse=True)[:2] == [max(map(parse, xs))] * 2 for xs in best_lists)  # a tie for the largest
    if over:
        assert any(c["func"] == over[0] and c["kind"] == "equal" and any(_call_or_error(parse, [x])[0] == "raises" for x in c["args"][0]) for c in key["checks"])
    assert re.search(rf"{helper}\(", files[f"test_{module}"])  # the visible tests show the new form
    assert len(exts) == 2 and len(params["functions"]) == 3


def _strings(args):
    for a in args:
        if isinstance(a, str):
            yield a
        elif isinstance(a, list):
            yield from (x for x in a if isinstance(x, str))


# --------------------------------------------------------------------------- #
# cli-flags
# --------------------------------------------------------------------------- #


def _cli_expected(argv, inputs):
    """(stdout, exit) per the instruction, from a hand-rolled reading of argv."""
    path, rest = argv[2], argv[3:]
    opts = {"limit": None, "format": "text", "min": 1, "only": [], "exclude": []}
    formats = ["text", "json", "csv"]
    i = 0
    while i < len(rest):
        flag, value = rest[i], rest[i + 1]
        i += 2
        if flag in ("--limit", "--min-count"):
            if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
                return "", 2
            opts["limit" if flag == "--limit" else "min"] = int(value)
        elif flag == "--format":
            if value not in formats:
                return "", 2
            opts["format"] = value
        else:
            opts[flag[2:]].append(value)
    if opts["only"] and opts["exclude"]:
        return "", 2
    if path not in inputs:
        return "", 1
    counts = Counter(line.split("\t")[1] for line in inputs[path].splitlines())
    rows = sorted(((n, c) for n, c in counts.items() if c >= opts["min"]), key=lambda nc: (-nc[1], nc[0]))
    rows = [r for r in rows if (not opts["only"] or r[0] in opts["only"]) and r[0] not in opts["exclude"]]
    rows = rows[: opts["limit"]] if opts["limit"] else rows
    if opts["format"] == "json":
        return json.dumps([{"category": n, "count": c} for n, c in rows]) + "\n", 0
    if opts["format"] == "csv":
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows([("category", "count"), *rows])
        return buf.getvalue(), 0
    return "".join(f"{n}: {c}\n" for n, c in rows), 0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_cli_flags_key_and_traps(rendered, tmp_path, difficulty, seed):
    d, params = rendered("cli-flags", difficulty, seed)
    key, files = _key(d), _files(d)
    checks = {c["name"]: c for c in key["checks"]}
    for c in key["checks"]:
        assert _cli_expected(c["argv"], c.get("inputs", {})) == (c["stdout"], c["exit"]), c
    # the original script's output is the default output
    (tmp_path / "report.py").write_text(files["report.py"])
    (tmp_path / "events.log").write_text(checks["default"]["inputs"]["events.log"])
    run = subprocess.run([sys.executable, "report.py", "events.log"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert run.stdout == checks["default"]["stdout"]
    log = checks["default"]["inputs"]["events.log"]
    counts = Counter(line.split("\t")[1] for line in log.splitlines())
    instruction = (d / "instruction.md").read_text()
    example = instruction.split("```\n")[1]
    assert _cli_expected(["python3", "report.py", "sample.log", "--limit", "2", "--format", "json"], {"sample.log": files["sample.log"]}) == (example, 0)
    assert all(c["stdout"] != example for c in key["checks"])
    assert any('"' in n for n in counts) and '\\"' in checks["json"]["stdout"]
    assert {checks["limit_zero"]["exit"], checks["limit_negative"]["exit"]} == {2}
    if difficulty != "easy":
        assert any("," in n for n in counts)
        boundary = int(checks["min_count_boundary"]["argv"][-1])
        assert boundary in counts.values()
    if difficulty == "hard":
        c = checks["exclude_top_then_limit"]
        top = max(counts, key=lambda n: (counts[n], [-ord(ch) for ch in n]))
        assert c["argv"][c["argv"].index("--exclude") + 1] == top and len(c["stdout"].splitlines()) == 2
        assert checks["only_and_exclude"]["exit"] == 2 and checks["missing_log"]["exit"] == 1
        assert len(checks["only_repeated"]["stdout"].splitlines()) == 2


# --------------------------------------------------------------------------- #
# One instruction per family, difficulty in the data
# --------------------------------------------------------------------------- #


def _template(family: str, spec) -> tuple[str, str]:
    """(variant, instruction with the instance's own values blanked out). Variants are the
    seed's choice of domain or conversion, whose instructions legitimately differ."""
    text, params = spec.instruction, spec.params
    if family == "cli-flags":
        return "", re.sub(r"```\n.*?```", "```{example}```", text, flags=re.S)
    if family == "write-validator":
        return params["domain"], re.sub(r"Schema:\n.*?\nRules:", "Schema:\n{fields}\nRules:", text, flags=re.S)
    if family == "refactor-preserve":
        return params["domain"], text
    if family == "implement-function":
        a, b = params["functions"]
        return "", text.replace(f"`{a}`", "`{a}`", 1).replace(f"`{b}`", "`{b}`", 1)
    if params["conversion"] == "kv-csv":
        return "kv-csv", re.sub(r"header `[^`]*`", "header `{header}`", text)
    return "fixed-jsonl", re.sub(r"(\| `\w+` \| \d+-\d+ \| \w+ \|\n)+", "{table}\n", text)


@pytest.mark.parametrize("family", list(MODULES))
def test_instruction_is_the_same_at_every_difficulty(family):
    texts: dict[str, dict[str, set[str]]] = {}
    for difficulty in DIFFICULTIES:
        for seed in range(12):
            variant, text = _template(family, _build(family, difficulty, seed))
            texts.setdefault(variant, {}).setdefault(difficulty, set()).add(text)
    for variant, by_level in texts.items():
        assert len(set().union(*by_level.values())) == 1, (family, variant)
    assert any(len(by_level) == 3 for by_level in texts.values())


def test_checks_families_say_the_module_is_tested_on_its_own():
    """The `checks` grader copies only the module file into an empty directory, so a helper in
    another module fails the import; each instruction says so."""
    for family in ("implement-function", "write-validator", "refactor-preserve"):
        for difficulty in DIFFICULTIES:
            assert "importing that file on its own, from a directory that holds no other file" in _build(family, difficulty, 1).instruction


# --------------------------------------------------------------------------- #
# cli-flags
# --------------------------------------------------------------------------- #


def test_cli_flags_instruction_defines_every_output():
    text = _build("cli-flags", "easy", 1).instruction
    assert "An empty report is just the header line." in text and "An empty report is `[]`." in text
    assert "every other field is written as it is, without quotes" in text
    assert "`--format csv`" not in text and "`text`, `json` or `csv`" in text and "exit with status 1" in text


def test_cli_flags_only_min_count_keeps_a_row():
    """`--only a --only b --min-count min+1` on two categories with equal counts left only the
    CSV header; a and b now have distinct counts, so one row remains."""
    for seed in range(150):
        checks = {c["name"]: c for c in _build("cli-flags", "hard", seed).grader.checks}
        c = checks["only_min_count"]
        only = [c["argv"][i + 1] for i, a in enumerate(c["argv"]) if a == "--only"]
        counts = Counter(line.split("\t")[1] for line in c["inputs"]["events.log"].splitlines())
        assert len({counts[n] for n in only}) == 2 and len(c["stdout"].splitlines()) == 2, seed


def test_cli_flags_top_count_is_unique_by_construction():
    mod = FAMILY_MODULES["cli-flags"]
    for i in range(500):
        rng = rng_for(FAMILIES["cli-flags"], "hard", i)
        values = sorted(mod._counts(rng, rng.choice(DIFFICULTIES)).values(), reverse=True)
        assert values[0] > values[1] and len(set(values)) == len(values) - 1


# --------------------------------------------------------------------------- #
# write-validator
# --------------------------------------------------------------------------- #


def _codes_for(spec, record):
    return _codes(record, _schema_from_instruction(spec.instruction), True, True)


def test_write_validator_ordering_record_breaks_only_plain_rules():
    """The record planted for code ordering has two plain errors; its int field used to get
    `True`, so the bool trap, not the ordering, decided the check."""
    for difficulty in DIFFICULTIES:
        for seed in range(40):
            spec = _build("write-validator", difficulty, seed)
            names = [f[0] for f in _schema_from_instruction(spec.instruction)]
            (c,) = [c for c in spec.grader.checks if c["name"].startswith("multi_order_")]
            record = c["args"][0]
            codes = _codes_for(spec, record)
            assert len(codes) == 2 and not any(type(v) is bool for v in record.values()), (seed, record)
            order = [next(i for i, n in enumerate(names) if code.startswith(n + "_")) for code in codes]
            assert order == sorted(order, reverse=True) and order[0] != order[1]


def _validator_traps(spec, record):
    """The traps a record exhibits, judged from the instruction's schema."""
    schema = _schema_from_instruction(spec.instruction)
    if type(record) is not dict:
        return {"assumes-object"}
    ident = next(f for f in schema if f[1] == "ident")
    num = next(f for f in schema if f[1] == "int")
    out = set()
    v = record.get(num[0])
    if v is True or v is False:
        out.add("bool-is-int")
    if type(v) is str and v.isdigit():
        out.add("coerces-strings")
    if type(v) is int and v in (num[3]["min"], num[3]["max"]):
        out.add("range-off-by-one")
    s = record.get(ident[0])
    if type(s) is str and s.endswith("\n"):
        out.add("dollar-newline")
    if type(s) is str and re.fullmatch(r"[a-z][a-z0-9_-]*", s) and not set(s) <= ident[3]["chars"]:
        out.add("re-match-prefix")
    if any(k not in {f[0] for f in schema} for k in record):
        out.add("unknown-fields-ignored")
    return out


def test_write_validator_examples_show_the_traps_by_difficulty():
    shown_traps = {"assumes-object", "bool-is-int", "coerces-strings", "range-off-by-one", "dollar-newline", "re-match-prefix", "unknown-fields-ignored"}
    for seed in range(10):
        by_level = {}
        for difficulty in DIFFICULTIES:
            spec = _build("write-validator", difficulty, seed)
            examples = json.loads(spec.files["examples.json"])
            by_level[difficulty] = set().union(*(_validator_traps(spec, e["record"]) for e in examples))
            for e in examples:
                assert _codes_for(spec, e["record"]) == e["expected"]
        assert by_level["easy"] == shown_traps and by_level["hard"] == set(), seed


# --------------------------------------------------------------------------- #
# refactor-preserve
# --------------------------------------------------------------------------- #


def _visible_calls(test_source: str, func: str) -> list[list]:
    return [[ast.literal_eval(a) for a in node.args] for node in ast.walk(ast.parse(test_source)) if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == func]


def test_refactor_preserve_legacy_test_exercises_each_legacy_rule():
    """Easy's visible legacy test used [a, b, a.upper()], where .upper() changes nothing for a
    bare number; now each string relies on a legacy rule."""
    for seed in range(20):
        spec = _build("refactor-preserve", "easy", seed)
        tests = next(v for k, v in spec.files.items() if k.startswith("test_"))
        (legacy,) = re.findall(r"def test_legacy_forms\(\):\n    assert \w+\.\w+\((\[.*?\])\)", tests)
        strings = ast.literal_eval(legacy)
        assert any(s != s.lower() for s in strings) and any(s != s.strip() for s in strings)
        if spec.params["domain"] == "timesheet":
            assert any(s.strip().isdigit() for s in strings)
        else:
            assert any(re.fullmatch(r"[0-9]+b", s.strip()) for s in strings)


def test_refactor_preserve_invalid_limit_is_not_a_second_helper_case():
    """The third function's invalid limit was another reversed-order duration, the case
    `helper_invalid_5` already checks; it is now a form the helper checks do not use."""
    for difficulty in DIFFICULTIES:
        for seed in range(20):
            checks = _build("refactor-preserve", difficulty, seed).grader.checks
            forms = {re.sub("[0-9]+", "N", c["args"][0]) for c in checks if c["name"].startswith("helper_invalid")}
            (limit,) = [c["args"][1] for c in checks if c["name"] == "over_invalid_limit"]
            assert re.sub("[0-9]+", "N", limit) not in forms


def test_refactor_preserve_difficulty_is_in_the_data():
    for seed in range(10):
        shown, invalid = {}, {}
        for difficulty in DIFFICULTIES:
            spec = _build("refactor-preserve", difficulty, seed)
            assert len(spec.params["extensions"]) == 2 and len(spec.params["functions"]) == 3
            tests = next(v for k, v in spec.files.items() if k.startswith("test_"))
            shown[difficulty] = len(re.findall(r"def test_\w+_new_form_", tests))
            invalid[difficulty] = sum(c["name"].startswith("helper_invalid") for c in spec.grader.checks)
        assert shown["easy"] > shown["medium"] > shown["hard"] >= 1
        assert invalid["easy"] < invalid["medium"] < invalid["hard"]


# --------------------------------------------------------------------------- #
# implement-function
# --------------------------------------------------------------------------- #


def test_implement_function_shortcut_names_are_unique_across_the_catalog():
    """Any two catalog entries can share an instance, so no wrong variant may share a name
    (merge_intervals and compress_ranges both had `unsorted-input`)."""
    import random

    names = [n for e in FAMILY_MODULES["implement-function"].CATALOG.values() for n in e.wrong(e.params(random.Random(0)))]
    assert len(names) == len(set(names)) and "hardcode-examples" not in names


def test_implement_function_examples_cover_edges_by_difficulty():
    """Easy's docstring examples show every trap its hidden checks test, hard's none; the two
    functions are drawn the same way at every difficulty."""
    funcs = {d: Counter() for d in DIFFICULTIES}
    for seed in range(12):
        for difficulty in DIFFICULTIES:
            spec = _build("implement-function", difficulty, seed)
            examples = _examples(spec.files["toolkit.py"])
            for func in spec.params["functions"]:
                funcs[difficulty][func] += 1
                p = spec.params["spec"][func]
                as_checks = [
                    {"args": a, "kind": "raises"} if r is RAISES else {"args": a, "kind": "equal", "expected": r}
                    for a, r in examples[func]
                ]
                hidden_raises = any(c["kind"] == "raises" for c in spec.grader.checks if c["func"] == func)
                if difficulty == "easy":
                    _traps_present(func, as_checks, p, mutation=False)
                    assert hidden_raises == any(c["kind"] == "raises" for c in as_checks)
                if difficulty == "hard":
                    assert all(c["kind"] == "equal" for c in as_checks) and len(as_checks) == 2
    assert all(len(c) >= 6 for c in funcs.values())


# --------------------------------------------------------------------------- #
# format-converter
# --------------------------------------------------------------------------- #


def test_format_converter_instruction_matches_the_grader():
    """The grader compares stdout after text-mode decoding (`\\r\\n` becomes `\\n`) and ignores
    trailing line breaks, and runs the converter in a directory without the examples: the
    instruction says both, and a converter writing CRLF without a final line break scores 1."""
    for seed in range(6):
        spec = _build("format-converter", "medium", seed)
        text = spec.instruction
        assert "Lines may end with `\\n` or `\\r\\n`, and the line break after the last line is optional." in text
        assert "from a directory that holds only `convert.py`" in text and "cannot read the examples" in text
        assert "Blank lines (empty or containing only spaces)" in text
        key = verifier_key(spec)
        outputs = {c["argv"][-1]: c["stdout"].replace("\n", "\r\n").rstrip("\r\n") for c in key["checks"]}
        script = f"import sys\nsys.stdout.write({outputs!r}[sys.argv[1]])\n"
        view = grade_runtime.MemoryView({"/app/convert.py": (script.encode(), 0o644)})
        assert grade_runtime.grade(key, view, trusted=True)[0] == 1.0


def test_format_converter_data_by_difficulty():
    """Every difficulty can contain every case of the grammar; the data decides which do. A
    fixed-width export always has an optional integer field (for spaces-only integers), and
    from medium on both conversions contain a spaces-only blank line."""
    for seed in range(30):
        for difficulty in DIFFICULTIES:
            spec = _build("format-converter", difficulty, seed)
            if spec.params["conversion"] == "fixed-jsonl":
                assert any(k == "int" and n != "id" for n, _, _, k in spec.params["fields"])
            if difficulty != "easy":
                for c in spec.grader.checks:
                    ((_, text),) = c["inputs"].items()
                    assert not text or any(ln and not ln.strip() for ln in text.splitlines())
