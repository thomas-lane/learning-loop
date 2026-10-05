"""The write-code families (implement-function, format-converter, write-validator,
refactor-preserve, cli-flags): deterministic, clean of stray randomness, and their answer
keys and traps re-derived here from the rendered files with independent implementations,
never with the families' own reference or wrong solutions."""

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
from learning_loop.tasks.render import render

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


def _load(module: str):
    path = FAMILY_DIR / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"write_code_{module}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAMILY


FAMILIES = {name: _load(module) for name, module in MODULES.items()}


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
        assert fam.name == name and fam.cluster == "write-code" and fam.version == 1 and fam.profile == "python@1"
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


def _examples(source: str) -> dict[str, list[tuple[list, object]]]:
    """Function name -> [(args, result)] from the doctest examples in its docstring."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            ex = []
            for e in doctest.DocTestParser().get_examples(ast.get_docstring(node) or ""):
                call = ast.parse(e.source, mode="eval").body
                ex.append(([ast.literal_eval(a) for a in call.args], ast.literal_eval(e.want.strip())))
            out[node.name] = ex
    return out


def _traps_present(func, checks, p):
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
        assert any(c["kind"] == "no_mutation" and c["args"][0] != sorted(c["args"][0]) for c in checks)
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
    assert "raise NotImplementedError" in source and key["module"] == module[:-3]
    examples = _examples(source)
    assert sorted(examples) == sorted(params["functions"])
    for func in params["functions"]:
        p, ref = params["spec"][func], REFS[func]
        for args, want in examples[func]:
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
    assert [f[0] for f in schema] == params["fields"]
    api = "is_valid" if "`is_valid(record)`" in instruction else "validate"
    unknown_rule, not_object = "`unknown_field`" in instruction, "not_an_object" in instruction

    def result(record):
        codes = _codes(record, schema, unknown_rule, not_object)
        return not codes if api == "is_valid" else codes

    for e in json.loads(files["examples.json"]):
        assert result(e["record"]) == e["expected"]
    shown = [e["record"] for e in json.loads(files["examples.json"])]
    records = [c["args"][0] for c in key["checks"]]
    assert all(c["func"] == api and c["kind"] == "equal" for c in key["checks"])
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
        assert any(type(r.get(num[0])) is int and r[num[0]] == bound and result(r) in (True, []) for r in dicts)
    assert any(isinstance(r.get(ident[0]), str) and r[ident[0]].endswith("\n") for r in dicts)
    assert any(isinstance(r.get(ident[0]), str) and re.match(r"[a-z][a-z0-9_-]*", r[ident[0]]) and not set(r[ident[0]]) <= ident[3]["chars"] for r in dicts)
    if api == "validate":
        names = [f[0] for f in schema]

        def schema_order(code):
            return next((i for i, n in enumerate(names) if code.startswith(n + "_")), len(names))

        multi = [result(r) for r in dicts if len(result(r)) >= 2]
        assert multi and any(sorted(cs, key=schema_order) != cs for cs in multi)  # checking in schema order is not enough
    if difficulty == "hard":
        date = next(f[0] for f in schema if f[1] == "date")
        assert any(r.get(date) == "2026-2-03" for r in dicts) and any(r.get(date) == "20260203" for r in dicts)
        assert any("unknown_field" in result(r) for r in dicts) and any(type(r) is not dict for r in records)


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


def _strings(args):
    for a in args:
        if isinstance(a, str):
            yield a
        elif isinstance(a, list):
            yield from (x for x in a if isinstance(x, str))


# --------------------------------------------------------------------------- #
# cli-flags
# --------------------------------------------------------------------------- #


def _cli_expected(argv, inputs, difficulty):
    """(stdout, exit) per the instruction, from a hand-rolled reading of argv."""
    path, rest = argv[2], argv[3:]
    opts = {"limit": None, "format": "text", "min": 1, "only": [], "exclude": []}
    formats = ["text", "json"] + (["csv"] if difficulty != "easy" else [])
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
        assert _cli_expected(c["argv"], c.get("inputs", {}), difficulty) == (c["stdout"], c["exit"]), c
    # the original script's output is the default output
    (tmp_path / "report.py").write_text(files["report.py"])
    (tmp_path / "events.log").write_text(checks["default"]["inputs"]["events.log"])
    run = subprocess.run([sys.executable, "report.py", "events.log"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert run.stdout == checks["default"]["stdout"]
    log = checks["default"]["inputs"]["events.log"]
    counts = Counter(line.split("\t")[1] for line in log.splitlines())
    instruction = (d / "instruction.md").read_text()
    example = instruction.split("```\n")[1]
    assert _cli_expected(["python3", "report.py", "sample.log", "--limit", "2", "--format", "json"], {"sample.log": files["sample.log"]}, difficulty) == (example, 0)
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
