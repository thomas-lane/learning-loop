"""write-validator: write a validator for a schema stated in the instruction.

The agent fills in /app/validator.py. Easy asks for `is_valid(record) -> bool`; medium and
hard ask for `validate(record) -> list[str]`, the sorted, distinct error codes of a record
(`[]` when it is valid). The schema's fields come from one of two domains (an account or a
device), with seed-drawn bounds, lengths and allowed values. Field kinds: an identifier string
(length and character rules), an integer range, an enum, an email address, a tag list and a
calendar date. /app/examples.json shows a few records with their results.

Grading: `Checks` (kind `equal`) on hidden records: valid ones (including every boundary:
the range ends, the length limits, a leap day), records with one planted error per rule, and
records with several errors.

Traps (each a full validator with one mistake, declared as a shortcut, and each caught by a
planted hidden record): `hardcode-examples` (answers the example records only),
`bool-is-int` (`isinstance(v, int)` accepts true; the integer range includes 1),
`coerces-strings` (`int(v)` accepts "7"), `range-off-by-one` (exclusive bounds),
`re-match-prefix` (`re.match` without an end anchor accepts "abc-def"); medium and hard add
`dollar-newline` (`^...$` accepts a trailing newline), `stops-at-first-error` and
`unsorted-codes` (multi-error records whose schema order is not alphabetical); hard adds
`strptime-date` (accepts "2026-2-03"), `fromisoformat-date` (accepts "20260203"),
`unknown-fields-ignored` and `assumes-object` (non-object inputs).
"""

import json

from learning_loop.tasks.runtime.grade import run_checks
from learning_loop.tasks.spec import Checks, Family, Reject, Solution, TaskSpec

ENGINE = r'''import datetime
import json
import re

SCHEMA = @SCHEMA@
KNOWN = {name for name, _, _, _ in SCHEMA}
LABEL = re.compile(r"[a-z0-9-]+")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _email_ok(v):
    local, at, domain = v.partition("@")
    if not at or not local or " " in local or "@" in domain:
        return False
    labels = domain.split(".")
    return len(labels) >= 2 and all(LABEL.fullmatch(label) for label in labels)


def _date_ok(v):
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", v):
        return False
    try:
        datetime.date(int(v[:4]), int(v[5:7]), int(v[8:]))
    except ValueError:
        return False
    return True


def _check(name, kind, opts, v):
    if kind == "int":
        if not _is_int(v):
            return [name + "_type"]
        return [] if opts["min"] <= v <= opts["max"] else [name + "_range"]
    if kind == "ident":
        if not isinstance(v, str):
            return [name + "_type"]
        out = []
        if not opts["min_len"] <= len(v) <= opts["max_len"]:
            out.append(name + "_length")
        if not re.fullmatch(opts["pattern"], v):
            out.append(name + "_format")
        return out
    if kind == "enum":
        return [] if isinstance(v, str) and v in opts["values"] else [name + "_invalid"]
    if kind in ("email", "date"):
        if not isinstance(v, str):
            return [name + "_type"]
        ok = _email_ok(v) if kind == "email" else _date_ok(v)
        return [] if ok else [name + "_format"]
    if kind == "tags":
        if not isinstance(v, list):
            return [name + "_type"]
        out = []
        if any(not isinstance(t, str) or t == "" for t in v):
            out.append(name + "_item")
        strings = [t for t in v if isinstance(t, str)]
        if len(set(strings)) != len(strings):
            out.append(name + "_duplicate")
        if len(v) > opts["max_items"]:
            out.append(name + "_too_many")
        return out
    raise AssertionError(kind)


def _errors(record):
    if not isinstance(record, dict):
        return ["not_an_object"]
    errors = []
    for name, kind, required, opts in SCHEMA:
        if name not in record:
            if required:
                errors.append(name + "_missing")
            continue
        errors += _check(name, kind, opts, record[name])
    if @UNKNOWN@ and any(key not in KNOWN for key in record):
        errors.append("unknown_field")
    return sorted(set(errors))
'''
API = {
    "is_valid": '''

def is_valid(record):
    """True if `record` satisfies the schema."""
    return not _errors(record)
''',
    "validate": '''

def validate(record):
    """The sorted, distinct error codes of `record` ([] when it is valid)."""
    return _errors(record)
''',
}

# Each variant: (old, new) replacements in ENGINE that make one plausible mistake.
VARIANTS = {
    "bool-is-int": [("    return isinstance(v, int) and not isinstance(v, bool)", "    return isinstance(v, int)")],
    "coerces-strings": [(
        '        if not _is_int(v):\n            return [name + "_type"]\n        return [] if opts["min"]',
        '        try:\n            v = int(v)\n        except (TypeError, ValueError):\n            return [name + "_type"]\n        return [] if opts["min"]',
    )],
    "range-off-by-one": [('opts["min"] <= v <= opts["max"]', 'opts["min"] < v < opts["max"]')],
    "re-match-prefix": [('if not re.fullmatch(opts["pattern"], v):', 'if not re.match(opts["pattern"], v):')],
    "dollar-newline": [('if not re.fullmatch(opts["pattern"], v):', 'if not re.match("^" + opts["pattern"] + "$", v):')],
    "stops-at-first-error": [("        errors += _check(name, kind, opts, record[name])\n", "        errors += _check(name, kind, opts, record[name])\n        if errors:\n            return errors[:1]\n")],
    "unsorted-codes": [("    return sorted(set(errors))\n", "    return errors\n")],
    "strptime-date": [(
        '    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", v):\n        return False\n    try:\n        datetime.date(int(v[:4]), int(v[5:7]), int(v[8:]))',
        '    try:\n        datetime.datetime.strptime(v, "%Y-%m-%d")',
    )],
    "fromisoformat-date": [(
        '    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", v):\n        return False\n    try:\n        datetime.date(int(v[:4]), int(v[5:7]), int(v[8:]))',
        "    try:\n        datetime.date.fromisoformat(v)",
    )],
    "unknown-fields-ignored": [("    if @UNKNOWN@ and any(key not in KNOWN for key in record):\n        errors.append(\"unknown_field\")\n", "")],
    "assumes-object": [('    if not isinstance(record, dict):\n        return ["not_an_object"]\n', "")],
}

DOMAINS = {
    "account": {
        "noun": "user account",
        "ident": ("username", "lowercase letters, digits and underscores, starting with a letter", "[a-z][a-z0-9_]*", 3, [12, 16]),
        "int": ("level", [0, 1], [50, 99]),
        "enum": ("role", [["admin", "editor", "viewer"], ["owner", "member", "guest"]]),
        "email": "email",
        "tags": ("interests", [3, 4]),
        "date": "joined",
    },
    "device": {
        "noun": "device inventory",
        "ident": ("hostname", "lowercase letters, digits and hyphens, starting with a letter", "[a-z][a-z0-9-]*", 2, [10, 15]),
        "int": ("cpus", [1], [64, 128]),
        "enum": ("env", [["prod", "staging", "dev"], ["live", "test", "lab"]]),
        "email": "owner",
        "tags": ("labels", [4, 5]),
        "date": "installed",
    },
}

INSTRUCTION = """Write {what} in `/app/validator.py` that checks a {noun} record (a JSON object, given to you as a Python dict) against the schema below. {returns}

Schema:
{fields}
Rules:
- A key that is present is never missing, even when its value is null; a null value is simply of the wrong type.
- Integers are Python `int` values. Booleans (`true`/`false`) are not integers, and neither are floats such as `7.0` or numeric strings such as `"7"`.
- Ranges and length limits include both ends.
{unknown}{not_object}
`/app/examples.json` lists some records with their expected results. Use only the Python standard library.
"""

STUB = {
    "is_valid": '"""Validate {noun} records."""\n\n\ndef is_valid(record):\n    """True if `record` satisfies the schema."""\n    raise NotImplementedError\n',
    "validate": '"""Validate {noun} records."""\n\n\ndef validate(record):\n    """The sorted, distinct error codes of `record` ([] when it is valid)."""\n    raise NotImplementedError\n',
}


def _schema(rng, dom, p):
    """[(name, kind, required, opts)] in the order the instruction lists them."""
    d = DOMAINS[dom]
    name, words, pattern, min_len, max_lens = d["ident"]
    fields = {
        "ident": (name, "ident", True, {"min_len": min_len, "max_len": rng.choice(max_lens), "pattern": pattern, "words": words}),
        "int": (d["int"][0], "int", p["int_required"], {"min": rng.choice(d["int"][1]), "max": rng.choice(d["int"][2])}),
        "enum": (d["enum"][0], "enum", True, {"values": rng.choice(d["enum"][1])}),
        "email": (d["email"], "email", True, {}),
        "tags": (d["tags"][0], "tags", False, {"max_items": rng.choice(d["tags"][1])}),
        "date": (d["date"], "date", True, {}),
    }
    kinds = list(p["kinds"])
    for _ in range(50):
        rng.shuffle(kinds)
        schema = [fields[k] for k in kinds]
        names = [f[0] for f in schema]
        if names != sorted(names):  # schema order differs from code order, so unsorted output is visible
            return schema
    raise Reject("schema order is alphabetical")


def _describe(field, api):
    name, kind, required, o = field
    req = "required" if required else "optional"
    if kind == "ident":
        what, suffixes = f"a string of {o['min_len']} to {o['max_len']} characters, made only of {o['words']}", ["type", "length", "format"]
        note = "a string can have both `{n}_length` and `{n}_format`"
    elif kind == "int":
        what, suffixes, note = f"an integer from {o['min']} to {o['max']}", ["type", "range"], ""
    elif kind == "enum":
        what = "one of " + ", ".join(f'`"{v}"`' for v in o["values"]) + " (exactly, case included)"
        suffixes, note = ["invalid"], "any other value, of any type, is `{n}_invalid`"
    elif kind == "email":
        what = (
            "an email address: a string with exactly one `@`; the part before it is non-empty and has no spaces, and the part after it is "
            "two or more non-empty labels separated by dots, each made only of lowercase letters, digits and hyphens"
        )
        suffixes, note = ["type", "format"], ""
    elif kind == "tags":
        what, suffixes = f"a list of at most {o['max_items']} non-empty strings with no duplicates", ["type", "item", "duplicate", "too_many"]
        note = (
            "`{n}_item` when some element is not a non-empty string, `{n}_duplicate` when two string elements are equal, "
            f"`{{n}}_too_many` when it has more than {o['max_items']} elements"
        )
    else:
        what = "a calendar date written `YYYY-MM-DD` (four-digit year, two-digit month and day) that exists, e.g. not February 30"
        suffixes, note = ["type", "format"], ""
    line = f"- `{name}` ({req}): {what}."
    if api == "validate":
        all_codes = ([f"{name}_missing"] if required else []) + [f"{name}_{s}" for s in suffixes]
        line += " Codes: " + ", ".join(f"`{c}`" for c in all_codes)
        if "type" in suffixes:
            line += f" (`{name}_type` when the value is not {'a list' if kind == 'tags' else 'an integer' if kind == 'int' else 'a string'}, and then no other code for it)"
        line += "." if not note else "; " + note.format(n=name) + "."
    return line + "\n"


def _valid_value(rng, field, boundary=None):
    name, kind, _, o = field
    if kind == "ident":
        n = {"min": o["min_len"], "max": o["max_len"]}.get(boundary, rng.randint(o["min_len"], min(o["max_len"], 9)))
        sep = "_" if "_" in o["pattern"] else "-"
        body = "".join(rng.choice("abcdefghkmnprstuvwxyz0123456789" + sep) for _ in range(n - 1))
        return rng.choice("abcdefghkmnprstuvwxyz") + body
    if kind == "int":
        return {"min": o["min"], "max": o["max"]}.get(boundary, rng.randint(o["min"] + 1, o["max"] - 1))
    if kind == "enum":
        return rng.choice(o["values"])
    if kind == "email":
        return f"{rng.choice(['ana', 'bo.li', 'c_ruiz', 'dee+ops', 'eli'])}@{rng.choice(['example', 'corp-mail', 'lab7'])}.{rng.choice(['com', 'org', 'co.uk'])}"
    if kind == "tags":
        n = o["max_items"] if boundary == "max" else rng.randint(0, o["max_items"] - 1)
        return rng.sample(["red", "blue", "ops", "beta", "gpu", "east", "west", "core"], n)
    if kind == "date":
        if boundary == "leap":
            return "2024-02-29"
        return f"{rng.randint(2019, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    raise AssertionError(kind)


def _bad_values(rng, field, unknown_rule):
    """[(label, value)] that break exactly this field's rules."""
    name, kind, _, o = field
    if kind == "ident":
        sep = "_" if "_" in o["pattern"] else "-"
        other = "-" if sep == "_" else "_"
        ok = _valid_value(rng, field)
        return [
            ("ident_number", rng.randint(100, 999)),
            ("ident_short", ok[: o["min_len"] - 1]),
            ("ident_long", (ok + "x" * 20)[: o["max_len"] + 1]),
            ("ident_digit_first", str(rng.randint(1, 9)) + ok[1:]),
            ("ident_bad_char", ok[:2] + other + ok[2:]),
            ("ident_newline", ok + "\n"),
            ("ident_upper", ok[0].upper() + ok[1:]),
            ("ident_long_and_bad", ("Q" + ok + "y" * 20)[: o["max_len"] + 2]),
        ]
    if kind == "int":
        mid = rng.randint(o["min"] + 1, o["max"] - 1)
        return [("int_true", True), ("int_string", str(mid)), ("int_float", float(mid)), ("int_null", None), ("int_below", o["min"] - 1), ("int_above", o["max"] + 1)]
    if kind == "enum":
        v = rng.choice(o["values"])
        return [("enum_case", v.capitalize()), ("enum_other", "root"), ("enum_number", 1), ("enum_null", None)]
    if kind == "email":
        return [("email_no_dot", "ana@example"), ("email_two_at", "ana@@example.com"), ("email_space", "ana lee@example.com"), ("email_no_local", "@example.com"), ("email_empty_label", "ana@example..com"), ("email_upper", "ana@Example.com"), ("email_number", 42)]
    if kind == "tags":
        m = o["max_items"]
        many = ["t%d" % i for i in range(m + 1)]
        return [("tags_string", "red,blue"), ("tags_empty_item", ["red", ""]), ("tags_number_item", ["red", 7]), ("tags_duplicate", ["red", "blue", "red"]), ("tags_too_many", many), ("tags_many_dupes", ["ops"] * (m + 1))]
    if kind == "date":
        return [("date_feb30", "2026-02-30"), ("date_unpadded", "2026-2-03"), ("date_basic", "20260203"), ("date_not_leap", "2025-02-29"), ("date_month13", "2026-13-01"), ("date_number", 20260203)]
    raise AssertionError(kind)


def _valid_record(rng, schema):
    rec = {}
    for f in schema:
        if f[2] or rng.random() < 0.6:
            rec[f[0]] = _valid_value(rng, f)
    return rec


def _cases(rng, schema, p):
    """[(label, input)] hidden inputs; the expected results come from the reference."""
    out = [("valid", _valid_record(rng, schema)) for _ in range(3)]
    full = {f[0]: _valid_value(rng, f) for f in schema}
    out.append(("valid_all_fields", full))
    for f in schema:
        for b in {"ident": ["min", "max"], "int": ["min", "max"], "tags": ["max"], "date": ["leap"]}.get(f[1], []):
            rec = _valid_record(rng, schema)
            rec[f[0]] = _valid_value(rng, f, b)
            out.append((f"boundary_{f[0]}_{b}", rec))
    if not p["unknown_rule"]:
        rec = _valid_record(rng, schema)
        rec["note"] = "extra keys are allowed"
        out.append(("extra_key_allowed", rec))
    for f in schema:
        for label, v in _bad_values(rng, f, p["unknown_rule"]):
            rec = _valid_record(rng, schema)
            rec[f[0]] = v
            out.append((label, rec))
        if f[2]:
            rec = _valid_record(rng, schema)
            rec.pop(f[0], None)
            out.append((f"missing_{f[0]}", rec))
    if p["multi"]:
        for i in range(3):
            rec = _valid_record(rng, schema)
            bad = rng.sample(schema, 3)
            for f in bad:
                rec[f[0]] = rng.choice(_bad_values(rng, f, p["unknown_rule"]))[1]
            out.append((f"multi_{i}", rec))
        # planted: two errors whose schema order is the reverse of their alphabetical order
        first, second = next((a, b) for i, a in enumerate(schema) for b in schema[i + 1 :] if a[0] > b[0])
        rec = _valid_record(rng, schema)
        rec[first[0]] = _bad_values(rng, first, False)[0][1]
        rec[second[0]] = _bad_values(rng, second, False)[0][1]
        out.append(("multi_order", rec))
    if p["unknown_rule"]:
        rec = _valid_record(rng, schema)
        rec["nickname"] = "zed"
        out.append(("unknown_key", rec))
        rec = _valid_record(rng, schema)
        rec[schema[0][0]] = _bad_values(rng, schema[0], True)[0][1]
        rec["Extra"] = 1
        out.append(("unknown_and_bad", rec))
    if p["not_object"]:
        out += [("not_object_list", [_valid_record(rng, schema)]), ("not_object_string", "username=ann"), ("not_object_null", None)]
    return out


def _source(schema, api, unknown_rule, variant=None):
    plain = [(n, k, r, {key: v for key, v in o.items() if key != "words"}) for n, k, r, o in schema]
    src = ENGINE
    for old, new in VARIANTS.get(variant, []):
        if old not in src:
            raise AssertionError(f"{variant}: {old!r}")
        src = src.replace(old, new)
    src = src.replace("@SCHEMA@", _pretty(plain)).replace("@UNKNOWN@", repr(unknown_rule))
    return src + API[api]


def _pretty(schema):
    return "[\n" + "".join(f"    {tuple(f)!r},\n" for f in schema) + "]"


def _exec(src):
    ns = {}
    exec(compile(src, "<validator>", "exec"), ns)  # noqa: S102 - our own reference/wrong sources
    return ns


def _hardcoded(api, examples):
    table = {json.dumps(e["record"], sort_keys=True): e["expected"] for e in examples}
    default = "True" if api == "is_valid" else "[]"
    return (
        "import json\n\n"
        f"EXPECTED = {json.dumps(table, indent=4, sort_keys=True)}\n\n\n"
        f"def {api}(record):\n"
        f"    return EXPECTED.get(json.dumps(record, sort_keys=True), {default})\n"
    )


def _write(source):
    return f"cat > /app/validator.py <<'PY'\n{source}PY\n"


def build(ctx):
    rng, p = ctx.rng, ctx.params
    api = p["api"]
    dom = rng.choice(sorted(DOMAINS))
    schema = _schema(rng, dom, p)
    ref = _exec(_source(schema, api, p["unknown_rule"]))[api]

    def expect(v):
        return ref(json.loads(json.dumps(v)))

    examples = []
    pool = [("valid", _valid_record(rng, schema)) for _ in range(p["examples"])] + _cases(rng, schema, p)
    picks = rng.sample(range(p["examples"], len(pool)), p["examples"] // 2)
    for i in list(range(p["examples"] - p["examples"] // 2)) + picks:
        examples.append({"record": pool[i][1], "expected": expect(pool[i][1])})
    shown = {json.dumps(e["record"], sort_keys=True) for e in examples}
    cases = [(label, v) for label, v in _cases(rng, schema, p) if json.dumps(v, sort_keys=True) not in shown]
    checks = tuple({"name": f"{label}_{i}", "func": api, "kind": "equal", "args": [v], "expected": expect(v)} for i, (label, v) in enumerate(cases))
    if api == "is_valid" and sum(c["expected"] is True for c in checks) < 3:
        raise Reject("too few valid hidden records")
    variants = [v for v in p["shortcuts"]]
    sources = {v: _source(schema, api, p["unknown_rule"], v) for v in variants}
    for v, src in sources.items():  # by construction every variant fails a planted record
        if all(run_checks(_exec(src), list(checks)).values()):
            raise Reject(f"no hidden record catches {v}")
    sources["hardcode-examples"] = _hardcoded(api, examples)
    oracle = _source(schema, api, p["unknown_rule"])
    d = DOMAINS[dom]
    fields_text = "".join(_describe(f, api) for f in schema)
    returns = (
        "`is_valid(record)` returns `True` if the record satisfies every rule and `False` otherwise."
        if api == "is_valid"
        else "`validate(record)` returns the list of error codes for the record: each code at most once, sorted alphabetically, and `[]` for a valid record. A field can produce several codes, and every field is checked (do not stop at the first error)."
    )
    instruction = INSTRUCTION.format(
        what="a function `is_valid(record)`" if api == "is_valid" else "a function `validate(record)`",
        noun=d["noun"],
        returns=returns,
        fields=fields_text,
        unknown="- Any key that is not in the schema adds the code `unknown_field` (once, however many such keys there are).\n" if p["unknown_rule"] else "- Keys that are not in the schema are allowed and ignored.\n",
        not_object="- If the record is not a JSON object (a dict), the result is `[\"not_an_object\"]` and nothing else.\n" if p["not_object"] else "",
    )
    return TaskSpec(
        instruction=instruction,
        files={"validator.py": STUB[api].format(noun=d["noun"]), "examples.json": json.dumps(examples, indent=2) + "\n"},
        grader=Checks("/app/validator.py", "validator", checks),
        oracle=Solution(_write(oracle), lambda f, s=oracle: {"/app/validator.py": s}),
        shortcuts={k: Solution(_write(v), lambda f, s=v: {"/app/validator.py": s}) for k, v in sorted(sources.items())},
        params={"domain": dom, "fields": [f[0] for f in schema], "api": api, "n_checks": len(checks)},
    )


FAMILY = Family(
    name="write-validator",
    version=1,
    cluster="write-code",
    category="coding",
    skills=("python", "reading-specs", "edge-cases", "validation"),
    difficulties={
        "easy": {
            "api": "is_valid", "kinds": ["ident", "int", "enum"], "int_required": True, "unknown_rule": False, "not_object": False, "multi": False,
            "examples": 8, "shortcuts": ["bool-is-int", "coerces-strings", "range-off-by-one", "re-match-prefix"],
        },
        "medium": {
            "api": "validate", "kinds": ["ident", "int", "enum", "email"], "int_required": False, "unknown_rule": False, "not_object": False, "multi": True,
            "examples": 6, "shortcuts": ["bool-is-int", "coerces-strings", "range-off-by-one", "re-match-prefix", "dollar-newline", "stops-at-first-error", "unsorted-codes"],
        },
        "hard": {
            "api": "validate", "kinds": ["ident", "int", "enum", "email", "tags", "date"], "int_required": False, "unknown_rule": True, "not_object": True, "multi": True,
            "examples": 4,
            "shortcuts": [
                "bool-is-int", "coerces-strings", "range-off-by-one", "re-match-prefix", "dollar-newline", "stops-at-first-error", "unsorted-codes",
                "strptime-date", "fromisoformat-date", "unknown-fields-ignored", "assumes-object",
            ],
        },
    },
    build=build,
)
