"""write-validator: write a validator for a schema stated in the instruction.

The agent fills in `validate(record) -> list[str]` in /app/validator.py: the sorted, distinct
error codes of a record (`[]` when it is valid). The instruction is the same at every
difficulty: a schema of six fields from one of two domains (an account or a device), with
seed-drawn bounds, lengths, allowed values and whether the integer is required, and one rule
text (missing vs null, what counts as an integer, inclusive bounds, `unknown_field`,
`not_an_object`). Field kinds: an identifier string (length and character rules), an integer
range, an enum, an email address, a tag list and a calendar date. /app/examples.json shows
records with their expected codes.

Grading: `Checks` (kind `equal`) on hidden records: valid ones (including every boundary: the
range ends, the length limits, a leap day), records that break one rule of one field, records
with several errors, and the trap records of the difficulty. No hidden record equals an
example.

Traps (each a full validator with one mistake, declared as a shortcut where the difficulty
plants it, and each caught by the hidden records planted for it, `TRAPS`): `hardcode-examples` (answers the example records
only), `bool-is-int` (`isinstance(v, int)` accepts true; the integer range includes 1),
`coerces-strings` (`int(v)` accepts "7"), `range-off-by-one` (exclusive bounds; records at
both range ends), `re-match-prefix` (`re.match` without an end anchor accepts "ab-cd"),
`dollar-newline` (`^...$` accepts a trailing newline), `stops-at-first-error` (records with
three errors), `unsorted-codes` (two errors whose schema order is the reverse of their
alphabetical order), `unknown-fields-ignored`, `assumes-object` (non-object inputs),
`strptime-date` (accepts "2026-2-03") and `fromisoformat-date` (accepts "20260203"). The
records with several errors use only plain bad values (`PLAIN_BAD`), never another trap's, so
each checks what it is planted for.

Difficulty is in the data, not the instruction: easy's hidden records leave out the two
date-library traps and its examples show one record for every trap it plants; medium also
plants `fromisoformat-date` and its examples show half of its traps; hard plants the same
traps as medium and its examples show none. `strptime-date` is planted at no difficulty
(`unplanted`). Each step also has fewer plain examples (valid records and plain errors).
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


def validate(record):
    """The sorted, distinct error codes of `record` ([] when it is valid)."""
    if not isinstance(record, dict):
        return ["not_an_object"]
    errors = []
    for name, kind, required, opts in SCHEMA:
        if name not in record:
            if required:
                errors.append(name + "_missing")
            continue
        errors += _check(name, kind, opts, record[name])
    if any(key not in KNOWN for key in record):
        errors.append("unknown_field")
    return sorted(set(errors))
'''

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
    "unknown-fields-ignored": [("    if any(key not in KNOWN for key in record):\n        errors.append(\"unknown_field\")\n", "")],
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

INSTRUCTION = """Write a function `validate(record)` in `/app/validator.py` that checks a {noun} record (a JSON object, given to you as a Python dict) against the schema below. It returns the list of error codes for the record: each code at most once, sorted alphabetically, and `[]` for a valid record. A field can produce several codes, and every field is checked.

Schema:
{fields}
Rules:
- A key that is present is never missing, even when its value is null; a null value is simply of the wrong type.
- An integer is a Python `int` that is not a `bool`: `True` and `False` are not integers here. Floats such as `7.0` and numeric strings such as `"7"` are not integers either.
- Ranges and length limits include both ends.
- Any key that is not in the schema adds the code `unknown_field` (once, however many such keys there are).
- If the record is not a JSON object (a dict), the result is `["not_an_object"]` and nothing else.

`/app/examples.json` lists some records with their expected results. Put all of your code in `/app/validator.py`: it is tested by importing that file on its own, from a directory that holds no other file. Use only the Python standard library.
"""

STUB = '"""Validate {noun} records."""\n\n\ndef validate(record):\n    """The sorted, distinct error codes of `record` ([] when it is valid)."""\n    raise NotImplementedError\n'

# Shortcut -> the labels of the hidden records planted to catch it.
TRAPS = {
    "bool-is-int": ("int_true",),
    "coerces-strings": ("int_string",),
    "range-off-by-one": ("boundary_int_min", "boundary_int_max"),
    "re-match-prefix": ("ident_bad_char",),
    "dollar-newline": ("ident_newline",),
    "stops-at-first-error": ("multi",),
    "unsorted-codes": ("multi_order",),
    "unknown-fields-ignored": ("unknown_key", "unknown_and_bad"),
    "assumes-object": ("not_object_list", "not_object_string", "not_object_null"),
    "strptime-date": ("date_unpadded",),
    "fromisoformat-date": ("date_basic",),
}
TRAP_LABELS = {label for labels in TRAPS.values() for label in labels}

# Per field kind, the bad values that break a plain rule and set off no trap (bool, numeric
# string, float, prefix match, trailing newline, date-library leniency).
PLAIN_BAD = {
    "ident": ("ident_number", "ident_short", "ident_long", "ident_digit_first", "ident_upper"),
    "int": ("int_null", "int_below", "int_above"),
    "enum": ("enum_case", "enum_other", "enum_number", "enum_null"),
    "email": ("email_no_dot", "email_two_at", "email_space", "email_no_local", "email_empty_label", "email_upper", "email_number"),
    "tags": ("tags_string", "tags_empty_item", "tags_number_item", "tags_duplicate", "tags_too_many"),
    "date": ("date_feb30", "date_not_leap", "date_month13", "date_number"),
}
KINDS = ["ident", "int", "enum", "email", "tags", "date"]


def _schema(rng, dom):
    """[(name, kind, required, opts)] in the order the instruction lists them."""
    d = DOMAINS[dom]
    name, words, pattern, min_len, max_lens = d["ident"]
    fields = {
        "ident": (name, "ident", True, {"min_len": min_len, "max_len": rng.choice(max_lens), "pattern": pattern, "words": words}),
        "int": (d["int"][0], "int", rng.choice([True, False]), {"min": rng.choice(d["int"][1]), "max": rng.choice(d["int"][2])}),
        "enum": (d["enum"][0], "enum", True, {"values": rng.choice(d["enum"][1])}),
        "email": (d["email"], "email", True, {}),
        "tags": (d["tags"][0], "tags", False, {"max_items": rng.choice(d["tags"][1])}),
        "date": (d["date"], "date", True, {}),
    }
    kinds = list(KINDS)
    rng.shuffle(kinds)
    schema = [fields[k] for k in kinds]
    if [f[0] for f in schema] == sorted(f[0] for f in schema):
        raise Reject("schema order is alphabetical, so unsorted codes could not be told apart")
    return schema


def _describe(field):
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
    all_codes = ([f"{name}_missing"] if required else []) + [f"{name}_{s}" for s in suffixes]
    line = f"- `{name}` ({req}): {what}. Codes: " + ", ".join(f"`{c}`" for c in all_codes)
    if "type" in suffixes:
        line += f" (`{name}_type` when the value is not {'a list' if kind == 'tags' else 'an integer' if kind == 'int' else 'a string'}, and then no other code for it)"
    return line + ("." if not note else "; " + note.format(n=name) + ".") + "\n"


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


def _bad_values(rng, field):
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


def _plain_bad(rng, field):
    return dict(_bad_values(rng, field))[rng.choice(PLAIN_BAD[field[1]])]


def _cases(rng, schema):
    """[(label, input)] for every kind of hidden record; expected results come from the reference."""
    out = [("valid", _valid_record(rng, schema)) for _ in range(3)]
    out.append(("valid_all_fields", {f[0]: _valid_value(rng, f) for f in schema}))
    for f in schema:
        for b in {"ident": ["min", "max"], "int": ["min", "max"], "tags": ["max"], "date": ["leap"]}.get(f[1], []):
            rec = _valid_record(rng, schema)
            rec[f[0]] = _valid_value(rng, f, b)
            out.append((f"boundary_{f[1]}_{b}", rec))
    for f in schema:
        for label, v in _bad_values(rng, f):
            rec = _valid_record(rng, schema)
            rec[f[0]] = v
            out.append((label, rec))
        if f[2]:
            rec = _valid_record(rng, schema)
            rec.pop(f[0], None)
            out.append((f"missing_{f[1]}", rec))
    for _ in range(3):
        rec = _valid_record(rng, schema)
        for f in rng.sample(schema, 3):
            rec[f[0]] = _plain_bad(rng, f)
        out.append(("multi", rec))
    # two errors whose schema order is the reverse of their alphabetical order
    first, second = next((a, b) for i, a in enumerate(schema) for b in schema[i + 1 :] if a[0] > b[0])
    rec = _valid_record(rng, schema)
    rec[first[0]], rec[second[0]] = _plain_bad(rng, first), _plain_bad(rng, second)
    out.append(("multi_order", rec))
    rec = _valid_record(rng, schema)
    rec["nickname"] = "zed"
    out.append(("unknown_key", rec))
    rec = _valid_record(rng, schema)
    rec[schema[0][0]] = _plain_bad(rng, schema[0])
    rec["Extra"] = 1
    out.append(("unknown_and_bad", rec))
    out += [("not_object_list", [_valid_record(rng, schema)]), ("not_object_string", "username=ann"), ("not_object_null", None)]
    return out


def _source(schema, variant=None):
    plain = [(n, k, r, {key: v for key, v in o.items() if key != "words"}) for n, k, r, o in schema]
    src = ENGINE
    for old, new in VARIANTS.get(variant, []):
        if old not in src:
            raise AssertionError(f"{variant}: {old!r}")
        src = src.replace(old, new)
    return src.replace("@SCHEMA@", _pretty(plain))


def _pretty(schema):
    return "[\n" + "".join(f"    {tuple(f)!r},\n" for f in schema) + "]"


def _exec(src):
    ns = {}
    exec(compile(src, "<validator>", "exec"), ns)  # noqa: S102 - our own reference/wrong sources
    return ns


def _hardcoded(examples):
    table = {json.dumps(e["record"], sort_keys=True): e["expected"] for e in examples}
    return (
        "import json\n\n"
        f"EXPECTED = {json.dumps(table, indent=4, sort_keys=True)}\n\n\n"
        "def validate(record):\n"
        "    return EXPECTED.get(json.dumps(record, sort_keys=True), [])\n"
    )


def _write(source):
    return f"cat > /app/validator.py <<'PY'\n{source}PY\n"


def _key(v):
    return json.dumps(v, sort_keys=True)


def _examples(rng, schema, p, planted):
    """[(label, record)] for examples.json: plain examples (valid records first, then plain
    errors), and one record for each trap the difficulty shows, all from a draw of their own."""
    pool = _cases(rng, schema)
    valid = [v for label, v in pool if label.startswith("valid")]
    plain_errors = [(label, v) for label, v in pool if not label.startswith(("valid", "boundary_")) and label not in TRAP_LABELS]
    n_errors = p["plain_examples"] // 2
    out = [("valid", v) for v in valid[: p["plain_examples"] - n_errors]] + rng.sample(plain_errors, n_errors)
    shown = sorted(planted) if p["show_traps"] == "all" else sorted(rng.sample(sorted(planted), len(planted) // 2)) if p["show_traps"] == "half" else []
    for trap in shown:
        label = rng.choice(TRAPS[trap])
        out.append((label, rng.choice([v for lb, v in pool if lb == label])))
    rng.shuffle(out)
    return out, shown


def build(ctx):
    rng, p = ctx.rng, ctx.params
    dom = rng.choice(sorted(DOMAINS))
    schema = _schema(rng, dom)
    oracle = _source(schema)
    ref = _exec(oracle)["validate"]

    def expect(v):
        return ref(json.loads(json.dumps(v)))

    planted = [t for t in TRAPS if t not in p["unplanted"]]
    dropped = {label for t in p["unplanted"] for label in TRAPS[t]}
    shown_records, shown_traps = _examples(rng, schema, p, planted)
    examples = [{"record": v, "expected": expect(v)} for _, v in shown_records]
    shown = {_key(e["record"]) for e in examples}
    cases = [(label, v) for label, v in _cases(rng, schema) if label not in dropped and _key(v) not in shown]
    checks = tuple({"name": f"{label}_{i}", "func": "validate", "kind": "equal", "args": [v], "expected": expect(v)} for i, (label, v) in enumerate(cases))
    sources = {t: _source(schema, t) for t in planted}
    for t, src in sources.items():  # each trap's planted records catch it unless an example took them all
        if all(run_checks(_exec(src), list(checks)).values()):
            raise Reject(f"no hidden record catches {t}")
    sources["hardcode-examples"] = _hardcoded(examples)
    d = DOMAINS[dom]
    instruction = INSTRUCTION.format(noun=d["noun"], fields="".join(_describe(f) for f in schema))
    return TaskSpec(
        instruction=instruction,
        files={"validator.py": STUB.format(noun=d["noun"]), "examples.json": json.dumps(examples, indent=2) + "\n"},
        grader=Checks("/app/validator.py", "validator", checks),
        oracle=Solution(_write(oracle), lambda f, s=oracle: {"/app/validator.py": s}),
        shortcuts={k: Solution(_write(v), lambda f, s=v: {"/app/validator.py": s}) for k, v in sorted(sources.items())},
        params={"domain": dom, "fields": [f[0] for f in schema], "planted": planted, "shown_traps": shown_traps, "n_checks": len(checks)},
    )


FAMILY = Family(
    name="write-validator",
    version=4,
    cluster="write-code",
    category="coding",
    skills=("python", "reading-specs", "edge-cases", "validation"),
    difficulties={
        "easy": {"unplanted": ["strptime-date", "fromisoformat-date"], "show_traps": "all", "plain_examples": 6},
        "medium": {"unplanted": ["strptime-date"], "show_traps": "half", "plain_examples": 5},
        "hard": {"unplanted": ["strptime-date"], "show_traps": "none", "plain_examples": 4},
    },
    build=build,
)
