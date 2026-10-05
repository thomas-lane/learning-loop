"""fix-parser: fix the bugs injected into a small INI reader/writer (partial credit).

`/app/iniconf.py` holds 4 / 5 / 6 functions (value unquoting, the parser, boolean and list
getters; medium adds merging two configs, hard adds writing INI text). The module docstring
specifies the format; 1 / 2 / 4 functions carry a bug from a catalog of classic parser
mistakes (inline-comment stripping, sloppy unquoting, case folding, duplicate detection,
`;` comments, case-sensitive booleans, unstripped or empty list items, shallow merges,
missing quotes on output). Errors are specified: malformed input raises the module's
`ParseError`, invalid booleans `ValueError` and missing sections or keys `KeyError`. The
visible tests in `/app/test_iniconf.py` expose all / half / a quarter of the bugs; a function
with a bug no visible test exposes gets no visible test. Hidden checks use fresh
seed-specific configs that always contain each feature the format defines (both comment
characters, indented comments, mixed-case keys, section names and values, `=` and `#`
inside values, quoted values with inner spaces, a value ending in a quote, the same key in
two sections) and every specified error.

Traps, each declared as a shortcut that must fail:
- `visible-bugs-only` (when some bug has no visible test): fix only the bugs the visible
  tests expose;
- `hardcode-visible`: return the visible tests' expected values for their exact inputs and
  keep the buggy code for everything else;
- `wrong-fix-<function>`: every bug fixed correctly except one, which gets a plausible wrong
  fix from the catalog (e.g. lower-casing the whole line instead of the key, tracking
  duplicate keys across sections, `value.strip('"')` as unquoting, copying only the outer
  dict when merging).

Expected values come from the data each config is generated from, not from the module's
code, and `build` checks that the correct module passes every hidden check. `build` redraws
a visible test until it exposes its function's bug (or, for a function without a visible
bug, passes with the hidden-only bugs still in place).
"""

import json

from learning_loop.tasks.runtime.grade import run_checks
from learning_loop.tasks.spec import Checks, Family, Reject, Solution, TaskSpec

MODULE = "iniconf"
PATH = f"/app/{MODULE}.py"

HEADER = '''"""A small INI reader and writer.

The format `parse` accepts:

- Each line is stripped of surrounding whitespace before it is interpreted.
- Blank lines are ignored, and so is a comment: a line whose first character is `#` or `;`.
  Anywhere else `#` and `;` are ordinary characters (there are no inline comments).
- `[name]` starts the section `name` (the text between the brackets, stripped). Section
  names are case-sensitive.
- `key = value` adds `key` to the current section. The line is split at its first `=`; the
  key is stripped and lower-cased, and the rest is read with `parse_value`.
- Any other line, a key before the first section, a section name that appears twice, or a
  key that appears twice in the same section is an error: `parse` raises `ParseError`.

A config is a dict {section name: {key: value}} of strings, in file order.
"""


class ParseError(ValueError):
    """Malformed INI text."""
'''

CORRECT = {
    "parse_value": '''def parse_value(raw: str) -> str:
    """The value of a `key = value` line, from the text after the `=`: `raw` without
    surrounding whitespace. If that text is at least two characters long and starts and ends
    with a double quote, those two quotes are removed and the text between them is kept
    exactly, including its whitespace."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value
''',
    "parse": '''def parse(text: str) -> dict[str, dict[str, str]]:
    """Parse INI `text` (see the module docstring) into a config."""
    config: dict[str, dict[str, str]] = {}
    section = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line[0] == "[" and line[-1] == "]":
            name = line[1:-1].strip()
            if name in config:
                raise ParseError(f"line {lineno}: duplicate section {name!r}")
            section = config[name] = {}
        elif "=" in line:
            if section is None:
                raise ParseError(f"line {lineno}: key outside a section")
            key, value = line.split("=", 1)
            key = key.strip().lower()
            if key in section:
                raise ParseError(f"line {lineno}: duplicate key {key!r}")
            section[key] = parse_value(value)
        else:
            raise ParseError(f"line {lineno}: cannot parse {line!r}")
    return config
''',
    "get_bool": '''def get_bool(config: dict, section: str, key: str) -> bool:
    """`config[section][key]` as a bool: 1, yes, true and on are True; 0, no, false and off
    are False, in any letter case. Raises ValueError for any other value, and KeyError when
    the section or the key is missing."""
    value = config[section][key].lower()
    if value in ("1", "yes", "true", "on"):
        return True
    if value in ("0", "no", "false", "off"):
        return False
    raise ValueError(f"not a boolean: {value!r}")
''',
    "get_list": '''def get_list(config: dict, section: str, key: str) -> list[str]:
    """`config[section][key]` split at commas into a list of items, each stripped of
    surrounding whitespace; empty items are dropped. A key missing from the section gives
    []; a missing section raises KeyError."""
    value = config[section].get(key, "")
    return [item.strip() for item in value.split(",") if item.strip()]
''',
    "merge": '''def merge(base: dict, override: dict) -> dict:
    """A new config with every section of `base` and of `override`. A section in both keeps
    the keys of both, and for a key in both the value from `override` wins. Neither argument
    is modified."""
    merged = {name: dict(keys) for name, keys in base.items()}
    for name, keys in override.items():
        merged.setdefault(name, {}).update(keys)
    return merged
''',
    "dumps": '''def dumps(config: dict) -> str:
    """INI text for `config`: for each section in order, a `[name]` line followed by one
    `key = value` line per key, in order, with one blank line between sections. A value with
    leading or trailing whitespace is written in double quotes; other values are written as
    they are. The text ends with a single newline; an empty config gives ""."""
    blocks = []
    for name, keys in config.items():
        lines = [f"[{name}]"]
        for key, value in keys.items():
            if value != value.strip():
                value = f'"{value}"'
            lines.append(f"{key} = {value}")
        blocks.append("\\n".join(lines) + "\\n")
    return "\\n".join(blocks)
''',
}

_UNQUOTE = "    if len(value) >= 2 and value[0] == '\"' and value[-1] == '\"':\n        return value[1:-1]\n    return value\n"
_DUP_KEY = "            if key in section:\n                raise ParseError(f\"line {lineno}: duplicate key {key!r}\")\n"

# func -> bug -> {"bug": swaps applied to CORRECT[func], "naive": swaps for a plausible wrong fix, or None}
BUGS = {
    "parse_value": {
        "cuts_at_hash": {
            "bug": [("value = raw.strip()", 'value = raw.split("#")[0].strip()')],
            "naive": [("value = raw.strip()", 'value = raw.split(" #")[0].strip()')],
        },
        "no_unquoting": {
            "bug": [(_UNQUOTE, "    return value\n")],
            "naive": [(_UNQUOTE, "    return value.strip('\"')\n")],
        },
    },
    "parse": {
        "keys_keep_case": {
            "bug": [("key = key.strip().lower()", "key = key.strip()")],
            "naive": [("key = key.strip().lower()", "key = key.strip()"), ("        line = raw.strip()\n", "        line = raw.strip().lower()\n")],
        },
        "duplicate_key_overwrites": {
            "bug": [(_DUP_KEY, "")],
            "naive": [
                ("    section = None\n", "    section = None\n    seen = set()\n"),
                (_DUP_KEY, "            if key in seen:\n                raise ParseError(f\"line {lineno}: duplicate key {key!r}\")\n            seen.add(key)\n"),
            ],
        },
        "hash_comments_only": {
            "bug": [('if not line or line[0] in "#;":', 'if not line or line[0] == "#":')],
            "naive": None,
        },
    },
    "get_bool": {
        "case_sensitive": {
            "bug": [("value = config[section][key].lower()", "value = config[section][key]")],
            "naive": [
                ("value = config[section][key].lower()", "value = config[section][key]"),
                ('if value in ("1", "yes", "true", "on"):', 'if value in ("1", "yes", "true", "on", "Yes", "True", "On"):'),
                ('if value in ("0", "no", "false", "off"):', 'if value in ("0", "no", "false", "off", "No", "False", "Off"):'),
            ],
        },
        "anything_else_false": {
            "bug": [('    if value in ("0", "no", "false", "off"):\n        return False\n    raise ValueError(f"not a boolean: {value!r}")\n', "    return False\n")],
            "naive": None,
        },
    },
    "get_list": {
        "items_not_stripped": {
            "bug": [("return [item.strip() for item in value.split(\",\") if item.strip()]", "return [item for item in value.split(\",\") if item]")],
            "naive": [("return [item.strip() for item in value.split(\",\") if item.strip()]", "return [item for item in value.split(\", \") if item]")],
        },
        "keeps_empty_items": {
            "bug": [("return [item.strip() for item in value.split(\",\") if item.strip()]", "return [item.strip() for item in value.split(\",\")]")],
            "naive": [
                (
                    "return [item.strip() for item in value.split(\",\") if item.strip()]",
                    "if not value:\n        return []\n    return [item.strip() for item in value.split(\",\")]",
                )
            ],
        },
    },
    "merge": {
        "shallow": {
            "bug": [
                (
                    "    merged = {name: dict(keys) for name, keys in base.items()}\n    for name, keys in override.items():\n        merged.setdefault(name, {}).update(keys)\n    return merged\n",
                    "    return {**base, **override}\n",
                )
            ],
            "naive": [("merged = {name: dict(keys) for name, keys in base.items()}", "merged = dict(base)")],
        },
        "base_wins": {
            "bug": [("        merged.setdefault(name, {}).update(keys)\n", "        for key, value in keys.items():\n            merged.setdefault(name, {}).setdefault(key, value)\n")],
            "naive": None,
        },
    },
    "dumps": {
        "never_quotes": {
            "bug": [("            if value != value.strip():\n                value = f'\"{value}\"'\n", "")],
            "naive": [("if value != value.strip():", 'if " " in value:')],
        },
        "blank_after_every_section": {
            "bug": [('    return "\\n".join(blocks)\n', '    return "".join(block + "\\n" for block in blocks)\n')],
            "naive": [('    return "\\n".join(blocks)\n', '    return "".join(block + "\\n" for block in blocks).rstrip("\\n") + "\\n"\n')],
        },
    },
}


# --------------------------------------------------------------------------- #
# Generated configs: the data first, then its INI text (so expectations do not come from
# the module's code)
# --------------------------------------------------------------------------- #

SECTIONS = ["server", "database", "cache", "logging", "auth", "mail", "storage", "metrics", "queue", "search"]
KEYS = ["host", "port", "timeout", "user", "path", "level", "retries", "mode", "region", "endpoint", "name", "limit"]
WORDS = ["alpha", "bravo", "delta", "omega", "north", "local", "primary", "backup", "fast", "blue"]


def _cap(rng, word):
    """`word` spelled with some capitals: "Host", "HOST" or "hostName"-style."""
    i = rng.randint(1, len(word) - 1)
    return rng.choice([word.capitalize(), word.upper(), word[:i] + word[i].upper() + word[i + 1 :]])


def _value_parts(rng, kind):
    """(text after '=' as written, parsed value) for one value of `kind`."""
    w = rng.choice(WORDS)
    n = rng.randint(2, 999)
    if kind == "plain":
        v = rng.choice([w, str(n), f"{w}-{n}", f"/srv/{w}/{n}"])
        return v, v
    if kind == "mixed":
        v = f"{_cap(rng, w)}{n}"
        return v, v
    if kind == "equals":
        v = rng.choice([f"user={w};level={n}", f"a={n}", f"{w}=={n}"])
        return v, v
    if kind == "hash":
        v = rng.choice([f"{w} # {n}", f"color #{n:03d}", f"{w}#{n}; {w}"])
        return v, v
    if kind == "quoted":
        inner = rng.choice([f"  {w} {n}  ", f" {w}", f"{w} # {n} "])
        return f'"{inner}"', inner
    if kind == "inch":
        v = f'{n}"'
        return v, v
    raise KeyError(kind)


def _render(rng, sections, *, upper_keys, comments, indent_comments):
    """INI text for [(section, [(key, raw value)])], with random spacing around `=`. With the
    flags, the text always has a `#` and a `;` comment line, an indented comment and a key
    spelled with capitals (more of each at random)."""
    lines = []
    for i, (name, items) in enumerate(sections):
        if comments and (i < 2 or rng.random() < 0.5):
            lines.append(("#;"[i] if i < 2 else rng.choice("#;")) + f" {rng.choice(WORDS)} settings")
        if i:
            lines.append("")
        lines.append(f"[{name}]" if rng.random() < 0.7 else f"[ {name} ]")
        for j, (key, raw) in enumerate(items):
            k = _cap(rng, key) if upper_keys and ((i, j) == (0, 0) or rng.random() < 0.5) else key
            sep = rng.choice(["=", " = ", " =", "= ", "  =  "])
            lines.append(f"{k}{sep}{raw}")
            if indent_comments and ((i, j) == (0, 0) or rng.random() < 0.25):
                lines.append(rng.choice(["  # ", "\t; "]) + f"old {key}")
    return "\n".join(lines) + "\n"


def _config(rng, kinds, *, n_sections, mixed_sections, shared_key):
    """([(section, [(key, raw)])], expected config): each value kind in `kinds` appears at least once."""
    names = rng.sample(SECTIONS, n_sections)
    if mixed_sections:
        names[0] = _cap(rng, names[0])
    sections, expected = [], {}
    pool = list(kinds) + [rng.choice(["plain", "plain", "mixed"]) for _ in range(rng.randint(1, 3) * n_sections)]
    rng.shuffle(pool)
    per = [[] for _ in names]
    for i, kind in enumerate(pool):
        per[i % n_sections].append(kind)
    common = rng.choice(KEYS)
    for idx, (name, ks) in enumerate(zip(names, per)):
        keys = rng.sample([k for k in KEYS if k != common], len(ks))
        if shared_key and idx < 2:
            keys[0] = common  # the same key in two sections is fine
        items, out = [], {}
        for key, kind in zip(keys, ks):
            raw, parsed = _value_parts(rng, kind)
            items.append((key, raw))
            out[key] = parsed
        sections.append((name, items))
        expected[name] = out
    return sections, expected


def _rich_text(rng, kinds, **flags):
    sections, expected = _config(rng, kinds, n_sections=rng.randint(2, 3), mixed_sections=flags.get("mixed_sections", False), shared_key=flags.get("shared_key", False))
    text = _render(rng, sections, upper_keys=flags.get("upper_keys", False), comments=flags.get("comments", False), indent_comments=flags.get("indent_comments", False))
    return text, expected


def _eq(name, func, args, expected):
    return {"name": name, "func": func, "kind": "equal", "args": args, "expected": expected}


def _raises(name, func, args, exception):
    return {"name": name, "func": func, "kind": "raises", "args": args, "exception": exception}


def _simple_config(rng, n=2):
    names = rng.sample(SECTIONS, n)
    return {name: {k: rng.choice(WORDS) for k in rng.sample(KEYS, rng.randint(2, 4))} for name in names}


def _bool_word(rng, truth, style):
    word = rng.choice(["yes", "true", "on"] if truth else ["no", "false", "off"])
    if style == "upper":
        return word.upper()
    if style == "mixed":
        return rng.choice([w for w in (word.upper()[0] + word[1:-1] + word[-1].upper(), word[0] + word[1:].upper()) if w != word.capitalize()])
    return word


def _dumps_ref(config):
    out = []
    for name, keys in config.items():
        block = f"[{name}]\n" + "".join(f"{k} = " + (f'"{v}"' if v != v.strip() else v) + "\n" for k, v in keys.items())
        out.append(block)
    return "\n".join(out)


def _cases(rng, func):
    """The hidden checks for `func`, with fresh inputs; the visible tests draw from the same kinds."""
    w, n = rng.choice(WORDS), rng.randint(2, 999)
    if func == "parse_value":
        return [
            _eq("parse_value_hash_inside", func, [f"  {w} # {n}  "], f"{w} # {n}"),
            _eq("parse_value_hash_no_space", func, [f" {w}#{n}"], f"{w}#{n}"),
            _eq("parse_value_quoted", func, [f' "  {w} {n} " '], f"  {w} {n} "),
            _eq("parse_value_ends_with_quote", func, [f" {n}\" "], f'{n}"'),
            _eq("parse_value_unbalanced_quote", func, [f' "{w} {n}'], f'"{w} {n}'),
        ]
    if func == "parse":
        text, expected = _rich_text(
            rng, ["equals", "hash", "quoted", "inch", "mixed"], mixed_sections=True, shared_key=True, upper_keys=True, comments=True, indent_comments=True
        )
        cfg = _simple_config(rng, 2)
        (s1, k1), (_, k2) = [(s, list(v)) for s, v in cfg.items()]
        base = _render(rng, [(s, list(v.items())) for s, v in cfg.items()], upper_keys=False, comments=False, indent_comments=False)
        lines = base.splitlines()
        dup_key = base + f"{k2[0].upper()} = {w}\n"  # the last section already has this key
        dup_section = base + f"\n[{s1}]\nextra = {n}\n"
        orphan = f"{k1[0]} = {w}\n" + base
        garbage = "\n".join(lines[:2] + [f"{w} {n}"] + lines[2:]) + "\n"
        return [
            _eq("parse_full", func, [text], expected),
            _raises("parse_duplicate_key", func, [dup_key], "ParseError"),
            _raises("parse_duplicate_section", func, [dup_section], "ParseError"),
            _raises("parse_key_before_section", func, [orphan], "ParseError"),
            _raises("parse_unparseable_line", func, [garbage], "ParseError"),
        ]
    if func == "get_bool":
        cfg = {"flags": {}}
        for i, (truth, style) in enumerate([(True, "upper"), (False, "mixed"), (True, "lower"), (False, "upper")]):
            cfg["flags"][f"opt{i}"] = _bool_word(rng, truth, style)
        cfg["flags"]["one"], cfg["flags"]["zero"] = "1", "0"
        cfg["flags"]["bad"] = rng.choice(["maybe", "2", "y", "enabled", "none"])
        return [
            _eq("get_bool_upper_true", func, [cfg, "flags", "opt0"], True),
            _eq("get_bool_mixed_false", func, [cfg, "flags", "opt1"], False),
            _eq("get_bool_upper_false", func, [cfg, "flags", "opt3"], False),
            _eq("get_bool_digit_one", func, [cfg, "flags", "one"], True),
            _eq("get_bool_digit_zero", func, [cfg, "flags", "zero"], False),
            _raises("get_bool_invalid", func, [cfg, "flags", "bad"], "ValueError"),
            _raises("get_bool_missing_key", func, [cfg, "flags", "absent"], "KeyError"),
        ]
    if func == "get_list":
        items = rng.sample(WORDS, rng.randint(3, 5))
        seps = [rng.choice([",", ", ", " ,", " ,  ", ",\t"]) for _ in items[1:]]
        seps[0] = rng.choice([",", " ,"])  # a separator without a following space
        seps[-1] = rng.choice([", ", " ,  ", ",\t"])  # and one with surrounding whitespace
        spaced = items[0] + "".join(s + it for s, it in zip(seps, items[1:]))
        gaps = rng.sample(WORDS, 3)
        gappy = f"{gaps[0]},,{gaps[1]}, ,{gaps[2]},"
        cfg = {"app": {"hosts": spaced, "tags": gappy, "empty": ""}}
        return [
            _eq("get_list_spacing", func, [cfg, "app", "hosts"], items),
            _eq("get_list_empty_items", func, [cfg, "app", "tags"], gaps),
            _eq("get_list_empty_value", func, [cfg, "app", "empty"], []),
            _eq("get_list_missing_key", func, [cfg, "app", "absent"], []),
            _raises("get_list_missing_section", func, [cfg, "nope", "hosts"], "KeyError"),
        ]
    if func == "merge":
        base = _simple_config(rng, 3)
        shared = list(base)[rng.randrange(3)]
        old_key = next(iter(base[shared]))
        new_key = rng.choice([k for k in KEYS if k not in base[shared]])
        extra = rng.choice([s for s in SECTIONS if s not in base])
        override = {shared: {old_key: f"{w}-{n}", new_key: str(n)}, extra: {"enabled": "yes"}}
        expected = {name: dict(keys) for name, keys in base.items()}
        expected[shared].update(override[shared])
        expected[extra] = dict(override[extra])
        return [
            _eq("merge_overlap", func, [base, override], expected),
            {"name": "merge_base_unchanged", "func": func, "kind": "no_mutation", "args": [base, override]},
        ]
    if func == "dumps":
        # dict arguments reach the function in sorted key order (see _canonical), so the
        # config is built in that order
        names = sorted(rng.sample(SECTIONS, 3))
        cfg = {name: {k: rng.choice(WORDS) for k in sorted(rng.sample(KEYS, 3))} for name in names}
        k0, k1 = list(cfg[names[0]])[:2]
        cfg[names[0]][k0] = f"{w} {n}"  # inner space only: written as is
        cfg[names[0]][k1] = rng.choice([f"  {w}", f"{w} ", f" {w} {n} "])  # surrounding space: quoted
        return [
            _eq("dumps_config", func, [cfg], _dumps_ref(cfg)),
            _eq("dumps_one_section", func, [{names[1]: cfg[names[1]]}], _dumps_ref({names[1]: cfg[names[1]]})),
            _eq("dumps_empty", func, [{}], ""),
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


INSTRUCTION = """Our services read their settings with the small INI module `/app/iniconf.py` ({functions}), and some settings come out wrong. {hint}

Fix `iniconf.py` so that every function behaves exactly as the module docstring and its own docstring describe, including the errors they raise (`ParseError`, `ValueError`, `KeyError`). Keep the function names, signatures and the `ParseError` class, and use only the Python standard library. You may add tests to `/app/test_iniconf.py` (run it with `python3 test_iniconf.py`), but do not change what the existing tests assert.
"""

HINTS = {
    "easy": "One function has a bug, and a test in `/app/test_iniconf.py` fails because of it.",
    "medium": "Some tests in `/app/test_iniconf.py` fail, and some bugs may not be covered by those tests at all.",
    "hard": "Several functions have bugs; the tests in `/app/test_iniconf.py` catch only a few of them.",
}


def build(ctx):
    return build_fix(ctx, INSTRUCTION, extra_imports=("ParseError",))


FAMILY = Family(
    name="fix-parser",
    version=1,
    cluster="fix-code",
    category="debugging",
    skills=("python", "debugging", "parsing", "reading-docs"),
    difficulties={
        "easy": {"functions": ["parse_value", "parse", "get_bool", "get_list"], "n_bugs": 1, "visible_bug_tests": 1.0, "hint": HINTS["easy"]},
        "medium": {"functions": ["parse_value", "parse", "get_bool", "get_list", "merge"], "n_bugs": 2, "visible_bug_tests": 0.5, "hint": HINTS["medium"]},
        "hard": {"functions": ["parse_value", "parse", "get_bool", "get_list", "merge", "dumps"], "n_bugs": 4, "visible_bug_tests": 0.25, "hint": HINTS["hard"]},
    },
    build=build,
)
