"""The config-repair cluster (fix-json-config, resolve-conflicts, env-reconcile, cron-translate,
toml-migrate): deterministic rendering, clean randomness, and each family's answer and traps
re-derived here from the rendered files, independently of the families' own solution code."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.render import render
from learning_loop.tasks.runtime.grade import parse_crontab

MODULES = ["fix_json_config", "resolve_conflicts", "env_reconcile", "cron_translate", "toml_migrate"]
FAMILY_DIR = REPO_ROOT / "evaluation" / "families"


def _load(module: str):
    spec = importlib.util.spec_from_file_location(f"config_repair_{module}", FAMILY_DIR / f"{module}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAMILY


FAMS = {m: _load(m) for m in MODULES}
CASES = [(m, d) for m in MODULES for d in FAMS[m].difficulties]
SEEDS = [1, 2, 3]


def _key(d: Path) -> dict:
    return json.loads((d / "tests" / "key.json").read_text())


def _files(d: Path) -> dict[str, str]:
    root = d / "environment" / "files"
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- #
# fix-json-config
# --------------------------------------------------------------------------- #

_JSON_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'[^\'\n]*\'|//[^\n]*|/\*.*?\*/|\s+|.', re.S)


def _tokens(text):
    return [(m.start(), m.group()) for m in _JSON_TOKEN.finditer(text)]


def _relaxed(text, hook=None):
    out: list[str] = []
    for _, t in _tokens(text):
        if t.startswith("//") or t.startswith("/*"):
            continue
        if t.startswith("'"):
            t = json.dumps(t[1:-1])
        if t in "}]":
            while out and out[-1].isspace():
                out.pop()
            if out and out[-1] == ",":
                out.pop()
        out.append(t)
    return json.loads("".join(out), object_pairs_hook=hook)


def _first_wins(pairs):
    out = {}
    for k, v in pairs:
        out.setdefault(k, v)
    return out


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_fix_json_config_traps(tmp_path, difficulty, seed):
    render(FAMS["fix_json_config"], difficulty, seed, tmp_path / "t")
    files, expected = _files(tmp_path / "t"), _key(tmp_path / "t")["expected"]
    text = files["config.json"]
    toks = _tokens(text)
    assert _relaxed(text, _first_wins) == expected
    assert _relaxed(text) != expected  # json.loads keeps the last duplicate
    strings = [t for _, t in toks if t[0] in "\"'"]
    assert any("//" in s for s in strings)  # a `//` comment stripper that ignores strings breaks these
    before_comments = [text[text.rfind("\n", 0, pos) + 1 : pos] for pos, t in toks if t.startswith("//")]
    assert any(":" in b and b.strip() for b in before_comments)  # a value shares its line with a comment
    # the validator rejects the file as it is and accepts the expected config
    (tmp_path / "ok.json").write_text(json.dumps(expected))
    validate = tmp_path / "t" / "environment" / "files" / "validate.py"
    assert subprocess.run([sys.executable, str(validate), str(tmp_path / "ok.json")], capture_output=True, text=True).stdout.strip() == "OK"
    assert subprocess.run([sys.executable, str(validate), str(tmp_path / "t" / "environment" / "files" / "config.json")], capture_output=True).returncode == 1
    if difficulty != "easy":
        assert any(s.startswith("'") for s in strings)
        assert any(s.startswith('"') and "'" in s for s in strings)  # replacing every ' with " breaks this string
    if difficulty == "hard":
        assert any(t.startswith("/*") for _, t in toks)
        assert any("/*" in s for s in strings)
        assert any(s.startswith("'") and '"' in s for s in strings)


# --------------------------------------------------------------------------- #
# resolve-conflicts
# --------------------------------------------------------------------------- #

_BLOCK = re.compile(r"^<<<<<<< HEAD\n(.*?)(?:^\|\|\|\|\|\|\| [^\n]*\n.*?)?^=======\n(.*?)^>>>>>>> [^\n]*\n", re.S | re.M)


def _resolve(text, how):
    def repl(m):
        section = re.findall(r"^\[(\w+)\]$", text[: m.start()], re.M)[-1]
        ours, theirs = m.group(1).splitlines(), m.group(2).splitlines()
        rule = how if how in ("ours", "theirs") else {"release": "ours", "dependencies": "theirs", "allowed_hosts": "union"}[section]
        lines = ours if rule == "ours" else theirs if rule == "theirs" else ours + [t for t in theirs if t not in ours]
        return "".join(ln + "\n" for ln in lines)

    return _BLOCK.sub(repl, text)


def _tree_matches(key, resolved):
    if key["kind"] == "exact":
        (text,) = resolved.values()
        return text.strip() == key["expected"]
    return {rel: hashlib.sha256(t.encode()).hexdigest() for rel, t in resolved.items()} == {rel: e["sha256"] for rel, e in key["expected"].items()}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_resolve_conflicts_traps(tmp_path, difficulty, seed):
    render(FAMS["resolve_conflicts"], difficulty, seed, tmp_path)
    key = _key(tmp_path)
    files = {rel[len("config/") :]: t for rel, t in _files(tmp_path).items()}
    assert _tree_matches(key, {rel: _resolve(t, "rules") for rel, t in files.items()})
    assert not any("<<<<<<<" in _resolve(t, "rules") for t in files.values())
    assert not _tree_matches(key, {rel: _resolve(t, "ours") for rel, t in files.items()})
    assert not _tree_matches(key, {rel: _resolve(t, "theirs") for rel, t in files.items()})
    keep_both = {rel: "".join(ln for ln in t.splitlines(keepends=True) if not re.match(r"(<<<<<<<|=======|>>>>>>>)", ln)) for rel, t in files.items()}
    assert not _tree_matches(key, keep_both)
    hosts_blocks = [m for t in files.values() for m in _BLOCK.finditer(t) if re.findall(r"^\[(\w+)\]$", t[: m.start()], re.M)[-1] == "allowed_hosts"]
    assert any(set(m.group(1).splitlines()) & set(m.group(2).splitlines()) for m in hosts_blocks)  # concatenating both sides duplicates a host
    if difficulty != "easy":
        assert "=======" in files["README.md"].splitlines()
    if difficulty == "hard":
        blocks = [m for t in files.values() for m in _BLOCK.finditer(t)]
        assert blocks and all("|||||||" in m.group(0) for m in blocks)
        assert any("<<<<<<<" in t for rel, t in files.items() if rel.startswith("legacy/"))


# --------------------------------------------------------------------------- #
# env-reconcile
# --------------------------------------------------------------------------- #


def _dotenv(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        k, v = line.split("=", 1)
        out[k] = v
    return out


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_env_reconcile_traps(tmp_path, difficulty, seed):
    render(FAMS["env_reconcile"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path), _key(tmp_path)["expected"]
    tmpl, cur = _dotenv(files[".env.example"]), _dotenv(files[".env"])
    layer_names = ["deploy/overrides.env"] if "deploy/overrides.env" in files else ["deploy/common.env", "deploy/production.env"]
    layers = [_dotenv(files[n]) for n in layer_names]
    merged = {}
    for k in tmpl:
        merged[k] = next(v for v in [lay.get(k, "") for lay in reversed(layers)] + [cur.get(k, ""), tmpl[k]] if v != "")
    assert merged == expected
    over = {k: v for lay in layers for k, v in lay.items() if v != ""}
    assert any(k in tmpl and cur.get(k) and over.get(k) and cur[k] != over[k] for k in tmpl)  # current-beats-overrides fails
    assert any(k not in tmpl for k in cur)  # keeping unknown keys fails
    assert any(cur.get(k) == "" and tmpl[k] and k not in over for k in tmpl)  # treating KEY= as set fails
    assert any(expected[k] == cur.get(k) != tmpl[k] and k not in over for k in tmpl)  # ignoring the current file fails
    assert "=" in expected["DATABASE_URL"]  # awk -F= ... $2 cuts it
    if difficulty != "easy":
        assert any(v == "" and cur.get(k) for lay in layers for k, v in lay.items() if k in tmpl)  # an empty override must not clear
        assert any(k not in tmpl for lay in layers for k in lay)
    if difficulty == "hard":
        assert any(k in layers[0] and k in layers[1] and layers[0][k] != layers[1][k] for k in tmpl)
        exported = [ln.split()[1].split("=")[0] for ln in files[".env"].splitlines() if ln.startswith("export ")]
        assert any(expected[k] == cur[k] != tmpl[k] and k not in over for k in exported if k in tmpl)


# --------------------------------------------------------------------------- #
# cron-translate
# --------------------------------------------------------------------------- #

_DAY = {d: i for i, d in enumerate(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])}
_MONTH = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}
_T = r"(\d{1,2}):(\d{2})(?: (AM|PM))?"


def _hm(h, m, ampm):
    h = int(h)
    if ampm:
        h = h % 12 + (12 if ampm == "PM" else 0)
    return f"{int(m)} {h}"


def _names(text, table):
    return sorted(table[w.rstrip("s") if w.rstrip("s") in table else w] for w in re.split(r", | and ", text))


def _cron(s):
    """Five cron fields for one schedule phrase (an independent translation)."""
    if m := re.fullmatch(r"every (\d+) minutes", s):
        return f"*/{m[1]} * * * *"
    if s in ("every quarter hour", "every half hour"):
        return f"*/{15 if 'quarter' in s else 30} * * * *"
    if m := re.fullmatch(r"every (\d+) hours, on the hour", s):
        return f"0 */{m[1]} * * *"
    if m := re.fullmatch(r"every (\d+) minutes from (\d+):00 through (\d+):59, Monday through Friday", s):
        return f"*/{m[1]} {int(m[2])}-{int(m[3])} * * 1-5"
    if m := re.fullmatch(r"every day at " + _T, s):
        return f"{_hm(*m.groups())} * * *"
    if m := re.fullmatch(r"every (\w+day) at " + _T, s):
        return f"{_hm(*m.groups()[1:])} * * {_DAY[m[1]]}"
    if m := re.fullmatch(r"Monday through Friday at " + _T, s):
        return f"{_hm(*m.groups())} * * 1-5"
    if m := re.fullmatch(r"on the (\d+)\w\w of every month at " + _T, s):
        return f"{_hm(*m.groups()[1:])} {m[1]} * *"
    if m := re.fullmatch(r"on the (\d+)\w\w and the (\d+)\w\w of every month at " + _T, s):
        return f"{_hm(*m.groups()[2:])} {m[1]},{m[2]} * *"
    if m := re.fullmatch(r"at " + _T + r" on the 1st of (.+)", s):
        return f"{_hm(*m.groups()[:3])} 1 {','.join(map(str, _names(m[4], _MONTH)))} *"
    if m := re.fullmatch(r"on (.+) at " + _T, s):
        return f"{_hm(*m.groups()[1:])} * * {','.join(map(str, _names(m[1], _DAY)))}"
    raise AssertionError(f"unrecognized schedule {s!r}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_cron_translate_traps(tmp_path, difficulty, seed):
    render(FAMS["cron_translate"], difficulty, seed, tmp_path)
    text, expected = _files(tmp_path)["jobs.txt"], _key(tmp_path)["expected"]
    jobs = [dict(ln.split(": ", 1) for ln in block.splitlines()) for block in text.split("\n\n")[1:]]
    active = [j for j in jobs if j.get("status", "active") == "active"]
    lines = [f"{_cron(j['schedule'])} {j['command']}" for j in active]
    assert parse_crontab("\n".join(lines)) == expected
    # an equivalent spelling is graded the same: steps expanded to lists, weekday numbers as names
    names = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
    respelled = []
    for ln in lines:
        f = ln.split(None, 5)
        f[0] = ",".join(str(v) for v in range(0, 60, int(f[0][2:]))) if f[0].startswith("*/") else f[0]
        f[4] = ",".join(names[int(d)] for d in f[4].split(",")) if re.fullmatch(r"\d(,\d)*", f[4]) else f[4]
        respelled.append(" ".join(f))
    assert parse_crontab("\n".join(respelled)) == expected
    fields = [ln.split()[:5] for ln in lines]
    assert any("0" in f[4].replace("-", ",").split(",") for f in fields)  # Sunday = 1 numbering fails
    assert any(f[0].startswith("*/") for f in fields)  # every N minutes written as one minute fails
    assert any(f[2] != "*" and f[4] == "*" for f in fields)  # swapping day-of-month and day-of-week fails
    if difficulty != "easy":
        assert any(re.search(r"\d:\d\d PM", j["schedule"]) and not re.search(r"\b12:\d\d PM", j["schedule"]) for j in active)
    if difficulty == "hard":
        assert len(active) < len(jobs)
        assert any(re.search(r"\b12:\d\d AM", j["schedule"]) for j in active)


# --------------------------------------------------------------------------- #
# toml-migrate
# --------------------------------------------------------------------------- #


def _factor(before, after):
    """The multiplier a changelog's unit notes describe, e.g. "(seconds)" -> "(milliseconds)"."""
    if (before, after) == ("seconds", "milliseconds"):
        return 1000
    for text in (before, after):
        if m := re.search(r"= (\d+) ", text or ""):
            return int(m[1])
    raise AssertionError((before, after))


def _section(changelog, title):
    return changelog.split(f"## {title}")[1].split("\n## ")[0]


def _migrate(cfg, section):
    """Apply the changelog bullets of one schema section (schema 1 names) to `cfg`."""
    moves = []
    for line in section.splitlines():
        if m := re.match(r"- `\[(\w+)\] (\w+)` (?:is|will be) renamed to `(\w+)`", line):
            t = cfg[m[1]]
            t[m[3]] = t.pop(m[2])
        elif m := re.match(r"- `\[\[?(\w+)\]?\] (\w+)`(?: \(([^)]*)\))? is replaced by `(\w+)` \(([^)]*)\)", line):
            factor = _factor(m[3], m[5])
            for t in cfg[m[1]] if isinstance(cfg[m[1]], list) else [cfg[m[1]]]:
                t[m[4]] = round(t.pop(m[2]) * factor)
        elif m := re.match(r"- `\[(\w+)\] (\w+)` (?:is|will be) removed", line):
            del cfg[m[1]][m[2]]
        elif m := re.match(r"- `\[(\w+)\]` moves to `\[([\w.]+)\]`", line):
            moves.append((m[1], m[2]))
        elif m := re.match(r"- `schema_version` is now `(\d+)`", line):
            cfg["schema_version"] = int(m[1])
    for old, new in moves:
        *parents, leaf = new.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = cfg.pop(old)
    return cfg


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_toml_migrate_traps(tmp_path, difficulty, seed):
    render(FAMS["toml_migrate"], difficulty, seed, tmp_path)
    files, expected = _files(tmp_path), _key(tmp_path)["expected"]
    v1, changelog = tomllib.loads(files["config.toml"]), files["CHANGELOG.md"]
    schema2 = _section(changelog, "Config schema 2")
    assert _migrate(tomllib.loads(files["config.toml"]), schema2) == expected
    assert "is removed" in schema2 and "moves to" in schema2 and "is replaced by" in schema2
    if difficulty != "easy":
        converted = re.findall(r"- `\[(\w+)\] (\w+)` \(seconds\)", schema2)
        assert any(isinstance(v1[t][k], float) for t, k in converted)  # v * 1000 stays a float
    if difficulty == "hard":
        assert len(v1["workers"]) >= 2 and "[[workers]] interval" in schema2
        removed = re.findall(r"- `\[(\w+)\] (\w+)` is removed", schema2)
        assert any(k in other for t, k in removed for name, other in v1.items() if name != t and isinstance(other, dict))
        planned = _section(changelog, "Config schema 3")
        assert _migrate(tomllib.loads(files["config.toml"]), schema2 + planned) != expected
