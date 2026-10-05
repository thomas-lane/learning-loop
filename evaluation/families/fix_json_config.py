"""fix-json-config: repair a hand-edited JSON config in place so that its validator accepts it,
keeping every value.

Task: `/app/config.json` has `//` comments, trailing commas and keys repeated within one
object (medium/hard add single-quoted strings, commented-out settings and nested duplicates;
hard adds `/* */` block comments and a whole duplicated object, and its instruction no longer
lists the kinds of damage). The visible `/app/validate.py` accepts only strict JSON without
duplicate keys that has the required settings with the right types. The instruction states the
rule for duplicates: the first occurrence is the one the service used. The grader parses the
file as JSON and compares it with the intended config: every setting, with first occurrences.

Construction: the config is a list of entries rendered by hand, so every defect is planted.
Every instance has URL strings containing `//`, a value followed by an inline `// ...`
comment on the same line, and at least one duplicated key whose later value differs.
Medium/hard also put an apostrophe inside a double-quoted string; hard adds `/*` inside a
string (a log glob) and a single-quoted string that contains double quotes.

Traps (declared shortcuts, all failing by construction):
- `last-wins`: a proper repair parsed with plain `json.loads`, which keeps the *last* duplicate;
- `drop-duplicates`: removes every occurrence of a duplicated key (validator-pleasing, lossy);
- `regex-strip`: `re.sub(r"//.*", "")` etc., which also cuts `https://...` strings, so the
  result does not parse and the file is left as it was;
- `grep-v-comments`: drops every line containing `//` before repairing, losing the values
  next to inline comments and the URL settings;
- medium/hard `quote-swap`: replaces every `'` with `"`, which breaks the apostrophe string.

Every solution, right or wrong, is one Python source (`SOLVER`) run with a mode, as in
csv-revenue: the shell form runs it on `/app/config.json`, the model on the generated text.
"""

import json

from learning_loop.tasks.spec import Family, ParsedAnswer, Solution, TaskSpec

SERVICES = ["billing", "orders", "inventory", "search", "notify", "ledger", "gateway", "catalog"]
ORIGIN_HOSTS = ["app", "admin", "partners", "status", "docs", "shop", "beta"]
TEAMS = ["platform", "payments", "growth", "infra", "core", "data"]
COMMENTS = [
    "keep in sync with the load balancer config",
    "TODO: move this to the secrets manager",
    "values below were tuned during the March incident",
    "ask #ops before changing",
    "staging uses the same layout",
    "do not commit real credentials here",
]
INLINE = {
    "port": ["behind the proxy", "default"],
    "workers": ["one per core", "bumped for the sale"],
    "pool_size": ["max connections", "per worker"],
    "timeout_s": ["seconds", "was 30 before"],
    "beta_ui": ["rollout flag", "off in production"],
    "rate_limit": ["requests per minute", "per client"],
    "cache_ttl_s": ["seconds", "tune with care"],
}
MOTDS = ["Don't forget to rotate the keys", "Maintenance window: Sunday's 02:00 slot", "We're migrating the database this week"]

INSTRUCTION = """Someone hand-edited our service config `/app/config.json` and now it no longer loads: it contains {defects}.

Repair the file in place so that `python3 /app/validate.py` prints `OK`, following these rules:

- Keep every setting with exactly the value it has now. Do not change any value's type; a single-quoted value is a string with the same text.
- Comments are not settings: remove them, including commented-out settings.
- Some keys appear more than once in the same object (left over from a bad merge). The service always used the **first** occurrence: keep that one and remove the later ones.
- Any valid JSON formatting is fine.
"""

INSTRUCTION_HARD = """`/app/config.json` was edited by hand and no longer loads. Repair it in place so that `python3 /app/validate.py` prints `OK`.

Keep every setting with exactly its current value and type (a single-quoted value is a string with the same text). Comments are not settings. Where a key appears more than once in the same object, the service used the first occurrence, so that is the value to keep. Any valid JSON formatting is fine.
"""

VALIDATOR = '''"""Pre-deploy check for /app/config.json: strict JSON, no duplicated keys, and the
required settings present with the right types. Prints OK or the problems found."""

import json
import sys

REQUIRED = {
    "service": str,
    "version": str,
    "server.host": str,
    "server.port": int,
    "server.base_url": str,
    "database.url": str,
    "database.pool_size": int,
    "logging.level": str,
}


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def main(path="/app/config.json"):
    try:
        with open(path) as f:
            cfg = json.load(f, object_pairs_hook=_no_duplicates)
    except ValueError as e:
        print(f"invalid JSON: {e}")
        return 1
    errors = []
    for dotted, typ in REQUIRED.items():
        node = cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                errors.append(f"missing: {dotted}")
                break
            node = node[part]
        else:
            if not isinstance(node, typ) or isinstance(node, bool):
                errors.append(f"wrong type: {dotted} should be {typ.__name__}")
    for e in errors:
        print(e)
    if errors:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
'''

SOLVER = r'''
import json
import re


def _first(pairs):
    out = {}
    for key, value in pairs:
        out.setdefault(key, value)
    return out


def _drop_duplicates(pairs):
    counts = {}
    for key, _ in pairs:
        counts[key] = counts.get(key, 0) + 1
    return {key: value for key, value in pairs if counts[key] == 1}


def strict(text):
    """Strict JSON text from hand-edited JSON: comments outside strings removed, single-quoted
    strings re-quoted, trailing commas before } and ] dropped."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            body = text[i + 1 : j]
            if c == "'":
                body = body.replace("\\'", "'")
                body = re.sub(r'(?<!\\)"', r'\\"', body)
            out.append('"' + body + '"')
            i = j + 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            i = text.index("*/", i + 2) + 2
            continue
        if c in "}]":
            k = len(out) - 1
            while k >= 0 and out[k].isspace():
                k -= 1
            if k >= 0 and out[k] == ",":
                del out[k]
        out.append(c)
        i += 1
    return "".join(out)


def fix(text, mode):
    """The repaired file text, or None when this method cannot produce one (the file is left as it is)."""
    try:
        if mode == "regex-strip":
            s = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
            s = re.sub(r"//.*", "", s)
            s = re.sub(r",(\s*[}\]])", r"\1", s)
            obj = json.loads(s.replace("'", '"'))
        else:
            if mode == "grep-v-comments":
                text = "".join(ln for ln in text.splitlines(keepends=True) if "//" not in ln)
            if mode == "quote-swap":
                text = text.replace("'", '"')
            hook = {"last-wins": None, "drop-duplicates": _drop_duplicates}.get(mode, _first)
            obj = json.loads(strict(text), object_pairs_hook=hook)
    except (ValueError, IndexError):
        return None
    return json.dumps(obj, indent=2) + "\n"
'''

_SHELL = """python3 - <<'PY'
{solver}
with open("/app/config.json") as f:
    fixed = fix(f.read(), {mode!r})
if fixed is not None:
    with open("/app/config.json", "w") as f:
        f.write(fixed)
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _solution(mode):
    def model(files):
        fixed = _NS["fix"](files["config.json"].decode(), mode)
        return {} if fixed is None else {"/app/config.json": fixed}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


# --------------------------------------------------------------------------- #
# The config as a list of entries, rendered by hand with planted defects
# --------------------------------------------------------------------------- #


def _e(key, value, **opts):
    return {"k": key, "v": value, **opts}


def _obj(entries):
    return {"obj": entries, "trailing": False}


def _arr(items):
    return {"arr": items, "trailing": False}


def _is_obj(v):
    return isinstance(v, dict) and "obj" in v


def _is_arr(v):
    return isinstance(v, dict) and "arr" in v


def _scalar(v, sq):
    return "'" + v + "'" if sq else json.dumps(v)


def _render_obj(o, ind):
    pad = "  " * (ind + 1)
    lines = ["{"]
    real = [i for i, e in enumerate(o["obj"]) if "commented" not in e]
    for i, e in enumerate(o["obj"]):
        for c in e.get("before", []):
            lines.append(f"{pad}// {c}")
        if e.get("block"):
            block = e["block"]
            lines.append(f"{pad}/* {block[0]}")
            lines += [f"{pad}   {b}" for b in block[1:]]
            lines[-1] += " */"
        if "commented" in e:
            lines.append(f"{pad}// {e['commented']}")
            continue
        v = e["v"]
        if _is_obj(v):
            vl = _render_obj(v, ind + 1)
        elif _is_arr(v):
            vl = ["[" + ", ".join(_scalar(x, False) for x in v["arr"]) + ("," if v["trailing"] else "") + "]"]
        else:
            vl = [_scalar(v, e.get("sq", False))]
        vl[-1] += "" if i == real[-1] and not o["trailing"] else ","
        if e.get("inline"):
            vl[-1] += f" // {e['inline']}"
        key = "'" + e["k"] + "'" if e.get("sqkey") else json.dumps(e["k"])
        lines.append(f"{pad}{key}: {vl[0]}")
        lines += vl[1:]
    lines.append("  " * ind + "}")
    return lines


def _truth(v):
    if _is_obj(v):
        out = {}
        for e in v["obj"]:
            if "commented" not in e and not e.get("dup"):
                out[e["k"]] = _truth(e["v"])
        return out
    if _is_arr(v):
        return list(v["arr"])
    return v


def _other(rng, v):
    """A different value of the same type, for a later duplicate."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + rng.choice([-1, 1]) * rng.randint(1, max(2, v // 2))
    if v.startswith("http") or v.startswith("postgres"):
        return v.replace(".internal", "-old.internal") if ".internal" in v else v.replace(".example.com", ".example.net")
    return v + "-old" if not v[-1].isdigit() else v[:-1] + str((int(v[-1]) + 1) % 10)


def _config(rng, p):
    name = rng.choice(SERVICES)
    server = _obj([
        _e("host", rng.choice(["0.0.0.0", "127.0.0.1"])),
        _e("port", rng.randint(8000, 8999)),
        _e("base_url", f"https://{name}.example.com/api/v{rng.randint(1, 3)}"),
        _e("workers", rng.randint(2, 16)),
    ])
    database = _obj([
        _e("url", f"postgres://{name}-db.internal:5432/{name}"),
        _e("pool_size", rng.randint(5, 40)),
        _e("timeout_s", rng.choice([5, 10, 15, 30, 60])),
    ])
    logging = _obj([_e("level", rng.choice(["debug", "info", "warning"])), _e("file", f"/var/log/{name}/app.log")])
    features = _obj([
        _e("beta_ui", rng.choice([True, False])),
        _e("rate_limit", rng.choice([50, 100, 200, 500])),
        _e("cache_ttl_s", rng.choice([30, 60, 300, 900])),
    ])
    origins = _arr([f"https://{h}.example.com" for h in rng.sample(ORIGIN_HOSTS, rng.randint(2, 3))])
    top = [
        _e("service", name),
        _e("version", f"{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}"),
        _e("server", server),
        _e("database", database),
        _e("logging", logging),
        _e("features", features),
        _e("allowed_origins", origins),
    ]
    if p["single_quotes"]:
        database["obj"].append(_e("replica_url", f"postgres://{name}-replica.internal:5432/{name}", sq=True))
        top.insert(2, _e("owner", f"team-{rng.choice(TEAMS)}", sq=True, sqkey=rng.random() < 0.5))
        top.insert(rng.randint(3, len(top)), _e("motd", rng.choice(MOTDS)))
        logging["obj"].insert(1, {"commented": f'"format": "{rng.choice(["json", "text"])}",'})
    if p["block_comments"]:
        logging["obj"].append(_e("glob", f"/var/log/{name}/*.log"))
        top.insert(1, _e("banner", f'Welcome to "{name.title()}"', sq=True))
        features["obj"][1]["block"] = ["Rate limiting was tuned after the March incident.", f"Do not raise above {rng.choice([800, 1000])} without asking ops."]
    return top, {"server": server, "database": database, "logging": logging, "features": features, "origins": origins}


def _plant(rng, top, objs, p):
    # trailing commas: at least one object and the origins list
    objs["origins"]["trailing"] = True
    for key in rng.sample(["server", "database", "logging", "features"], p["trailing_objects"]):
        objs[key]["trailing"] = True
    # full-line comments before a few entries
    scalars = [(o, e) for o in ("server", "database", "logging", "features") for e in objs[o]["obj"] if "commented" not in e]
    for _, e in rng.sample(scalars, 2):
        e.setdefault("before", []).append(rng.choice(COMMENTS))
    # inline comments after values (always at least one on a line that is not otherwise special)
    plain = [e for _, e in scalars if not isinstance(e["v"], str) and not e.get("inline")]
    for e in rng.sample(plain, p["inline_comments"]):
        e["inline"] = rng.choice(INLINE[e["k"]])
    # duplicated keys: a later entry in the same object with a different value
    dup_objs = rng.sample(["server", "database", "features"], p["duplicates"])
    for key in dup_objs:
        ents = objs[key]["obj"]
        cands = [i for i, e in enumerate(ents) if "commented" not in e and not _is_obj(e["v"]) and not _is_arr(e["v"]) and not e.get("sq")]
        i = rng.choice(cands)
        orig = ents[i]
        ents.insert(rng.randint(i + 1, len(ents)), _e(orig["k"], _other(rng, orig["v"]), dup=True))
    if p["top_duplicate"]:
        # the whole `logging` object again, later, with changed values
        later = _obj([_e("level", rng.choice(["error", "critical"])), _e("file", "/tmp/app.log")])
        top.insert(rng.randint(top.index(next(e for e in top if e["k"] == "logging")) + 1, len(top)), _e("logging", later, dup=True))


def build(ctx):
    rng, p = ctx.rng, ctx.params
    top, objs = _config(rng, p)
    _plant(rng, top, objs, p)
    root = _obj(top)
    text = "\n".join(_render_obj(root, 0)) + "\n"
    expected = _truth(root)
    shortcuts = {m: _solution(m) for m in ("last-wins", "drop-duplicates", "regex-strip", "grep-v-comments")}
    if p["single_quotes"]:
        shortcuts["quote-swap"] = _solution("quote-swap")
    defects = "comments, trailing commas, single-quoted strings and duplicated keys" if p["single_quotes"] else "comments, trailing commas and duplicated keys"
    return TaskSpec(
        instruction=INSTRUCTION_HARD if p["block_comments"] else INSTRUCTION.format(defects=defects),
        files={"config.json": text, "validate.py": VALIDATOR},
        grader=ParsedAnswer("/app/config.json", "json", expected),
        oracle=_solution("oracle"),
        shortcuts=shortcuts,
        params={k: p[k] for k in sorted(p)},
    )


FAMILY = Family(
    name="fix-json-config",
    version=1,
    cluster="config-repair",
    category="config",
    skills=("json", "parsing", "config-repair", "python"),
    difficulties={
        "easy": {"single_quotes": False, "block_comments": False, "trailing_objects": 1, "inline_comments": 1, "duplicates": 1, "top_duplicate": False},
        "medium": {"single_quotes": True, "block_comments": False, "trailing_objects": 2, "inline_comments": 2, "duplicates": 2, "top_duplicate": False},
        "hard": {"single_quotes": True, "block_comments": True, "trailing_objects": 3, "inline_comments": 3, "duplicates": 3, "top_duplicate": True},
    },
    build=build,
)
