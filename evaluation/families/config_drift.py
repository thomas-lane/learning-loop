"""config-drift: which hosts' configuration dumps differ from the baseline, on which settings,
and with which values?

The instruction is the same at every difficulty: it says that every `.json` file under
`/app/configs/hosts/`, at any depth, is a host's dump, defines settings as leaf values, and
says that an object present on only one side is reported as the settings inside it. The
data per difficulty:

- easy: drifted hosts differ only in changed values; every dump is directly in `hosts/`;
- medium: also settings missing on the host and settings only the host has, always inside
  objects both sides have, so every difference is a single leaf;
- hard: also a number written as a string, an object (a section or subsection such as `db`
  or `db.pool`) the host lacks and a section only the host has, each reported as the
  settings inside it; drifted hosts sit in region subdirectories of `hosts/`.

Construction: a baseline settings tree (`/app/configs/baseline.json`, every setting nested
in a section) and one JSON dump per host. Every dump, the baseline included, is written with
its own key order and indentation (2 or 4 spaces, or on one line), because order and
whitespace do not count. Drifted hosts get 1-3 drifts each (an object drift gives one
difference per setting inside the object), and every kind of drift the level requires
appears on some host. The answer maps each drifted host to
{dotted setting: {"baseline", "host"}}, with null for a missing side. The grader compares
numbers by value (`1` equals `1.0`, both JSON numbers); a number and a string still differ
(`8080` against `"8080"`), as the instruction says.

Traps, by construction (each a declared shortcut where it applies): some hosts differ from
the baseline only in key order or whitespace, so comparing file text flags them (they would
appear with no differences; `textual`); every setting is nested, so comparing only
top-level keys reports sections instead of settings (`top-level`). Medium/hard: a host has
a setting the baseline lacks, so walking only the baseline's settings misses it
(`baseline-keys-only`). Hard: a port is written as "8080" on one host, so comparing values
as strings misses it (`str-compare`); reporting an object that exists on only one side as
one entry instead of its settings fails (`whole-object`); drifted hosts sit in region
subdirectories, so reading only `hosts/*.json` misses them (`top-dir-only`).

Every solution is one Python source (`SOLVER`) run with a mode, as in csv-revenue.
"""

import copy
import json

from learning_loop.tasks.spec import Family, ParsedAnswer, Reject, Solution, TaskSpec

# Section -> setting -> candidate values (nested dicts are subsections).
SECTIONS = {
    "server": {"host": ["0.0.0.0", "127.0.0.1"], "port": [8080, 8000, 9000, 8443], "workers": [2, 4, 8, 16], "keepalive_sec": [5, 15, 30, 60]},
    "db": {"host": ["db.internal", "pg-primary.internal", "pg.service.local"], "port": [5432, 6432], "pool": {"min": [1, 2, 4], "max": [10, 20, 40, 80]}, "timeout_ms": [1000, 3000, 5000, 10000]},
    "cache": {"enabled": [True, False], "backend": ["redis", "memcached"], "ttl_sec": [60, 300, 600, 3600], "redis": {"host": ["cache.internal", "redis.service.local"], "port": [6379, 6380]}},
    "logging": {"level": ["info", "warning", "debug", "error"], "format": ["json", "text"], "sample_rate": [0.1, 0.25, 0.5, 1.0]},
    "features": {"new_checkout": [True, False], "beta_search": [True, False], "dark_mode": [True, False], "bulk_export": [True, False]},
    "limits": {"max_body_kb": [512, 1024, 2048, 4096], "rate": {"per_minute": [60, 120, 600, 1200], "burst": [10, 20, 50]}},
    "tls": {"enabled": [True, False], "min_version": ["1.2", "1.3"], "cert_path": ["/etc/ssl/app.pem", "/etc/ssl/certs/site.pem"]},
    "metrics": {"enabled": [True, False], "port": [9100, 9102, 9200], "path": ["/metrics", "/internal/metrics"]},
}
# Settings a host may have that no baseline has: section -> setting -> candidate values.
EXTRA_LEAVES = {
    "server": {"trace_requests": [True, False]},
    "db": {"statement_cache": [128, 256, 512]},
    "cache": {"compress": [True, False]},
    "logging": {"verbose_sql": [True, False]},
    "features": {"instant_refund": [True, False]},
    "limits": {"max_header_kb": [8, 16, 32]},
    "tls": {"hsts": [True, False]},
    "metrics": {"histograms": [True, False]},
}
# Sections a host may have that the baseline lacks, besides the SECTIONS the baseline leaves out.
EXTRA_SECTIONS = {
    "debug": {"pprof": [True, False], "trace_sql": [True, False], "dump_dir": ["/tmp/debug", "/var/tmp/app-debug"]},
    "profiling": {"enabled": [True, False], "sample_hz": [10, 50, 100]},
}
REGIONS = ["eu", "us", "ap"]

INSTRUCTION = """We keep the intended configuration of our web hosts in `/app/configs/baseline.json`. Every `.json` file under `/app/configs/hosts/`, at any depth, is a JSON dump of one host's live configuration; a dump's file name without `.json` is the host name. The dumps come from different tools, so key order, indentation and other whitespace vary, and those differences do not matter.

Find every setting on which a host differs from the baseline. Settings are the leaf values (anything that is not a JSON object), named by their dotted path, e.g. `db.pool.max`. A host differs on a setting when its value differs (including its JSON type: `8080` and `"8080"` differ), when the host lacks a setting the baseline has, or when the host has a setting the baseline lacks. When an object (a section such as `db`, or a subsection such as `db.pool`) exists on only one side, every setting inside it is a separate difference: report those settings, not the object.

Write a JSON object to `/app/answer.json` that maps each host that differs to an object mapping each of its differing settings to `{"baseline": <baseline value>, "host": <host value>}`, with `null` on the side where the setting is missing (no setting is ever `null`). Leave out hosts that match the baseline. For example: `{"web-99": {"db.pool.max": {"baseline": 20, "host": 40}}}`.
"""

SOLVER = r'''
import json


def leaves(obj, prefix=""):
    out = {}
    for k, v in obj.items():
        path = prefix + k
        if isinstance(v, dict):
            out.update(leaves(v, path + "."))
        else:
            out[path] = v
    return out


def same(a, b, mode):
    if mode == "str-compare":
        return str(a) == str(b)
    return type(a) is type(b) and a == b


def objects_diff(b, h, prefix=""):
    """Like the oracle, but an object on only one side is one entry holding the object."""
    out = {}
    for k in sorted(set(b) | set(h)):
        path = prefix + k
        if isinstance(b.get(k), dict) and isinstance(h.get(k), dict):
            out.update(objects_diff(b[k], h[k], path + "."))
        elif k not in h or k not in b or not same(b[k], h[k], "oracle"):
            out[path] = {"baseline": b.get(k), "host": h.get(k)}
    return out


def diff(base, host, mode):
    if mode == "whole-object":
        return objects_diff(base, host)
    if mode == "top-level":
        b, h = base, host
    else:
        b, h = leaves(base), leaves(host)
    keys = sorted(b) if mode == "baseline-keys-only" else sorted(set(b) | set(h))
    out = {}
    for k in keys:
        if k not in h or k not in b or not same(b[k], h[k], mode):
            out[k] = {"baseline": b.get(k), "host": h.get(k)}
    return out


def answer(baseline_text, hosts, mode):
    """hosts: [(path relative to the hosts directory, text)]. Returns {host: {setting: {...}}}."""
    base = json.loads(baseline_text)
    out = {}
    for rel, text in sorted(hosts):
        if not rel.endswith(".json") or (mode == "top-dir-only" and "/" in rel):
            continue
        name = rel.rsplit("/", 1)[-1][: -len(".json")]
        d = diff(base, json.loads(text), mode)
        if d or (mode == "textual" and text != baseline_text):
            out[name] = d
    return out
'''

_SHELL = """python3 - <<'PY'
{solver}
import os

hosts = []
for d, _, names in os.walk("/app/configs/hosts"):
    for n in names:
        p = os.path.join(d, n)
        hosts.append((os.path.relpath(p, "/app/configs/hosts"), open(p).read()))
result = answer(open("/app/configs/baseline.json").read(), hosts, {mode!r})
with open("/app/answer.json", "w") as f:
    f.write(json.dumps(result, indent=2, sort_keys=True) + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _solution(mode):
    def model(files):
        hosts = [(rel[len("configs/hosts/"):], data.decode()) for rel, data in files.items() if rel.startswith("configs/hosts/")]
        result = _NS["answer"](files["configs/baseline.json"].decode(), hosts, mode)
        return {"/app/answer.json": json.dumps(result, indent=2, sort_keys=True) + "\n"}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def _draw(rng, spec):
    return {k: _draw(rng, v) if isinstance(v, dict) else rng.choice(v) for k, v in spec.items()}


def _shuffled(rng, obj):
    items = list(obj.items())
    rng.shuffle(items)
    return {k: _shuffled(rng, v) if isinstance(v, dict) else v for k, v in items}


def _dump(rng, obj, style):
    """`obj` as one dump tool would write it: own key order and whitespace."""
    obj = _shuffled(rng, obj) if style["shuffle"] else obj
    if style["indent"] is None:
        return json.dumps(obj, separators=style["separators"]) + "\n"
    return json.dumps(obj, indent=style["indent"]) + "\n"


def _styles(rng):
    return {"shuffle": rng.random() < 0.7, "indent": rng.choice([2, 4, None]), "separators": rng.choice([(",", ":"), (", ", ": ")])}


def _paths(obj, prefix=()):
    for k, v in obj.items():
        if isinstance(v, dict):
            yield from _paths(v, prefix + (k,))
        else:
            yield prefix + (k,)


def _get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _set(obj, path, value):
    for k in path[:-1]:
        obj = obj.setdefault(k, {})
    obj[path[-1]] = value


def _delete(obj, path):
    for k in path[:-1]:
        obj = obj[k]
    del obj[path[-1]]


def _choices(path):
    spec = SECTIONS
    for k in path:
        spec = spec[k]
    return spec


def _drift(rng, host, base, kind, used):
    """Apply one drift of `kind` to `host`; returns False when it cannot apply. `used` holds the
    setting paths earlier drifts of this host touched, which later ones leave alone."""
    paths = [x for x in _paths(base) if x not in used]
    if kind == "value":
        path = rng.choice(paths)
        old = _get(base, path)
        new = rng.choice([v for v in _choices(path) if not (type(v) is type(old) and v == old)])
        _set(host, path, new)
    elif kind == "missing":  # one setting; its object stays on both sides
        path = rng.choice(paths)
        _delete(host, path)
    elif kind == "extra":  # one setting, inside a section both sides have
        cands = [(s, k) for s in sorted(EXTRA_LEAVES) if s in base and s in host for k in sorted(EXTRA_LEAVES[s]) if (s, k) not in used]
        if not cands:
            return False
        path = rng.choice(cands)
        _set(host, path, rng.choice(EXTRA_LEAVES[path[0]][path[1]]))
    elif kind == "type":
        ports = [x for x in paths if x[-1] == "port"]
        if not ports:
            return False
        path = rng.choice(ports)
        _set(host, path, str(_get(base, path)))
    elif kind == "missing-object":  # a section or subsection none of whose settings drifted yet
        objs = sorted({x[:n] for x in _paths(base) for n in range(1, len(x))})
        objs = [o for o in objs if not any(u[: len(o)] == o for u in used)]
        if not objs:
            return False
        path = rng.choice(objs)
        _delete(host, path)
        used.update(x for x in _paths(base) if x[: len(path)] == path)
        return True
    elif kind == "extra-object":  # a whole section the baseline lacks
        specs = {**SECTIONS, **EXTRA_SECTIONS}
        section = rng.choice([s for s in sorted(specs) if s not in base and s not in host])
        host[section] = _draw(rng, specs[section])
        used.update((section,) + x for x in _paths(host[section]))
        return True
    else:
        raise ValueError(kind)
    used.add(path)
    return True


def build(ctx):
    rng, p = ctx.rng, ctx.params
    sections = rng.sample(sorted(SECTIONS), p["sections"])
    base = {s: _draw(rng, SECTIONS[s]) for s in sections}
    n = p["hosts"]
    prefix = rng.choice(["web", "app", "api"])
    names = [f"{prefix}-{i:02d}" for i in rng.sample(range(1, 40), n)]
    roles = ["drift"] * p["drifted"] + ["format"] * p["format_only"]
    roles += ["same"] * (n - len(roles))
    rng.shuffle(roles)
    required = list(p["required_kinds"])
    rng.shuffle(required)
    drifted = [i for i, r in enumerate(roles) if r == "drift"]
    plans = {i: [] for i in drifted}
    for j, kind in enumerate(required):  # every required kind appears on some drifted host
        plans[drifted[j % len(drifted)]].append(kind)
    for i in drifted:
        while len(plans[i]) < rng.randint(1, 3):
            plans[i].append(rng.choice(p["kinds"]))

    base_style = _styles(rng)
    files = {"configs/baseline.json": _dump(rng, base, base_style)}
    locations = {}
    for i, (name, role) in enumerate(zip(names, roles)):
        host = copy.deepcopy(base)
        if role == "drift":
            used = set()
            for kind in plans[i]:
                if not _drift(rng, host, base, kind, used):
                    raise Reject(f"cannot apply a {kind} drift")
        if role == "same":
            text = files["configs/baseline.json"]
        else:
            style = _styles(rng)
            text = _dump(rng, host, style)
            if text == files["configs/baseline.json"]:
                style["shuffle"], style["indent"] = True, 4 if style["indent"] != 4 else 2
                text = _dump(rng, host, style)
        if p["regions"]:
            region = rng.choice(REGIONS) if role == "drift" or rng.random() < 0.6 else None
        else:
            region = None
        loc = f"configs/hosts/{region}/{name}.json" if region else f"configs/hosts/{name}.json"
        files[loc] = text
        locations[name] = loc
    if p["regions"] and not any("/" in locations[names[i]][len("configs/hosts/"):] for i in drifted):
        raise Reject("no drifted host in a region subdirectory")
    if not any(roles[i] == "format" and files[locations[names[i]]] != files["configs/baseline.json"] for i in range(n)):
        raise Reject("no host that differs only in formatting")

    oracle = _solution("oracle")
    fbytes = {k: v.encode() for k, v in files.items()}
    expected = json.loads(oracle.model(fbytes)["/app/answer.json"])
    if sorted(expected) != sorted(names[i] for i in drifted):
        raise Reject("a drift cancelled out")
    shortcuts = {m: _solution(m) for m in ("textual", "top-level")}
    if "extra" in required:
        shortcuts["baseline-keys-only"] = _solution("baseline-keys-only")
    if "type" in required:
        shortcuts["str-compare"] = _solution("str-compare")
    if {"missing-object", "extra-object"} & set(required):
        shortcuts["whole-object"] = _solution("whole-object")
    if p["regions"]:
        shortcuts["top-dir-only"] = _solution("top-dir-only")
    return TaskSpec(
        instruction=INSTRUCTION,
        files=files,
        grader=ParsedAnswer("/app/answer.json", "json", expected, numbers="value"),
        oracle=oracle,
        shortcuts=shortcuts,
        params={"hosts": n, "drifted": sorted(expected), "sections": sections},
    )


FAMILY = Family(
    name="config-drift",
    version=2,
    cluster="diagnosis",
    category="data",
    skills=("json", "diffing", "python", "diagnosis"),
    difficulties={
        "easy": {"sections": 4, "hosts": 4, "drifted": 1, "format_only": 2, "kinds": ["value"], "required_kinds": ["value"], "regions": False},
        "medium": {"sections": 5, "hosts": 6, "drifted": 2, "format_only": 3, "kinds": ["value", "missing", "extra"], "required_kinds": ["value", "extra", "missing"], "regions": False},
        "hard": {"sections": 6, "hosts": 8, "drifted": 3, "format_only": 3, "kinds": ["value", "missing", "extra"], "required_kinds": ["type", "extra", "missing-object", "extra-object", "value"], "regions": True},
    },
    build=build,
)
