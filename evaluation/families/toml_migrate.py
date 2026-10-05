"""toml-migrate: migrate a TOML config from schema 1 to schema 2 by following a changelog.

Task: `/app/config.toml` is a schema 1 config; `/app/CHANGELOG.md` lists the schema 2 changes
(renamed keys, tables moved under a new parent, unit changes such as seconds to
milliseconds, removed keys), in schema 1 names. The agent rewrites the file in place. Python's
`tomllib` can read TOML but the image has no TOML writer, so the agent writes TOML text itself;
the structure is plain tables, an array of tables and scalar/array values, all writable by
hand. The grader parses the file as TOML and compares it with the expected structure.

Construction: each instance draws its changes from a pool: easy one of each kind; medium
two of each, with a fractional seconds value (2.5 s must become the integer 2500); hard adds
`[[workers]]` (several entries, each with a seconds value to convert), a removed key whose
name also exists in another table (where it stays) and a "planned, not released" schema 3
section that must not be applied.

Traps (declared shortcuts, all failing by construction):
- `no-unit-conversion`: renames the converted keys but keeps their old numbers;
- `keep-removed`: keeps the removed keys;
- `partial-move`: moves all but the last moved table (easy: none);
- medium/hard `float-ms`: multiplies a fractional seconds value without making it an integer;
- hard `first-worker-only`: converts only the first `[[workers]]` entry;
- hard `remove-everywhere`: deletes the removed key from every table (`sed -i '/^debug = /d'`);
- hard `apply-planned`: also applies the unreleased schema 3 changes.

Every solution is one Python source (`SOLVER`: apply a list of changes, then write TOML)
run with a mode and the instance's change list; the model runs the same source.
"""

import tomllib

from learning_loop.tasks.spec import Family, ParsedAnswer, Solution, TaskSpec

SERVICES = ["billing", "orders", "search", "notify", "ledger", "gateway", "catalog", "reports"]
WORKERS = ["email", "webhooks", "exports", "thumbnails", "reindex", "billing-sync"]

SOLVER = r'''
import copy
import json


def _tables(cfg, name, mode):
    t = cfg[name]
    if isinstance(t, list):
        return t[:1] if mode == "first-worker-only" else t
    return [t]


def migrate(cfg, changes, planned, mode):
    """Apply `changes` (schema 1 names) to the parsed config: key changes first, then table moves."""
    cfg = copy.deepcopy(cfg)
    ops = changes + (planned if mode == "apply-planned" else [])
    for op in ops:
        kind = op["op"]
        if kind == "set":
            cfg[op["key"]] = op["value"]
        elif kind == "rename":
            for t in _tables(cfg, op["table"], mode):
                t[op["new"]] = t.pop(op["key"])
        elif kind == "convert":
            for t in _tables(cfg, op["table"], mode):
                v = t.pop(op["key"])
                if mode == "no-unit-conversion":
                    t[op["new"]] = v
                elif mode == "float-ms":
                    t[op["new"]] = v * op["factor"]
                else:
                    t[op["new"]] = int(round(v * op["factor"]))
        elif kind == "remove":
            if mode == "keep-removed":
                continue
            if mode == "remove-everywhere":
                for t in cfg.values():
                    for d in (t if isinstance(t, list) else [t]):
                        if isinstance(d, dict):
                            d.pop(op["key"], None)
            else:
                for t in _tables(cfg, op["table"], mode):
                    t.pop(op["key"])
    moves = [op for op in ops if op["op"] == "move"]
    if mode == "partial-move":
        moves = moves[:-1]
    for op in moves:
        table = cfg.pop(op["table"])
        parent = cfg
        parts = op["to"].split(".")
        for p in parts[:-1]:
            parent = parent.setdefault(p, {})
        parent[parts[-1]] = table
    return cfg


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise TypeError(type(v).__name__)


def _is_array_of_tables(v):
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def dump(cfg):
    """TOML text for nested tables, arrays of tables and scalar or array values."""
    lines = []

    def table(path, t):
        plain = [(k, v) for k, v in t.items() if not isinstance(v, dict) and not _is_array_of_tables(v)]
        if path and plain:
            lines.append("[" + path + "]")
        for k, v in plain:
            lines.append(k + " = " + _value(v))
        if plain:
            lines.append("")
        for k, v in t.items():
            sub = path + "." + k if path else k
            if isinstance(v, dict):
                table(sub, v)
            elif _is_array_of_tables(v):
                for item in v:
                    lines.append("[[" + sub + "]]")
                    for ik, iv in item.items():
                        lines.append(ik + " = " + _value(iv))
                    lines.append("")

    table("", cfg)
    return "\n".join(lines).rstrip("\n") + "\n"
'''

_SHELL = """python3 - <<'PY'
{solver}
import tomllib

with open("/app/config.toml", "rb") as f:
    cfg = tomllib.load(f)
out = migrate(cfg, {changes!r}, {planned!r}, {mode!r})
with open("/app/config.toml", "w") as f:
    f.write(dump(out))
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _solution(mode, changes, planned):
    def model(files):
        cfg = tomllib.loads(files["config.toml"].decode())
        return {"/app/config.toml": _NS["dump"](_NS["migrate"](cfg, changes, planned, mode))}

    return Solution(_SHELL.format(solver=SOLVER, changes=changes, planned=planned, mode=mode), model)


# --------------------------------------------------------------------------- #
# The schema 1 config and the pool of changes
# --------------------------------------------------------------------------- #


def _config(rng, p):
    """[(table, [(key, value, comment)])]; table "" is the top level, "workers" an array of tables."""
    name = rng.choice(SERVICES)
    secs = [1.5, 2.5, 7.5, 0.5] if p["fractional"] else [None]
    frac = rng.choice(secs)
    tables = [
        ("", [("schema_version", 1, None), ("service", name, None)]),
        ("server", [
            ("host", rng.choice(["0.0.0.0", "127.0.0.1"]), None),
            ("listen_port", rng.randint(8000, 8999), None),
            ("request_timeout", rng.choice([10, 15, 30, 45, 60]), "seconds"),
            ("max_body_kb", rng.choice([256, 512, 1024, 2048]), "kilobytes"),
            ("debug", rng.choice([True, False]), None),
        ]),
        ("database", [
            ("url", f"postgres://{name}-db.internal:5432/{name}", None),
            ("pool", rng.randint(5, 40), None),
            ("connect_timeout", frac if frac is not None else rng.choice([3, 5, 10]), "seconds"),
        ]),
        ("cache", [
            ("backend", rng.choice(["redis", "memcached"]), None),
            ("ttl", rng.choice([60, 300, 600, 3600]), "seconds"),
            ("servers", [f"cache-{i}.internal:6379" for i in range(1, rng.randint(2, 3) + 1)], None),
        ]),
        ("logging", [
            ("level", rng.choice(["debug", "info", "warning"]), None),
            ("colorize", rng.choice([True, False]), None),
            ("retention_days", rng.choice([3, 7, 14, 30]), "days"),
            ("debug", rng.choice([True, False]), "log the raw requests"),
        ]),
    ]
    if p["workers"]:
        for w in rng.sample(WORKERS, p["workers"]):
            tables.append(("workers", [("name", w, None), ("interval", rng.choice([15, 30, 60, 120, 300]), "seconds"), ("concurrency", rng.randint(1, 8), None)]))
    return tables


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return repr(v)


def _render(tables):
    lines = []
    for t, entries in tables:
        if t == "workers":
            lines.append("[[workers]]")
        elif t:
            lines.append(f"[{t}]")
        width = max([len(f"{k} = {_value(v)}") for k, v, c in entries if c is not None], default=0)
        for k, v, c in entries:
            kv = f"{k} = {_value(v)}"
            lines.append(kv if c is None else f"{kv.ljust(width)}  # {c}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# Change pool: (op, text). Texts use schema 1 names.
RENAMES = [
    ({"op": "rename", "table": "server", "key": "listen_port", "new": "port"}, "`[server] listen_port` is renamed to `port`."),
    ({"op": "rename", "table": "server", "key": "host", "new": "bind_address"}, "`[server] host` is renamed to `bind_address`."),
    ({"op": "rename", "table": "database", "key": "pool", "new": "pool_size"}, "`[database] pool` is renamed to `pool_size`."),
    ({"op": "rename", "table": "cache", "key": "backend", "new": "driver"}, "`[cache] backend` is renamed to `driver`."),
    ({"op": "rename", "table": "logging", "key": "level", "new": "min_level"}, "`[logging] level` is renamed to `min_level`."),
]
CONVERTS = [
    ({"op": "convert", "table": "server", "key": "request_timeout", "new": "request_timeout_ms", "factor": 1000}, "`[server] request_timeout` (seconds) is replaced by `request_timeout_ms` (milliseconds)."),
    ({"op": "convert", "table": "database", "key": "connect_timeout", "new": "connect_timeout_ms", "factor": 1000}, "`[database] connect_timeout` (seconds) is replaced by `connect_timeout_ms` (milliseconds)."),
    ({"op": "convert", "table": "cache", "key": "ttl", "new": "ttl_ms", "factor": 1000}, "`[cache] ttl` (seconds) is replaced by `ttl_ms` (milliseconds)."),
    ({"op": "convert", "table": "server", "key": "max_body_kb", "new": "max_body_bytes", "factor": 1024}, "`[server] max_body_kb` (kilobytes, 1 KB = 1024 bytes) is replaced by `max_body_bytes` (bytes)."),
    ({"op": "convert", "table": "logging", "key": "retention_days", "new": "retention_hours", "factor": 24}, "`[logging] retention_days` is replaced by `retention_hours` (1 day = 24 hours)."),
]
WORKER_CONVERT = ({"op": "convert", "table": "workers", "key": "interval", "new": "interval_ms", "factor": 1000}, "`[[workers]] interval` (seconds) is replaced by `interval_ms` (milliseconds) in every worker.")
MOVES = [
    ({"op": "move", "table": "database", "to": "storage.database"}, "`[database]` moves to `[storage.database]`."),
    ({"op": "move", "table": "cache", "to": "storage.cache"}, "`[cache]` moves to `[storage.cache]`."),
    ({"op": "move", "table": "logging", "to": "observability.logging"}, "`[logging]` moves to `[observability.logging]`."),
]
REMOVES = [
    ({"op": "remove", "table": "logging", "key": "colorize"}, "`[logging] colorize` is removed."),
    ({"op": "remove", "table": "cache", "key": "servers"}, "`[cache] servers` is removed (the driver discovers the servers)."),
]
SERVER_DEBUG = ({"op": "remove", "table": "server", "key": "debug"}, "`[server] debug` is removed.")
PLANNED = [
    ({"op": "rename", "table": "server", "key": "max_body_kb", "new": "body_limit_kb"}, "`[server] max_body_kb` will be renamed to `body_limit_kb`."),
    ({"op": "remove", "table": "database", "key": "pool"}, "`[database] pool` will be removed (pools become automatic)."),
    ({"op": "rename", "table": "logging", "key": "debug", "new": "trace_requests"}, "`[logging] debug` will be renamed to `trace_requests`."),
]

INSTRUCTION = """`/app/config.toml` still uses config schema 1. Migrate it in place to schema 2, following the schema 2 changes in `/app/CHANGELOG.md`.

Settings the changelog does not mention keep their values. A converted value is an integer in its new unit. {hint}
"""

HINT = "Python's `tomllib` can read the file; there is no TOML writer installed, so write the TOML text yourself."


def _changelog(rng, items, planned):
    rng.shuffle(items)
    lines = [
        "# Changelog",
        "",
        "## Config schema 2",
        "",
        "Table and key names below are the schema 1 names. Everything not listed is unchanged.",
        "",
        "- `schema_version` is now `2`.",
    ]
    lines += [f"- {t}" for t in items]
    if planned:
        lines += ["", "## Config schema 3 (planned, not released)", ""] + [f"- {t}" for t in planned]
    lines += ["", "## Config schema 1", "", "- First versioned config format."]
    return "\n".join(lines) + "\n"


def build(ctx):
    rng, p = ctx.rng, ctx.params
    tables = _config(rng, p)
    text = _render(tables)
    picked = rng.sample(RENAMES, p["renames"]) + rng.sample(CONVERTS, p["converts"]) + rng.sample(MOVES, p["moves"]) + rng.sample(REMOVES, p["removes"])
    if p["fractional"]:
        picked = [c for c in picked if c[0].get("key") != "connect_timeout"]
        picked.append(CONVERTS[1])  # the fractional seconds value is always converted
    if p["workers"]:
        picked += [WORKER_CONVERT, SERVER_DEBUG]
    touched = {(c[0].get("table"), c[0].get("key")) for c in picked}
    free = [c for c in PLANNED if (c[0]["table"], c[0]["key"]) not in touched]  # schema 3 changes apply to keys schema 2 leaves alone
    planned = rng.sample(free, min(2, len(free))) if p["planned"] else []
    changes = [{"op": "set", "key": "schema_version", "value": 2}] + [c[0] for c in picked]
    planned_ops = [c[0] for c in planned]
    cfg = tomllib.loads(text)
    expected = _NS["migrate"](cfg, changes, planned_ops, "oracle")
    shortcuts = {m: _solution(m, changes, planned_ops) for m in ("no-unit-conversion", "keep-removed", "partial-move")}
    if p["fractional"]:
        shortcuts["float-ms"] = _solution("float-ms", changes, planned_ops)
    if p["workers"]:
        for m in ("first-worker-only", "remove-everywhere"):
            shortcuts[m] = _solution(m, changes, planned_ops)
    if p["planned"]:
        shortcuts["apply-planned"] = _solution("apply-planned", changes, planned_ops)
    return TaskSpec(
        instruction=INSTRUCTION.format(hint="" if p["planned"] else HINT).rstrip() + "\n",
        files={"config.toml": text, "CHANGELOG.md": _changelog(rng, [c[1] for c in picked], [c[1] for c in planned])},
        grader=ParsedAnswer("/app/config.toml", "toml", expected),
        oracle=_solution("oracle", changes, planned_ops),
        shortcuts=shortcuts,
        params={k: p[k] for k in sorted(p)},
    )


FAMILY = Family(
    name="toml-migrate",
    version=1,
    cluster="config-repair",
    category="config",
    skills=("toml", "migration", "unit-conversion", "config-repair"),
    difficulties={
        "easy": {"renames": 1, "converts": 1, "moves": 1, "removes": 1, "fractional": False, "workers": 0, "planned": False},
        "medium": {"renames": 2, "converts": 2, "moves": 2, "removes": 2, "fractional": True, "workers": 0, "planned": False},
        "hard": {"renames": 3, "converts": 2, "moves": 3, "removes": 2, "fractional": True, "workers": 3, "planned": True},
    },
    build=build,
)
