"""traceback-locate: a nightly job crashed; from its traceback and source, name the line that
set the bad setting value the run used (not the line that raised).

Construction: a small package `/app/<pkg>/` started with `python3 -m <pkg>.main`. The job
loads a settings profile and hands it down a call chain to a consumer that reads one key,
`settings["<key>"]`, and passes it to a helper in `util.py` that raises: `int(value)` on a
malformed string such as "30s" or "1,000" (ValueError, whose message shows the literal), or
`total // parts` on 0 (ZeroDivisionError, medium/hard only, whose message shows nothing).
The traceback in `/app/incident/traceback.txt` is exactly what Python prints for that code
(the unit test runs the package and compares). The answer is the dictionary entry that holds
the value the run used: on easy/medium the active profile's entry in `settings.py`; on hard
the active profile's entry in `overrides.py`, which `load_settings` applies on top of a
profile that has a good value. The active profile is a literal in `main.py` on easy and is
imported from `deploy.py` (which also mentions an older profile in a comment) on
medium/hard; it is never `prod`.

Traps: the root-cause line is never in the traceback, so reporting the last frame, the first
frame in the package's own code or the frame that read the setting fails. The same bad
literal sits in other, unused profiles (and on hard in another profile's overrides) before
the real one, so taking the first `grep` hit of the literal fails. Medium/hard: assuming the
`prod` profile fails. Hard: the worker re-raises the error as a RuntimeError, so the
traceback has two sections and its last frame is the re-raise; ignoring the overrides and
reporting the profile's own entry fails.

Every solution, right or wrong, is one Python source (`SOLVER`) run with a mode, as in
csv-revenue, so the shell and model forms cannot drift apart.
"""

from learning_loop.tasks.spec import Family, ParsedAnswer, Reject, Solution, TaskSpec

PACKAGES = ["shipping", "billing", "ingest", "reports", "notify", "catalog"]
JOBS = {"shipping": "label export", "billing": "invoice run", "ingest": "feed import", "reports": "report build", "notify": "digest mailer", "catalog": "price sync"}

# key -> (consumer module, consumer function, local name, module docstring, good values, bad literal format)
INT_KEYS = {
    "timeout_sec": ("client", "make_client", "timeout", "HTTP client for the upstream API.", [10, 20, 30, 45, 60], "{}s"),
    "poll_interval": ("scheduler", "make_schedule", "interval", "Polling schedule for the work queue.", [5, 10, 15, 30], "{}s"),
    "batch_size": ("batcher", "make_batcher", "size", "Groups records into write batches.", [1000, 2000, 2500, 5000], "{:,}"),
    "cache_ttl": ("cache", "make_cache", "ttl", "In-memory cache for lookups.", [120, 300, 600, 900], "minutes"),
    "page_size": ("paging", "make_pager", "size", "Pagination over the upstream listing.", [25, 50, 100, 200], "{}.0"),
}
ZERO_KEY = "workers"
ZERO_CONSUMER = ("planner", "plan_batches", "per_worker", "Splits the job queue between workers.")
LOG_LEVELS = ["debug", "info", "warning"]
NOISE_MODULES = {
    "metrics": '"""Counters reported at the end of a run."""\n\nCOUNTERS = {{}}\n\n\ndef bump(name, by=1):\n    COUNTERS[name] = COUNTERS.get(name, 0) + by\n    return COUNTERS[name]\n',
    "formatting": '"""Formatting helpers for log lines."""\n\n\ndef human_size(n):\n    for unit in ("B", "KB", "MB"):\n        if n < 1024:\n            return f"{{n}} {{unit}}"\n        n //= 1024\n    return f"{{n}} GB"\n',
    "retry": '"""Retry helper for flaky upstream calls."""\n\n\ndef with_retries(fn, attempts):\n    last = None\n    for _ in range(attempts):\n        try:\n            return fn()\n        except OSError as exc:\n            last = exc\n    raise last\n',
}

INSTRUCTION = """Last night's {job} job crashed. Its log, with the Python traceback, is in `/app/incident/traceback.txt`. The job is the `{pkg}` package in `/app/{pkg}/`, started with `cd /app && python3 -m {pkg}.main`.

The line that raised the exception is correct: it was handed a bad setting value. Find the line of source code that set the value this run actually used, i.e. the dictionary entry you would have to change to fix the crash.{extra}

Write a JSON object to `/app/answer.json` with two keys: `"file"`, the absolute path of that source file as it would appear in a traceback, and `"line"`, its line number as an integer. For example: `{{"file": "/app/{pkg}/example.py", "line": 12}}`.
"""

SOLVER = r'''
import ast
import re

FRAME = re.compile(r'^  File "([^"]+)", line (\d+), in (\S+)$')


def sections(tb):
    """The traceback's sections in printed order, each a list of (file, line, func, source)."""
    out, cur = [], None
    lines = tb.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("Traceback (most recent call last):"):
            cur = []
            out.append(cur)
            continue
        m = FRAME.match(ln)
        if m and cur is not None:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            src = nxt.strip() if nxt.startswith("    ") else ""
            cur.append((m.group(1), int(m.group(2)), m.group(3), src))
    return out


def _const(sources, module, name, depth=0):
    """The string constant that `name` in `module` resolves to, following `from m import n`."""
    if module not in sources or depth > 5:
        return None
    for node in ast.parse(sources[module]).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node.value.value
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if (a.asname or a.name) == name:
                    return _const(sources, node.module, a.name, depth + 1)
    return None


def _entry(sources, module, var, profile, key):
    """Line of `var[profile][key]` in `module`, where `var` is a dict literal of dict literals."""
    if module not in sources:
        return None
    for node in ast.parse(sources[module]).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == profile and isinstance(v, ast.Dict):
                    for k2 in v.keys:
                        if isinstance(k2, ast.Constant) and k2.value == key:
                            return k2.lineno
    return None


def answer(tb, sources, mode):
    """tb: traceback text; sources: {dotted module name: source}. Returns (file, line) or None."""
    secs = sections(tb)
    frames = [f for s in secs for f in s]
    if not frames:
        return None
    if mode == "last-frame":
        return frames[-1][:2]
    if mode == "first-app-frame":
        return next(f for f in frames if f[0].startswith("/app/"))[:2]
    first = secs[0]  # the original exception (printed first when it was re-raised)
    if mode == "caller-frame":
        return first[-2][:2]
    if mode == "grep-first":
        m = re.search(r"invalid literal for int\(\) with base 10: '(.*)'", tb)
        if not m:
            return None
        for name in sorted(sources):
            for i, ln in enumerate(sources[name].splitlines(), 1):
                if m.group(1) in ln:
                    return ("/app/" + name.replace(".", "/") + ".py", i)
        return None
    key = None
    for _, _, _, src in first:
        k = re.search(r'settings\["(\w+)"\]', src)
        if k:
            key = k.group(1)
    pkg = next(f for f in frames if f[0].startswith("/app/"))[0].split("/")[2]
    profile = "prod" if mode == "assume-prod" else _const(sources, pkg + ".main", "PROFILE")
    if mode != "profile-line":
        line = _entry(sources, pkg + ".overrides", "OVERRIDES", profile, key)
        if line is not None:
            return ("/app/" + pkg + "/overrides.py", line)
    line = _entry(sources, pkg + ".settings", "PROFILES", profile, key)
    return None if line is None else ("/app/" + pkg + "/settings.py", line)
'''

_SHELL = """python3 - <<'PY'
{solver}
import json
import os

sources = {{}}
for d, _, names in os.walk("/app"):
    for n in names:
        if n.endswith(".py"):
            p = os.path.join(d, n)
            sources[os.path.relpath(p, "/app")[:-3].replace("/", ".")] = open(p).read()
result = answer(open("/app/incident/traceback.txt").read(), sources, {mode!r})
if result is not None:
    with open("/app/answer.json", "w") as f:
        f.write(json.dumps({{"file": result[0], "line": result[1]}}) + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _solution(mode):
    def model(files):
        sources = {rel[:-3].replace("/", "."): data.decode() for rel, data in files.items() if rel.endswith(".py")}
        result = _NS["answer"](files["incident/traceback.txt"].decode(), sources, mode)
        return {} if result is None else {"/app/answer.json": '{"file": "%s", "line": %d}\n' % result}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def _line(text, needle):
    """1-based number of the only line of `text` that contains `needle`."""
    hits = [i for i, ln in enumerate(text.splitlines(), 1) if needle in ln]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def _entry_line(text, profile, key):
    """1-based line of `key` inside the `profile` block of a dict literal written by `_dict_literal`."""
    lines = text.splitlines()
    start = lines.index(f'    "{profile}": {{')
    return next(i for i, ln in enumerate(lines[start:], start + 1) if ln.startswith(f'        "{key}": '))


def _dict_literal(var, entries):
    """`var = {profile: {key: value}}` with one entry per line; values are Python reprs."""
    out = [f"{var} = {{"]
    for profile, kv in entries:
        out.append(f'    "{profile}": {{')
        out += [f'        "{k}": {v},' for k, v in kv]
        out.append("    },")
    out.append("}")
    return "\n".join(out) + "\n"


def build(ctx):
    rng, p = ctx.rng, ctx.params
    pkg = rng.choice(PACKAGES)
    job = JOBS[pkg]
    zero = p["zero_kind"] and rng.random() < 0.4
    profiles = list(p["profiles"])
    active = rng.choice(p["active_choices"])
    others = [x for x in profiles if x != active]

    if zero:
        key = ZERO_KEY
        mod, fn, local, doc = ZERO_CONSUMER
        bad = 0
    else:
        key = rng.choice(sorted(INT_KEYS))
        mod, fn, local, doc, goods, fmt = INT_KEYS[key]
        good = rng.choice(goods)
        bad = f"{good // 60}m" if fmt == "minutes" else fmt.format(good)
    bad_src = repr(bad).replace("'", '"')

    # Profiles: every key with a good value; the bad literal in the active profile (easy/medium)
    # or in its overrides (hard), and in decoy profiles listed before it.
    keys = sorted(INT_KEYS) + [ZERO_KEY, "log_level"]
    rng.shuffle(keys)
    decoys = set(rng.sample(others, p["decoys"]))
    if "dev" in others:
        decoys.add("dev")  # first in the file, so the first grep hit is a decoy
    values = {}
    for prof in profiles:
        kv = []
        for k in keys:
            if k == "log_level":
                v = repr(rng.choice(LOG_LEVELS)).replace("'", '"')
            elif k == ZERO_KEY:
                v = str(rng.randint(2, 8))
            else:
                v = str(rng.choice(INT_KEYS[k][4]))
            kv.append((k, v))
        values[prof] = kv
    overrides = None
    if p["overrides"]:
        for prof in decoys:
            values[prof] = [(k, bad_src if k == key else v) for k, v in values[prof]]
        decoy = sorted(decoys, key=profiles.index)[0]
        extra_key = rng.choice([k for k in sorted(INT_KEYS) if k != key])
        ov_active = [(key, bad_src), (extra_key, str(rng.choice(INT_KEYS[extra_key][4])))]
        rng.shuffle(ov_active)
        overrides = [(decoy, [(key, bad_src)]), (active, ov_active)]
        if "prod" not in (decoy, active):
            overrides.append(("prod", [(extra_key, str(rng.choice(INT_KEYS[extra_key][4])))]))
    else:
        for prof in decoys | {active}:
            values[prof] = [(k, bad_src if k == key else v) for k, v in values[prof]]

    files = {}
    P = f"/app/{pkg}"
    files[f"{pkg}/__init__.py"] = f'"""The {job} job."""\n'
    if p["deploy"]:
        old = rng.choice([x for x in others if x != "dev"])
        files[f"{pkg}/deploy.py"] = (
            '"""Deployment selection for this host."""\n\n'
            f'# The profile this host runs (was "{old}" until the {rng.choice(["March", "June", "August"])} migration).\n'
            f'ACTIVE_PROFILE = "{active}"\n'
        )
        main_imports = f"from {pkg}.deploy import ACTIVE_PROFILE as PROFILE\nfrom {pkg}.settings import load_settings\nfrom {pkg}.worker import run\n"
        profile_line = ""
    else:
        main_imports = f"from {pkg}.settings import load_settings\nfrom {pkg}.worker import run\n"
        profile_line = f'\nPROFILE = "{active}"\n'
    main = (
        f'"""Nightly {job} job.\n\nRun with: cd /app && python3 -m {pkg}.main\n"""\n\n'
        f"{main_imports}{profile_line}\n\n"
        "def main():\n    settings = load_settings(PROFILE)\n    run(settings)\n\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )
    files[f"{pkg}/main.py"] = main

    load_doc = "The settings of profile `name`, with that profile's overrides applied." if overrides else "The settings of profile `name`."
    settings = f'"""Settings profiles for the {job} job."""\n\n'
    if overrides:
        settings += f"from {pkg}.overrides import OVERRIDES\n\n"
    settings += _dict_literal("PROFILES", [(prof, values[prof]) for prof in profiles])
    settings += f'\n\ndef load_settings(name):\n    """{load_doc}"""\n    settings = dict(PROFILES[name])\n'
    if overrides:
        settings += "    settings.update(OVERRIDES.get(name, {}))\n"
    settings += "    return settings\n"
    files[f"{pkg}/settings.py"] = settings
    if overrides:
        files[f"{pkg}/overrides.py"] = '"""Per-profile overrides, applied on top of PROFILES by load_settings."""\n\n' + _dict_literal("OVERRIDES", overrides)

    helper = "split_evenly" if zero else "parse_int"
    files[f"{pkg}/util.py"] = (
        '"""Small helpers shared by the job\'s modules."""\n\n\n'
        'def parse_int(value):\n    """An integer setting, given as an int or a string of digits."""\n    return int(value)\n\n\n'
        'def split_evenly(total, parts):\n    """How many of `total` items each of `parts` parts gets."""\n    return total // parts\n'
    )
    call = f'{local} = {helper}(JOB_COUNT, settings["{key}"])' if zero else f'{local} = {helper}(settings["{key}"])'
    files[f"{pkg}/{mod}.py"] = (
        f'"""{doc}"""\n\nfrom {pkg}.util import {helper}\n\n'
        + (f"JOB_COUNT = {rng.randint(60, 240)}\n" if zero else "")
        + f'\n\ndef {fn}(settings):\n    {call}\n    return {{"{local}": {local}}}\n'
    )
    if p["pipeline"]:
        files[f"{pkg}/pipeline.py"] = (
            f'"""Builds what a run needs before it starts."""\n\nfrom {pkg}.{mod} import {fn}\n\n\n'
            f'def prepare(settings):\n    level = settings["log_level"]\n    resource = {fn}(settings)\n    print("{job}: log level", level)\n    return resource\n'
        )
        first_call, first_mod = "prepare", "pipeline"
    else:
        first_call, first_mod = fn, mod
    if p["reraise"]:
        body = f'    try:\n        resource = {first_call}(settings)\n    except Exception as exc:\n        raise RuntimeError("{job} job setup failed") from exc\n'
    else:
        body = f"    resource = {first_call}(settings)\n"
    files[f"{pkg}/worker.py"] = f'"""Runs one pass of the {job} job."""\n\nfrom {pkg}.{first_mod} import {first_call}\n\n\ndef run(settings):\n{body}    print("{job}: ready", resource)\n'
    for name in rng.sample(sorted(NOISE_MODULES), p["noise"]):
        files[f"{pkg}/{name}.py"] = NOISE_MODULES[name].format()

    # The traceback Python prints for this code.
    def fr(module, func, needle):
        text = files[f"{pkg}/{module}.py"]
        return f'  File "{P}/{module}.py", line {_line(text, needle)}, in {func}\n    {needle.strip()}\n'

    inner = [fr("worker", "run", f"    resource = {first_call}(settings)")]
    if p["pipeline"]:
        inner.append(fr("pipeline", "prepare", f"    resource = {fn}(settings)"))
    inner.append(fr(mod, fn, f"    {call}"))
    inner.append(fr("util", helper, "    return total // parts" if zero else "    return int(value)"))
    error = "ZeroDivisionError: integer division or modulo by zero" if zero else f"ValueError: invalid literal for int() with base 10: {bad!r}"
    outer = [
        '  File "<frozen runpy>", line 198, in _run_module_as_main\n',
        '  File "<frozen runpy>", line 88, in _run_code\n',
        fr("main", "<module>", "    main()"),
        fr("main", "main", "    run(settings)"),
    ]
    head = "Traceback (most recent call last):\n"
    if p["reraise"]:
        outer.append(fr("worker", "run", f'        raise RuntimeError("{job} job setup failed") from exc'))
        tb = head + "".join(inner) + error + "\n\nThe above exception was the direct cause of the following exception:\n\n" + head + "".join(outer) + f"RuntimeError: {job} job setup failed\n"
    else:
        tb = head + "".join(outer + inner) + error + "\n"
    day = rng.randint(1, 28)
    stamp = f"2026-09-{day:02d}T02:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
    files["incident/traceback.txt"] = f"{stamp} cron: starting {job} (python3 -m {pkg}.main)\n{tb}{stamp} cron: {job} exited with status 1\n"

    mod_path = "overrides" if overrides else "settings"
    target = (f"{P}/{mod_path}.py", _entry_line(files[f"{pkg}/{mod_path}.py"], active, key))

    shortcuts = {m: _solution(m) for m in ("last-frame", "first-app-frame", "caller-frame")}
    if not zero:
        shortcuts["grep-first"] = _solution("grep-first")
    if p["deploy"]:
        shortcuts["assume-prod"] = _solution("assume-prod")
    if overrides:
        shortcuts["profile-line"] = _solution("profile-line")
    oracle = _solution("oracle")
    if oracle.model({k: v.encode() for k, v in files.items()}) != {"/app/answer.json": '{"file": "%s", "line": %d}\n' % target}:
        raise Reject("the oracle does not find the planted line")
    extra = " The settings may be overridden in more than one place." if overrides else ""
    return TaskSpec(
        instruction=INSTRUCTION.format(job=job, pkg=pkg, extra=extra),
        files=files,
        grader=ParsedAnswer("/app/answer.json", "json", {"file": target[0], "line": target[1]}),
        oracle=oracle,
        shortcuts=shortcuts,
        params={"package": pkg, "key": key, "kind": "zero-division" if zero else "int-parse", "active_profile": active, "file": target[0], "line": target[1]},
    )


FAMILY = Family(
    name="traceback-locate",
    version=1,
    cluster="diagnosis",
    category="debugging",
    skills=("python", "tracebacks", "code-reading", "diagnosis"),
    difficulties={
        "easy": {"profiles": ["dev", "staging", "prod"], "active_choices": ["staging", "prod"], "decoys": 0, "deploy": False, "pipeline": False, "overrides": False, "reraise": False, "zero_kind": False, "noise": 0},
        "medium": {"profiles": ["dev", "test", "staging", "prod"], "active_choices": ["test", "staging"], "decoys": 1, "deploy": True, "pipeline": True, "overrides": False, "reraise": False, "zero_kind": True, "noise": 1},
        "hard": {"profiles": ["dev", "test", "staging", "prod"], "active_choices": ["test", "staging"], "decoys": 1, "deploy": True, "pipeline": True, "overrides": True, "reraise": True, "zero_kind": True, "noise": 3},
    },
    build=build,
)
