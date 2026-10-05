"""env-reconcile: rewrite `/app/.env` from a template, the current file and deployment overrides.

Task: `/app/.env.example` lists the allowed keys with their defaults (an empty default marks a
required key); `/app/.env` is the current file, with stale unknown keys and some empty
values; the overrides (easy/medium `/app/deploy/overrides.env`; hard `deploy/common.env` and
`deploy/production.env`) hold deployment values. The instruction states the rules: exactly
the template's keys, each once; value from the highest source that sets it (overrides, then
the current file, then the template default); an empty value counts as not set in every file;
`KEY=value` lines. The grader parses the result as dotenv (duplicate keys fail) and compares
it with the expected mapping.

Construction: keys are dealt roles that plant each trap: a key set differently in the
overrides and the current file, a key only the current file sets (differently from its
default), a key empty in the current file whose default applies, required keys filled from
the current file and from the overrides, and unknown keys in the current file. DATABASE_URL
is always present and every value of it contains `=` (`?sslmode=...`). Medium adds an
unknown key in the overrides and an override with an empty value; hard has two override
layers that disagree on a key and `export ` prefixes in the current file.

Traps (declared shortcuts, all failing by construction):
- `env-wins`: the current file beats the overrides;
- `ignore-current`: template defaults plus overrides only;
- `empty-is-set`: `KEY=` in the current file keeps the key empty;
- `keep-unknown`: keys outside the template stay;
- `append-overrides`: `cat overrides >> .env`, which leaves duplicate keys;
- `awk-split`: an awk merge with `-F=` that reads only `$2`, cutting values at their second `=`;
- medium/hard `override-empty-clears`: an empty override empties the key;
- hard `common-wins`: the override layers in the wrong order;
- hard `export-as-unknown`: `export KEY=...` lines read as an unknown key `export KEY`.

The Python methods are one source (`SOLVER`) run with a mode, as in csv-revenue; `awk-split`
and `append-overrides` are shell one-liners with models that follow awk and `cat` exactly.
"""

from learning_loop.tasks.spec import Family, ParsedAnswer, Reject, Solution, TaskSpec

APPS = ["billing", "orders", "search", "notify", "ledger", "gateway", "catalog", "reports"]
WORDS = ["amber", "cobalt", "delta", "ember", "fjord", "granite", "harbor", "indigo", "juniper", "kestrel"]
UNKNOWN = ["LEGACY_API_HOST", "OLD_FEATURE_FLAG", "DEBUG_TOOLBAR", "MEMCACHE_SERVERS", "STATSD_PREFIX", "TMP_EXPORT_DIR"]


def _secret(rng, n):
    return "".join(rng.choice("abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(n))


GENERATORS = {
    "APP_ENV": lambda rng, app: rng.choice(["development", "staging", "production", "qa"]),
    "PORT": lambda rng, app: str(rng.randint(3000, 9999)),
    "LOG_LEVEL": lambda rng, app: rng.choice(["debug", "info", "warning", "error"]),
    "DATABASE_URL": lambda rng, app: f"postgres://{app}:{_secret(rng, 8)}@db-{rng.randint(1, 9)}.internal:5432/{app}?sslmode={rng.choice(['require', 'verify-full', 'prefer'])}",
    "REDIS_URL": lambda rng, app: f"redis://cache-{rng.randint(1, 9)}.internal:6379/{rng.randint(0, 15)}",
    "SECRET_KEY": lambda rng, app: _secret(rng, 24),
    "WORKERS": lambda rng, app: str(rng.randint(2, 32)),
    "REQUEST_TIMEOUT": lambda rng, app: str(rng.choice([5, 10, 15, 30, 45, 60, 90])),
    "CACHE_TTL": lambda rng, app: str(rng.choice([60, 120, 300, 600, 900, 3600])),
    "SMTP_HOST": lambda rng, app: f"smtp-{rng.choice(WORDS)}.internal",
    "SMTP_PORT": lambda rng, app: rng.choice(["25", "465", "587", "2525"]),
    "S3_BUCKET": lambda rng, app: f"{app}-{rng.choice(WORDS)}-{rng.randint(10, 99)}",
    "SENTRY_DSN": lambda rng, app: f"https://{_secret(rng, 12).lower()}@sentry.internal/{rng.randint(2, 40)}",
    "FEATURE_SIGNUP": lambda rng, app: rng.choice(["true", "false", "beta"]),
    "MAX_UPLOAD_MB": lambda rng, app: str(rng.choice([5, 10, 25, 50, 100, 250])),
    "API_BASE_URL": lambda rng, app: f"https://{rng.choice(WORDS)}.example.com/{app}",
}
UNKNOWN_GEN = lambda rng, app: rng.choice([f"{rng.choice(WORDS)}.internal", str(rng.randint(1, 500)), "true", f"/tmp/{app}"])  # noqa: E731

TEMPLATE_HEADER = "every supported setting, with its default"

INSTRUCTION = """Bring `/app/.env` in line with the template `/app/.env.example` and the deployment overrides in {layers}. Rewrite `/app/.env` in place so that:

- it sets exactly the keys listed in `/app/.env.example`, each once; a key that is not in the template is dropped, even when another file sets it;
- each key's value comes from the first of these that sets it: {precedence};
- an empty value (`KEY=`) counts as not set, in every file;
- template keys with an empty value have no default (they are required); each of them is set in one of the other files;
- every line is `KEY=value`, with no quotes, no `export` and no spaces around `=` (comment lines and blank lines are allowed).{export}
"""

EXPORT = "\n\nThe current `/app/.env` is also sourced by a shell script, so some of its lines start with `export `; such a line sets the key after `export`."

SOLVER = r'''
def parse(text, strip_export=True):
    """[(key, value)] in file order; comment and blank lines skipped."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if strip_export and s.startswith("export "):
            s = s[len("export "):].strip()
        key, _, value = s.partition("=")
        out.append((key.strip(), value.strip()))
    return out


def reconcile(template, current, layers, mode):
    """template, current: file texts; layers: override file texts, lowest precedence first."""
    tmpl = parse(template)
    cur = ("current", parse(current, strip_export=(mode != "export-as-unknown")))
    lays = [("override", parse(t)) for t in layers]
    if mode == "common-wins":
        lays = lays[::-1]
    sources = [cur] + lays  # increasing precedence
    if mode == "env-wins":
        sources = lays + [cur]
    if mode == "ignore-current":
        sources = lays
    values = {k: v for k, v in tmpl if v != ""}
    for src, pairs in sources:
        for k, v in pairs:
            empty_sets = (mode == "empty-is-set" and src == "current") or (mode == "override-empty-clears" and src == "override")
            if v != "" or empty_sets:
                values[k] = v
    keys = [k for k, _ in tmpl]
    if mode == "keep-unknown":
        for _, pairs in sources:
            keys += [k for k, _ in pairs if k not in keys]
    return "".join(f"{k}={values.get(k, '')}\n" for k in keys)
'''

_SHELL = """python3 - <<'PY'
{solver}
def read(path):
    with open(path) as f:
        return f.read()


layers = [read(p) for p in {layers!r}]
text = reconcile(read("/app/.env.example"), read("/app/.env"), layers, {mode!r})
with open("/app/.env", "w") as f:
    f.write(text)
PY
"""

AWK = """awk -F= 'FNR == 1 {{ f++ }}
/^[ \\t]*(#|$)/ {{ next }}
{{ k = $1; v = $2 }}
f == 1 {{ order[++n] = k; val[k] = v; next }}
v != "" {{ val[k] = v }}
END {{ for (i = 1; i <= n; i++) print order[i] "=" val[order[i]] }}' /app/.env.example /app/.env {layers} > /app/.env.new && mv /app/.env.new /app/.env
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _text(files, rel):
    return files[rel].decode()


def _solution(mode, layer_files):
    def model(files):
        text = _NS["reconcile"](_text(files, ".env.example"), _text(files, ".env"), [_text(files, r) for r in layer_files], mode)
        return {"/app/.env": text}

    return Solution(_SHELL.format(solver=SOLVER, layers=[f"/app/{r}" for r in layer_files], mode=mode), model)


def _awk_model(layer_files):
    """What the awk merge prints: fields split on every `=`, so the value is only `$2`."""

    def model(files):
        order, val = [], {}
        for f, rel in enumerate([".env.example", ".env", *layer_files], start=1):
            for line in _text(files, rel).splitlines():
                if line.lstrip(" \t").startswith("#") or line.strip(" \t") == "":
                    continue
                parts = line.split("=")
                k, v = parts[0], parts[1] if len(parts) > 1 else ""
                if f == 1:
                    order.append(k)
                    val[k] = v
                elif v != "":
                    val[k] = v
        return {"/app/.env": "".join(f"{k}={val[k]}\n" for k in order)}

    return Solution(AWK.format(layers=" ".join(f"/app/{r}" for r in layer_files)), model)


def _append_model(layer_files):
    def model(files):
        return {"/app/.env": _text(files, ".env") + "".join(_text(files, r) for r in layer_files)}

    return Solution(f"cat {' '.join(f'/app/{r}' for r in layer_files)} >> /app/.env\n", model)


# --------------------------------------------------------------------------- #
# Generation: every key gets a role that decides where its values appear
# --------------------------------------------------------------------------- #


def _differs(rng, gen, app, *avoid):
    for _ in range(50):
        v = gen(rng, app)
        if v not in avoid:
            return v
    raise Reject("could not draw a distinct value")


def _roles(rng, p):
    roles = ["override_beats_env", "env_value", "env_empty", "required_from_env", "required_from_override"]
    if p["override_extras"]:
        roles += ["override_empty"]
    if p["layers"] == 2:
        roles += ["layer_order", "export_env_value"]
    while len(roles) < p["n_keys"]:
        roles.append(rng.choice(["default_only", "default_only", "env_value", "override_only", "env_same"]))
    rng.shuffle(roles)
    return roles


def build(ctx):
    rng, p = ctx.rng, ctx.params
    app = rng.choice(APPS)
    names = ["DATABASE_URL"] + rng.sample([k for k in GENERATORS if k != "DATABASE_URL"], p["n_keys"] - 1)
    rng.shuffle(names)
    roles = _roles(rng, p)
    n_layers = p["layers"]
    template, current, layers = [], [], [[] for _ in range(n_layers)]
    expected, exported = {}, set()
    for key, role in zip(names, roles):
        gen = GENERATORS[key]
        d = gen(rng, app)
        top = rng.randrange(n_layers)  # the layer an override goes to
        if role == "override_beats_env":
            e = _differs(rng, gen, app, d)
            o = _differs(rng, gen, app, d, e)
            template.append((key, d)), current.append((key, e)), layers[top].append((key, o))
            expected[key] = o
        elif role == "env_value":
            e = _differs(rng, gen, app, d)
            template.append((key, d)), current.append((key, e))
            expected[key] = e
        elif role == "export_env_value":
            e = _differs(rng, gen, app, d)
            template.append((key, d)), current.append((key, e))
            exported.add(key)
            expected[key] = e
        elif role == "env_same":
            template.append((key, d)), current.append((key, d))
            expected[key] = d
        elif role == "env_empty":
            template.append((key, d)), current.append((key, ""))
            expected[key] = d
        elif role == "required_from_env":
            e = gen(rng, app)
            template.append((key, "")), current.append((key, e))
            expected[key] = e
        elif role == "required_from_override":
            o = gen(rng, app)
            template.append((key, "")), layers[top].append((key, o))
            expected[key] = o
        elif role == "override_only":
            o = _differs(rng, gen, app, d)
            template.append((key, d)), layers[top].append((key, o))
            expected[key] = o
        elif role == "override_empty":
            e = _differs(rng, gen, app, d)
            template.append((key, d)), current.append((key, e)), layers[top].append((key, ""))
            expected[key] = e
        elif role == "layer_order":
            c = _differs(rng, gen, app, d)
            q = _differs(rng, gen, app, d, c)
            template.append((key, d)), current.append((key, _differs(rng, gen, app, c, q))), layers[0].append((key, c)), layers[1].append((key, q))
            expected[key] = q
        else:  # default_only
            template.append((key, d))
            expected[key] = d
    for key in rng.sample(UNKNOWN, p["unknown_keys"]):
        current.insert(rng.randint(0, len(current)), (key, UNKNOWN_GEN(rng, app)))
    if p["override_extras"]:
        layers[-1].insert(rng.randint(0, len(layers[-1])), (rng.choice([k for k in UNKNOWN if k not in dict(current)]), UNKNOWN_GEN(rng, app)))
    if p["layers"] == 2:
        exported |= {k for k, _ in current if rng.random() < 0.4}
    rng.shuffle(current)
    for lay in layers:
        rng.shuffle(lay)

    def render(pairs, header, exports=frozenset()):
        lines = [f"# {header}"]
        for k, v in pairs:
            if header == TEMPLATE_HEADER and v == "":
                lines.append("# required")
            lines.append(f"{'export ' if k in exports else ''}{k}={v}")
        return "\n".join(lines) + "\n"

    layer_files = ["deploy/overrides.env"] if n_layers == 1 else ["deploy/common.env", "deploy/production.env"]
    files = {
        ".env.example": render(template, TEMPLATE_HEADER),
        ".env": render(current, f"local settings for {app} (edited by hand)", exported),
    }
    for rel, lay in zip(layer_files, layers):
        files[rel] = render(lay, f"deployment overrides ({rel.split('/')[-1]})")
    if n_layers == 1:
        layers_text, precedence = "`/app/deploy/overrides.env`", "`/app/deploy/overrides.env`, then the current `/app/.env`, then the default in `/app/.env.example`"
    else:
        layers_text = "`/app/deploy/common.env` and `/app/deploy/production.env`"
        precedence = "`/app/deploy/production.env`, then `/app/deploy/common.env`, then the current `/app/.env`, then the default in `/app/.env.example`"
    shortcuts = {m: _solution(m, layer_files) for m in ("env-wins", "ignore-current", "empty-is-set", "keep-unknown")}
    shortcuts["append-overrides"] = _append_model(layer_files)
    shortcuts["awk-split"] = _awk_model(layer_files)
    if p["override_extras"]:
        shortcuts["override-empty-clears"] = _solution("override-empty-clears", layer_files)
    if n_layers == 2:
        shortcuts["common-wins"] = _solution("common-wins", layer_files)
        shortcuts["export-as-unknown"] = _solution("export-as-unknown", layer_files)
    return TaskSpec(
        instruction=INSTRUCTION.format(layers=layers_text, precedence=precedence, export=EXPORT if exported else ""),
        files=files,
        grader=ParsedAnswer("/app/.env", "dotenv", expected),
        oracle=_solution("oracle", layer_files),
        shortcuts=shortcuts,
        params={k: p[k] for k in sorted(p)},
    )


FAMILY = Family(
    name="env-reconcile",
    version=1,
    cluster="config-repair",
    category="config",
    skills=("dotenv", "precedence", "config-repair", "shell"),
    difficulties={
        "easy": {"n_keys": 7, "layers": 1, "unknown_keys": 1, "override_extras": False},
        "medium": {"n_keys": 10, "layers": 1, "unknown_keys": 2, "override_extras": True},
        "hard": {"n_keys": 13, "layers": 2, "unknown_keys": 3, "override_extras": True},
    },
    build=build,
)
