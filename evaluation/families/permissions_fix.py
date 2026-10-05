"""permissions-fix: set file modes under /app/site from a glob -> mode policy.

/app/perms.policy has one `<pattern> <mode>` rule per line. A pattern matches a file's path
relative to /app/site, where `*` matches any characters including `/` and a leading `.`; the
last matching rule wins, and files no rule matches keep their mode. The grader is a FileTree
over /app/site with every file's content and mode.

Construction: the rules go from general to specific, and some files match two rules with
different modes (`secrets/*` then `secrets/*.pub`; on medium/hard `*.env` then
`config/prod/*`; on hard `secrets/*` then `*.key`), so applying the first matching rule
fails. Files that a rule matches only because `*` crosses `/` sit in subfolders
(`scripts/x.sh` under `*.sh`; on medium/hard also `public/css/x.css` under `public/*`) and on
medium/hard a hidden `.env` matches `*.env`, so running `chmod <mode> <pattern>` with shell
globbing fails. Some files match no rule and have modes other than 0644, so resetting
every file to 0644 before applying the rules fails. Every file starts with a mode
other than its target, so doing nothing fails. Every trap holds by construction, so the
first draw is always used.
"""

from learning_loop.tasks.spec import Family, FileState, FileTree, Solution, TaskSpec, tree_entry

SCRIPTS = ["deploy", "backup", "restart", "migrate", "healthcheck", "rotate-logs", "warm-cache"]
ASSETS = ["site", "main", "theme", "print", "app", "vendor"]
KEYS = ["api", "stripe", "smtp", "github", "s3", "deploy"]
DATA = ["cache.db", "sessions.db", "queue.db"]
MESSY = [0o600, 0o640, 0o644, 0o660, 0o664, 0o666, 0o700, 0o750, 0o755, 0o775, 0o777]
UNMATCHED_MODES = [0o600, 0o640, 0o660, 0o664, 0o666, 0o750, 0o775]  # never 0644

INSTRUCTION = """The deployment in `/app/site/` has messy file permissions. `/app/perms.policy` lists the modes its files must have, one rule per line: `<pattern> <mode>`, where the pattern is matched against a file's path relative to `/app/site/` (such as `bin/deploy.sh`) and the mode is octal. Lines starting with `#` and empty lines are comments.

- In a pattern, `*` matches any sequence of characters, including none, `/` and a leading `.`; every other character matches itself. A pattern must match the whole path.{examples}
- When several rules match a file, the **last** matching rule wins.
- A file that no rule matches keeps its current mode.

Set the mode of every regular file under `/app/site/`, in every subfolder, accordingly. Leave directory modes as they are, and do not change, add, remove or rename any file.
"""
EXAMPLES = " So `*.sh` matches `run.sh` and `tools/x/run.sh`, and `*.env` matches `.env`."

SOLVER = '''
import re


def rules(policy):
    """[(pattern, mode)] in file order."""
    out = []
    for line in policy.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pattern, mode = line.split()
            out.append((pattern, int(mode, 8)))
    return out


def matches(pattern, path):
    return re.fullmatch(".*".join(re.escape(part) for part in pattern.split("*")), path, re.S) is not None


def modes(policy, paths, mode="oracle"):
    """{path: new mode} for the files some rule matches (`first-match` keeps the first rule)."""
    out = {}
    for path in paths:
        hits = [m for pattern, m in rules(policy) if matches(pattern, path)]
        if hits:
            out[path] = hits[0] if mode == "first-match" else hits[-1]
    return out
'''

_PY_SHELL = """python3 - <<'PY'
{solver}
import os

root = "/app/site"
paths = []
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        full = os.path.join(dirpath, name)
        if os.path.isfile(full) and not os.path.islink(full):
            paths.append(os.path.relpath(full, root))
with open("/app/perms.policy") as f:
    policy = f.read()
for path, m in sorted(modes(policy, paths, {mode!r}).items()):
    os.chmod(os.path.join(root, path), m)
PY
"""

SHELL_GLOB = """cd /app/site
grep -v '^#' /app/perms.policy | while read -r pattern mode; do
  for f in $pattern; do [ -f "$f" ] && chmod "$mode" "$f"; done
done
"""

RESET_THEN_APPLY = """find /app/site -type f -exec chmod 644 {} +
cd /app/site
grep -Ev '^(#|$)' /app/perms.policy | while read -r pattern mode; do
  find . -type f -path "./$pattern" -exec chmod "$mode" {} +
done
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of the Python solutions


def _site(files):
    return {rel[len("site/"):]: data for rel, data in files.items() if rel.startswith("site/")}


def _states(files, new_modes):
    site = _site(files)
    return {f"/app/site/{path}": FileState(site[path], m) for path, m in new_modes.items()}


def _py_solution(mode):
    return Solution(_PY_SHELL.format(solver=SOLVER, mode=mode), lambda f: _states(f, _NS["modes"](f["perms.policy"].decode(), list(_site(f)), mode)))


def _glob(paths, pattern):
    """What bash expands an unquoted pattern to (no dotglob): components matched one level at a
    time, `*` not crossing `/` and not matching a leading `.`. Returns files and directories
    (the shell form chmods only the files)."""
    entries = set(paths) | {p.rsplit("/", k)[0] for p in paths for k in range(1, p.count("/") + 1)}
    current = [""]
    for comp in pattern.split("/"):
        nxt = []
        for base in current:
            for e in sorted(entries):
                parent, _, name = e.rpartition("/")
                if parent != base:
                    continue
                if "*" in comp:
                    ok = _NS["matches"](comp, name) and not (name.startswith(".") and not comp.startswith("."))
                else:
                    ok = name == comp
                if ok:
                    nxt.append(e)
        current = nxt
    return current


def _shell_glob(files):
    site = _site(files)
    out = {}
    for pattern, m in _NS["rules"](files["perms.policy"].decode()):
        for e in _glob(list(site), pattern):
            if e in site:
                out[e] = m
    return _states(files, out)


def _reset_then_apply(files):
    site = _site(files)
    new = {path: 0o644 for path in site}
    new.update(_NS["modes"](files["perms.policy"].decode(), list(site)))
    return _states(files, new)


def _layout(rng, level):
    """(files {path: kind}, rules [(pattern, mode)]) for a difficulty level 0/1/2."""
    s = rng.sample(SCRIPTS, 4)
    a = rng.sample(ASSETS, 3)
    k = rng.sample(KEYS, 3)
    files = {
        "index.html": "page",
        f"public/{a[0]}.css": "asset",
        f"public/{a[1]}.js": "asset",
        f"bin/{s[0]}.sh": "script",
        f"bin/{s[1]}": "script",
        f"scripts/{s[2]}.sh": "script",
        f"secrets/{k[0]}.key": "secret",
        f"secrets/{k[1]}.pub": "public-key",
        f"data/{rng.choice(DATA)}": "data",
    }
    rules = [
        ("public/*", rng.choice([0o644, 0o444])),
        ("*.sh", rng.choice([0o755, 0o750])),
        ("bin/*", rng.choice([0o755, 0o750])),
        ("secrets/*", rng.choice([0o600, 0o640])),
        ("secrets/*.pub", rng.choice([0o644, 0o444])),
    ]
    if level >= 1:
        files.update({f"public/css/{a[2]}.css": "asset", f"bin/tools/{s[3]}.sh": "script", ".env": "env", "config/app.env": "env", "config/prod/app.env": "env"})
        rules += [("*.env", rng.choice([0o640, 0o660])), ("config/prod/*", rng.choice([0o600, 0o400]))]
    if level >= 2:
        files.update({f"secrets/ssh/{k[2]}.key": "secret", f"secrets/ssh/{k[2]}.pub": "public-key", "logs/app.log": "log", "logs/archive/app-1.log": "log", "vendor/bin/tool": "data", "docs/README.md": "page"})
        rules += [("logs/*", rng.choice([0o640, 0o600])), ("*.key", 0o400)]
    return files, rules


CATEGORIES = {"public/*": "web assets", "*.sh": "scripts", "bin/*": "scripts", "secrets/*": "secrets", "secrets/*.pub": "secrets", "*.env": "environment files", "config/prod/*": "environment files", "logs/*": "logs", "*.key": "private keys"}


def _policy(rules, comments):
    """The policy text; with comments, a header line and a comment before each group of rules."""
    lines = ["# File modes for /app/site: <pattern> <mode>"] if comments else []
    previous = None
    for pattern, mode in rules:
        if comments and CATEGORIES[pattern] != previous:
            lines += ["", f"# {CATEGORIES[pattern]}"]
            previous = CATEGORIES[pattern]
        lines.append(f"{pattern:<15} {mode:04o}")
    return "\n".join(lines) + "\n"


def _content(rng, path, kind):
    if kind == "script":
        return f"#!/bin/sh\nset -e\necho running {path.rsplit('/', 1)[-1]}\n"
    if kind in ("secret", "public-key"):
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz0123456789") for _ in range(40)) + "\n"
    if kind == "env":
        return f"DATABASE_URL=postgres://db:{rng.randint(5000, 6000)}/app\nLOG_LEVEL={rng.choice(['info', 'debug', 'warn'])}\n"
    return f"{path} {rng.randint(1000, 99999)}\n"


def build(ctx):
    rng, p = ctx.rng, ctx.params
    layout, rules = _layout(rng, p["level"])
    policy = _policy(rules, comments=p["level"] >= 2)
    target = _NS["modes"](policy, list(layout))
    files, modes = {"perms.policy": policy}, {"perms.policy": 0o644}
    expected = {}
    for path in sorted(layout):
        content = _content(rng, path, layout[path])
        mode = rng.choice([m for m in MESSY if m != target[path]]) if path in target else rng.choice(UNMATCHED_MODES)
        files[f"site/{path}"], modes[f"site/{path}"] = content, mode
        expected[path] = tree_entry(content, target.get(path, mode))
    return TaskSpec(
        instruction=INSTRUCTION.format(examples=EXAMPLES if p["level"] < 2 else ""),
        files=files,
        modes=modes,
        grader=FileTree("/app/site", expected),
        oracle=_py_solution("oracle"),
        shortcuts={
            "first-match": _py_solution("first-match"),
            "shell-glob": Solution(SHELL_GLOB, _shell_glob),
            "reset-then-apply": Solution(RESET_THEN_APPLY, _reset_then_apply),
        },
        params={"files": len(layout), "rules": len(rules)},
    )


FAMILY = Family(
    name="permissions-fix",
    version=1,
    cluster="filesystem",
    category="shell",
    skills=("shell", "permissions", "chmod", "globs"),
    difficulties={"easy": {"level": 0}, "medium": {"level": 1}, "hard": {"level": 2}},
    build=build,
)
