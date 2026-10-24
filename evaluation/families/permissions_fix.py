"""permissions-fix: set file modes under /app/site from a glob -> mode policy.

/app/perms.policy has one `<pattern> <mode>` rule per line. A pattern matches a file's path
relative to /app/site, where `*` matches any characters including `/` and a leading `.`; the
last matching rule wins, and files no rule matches keep their mode. The grader is a FileTree
over /app/site with every file's content and mode. The instruction, including its examples of
`*` crossing `/` and matching a leading `.`, is the same at every difficulty.

Data per difficulty: easy has 10 files under `public/`, `bin/`, `scripts/`, `secrets/` and
`data/` plus `index.html`, and 5 rules (`public/*`, `*.sh`, `bin/*`, `secrets/*`,
`secrets/*.pub`); `scripts/` also holds a backup `<name>.sh.bak` that no rule matches
(`decoy`). Medium adds `public/css/`, `bin/tools/`, a hidden `.env` and `config/` files with
the rules `*.env` and `config/prod/*`, and route files `routes/index.js`, `routes/[id].js` and
`routes/[id]/edit.js` with the rules `routes/*` and then `routes/[id]*`, whose brackets are
literal characters (`brackets`). Hard adds `secrets/ssh/`, `logs/`, `vendor/` and `docs/`
files with the rules `logs/*` and `*.key`, and a comment line before each group of rules.

Traps and the shortcut that checks each:

- the rules go from general to specific, and some files match two rules with different modes
  (`secrets/*` then `secrets/*.pub`; on medium/hard `*.env` then `config/prod/*`; on hard
  `secrets/*` then `*.key`), so applying the first matching rule (`first-match`) fails;
- files that a rule matches only because `*` crosses `/` sit in subfolders (`scripts/x.sh`
  under `*.sh`; on medium/hard also `public/css/x.css` under `public/*`) and on medium/hard a
  hidden `.env` matches `*.env`, so running `chmod <mode> <pattern>` with shell globbing
  (`shell-glob`) fails;
- some files match no rule and have modes other than 0644, so resetting every file to 0644
  before applying the rules (`reset-then-apply`) fails;
- the backup `scripts/<name>.sh.bak` starts with `scripts/<name>.sh`, which `*.sh` matches,
  and its mode differs from that rule's, so matching a pattern against the start of the path
  instead of the whole path (`prefix-match`, `re.match` without an end anchor) fails;
- medium/hard: in `routes/[id]*` the brackets match themselves, but glob matchers (Python's
  `fnmatch`, `find -path`, bash `[[ == ]]`) read `[id]` as one character `i` or `d`, so they
  apply the rule to `routes/index.js` and miss the `[id]` files (`fnmatch`). Without bracketed
  patterns `fnmatch` has exactly the stated semantics, so the shortcut is declared only with
  `brackets`;
- every file starts with a mode other than its target, so doing nothing fails.

Every trap holds by construction, so the first draw is always used.
"""

import fnmatch

from learning_loop.tasks.spec import Family, FileState, FileTree, Solution, TaskSpec, tree_entry

SCRIPTS = ["deploy", "backup", "restart", "migrate", "healthcheck", "rotate-logs", "warm-cache"]
ASSETS = ["site", "main", "theme", "print", "app", "vendor"]
KEYS = ["api", "stripe", "smtp", "github", "s3", "deploy"]
DATA = ["cache.db", "sessions.db", "queue.db"]
MESSY = [0o600, 0o640, 0o644, 0o660, 0o664, 0o666, 0o700, 0o750, 0o755, 0o775, 0o777]
UNMATCHED_MODES = [0o600, 0o640, 0o660, 0o664, 0o666, 0o750, 0o775]  # never 0644

INSTRUCTION = """The deployment in `/app/site/` has messy file permissions. `/app/perms.policy` lists the modes its files must have, one rule per line: `<pattern> <mode>`, where the pattern is matched against a file's path relative to `/app/site/` (such as `bin/deploy.sh`) and the mode is octal. Lines starting with `#` and empty lines are comments.

- In a pattern, `*` matches any sequence of characters, including none, `/` and a leading `.`; every other character matches itself. A pattern must match the whole path.
- When several rules match a file, the **last** matching rule wins.
- A file that no rule matches keeps its current mode.

Set the mode of every regular file under `/app/site/`, in every subfolder, accordingly. Leave directory modes as they are, and do not change, add, remove or rename any file.
"""

SOLVER = '''
import fnmatch
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


def matches(pattern, path, mode="oracle"):
    """Whether `pattern` matches `path`: as stated (`*` is the only wildcard, the whole path
    must match), from the start only (`prefix-match`), or as a glob (`fnmatch`)."""
    if mode == "fnmatch":
        return fnmatch.fnmatchcase(path, pattern)
    regex = ".*".join(re.escape(part) for part in pattern.split("*"))
    if mode == "prefix-match":
        return re.match(regex, path, re.S) is not None
    return re.fullmatch(regex, path, re.S) is not None


def modes(policy, paths, mode="oracle"):
    """{path: new mode} for the files some rule matches (`first-match` keeps the first rule)."""
    out = {}
    for path in paths:
        hits = [m for pattern, m in rules(policy) if matches(pattern, path, mode)]
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
    """What bash expands an unquoted pattern to (no dotglob, no nullglob): components matched
    one level at a time as globs (`*`, `[...]`), `*` not crossing `/` and not matching a leading
    `.`; a pattern that matches nothing stays as it is. Returns files and directories (the shell
    form chmods only the files)."""
    entries = set(paths) | {p.rsplit("/", k)[0] for p in paths for k in range(1, p.count("/") + 1)}
    current = [""]
    for comp in pattern.split("/"):
        nxt = []
        for base in current:
            for e in sorted(entries):
                parent, _, name = e.rpartition("/")
                if parent != base:
                    continue
                if any(c in comp for c in "*?["):
                    ok = fnmatch.fnmatchcase(name, comp) and not (name.startswith(".") and not comp.startswith("."))
                else:
                    ok = name == comp
                if ok:
                    nxt.append(e)
        current = nxt
    return current or ([pattern] if pattern in entries else [])


def _shell_glob(files):
    site = _site(files)
    out = {}
    for pattern, m in _NS["rules"](files["perms.policy"].decode()):
        for e in _glob(list(site), pattern):
            if e in site:
                out[e] = m
    return _states(files, out)


def _reset_then_apply(files):
    """`find -path` matches like `fnmatch`: `*` crosses `/`, and `[...]` is a character class."""
    site = _site(files)
    new = {path: 0o644 for path in site}
    new.update(_NS["modes"](files["perms.policy"].decode(), list(site), "fnmatch"))
    return _states(files, new)


def _layout(rng, level, decoy, brackets):
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
    if decoy:
        files[f"scripts/{s[2]}.sh.bak"] = "script"
    if brackets:
        files.update({"routes/index.js": "page", "routes/[id].js": "page", "routes/[id]/edit.js": "page"})
        rules += [("routes/*", rng.choice([0o644, 0o444])), ("routes/[id]*", rng.choice([0o640, 0o600]))]
    if level >= 1:
        files.update({f"public/css/{a[2]}.css": "asset", f"bin/tools/{s[3]}.sh": "script", ".env": "env", "config/app.env": "env", "config/prod/app.env": "env"})
        rules += [("*.env", rng.choice([0o640, 0o660])), ("config/prod/*", rng.choice([0o600, 0o400]))]
    if level >= 2:
        files.update({f"secrets/ssh/{k[2]}.key": "secret", f"secrets/ssh/{k[2]}.pub": "public-key", "logs/app.log": "log", "logs/archive/app-1.log": "log", "vendor/bin/tool": "data", "docs/README.md": "page"})
        rules += [("logs/*", rng.choice([0o640, 0o600])), ("*.key", 0o400)]
    return files, rules


CATEGORIES = {"public/*": "web assets", "routes/*": "routes", "routes/[id]*": "routes", "*.sh": "scripts", "bin/*": "scripts", "secrets/*": "secrets", "secrets/*.pub": "secrets", "*.env": "environment files", "config/prod/*": "environment files", "logs/*": "logs", "*.key": "private keys"}


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
    layout, rules = _layout(rng, p["level"], p["decoy"], p["brackets"])
    policy = _policy(rules, comments=p["level"] >= 2)
    target = _NS["modes"](policy, list(layout))
    files, modes = {"perms.policy": policy}, {"perms.policy": 0o644}
    expected = {}
    for path in sorted(layout):
        content = _content(rng, path, layout[path])
        if path in target:
            mode = rng.choice([m for m in MESSY if m != target[path]])
        else:  # never the mode a rule matching only the start of the path would set
            mode = rng.choice([m for m in UNMATCHED_MODES if m != _NS["modes"](policy, [path], "prefix-match").get(path)])
        files[f"site/{path}"], modes[f"site/{path}"] = content, mode
        expected[path] = tree_entry(content, target.get(path, mode))
    shortcuts = {
        "first-match": _py_solution("first-match"),
        "shell-glob": Solution(SHELL_GLOB, _shell_glob),
        "reset-then-apply": Solution(RESET_THEN_APPLY, _reset_then_apply),
    }
    if p["decoy"]:
        shortcuts["prefix-match"] = _py_solution("prefix-match")
    if p["brackets"]:
        shortcuts["fnmatch"] = _py_solution("fnmatch")
    return TaskSpec(
        instruction=INSTRUCTION,
        files=files,
        modes=modes,
        grader=FileTree("/app/site", expected),
        oracle=_py_solution("oracle"),
        shortcuts=shortcuts,
        params={"files": len(layout), "rules": len(rules)},
    )


FAMILY = Family(
    name="permissions-fix",
    version=4,
    cluster="filesystem",
    category="shell",
    skills=("shell", "permissions", "chmod", "globs"),
    difficulties={
        "easy": {"level": 0, "decoy": True, "brackets": False},
        "medium": {"level": 1, "decoy": True, "brackets": True},
        "hard": {"level": 2, "decoy": True, "brackets": True},
    },
    build=build,
)
