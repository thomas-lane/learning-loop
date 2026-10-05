"""resolve-conflicts: resolve the merge-conflict blocks in service config files per stated rules.

Task: the files under `/app/config/` (INI-style service settings) contain conflict blocks
(`<<<<<<< HEAD` ... `=======` ... `>>>>>>> branch`). The instruction gives one rule per
section: `[release]` keeps ours (HEAD), `[dependencies]` keeps theirs, `[allowed_hosts]`
keeps both (ours, then the theirs lines not already in ours). Everything outside the blocks
stays byte for byte. Easy has one file (ExactAnswer on it); medium has three plus a
`README.md` and hard four plus the README, one of them in `legacy/` (FileTree over
`/app/config`, so a stray backup file also fails). Hard writes every block in diff3 style
(a `||||||| merged common ancestors` section with the base lines, which are never kept).

Construction: every instance has a `[release]` block whose sides differ (taking theirs
everywhere fails), a `[dependencies]` block whose sides differ (taking ours everywhere fails)
and an `[allowed_hosts]` block where both sides added one same host (concatenating the sides
duplicates it). The README is a Markdown file whose setext headings are underlined with `=`
lines, one exactly `=======`.

Traps (declared shortcuts, all failing by construction):
- `all-ours`, `all-theirs`: one side for every block;
- `keep-both`: `sed -i -E '/^(<<<<<<<|=======|>>>>>>>)/d'` over every file, which keeps both
  versions (and on medium/hard deletes the README's underlines; on hard it keeps the diff3
  base sections);
- `union-duplicates`: the right rules, but `[allowed_hosts]` keeps both sides without removing
  the repeated host;
- hard `base-as-ours`: a two-way parser that reads everything up to `=======` as ours,
  so the `|||||||` line and the base lines land in `[release]`;
- hard `top-level-only`: resolves `/app/config/*.conf` but not `legacy/`.

The resolvers are one Python source (`SOLVER`) run with a mode, used by both the shell and the
model forms; `keep-both` is the sed one-liner with a line-by-line model of it.
"""

import re

from learning_loop.tasks.spec import ExactAnswer, Family, FileTree, Solution, TaskSpec, tree_entry

SERVICES = ["billing", "orders", "search", "notify", "ledger", "gateway", "catalog", "auth", "media", "reports"]
CODENAMES = ["falcon", "heron", "otter", "lynx", "marten", "osprey", "badger", "ibis"]
PACKAGES = ["requests", "urllib3", "sqlalchemy", "pydantic", "redis", "celery", "boto3", "jinja2", "click", "attrs", "httpx", "gunicorn"]
HOSTS = ["api", "cdn", "admin", "status", "partners", "metrics", "auth", "uploads", "billing", "search", "docs", "hooks", "edge", "mail"]
BRANCHES = ["feature/upgrade-deps", "release/next", "feature/new-hosts", "hotfix/pin-versions", "feature/edge-routing"]

INSTRUCTION = """A merge left conflict blocks in {where}. Resolve every conflict in place.

A conflict block starts with a `<<<<<<< HEAD` line and ends with a `>>>>>>> <branch>` line. **Ours** are the lines after `<<<<<<< HEAD` up to the next marker line; **theirs** are the lines between the `=======` line and the `>>>>>>>` line.{diff3} Resolve each block by the section (`[name]` header) it is in:

- `[release]`: keep **ours**.
- `[dependencies]`: keep **theirs**.
- `[allowed_hosts]`: keep **both**: the lines from ours, followed by the lines from theirs that are not already in ours, each in their original order.

Remove every marker line and keep every line outside the conflict blocks exactly as it is. Leave no other files (such as backups) in `/app/config/`.
"""

DIFF3 = " Some blocks also show the common ancestor's lines, after a `||||||| merged common ancestors` line and before `=======`; those lines are never kept."

README = """Service configuration
=====================

One `.conf` file per service. Each file has three sections.

Release
=======

`[release]` holds the version that is deployed. Only the release manager changes it.

Dependencies
------------

`[dependencies]` pins the packages the service is built with, one `name==version` per line.

Hosts
-----

`[allowed_hosts]` lists the internal hosts the service may call, one per line.
"""

SOLVER = r'''
import re

RULES = {"release": "ours", "dependencies": "theirs", "allowed_hosts": "union"}


def _pick(section, ours, theirs, mode):
    if mode == "all-ours":
        return ours
    if mode == "all-theirs":
        return theirs
    rule = RULES[section]
    if rule == "ours":
        return ours
    if rule == "theirs":
        return theirs
    if mode == "union-duplicates":
        return ours + theirs
    seen = {ln.strip() for ln in ours}
    return ours + [ln for ln in theirs if ln.strip() not in seen]


def resolve(text, mode):
    out, section, state = [], None, None
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\n")
        if state is None:
            if bare.startswith("<<<<<<<"):
                state, ours, theirs = "ours", [], []
                continue
            m = re.match(r"\[([^\]]+)\]\s*$", bare)
            if m:
                section = m.group(1)
            out.append(line)
        elif state == "ours":
            if bare.startswith("|||||||") and mode != "base-as-ours":
                state = "base"
            elif bare == "=======":
                state = "theirs"
            else:
                ours.append(line)
        elif state == "base":
            if bare == "=======":
                state = "theirs"
        elif bare.startswith(">>>>>>>"):
            out += _pick(section, ours, theirs, mode)
            state = None
        else:
            theirs.append(line)
    return "".join(out)


def selected(rel, mode):
    """Whether this method touches the file at `rel` (relative to /app/config)."""
    if mode == "top-level-only":
        return "/" not in rel and rel.endswith(".conf")
    return True
'''

_SHELL = """python3 - <<'PY'
{solver}
import os

root = "/app/config"
for dirpath, dirnames, filenames in os.walk(root):
    dirnames.sort()
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        if not selected(os.path.relpath(path, root), {mode!r}):
            continue
        with open(path) as f:
            text = f.read()
        fixed = resolve(text, {mode!r})
        if fixed != text:
            with open(path, "w") as f:
                f.write(fixed)
PY
"""

KEEP_BOTH = "find /app/config -type f -exec sed -i -E '/^(<<<<<<<|=======|>>>>>>>)/d' {} +\n"

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _configs(files):
    return {rel[len("config/"):]: data.decode() for rel, data in files.items() if rel.startswith("config/")}


def _solution(mode):
    def model(files):
        out = {}
        for rel, text in _configs(files).items():
            if _NS["selected"](rel, mode):
                fixed = _NS["resolve"](text, mode)
                if fixed != text:
                    out[f"/app/config/{rel}"] = fixed
        return out

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def _keep_both(files):
    marker = re.compile(r"(<<<<<<<|=======|>>>>>>>)")
    out = {}
    for rel, text in _configs(files).items():
        kept = "".join(ln for ln in text.splitlines(keepends=True) if not marker.match(ln))
        if kept != text:
            out[f"/app/config/{rel}"] = kept
    return out


# --------------------------------------------------------------------------- #
# Generating the conflicted files
# --------------------------------------------------------------------------- #


def _ver(rng):
    return f"{rng.randint(1, 4)}.{rng.randint(0, 12)}.{rng.randint(0, 20)}"


def _other_ver(rng, v):
    while True:
        w = _ver(rng)
        if w != v:
            return w


def _block(rng, ours, theirs, base, diff3):
    lines = ["<<<<<<< HEAD"] + ours
    if diff3:
        lines += ["||||||| merged common ancestors"] + base
    return lines + ["======="] + theirs + [f">>>>>>> {rng.choice(BRANCHES)}"]


def _file(rng, service, kinds, diff3):
    """(conflicted text, resolved text) for one service file with conflicts in the sections `kinds`."""
    # release
    version, codename = _ver(rng), rng.choice(CODENAMES)
    rel_clean = [f"version = {version}", f"codename = {codename}", f"channel = {rng.choice(['stable', 'beta'])}"]
    rel_conf = list(rel_clean)
    if "release" in kinds:
        theirs_v = _other_ver(rng, version)
        ours, theirs, base = [f"version = {version}"], [f"version = {theirs_v}"], [f"version = {_other_ver(rng, version)}"]
        if rng.random() < 0.5:  # a two-line hunk: the codename changed too
            other = rng.choice([c for c in CODENAMES if c != codename])
            ours.append(f"codename = {codename}")
            theirs.append(f"codename = {other}")
            base.append(f"codename = {rng.choice(CODENAMES)}")
            rel_conf = _block(rng, ours, theirs, base, diff3) + rel_clean[2:]
        else:
            rel_conf = _block(rng, ours, theirs, base, diff3) + rel_clean[1:]
    # dependencies
    pkgs = rng.sample(PACKAGES, rng.randint(4, 6))
    pins = {p: _ver(rng) for p in pkgs}
    dep_clean = [f"{p}=={pins[p]}" for p in pkgs]
    dep_conf, dep_res = list(dep_clean), list(dep_clean)
    if "dependencies" in kinds:
        i = rng.randrange(len(pkgs))
        p = pkgs[i]
        newer = _other_ver(rng, pins[p])
        ours, theirs, base = [f"{p}=={pins[p]}"], [f"{p}=={newer}"], [f"{p}=={_other_ver(rng, newer)}"]
        res = [f"{p}=={newer}"]
        extra = rng.choice([q for q in PACKAGES if q not in pkgs])
        if rng.random() < 0.5:  # theirs also adds a package
            theirs.append(f"{extra}=={_ver(rng)}")
            res = list(theirs)
        dep_conf = dep_clean[:i] + _block(rng, ours, theirs, base, diff3) + dep_clean[i + 1 :]
        dep_res = dep_clean[:i] + res + dep_clean[i + 1 :]
    # allowed_hosts
    hosts = [f"{h}.{service}.internal" for h in rng.sample(HOSTS, rng.randint(6, 8))]
    fixed_n = rng.randint(1, 2)
    host_clean = hosts[:fixed_n]
    host_conf, host_res = list(host_clean), list(host_clean)
    if "allowed_hosts" in kinds:
        rest = hosts[fixed_n:]
        shared = rest[0]
        ours_only = rest[1 : 1 + rng.randint(1, 2)]
        theirs_only = rest[1 + len(ours_only) : 1 + len(ours_only) + rng.randint(1, 2)]
        ours = ours_only + [shared]
        rng.shuffle(ours)
        theirs = [shared] + theirs_only
        rng.shuffle(theirs)
        host_conf = host_clean + _block(rng, ours, theirs, [], diff3)
        host_res = host_clean + ours + [h for h in theirs if h not in ours]
    head = [f"# {service} service settings", ""]
    conflicted = head + ["[release]"] + rel_conf + ["", "[dependencies]"] + dep_conf + ["", "[allowed_hosts]"] + host_conf
    resolved = head + ["[release]"] + rel_clean + ["", "[dependencies]"] + dep_res + ["", "[allowed_hosts]"] + host_res
    return "\n".join(conflicted) + "\n", "\n".join(resolved) + "\n"


def _assign(rng, n_files):
    """Conflict kinds per file: every file has at least one, every kind appears somewhere."""
    kinds = ["release", "dependencies", "allowed_hosts"]
    if n_files == 1:
        return [set(kinds)]
    while True:
        plan = [{k for k in kinds if rng.random() < 0.6} for _ in range(n_files)]
        if all(plan) and all(any(k in s for s in plan) for k in kinds):
            return plan


def build(ctx):
    rng, p = ctx.rng, ctx.params
    services = rng.sample(SERVICES, p["n_files"])
    names = [f"{s}.conf" for s in services]
    if p["subdir"]:
        names[-1] = f"legacy/{names[-1]}"
    plan = _assign(rng, p["n_files"])
    files, resolved = {}, {}
    for name, service, kinds in zip(names, services, plan):
        files[f"config/{name}"], resolved[name] = _file(rng, service, kinds, p["diff3"])
    if p["readme"]:
        files["config/README.md"] = README
        resolved["README.md"] = README
    if p["n_files"] == 1:
        where = f"`/app/config/{names[0]}`"
        grader = ExactAnswer(f"/app/config/{names[0]}", resolved[names[0]].strip())
    else:
        where = "the files under `/app/config/`" + (" (including its subdirectories)" if p["subdir"] else "")
        grader = FileTree("/app/config", {rel: tree_entry(text) for rel, text in sorted(resolved.items())})
    shortcuts = {m: _solution(m) for m in ("all-ours", "all-theirs", "union-duplicates")}
    shortcuts["keep-both"] = Solution(KEEP_BOTH, _keep_both)
    if p["diff3"]:
        shortcuts["base-as-ours"] = _solution("base-as-ours")
    if p["subdir"]:
        shortcuts["top-level-only"] = _solution("top-level-only")
    return TaskSpec(
        instruction=INSTRUCTION.format(where=where, diff3=DIFF3 if p["diff3"] else ""),
        files=files,
        grader=grader,
        oracle=_solution("oracle"),
        shortcuts=shortcuts,
        params={k: p[k] for k in sorted(p)},
    )


FAMILY = Family(
    name="resolve-conflicts",
    version=1,
    cluster="config-repair",
    category="config",
    skills=("merge-conflicts", "text-editing", "config-repair"),
    difficulties={
        "easy": {"n_files": 1, "subdir": False, "diff3": False, "readme": False},
        "medium": {"n_files": 3, "subdir": False, "diff3": False, "readme": True},
        "hard": {"n_files": 4, "subdir": True, "diff3": True, "readme": True},
    },
    build=build,
)
