"""broken-symlinks: find the broken symlinks under /app/srv and the existing file each one
should point to now, following the rename log.

Construction: a deployment tree of files in `/app/srv/` and symlinks (`TaskSpec.symlinks`)
to some of them, relative or absolute. A history of renames, oldest first, is written to
`/app/renames.log` (`<timestamp> <file|dir> <old> -> <new>`, paths relative to /app/srv).
Each broken link points to a path the history renamed away; the intended file is what
replaying the log on that path gives, and it always exists. Every old path is unique in the
history, so the replay is unambiguous. Unrelated renames are mixed in.

Traps, by construction: at least one broken link sits in a subdirectory (checking only the
top level fails, e.g. `find -maxdepth 1 -xtype l`), working links exist (reporting every
link fails), and the raw link text never names the intended file (reporting the dangling
target fails). Medium/hard: some files were renamed two or three times (following one
rename fails), and some relative targets start with `../` from a subdirectory (resolving a
relative target from /app/srv instead of the link's directory fails). Hard: a directory
rename moved a linked file, and the file was renamed again afterwards (ignoring directory
renames fails).

Every solution is one Python source (`SOLVER`) run with a mode, as in csv-revenue. The
model form gets the symlinks from the spec, since the files a model reads are regular files.
"""

import posixpath
from datetime import datetime, timedelta

from learning_loop.tasks.spec import Family, ParsedAnswer, Reject, Solution, TaskSpec

ROOT = "srv"
DIRS = {
    "assets/images": ["logo.png", "banner.jpg", "icon.svg", "hero.webp"],
    "assets/css": ["site.css", "print.css", "theme.css"],
    "docs/guides": ["install.md", "upgrade.md", "faq.md"],
    "conf": ["app.yml", "db.yml", "cache.yml"],
    "shared/data": ["regions.csv", "rates.json", "holidays.txt"],
    "releases/stable": ["notes.txt", "checksums.txt", "manifest.json"],
    "bin": ["deploy.sh", "backup.sh", "migrate.sh"],
}
DIR_RENAMES = {"assets/images": "assets/img", "docs/guides": "docs/howto", "shared/data": "shared/datasets", "releases/stable": "releases/rc"}
LINK_DIRS = ["", "www", "www/static", "current", "etc/app"]
SUFFIXES = ["-old", "-v1", "-v2", "-2025", "-draft", "-bak", "-prev"]
OLD_STEMS = ["main", "default", "base", "legacy", "current", "primary", "common", "core", "latest", "stable", "local"]

INSTRUCTION = """The deployment tree `/app/srv/` uses symlinks to point at shared files, and some of them broke when files{dirs} were renamed. Every rename is recorded in `/app/renames.log`, oldest first, one per line: `<timestamp> <file|dir> <old path> -> <new path>`, with paths relative to `/app/srv`.{dir_rule}

Find every broken symlink under `/app/srv/`, at any depth, and the existing file it should point to now: take the path the link points to (a relative target is relative to the link's own directory) and follow the renames in the log to the file that path became.

Write a JSON object to `/app/answer.json` that maps each broken symlink's path to that file's path, both relative to `/app/srv` (for example `{{"site/a.png": "media/b.png"}}`). Leave out symlinks that work.
"""

SOLVER = r'''
import posixpath

ROOT = "/app/srv"


def parse_log(text):
    """[(kind, old, new)] in log order."""
    events = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[1] in ("file", "dir") and parts[3] == "->":
            events.append((parts[1], parts[2], parts[4]))
    return events


def replay(path, events, mode):
    for kind, old, new in events:
        if kind == "file" and path == old:
            path = new
        elif kind == "dir" and mode != "file-only" and path.startswith(old + "/"):
            path = new + path[len(old):]
        else:
            continue
        if mode == "one-step":
            break
    return path


def resolve(link, target, mode):
    """The path (relative to the root) that a link's target text names."""
    if target.startswith("/"):
        return posixpath.normpath(target[len(ROOT) + 1:]) if target.startswith(ROOT + "/") else target
    if mode == "root-relative":
        return posixpath.normpath(target)
    return posixpath.normpath(posixpath.join(posixpath.dirname(link), target))


def is_file(tree, path, depth=0):
    entry = tree.get(path)
    if entry is None or depth > 20:
        return False
    if entry[0] == "link":
        return is_file(tree, resolve(path, entry[1], "oracle"), depth + 1)
    return entry[0] == "file"


def answer(tree, log, mode):
    """tree: {path relative to the root: ("file",) | ("link", target text)}. Returns {link: file}."""
    events = parse_log(log)
    out = {}
    for path in sorted(tree):
        entry = tree[path]
        if entry[0] != "link" or (mode == "top-level-only" and "/" in path):
            continue
        broken = not is_file(tree, path)
        if not broken and mode != "all-links":
            continue
        if mode == "dangling-text":
            out[path] = entry[1]
            continue
        dest = resolve(path, entry[1], mode)
        out[path] = replay(dest, events, mode) if broken else dest
    return out
'''

_SHELL = """python3 - <<'PY'
{solver}
import json
import os

tree = {{}}
for d, dirs, names in os.walk(ROOT):
    for n in dirs + names:
        p = os.path.join(d, n)
        rel = os.path.relpath(p, ROOT)
        if os.path.islink(p):
            tree[rel] = ("link", os.readlink(p))
        elif os.path.isfile(p):
            tree[rel] = ("file",)
result = answer(tree, open("/app/renames.log").read(), {mode!r})
with open("/app/answer.json", "w") as f:
    f.write(json.dumps(result, indent=2, sort_keys=True) + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _render(result):
    import json

    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def _solution(mode, links):
    def model(files):
        tree = {rel[len(ROOT) + 1:]: ("file",) for rel in files if rel.startswith(ROOT + "/")}
        tree.update({rel[len(ROOT) + 1:]: ("link", target) for rel, target in links.items()})
        return {"/app/answer.json": _render(_NS["answer"](tree, files["renames.log"].decode(), mode))}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def build(ctx):
    rng, p = ctx.rng, ctx.params
    dirs = rng.sample(sorted(DIRS), p["dirs"])
    dir_renamed = rng.choice([d for d in dirs if d in DIR_RENAMES]) if p["dir_rename"] else None
    if p["dir_rename"] and dir_renamed is None:
        raise Reject("no renameable directory")
    final = [f"{d}/{n}" for d in dirs for n in rng.sample(DIRS[d], rng.randint(2, len(DIRS[d])))]
    taken = set(final) | {d for f in final for d in [posixpath.dirname(f)]}
    if dir_renamed:
        taken.add(DIR_RENAMES[dir_renamed])

    def older(path):
        """An unused earlier name for the file `path`: its stem with a suffix, or another stem."""
        stem, dot, ext = posixpath.basename(path).partition(".")
        cands = [f"{stem}{s}" for s in SUFFIXES] + [s for s in OLD_STEMS if s != stem]
        rng.shuffle(cands)
        for c in cands:
            cand = posixpath.join(posixpath.dirname(path), f"{c}{dot}{ext}")
            if cand not in taken:
                taken.add(cand)
                return cand
        raise Reject("no free old name")

    # Choose the broken links' files, then give each a rename chain ending at it.
    n_broken = p["broken"]
    if n_broken > len(final):
        raise Reject("not enough files")
    broken_files = rng.sample(final, n_broken)
    if dir_renamed and not any(posixpath.dirname(f) == dir_renamed for f in broken_files):
        broken_files[0] = rng.choice([f for f in final if posixpath.dirname(f) == dir_renamed and f not in broken_files])
    lengths = [rng.randint(1, p["max_chain"]) for _ in broken_files]
    if p["max_chain"] > 1 and max(lengths) < 2:
        lengths[0] = rng.randint(2, p["max_chain"])
    via_dir = None
    if dir_renamed:
        inside = [i for i, f in enumerate(broken_files) if posixpath.dirname(f) == dir_renamed]
        if not inside:
            raise Reject("no broken link into the renamed directory")
        via_dir = inside[0]
    chains = []  # per broken file: [p0, p1, ..., final]
    for f, n in zip(broken_files, lengths):
        chain = [f]
        for _ in range(n):
            chain.insert(0, older(f))
        chains.append(chain)
    # Unrelated renames of files no link points to.
    noise = []
    for f in rng.sample([x for x in final if x not in broken_files], min(p["noise"], len(final) - n_broken)):
        noise.append([older(f), f])

    # Events, oldest first: the directory rename comes first, so every later file rename inside
    # it already uses the new name; each chain's steps keep their order.
    histories = chains + noise
    slots = [k for k, c in enumerate(histories) for _ in range(len(c) - 1)]
    rng.shuffle(slots)  # interleave the chains; each chain's own steps stay in order
    pos = [0] * len(histories)
    events = []
    for k in slots:
        c = histories[k]
        events.append(("file", c[pos[k]], c[pos[k] + 1]))
        pos[k] += 1
    if dir_renamed:
        events.insert(0, ("dir", DIR_RENAMES[dir_renamed], dir_renamed))

    # The path each broken link was made to point to: its chain's first name, with the
    # renamed directory's old name for the file whose link predates the directory rename.
    origins = []
    for i, chain in enumerate(chains):
        p0 = chain[0]
        if i == via_dir:
            p0 = DIR_RENAMES[dir_renamed] + p0[len(dir_renamed):]
        origins.append(p0)

    links = {}
    link_dirs = list(p["link_dirs"])

    def place(target_rel, sub, absolute):
        base = posixpath.basename(target_rel)
        for d in rng.sample(link_dirs, len(link_dirs)) if sub is None else [sub]:
            path = posixpath.join(d, base) if d else base
            if f"{ROOT}/{path}" not in links and path not in taken:
                taken.add(path)
                text = f"/app/{ROOT}/{target_rel}" if absolute else posixpath.relpath(target_rel, d or ".")
                links[f"{ROOT}/{path}"] = text
                return path
        raise Reject("no free link path")

    broken_paths = []
    deep = [d for d in link_dirs if "/" in d] or [d for d in link_dirs if d]
    for i, p0 in enumerate(origins):
        if i == 0:
            sub = rng.choice(deep)  # a broken link in a subdirectory, relative with ../
            absolute = False
        else:
            sub = None
            absolute = rng.random() < p["absolute_rate"]
        broken_paths.append(place(p0, sub, absolute))
    for f in rng.sample([x for x in final if x not in broken_files] or final, min(p["working"], len(final))):
        place(f, None, rng.random() < p["absolute_rate"])
    if not any("/" in x for x in broken_paths):
        raise Reject("no broken link in a subdirectory")

    files = {f"{ROOT}/{f}": f"# {f}\n{rng.randint(1000, 9999)}\n" for f in final}
    start = datetime(2026, 6, 1) + timedelta(days=rng.randint(0, 30))
    log = []
    for kind, old, new in events:
        start += timedelta(days=rng.randint(1, 9), seconds=rng.randint(0, 86399))
        log.append(f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')} {kind} {old} -> {new}")
    files["renames.log"] = "\n".join(log) + "\n"

    expected = {bp: c[-1] for bp, c in zip(broken_paths, chains)}
    oracle = _solution("oracle", links)
    got = _NS["answer"]({**{f: ("file",) for f in final}, **{k[len(ROOT) + 1:]: ("link", v) for k, v in links.items()}}, files["renames.log"], "oracle")
    if got != expected:
        raise Reject("the oracle disagrees with the construction")
    shortcuts = {m: _solution(m, links) for m in ("top-level-only", "dangling-text", "all-links")}
    if p["max_chain"] > 1:
        shortcuts["one-step"] = _solution("one-step", links)
        shortcuts["root-relative"] = _solution("root-relative", links)
    if dir_renamed:
        shortcuts["file-only"] = _solution("file-only", links)
    return TaskSpec(
        instruction=INSTRUCTION.format(
            dirs=" and directories" if dir_renamed else "",
            dir_rule=" Renaming a directory moves everything inside it." if dir_renamed else "",
        ),
        files=files,
        symlinks=links,
        grader=ParsedAnswer("/app/answer.json", "json", expected),
        oracle=oracle,
        shortcuts=shortcuts,
        params={"broken": len(expected), "links": len(links), "renames": len(events), "dir_rename": dir_renamed},
    )


FAMILY = Family(
    name="broken-symlinks",
    version=1,
    cluster="diagnosis",
    category="shell",
    skills=("shell", "symlinks", "filesystem", "diagnosis"),
    difficulties={
        "easy": {"dirs": 3, "broken": 2, "working": 2, "max_chain": 1, "noise": 1, "dir_rename": False, "absolute_rate": 0.0, "link_dirs": ["", "www"]},
        "medium": {"dirs": 4, "broken": 4, "working": 3, "max_chain": 3, "noise": 2, "dir_rename": False, "absolute_rate": 0.3, "link_dirs": ["", "www", "www/static", "current"]},
        "hard": {"dirs": 5, "broken": 5, "working": 4, "max_chain": 3, "noise": 3, "dir_rename": True, "absolute_rate": 0.3, "link_dirs": ["", "www", "www/static", "current", "etc/app"]},
    },
    build=build,
)
