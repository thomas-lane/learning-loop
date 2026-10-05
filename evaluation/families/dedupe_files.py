"""dedupe-files: delete duplicate files across dated backup snapshots, keeping the oldest copy.

/app/backups holds snapshot folders `backup-DD-MM-YYYY`, each a copy of some documents at that
date. Files are duplicates when their bytes are identical, whatever their names. From each
set of duplicates the copy in the oldest snapshot stays (by the date in the folder name),
ties within that snapshot going to the path that sorts first; every other copy is deleted.
The grader is a FileTree over /app/backups: exactly the surviving files, unchanged.

Construction: the oldest snapshot is in late August and the others follow it, one later in
August and the rest in early September, so with day-first names the oldest snapshot sorts
neither first nor last. A planted document is unchanged in every snapshot, so keeping the
copy whose path sorts first, keeping the newest copy and deleting every copy all fail. A
second planted document is edited by changing one digit, so two versions share a name and a
size: deduplicating by name or by size deletes a version that must stay. Other documents are
edited, added or missing at random. On medium/hard the oldest snapshot also holds a second
copy of one document, so the tie rule matters. Every trap holds by construction, so the first
draw is always used.
"""

from datetime import datetime, timedelta

from learning_loop.tasks.spec import Family, FileTree, Solution, TaskSpec, tree_entry

DOCS = {
    "flat": ["todo.txt", "budget.csv", "notes.md", "contacts.csv", "config.ini", "plan.txt", "recipes.md", "hosts.txt"],
    "nested": ["home/todo.txt", "home/notes.md", "work/budget.csv", "work/plan.txt", "etc/config.ini", "etc/hosts.txt", "home/recipes.md", "work/contacts.csv", "work/minutes.md"],
    "deep": [
        "home/notes/todo.txt",
        "home/notes/ideas.md",
        "home/recipes.md",
        "work/reports/q2.csv",
        "work/reports/q3.csv",
        "work/plan.txt",
        "work/team/contacts.csv",
        "etc/app/config.ini",
        "etc/app/hosts.txt",
        "etc/backup.ini",
        "work/team/minutes.md",
    ],
}
WORDS = ["alpha", "boiler", "cable", "dentist", "invoice", "garden", "laptop", "meeting", "renewal", "server", "ticket", "visa"]

INSTRUCTION = """`/app/backups/` holds backup snapshots, one folder per snapshot named `backup-DD-MM-YYYY` after the day it was taken (day, month, year: `backup-05-09-2026` is 5 September 2026). Each snapshot is a copy of some files as they were on that day.

Free up space by removing duplicate files. Two files are duplicates when their contents are byte-for-byte identical, whatever their names or folders. From each set of duplicates keep exactly one copy: the one in the **oldest** snapshot (by the date in the folder name); if that snapshot holds more than one copy, keep the one whose path sorts first in byte order (as `LC_ALL=C sort` orders them). Delete every other copy. Files whose contents are unique stay.
{note}
Change nothing else: do not move, rename or edit any file.
"""
NOTES = {
    "easy": "\nA name says nothing about the contents: a file can keep its name while its contents change from one snapshot to the next.\n",
    "medium": "\nA name says nothing about the contents: a file can keep its name while its contents change from one snapshot to the next.\n",
    "hard": "",
}

SOLVER = '''
import re


def snapshot_date(rel):
    d, m, y = re.fullmatch(r"backup-(\\d\\d)-(\\d\\d)-(\\d{4})", rel.split("/")[0]).groups()
    return (int(y), int(m), int(d))


def keep(files, mode):
    """files: {path relative to /app/backups: bytes}. Returns the set of paths that stay."""
    groups = {}
    for rel in sorted(files):
        groups.setdefault(files[rel], []).append(rel)
    if mode == "keep-newest":
        pick = lambda paths: min(paths, key=lambda r: (tuple(-x for x in snapshot_date(r)), r.encode()))
    else:
        pick = lambda paths: min(paths, key=lambda r: (snapshot_date(r), r.encode()))
    return {pick(paths) for paths in groups.values()}
'''

_PY_SHELL = """python3 - <<'PY'
{solver}
import os

root = "/app/backups"
files = {{}}
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        path = os.path.join(dirpath, name)
        if os.path.isfile(path) and not os.path.islink(path):
            with open(path, "rb") as f:
                files[os.path.relpath(path, root)] = f.read()
stay = keep(files, {mode!r})
for rel in sorted(files):
    if rel not in stay:
        os.remove(os.path.join(root, rel))
PY
"""

_FIND = "find /app/backups -type f"
BY_NAME = _FIND + " | sort | awk -F/ 'seen[$NF]++' | xargs -r rm --\n"
BY_SIZE = _FIND + " -printf '%s %p\\n' | sort -k2 | awk 'seen[$1]++ { print $2 }' | xargs -r rm --\n"
KEEP_FIRST_PATH = _FIND + " -exec sha256sum {} + | sort | awk 'seen[$1]++ { print $2 }' | xargs -r rm --\n"
DELETE_ALL = _FIND + " -exec sha256sum {} + | sort | uniq -w64 -D | cut -c67- | xargs -r rm --\n"

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of the Python solutions


def _backups(files):
    return {rel[len("backups/"):]: data for rel, data in files.items() if rel.startswith("backups/")}


def _deleted(files, stay):
    return {f"/app/backups/{rel}": None for rel in _backups(files) if rel not in stay}


def _py_solution(mode):
    return Solution(_PY_SHELL.format(solver=SOLVER, mode=mode), lambda f: _deleted(f, _NS["keep"](_backups(f), mode)))


def _first_per(files, key):
    """Paths that survive `sort | awk 'seen[key]++' | rm`: the first path (byte order) per key."""
    seen, stay = set(), set()
    for rel in sorted(files, key=str.encode):
        if key(rel) not in seen:
            seen.add(key(rel))
            stay.add(rel)
    return stay


def _delete_all(files):
    counts = {}
    for data in files.values():
        counts[data] = counts.get(data, 0) + 1
    return {rel for rel, data in files.items() if counts[data] == 1}


def _shell_solution(shell, stay_of):
    return Solution(shell, lambda f: _deleted(f, stay_of(_backups(f))))


def _content(rng, path):
    ext = path.rsplit(".", 1)[1]
    n = rng.randint(3, 7)
    if ext == "csv":
        rows = [f"2026-{rng.randint(1, 8):02d}-{rng.randint(10, 28)},{rng.choice(WORDS)},{rng.randint(100, 9999)}.{rng.randint(10, 99)}" for _ in range(n)]
        return "date,item,amount\n" + "\n".join(rows) + "\n"
    if ext == "ini":
        return "[main]\n" + "\n".join(f"{rng.choice(WORDS)}_{i} = {rng.randint(10, 99999)}" for i in range(n)) + "\n"
    if ext == "md":
        return f"# {rng.choice(WORDS).title()}\n\n" + "\n".join(f"- {rng.choice(WORDS)} {rng.choice(WORDS)} x{rng.randint(10, 999)}" for _ in range(n)) + "\n"
    return "\n".join(f"{rng.choice(WORDS)} {rng.choice(WORDS)} {rng.randint(10, 9999)}" for _ in range(n)) + "\n"


def _same_size_edit(rng, text):
    """Change one digit, so the size stays the same."""
    spots = [i for i, ch in enumerate(text) if ch.isdigit()]
    i = rng.choice(spots)
    return text[:i] + rng.choice([d for d in "0123456789" if d != text[i]]) + text[i + 1 :]


def _grow_edit(rng, text):
    return text + f"{rng.choice(WORDS)} {rng.choice(WORDS)} {rng.randint(10, 9999)}\n"


def _dates(rng, n):
    first = datetime(2026, 8, rng.randint(18, 24))
    second = first + timedelta(days=rng.randint(2, 31 - first.day))
    rest = sorted(rng.sample(range(1, 18), n - 2))
    return [first, second] + [datetime(2026, 9, d) for d in rest]


def build(ctx):
    rng, p = ctx.rng, ctx.params
    dates = _dates(rng, p["snapshots"])
    snaps = [f"backup-{d.day:02d}-{d.month:02d}-2026" for d in dates]  # chronological order
    pool = rng.sample(DOCS[p["layout"]], p["docs"])
    planted_same, planted_edit = pool[0], pool[1]
    current = {doc: _content(rng, doc) for doc in pool}
    start = {doc: 0 if doc in (planted_same, planted_edit) or rng.random() < 0.7 else rng.randrange(1, len(snaps)) for doc in pool}
    edit_at = rng.randrange(1, len(snaps))
    files = {}
    for i, snap in enumerate(snaps):
        for doc in pool:
            if i < start[doc]:
                continue
            if doc == planted_edit and i == edit_at:
                current[doc] = _same_size_edit(rng, current[doc])
            elif doc not in (planted_same, planted_edit) and i > start[doc] and rng.random() < 0.35:
                current[doc] = (_same_size_edit if rng.random() < 0.5 else _grow_edit)(rng, current[doc])
            if doc in (planted_same, planted_edit) or rng.random() < 0.85:
                files[f"{snap}/{doc}"] = current[doc]
    if p["inner_copy"]:  # a second copy inside the oldest snapshot, so the tie rule decides
        doc = rng.choice([d for d in pool if f"{snaps[0]}/{d}" in files])
        folder, _, name = doc.rpartition("/")
        alt = rng.choice([f"old/{name}", f"copy-of-{name}", f"{name.rsplit('.', 1)[0]}-copy.{name.rsplit('.', 1)[1]}"])
        files[f"{snaps[0]}/{folder + '/' if folder else ''}{alt}"] = files[f"{snaps[0]}/{doc}"]
    as_bytes = {rel: text.encode() for rel, text in files.items()}
    stay = _NS["keep"](as_bytes, "oracle")
    return TaskSpec(
        instruction=INSTRUCTION.format(note=NOTES[ctx.difficulty]),
        files={f"backups/{rel}": text for rel, text in files.items()},
        grader=FileTree("/app/backups", {rel: tree_entry(as_bytes[rel]) for rel in sorted(stay)}),
        oracle=_py_solution("oracle"),
        shortcuts={
            "by-name": _shell_solution(BY_NAME, lambda f: _first_per(f, lambda rel: rel.rsplit("/", 1)[-1])),
            "by-size": _shell_solution(BY_SIZE, lambda f: _first_per(f, lambda rel: len(f[rel]))),
            "keep-first-path": _shell_solution(KEEP_FIRST_PATH, lambda f: _first_per(f, lambda rel: f[rel])),
            "delete-all-copies": _shell_solution(DELETE_ALL, _delete_all),
            "keep-newest": _py_solution("keep-newest"),
        },
        params={"snapshots": snaps, "docs": len(pool), "files": len(files), "kept": len(stay)},
    )


FAMILY = Family(
    name="dedupe-files",
    version=1,
    cluster="filesystem",
    category="shell",
    skills=("shell", "files", "hashing", "dates"),
    difficulties={
        "easy": {"snapshots": 3, "docs": 5, "layout": "flat", "inner_copy": False},
        "medium": {"snapshots": 4, "docs": 7, "layout": "nested", "inner_copy": True},
        "hard": {"snapshots": 5, "docs": 9, "layout": "deep", "inner_copy": True},
    },
    build=build,
)
