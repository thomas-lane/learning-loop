"""bulk-rename: give every file under /app/docs a standard name, resolving collisions by rule.

The standard name is the current name in lowercase with every run of spaces and underscores
replaced by one hyphen. Files are renamed in place (recursively, in every folder); folder
names and hidden files stay as they are. When several files in one folder map to the same
name, they are ordered by their current names in byte order: the first gets the name, the
others get `-2`, `-3`, ... before the extension. A file that already has the standard name
takes part like any other. The grader is a FileTree over /app/docs: exactly the renamed files.

Construction: every folder has collision groups, so renaming in a loop that overwrites
(`mv -f`) loses files and one that refuses to overwrite (`mv -n`) leaves names unchanged. One
group holds a file that already has the standard name and a spelling with a capital first
letter, which sorts before it, so "add a suffix if the name is taken" numbers the wrong
file. Names in a folder differ ignoring case, so rendering on a case-insensitive file system
loses nothing. A hidden file whose name would change (`.DS_Store`) sits in every folder, so renaming
hidden files fails. Medium/hard add folders whose names would change and whose files need
renaming, so renaming folders too, or only the top level, fails. `build` redraws when a
suffixed name would clash with another file's new name.
"""

from learning_loop.tasks.spec import Family, FileTree, Reject, Solution, TaskSpec, tree_entry

SLUGS = [
    "budget plan",
    "meeting notes",
    "travel claims",
    "project brief",
    "team roster",
    "q3 report",
    "client list",
    "price sheet",
    "launch checklist",
    "vendor contacts",
    "draft agenda",
    "release notes",
    "board minutes",
    "hiring plan",
    "expense summary",
    "office moves",
]
EXTS = ["txt", "md", "csv"]
SEPARATORS = [" ", "_", "  ", " _ ", "__"]
HIDDEN = [".DS_Store", ".Sync_Index"]
WORDS = ["agenda", "budget", "client", "deadline", "invoice", "review", "summary", "travel", "vendor"]

INSTRUCTION = """The files in `/app/docs/` and its subfolders have inconsistent names. Rename every file to the standard form:

- the whole name in lowercase;
- every run of spaces and/or underscores replaced by a single hyphen (`-`).

For example, `Project_Brief  v2.TXT` becomes `project-brief-v2.txt`. Each file stays in its folder, and folder names stay as they are. Hidden files (names starting with `.`) keep their names.

Collisions: when several files in the same folder would get the same new name, order them by their current names in byte order (as `LC_ALL=C sort` orders them, so capital letters come before lowercase ones). The first gets the new name, and the others get `-2`, `-3`, ... inserted before the extension, in that order: `Draft A.txt` and `draft_a.txt` become `draft-a.txt` and `draft-a-2.txt`.{note} File contents must not change, and no file may be lost.
"""
NOTES = {
    "easy": " A file that already has the standard name takes part in this ordering like any other.",
    "medium": " A file that already has the standard name takes part in this ordering like any other.",
    "hard": "",
}

SOLVER = '''
import re


def standard(name):
    return re.sub(r"[ _]+", "-", name.lower())


def suffixed(name, k):
    stem, _, ext = name.rpartition(".")
    return f"{stem}-{k}.{ext}"


def plan_folder(names, include_hidden=False):
    """{current name: new name} for the files of one folder."""
    groups = {}
    for name in sorted(names, key=str.encode):
        if include_hidden or not name.startswith("."):
            groups.setdefault(standard(name), []).append(name)
    out = {}
    for new, olds in groups.items():
        for i, old in enumerate(olds):
            out[old] = new if i == 0 else suffixed(new, i + 1)
    return out


def plan(paths, mode):
    """paths: file paths relative to /app/docs. Returns {old path: new path}."""
    folders = {}
    for rel in paths:
        folder, _, name = rel.rpartition("/")
        folders.setdefault(folder, []).append(name)
    out = {}
    for folder, names in folders.items():
        renames = {} if mode == "top-level-only" and folder else plan_folder(names, include_hidden=mode == "include-hidden")
        folder_new = folder
        if mode == "rename-folders" and folder:
            folder_new = "/".join(standard(part) for part in folder.split("/"))
        for old in names:
            new = renames.get(old, old)
            out[f"{folder}/{old}" if folder else old] = f"{folder_new}/{new}" if folder_new else new
    return out
'''

_PY_SHELL = """python3 - <<'PY'
{solver}
import os

root = "/app/docs"
paths = []
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        full = os.path.join(dirpath, name)
        if os.path.isfile(full) and not os.path.islink(full):
            paths.append(os.path.relpath(full, root))
moves = {{old: new for old, new in plan(paths, {mode!r}).items() if old != new}}
staged = []
for i, (old, new) in enumerate(sorted(moves.items())):  # two phases, so no rename overwrites a file
    tmp = os.path.join(root, os.path.dirname(old), ".rename-%d.tmp" % i)
    os.rename(os.path.join(root, old), tmp)
    staged.append((tmp, new))
for tmp, new in staged:
    os.makedirs(os.path.join(root, os.path.dirname(new)), exist_ok=True)
    os.rename(tmp, os.path.join(root, new))
for dirpath, dirnames, filenames in os.walk(root, topdown=False):  # folders emptied by renaming them
    if dirpath != root and not os.listdir(dirpath):
        os.rmdir(dirpath)
PY
"""

_LOOP = """cd /app/docs
find . -type f ! -name '.*' | sort | while IFS= read -r f; do
  d=$(dirname "$f"); b=$(basename "$f")
  n=$(printf '%s' "$b" | tr 'A-Z' 'a-z' | sed -E 's/[ _]+/-/g')
{body}done
"""
OVERWRITE = _LOOP.format(body='  [ "$b" != "$n" ] && mv -f "$f" "$d/$n"\n')
NO_CLOBBER = _LOOP.format(body='  [ "$b" != "$n" ] && mv -n "$f" "$d/$n"\n')
SUFFIX_IF_TAKEN = _LOOP.format(
    body='  [ "$b" = "$n" ] && continue\n  t="$d/$n"; i=2\n  while [ -e "$t" ]; do t="$d/${n%.*}-$i.${n##*.}"; i=$((i+1)); done\n  mv "$f" "$t"\n'
)

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of the Python solutions


def _docs(files):
    return {rel[len("docs/"):]: data for rel, data in files.items() if rel.startswith("docs/")}


def _result(files, final):
    """Artifacts for a final state {path under /app/docs: bytes}."""
    docs = _docs(files)
    out = {f"/app/docs/{rel}": None for rel in docs if rel not in final}
    out.update({f"/app/docs/{rel}": data for rel, data in final.items() if docs.get(rel) != data})
    return out


def _apply_plan(docs, moves):
    return {moves.get(rel, rel): data for rel, data in docs.items()}


def _py_solution(mode):
    def model(files):
        docs = _docs(files)
        return _result(files, _apply_plan(docs, _NS["plan"](list(docs), mode)))

    return Solution(_PY_SHELL.format(solver=SOLVER, mode=mode), model)


def _loop(files, how):
    """Simulate the `find | sort | while read` loop: each in-scope file in byte order of its
    path, `how` deciding the target (or None to leave it) from the current state."""
    state = dict(_docs(files))
    for rel in sorted((r for r in state if not r.rpartition("/")[2].startswith(".")), key=str.encode):
        if rel not in state:
            continue
        folder, _, name = rel.rpartition("/")
        new = _NS["standard"](name)
        if new == name:
            continue
        target = how(state, folder, new)
        if target is not None:
            state[target] = state.pop(rel)
    return state


def _join(folder, name):
    return f"{folder}/{name}" if folder else name


def _overwrite(state, folder, new):
    return _join(folder, new)


def _no_clobber(state, folder, new):
    return None if _join(folder, new) in state else _join(folder, new)


def _suffix_if_taken(state, folder, new):
    t, i = _join(folder, new), 2
    while t in state:
        t, i = _join(folder, _NS["suffixed"](new, i)), i + 1
    return t


def _variant(rng, slug, ext):
    words = []
    for w in slug.split():
        words.append(rng.choice([w, w.capitalize(), w.upper()]))
    out = words[0]
    for w in words[1:]:
        out += rng.choice(SEPARATORS) + w
    return out + "." + rng.choice([ext, ext, ext.upper()])


def _text(rng):
    return "\n".join(f"{rng.choice(WORDS)} {rng.choice(WORDS)} {rng.randint(10, 99999)}" for _ in range(rng.randint(2, 6))) + "\n"


def _folder_names(rng, folder, slugs, groups, singles, trap):
    names = []
    for k, size in enumerate(groups):
        slug, ext = slugs.pop(), rng.choice(EXTS)
        std = _NS["standard"](slug.replace(" ", "_") + "." + ext)
        # Names that differ only in letter case would overwrite each other on a case-insensitive
        # file system (e.g. when rendering on macOS), so a group's names differ ignoring case.
        group = {}
        if trap and k == 0:  # the standard name plus a spelling that sorts before it
            first = slug.capitalize().replace(" ", rng.choice(SEPARATORS)) + "." + ext
            group = {std: std, first.lower(): first}
        while len(group) < size:
            name = rng.choice([std, _variant(rng, slug, ext)])
            group.setdefault(name.lower(), name)
        names += sorted(group.values())
    for _ in range(singles):
        slug, ext = slugs.pop(), rng.choice(EXTS)
        names.append(_NS["standard"](slug + "." + ext) if rng.random() < 0.25 else _variant(rng, slug, ext))
    if folder and all(_NS["standard"](n) == n for n in names):
        raise Reject(f"no file in {folder!r} needs renaming")
    return names


def build(ctx):
    rng, p = ctx.rng, ctx.params
    slugs = rng.sample(SLUGS, sum(len(f["groups"]) + f["singles"] for f in p["folders"].values()))
    folders = {}
    for folder, spec in p["folders"].items():
        folders[folder] = _folder_names(rng, folder, slugs, spec["groups"], spec["singles"], trap=folder == "")
        folders[folder].append(rng.choice(HIDDEN))
    docs = {_join(folder, name): _text(rng) for folder, names in folders.items() for name in names}
    if len({rel.lower() for rel in docs}) != len(docs):
        raise Reject("two paths differ only in letter case")
    if len(set(docs.values())) != len(docs):
        raise Reject("two files have the same contents")
    moves = _NS["plan"](list(docs), "oracle")
    final = _apply_plan(docs, moves)
    if len(final) != len(docs):
        raise Reject("a suffixed name clashes with another file's new name")
    files = {f"docs/{rel}": text for rel, text in docs.items()}
    shortcuts = {
        "overwrite": Solution(OVERWRITE, lambda f: _result(f, _loop(f, _overwrite))),
        "no-clobber": Solution(NO_CLOBBER, lambda f: _result(f, _loop(f, _no_clobber))),
        "suffix-if-taken": Solution(SUFFIX_IF_TAKEN, lambda f: _result(f, _loop(f, _suffix_if_taken))),
        "include-hidden": _py_solution("include-hidden"),
    }
    if len(folders) > 1:
        shortcuts["rename-folders"] = _py_solution("rename-folders")
        shortcuts["top-level-only"] = _py_solution("top-level-only")
    return TaskSpec(
        instruction=INSTRUCTION.format(note=NOTES[ctx.difficulty]),
        files=files,
        grader=FileTree("/app/docs", {rel: tree_entry(text) for rel, text in sorted(final.items())}),
        oracle=_py_solution("oracle"),
        shortcuts=shortcuts,
        params={"folders": sorted(folders), "files": len(docs), "renamed": sum(1 for a, b in moves.items() if a != b)},
    )


FAMILY = Family(
    name="bulk-rename",
    version=1,
    cluster="filesystem",
    category="shell",
    skills=("shell", "files", "rename", "collisions"),
    difficulties={
        "easy": {"folders": {"": {"groups": [2, 2], "singles": 4}}},
        "medium": {"folders": {"": {"groups": [2, 3], "singles": 3}, "Old Drafts": {"groups": [2], "singles": 2}}},
        "hard": {
            "folders": {
                "": {"groups": [2, 3], "singles": 3},
                "Old Drafts": {"groups": [2], "singles": 2},
                "Clients": {"groups": [], "singles": 2},
                "Clients/Acme_Corp": {"groups": [3, 2], "singles": 1},
            }
        },
    },
    build=build,
)
