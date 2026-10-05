"""The filesystem families (cluster C): deterministic rendering, and their answers and traps
re-derived here from the rendered files, independently of each family's own solution models.

The modules are loaded by path, so these tests do not depend on the family registry."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import tarfile
from collections import defaultdict
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.family_lint import nondeterminism
from learning_loop.tasks.render import render

MODULES = ["organize_files", "dedupe_files", "bulk_rename", "extract_subset", "permissions_fix"]
SEEDS = [1, 2, 3]


def _path(module: str) -> Path:
    return REPO_ROOT / "evaluation" / "families" / f"{module}.py"


def _load(module: str):
    spec = importlib.util.spec_from_file_location(f"fs_family_{module}", _path(module))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAMILY


FAMILIES = {m: _load(m) for m in MODULES}


def _render(tmp_path: Path, module: str, difficulty: str, seed: int) -> tuple[Path, dict]:
    d = tmp_path / f"{module}-{difficulty}-{seed}"
    render(FAMILIES[module], difficulty, seed, d)
    return d / "environment" / "files", json.loads((d / "tests" / "key.json").read_text())


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("module", MODULES)
def test_clean_of_stray_randomness(module):
    assert nondeterminism(_path(module).read_text(), module) == []


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- organize-files


def _ticket_header(text: str) -> dict[str, str]:
    head = text.split("\n\n", 1)[0]
    out = {}
    for line in head.splitlines():
        k, v = line.split(":", 1)
        out.setdefault(k, v.strip().lower())
    return out


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_organize_files_answer_and_traps(tmp_path, difficulty, seed):
    files, key = _render(tmp_path, "organize_files", difficulty, seed)
    tickets = {n: t.decode() for n, t in _tree(files / "tickets").items()}
    assert all("/" not in n for n in tickets)
    queues = {}
    for name, text in tickets.items():
        h = _ticket_header(text)
        queues[name] = "urgent" if h.get("Priority") == "urgent" else h.get("Team") or "unassigned"
    assert key["expected"] == {f"{q}/{n}": {"sha256": _sha(tickets[n].encode())} for n, q in queues.items()}
    # by name: some ticket's name prefix is not its queue; ignoring urgency: an urgent ticket has a team
    assert any(n.split("-")[0] != q for n, q in queues.items())
    assert any(q == "urgent" and "Team" in _ticket_header(tickets[n]) for n, q in queues.items())
    if difficulty != "easy":
        assert "unassigned" in queues.values()
        body_lines = {n: t.split("\n\n", 1)[1].splitlines() for n, t in tickets.items()}
        # grep over the whole file: a Team line in the body of a ticket without one in its header,
        # and `Priority: urgent` in the body of a ticket whose header has no Priority
        assert any("Team" not in _ticket_header(t) and any(ln.startswith("Team:") for ln in body_lines[n]) for n, t in tickets.items())
        assert any("Priority" not in _ticket_header(t) and "Priority: urgent" in body_lines[n] for n, t in tickets.items())
    if difficulty == "hard":
        raw_team = {n: re.search(r"^Team:(.*)$", t.split("\n\n", 1)[0], re.M) for n, t in tickets.items()}
        assert any(m and m.group(1).strip() != m.group(1).strip().lower() and queues[n] != "urgent" for n, m in raw_team.items())
        spellings = defaultdict(set)  # one spelling per team: raw-value folders never differ only in case
        for m in raw_team.values():
            if m:
                spellings[m.group(1).strip().lower()].add(m.group(1).strip())
        assert all(len(v) == 1 for v in spellings.values())


# --------------------------------------------------------------------------- dedupe-files


def _snapshot_key(rel: str):
    d, m, y = re.fullmatch(r"backup-(\d\d)-(\d\d)-(\d{4})", rel.split("/")[0]).groups()
    return (int(y), int(m), int(d))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_dedupe_files_answer_and_traps(tmp_path, difficulty, seed):
    files, key = _render(tmp_path, "dedupe_files", difficulty, seed)
    tree = _tree(files / "backups")
    groups = defaultdict(list)
    for rel, data in tree.items():
        groups[data].append(rel)
    oldest = {data: sorted(paths, key=lambda r: (_snapshot_key(r), r.encode()))[0] for data, paths in groups.items()}
    assert key["expected"] == {rel: {"sha256": _sha(data)} for data, rel in oldest.items()}
    dups = {data: paths for data, paths in groups.items() if len(paths) > 1}
    assert dups  # deleting every copy loses files
    assert any(min(p, key=str.encode) != oldest[d] for d, p in dups.items())  # keeping the first path
    assert any(max(p, key=lambda r: (_snapshot_key(r), r)) != oldest[d] for d, p in dups.items())  # keeping the newest
    kept = list(oldest.values())
    assert len({r.rsplit("/", 1)[-1] for r in kept}) < len(kept)  # by name: two kept files share a name
    assert len({len(tree[r]) for r in kept}) < len(kept)  # by size: two kept files share a size
    names = sorted({r.split("/")[0] for r in tree})
    chronological = sorted(names, key=_snapshot_key)
    assert names[0] != chronological[0] and names[-1] != chronological[0]
    if difficulty != "easy":  # the tie rule decides inside the oldest snapshot
        assert any(len([r for r in p if r.split("/")[0] == chronological[0]]) > 1 for p in dups.values())


# --------------------------------------------------------------------------- bulk-rename


def _std(name: str) -> str:
    out, run = "", False
    for ch in name.lower():
        if ch in " _":
            if not run:
                out += "-"
            run = True
        else:
            out += ch
            run = False
    return out


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_bulk_rename_answer_and_traps(tmp_path, difficulty, seed):
    files, key = _render(tmp_path, "bulk_rename", difficulty, seed)
    tree = _tree(files / "docs")
    assert len({rel.lower() for rel in tree}) == len(tree)  # nothing lost on a case-insensitive file system
    folders = defaultdict(list)
    for rel in tree:
        folder, _, name = rel.rpartition("/")
        folders[folder].append(name)
    expected, standard_not_first = {}, False
    for folder, names in folders.items():
        assert any(n.startswith(".") and _std(n) != n for n in names)  # renaming hidden files changes them
        by_new = defaultdict(list)
        for n in sorted(names, key=str.encode):
            if n.startswith("."):
                expected[os.path.join(folder, n)] = tree[os.path.join(folder, n)]
            else:
                by_new[_std(n)].append(n)
        for new, olds in by_new.items():
            if len(olds) > 1 and new in olds[1:]:
                standard_not_first = True
            for i, old in enumerate(olds):
                stem, ext = new.rsplit(".", 1)
                target = new if i == 0 else f"{stem}-{i + 1}.{ext}"
                expected[os.path.join(folder, target)] = tree[os.path.join(folder, old)]
    assert len(expected) == len(tree)
    assert key["expected"] == {rel: {"sha256": _sha(data)} for rel, data in expected.items()}
    assert standard_not_first  # "add a suffix when the name is taken" numbers the wrong file
    assert any(len({_std(n) for n in names if not n.startswith(".")}) < len([n for n in names if not n.startswith(".")]) for names in folders.values())
    if difficulty != "easy":
        sub = [f for f in folders if f]
        assert sub and all(any(_std(n) != n for n in folders[f] if not n.startswith(".")) for f in sub)  # top level only
        assert any(_std(part) != part for f in sub for part in f.split("/"))  # renaming folders


# --------------------------------------------------------------------------- extract-subset


def _walk_archive(blob: bytes, prefix: str, out: dict, decoys: list, levels: list, depth: int = 0):
    """Collect the `.conf` files ({output path: (depth, bytes)}), the `conf` decoys and the
    inner archives of a gzipped tar, checking that it was built deterministically."""
    assert blob[4:8] == b"\0\0\0\0"  # gzip header mtime 0
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob)), mode="r:") as tf:
        members = tf.getmembers()
        assert [m.name for m in members] == sorted(m.name for m in members)
        assert all(m.uid == 0 and m.gid == 0 and m.mtime == members[0].mtime for m in members)
        for m in members:
            data = tf.extractfile(m).read()
            stem = re.sub(r"\.(tar\.gz|tgz)$", "", m.name)
            if stem != m.name:
                levels.append((depth + 1, m.name))
                _walk_archive(data, os.path.join(prefix, stem), out, decoys, levels, depth + 1)
            elif m.name.endswith(".conf"):
                out[os.path.join(prefix, m.name)] = (depth, data)
            elif "conf" in m.name:
                decoys.append((depth, m.name))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_extract_subset_answer_and_traps(tmp_path, difficulty, seed):
    files, key = _render(tmp_path, "extract_subset", difficulty, seed)
    assert sorted(_tree(files)) == ["bundle.tar.gz"]
    out, decoys, levels = {}, [], []
    _walk_archive((files / "bundle.tar.gz").read_bytes(), "", out, decoys, levels)
    assert key["expected"] == {rel: {"sha256": _sha(data)} for rel, (_, data) in out.items()}
    assert {0, 1} <= {d for d, _ in decoys}  # `*conf*` matches decoys in the bundle and inside archives
    assert any("/" in rel for rel in out)  # flattening changes paths
    assert any(depth >= 1 for depth, _ in out.values())  # skipping inner archives, or extracting them at the top
    basenames = [rel.rsplit("/", 1)[-1] for rel in out]
    if difficulty != "easy":
        assert len(set(basenames)) < len(basenames)
    if difficulty == "hard":
        assert any(name.endswith(".tgz") for _, name in levels)
        assert any(depth == 2 for depth, _ in levels)
        assert any(depth == 2 for depth, _ in out.values())  # a .conf two archives deep


# --------------------------------------------------------------------------- permissions-fix


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_permissions_fix_answer_and_traps(tmp_path, difficulty, seed):
    files, key = _render(tmp_path, "permissions_fix", difficulty, seed)
    site = files / "site"
    initial = {str(p.relative_to(site)): stat.S_IMODE(p.stat().st_mode) for p in site.rglob("*") if p.is_file()}
    rules = []
    for line in (files / "perms.policy").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            pattern, mode = line.split()
            rules.append((pattern, int(mode, 8)))
    assert not any(c in p for p, _ in rules for c in "?[")  # fnmatch semantics equal the stated ones
    hits = {path: [m for p, m in rules if fnmatch.fnmatchcase(path, p)] for path in initial}
    target = {path: h[-1] if h else initial[path] for path, h in hits.items()}
    assert key["expected"] == {path: {"sha256": _sha((site / path).read_bytes()), "mode": f"{m:04o}"} for path, m in target.items()}
    assert all(initial[p] != target[p] for p in initial if hits[p])  # doing nothing leaves every matched file wrong
    assert any(h and h[0] != h[-1] for h in hits.values())  # first match
    assert any(not h and initial[p] != 0o644 for p, h in hits.items())  # resetting everything to 0644
    # shell globbing: `*` stops at `/` and skips dotfiles, so some matched file is reached by no glob
    def bash_glob(pattern: str, path: str) -> bool:
        pp, ff = pattern.split("/"), path.split("/")
        return len(pp) == len(ff) and all(fnmatch.fnmatchcase(f, p) and not (f.startswith(".") and not p.startswith(".")) for p, f in zip(pp, ff))

    assert any(hits[p] and not any(bash_glob(pat, p) for pat, _ in rules) for p in initial)
    if difficulty != "easy":
        assert ".env" in initial and hits[".env"]
