"""The filesystem families (cluster C): deterministic rendering, and their answers and traps
re-derived here from the rendered files, independently of each family's own solution models.

The modules are loaded by path, so these tests do not depend on the family registry."""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import stat
from collections import defaultdict
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.family_lint import nondeterminism
from learning_loop.tasks.generate import generate_spec
from learning_loop.tasks.render import render

MODULES = ["organize_files", "dedupe_files", "bulk_rename", "permissions_fix"]
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


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("seed", SEEDS)
def test_instruction_is_the_same_at_every_difficulty(module, seed):
    """Difficulty comes from the data alone: these instructions draw no instance values, so the
    text is identical at every level, and it states every rule and note any level needs."""
    fam = FAMILIES[module]
    texts = {d: generate_spec(fam, d, seed).spec.instruction for d in fam.difficulties}
    assert len(set(texts.values())) == 1, texts


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
    # every level: tickets without a Team line, and forwarded messages in bodies
    assert "unassigned" in queues.values()
    body_lines = {n: t.split("\n\n", 1)[1].splitlines() for n, t in tickets.items()}
    # grep over the whole file: a Team line in the body of a ticket without one in its header,
    # and `Priority: urgent` in the body of a ticket whose header has no Priority
    assert any("Team" not in _ticket_header(t) and any(ln.startswith("Team:") for ln in body_lines[n]) for n, t in tickets.items())
    assert any("Priority" not in _ticket_header(t) and "Priority: urgent" in body_lines[n] for n, t in tickets.items())
    heads = [t.split("\n\n", 1)[0] for t in tickets.values()]
    messy = any(re.search(r"^(Team|Priority):(?! \S)|^(Team|Priority):.*([A-Z]|\s$)", h, re.M) for h in heads)
    assert messy == (difficulty != "easy")  # spacing and letter case vary from medium on
    assert len(tickets) == {"easy": 10, "medium": 12, "hard": 20}[difficulty]
    if difficulty != "easy":
        raw_team = {n: re.search(r"^Team:(.*)$", t.split("\n\n", 1)[0], re.M) for n, t in tickets.items()}
        assert any(m and m.group(1).strip() != m.group(1).strip().lower() and queues[n] != "urgent" for n, m in raw_team.items())
        spellings = defaultdict(set)  # one spelling per team: raw-value folders never differ only in case
        for m in raw_team.values():
            if m:
                spellings[m.group(1).strip().lower()].add(m.group(1).strip())
        assert all(len(v) == 1 for v in spellings.values())


def _instruction(tmp_path: Path, module: str, difficulty: str, seed: int = 1) -> str:
    d = tmp_path / f"{module}-{difficulty}-{seed}-instruction"
    render(FAMILIES[module], difficulty, seed, d)
    return (d / "instruction.md").read_text()


@pytest.mark.parametrize("seed", SEEDS)
def test_organize_files_header_lines_match_the_stated_format(tmp_path, seed):
    """The instruction states the header format the data uses: `<field>:<value>` with optional
    spaces around the value (no `Name: value` placeholder an agent could take literally), and
    hard tickets do write values without a space after the colon."""
    text = _instruction(tmp_path, "organize_files", "hard", seed)
    assert "`<field>:<value>`" in text and "`Team:billing`" in text and "Name: value" not in text
    files, _ = _render(tmp_path, "organize_files", "hard", seed)
    heads = [t.decode().split("\n\n", 1)[0].splitlines() for t in _tree(files / "tickets").values()]
    assert all(re.fullmatch(r"[A-Z][a-z]+:.*\S.*", line) for head in heads for line in head)
    assert any(re.fullmatch(r"(Team|Priority):\S.*", line) for head in heads for line in head)


def test_organize_files_names_lowercase_folders(tmp_path):
    """Hard Priority and Team values vary in letter case; the instruction fixes the folder names
    in lowercase, so no agent needs two folders that differ only in case."""
    text = _instruction(tmp_path, "organize_files", "hard")
    assert "Queue folder names are always lowercase" in text
    files, key = _render(tmp_path, "organize_files", "hard", 1)
    raw = "".join(t.decode().split("\n\n", 1)[0] for t in _tree(files / "tickets").values())
    assert re.search(r"^(Team|Priority):\s*\S*[A-Z]", raw, re.M)
    assert all(rel.split("/")[0] == rel.split("/")[0].lower() for rel in key["expected"])


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
    assert len(names) == {"easy": 3, "medium": 3, "hard": 4}[difficulty]  # snapshots; size drives difficulty
    chronological = sorted(names, key=_snapshot_key)
    assert names[0] != chronological[0] and names[-1] != chronological[0]
    # the tie rule decides inside the oldest snapshot on hard only
    tie = any(len([r for r in p if r.split("/")[0] == chronological[0]]) > 1 for p in dups.values())
    assert tie == (difficulty == "hard")


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


def _groups(names: list[str]) -> dict[str, list[str]]:
    out = defaultdict(list)
    for n in sorted(names, key=str.encode):
        if not n.startswith("."):
            out[_std(n)].append(n)
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
    # no level has a collision group holding a file already at the standard name, so renaming
    # a group's files one by one never overwrites a file that is still to be renamed
    assert not standard_not_first
    existing = any(len(olds) > 1 and new in olds for names in folders.values() for new, olds in _groups(names).items())
    assert not existing
    # extensions need lowercasing on hard only
    upper_ext = any(n.rpartition(".")[2] != n.rpartition(".")[2].lower() for names in folders.values() for n in names if not n.startswith("."))
    assert upper_ext == (difficulty == "hard")
    if difficulty == "hard":
        assert any(n.rpartition(".")[2].isupper() for names in folders.values() for n in names if not n.startswith("."))
    assert any(len({_std(n) for n in names if not n.startswith(".")}) < len([n for n in names if not n.startswith(".")]) for names in folders.values())
    if difficulty != "easy":
        sub = [f for f in folders if f]
        assert sub and all(any(_std(n) != n for n in folders[f] if not n.startswith(".")) for f in sub)  # top level only
        assert any(_std(part) != part for f in sub for part in f.split("/"))  # renaming folders
    else:
        assert list(folders) == [""]  # subfolders only on harder levels


@pytest.mark.parametrize("seed", SEEDS)
def test_bulk_rename_suffix_if_taken_is_declared_with_its_trap(seed):
    """The `suffix-if-taken` loop is correct when no file already has its standard name, so it
    is declared exactly when `existing` plants that collision: on no level, and on a level that
    turns `existing` on, where generation checks that the loop fails."""
    fam = FAMILIES["bulk_rename"]
    for difficulty in fam.difficulties:
        assert "suffix-if-taken" not in generate_spec(fam, difficulty, seed).spec.shortcuts
    knob = dataclasses.replace(fam, difficulties={"existing": {**fam.difficulties["hard"], "existing": True}})
    gen = generate_spec(knob, "existing", seed)
    assert "suffix-if-taken" in gen.spec.shortcuts and gen.shortcut_rewards["suffix-if-taken"] < 1
    names = [rel.rpartition("/")[2] for rel in gen.spec.files if rel.count("/") == 1]
    assert any(len(olds) > 1 and new in olds[1:] for new, olds in _groups(names).items())


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

    def stated(pattern: str, path: str, whole: bool = True) -> bool:  # `*` is the only wildcard
        regex = ".*".join(re.escape(part) for part in pattern.split("*"))
        return bool((re.fullmatch if whole else re.match)(regex, path, re.S))

    hits = {path: [m for p, m in rules if stated(p, path)] for path in initial}
    target = {path: h[-1] if h else initial[path] for path, h in hits.items()}
    # a glob matcher (`fnmatch`, `find -path`) reads `[...]` as a character class: on medium/hard
    # it gives some file another mode; on easy no pattern has a glob character but `*`
    globbed = {path: ([m for p, m in rules if fnmatch.fnmatchcase(path, p)] or [initial[path]])[-1] for path in initial}
    assert any(c in p for p, _ in rules for c in "?[") == (difficulty != "easy")
    assert (globbed != target) == (difficulty != "easy")
    # every level: a file no rule matches as a whole, but one does from the start, with another mode
    assert any(not hits[p] and any(stated(pat, p, whole=False) and m != initial[p] for pat, m in rules) for p in initial)
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
