"""Task instances, explicit split/panel files, materialization and leakage checks.

Split files (`evaluation/splits/*.yaml`) are authored by hand:

    schema_version: 1
    name: pilot
    held_out_families: [csv-revenue]         # never allowed in a train panel
    families:
      log-triage: {skills: [shell, logs]}    # optional default skills per family
    instances:
      - {id: log-triage/easy/s1, family: log-triage, difficulty: easy, seed: 1, split: train}
      - {id: log-triage/static, family: log-triage, static: evaluation/tasks/log-triage, split: dev}
    panels:
      train: {split: train, instances: [log-triage/easy/s1]}

Every instance is declared once, with exactly one split. A panel is a named
list of instances of one split (an instance may appear in several panels of
its split). Generated instances come from `evaluation/generators` (family,
difficulty, seed); static instances are copied from a task directory.

`materialize()` writes instance task dirs and returns content-hashed
`TaskInstance` records. `validate_splits()` rejects: an instance id declared
twice or with two splits, a panel mixing splits, held-out families in any
train panel/split, the same generator coordinates under two ids, and identical
learner-visible content (`public_content_hash`) in different splits. Disjoint
seeds alone are not treated as proof of disjoint content.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import REPO_ROOT, repo_path
from .interfaces import StateSpec
from .records import RestoreCapability, Split, TaskInstance
from .storage import atomic_write_json, sha256_tree


class SplitValidationError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FamilyDef(_Strict):
    skills: list[str] = Field(default_factory=list)
    description: str = ""


class InstanceDef(_Strict):
    instance_id: str = Field(alias="id")
    family: str
    split: Split
    difficulty: str | None = None
    seed: int | None = None  # generator seed (generated instances)
    static: str | None = None  # task dir (static instances), relative to the repo root
    skills: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "InstanceDef":
        if (self.static is None) == (self.seed is None):
            raise ValueError(f"{self.instance_id}: give exactly one of `seed` (generated) or `static` (task dir)")
        if self.static is None and self.difficulty is None:
            raise ValueError(f"{self.instance_id}: generated instances need a difficulty")
        return self

    @property
    def generator_key(self) -> tuple[str, str | None, int | None, str | None]:
        return (self.family, self.difficulty, self.seed, self.static)


class Panel(_Strict):
    name: str
    split: Split
    description: str = ""
    instances: list[InstanceDef]


class _PanelRaw(_Strict):
    split: Split
    description: str = ""
    instances: list[str]


class _SplitsRaw(_Strict):
    schema_version: Literal[1] = 1
    name: str
    description: str = ""
    held_out_families: list[str] = Field(default_factory=list)
    families: dict[str, FamilyDef] = Field(default_factory=dict)
    instances: list[InstanceDef]
    panels: dict[str, _PanelRaw]


class Splits(_Strict):
    name: str
    path: str
    description: str = ""
    held_out_families: list[str]
    families: dict[str, FamilyDef]
    instances: dict[str, InstanceDef]
    panels: dict[str, Panel]

    def skills_for(self, inst: InstanceDef) -> list[str]:
        fam = self.families.get(inst.family)
        return sorted(set(inst.skills) | set(fam.skills if fam else []))


def load_splits(path: str | Path) -> Splits:
    """Parse + structurally validate a split file (content checks need materialize())."""
    p = repo_path(path)
    raw = _SplitsRaw.model_validate(yaml.safe_load(p.read_text()))
    errors: list[str] = []
    instances: dict[str, InstanceDef] = {}
    for inst in raw.instances:
        if inst.instance_id in instances:
            errors.append(f"instance {inst.instance_id!r} declared more than once")
            continue
        instances[inst.instance_id] = inst
    panels: dict[str, Panel] = {}
    for name, pr in raw.panels.items():
        defs = []
        for iid in pr.instances:
            if iid not in instances:
                errors.append(f"panel {name!r}: unknown instance {iid!r}")
                continue
            defs.append(instances[iid])
        if len(set(pr.instances)) != len(pr.instances):
            errors.append(f"panel {name!r}: duplicate instance ids")
        panels[name] = Panel(name=name, split=pr.split, description=pr.description, instances=defs)
    splits = Splits(
        name=raw.name,
        path=str(p),
        description=raw.description,
        held_out_families=raw.held_out_families,
        families=raw.families,
        instances=instances,
        panels=panels,
    )
    errors += structural_errors(splits)
    if errors:
        raise SplitValidationError(f"{p}: " + "; ".join(errors))
    return splits


def structural_errors(splits: Splits) -> list[str]:
    from evaluation.generators import GENERATORS

    errs: list[str] = []
    for name, panel in splits.panels.items():
        for inst in panel.instances:
            if inst.split != panel.split:
                errs.append(f"panel {name!r} ({panel.split.value}) contains {inst.instance_id!r} of split {inst.split.value}")
            if panel.split == Split.TRAIN and inst.family in splits.held_out_families:
                errs.append(f"held-out family {inst.family!r} appears in train panel {name!r} ({inst.instance_id})")
    for inst in splits.instances.values():
        if inst.split == Split.TRAIN and inst.family in splits.held_out_families:
            errs.append(f"held-out family {inst.family!r} has a train-split instance {inst.instance_id!r}")
        if inst.static is None:
            gen = GENERATORS.get(inst.family)
            if gen is None:
                errs.append(f"{inst.instance_id}: no generator for family {inst.family!r}")
            elif inst.difficulty not in gen.DIFFICULTIES:
                errs.append(f"{inst.instance_id}: unknown difficulty {inst.difficulty!r} for {inst.family}")
        elif not repo_path(inst.static).joinpath("task.toml").exists():
            errs.append(f"{inst.instance_id}: static task dir {inst.static!r} has no task.toml")
    seen: dict[tuple[Any, ...], str] = {}
    for inst in splits.instances.values():
        key = inst.generator_key
        if key in seen:
            errs.append(f"{inst.instance_id!r} and {seen[key]!r} are the same task ({key})")
        else:
            seen[key] = inst.instance_id
    return errs


# --------------------------------------------------------------------------- #
# Task-dir helpers
# --------------------------------------------------------------------------- #


def read_task_toml(task_dir: Path) -> dict[str, Any]:
    return tomllib.loads((Path(task_dir) / "task.toml").read_text())


NETWORK_CAVEAT = "network: public (unmodeled)"


def load_state_spec(task_dir: Path) -> StateSpec:
    """The task's `[metadata.learning_loop]` replay contract (NONE when absent).

    Deterministic replay should run without network access, but Harbor 0.23.0's
    Docker backend rejects `network_mode = "no-network"` on hosts whose kernel
    lacks its egress-control support (Docker Desktop on this Mac; see
    evaluation/README.md). Such tasks run with the public network, and the spec
    records that as the caveat `NETWORK_CAVEAT` (it travels with every plan into
    `episode_start`) instead of claiming isolation."""
    toml = read_task_toml(task_dir)
    meta = toml.get("metadata", {}).get("learning_loop")
    if meta is None:
        return StateSpec()
    spec = StateSpec.model_validate(meta)
    if spec.restore == RestoreCapability.DETERMINISTIC_REPLAY and not spec.fingerprint_paths:
        raise ValueError(f"{task_dir}: deterministic replay requires fingerprint_paths")
    network = (toml.get("environment") or {}).get("network_mode", "public")
    if spec.restore == RestoreCapability.DETERMINISTIC_REPLAY and network != "no-network":
        caveat = NETWORK_CAVEAT if network == "public" else f"network: {network} (unmodeled)"
        if caveat not in spec.caveats:
            spec.caveats.append(caveat)
    for n in spec.observation_normalizers:
        if set(n) != {"pattern", "replacement", "reason"}:
            raise ValueError(f"{task_dir}: observation normalizers need exactly pattern/replacement/reason")
    return spec


def read_instruction(task_dir: Path) -> str:
    return (Path(task_dir) / "instruction.md").read_text()


def public_content_hash(task_dir: Path) -> str:
    """Hash of learner-visible inputs: instruction + environment/ (not tests/solution)."""
    d = Path(task_dir)
    h = hashlib.sha256()
    h.update(hashlib.sha256((d / "instruction.md").read_bytes()).hexdigest().encode())
    env = d / "environment"
    h.update((sha256_tree(env) if env.exists() else "-").encode())
    return h.hexdigest()


def _instance_dir_name(inst: InstanceDef) -> str:
    return inst.instance_id.replace("/", "__")


def materialize_instance(splits: Splits, inst: InstanceDef, dest_root: Path) -> TaskInstance:
    from evaluation.generators import generate, generator_id

    dest = Path(dest_root) / _instance_dir_name(inst)
    tmp = dest.with_name(dest.name + ".tmp")
    params: dict[str, Any] = {}
    if not dest.exists():
        if tmp.exists():
            shutil.rmtree(tmp)
        if inst.static is not None:
            shutil.copytree(repo_path(inst.static), tmp, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        else:
            generate(inst.family, inst.difficulty or "", inst.seed or 0, tmp)
        tmp.rename(dest)
    toml = read_task_toml(dest)
    meta = toml.get("metadata", {})
    if "params_json" in meta:
        params = json.loads(meta["params_json"])
    spec = load_state_spec(dest)
    gen = None
    if inst.static is None:
        gen = generator_id(inst.family)
        if meta.get("generator") != gen:
            raise SplitValidationError(f"{dest}: generated by {meta.get('generator')!r}, current generator is {gen!r}; use a fresh directory")
    difficulty = inst.difficulty or meta.get("difficulty")
    return TaskInstance(
        instance_id=inst.instance_id,
        family=inst.family,
        difficulty=difficulty,
        skills=splits.skills_for(inst) or list(meta.get("skills", [])),
        generator=gen,
        generator_seed=inst.seed,
        params=params,
        task_dir=str(dest),
        content_hash=sha256_tree(dest),
        public_content_hash=public_content_hash(dest),
        restore=spec.restore,
    )


def materialize(splits: Splits, dest_root: str | Path, panels: list[str] | None = None, ids: list[str] | None = None) -> dict[str, TaskInstance]:
    """Write task dirs for the selected instances (default: all) under dest_root.

    Existing directories are reused (content is re-hashed, so edits are visible
    in `content_hash`). Writes `dest_root/instances.json` (merged index).
    """
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    wanted: list[str] = []
    if panels is None and ids is None:
        wanted = list(splits.instances)
    for p in panels or []:
        if p not in splits.panels:
            raise SplitValidationError(f"unknown panel {p!r}")
        wanted += [i.instance_id for i in splits.panels[p].instances]
    wanted += list(ids or [])
    out: dict[str, TaskInstance] = {}
    for iid in dict.fromkeys(wanted):
        out[iid] = materialize_instance(splits, splits.instances[iid], dest_root)
    index_path = dest_root / "instances.json"
    index: dict[str, Any] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    index.update({k: v.model_dump(mode="json") for k, v in out.items()})
    atomic_write_json(index_path, dict(sorted(index.items())))
    return out


def validate_splits(splits: Splits, instances: dict[str, TaskInstance] | None = None) -> list[str]:
    """Raise SplitValidationError on any leakage problem; return informational notes."""
    errs = structural_errors(splits)
    notes: list[str] = []
    if instances:
        by_hash: dict[str, list[str]] = {}
        for iid, ti in instances.items():
            if ti.public_content_hash:
                by_hash.setdefault(ti.public_content_hash, []).append(iid)
        for h, ids in by_hash.items():
            split_set = {splits.instances[i].split for i in ids if i in splits.instances}
            if len(split_set) > 1:
                errs.append(f"identical learner-visible content in different splits: {sorted(ids)}")
            elif len(ids) > 1:
                notes.append(f"identical learner-visible content within one split: {sorted(ids)}")
        for iid, ti in instances.items():
            inst = splits.instances.get(iid)
            if inst is not None and ti.family != inst.family:
                errs.append(f"{iid}: materialized family {ti.family!r} != declared {inst.family!r}")
    if errs:
        raise SplitValidationError("; ".join(errs))
    return notes


def assert_exportable(splits: Splits, instance_id: str, forbidden_families: list[str] | None = None) -> None:
    """Dataset-export boundary: only train-split instances of allowed families."""
    inst = splits.instances.get(instance_id)
    if inst is None:
        raise SplitValidationError(f"{instance_id}: not declared in {splits.name}")
    if inst.split != Split.TRAIN:
        raise SplitValidationError(f"{instance_id}: split {inst.split.value} may not be exported for training")
    if inst.family in set(splits.held_out_families) | set(forbidden_families or []):
        raise SplitValidationError(f"{instance_id}: family {inst.family!r} is held out")


def repo_relative(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
