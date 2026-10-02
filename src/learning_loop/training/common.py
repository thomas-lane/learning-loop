"""Shared trainer plumbing: dataset loading, checkpoint identity, atomic publication, devices.

Publication contract (both trainers):
- checkpoint_id = "c<cycle:03d>-<sha12>" over (run, cycle, dataset sha256, incoming
  checkpoint id, training config, seed, trainer). Same inputs -> same id.
- Files are written to `output_root/.tmp-<id>-<pid>-<rand>/`, verified (including the
  post-save reload check), made read-only, then renamed to `output_root/<id>/` in one `os.rename`.
- An existing `output_root/<id>/` is never overwritten: if its checkpoint.json records
  the same id and dataset it is returned as-is (idempotent retry); otherwise it is an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import shutil
from pathlib import Path
from typing import Any

from ..config import ModelProfile, TrainingConfig, load_model_profile
from ..records import CheckpointRecord, PreferenceExample
from ..storage import atomic_write_json, read_json, read_jsonl, sha256_file, sha256_json

CHECKPOINT_JSON = "checkpoint.json"


class NoTrainableExamples(RuntimeError):
    """Empty dataset, or every example dropped during rendering. The caller records a no-update cycle."""

    def __init__(self, message: str, render: dict[str, Any] | None = None):
        super().__init__(message)
        self.render = render  # RenderReport.summary() when examples were rendered


class TrainingRequestError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def load_preference_dataset(dataset_dir: str | Path) -> tuple[list[PreferenceExample], str, dict[str, Any]]:
    """(examples, sha256 of preferences.jsonl, info). Provenance is read only for counts."""
    d = Path(dataset_dir)
    prefs = d / "preferences.jsonl"
    if not prefs.exists():
        raise TrainingRequestError(f"{d}: missing preferences.jsonl")
    examples = [PreferenceExample.model_validate(r) for r in read_jsonl(prefs)]
    ids = [e.pair_id for e in examples]
    if len(ids) != len(set(ids)):
        raise TrainingRequestError(f"{d}: duplicate pair_id in preferences.jsonl")
    info: dict[str, Any] = {"n_examples": len(examples)}
    prov = d / "provenance.jsonl"
    if prov.exists():
        kinds: dict[str, int] = {}
        modes: dict[str, int] = {}
        for r in read_jsonl(prov):
            kinds[r.get("kind", "verified")] = kinds.get(r.get("kind", "verified"), 0) + 1
            modes[r.get("verification_mode")] = modes.get(r.get("verification_mode"), 0) + 1
        info["provenance_kinds"] = kinds
        info["verification_modes"] = modes
    if (d / "manifest.json").exists():
        info["manifest_sha256"] = sha256_file(d / "manifest.json")
    return examples, sha256_file(prefs), info


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def derive_checkpoint_id(
    run_id: str, cycle: int, dataset_sha256: str, incoming_id: str, training_config: dict[str, Any], seed: int, trainer: str
) -> str:
    h = sha256_json(
        {
            "run_id": run_id,
            "cycle": cycle,
            "dataset_sha256": dataset_sha256,
            "incoming": incoming_id,
            "training_config": training_config,
            "seed": seed,
            "trainer": trainer,
        }
    )
    return f"c{cycle:03d}-{h[:12]}"


def request_sha256(request_dump: dict[str, Any]) -> str:
    return sha256_json(request_dump)


def adapter_sha256(adapter_dir: Path, files: tuple[str, ...]) -> str:
    """Hash over the named adapter files (name + content), in the given order."""
    h = hashlib.sha256()
    for name in files:
        p = Path(adapter_dir) / name
        h.update(f"{name}\n{sha256_file(p)}\n".encode())
    return h.hexdigest()


def resolve_training_config(raw: dict[str, Any], expected_trainer: str) -> TrainingConfig:
    cfg = TrainingConfig.model_validate(raw)
    if cfg.trainer != expected_trainer:
        raise TrainingRequestError(f"training_config.trainer={cfg.trainer!r} but this trainer is {expected_trainer!r}")
    return cfg


def resolve_profile(request_profile: str, incoming_profile: str, base_model: str, base_revision: str) -> ModelProfile:
    if request_profile != incoming_profile:
        raise TrainingRequestError(f"request profile {request_profile!r} != incoming checkpoint profile {incoming_profile!r}")
    profile = load_model_profile(request_profile)
    if (profile.base_model, profile.base_revision) != (base_model, base_revision):
        raise TrainingRequestError(
            f"incoming checkpoint base {base_model}@{base_revision} != profile {profile.base_model}@{profile.base_revision}"
        )
    return profile


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #


def existing_publication(output_root: Path, checkpoint_id: str, dataset_sha256: str) -> CheckpointRecord | None:
    final = Path(output_root) / checkpoint_id
    if not final.exists():
        return None
    cj = final / CHECKPOINT_JSON
    if not cj.exists():
        raise FileExistsError(f"{final} exists without {CHECKPOINT_JSON}; refusing to overwrite")
    rec = CheckpointRecord.model_validate(read_json(cj))
    if rec.checkpoint.checkpoint_id != checkpoint_id or rec.dataset_sha256 != dataset_sha256:
        raise FileExistsError(f"{final} holds a different checkpoint; refusing to overwrite")
    return rec


def make_staging_dir(output_root: Path, checkpoint_id: str) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp = output_root / f".tmp-{checkpoint_id}-{os.getpid()}-{secrets.token_hex(4)}"
    tmp.mkdir()
    return tmp


def publish(staging: Path, output_root: Path, record: CheckpointRecord) -> Path:
    """Write checkpoint.json into staging, then atomically rename staging -> output_root/<id>."""
    cid = record.checkpoint.checkpoint_id
    final = Path(output_root) / cid
    atomic_write_json(staging / CHECKPOINT_JSON, record)
    for f in staging.rglob("*"):
        if f.is_file():
            f.chmod(0o444)  # published files are read-only; the directory is never rewritten
    if final.exists():
        raise FileExistsError(f"{final} appeared during publication; refusing to overwrite")
    os.rename(staging, final)  # same filesystem: atomic; fails if final exists and is non-empty
    return final / CHECKPOINT_JSON


def discard_staging(staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #


def resolve_device(
    requested: str, allow_cpu_fallback: bool, supported: list[str] | tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Pick the training/serving device. CPU is used only when requested or explicitly allowed.

    `supported` (a model profile's `supported_train_devices`; empty/None = not declared)
    restricts the choice: "auto" only considers supported accelerators, the CPU fallback
    only applies when CPU is supported, and a resolved device outside the list raises
    TrainingRequestError. The list and the outcome are recorded in the returned info."""
    import torch

    sup = list(supported or [])
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    info: dict[str, Any] = {"requested": requested, "cuda_available": cuda, "mps_available": mps, "cpu_fallback": False,
                            "supported_train_devices": sup or None}

    def ok(d: str) -> bool:
        return not sup or d in sup

    if requested == "auto":
        if cuda and ok("cuda"):
            dev = "cuda"
        elif mps and ok("mps"):
            dev = "mps"
        else:
            dev = None
    elif requested == "cuda":
        dev = "cuda" if cuda else None
    elif requested == "mps":
        dev = "mps" if mps else None
    elif requested == "cpu":
        dev = "cpu"
    else:
        raise TrainingRequestError(f"unknown device {requested!r}")
    if dev is None:
        if not allow_cpu_fallback:
            raise TrainingRequestError(
                f"device {requested!r} unavailable (cuda={cuda}, mps={mps}, supported={sup or 'any'}) "
                "and allow_cpu_fallback is false"
            )
        dev = "cpu"
        info["cpu_fallback"] = True
    if not ok(dev):
        raise TrainingRequestError(
            f"device {dev!r} (requested {requested!r}{', CPU fallback' if info['cpu_fallback'] else ''}) "
            f"is not in the model profile's supported_train_devices {sup}"
        )
    info["supported_device_check"] = "ok" if sup else "not_declared"
    info["device"] = dev
    info["platform"] = f"{platform.system()} {platform.machine()}"
    return info


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
