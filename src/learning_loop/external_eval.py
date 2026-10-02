"""Version-pinned external Harbor evaluations (e.g. Terminal-Bench) of saved checkpoints.

This only produces a Harbor job config plus a protocol note; it never feeds
results into training or checkpoint selection (the coordinator never reads
`runs/external/`). The agent is this repository's ToolAgent (bash/read_file/
write_file tools over an OpenAI-compatible endpoint), i.e. a custom-agent
protocol, not the benchmark's reference harness — report it that way.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .config import REPO_ROOT, load_machine
from .storage import atomic_write_json, atomic_write_text, now_iso, read_json

_UNPINNED = {"", "latest", "head", "main", "master"}


def parse_dataset(spec: str) -> tuple[str, str]:
    if "@" not in spec:
        raise ValueError("external datasets must be version-pinned: use name@version")
    name, version = spec.rsplit("@", 1)
    if version.lower() in _UNPINNED or not re.fullmatch(r"[A-Za-z0-9._+-]+", version):
        raise ValueError(f"version {version!r} is not an exact pin")
    return name, version


def checkpoint_identity(checkpoint: str, model_profile: str | None) -> str:
    if checkpoint == "base":
        if not model_profile:
            raise ValueError("--checkpoint base needs --model-profile to resolve the exact base identity")
        from .config import load_model_profile
        from .coordinator import base_checkpoint

        return base_checkpoint(load_model_profile(model_profile)).checkpoint_id
    return read_json(Path(checkpoint) / "checkpoint.json")["checkpoint"]["checkpoint_id"]


def build_external_job(dataset: str, checkpoint: str, machines: str, out: str, n_tasks: int | None = None, task_names: list[str] | None = None, model_profile: str | None = None) -> tuple[Path, list[str]]:
    from harbor.models.job.config import JobConfig

    name, version = parse_dataset(dataset)
    machine = load_machine(machines)
    inf = machine.inference
    if inf.mode == "scripted":
        raise ValueError("external evaluation needs a real model endpoint")
    api_base = inf.api_base or f"http://127.0.0.1:{inf.port}/v1"
    ckpt_id = checkpoint_identity(checkpoint, model_profile)
    if inf.mode == "external" and inf.served_checkpoint_id != ckpt_id:
        raise ValueError(f"external endpoint declares {inf.served_checkpoint_id!r}, not {ckpt_id!r}")
    served = (inf.served_model_name or ckpt_id) if inf.mode == "external" else ckpt_id
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{name}-{version}-{ckpt_id}")
    out_dir = (REPO_ROOT / out / safe) if not Path(out).is_absolute() else Path(out) / safe
    job = {
        "job_name": safe,
        "jobs_dir": str(out_dir / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "agents": [
            {
                "import_path": "evaluation.agents.tool_agent:ToolAgent",
                "model_name": served,
                "kwargs": {"api_base": api_base, **({"api_key_env": inf.api_key_env} if inf.api_key_env else {})},
            }
        ],
        "datasets": [{"name": name, "version": version, **({"n_tasks": n_tasks} if n_tasks else {}), **({"task_names": task_names} if task_names else {})}],
    }
    JobConfig.model_validate(job)  # fail early on schema drift in the pinned Harbor
    cfg_path = out_dir / "job.yaml"
    atomic_write_text(cfg_path, yaml.safe_dump(job, sort_keys=False))
    atomic_write_json(
        out_dir / "protocol.json",
        {
            "dataset": {"name": name, "version": version, "subset": {"n_tasks": n_tasks, "task_names": task_names}},
            "checkpoint": checkpoint,
            "checkpoint_id": ckpt_id,
            "served_model_name": served,
            "serving_backend": inf.backend,
            "serving_mode": inf.mode,
            "agent_protocol": "custom: repository ToolAgent (bash/read_file/write_file), not the benchmark reference agent",
            "training_isolation": "results are never read by the learning loop or used for checkpoint selection",
            "created_at": now_iso(),
        },
    )
    argv = ["uv", "run", "harbor", "run", "-c", str(cfg_path), "-y"]  # project venv: ToolAgent imports learning_loop
    return cfg_path, argv
