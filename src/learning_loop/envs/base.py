"""Shared tool-execution logic for environment sessions.

Both the Harbor (Docker) session and the local fixture session run the same
tool handlers (`evaluation/agents/tools.py`) against an exec target, so tool
schemas, output formats and truncation are identical across backends.

`execute()` returns command-level failures inside the ToolExecution and lets
`EnvInfraError` (the environment transport failed) propagate, so the episode
loop stops as INFRA instead of showing the learner a transport error.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from evaluation.agents.tools import EnvInfraError, ToolConfig, run_tool, truncate

from ..interfaces import EnvCapabilities, StateSpec
from ..records import ToolExecution


__all__ = ["EnvInfraError", "FingerprintError", "ToolSession", "build_inputs_identity"]


class FingerprintError(RuntimeError):
    pass


def build_inputs_identity(kind: str, context_dir: Path, *, prebuilt_image: str | None = None, image_id: str | None = None) -> dict[str, Any]:
    """Identity of an environment image from its build inputs.

    `identity` = sha256 over the build context tree (Dockerfile + files) and a
    prebuilt image reference; `base_images` are the Dockerfile's FROM references
    and `base_pinned` says whether all of them are pinned by digest. `image_id`
    (the locally built image id, when reachable) is informational only: rebuilds
    of identical inputs get new ids (layer timestamps), so replay compares
    `identity`.
    """
    from ..storage import sha256_tree

    context_dir = Path(context_dir)
    tree = sha256_tree(context_dir) if context_dir.exists() else None
    dockerfile = context_dir / "Dockerfile"
    bases: list[str] = []
    if dockerfile.is_file():
        for line in dockerfile.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].upper() == "FROM":
                ref = next((p for p in parts[1:] if not p.startswith("--")), None)
                if ref:
                    bases.append(ref)
    refs = [prebuilt_image] if prebuilt_image else bases
    h = hashlib.sha256(json.dumps({"kind": kind, "tree": tree, "prebuilt_image": prebuilt_image}, sort_keys=True).encode()).hexdigest()
    return {
        "kind": kind,
        "identity": f"sha256:{h}",
        "context_sha256": tree,
        "prebuilt_image": prebuilt_image,
        "base_images": bases,
        "base_pinned": bool(refs) and all("@sha256:" in r for r in refs),
        "image_id": image_id,
    }


class ToolSession:
    """Base class: subclasses provide `target` (exec/upload_file) and fingerprints."""

    capabilities: EnvCapabilities
    target: Any
    tool_cfg: ToolConfig

    async def _cpu_usage_sec(self) -> float | None:
        """Cumulative CPU seconds of the environment's scope, or None."""
        return None

    async def image_identity(self) -> dict[str, Any] | None:
        """Identity of the environment image (see `build_inputs_identity`), or None."""
        return None

    async def execute(self, call_id: str, name: str, arguments: dict[str, Any], raw_arguments: str | None) -> ToolExecution:
        cpu0 = await self._cpu_usage_sec() if self.capabilities.measures_cpu else None
        t0 = time.monotonic()
        out = await run_tool(self.target, name, arguments, self.tool_cfg)
        duration = time.monotonic() - t0
        cpu1 = await self._cpu_usage_sec() if cpu0 is not None else None
        observation = truncate(out.raw_output, self.tool_cfg.max_output_chars)
        return ToolExecution(
            call_id=call_id,
            name=name,
            requested_arguments_raw=raw_arguments,
            executed_arguments=arguments if out.executed else None,
            executed=out.executed,
            exit_code=out.exit_code,
            stdout=out.stdout,
            stderr=out.stderr,
            raw_output=out.raw_output,
            observation=observation,
            truncated=observation != out.raw_output,
            duration_sec=round(duration, 6),
            cpu_time_sec=round(cpu1 - cpu0, 6) if cpu0 is not None and cpu1 is not None else None,
            peak_memory_bytes=None,
            error=out.error,
            timeout_sec=out.timeout_sec,
            timed_out=out.timed_out,
        )

    async def fingerprint_with_detail(self, spec: StateSpec) -> tuple[str, list[str]]:
        raise NotImplementedError

    async def fingerprint(self, spec: StateSpec) -> str:
        digest, _ = await self.fingerprint_with_detail(spec)
        return digest
