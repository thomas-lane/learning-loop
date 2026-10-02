"""Deterministic identities and seed streams.

Seeds and IDs are derived with SHA-256 over stable, explicit components, never
Python's process-salted `hash()` or asynchronous completion order. Each purpose
uses its own named stream so that, e.g., adding an audit never shifts the
learner's attempt seeds.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

Stream = Literal[
    "task_generation",
    "learner_attempt",  # eval + collect attempts (paired across checkpoints)
    "editor_proposal",
    "branch_continuation",  # shared by original/edited branches (matched)
    "training",
    "data_selection",
    "audit",
    "confirmation",
]

_SEED_BITS = 31  # fits every backend's `seed` field (llama.cpp, vLLM, torch)


def _canonical(parts: tuple[Any, ...]) -> bytes:
    return json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str).encode()


def derive_seed(root_seed: int, stream: Stream, *parts: Any) -> int:
    """seed = H(root, stream, parts...) mod 2^31. Stable across processes/hosts."""
    digest = hashlib.sha256(_canonical((root_seed, stream, *parts))).digest()
    return int.from_bytes(digest[:8], "big") % (1 << _SEED_BITS)


def stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    """Readable, content-derived identifier: `<prefix>-<hex>`."""
    digest = hashlib.sha256(_canonical((prefix, *parts))).hexdigest()
    return f"{prefix}-{digest[:length]}"


def request_seed(episode_seed: int | None, request_index: int) -> int | None:
    """Per-request seed within an episode (request 0, 1, ...)."""
    if episode_seed is None:
        return None
    digest = hashlib.sha256(_canonical(("request", episode_seed, request_index))).digest()
    return int.from_bytes(digest[:8], "big") % (1 << _SEED_BITS)


def attempt_seed(root_seed: int, instance_id: str, attempt_index: int) -> int:
    """Learner attempt seed. Deliberately independent of the checkpoint so paired
    checkpoint evaluations reuse the same declared instance/attempt schedule."""
    return derive_seed(root_seed, "learner_attempt", instance_id, attempt_index)


def continuation_seed(root_seed: int, proposal_id: str, repetition: int, purpose: str = "acceptance") -> int:
    """Matched across original/edited branches: the branch label is NOT an input."""
    stream: Stream = {
        "acceptance": "branch_continuation",
        "audit": "audit",
        "confirmation": "confirmation",
    }[purpose]  # type: ignore[assignment]
    return derive_seed(root_seed, stream, proposal_id, repetition)
