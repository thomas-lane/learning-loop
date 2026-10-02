"""Deterministic, versioned task generators, keyed by family name.

    from evaluation.generators import GENERATORS, generate
    params = generate("log-triage", "medium", seed=3, out_dir=Path("..."))

A generator writes a complete Harbor task directory. Changing a generator's
output for an existing (difficulty, seed) requires bumping its VERSION.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from . import count_errors, csv_revenue, fix_stats, log_triage

GENERATORS: dict[str, ModuleType] = {m.FAMILY: m for m in (log_triage, fix_stats, csv_revenue, count_errors)}


def generator_id(family: str) -> str:
    m = GENERATORS[family]
    return f"{m.FAMILY}@v{m.VERSION}"


def generate(family: str, difficulty: str, seed: int, out_dir: Path) -> dict[str, Any]:
    if family not in GENERATORS:
        raise KeyError(f"no generator for family {family!r} (have {sorted(GENERATORS)})")
    m = GENERATORS[family]
    if difficulty not in m.DIFFICULTIES:
        raise KeyError(f"{family}: unknown difficulty {difficulty!r} (have {sorted(m.DIFFICULTIES)})")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to generate into non-empty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return m.generate(out_dir, difficulty, seed)
