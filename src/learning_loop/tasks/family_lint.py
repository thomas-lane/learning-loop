"""Static check that a family module draws randomness only from `GenContext.rng`.

A family's output must be a pure function of (family, rng_version, difficulty, seed). Any
other source of randomness or time would make the same seed render different bytes on
different hosts or days, so these are rejected (checked by tests/unit over every family):

    import random / secrets / uuid / time   (and `from ... import` of them)
    os.urandom(...), <anything>.now(), .today(), .utcnow()

`random.Random` as a type annotation is not needed: families receive `ctx.rng`.
"""

from __future__ import annotations

import ast

FORBIDDEN_MODULES = {"random", "secrets", "uuid", "time"}
FORBIDDEN_CALLS = {"urandom", "now", "today", "utcnow"}


def nondeterminism(source: str, filename: str = "<family>") -> list[str]:
    """Violations as 'line N: ...' strings (empty when the module is clean)."""
    out: list[str] = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            out += [f"line {node.lineno}: import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
            out.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
            out.append(f"line {node.lineno}: .{node.func.attr}()")
    return out
