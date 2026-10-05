"""Task families, keyed by name. Each module defines `FAMILY` (see learning_loop.tasks.spec).

    from evaluation.families import FAMILIES
    from learning_loop.tasks.render import render
    render(FAMILIES["log-triage"], "medium", 3, Path("..."))
"""

from __future__ import annotations

from learning_loop.tasks.spec import Family

from . import count_errors, csv_revenue, fix_stats, log_triage

FAMILIES: dict[str, Family] = {m.FAMILY.name: m.FAMILY for m in (log_triage, fix_stats, csv_revenue, count_errors)}
