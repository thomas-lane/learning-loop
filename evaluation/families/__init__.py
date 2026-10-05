"""Task families, keyed by name. Each module defines `FAMILY` (see learning_loop.tasks.spec).

    from evaluation.families import FAMILIES
    from learning_loop.tasks.render import render
    render(FAMILIES["log-triage"], "medium", 3, Path("..."))
"""

from __future__ import annotations

from learning_loop.tasks.spec import Family

from . import (
    broken_symlinks,
    bulk_rename,
    cli_flags,
    config_drift,
    count_errors,
    cron_translate,
    csv_revenue,
    dedupe_contacts,
    dedupe_files,
    env_reconcile,
    error_burst,
    extract_subset,
    fix_dates,
    fix_json_config,
    fix_pagination,
    fix_parser,
    fix_shell_script,
    fix_stats,
    format_converter,
    implement_function,
    import_cycle,
    join_orders,
    json_aggregate,
    latency_p95,
    log_triage,
    organize_files,
    permissions_fix,
    refactor_preserve,
    resolve_conflicts,
    session_count,
    slow_query,
    sqlite_query,
    toml_migrate,
    traceback_locate,
    write_validator,
)

_MODULES = (
    broken_symlinks, bulk_rename, cli_flags, config_drift, count_errors, cron_translate, csv_revenue,
    dedupe_contacts, dedupe_files, env_reconcile, error_burst, extract_subset, fix_dates, fix_json_config,
    fix_pagination, fix_parser, fix_shell_script, fix_stats, format_converter, implement_function, import_cycle,
    join_orders, json_aggregate, latency_p95, log_triage, organize_files, permissions_fix, refactor_preserve,
    resolve_conflicts, session_count, slow_query, sqlite_query, toml_migrate, traceback_locate, write_validator,
)
FAMILIES: dict[str, Family] = {m.FAMILY.name: m.FAMILY for m in _MODULES}
assert len(FAMILIES) == len(_MODULES), "two family modules declare the same name"
