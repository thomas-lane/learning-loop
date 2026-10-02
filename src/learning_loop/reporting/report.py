"""CSV + one readable markdown report per run, and paired run comparisons.

`write_report()` takes plain record lists (see metrics.py) and writes:

    episodes.csv            one row per episode (all roles)
    summary_by_<keys>.csv   aggregated views (checkpoint/cycle/panel, family, difficulty, skill, role)
    stage_effort.csv        collection/editor/verification/training effort per cycle
    proposals.csv           every proposal with status and reasons
    verifications.csv       every verification with acceptance reasons
    branch_costs.csv        per-branch counterfactual cost parts
    report.md               readable summary of the above

`compare(run_a, run_b)` pairs unassisted evaluations of two runs (or two
checkpoints) on the same panel/instance/attempt seeds and writes
`comparison.md` + `paired_episodes.csv` when `out_dir` is given.

`write_run_report(run_dir)` / `compare_runs(a, b)` read the coordinator's run
layout directly (stage manifests -> item summary/proposal/verification files).

`compare_conditions(runs_a, runs_b)` compares two conditions over several runs
each, one run per independent learning-loop seed: a paired instance-level effect
per seed, then mean and spread across seeds (no interval below 3 seeds).
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.records import EditProposal, EpisodeSummary, TaskInstance, Usage, VerificationRecord
from ..core.storage import StageManifest, atomic_write_text, read_json
from ..editing.preferences import DATASET_FORMAT
from .metrics import (
    MIN_SEEDS_FOR_INTERVAL,
    SMALL_N_NOTE,
    CostRate,
    EpisodeRow,
    StageEffort,
    branch_cost_rows,
    branch_saving_summary,
    eval_rows,
    group_summaries,
    make_rows,
    model_identity_of,
    multi_seed_comparison,
    paired_comparison,
    proposal_stats,
    read_rows,
    stage_effort_table,
    usage_coverage,
    verification_effort_split,
)

EPISODE_INDEX = "episodes.jsonl"  # runs/<id>/episodes.jsonl, appended by the coordinator

DISCLAIMER = (
    "Uncertainty is computed over task instances (mean of per-instance means), never over correlated "
    f"attempts; intervals appear only with >= 5 instances, otherwise '{SMALL_N_NOTE}'. Fixture/smoke runs "
    "exercise the machinery and are not evidence of learning efficacy."
)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _cell(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=str)
    return "" if v is None else v


def to_csv(rows: list[dict[str, Any]]) -> str:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r.get(k)) for k in cols})
    return buf.getvalue()


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) or "-"
    return str(v).replace("|", "\\|")


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_none_\n"
    head = "| " + " | ".join(h for _, h in columns) + " |\n"
    sep = "|" + "|".join("---" for _ in columns) + "|\n"
    body = "".join("| " + " | ".join(_fmt(r.get(k)) for k, _ in columns) + " |\n" for r in rows)
    return head + sep + body


def _effort_view(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    for k in ("input", "output", "total"):
        out[f"{k}_tok_view"] = r.get(f"{k}_tokens")
    if r.get("n_retried_attempts"):
        out["retried_view"] = f"{r['n_retried_attempts']} ({_fmt(r.get('retried_total_tokens'))} tok, {r.get('retried_tokens_coverage') or 'n/a'} with usage)"
    else:
        out["retried_view"] = r.get("n_retried_attempts")
    return out


def _interval(il: dict[str, Any]) -> str:
    if il.get("mean") is None:
        return "n/a"
    if il.get("ci95_low") is None:
        return f"{_fmt(il['mean'])} ({il.get('interval')}, n={il['n_instances']})"
    return f"{_fmt(il['mean'])} [{_fmt(il['ci95_low'])}, {_fmt(il['ci95_high'])}] (n={il['n_instances']})"


def _with_views(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in summaries:
        out.append(
            {
                **s,
                "success": f"{s['n_success']}/{s['n_episodes']}",
                "instance_success_view": _interval(s["instance_success"]),
                "instance_tokens_view": _interval(s["instance_total_tokens"]),
            }
        )
    return out


_SUMMARY_COLS = [
    ("success", "success"),
    ("mean_partial_reward", "mean partial reward"),
    ("instance_success_view", "success (instance-level)"),
    ("total_tokens_mean", "mean tokens/episode"),
    ("total_tokens_mean_successful", "mean tokens (successful)"),
    ("instance_tokens_view", "tokens (instance-level)"),
    ("n_requests_sum", "requests"),
    ("n_tool_calls_sum", "tool calls"),
    ("n_infra_failures", "infra failures"),
]


def episode_csv_rows(rows: list[EpisodeRow]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        e = r.episode
        out.append(
            {
                "run_id": r.run_id,
                "cycle": r.cycle,
                "panel": r.panel,
                "role": e.role.value,
                "checkpoint_id": e.checkpoint_id,
                "episode_id": e.episode_id,
                "instance_id": e.instance_id,
                "family": r.family,
                "difficulty": r.difficulty,
                "skills": ";".join(r.skills),
                "attempt_index": e.attempt_index,
                "seed": e.seed,
                "success": e.success,
                "partial_reward": e.partial_reward,
                "stop_category": e.stop_category.value,
                "stop_reason": e.stop_reason,
                "input_tokens": e.usage.input_tokens,
                "output_tokens": e.usage.output_tokens,
                "total_tokens": e.usage.total,
                "cached_input_tokens_subset": e.usage.cached_input_tokens,
                "reasoning_tokens_subset": e.usage.reasoning_tokens,
                "usage_source": e.usage.source,
                "n_requests": e.n_requests,
                "n_tool_calls": e.n_tool_calls,
                "n_malformed_turns": e.n_malformed_turns,
                "tool_sec": e.timing.tool_sec,
                "endpoint_sec": e.timing.endpoint_sec,
                "total_sec": e.timing.total_sec,
                "queue_sec": e.timing.queue_sec,
                "tool_cpu_sec": e.tool_cpu_sec,
                "tool_peak_memory_bytes": e.tool_peak_memory_bytes,
                "infra_error": e.infra_error,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Run report
# --------------------------------------------------------------------------- #


def training_data_row(t: dict[str, Any]) -> dict[str, Any]:
    """One cycle's `training_data` (cycle.json) as a flat row; null when the trainer did not render
    (fixture trainer) or the cycle exported no pairs."""
    kt = t.get("kept_tokens") or {}
    return {
        "cycle": t.get("cycle"),
        "exported": t.get("n_exported"),
        "trained": t.get("n_trained"),
        "dropped": t.get("n_dropped"),
        "dropped_by_reason": t.get("dropped_by_reason") or None,
        "dropped_pairs": [f"{d.get('pair_id')}: {d.get('reason')}" for d in t.get("dropped") or []] or None,
        "max_length": t.get("max_length"),
        "tokens_min": kt.get("min"),
        "tokens_median": kt.get("median"),
        "tokens_max": kt.get("max"),
    }


def write_report(
    out_dir: str | Path,
    rows: list[EpisodeRow],
    *,
    proposals: Iterable[EditProposal] = (),
    verifications: Iterable[VerificationRecord] = (),
    stage_efforts: Iterable[StageEffort] = (),
    dataset_manifests: Iterable[dict[str, Any]] = (),
    rates: Mapping[str, CostRate] | None = None,
    comparisons: Iterable[dict[str, Any]] = (),
    title: str = "Run report",
    notes: Iterable[str] = (),
    training_data: Iterable[dict[str, Any]] = (),
) -> dict[str, Path]:
    out = Path(out_dir)
    proposals, verifications = list(proposals), list(verifications)
    efforts = stage_effort_table(list(stage_efforts), rates)
    manifests = list(dataset_manifests)
    paths: dict[str, Path] = {}

    def put(name: str, text: str) -> None:
        atomic_write_text(out / name, text)
        paths[name] = out / name

    put("episodes.csv", to_csv(episode_csv_rows(rows)))
    evals = eval_rows(rows)
    views: dict[str, list[dict[str, Any]]] = {}
    for keys in (("checkpoint", "cycle", "panel"), ("family",), ("difficulty",), ("skill",)):
        s = group_summaries(evals, keys)
        views["_".join(keys)] = s
        put(f"summary_by_{'_'.join(keys)}.csv", to_csv(s))
    by_role = group_summaries(rows, ("role", "cycle"))
    put("summary_by_role_cycle.csv", to_csv(by_role))
    put("stage_effort.csv", to_csv(efforts))
    put(
        "proposals.csv",
        to_csv(
            [
                {
                    "proposal_id": p.proposal_id,
                    "source_episode_id": p.source_episode_id,
                    "instance_id": p.instance_id,
                    "editor_id": p.editor_id,
                    "status": p.status,
                    "turn_index": p.turn_index,
                    "replacement": p.replacement.model_dump() if p.replacement else None,
                    "rejection_reasons": p.rejection_reasons,
                    "input_tokens": p.usage.input_tokens if p.usage else None,
                    "output_tokens": p.usage.output_tokens if p.usage else None,
                    "duration_sec": p.duration_sec,
                }
                for p in proposals
            ]
        ),
    )
    put(
        "verifications.csv",
        to_csv(
            [
                {
                    "verification_id": v.verification_id,
                    "proposal_id": v.proposal_id,
                    "purpose": v.purpose,
                    "mode": v.mode,
                    "acceptance_rule": v.acceptance_rule,
                    "accepted": v.accepted,
                    "evidence_label": v.evidence_label,
                    "reasons": v.reasons,
                    "mean_cost_original": v.mean_cost_original,
                    "mean_cost_edited": v.mean_cost_edited,
                    "mean_saving": v.mean_saving,
                    "operational_total_tokens": v.operational_usage.total if v.operational_usage else None,
                }
                for v in verifications
            ]
        ),
    )
    put("branch_costs.csv", to_csv(branch_cost_rows(verifications)))
    td_rows = [training_data_row(t) for t in training_data]
    put("training_data.csv", to_csv(td_rows))

    md = [f"# {title}\n\n", DISCLAIMER + "\n\n"]
    md += [f"- {n}\n" for n in notes]
    md.append("\n## Unassisted evaluation by checkpoint, cycle and panel\n\n")
    md.append(md_table(_with_views(views["checkpoint_cycle_panel"]), [("checkpoint", "checkpoint"), ("cycle", "cycle"), ("panel", "panel"), *_SUMMARY_COLS]))
    for key, label in (("family", "family"), ("difficulty", "difficulty"), ("skill", "skill")):
        md.append(f"\n### Evaluation by {label}\n\n")
        md.append(md_table(_with_views(views[key]), [(key, label), *_SUMMARY_COLS]))
    md.append("\n## All episodes by role and cycle (stop reasons)\n\n")
    md.append(
        md_table(
            _with_views(by_role),
            [("role", "role"), ("cycle", "cycle"), ("success", "success"), ("total_tokens_sum", "tokens (sum)"), ("n_with_tokens", "episodes with tokens"), ("stop_categories", "stop categories"), ("stop_reasons", "stop reasons")],
        )
    )
    md.append("\nCached-input and reasoning tokens are subsets of input/output and are never added to totals (see CSVs).\n")
    md.append("\n## Stage effort\n\n")
    md.append(
        md_table(
            [_effort_view(e) for e in efforts],
            [
                ("cycle", "cycle"), ("stage", "stage"), ("n_items", "items"), ("input_tok_view", "input tok"), ("output_tok_view", "output tok"),
                ("total_tok_view", "total tok"), ("tokens_coverage", "items with tokens"), ("duration_sec", "active seconds"),
                ("optimizer_steps", "optimizer steps"), ("attempts", "attempts"), ("infra_failures", "infra retries"),
                ("retried_view", "retried attempts (tokens)"), ("cost_usd", "USD"), ("cost_provenance", "rate provenance"),
            ],
        )
    )
    md.append(
        "\nToken columns sum the items that reported usage ('items with tokens'); missing usage is never counted as 0. "
        "'infra retries' counts executions re-run after infrastructure failures; 'retried attempts' is the usage spent by "
        "those interrupted/failed executions, reported separately from final results.\n"
    )
    if not rates:
        md.append("\nNo cost rates configured: monetary cost not computed.\n")
    ps = proposal_stats(proposals, verifications)
    md.append("\n## Proposals and verification\n\n")
    md.append(
        md_table(
            [ps],
            [("n_sources", "sources"), ("n_proposals", "proposals"), ("n_proposed_valid", "valid"), ("n_abstained", "abstained"), ("n_invalid", "invalid"), ("n_verified", "verified"), ("n_accepted", "accepted"), ("yield_per_source", "yield/source")],
        )
    )
    md.append(f"\nInvalid proposal reasons: {_fmt(ps['invalid_reasons'])}\n\nVerification rejection reasons: {_fmt(ps['rejection_reasons'])}\n")
    if ps["n_editor_infra_errors"]:
        md.append(
            f"\nEditor infrastructure/request errors (not counted as invalid proposals): {ps['n_editor_infra_errors']}; "
            f"yield per source with editor output: {_fmt(ps['yield_per_source_with_editor_output'])} "
            f"(n={ps['n_sources_with_editor_output']}).\n"
        )
    labels = sorted({v.evidence_label for v in verifications if v.accepted and v.evidence_label})
    if labels:
        md.append(f"\nEvidence labels of accepted records: {', '.join(labels)}.\n")
    bs = branch_saving_summary(verifications)
    md.append("\n### Branch cost differences (original - edited)\n\n")
    md.append(
        md_table(
            bs["by_mode"],
            [
                ("mode", "mode"), ("n_verifications", "verifications"), ("n_all_branches_ok", "all branches succeeded"),
                ("mean_saving_all_branches_ok", "mean saving (all branches succeeded)"), ("n_all_branches_ok_with_costs", "n (with costs)"),
                ("n_accepted", "accepted"), ("mean_saving_accepted", "mean saving (accepted)"),
                ("n_edited_failed_original_ok", "edited failed / original ok"), ("n_original_failed_edited_ok", "original failed / edited ok"),
                ("n_branch_pairs", "branch pairs"),
            ],
        )
    )
    if bs["by_mode"]:
        md.append(
            "\nSavings are conditioned on every branch succeeding; a cheaper branch that failed or stopped early is not a saving. "
            + " ".join(f"Mode {r['mode']}: {r['saving_unit']}." for r in bs["by_mode"])
            + " Modes are never pooled.\n"
        )
    ve = verification_effort_split(verifications)
    if ve["n_branches"]:
        md.append("\n### Replay vs continuation effort (continuation verification)\n\n")
        if ve["available"]:
            md.append(
                md_table(
                    [
                        {"part": "replay (no model calls)", "tokens": ve["replay_tokens"], "tool_sec": ve["replay_tool_sec"]["sum"], "cpu_sec": ve["replay_cpu_sec"]["sum"], "wall_sec": ve["replay_wall_sec"]["sum"], "n": ve["replay_tool_sec"]["n"]},
                        {"part": "continuation", "tokens": ve["continuation_tokens"]["sum"], "tool_sec": ve["continuation_tool_sec"]["sum"], "cpu_sec": None, "wall_sec": None, "n": ve["continuation_tool_sec"]["n"]},
                    ],
                    [("part", "part"), ("tokens", "tokens"), ("tool_sec", "tool sec"), ("cpu_sec", "tool CPU sec"), ("wall_sec", "wall sec"), ("n", "branches")],
                )
            )
        md.append(
            f"\nBranches: {ve['n_branches']}; branch tool seconds (replay + continuation): {_fmt(ve['branch_tool_sec']['sum'])}; "
            f"continuation tokens: {_fmt(ve['continuation_tokens']['sum'])} (n={ve['continuation_tokens']['n']}).\n"
        )
        if ve.get("note"):
            md.append(f"\n{ve['note']}.\n")
    md.append("\n## Training data actually used\n\n")
    dropped_total = sum(r["dropped"] or 0 for r in td_rows)
    if dropped_total:
        md.append(f"**{dropped_total} exported pair(s) were dropped before training** (reasons and pair ids in "
                  "`training_data.csv`); the trained data differs from the exported data.\n\n")
    elif td_rows:
        md.append("No exported pair was dropped before training.\n\n")
    md.append(md_table(td_rows, [("cycle", "cycle"), ("exported", "exported pairs"), ("trained", "trained on"), ("dropped", "dropped"),
                                 ("dropped_by_reason", "drop reasons"), ("max_length", "max_length"), ("tokens_median", "median tokens"),
                                 ("tokens_max", "longest trained")]))
    md.append("\n## Preference datasets\n\n")
    md.append(
        md_table(
            [
                {
                    "dataset": (m.get("meta") or {}).get("name") or m.get("dataset_sha256", "")[:12],
                    "n": m.get("n_examples"),
                    "modes": m.get("verification_modes"),
                    "kinds": m.get("kinds"),
                    "families": (m.get("composition") or {}).get("by_family"),
                    "difficulties": (m.get("composition") or {}).get("by_difficulty"),
                }
                for m in manifests
            ],
            [("dataset", "dataset"), ("n", "pairs"), ("modes", "modes"), ("kinds", "kinds"), ("families", "families"), ("difficulties", "difficulties")],
        )
    )
    for c in comparisons:
        md.append("\n" + comparison_markdown(c))
    put("report.md", "".join(md))
    return paths


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #


def comparison_markdown(c: dict[str, Any]) -> str:
    a, b = c["label_a"], c["label_b"]
    t = c["transitions"]
    cov = c["token_delta_coverage"]
    lines = [
        f"## Paired comparison: {a} -> {b}\n\n",
        f"Pairs matched on panel/instance/attempt/seed: {c['n_pairs']} (unpaired: {c['n_only_a']} only in {a}, {c['n_only_b']} only in {b}; seed mismatches: {c['n_seed_mismatch']}).\n\n",
        md_table([t], [("both_succeed", "both succeed"), ("gained", f"gained in {b}"), ("lost", f"lost in {b}"), ("both_fail", "both fail"), ("undetermined", "undetermined")]),
        f"\nSuccess change (instance-level, {b} - {a}): {_interval(c['success_delta_instance_level'])}\n\n",
        f"Token delta ({b} - {a}) on both-succeed pairs only: {_interval(c['token_delta_instance_level'])}; "
        f"coverage {cov['n_both_succeed_with_tokens']}/{cov['n_pairs']} pairs; token ratio {_fmt(c['token_ratio_both_succeed'])}.\n",
    ]
    units = c.get("token_units") or {}
    if units.get("note"):
        lines.append(
            f"\n**Token units:** {units['note']}. Model identity {a}: {', '.join(units.get('model_identity_a') or []) or 'n/a'}; "
            f"{b}: {', '.join(units.get('model_identity_b') or []) or 'n/a'}. Usage sources {a}: {', '.join(units.get('usage_sources_a') or []) or 'n/a'}; "
            f"{b}: {', '.join(units.get('usage_sources_b') or []) or 'n/a'}.\n"
        )
    if t["lost"]:
        lines.append(f"\n**{t['lost']} previously successful pair(s) failed under {b}.** Token savings above exclude them.\n")
    return "".join(lines)


def load_run_rows(run_dir: str | Path) -> list[EpisodeRow]:
    """Episode rows of a run: `<run>/episodes.jsonl` if present, else read from the
    coordinator's stage directories."""
    p = Path(run_dir) / EPISODE_INDEX
    if p.exists():
        ident = run_model_identity(run_dir)
        rows = read_rows(p)
        return [r if r.model_identity or ident is None else r.model_copy(update={"model_identity": ident}) for r in rows]
    return collect_run(run_dir).rows


def _run_meta(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run.json"
    return read_json(path) if path.exists() else {}


def run_model_identity(run_dir: str | Path, meta: Mapping[str, Any] | None = None) -> str | None:
    """`<learner profile>@<base_revision[:12]>` from run.json, or None."""
    meta = _run_meta(run_dir) if meta is None else meta
    prof = (meta.get("model_profiles") or {}).get("learner") or {}
    return model_identity_of(prof.get("name"), prof.get("base_revision"))


def run_loop_seed(run_dir: str | Path, meta: Mapping[str, Any] | None = None) -> int | None:
    meta = _run_meta(run_dir) if meta is None else meta
    return ((meta.get("experiment") or {}).get("seeds") or {}).get("loop_seed")


def select_eval_rows(rows: list[EpisodeRow], *, checkpoint: str | None = None, panel: str | None = None) -> list[EpisodeRow]:
    """Unassisted evaluations of one checkpoint (default: the latest evaluated cycle)."""
    ev = [r for r in eval_rows(rows) if panel is None or r.panel == panel]
    if checkpoint is not None:
        return [r for r in ev if r.episode.checkpoint_id == checkpoint]
    if not ev:
        return []
    last = max(-1 if r.cycle is None else r.cycle for r in ev)
    return [r for r in ev if (-1 if r.cycle is None else r.cycle) == last]


def compare(
    run_a: str | Path | list[EpisodeRow],
    run_b: str | Path | list[EpisodeRow],
    *,
    out_dir: str | Path | None = None,
    panel: str | None = None,
    checkpoint_a: str | None = None,
    checkpoint_b: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    rows_a = run_a if isinstance(run_a, list) else load_run_rows(run_a)
    rows_b = run_b if isinstance(run_b, list) else load_run_rows(run_b)
    sel_a = select_eval_rows(rows_a, checkpoint=checkpoint_a, panel=panel)
    sel_b = select_eval_rows(rows_b, checkpoint=checkpoint_b, panel=panel)
    la = label_a or (sel_a[0].episode.checkpoint_id if sel_a else "a")
    lb = label_b or (sel_b[0].episode.checkpoint_id if sel_b else "b")
    if la == lb:
        la, lb = f"{la} (a)", f"{lb} (b)"
    c = paired_comparison(sel_a, sel_b, la, lb)
    if out_dir is not None:
        out = Path(out_dir)
        atomic_write_text(out / "paired_episodes.csv", to_csv(c["pairs"]))
        atomic_write_text(out / "comparison.md", f"# Comparison\n\n{DISCLAIMER}\n\n" + comparison_markdown(c))
        atomic_write_text(out / "comparison.json", json.dumps({k: v for k, v in c.items() if k != "pairs"}, indent=2, default=str) + "\n")
    return c


# --------------------------------------------------------------------------- #
# Run directories (coordinator layout)
# --------------------------------------------------------------------------- #


@dataclass
class RunRecords:
    run_id: str | None
    rows: list[EpisodeRow] = field(default_factory=list)
    proposals: list[EditProposal] = field(default_factory=list)
    verifications: list[VerificationRecord] = field(default_factory=list)
    efforts: list[StageEffort] = field(default_factory=list)
    dataset_manifests: list[dict[str, Any]] = field(default_factory=list)
    cycles: list[dict[str, Any]] = field(default_factory=list)
    training_data: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _duration(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return None


def _is_stage_manifest(d: Any) -> bool:
    return isinstance(d, dict) and "stage" in d and "items" in d


def _active_seconds(intervals: list[list[str]]) -> float | None:
    """Sum of recorded (start, end) execution intervals; None if any is open/unparsable."""
    total = 0.0
    for iv in intervals:
        d = _duration(iv[0] if iv else None, iv[1] if len(iv) > 1 else None)
        if d is None:
            return None
        total += d
    return total


def _attempt_usages(d: Path) -> list[Usage | None]:
    """Model usage recorded by one interrupted/failed attempt directory: its
    verification's operational usage if it finished, else every episode summary
    (top-level or branch) and proposal found below it. Nothing recorded -> []."""
    if (d / "verification.json").exists():
        v = VerificationRecord.model_validate(read_json(d / "verification.json"))
        return [v.operational_usage]
    out: list[Usage | None] = []
    for f in sorted(d.rglob("summary.json")):
        try:
            out.append(EpisodeSummary.model_validate(read_json(f)).usage)
        except Exception:  # noqa: BLE001 - partial files of a crashed attempt
            out.append(None)
    if (d / "proposal.json").exists():
        out.append(EditProposal.model_validate(read_json(d / "proposal.json")).usage)
    return out


def collect_run(run_dir: str | Path) -> RunRecords:
    """Read a coordinator run directory: stage manifests and their item outputs
    (summary.json / proposal.json / verification.json), dataset manifests and
    per-cycle state. Interrupted attempt directories are not counted as results;
    their usage is reported separately per stage as retried attempts, next to the
    manifest's attempt and infra-failure counts."""
    run_dir = Path(run_dir)
    meta = _run_meta(run_dir)
    instances = {k: TaskInstance.model_validate(v) for k, v in (meta.get("instances") or {}).items()}
    collection_panel = ((meta.get("experiment") or {}).get("tasks") or {}).get("collection_panel")
    ident = run_model_identity(run_dir, meta)
    rec = RunRecords(run_id=meta.get("run_id") or run_dir.name)
    for mpath in sorted(run_dir.rglob("manifest.json")):
        rel = mpath.relative_to(run_dir)
        if rel.parts[0] in ("tasks", "checkpoints") or any(".interrupted-" in part for part in rel.parts):
            continue
        data = read_json(mpath)
        if isinstance(data, dict) and data.get("format") == DATASET_FORMAT:
            rec.dataset_manifests.append({**data, "meta": {**(data.get("meta") or {}), "name": str(rel.parent)}})
            continue
        if not _is_stage_manifest(data):
            continue
        m = StageManifest.model_validate(data)
        usages: list[Usage | None] = []
        retried: list[Usage | None] = []
        n_retried = 0
        for item in m.items.values():
            for name in item.interrupted_dirs:
                n_retried += 1
                retried += _attempt_usages(mpath.parent / "items" / name)
            if not item.output:
                continue
            out = run_dir / item.output
            if (out / "summary.json").exists():
                s = EpisodeSummary.model_validate(read_json(out / "summary.json"))
                panel = item.meta.get("panel") or (collection_panel if s.role.value == "collect" else None)
                rec.rows += make_rows([s], instances, run_id=rec.run_id, cycle=m.cycle, panel=panel, model_identity=ident)
                usages.append(s.usage)
            if (out / "proposal.json").exists():
                p = EditProposal.model_validate(read_json(out / "proposal.json"))
                rec.proposals.append(p)
                usages.append(p.usage)
            if (out / "verification.json").exists():
                v = VerificationRecord.model_validate(read_json(out / "verification.json"))
                rec.verifications.append(v)
                usages.append(v.operational_usage)
        stage_name = {"verify": "replay_verification", "edit": "editor", "collect": "collection", "eval": "evaluation"}.get(m.stage, m.stage)
        span = _duration(m.started_at, m.finished_at)
        active = _active_seconds(m.active_intervals) if m.active_intervals else None
        present = [u for u in usages if u is not None]
        rec.efforts.append(
            StageEffort(
                stage=stage_name,
                cycle=m.cycle,
                usage=Usage.sum(present) if present and len(present) == len(usages) else None,
                duration_sec=active if active is not None else span,
                n_items=len(m.items),
                extra={
                    "status": m.status,
                    "counts": m.counts(),
                    "manifest": str(rel),
                    "usage_coverage": usage_coverage(usages) if usages else None,
                    "attempts": sum(i.attempts for i in m.items.values()),
                    "infra_failures": sum(i.infra_failures for i in m.items.values()),
                    "n_retried_attempts": n_retried,
                    "retried_usage_coverage": usage_coverage(retried) if retried else None,
                    "wall_span_sec": span,
                    "duration_basis": "active_intervals" if active is not None else "started_at..finished_at span",
                },
            )
        )
    for cpath in sorted((run_dir / "cycles").glob("cycle-*/cycle.json")):
        st = read_json(cpath)
        rec.cycles.append({k: st.get(k) for k in ("cycle", "status", "update", "reference")} | {
            "learner_in": (st.get("learner_in") or {}).get("checkpoint_id"),
            "learner_out": (st.get("learner_out") or {}).get("checkpoint_id"),
        })
        if st.get("training_data"):
            rec.training_data.append({"cycle": st.get("cycle"), **st["training_data"]})
        ck = st.get("checkpoint_record")
        if ck:
            metrics = ck.get("metrics") or {}
            dur = (metrics.get("durations_sec") or {}).get("total", metrics.get("duration_sec"))
            rec.efforts.append(
                StageEffort(stage="training", cycle=st.get("cycle"), duration_sec=dur, optimizer_steps=ck.get("optimizer_steps"), n_items=ck.get("n_train_examples"), extra={"trainer": ck.get("trainer")})
            )
    rec.efforts.sort(key=lambda e: (-1 if e.cycle is None else e.cycle, e.stage))
    return rec


def write_run_report(run_dir: str | Path, *, rates: Mapping[str, CostRate] | None = None, out_dir: str | Path | None = None) -> Path:
    """Regenerate `<run>/reports/` from the run directory; returns report.md's path.

    Adds a paired comparison between the first and last evaluated checkpoints of
    the run on the same dev panels when both exist."""
    run_dir = Path(run_dir)
    rec = collect_run(run_dir)
    comparisons = []
    evals = eval_rows(rec.rows)
    cycles = sorted({r.cycle for r in evals if r.cycle is not None})
    if len(cycles) >= 2:
        first = [r for r in evals if r.cycle == cycles[0]]
        last = [r for r in evals if r.cycle == cycles[-1]]
        la, lb = f"cycle {cycles[0]}", f"cycle {cycles[-1]}"
        comparisons.append(paired_comparison(first, last, la, lb))
    notes = [f"run {rec.run_id}"]
    for c in rec.cycles:
        notes.append(
            f"cycle {c.get('cycle')}: {c.get('status')}, update={c.get('update')}, learner {c.get('learner_in')} -> {c.get('learner_out')}"
            + (f", DPO reference {c.get('reference')}" if c.get("reference") else "")
        )
    out = Path(out_dir) if out_dir is not None else run_dir / "reports"
    paths = write_report(
        out,
        rec.rows,
        proposals=rec.proposals,
        verifications=rec.verifications,
        stage_efforts=rec.efforts,
        dataset_manifests=rec.dataset_manifests,
        rates=rates,
        comparisons=comparisons,
        title=f"Run report: {rec.run_id}",
        notes=notes,
        training_data=rec.training_data,
    )
    return paths["report.md"]


def compare_runs(run_a: str | Path, run_b: str | Path, *, out_dir: str | Path | None = None, panel: str | None = None) -> str:
    """CLI entry: paired comparison of the latest evaluated checkpoints of two runs.
    Returns the markdown text (also written to out_dir when given)."""
    c = compare(run_a, run_b, out_dir=out_dir, panel=panel, label_a=Path(run_a).name, label_b=Path(run_b).name)
    return comparison_markdown(c)


# --------------------------------------------------------------------------- #
# Conditions over independent learning-loop seeds
# --------------------------------------------------------------------------- #

RunRef = str | Path | list[EpisodeRow]


def _run_label(run: RunRef, i: int) -> str:
    return Path(run).name if not isinstance(run, list) else f"run{i}"


def _pair_runs(runs_a: list[RunRef], runs_b: list[RunRef]) -> tuple[list[tuple[Any, RunRef, RunRef]], str]:
    """Match runs of the two conditions by loop seed (run.json) when every run has
    one and the seed sets are equal; a single b run (e.g. a frozen baseline) is
    compared with every a run; otherwise runs are matched by position."""
    seeds_a = [None if isinstance(r, list) else run_loop_seed(r) for r in runs_a]
    seeds_b = [None if isinstance(r, list) else run_loop_seed(r) for r in runs_b]
    if len(runs_b) == 1 and len(runs_a) > 1:
        return [(seeds_a[i] if seeds_a[i] is not None else _run_label(r, i), r, runs_b[0]) for i, r in enumerate(runs_a)], "every a run vs the single b run"
    if len(runs_a) != len(runs_b):
        raise ValueError(f"need the same number of runs per condition (or one b run): {len(runs_a)} vs {len(runs_b)}")
    known = all(s is not None for s in seeds_a + seeds_b)
    if known and len(set(seeds_a)) == len(seeds_a) and set(seeds_a) == set(seeds_b):
        by_b = dict(zip(seeds_b, runs_b, strict=True))
        return [(sd, r, by_b[sd]) for sd, r in sorted(zip(seeds_a, runs_a, strict=True), key=lambda t: t[0])], "matched by loop seed"
    return [(f"{_run_label(a, i)} | {_run_label(b, i)}", a, b) for i, (a, b) in enumerate(zip(runs_a, runs_b, strict=True))], "matched by position (loop seeds unknown or not equal)"


def _across_view(x: dict[str, Any]) -> str:
    if x.get("mean") is None:
        return "n/a"
    spread = f"sd {_fmt(x['sd'])}, range [{_fmt(x['min'])}, {_fmt(x['max'])}]" if x.get("sd") is not None else f"single value {_fmt(x['mean'])}"
    if x.get("ci95_low") is None:
        return f"{_fmt(x['mean'])} ({spread}; {x['interval']})"
    return f"{_fmt(x['mean'])} [{_fmt(x['ci95_low'])}, {_fmt(x['ci95_high'])}] ({spread}; {x['interval']})"


def multi_seed_markdown(c: dict[str, Any]) -> str:
    a, b = c["label_a"], c["label_b"]
    lines = [
        f"## Condition comparison across loop seeds: {a} -> {b}\n\n",
        f"Independent learning-loop seeds: {c['n_seeds']} ({c.get('pairing', 'pairing n/a')}). Within a seed, effects are paired "
        f"on panel/instance/attempt/seed and averaged over task instances; across seeds the seed is the unit. "
        f"Intervals need >= {c['min_seeds_for_interval']} seeds.\n\n",
        md_table(
            c["per_seed"],
            [("seed", "loop seed"), ("n_pairs", "pairs"), ("n_instances", "instances"), ("success_delta", f"success change ({b} - {a})"),
             ("gained", "gained"), ("lost", "lost"), ("token_delta_both_succeed", "token delta (both succeed)"), ("n_instances_tokens", "instances w/ tokens"),
             ("token_ratio_both_succeed", "token ratio"), ("token_units_note", "token units")],
        ),
        f"\nSuccess change across seeds: {_across_view(c['success_delta'])}\n\n",
        f"Token delta on both-succeed pairs across seeds: {_across_view(c['token_delta_both_succeed'])}\n\n",
        f"Token ratio across seeds: {_across_view(c['token_ratio_both_succeed'])}\n",
    ]
    if c["lost_total"]:
        lines.append(f"\n**{c['lost_total']} previously successful pair(s) failed under {b} (summed over seeds).**\n")
    if c["n_seeds"] < c["min_seeds_for_interval"]:
        lines.append(f"\nWith {c['n_seeds']} seed(s) this is a description of these runs, not evidence about the method.\n")
    return "".join(lines)


def compare_conditions(
    runs_a: list[RunRef],
    runs_b: list[RunRef],
    *,
    out_dir: str | Path | None = None,
    panel: str | None = None,
    checkpoint_a: str | None = None,
    checkpoint_b: str | None = None,
    label_a: str = "a",
    label_b: str = "b",
    min_seeds: int = MIN_SEEDS_FOR_INTERVAL,
) -> dict[str, Any]:
    """Compare two conditions, each given as several runs (one per independent
    learning-loop seed; run dirs or row lists). For each seed: the latest
    evaluated checkpoint of each run (or the given checkpoints) is compared with
    `paired_comparison`; then the per-seed effects are summarized across seeds
    (`metrics.multi_seed_comparison`). Returns the dict, with the rendered text
    under "markdown"; writes `conditions.md` / `conditions.json` / `per_seed.csv`
    to `out_dir` when given."""
    if not runs_a or not runs_b:
        raise ValueError("each condition needs at least one run")
    matched, how = _pair_runs(runs_a, runs_b)
    seed_pairs = []
    for seed, ra, rb in matched:
        rows_a = ra if isinstance(ra, list) else load_run_rows(ra)
        rows_b = rb if isinstance(rb, list) else load_run_rows(rb)
        seed_pairs.append((seed, select_eval_rows(rows_a, checkpoint=checkpoint_a, panel=panel), select_eval_rows(rows_b, checkpoint=checkpoint_b, panel=panel)))
    c = multi_seed_comparison(seed_pairs, label_a, label_b, min_seeds=min_seeds)
    c["pairing"] = how
    c["markdown"] = multi_seed_markdown(c)
    if out_dir is not None:
        out = Path(out_dir)
        atomic_write_text(out / "per_seed.csv", to_csv(c["per_seed"]))
        atomic_write_text(out / "conditions.md", f"# Condition comparison\n\n{DISCLAIMER}\n\n" + c["markdown"])
        atomic_write_text(out / "conditions.json", json.dumps({k: v for k, v in c.items() if k != "markdown"}, indent=2, default=str) + "\n")
    return c
