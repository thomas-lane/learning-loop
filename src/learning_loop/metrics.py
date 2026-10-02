"""Episode, stage and verification aggregation from plain record lists.

Inputs are plain lists of records plus a small index:

  - `EpisodeRow`: one EpisodeSummary + where it belongs (run, cycle, panel) and
    task metadata (family, difficulty, skills). Build them with `make_rows()`
    from summaries and a {instance_id: TaskInstance} index; the coordinator
    appends them to `runs/<id>/episodes.jsonl` (`append_rows`).
  - EditProposal / VerificationRecord lists, `StageEffort` records, dataset
    manifests.

Rules kept here:
  - Missing measurements stay missing: sums are over available values and
    always come with their coverage count; None is never read as 0.
  - Cached-input and reasoning tokens are subsets of input/output; they are
    reported next to, never added to, totals.
  - Complete success and partial reward are separate; ungraded / infra-failed
    episodes stay in the denominators.
  - Uncertainty is over task instances (mean over instances of the per-instance
    mean); a percentile bootstrap over instances is reported only when there
    are >= MIN_INSTANCES_FOR_INTERVAL instances, else "n too small for interval".
  - Paired comparisons report the full success-transition table; token deltas
    are computed only on pairs where both sides succeed, with their coverage,
    and only when both sides count tokens in the same units (same model
    profile/base revision and usage source); otherwise they are None with an
    explicit "incomparable token units" note.
  - Method comparisons across independent learning-loop seeds use the seed as
    the unit (`multi_seed_comparison`): one paired effect per seed, then mean and
    spread across seeds; no interval below MIN_SEEDS_FOR_INTERVAL seeds.
  - Money is computed only from an explicitly configured `CostRate` with
    provenance; there are no built-in prices.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .records import EditProposal, EpisodeRole, EpisodeSummary, StopCategory, TaskInstance, Usage, VerificationRecord
from .seeds import stable_id
from .storage import JsonlAppender, read_jsonl

MIN_INSTANCES_FOR_INTERVAL = 5
N_BOOTSTRAP = 2000
SMALL_N_NOTE = "n too small for interval"
INCOMPARABLE_TOKENS_NOTE = "incomparable token units"
MIN_SEEDS_FOR_INTERVAL = 3
FEW_SEEDS_NOTE = "too few seeds for interval"


class EpisodeRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode: EpisodeSummary
    run_id: str | None = None
    cycle: int | None = None
    panel: str | None = None
    family: str | None = None
    difficulty: str | None = None
    skills: list[str] = Field(default_factory=list)
    # Learner model identity "<model_profile>@<base_revision[:12]>": token counts are
    # only comparable between rows with the same identity (tokenizer/template).
    model_identity: str | None = None


_BASE_CKPT = re.compile(r"^base:(?P<profile>[^@]+)@(?P<rev>\w+)$")


def model_identity_of(profile_name: str | None, base_revision: str | None) -> str | None:
    if not profile_name or not base_revision:
        return None
    return f"{profile_name}@{base_revision[:12]}"


def token_identity(row: EpisodeRow) -> str | None:
    """The row's model identity, else parsed from a base checkpoint id
    (`base:<profile>@<rev12>`); None when unknown (e.g. a trained checkpoint id
    from a row without `model_identity`)."""
    if row.model_identity:
        return row.model_identity
    m = _BASE_CKPT.match(row.episode.checkpoint_id)
    return model_identity_of(m.group("profile"), m.group("rev")) if m else None


def make_rows(
    episodes: Iterable[EpisodeSummary],
    instances: Mapping[str, TaskInstance],
    *,
    run_id: str | None = None,
    cycle: int | None = None,
    panel: str | None = None,
    model_identity: str | None = None,
) -> list[EpisodeRow]:
    rows = []
    for e in episodes:
        inst = instances.get(e.instance_id)
        rows.append(
            EpisodeRow(
                episode=e,
                run_id=run_id,
                cycle=cycle,
                panel=panel,
                family=inst.family if inst else None,
                difficulty=inst.difficulty if inst else None,
                skills=list(inst.skills) if inst else [],
                model_identity=model_identity,
            )
        )
    return rows


def append_rows(path: str | Path, rows: Iterable[EpisodeRow]) -> None:
    out = JsonlAppender(Path(path))
    for r in rows:
        out.append(r)


def read_rows(path: str | Path) -> list[EpisodeRow]:
    return [EpisodeRow.model_validate(r) for r in read_jsonl(Path(path))]


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _avail(values: Iterable[float | int | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def agg(values: Iterable[float | int | None]) -> dict[str, Any]:
    """{sum, mean, n (available), n_missing} - missing values never count as 0."""
    vals = list(values)
    got = _avail(vals)
    return {
        "sum": sum(got) if got else None,
        "mean": sum(got) / len(got) if got else None,
        "n": len(got),
        "n_missing": len(vals) - len(got),
    }


def bootstrap_ci(values: list[float], seed: int, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def instance_level(values_by_instance: Mapping[str, list[float]], seed_parts: tuple[Any, ...] = ()) -> dict[str, Any]:
    """Mean over instances of the per-instance mean, with an instance bootstrap when n >= 5."""
    per = {k: sum(v) / len(v) for k, v in values_by_instance.items() if v}
    n = len(per)
    out: dict[str, Any] = {"mean": None, "n_instances": n, "ci95_low": None, "ci95_high": None, "interval": None}
    if n == 0:
        out["interval"] = "no data"
        return out
    vals = [per[k] for k in sorted(per)]
    out["mean"] = sum(vals) / n
    if n < MIN_INSTANCES_FOR_INTERVAL:
        out["interval"] = SMALL_N_NOTE
    else:
        lo, hi = bootstrap_ci(vals, int(stable_id("bootstrap", *seed_parts)[-12:], 16))
        out.update(ci95_low=lo, ci95_high=hi, interval=f"bootstrap over {n} instances, {N_BOOTSTRAP} resamples, 95% percentile")
    return out


# --------------------------------------------------------------------------- #
# Episode summaries
# --------------------------------------------------------------------------- #


def summarize(rows: list[EpisodeRow]) -> dict[str, Any]:
    eps = [r.episode for r in rows]
    n = len(eps)
    graded = [e for e in eps if e.success is not None]
    n_success = sum(1 for e in eps if e.success is True)
    usage = [e.usage for e in eps]
    tot = agg(u.total for u in usage)
    succ_tot = agg(e.usage.total for e in eps if e.success is True)
    by_inst_success: dict[str, list[float]] = defaultdict(list)
    by_inst_tokens: dict[str, list[float]] = defaultdict(list)
    for e in eps:
        by_inst_success[e.instance_id].append(1.0 if e.success is True else 0.0)
        if e.usage.total is not None:
            by_inst_tokens[e.instance_id].append(float(e.usage.total))
    return {
        "n_episodes": n,
        "n_instances": len({e.instance_id for e in eps}),
        "n_graded": len(graded),
        "n_ungraded": n - len(graded),
        "n_success": n_success,
        "success_rate": n_success / n if n else None,  # ungraded/infra count as not successful
        "mean_partial_reward": agg(e.partial_reward for e in eps)["mean"],
        "n_partial_reward": agg(e.partial_reward for e in eps)["n"],
        "stop_categories": dict(Counter(e.stop_category.value for e in eps)),
        "stop_reasons": dict(Counter(e.stop_reason for e in eps)),
        "n_infra_failures": sum(1 for e in eps if e.stop_category == StopCategory.INFRA),
        "input_tokens_sum": agg(u.input_tokens for u in usage)["sum"],
        "output_tokens_sum": agg(u.output_tokens for u in usage)["sum"],
        "total_tokens_sum": tot["sum"],
        "total_tokens_mean": tot["mean"],
        "n_with_tokens": tot["n"],
        "total_tokens_mean_successful": succ_tot["mean"],
        "cached_input_tokens_sum (subset of input)": agg(u.cached_input_tokens for u in usage)["sum"],
        "n_with_cached": agg(u.cached_input_tokens for u in usage)["n"],
        "reasoning_tokens_sum (subset of output)": agg(u.reasoning_tokens for u in usage)["sum"],
        "n_with_reasoning": agg(u.reasoning_tokens for u in usage)["n"],
        "usage_sources": dict(Counter(u.source for u in usage)),
        "n_requests_sum": sum(e.n_requests for e in eps),
        "n_tool_calls_sum": sum(e.n_tool_calls for e in eps),
        "n_malformed_turns_sum": sum(e.n_malformed_turns for e in eps),
        "tool_sec_sum": agg(e.timing.tool_sec for e in eps)["sum"],
        "endpoint_sec_sum": agg(e.timing.endpoint_sec for e in eps)["sum"],
        "total_sec_sum": agg(e.timing.total_sec for e in eps)["sum"],
        "queue_sec_sum": agg(e.timing.queue_sec for e in eps)["sum"],
        "tool_cpu_sec_sum": agg(e.tool_cpu_sec for e in eps)["sum"],
        "n_with_tool_cpu": agg(e.tool_cpu_sec for e in eps)["n"],
        "tool_peak_memory_bytes_max": max(_avail(e.tool_peak_memory_bytes for e in eps), default=None),
        "instance_success": instance_level(by_inst_success, ("success",)),
        "instance_total_tokens": instance_level(by_inst_tokens, ("tokens",)),
    }


GROUP_KEYS: dict[str, Callable[[EpisodeRow], list[Any]]] = {
    "checkpoint": lambda r: [r.episode.checkpoint_id],
    "cycle": lambda r: [r.cycle],
    "panel": lambda r: [r.panel],
    "family": lambda r: [r.family],
    "difficulty": lambda r: [r.difficulty],
    "skill": lambda r: list(r.skills) or [None],  # a row counts once per skill
    "role": lambda r: [r.episode.role.value],
}


def group_summaries(rows: list[EpisodeRow], by: tuple[str, ...]) -> list[dict[str, Any]]:
    """Summaries keyed by any combination of GROUP_KEYS, sorted by key."""
    groups: dict[tuple[Any, ...], list[EpisodeRow]] = defaultdict(list)
    for r in rows:
        combos: list[tuple[Any, ...]] = [()]
        for key in by:
            combos = [c + (v,) for c in combos for v in GROUP_KEYS[key](r)]
        for c in combos:
            groups[c].append(r)
    out = []
    for k in sorted(groups, key=lambda t: tuple("" if v is None else str(v) for v in t)):
        out.append({**dict(zip(by, k, strict=True)), **summarize(groups[k])})
    return out


# --------------------------------------------------------------------------- #
# Paired fixed-panel comparison
# --------------------------------------------------------------------------- #


def _pair_key(r: EpisodeRow) -> tuple[Any, ...]:
    e = r.episode
    return (r.panel, e.instance_id, e.attempt_index, e.seed)


def token_units(rows_a: list[EpisodeRow], rows_b: list[EpisodeRow]) -> dict[str, Any]:
    """Whether token counts of two row sets are in the same units.

    Incomparable when both sides have known model identities that differ, or
    when their usage sources differ (provider counts vs local recount vs fixture
    estimate). Unknown identities are reported, not assumed equal."""
    ids_a = sorted({str(token_identity(r)) for r in rows_a})
    ids_b = sorted({str(token_identity(r)) for r in rows_b})
    src_a = sorted({r.episode.usage.source for r in rows_a if r.episode.usage.total is not None})
    src_b = sorted({r.episode.usage.source for r in rows_b if r.episode.usage.total is not None})
    known = "None" not in ids_a and "None" not in ids_b
    reasons = []
    if known and ids_a != ids_b:
        reasons.append(f"model identity differs ({', '.join(ids_a)} vs {', '.join(ids_b)})")
    if src_a and src_b and src_a != src_b:
        reasons.append(f"usage source differs ({', '.join(src_a)} vs {', '.join(src_b)})")
    note = None
    if reasons:
        note = f"{INCOMPARABLE_TOKENS_NOTE}: " + "; ".join(reasons)
    elif not known:
        note = "model identity unknown on at least one side; token units assumed, not verified"
    return {
        "comparable": not reasons,
        "note": note,
        "model_identity_a": ids_a,
        "model_identity_b": ids_b,
        "usage_sources_a": src_a,
        "usage_sources_b": src_b,
    }


def paired_comparison(rows_a: list[EpisodeRow], rows_b: list[EpisodeRow], label_a: str = "a", label_b: str = "b") -> dict[str, Any]:
    """Pair on (panel, instance, attempt, seed). Every transition is reported;
    lost successes are never hidden behind cheaper remaining successes. Token
    deltas/ratio are None when the sides' token units differ (`token_units`)."""
    ia: dict[tuple[Any, ...], EpisodeRow] = {}
    ib: dict[tuple[Any, ...], EpisodeRow] = {}
    for idx, rows in ((ia, rows_a), (ib, rows_b)):
        for r in rows:
            k = _pair_key(r)
            if k in idx:
                raise ValueError(f"duplicate paired key {k}: dedupe logical trials first")
            idx[k] = r
    keys = sorted(set(ia) & set(ib), key=lambda t: tuple(str(x) for x in t))
    seed_mismatch = {(k[0], k[1], k[2]) for k in ia} & {(k[0], k[1], k[2]) for k in ib}
    seed_mismatch -= {(k[0], k[1], k[2]) for k in keys}
    units = token_units([ia[k] for k in keys], [ib[k] for k in keys])
    trans = Counter()
    pairs = []
    deltas_by_inst: dict[str, list[float]] = defaultdict(list)
    succ_delta_by_inst: dict[str, list[float]] = defaultdict(list)
    tok_a = tok_b = 0.0
    n_both = n_both_tok = 0
    for k in keys:
        a, b = ia[k].episode, ib[k].episode
        sa, sb = a.success, b.success
        if sa is None or sb is None:
            t = "undetermined"
        elif sa and sb:
            t = "both_succeed"
        elif sb and not sa:
            t = "gained"
        elif sa and not sb:
            t = "lost"
        else:
            t = "both_fail"
        trans[t] += 1
        delta = None
        if t == "both_succeed":
            n_both += 1
            if a.usage.total is not None and b.usage.total is not None:
                n_both_tok += 1
                if units["comparable"]:
                    delta = float(b.usage.total - a.usage.total)
                    deltas_by_inst[a.instance_id].append(delta)
                    tok_a += a.usage.total
                    tok_b += b.usage.total
        if t != "undetermined":
            succ_delta_by_inst[a.instance_id].append(float(bool(sb)) - float(bool(sa)))
        pairs.append(
            {
                "panel": k[0],
                "instance_id": k[1],
                "attempt_index": k[2],
                "seed": k[3],
                f"success_{label_a}": sa,
                f"success_{label_b}": sb,
                f"partial_reward_{label_a}": a.partial_reward,
                f"partial_reward_{label_b}": b.partial_reward,
                f"total_tokens_{label_a}": a.usage.total,
                f"total_tokens_{label_b}": b.usage.total,
                "transition": t,
                "token_delta_both_succeed": delta,
            }
        )
    n_pairs = len(keys)
    if units["comparable"]:
        delta_il = instance_level(deltas_by_inst, ("delta", label_a, label_b))
    else:
        delta_il = {"mean": None, "n_instances": 0, "ci95_low": None, "ci95_high": None, "interval": INCOMPARABLE_TOKENS_NOTE}
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": n_pairs,
        "n_only_a": len(set(ia) - set(ib)),
        "n_only_b": len(set(ib) - set(ia)),
        "n_seed_mismatch": len(seed_mismatch),
        "transitions": {t: trans.get(t, 0) for t in ("both_succeed", "gained", "lost", "both_fail", "undetermined")},
        "token_units": units,
        "token_delta_coverage": {"n_both_succeed_with_tokens": n_both_tok, "n_pairs": n_pairs, "fraction": n_both_tok / n_pairs if n_pairs else None},
        "token_delta_instance_level": delta_il,
        "token_ratio_both_succeed": (tok_b / tok_a) if units["comparable"] and n_both_tok and tok_a else None,
        "success_delta_instance_level": instance_level(succ_delta_by_inst, ("succ", label_a, label_b)),
        "pairs": pairs,
    }


# --------------------------------------------------------------------------- #
# Independent learning-loop seeds
# --------------------------------------------------------------------------- #

# Two-sided 95% Student-t critical values by degrees of freedom (df >= 31: normal 1.96 is
# within 3%; we use the df=30 value there, which is conservative).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def across_seeds(values: list[float | None], min_seeds: int = MIN_SEEDS_FOR_INTERVAL) -> dict[str, Any]:
    """Mean and spread of one per-seed effect across independent loop seeds.

    Seeds with no value (e.g. incomparable tokens) are counted as missing. A
    95% t-interval over seeds is given only with >= `min_seeds` seeds."""
    got = _avail(values)
    n = len(got)
    out: dict[str, Any] = {"mean": None, "sd": None, "min": None, "max": None, "n_seeds": n, "n_missing": len(values) - n,
                           "ci95_low": None, "ci95_high": None, "interval": None}
    if n == 0:
        out["interval"] = "no data"
        return out
    mean = sum(got) / n
    out.update(mean=mean, min=min(got), max=max(got))
    if n >= 2:
        out["sd"] = math.sqrt(sum((v - mean) ** 2 for v in got) / (n - 1))
    if n < max(2, min_seeds):
        out["interval"] = f"{FEW_SEEDS_NOTE} (n_seeds={n} < {max(2, min_seeds)})"
    else:
        half = _T95[min(n - 1, 30)] * out["sd"] / math.sqrt(n)
        out.update(ci95_low=mean - half, ci95_high=mean + half, interval=f"t-interval over {n} loop seeds (seed = unit)")
    return out


def multi_seed_comparison(
    seed_pairs: list[tuple[Any, list[EpisodeRow], list[EpisodeRow]]],
    label_a: str = "a",
    label_b: str = "b",
    min_seeds: int = MIN_SEEDS_FOR_INTERVAL,
) -> dict[str, Any]:
    """Per-seed paired instance-level effects of condition b vs a, then their mean
    and spread across seeds. `seed_pairs` = [(seed label, rows_a, rows_b)], one
    entry per independent learning-loop seed (rows already restricted to the
    evaluations being compared). Within a seed the unit is the task instance;
    across seeds the unit is the seed, never the pooled episodes."""
    per_seed = []
    for seed, ra, rb in seed_pairs:
        c = paired_comparison(ra, rb, label_a, label_b)
        per_seed.append(
            {
                "seed": seed,
                "n_pairs": c["n_pairs"],
                "success_delta": c["success_delta_instance_level"]["mean"],
                "n_instances": c["success_delta_instance_level"]["n_instances"],
                "token_delta_both_succeed": c["token_delta_instance_level"]["mean"],
                "n_instances_tokens": c["token_delta_instance_level"]["n_instances"],
                "token_ratio_both_succeed": c["token_ratio_both_succeed"],
                "lost": c["transitions"]["lost"],
                "gained": c["transitions"]["gained"],
                "token_units_note": c["token_units"]["note"],
            }
        )
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_seeds": len(per_seed),
        "min_seeds_for_interval": min_seeds,
        "per_seed": per_seed,
        "success_delta": across_seeds([p["success_delta"] for p in per_seed], min_seeds),
        "token_delta_both_succeed": across_seeds([p["token_delta_both_succeed"] for p in per_seed], min_seeds),
        "token_ratio_both_succeed": across_seeds([p["token_ratio_both_succeed"] for p in per_seed], min_seeds),
        "lost_total": sum(p["lost"] for p in per_seed),
        "gained_total": sum(p["gained"] for p in per_seed),
    }


# --------------------------------------------------------------------------- #
# Stage effort, proposals, verification
# --------------------------------------------------------------------------- #


class CostRate(BaseModel):
    """An explicit, sourced rate. No defaults: money is never assumed."""

    model_config = ConfigDict(extra="forbid")
    usd_per_million_input_tokens: float | None = None
    usd_per_million_output_tokens: float | None = None
    usd_per_hour: float | None = None
    provenance: str


class StageEffort(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str  # collection | evaluation | editor | replay_verification | audit | training
    cycle: int | None = None
    usage: Usage | None = None
    duration_sec: float | None = None
    optimizer_steps: int | None = None
    n_items: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def usage_coverage(usages: list[Usage | None]) -> dict[str, Any]:
    """Per-component sums over AVAILABLE values with coverage counts (never
    all-or-nothing): {input_tokens|output_tokens|total: agg(...), n_items, sources}."""
    us = list(usages)
    return {
        "input_tokens": agg(u.input_tokens if u else None for u in us),
        "output_tokens": agg(u.output_tokens if u else None for u in us),
        "total": agg(u.total if u else None for u in us),
        "n_items": len(us),
        "sources": dict(Counter(u.source if u else "none" for u in us)),
    }


def _coverage_view(cov: Mapping[str, Any] | None) -> str | None:
    if not cov or not cov.get("n_items"):
        return None
    t = cov["total"]
    return f"{t['n']}/{t['n'] + t['n_missing']}"


def stage_effort_table(efforts: list[StageEffort], rates: Mapping[str, CostRate] | None = None) -> list[dict[str, Any]]:
    """One row per stage effort. When `extra["usage_coverage"]` is present (see
    `usage_coverage`), token columns are sums over items that reported usage and
    `tokens_coverage` says how many did; money from tokens is then computed only
    when coverage is complete. `infra_failures`/`attempts` come from the stage
    manifest (retried executions); `retried_*` is usage spent by interrupted or
    infra-failed attempts, reported separately from the final results."""
    rows = []
    for s in efforts:
        u = s.usage
        cov = s.extra.get("usage_coverage")
        if cov:
            inp, outp, tot = cov["input_tokens"], cov["output_tokens"], cov["total"]
            input_tokens, output_tokens, total_tokens = inp["sum"], outp["sum"], tot["sum"]
            input_complete, output_complete = inp["n_missing"] == 0, outp["n_missing"] == 0
            usage_source = ",".join(sorted(cov.get("sources") or {})) or None
        else:
            input_tokens = u.input_tokens if u else None
            output_tokens = u.output_tokens if u else None
            total_tokens = u.total if u else None
            input_complete, output_complete = input_tokens is not None, output_tokens is not None
            usage_source = u.source if u else None
        retried = s.extra.get("retried_usage_coverage")
        row: dict[str, Any] = {
            "stage": s.stage,
            "cycle": s.cycle,
            "n_items": s.n_items,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens_coverage": _coverage_view(cov),
            "usage_source": usage_source,
            "duration_sec": s.duration_sec,
            "wall_span_sec": s.extra.get("wall_span_sec"),
            "optimizer_steps": s.optimizer_steps,
            "attempts": s.extra.get("attempts"),
            "infra_failures": s.extra.get("infra_failures"),
            "n_retried_attempts": s.extra.get("n_retried_attempts"),
            "retried_total_tokens": retried["total"]["sum"] if retried else None,
            "retried_tokens_coverage": _coverage_view(retried),
            "cost_usd": None,
            "cost_provenance": None,
        }
        rate = (rates or {}).get(s.stage)
        if rate is not None:
            parts: list[float | None] = []
            if rate.usd_per_million_input_tokens is not None:
                parts.append(None if not input_complete or input_tokens is None else input_tokens * rate.usd_per_million_input_tokens / 1e6)
            if rate.usd_per_million_output_tokens is not None:
                parts.append(None if not output_complete or output_tokens is None else output_tokens * rate.usd_per_million_output_tokens / 1e6)
            if rate.usd_per_hour is not None:
                parts.append(None if s.duration_sec is None else s.duration_sec / 3600 * rate.usd_per_hour)
            if parts and all(p is not None for p in parts):
                row["cost_usd"] = sum(parts)  # type: ignore[arg-type]
                row["cost_provenance"] = rate.provenance
            elif parts:
                row["cost_provenance"] = f"{rate.provenance} (incomplete measurements; not computed)"
        rows.append(row)
    return rows


def _reason_key(reason: str) -> str:
    return reason.split(":", 1)[0]


EDITOR_INFRA_PREFIX = "editor_infra_error"


def _is_editor_infra(p: EditProposal) -> bool:
    return p.status == "invalid" and any(_reason_key(r) == EDITOR_INFRA_PREFIX for r in p.rejection_reasons)


def proposal_stats(
    proposals: list[EditProposal], verifications: list[VerificationRecord], n_sources: int | None = None
) -> dict[str, Any]:
    """Proposal yield. Editor infrastructure/request failures (reason prefix
    `editor_infra_error`) are counted separately from invalid proposals: they
    say nothing about the editor's proposals. They stay in the per-source yield
    denominator (a lost proposal is lost yield); `n_sources_with_editor_output`
    gives the denominator without them."""
    acc = [v for v in verifications if v.purpose == "acceptance"]
    infra = [p for p in proposals if _is_editor_infra(p)]
    real = [p for p in proposals if not _is_editor_infra(p)]
    status = Counter(p.status for p in real)
    invalid_reasons = Counter(_reason_key(r) for p in real if p.status == "invalid" for r in p.rejection_reasons)
    rejection = Counter(_reason_key(r) for v in acc if not v.accepted for r in v.reasons)
    n_acc = sum(1 for v in acc if v.accepted)
    denom = n_sources if n_sources is not None else len({p.source_episode_id for p in proposals})
    infra_only_sources = {p.source_episode_id for p in infra} - {p.source_episode_id for p in real}
    denom_output = denom - len(infra_only_sources) if denom else denom
    return {
        "n_sources": denom,
        "n_proposals": len(proposals),
        "n_proposed_valid": status.get("proposed", 0),
        "n_abstained": status.get("abstained", 0),
        "n_invalid": status.get("invalid", 0),
        "invalid_reasons": dict(invalid_reasons),
        "n_editor_infra_errors": len(infra),
        "editor_infra_errors": sorted({r for p in infra for r in p.rejection_reasons if _reason_key(r) == EDITOR_INFRA_PREFIX}),
        "n_verified": len(acc),
        "n_accepted": n_acc,
        "n_rejected": len(acc) - n_acc,
        "rejection_reasons": dict(rejection),
        "acceptance_rate_of_verified": n_acc / len(acc) if acc else None,
        "yield_per_source": n_acc / denom if denom else None,
        "n_sources_with_editor_output": denom_output,
        "yield_per_source_with_editor_output": n_acc / denom_output if denom_output else None,
        "editor_usage": Usage.sum([p.usage for p in proposals if p.usage is not None]).model_dump() if proposals else None,
        "verification_usage": Usage.sum([v.operational_usage for v in verifications if v.operational_usage is not None]).model_dump()
        if verifications
        else None,
        "audits": {
            "n": sum(1 for v in verifications if v.purpose == "audit"),
            "n_accepted": sum(1 for v in verifications if v.purpose == "audit" and v.accepted),
        },
    }


def branch_cost_rows(verifications: list[VerificationRecord]) -> list[dict[str, Any]]:
    """One row per branch continuation with its counterfactual cost parts."""
    rows = []
    for v in verifications:
        for b in v.branches:
            c = b.cost
            rows.append(
                {
                    "verification_id": v.verification_id,
                    "proposal_id": v.proposal_id,
                    "purpose": v.purpose,
                    "mode": v.mode,
                    "accepted": v.accepted,
                    "branch": b.branch,
                    "repetition": b.repetition,
                    "continuation_seed": b.continuation_seed,
                    "success": b.episode.success,
                    "stop_reason": b.episode.stop_reason,
                    "replay_ok": b.replay_ok,
                    "shared_prefix_tokens": c.shared_prefix_tokens,
                    "intervention_request_input_tokens": c.intervention_request_input_tokens,
                    "intervention_tokens": c.intervention_tokens,
                    "intervention_tokens_source": c.intervention_tokens_source,
                    "continuation_tokens": c.continuation_tokens,
                    "total": c.total,
                }
            )
    return rows


SAVING_UNITS = {
    "continuation": "counterfactual episode tokens (prefix + fixed turn + continuation)",
    "local": "fixed-turn intervention tokens only (no continuation observed)",
}


def _branch_ok(v: VerificationRecord, b: Any) -> bool | None:
    """Did this branch do what it had to? Continuation: complete success after a
    model-finished stop with valid replay. Local: the fixed action executed
    without error after a valid replay (local branches are not graded)."""
    ep = b.episode
    if not b.replay_ok or ep.stop_category != StopCategory.MODEL:
        return False
    if v.mode == "local":
        return bool(ep.extra.get("executed", True)) and not ep.extra.get("tool_error")
    return ep.success


def _saving_row(mode: str, vs: list[VerificationRecord]) -> dict[str, Any]:
    all_ok = [v for v in vs if v.branches and all(_branch_ok(v, b) is True for b in v.branches)]
    s_ok = agg(v.mean_saving for v in all_ok)
    s_acc = agg(v.mean_saving for v in vs if v.accepted)
    n_pairs = e_fail_o_ok = o_fail_e_ok = n_undetermined = 0
    for v in vs:
        reps: dict[int, dict[str, Any]] = defaultdict(dict)
        for b in v.branches:
            reps[b.repetition][b.branch] = b
        for rep in reps.values():
            if "original" not in rep or "edited" not in rep:
                continue
            n_pairs += 1
            o, e = _branch_ok(v, rep["original"]), _branch_ok(v, rep["edited"])
            if o is None or e is None:
                n_undetermined += 1
            elif o and not e:
                e_fail_o_ok += 1
            elif e and not o:
                o_fail_e_ok += 1
    return {
        "mode": mode,
        "saving_unit": SAVING_UNITS.get(mode, mode),
        "n_verifications": len(vs),
        "n_all_branches_ok": len(all_ok),
        "mean_saving_all_branches_ok": s_ok["mean"],
        "n_all_branches_ok_with_costs": s_ok["n"],
        "n_accepted": s_acc["n"],
        "mean_saving_accepted": s_acc["mean"],
        "n_branch_pairs": n_pairs,
        "n_edited_failed_original_ok": e_fail_o_ok,
        "n_original_failed_edited_ok": o_fail_e_ok,
        "n_pairs_undetermined": n_undetermined,
    }


def branch_saving_summary(verifications: list[VerificationRecord]) -> dict[str, Any]:
    """Acceptance-verification savings (original - edited), one row per
    verification mode (their units differ and are never mixed). The mean saving
    is conditioned on EVERY branch succeeding (with its n): savings of failed or
    early-stopped branches are not savings. Pair-level counts show edits that
    lost a success (edited failed, original succeeded) and the reverse."""
    acc = [v for v in verifications if v.purpose == "acceptance"]
    modes = sorted({v.mode for v in acc})
    return {"n_verifications": len(acc), "by_mode": [_saving_row(m, [v for v in acc if v.mode == m]) for m in modes]}


REPLAY_TIMING_KEYS = ("replay_tool_sec", "replay_cpu_sec", "replay_wall_sec")


def verification_effort_split(verifications: list[VerificationRecord]) -> dict[str, Any]:
    """Replay vs continuation effort of continuation-mode branch episodes.

    Replay makes no model calls (0 tokens by construction). Its time is known
    only when the branch episode recorded `extra["replay_tool_sec"]` (and
    optionally `replay_cpu_sec` / `replay_wall_sec`); the continuation's tool time
    is then `timing.tool_sec - replay_tool_sec`. Without those keys the split is
    reported as unavailable, never guessed."""
    branches = [b for v in verifications if v.mode == "continuation" for b in v.branches]
    rep = {k: agg(b.episode.extra.get(k) for b in branches) for k in REPLAY_TIMING_KEYS}
    cont_tool = []
    for b in branches:
        rt, tt = b.episode.extra.get("replay_tool_sec"), b.episode.timing.tool_sec
        cont_tool.append(None if rt is None or tt is None else max(0.0, float(tt) - float(rt)))
    out = {
        "n_branches": len(branches),
        "available": bool(branches) and rep["replay_tool_sec"]["n"] > 0,
        "replay_tool_sec": rep["replay_tool_sec"],
        "replay_cpu_sec": rep["replay_cpu_sec"],
        "replay_wall_sec": rep["replay_wall_sec"],
        "replay_tokens": 0 if branches else None,
        "continuation_tool_sec": agg(cont_tool),
        "continuation_tokens": agg(b.cost.continuation_tokens for b in branches),
        "branch_total_sec": agg(b.episode.timing.total_sec for b in branches),
        "branch_tool_sec": agg(b.episode.timing.tool_sec for b in branches),
    }
    if not branches:
        out["note"] = "no continuation branches"
    elif not out["available"]:
        out["note"] = "replay/continuation time split unavailable: branch episodes did not record replay timing"
    elif rep["replay_tool_sec"]["n_missing"]:
        out["note"] = f"replay timing recorded for {rep['replay_tool_sec']['n']}/{len(branches)} branches"
    else:
        out["note"] = None
    return out


def eval_rows(rows: list[EpisodeRow]) -> list[EpisodeRow]:
    """Unassisted policy evaluations only (branches are never evaluations)."""
    return [r for r in rows if r.episode.role == EpisodeRole.EVAL]
